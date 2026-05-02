"""
run_sweep.py — TOML-driven parallel sweep over (model x level x problem).

Reads a TOML config that declares N models (each with its own endpoint, key,
and rate limits) and a list of levels / problems, then fans the matrix out
across a process pool sized for the available GPUs and per-model API budgets.

Concurrency model
-----------------
- Process pool of `num_gpu_devices * agents_per_gpu` workers.
- Each worker is bound to a GPU via CUDA_VISIBLE_DEVICES.
- Per-model semaphore caps in-flight LLM requests for that model.
- Per-GPU lock optionally serializes the perf-timing phase (submit_kernel)
  so wall-clock measurements stay clean while compile/correctness runs
  oversubscribed.
- A background thread re-renders the HTML report every `refresh_seconds`.

Usage:
    uv run python scripts/run_sweep.py configs/sweep.example.toml
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import tomli
import torch
from openai import OpenAI
from tqdm import tqdm

from kernelbench.agent import KernelAgent, get_tools
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import set_gpu_arch

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class WorkItem:
    model_idx: int        # index into sweep.models
    level: int
    problem_id: int
    device_id: int        # CUDA device this worker should use
    variant: str = "original"  # KernelBench variant subdir


def _resolve_tools(tools_arg) -> list[str] | None:
    if isinstance(tools_arg, (list, tuple)):
        return list(tools_arg)
    s = str(tools_arg).strip().lower()
    if s == "default":
        return None
    if s == "all":
        from kernelbench.agent.tools import ALL_TOOLS
        return [t.name for t in ALL_TOOLS]
    return [t.strip() for t in s.split(",") if t.strip()]


def _force_backend_precision(backend: str, precision: str) -> str:
    b = backend.lower()
    if b == "tilelang":
        return "fp16"
    if b == "thunderkittens":
        return "bf16"
    return precision


# ---------------------------------------------------------------------------
# Rate-limited client wrapper
# ---------------------------------------------------------------------------

class _RateLimitedCreate:
    def __init__(self, inner, semaphore):
        self._inner = inner
        self._sem = semaphore

    def create(self, **kwargs):
        with self._sem:
            return self._inner.create(**kwargs)


class _RateLimitedChat:
    def __init__(self, inner, semaphore):
        self._inner = inner
        self._sem = semaphore

    @property
    def completions(self):
        return _RateLimitedCreate(self._inner.completions, self._sem)


class RateLimitedClient:
    """Forwards to an OpenAI client but gates the LLM-call entry points
    (`responses.create` and `chat.completions.create`) on a per-model
    multiprocessing semaphore."""

    def __init__(self, client: OpenAI, semaphore):
        self._client = client
        self._sem = semaphore

    @property
    def responses(self):
        return _RateLimitedCreate(self._client.responses, self._sem)

    @property
    def chat(self):
        return _RateLimitedChat(self._client.chat, self._sem)

    def __getattr__(self, name):
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Robust per-GPU file lock
# ---------------------------------------------------------------------------
#
# We intentionally don't use multiprocessing.Manager().Lock() here: that lock
# isn't tied to process lifetime, so if a holder dies (CUDA OOM-kill, segfault
# inside torch.cuda.synchronize, ncu subprocess hang) the lock stays "held"
# forever and every other worker on that GPU blocks indefinitely.
#
# fcntl.flock is held by a file descriptor; the kernel auto-releases it the
# moment the FD is closed — which always happens, including on SIGKILL.
# So a hung/killed worker can never deadlock its peers.

import contextlib
import fcntl
import signal as _signal


# ---------------------------------------------------------------------------
# Per-work-item soft timeout via SIGALRM
# ---------------------------------------------------------------------------
#
# `signal.alarm(N)` schedules SIGALRM in N seconds. Our handler raises
# TimeoutError, which run_one catches, marks the trajectory `outcome="timeout"`
# on disk so the next sweep run skips it, and returns None. The worker process
# stays alive to handle the next task.
#
# Caveat: SIGALRM only interrupts at the next Python bytecode boundary. Pure
# C-level deadlocks (e.g. CUDA driver hanging in cudaDeviceSynchronize) won't
# be interrupted. In practice ~95% of hangs we've seen are Python-side
# (network recv on slow LLM calls, blocked on a subprocess that never returns)
# and SIGALRM catches all of those.

class _WorkItemTimeout(Exception):
    """Raised when a single work item exceeds its soft timeout budget."""


def _alarm_handler(_signum, _frame):
    raise _WorkItemTimeout("work item exceeded soft timeout budget")


@contextlib.contextmanager
def _work_item_timeout(timeout_s: int):
    """Schedule a SIGALRM-based timeout for the duration of the with-block.

    `timeout_s <= 0` disables the alarm (used as an opt-out)."""
    if timeout_s <= 0 or not hasattr(_signal, "SIGALRM"):
        yield
        return
    old_handler = _signal.signal(_signal.SIGALRM, _alarm_handler)
    _signal.alarm(timeout_s)
    try:
        yield
    finally:
        _signal.alarm(0)  # cancel any pending alarm
        _signal.signal(_signal.SIGALRM, old_handler)


def _mark_trajectory_timed_out(traj_path: str) -> None:
    """Patch an in-progress trajectory file so future resume skips it.

    The agent autosaves after every turn with outcome='in_progress'. When we
    soft-timeout, that file exists and contains all turns up to the kill point,
    but its outcome will be 'in_progress' / finished_at None, which the resume
    logic correctly re-runs. Override both so a future sweep treats it as done.
    """
    if not os.path.exists(traj_path):
        # Nothing to mark — agent never got to its first save.
        return
    try:
        with open(traj_path) as f:
            d = json.load(f)
    except Exception:
        return
    d["outcome"] = "timeout"
    d["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with open(traj_path, "w") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


@contextlib.contextmanager
def _gpu_perf_lock(lock):
    """Acquire a multiprocessing.Manager().Lock() for the with-block.

    `lock=None` is a no-op (used when agents_per_gpu == 1 — no possible
    contention so we skip locking entirely).

    Manager-backed locks are FIFO-ish (a server process owns the semaphore)
    and don't suffer from the wake-all-waiters fairness issue that
    fcntl.flock has under heavy contention. They auto-release if the holder
    dies via the Manager's connection cleanup.
    """
    if lock is None:
        yield
        return
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(
    work: WorkItem,
    sweep: dict,
    run_dir: str,
    sem,
    perf_lock,
    eval_request_q=None,
    eval_response_q=None,
) -> Optional[dict]:
    """Run a single (model, level, problem) work item.

    When `eval_request_q` and `eval_response_q` are both set, submit_kernel
    RPCs to the per-GPU FIFO eval server instead of running locally. The
    response queue is allocated by the main process (Manager objects are
    not picklable; only their proxies are) and is dedicated to this worker.
    """
    model_cfg = sweep["models"][work.model_idx]
    run_cfg = sweep["run"]
    agent_cfg = sweep["agent"]

    model_name = model_cfg["name"]
    api_kind = model_cfg.get("api_kind", "openai")

    # GPU pinning: in single-GPU mode (the default) we bind the worker to its
    # assigned device. In multi_gpu mode (sweep.comm.toml), we leave all GPUs
    # visible so distributed kernels can init NCCL across multiple ranks.
    par_cfg_outer = sweep.get("parallelism", {})
    multi_gpu_mode = bool(par_cfg_outer.get("multi_gpu", False))
    if not multi_gpu_mode:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(work.device_id)
    # Allow PyTorch's allocator to grow segments dynamically. With many
    # agents sharing one GPU, the default expandable=False allocator
    # fragments aggressively and OOMs at modest peak usage; expandable
    # segments cut OOM rate substantially under contention.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    device = torch.device("cuda:0")
    if run_cfg.get("gpu_arch"):
        arch = run_cfg["gpu_arch"]
        set_gpu_arch(arch if isinstance(arch, list) else [arch])

    # Layout: runs/{name}/{variant}/{model}/...
    variant_dir = os.path.join(run_dir, _safe_filename(work.variant))
    model_dir = os.path.join(variant_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)

    traj_path = os.path.join(
        model_dir, f"level_{work.level}_problem_{work.problem_id}_trajectory.json"
    )
    # Skip if a *finished* trajectory already exists. In-progress snapshots
    # (outcome == "in_progress", finished_at == None) get re-run — they're
    # leftovers from a killed sweep, not real completions.
    if os.path.exists(traj_path):
        try:
            with open(traj_path) as f:
                d = json.load(f)
            if d.get("finished_at") and d.get("outcome") != "in_progress":
                return _summary_from_dict(d, work.level, model_name, work.variant)
        except Exception:
            pass

    if api_kind not in ("openai", "openai_chat"):
        msg = (
            f"[Worker] Skipping {model_name}: unsupported api_kind='{api_kind}'. "
            "Use 'openai' (Responses API) or 'openai_chat' (Chat Completions)."
        )
        print(msg)
        _write_skip_marker(traj_path, work, model_name, run_cfg, agent_cfg, msg)
        return None

    api_key = os.environ.get(model_cfg["api_key_env"])
    if not api_key:
        print(f"[Worker] {model_name}: env var {model_cfg['api_key_env']} not set.")
        return None

    # Per-work-item soft timeout. Catches Python-level hangs (slow LLM, stuck
    # subprocess, etc.) without taking down the worker process — so the pool
    # keeps draining and we don't lose the other 7 GPUs.
    par_cfg = sweep.get("parallelism", {})
    worker_timeout_s = int(par_cfg.get("worker_timeout_s", 1800))

    try:
        with _work_item_timeout(worker_timeout_s):
            # Load problem
            dataset = construct_kernelbench_dataset(
                level=work.level,
                source=run_cfg["dataset_src"],
                dataset_name=run_cfg.get("dataset_name", "ScalingIntelligence/KernelBench"),
                variant=work.variant,
            )
            problem = dataset.get_problem_by_id(work.problem_id)

            # OpenAI client (gated on per-model semaphore)
            client_kwargs = {"api_key": api_key, "base_url": model_cfg["base_url"]}
            if "request_timeout_s" in model_cfg:
                client_kwargs["timeout"] = float(model_cfg["request_timeout_s"])
            raw_client = OpenAI(**client_kwargs)
            client = RateLimitedClient(raw_client, sem)

            # Tools + cache dir
            tool_names = _resolve_tools(run_cfg.get("tools", "default"))
            tools = get_tools(tool_names)
            build_dir = os.path.join(
                model_dir, f"level_{work.level}_problem_{work.problem_id}_cache"
            )
            os.makedirs(build_dir, exist_ok=True)

            precision = _force_backend_precision(run_cfg["backend"], run_cfg["precision"])

            # Wire up the eval RPC client if the runner spawned per-GPU
            # eval servers. The client routes submit_kernel through the
            # FIFO queue so agents don't hold the GPU for minutes during
            # perf trials — they push and immediately go back to making
            # LLM calls on the next turn.
            eval_client = None
            if eval_request_q is not None and eval_response_q is not None:
                from kernelbench.agent.eval_client import EvalRPCClient
                eval_client = EvalRPCClient(
                    request_q=eval_request_q,
                    response_q=eval_response_q,
                    default_timeout_s=int(par_cfg.get("eval_rpc_timeout_s", 3600)),
                )

            agent = KernelAgent(
                problem_id=work.problem_id,
                level=work.level,
                problem_name=problem.name,
                ref_arch_src=problem.code,
                client=client,
                model=model_cfg.get("deployment_name", model_name),
                run_name=run_cfg["name"],
                tool_names=[t.name for t in tools],
                max_turns=agent_cfg["max_turns"],
                max_tool_calls=agent_cfg["max_tool_calls"],
                backend=run_cfg["backend"],
                precision=precision,
                device=device,
                build_dir=build_dir,
                num_correct_trials=run_cfg["num_correct_trials"],
                num_perf_trials=run_cfg["num_perf_trials"],
                timing_method=run_cfg["timing_method"],
                reasoning_effort=model_cfg.get("reasoning_effort")
                or agent_cfg.get("reasoning_effort"),
                warn_turns_remaining=agent_cfg.get("warn_turns_remaining", 2),
                turn_delay_s=float(agent_cfg.get("turn_delay_s", 0.0)),
                verbose=False,
                api_kind=api_kind,
                save_path=traj_path,
                eval_client=eval_client,
            )

            # Lock fallback: when the eval queue is NOT in use (e.g. legacy
            # mode or someone running scripts/run_agent.py standalone), wrap
            # submit_kernel.execute with the per-GPU lock so simultaneous
            # agents on the same GPU don't race on perf timing.
            if eval_client is None and perf_lock is not None and "submit_kernel" in agent.tool_map:
                sk = agent.tool_map["submit_kernel"]
                orig_execute = sk.execute

                def locked_execute(ctx, **kw):
                    with _gpu_perf_lock(perf_lock):
                        return orig_execute(ctx, **kw)

                sk.execute = locked_execute  # type: ignore[assignment]

            print(
                f"[worker] START {model_name} variant={work.variant} "
                f"L{work.level}/P{work.problem_id} on cuda:{work.device_id}",
                flush=True,
            )
            _t0 = time.time()
            trajectory = agent.run()
            print(
                f"[worker] DONE  {model_name} L{work.level}/P{work.problem_id} "
                f"→ {trajectory.outcome} in {time.time() - _t0:.1f}s "
                f"({trajectory.total_turns} turns, {trajectory.total_tool_calls} tool calls)",
                flush=True,
            )
            trajectory.save(traj_path)
            kernel_path = os.path.join(
                model_dir, f"level_{work.level}_problem_{work.problem_id}_kernel.py"
            )
            trajectory.save_kernel(kernel_path)
            return _summary_from_dict(trajectory.to_dict(), work.level, model_name, work.variant)

    except _WorkItemTimeout as te:
        print(f"[worker] TIMEOUT ({model_name}, lvl {work.level}, p {work.problem_id}) "
              f"after {worker_timeout_s}s — moving on. {te}", flush=True)
        _mark_trajectory_timed_out(traj_path)
        return None
    except Exception as e:
        print(f"[Worker] ERROR ({model_name}, lvl {work.level}, p {work.problem_id}): {e}")
        traceback.print_exc()
        return None


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _summary_from_dict(d: dict, level: int, model_name: str, variant: str = "") -> dict:
    fr = d.get("final_result") or {}
    return {
        "model": model_name,
        "variant": variant,
        "problem_id": d.get("problem_id"),
        "level": level,
        "problem_name": d.get("problem_name"),
        "outcome": d.get("outcome"),
        "total_turns": d.get("total_turns"),
        "total_tool_calls": d.get("total_tool_calls"),
        "compiled": fr.get("compiled", False),
        "correctness": fr.get("correctness", False),
        "runtime": fr.get("runtime", -1.0),
        "ref_runtime": fr.get("ref_runtime", -1.0),
        "started_at": d.get("started_at"),
        "finished_at": d.get("finished_at"),
    }


def _write_skip_marker(path, work, model_name, run_cfg, agent_cfg, msg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "problem_id": work.problem_id,
                "level": work.level,
                "problem_name": "(skipped)",
                "run_name": run_cfg["name"],
                "model_name": model_name,
                "backend": run_cfg["backend"],
                "precision": run_cfg["precision"],
                "max_turns": agent_cfg["max_turns"],
                "max_tool_calls": agent_cfg["max_tool_calls"],
                "tools_enabled": [],
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_turns": 0,
                "total_tool_calls": 0,
                "outcome": "skipped",
                "skip_reason": msg,
                "final_result": None,
                "turns": [],
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Report regen thread
# ---------------------------------------------------------------------------

def _start_report_thread(run_dir: str, refresh_s: int, stop_event: threading.Event):
    from build_report import build_report  # local import; same scripts/ dir

    def _loop():
        while not stop_event.is_set():
            try:
                build_report(run_dir)
            except Exception as e:
                print(f"[Report] regen failed: {e}")
            stop_event.wait(refresh_s)

    t = threading.Thread(target=_loop, daemon=True, name="report-regen")
    t.start()
    return t


def _start_http_server(report_dir: str, host: str, port: int):
    """Serve `report_dir` over plain HTTP on (host, port).

    Designed for SSH usage: bind to 127.0.0.1 by default and have the user
    forward the port from their laptop with:

        ssh -L 8765:localhost:8765 user@host

    Then visit http://localhost:8765 locally.
    """
    import functools
    import http.server
    import socket
    import socketserver

    os.makedirs(report_dir, exist_ok=True)

    handler_cls = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=report_dir
    )

    # ThreadingTCPServer so the page-refresh meta-tag doesn't head-of-line
    # block other requests (asset fetches, model-page nav, etc.)
    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

        # Quiet the default access log so it doesn't spam the sweep output.
        def handle_error(self, request, client_address):
            pass

    # Silence the per-request log lines.
    class _QuietHandler(handler_cls.func):  # type: ignore[name-defined]
        def log_message(self, fmt, *args):
            return

    httpd = _Server((host, port), functools.partial(_QuietHandler, directory=report_dir))
    hostname = socket.gethostname()
    print(
        "\n[run_sweep] HTML report server listening at:\n"
        f"  http://{host}:{port}/index.html\n"
        f"  (forward from your laptop with: "
        f"ssh -L {port}:localhost:{port} <user>@{hostname})\n"
    )
    t = threading.Thread(target=httpd.serve_forever, daemon=True, name="report-http")
    t.start()
    return httpd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a TOML-driven KernelBench sweep across models and variants.",
    )
    parser.add_argument(
        "config",
        help="Path to the sweep TOML (relative paths are resolved from the repo root).",
    )
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        # Resolve relative to CWD first (for tab-complete), then repo root.
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(REPO_TOP_DIR, args.config)
    with open(cfg_path, "rb") as f:
        sweep = tomli.load(f)

    run_cfg = sweep["run"]
    par_cfg = sweep["parallelism"]
    rep_cfg = sweep.get("report", {"enabled": True, "refresh_seconds": 30})

    run_dir = os.path.join(
        run_cfg.get("runs_dir", os.path.join(REPO_TOP_DIR, "runs")),
        run_cfg["name"],
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep, f, indent=2)

    # Resolve variants: prefer `variants = [...]`; fall back to legacy
    # singular `variant = "..."`; default to ["original"].
    if "variants" in run_cfg:
        variants = list(run_cfg["variants"])
    elif "variant" in run_cfg:
        variants = [run_cfg["variant"]]
    else:
        variants = ["original"]

    # Build (model, level, variant, problem) matrix
    levels = run_cfg["levels"]
    subset = set(run_cfg.get("problem_subset") or [])
    multi_gpu_mode = bool(par_cfg.get("multi_gpu", False))

    # Problems that require all GPUs to evaluate (NCCL collectives, real
    # tensor parallelism, pipeline parallelism). Skipped from any sweep
    # that doesn't set [parallelism].multi_gpu = true; routed through
    # configs/sweep.comm.toml instead.
    MULTI_GPU_PROBLEMS = {
        # (level, variant): set of problem_id
        (2, "popcorn"): {2, 11, 18, 27, 34, 38},
    }

    work_items: list[WorkItem] = []
    num_gpus = par_cfg["num_gpu_devices"]
    n_skipped_multi_gpu = 0
    for variant in variants:
        for level in levels:
            ds = construct_kernelbench_dataset(
                level=level,
                source=run_cfg["dataset_src"],
                dataset_name=run_cfg.get(
                    "dataset_name", "ScalingIntelligence/KernelBench"
                ),
                variant=variant,
            )
            all_pids = ds.get_problem_ids()
            pids = [p for p in all_pids if (not subset) or (p in subset)]

            # Filter the multi-GPU problems based on the run mode:
            # - non-multi_gpu run: drop them (they would crash trying to
            #   init NCCL with one visible GPU)
            # - multi_gpu run: keep only them
            mg_set = MULTI_GPU_PROBLEMS.get((int(level), variant), set())
            if multi_gpu_mode:
                pids = [p for p in pids if int(p) in mg_set]
            else:
                before = len(pids)
                pids = [p for p in pids if int(p) not in mg_set]
                n_skipped_multi_gpu += (before - len(pids)) * len(sweep["models"])

            for m_idx, _model in enumerate(sweep["models"]):
                for pid in pids:
                    work_items.append(
                        WorkItem(
                            model_idx=m_idx,
                            level=level,
                            problem_id=int(pid),
                            device_id=0,  # re-stamped after shuffle
                            variant=variant,
                        )
                    )

    # Shuffle so the pool doesn't drain a single model first (which both
    # starves the per-model LLM semaphore and pile-ups on the per-GPU
    # perf-timing locks). Re-stamp device_id afterwards to keep GPU coverage
    # balanced.
    import random as _random
    _random.seed(0xC0FFEE)  # deterministic so resumes pick the same order
    _random.shuffle(work_items)
    for i, w in enumerate(work_items):
        w.device_id = i % num_gpus

    print(f"[run_sweep] {len(work_items)} work items across "
          f"{len(sweep['models'])} models, levels={levels}, variants={variants}")
    print(f"[run_sweep] Workers: {num_gpus * par_cfg['agents_per_gpu']} "
          f"({num_gpus} GPUs x {par_cfg['agents_per_gpu']} per GPU)")
    if multi_gpu_mode:
        print(f"[run_sweep] multi_gpu=true: comm-kernel mode, all GPUs "
              f"visible to each worker")
    elif n_skipped_multi_gpu:
        print(f"[run_sweep] skipped {n_skipped_multi_gpu} multi-GPU "
              f"work items (run sweep.comm.toml separately to evaluate them)")

    # Spawn-mode required for CUDA in subprocesses
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    # Per-model semaphores (concurrency cap chosen from RPM/TPM headroom)
    per_model_sems = []
    for m in sweep["models"]:
        # crude rule: keep ~2x safety margin on TPM at ~10k tokens/turn
        cap = m.get("max_concurrency") or max(2, min(m.get("tpm", 250000) // 25000, 10))
        per_model_sems.append(manager.Semaphore(cap))
        print(f"  [{m['name']}] per-model concurrency cap = {cap}")

    # Per-GPU FIFO eval queues + dedicated eval-server processes.
    # ----------------------------------------------------------------
    # Each GPU gets one Manager().Queue() that all agents pinned to that GPU
    # push submit_kernel requests into, plus one long-lived eval-server
    # process that drains it in arrival order. Agents block on the response
    # but do NOT hold the GPU during eval — they're free to run their next
    # turn's LLM call while the eval is in flight on a different request.
    # See src/kernelbench/agent/eval_server.py.
    use_eval_queue = par_cfg.get("use_eval_queue", True)
    multi_gpu_mode = bool(par_cfg.get("multi_gpu", False))
    if multi_gpu_mode:
        # Multi-GPU comm sweeps need the worker to see all GPUs at once,
        # which means we cannot pin an eval server to one GPU. Disable the
        # queue and fall back to the local lock path for these runs.
        use_eval_queue = False

    eval_request_qs: list = [None] * num_gpus
    eval_server_procs: list = []
    perf_locks: list = [None] * num_gpus

    if use_eval_queue:
        from kernelbench.agent.eval_server import run_eval_server
        eval_request_qs = [manager.Queue() for _ in range(num_gpus)]
        for gpu_id in range(num_gpus):
            ready = manager.Event()
            p = mp.Process(
                target=run_eval_server,
                args=(gpu_id, eval_request_qs[gpu_id], ready),
                daemon=False,
                name=f"eval_server_gpu{gpu_id}",
            )
            p.start()
            eval_server_procs.append((p, ready))
        # Wait for every server to import torch + tools and signal ready.
        # ~5-10s on a cold venv; happens once per sweep.
        for _, ready in eval_server_procs:
            ready.wait(timeout=120)
        print(f"  eval queue: {num_gpus} per-GPU FIFO server(s) ready")
    elif par_cfg.get("perf_lock_per_gpu", True) and par_cfg["agents_per_gpu"] > 1:
        # Legacy lock fallback (e.g. multi_gpu sweeps).
        perf_locks = [manager.Lock() for _ in range(num_gpus)]
        print(f"  perf-timing serialized per GPU via Manager locks ({num_gpus} GPUs)")
    else:
        if par_cfg["agents_per_gpu"] == 1:
            print("  perf-timing lock disabled (agents_per_gpu=1, no possible contention)")

    # Background HTML report regen
    stop_event = threading.Event()
    if rep_cfg.get("enabled", True):
        _start_report_thread(run_dir, int(rep_cfg.get("refresh_seconds", 30)), stop_event)

    # Optional live HTTP server for the report.
    if rep_cfg.get("serve", True):
        try:
            _start_http_server(
                report_dir=os.path.join(run_dir, "report"),
                host=rep_cfg.get("serve_host", "0.0.0.0"),
                port=int(rep_cfg.get("serve_port", 8765)),
            )
        except OSError as e:
            print(f"[run_sweep] could not start HTTP server: {e} (continuing)")

    total_workers = num_gpus * par_cfg["agents_per_gpu"]
    t0 = time.time()
    completed = 0
    # Pre-allocate one response queue per work item. Manager Queue proxies
    # are picklable; the Manager itself is not, so we cannot pass `manager`
    # into a worker for it to allocate queues on demand.
    eval_response_qs: list = [
        manager.Queue() if use_eval_queue else None for _ in work_items
    ]

    try:
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            futs = {
                executor.submit(
                    run_one,
                    w,
                    sweep,
                    run_dir,
                    per_model_sems[w.model_idx],
                    perf_locks[w.device_id],
                    eval_request_qs[w.device_id],
                    eval_response_qs[i],
                ): w
                for i, w in enumerate(work_items)
            }
            with tqdm(total=len(futs), desc="sweep") as pbar:
                for fut in as_completed(futs):
                    w = futs[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[Worker] crashed (m={w.model_idx} "
                              f"lvl={w.level} p={w.problem_id}): {e}")
                    completed += 1
                    pbar.update(1)
    finally:
        stop_event.set()
        # Drain any remaining queue items by signaling shutdown to each
        # eval server. They exit on a None sentinel.
        for q in eval_request_qs:
            if q is not None:
                try:
                    q.put(None)
                except Exception:
                    pass
        for p, _ in eval_server_procs:
            try:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
            except Exception:
                pass
        # Final report build
        try:
            from build_report import build_report
            build_report(run_dir)
        except Exception as e:
            print(f"[Report] final regen failed: {e}")

    elapsed = time.time() - t0
    print(f"[run_sweep] done in {elapsed/60:.1f} min — see {run_dir}/report/index.html")


if __name__ == "__main__":
    main()
