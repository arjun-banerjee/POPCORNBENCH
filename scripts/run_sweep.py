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
    uv run python scripts/run_sweep.py config=configs/sweep.example.toml
"""

from __future__ import annotations

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

import pydra
import tomli
import torch
from openai import OpenAI
from pydra import Config, REQUIRED
from tqdm import tqdm

from kernelbench.agent import KernelAgent, get_tools
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import set_gpu_arch

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SweepConfig(Config):
    def __init__(self):
        self.config = REQUIRED  # path to the sweep TOML

    def __repr__(self):
        return f"SweepConfig(config={self.config})"


@dataclass
class WorkItem:
    model_idx: int        # index into sweep.models
    level: int
    problem_id: int
    device_id: int        # CUDA device this worker should use


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
# Worker
# ---------------------------------------------------------------------------

def run_one(
    work: WorkItem,
    sweep: dict,
    run_dir: str,
    sem,
    perf_lock,
) -> Optional[dict]:
    """Run a single (model, level, problem) work item."""
    model_cfg = sweep["models"][work.model_idx]
    run_cfg = sweep["run"]
    agent_cfg = sweep["agent"]

    model_name = model_cfg["name"]
    api_kind = model_cfg.get("api_kind", "openai")

    # Bind GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(work.device_id)
    device = torch.device("cuda:0")
    if run_cfg.get("gpu_arch"):
        arch = run_cfg["gpu_arch"]
        set_gpu_arch(arch if isinstance(arch, list) else [arch])

    # Output dir is per-model so results don't collide
    model_dir = os.path.join(run_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)

    traj_path = os.path.join(
        model_dir, f"level_{work.level}_problem_{work.problem_id}_trajectory.json"
    )
    if os.path.exists(traj_path):
        try:
            with open(traj_path) as f:
                d = json.load(f)
            return _summary_from_dict(d, work.level, model_name)
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

    try:
        # Load problem
        dataset = construct_kernelbench_dataset(
            level=work.level,
            source=run_cfg["dataset_src"],
            dataset_name=run_cfg.get("dataset_name", "ScalingIntelligence/KernelBench"),
            variant=run_cfg.get("variant", "original"),
        )
        problem = dataset.get_problem_by_id(work.problem_id)

        # OpenAI client (gated on per-model semaphore)
        raw_client = OpenAI(api_key=api_key, base_url=model_cfg["base_url"])
        client = RateLimitedClient(raw_client, sem)

        # Tools + cache dir
        tool_names = _resolve_tools(run_cfg.get("tools", "default"))
        tools = get_tools(tool_names)
        build_dir = os.path.join(
            model_dir, f"level_{work.level}_problem_{work.problem_id}_cache"
        )
        os.makedirs(build_dir, exist_ok=True)

        precision = _force_backend_precision(run_cfg["backend"], run_cfg["precision"])

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
        )

        # Wrap submit_kernel.execute with a per-GPU perf lock so timing is
        # never measured concurrently with another agent on the same device.
        if perf_lock is not None and "submit_kernel" in agent.tool_map:
            sk = agent.tool_map["submit_kernel"]
            orig_execute = sk.execute

            def locked_execute(ctx, **kw):
                with perf_lock:
                    return orig_execute(ctx, **kw)

            sk.execute = locked_execute  # type: ignore[assignment]

        trajectory = agent.run()
        trajectory.save(traj_path)
        kernel_path = os.path.join(
            model_dir, f"level_{work.level}_problem_{work.problem_id}_kernel.py"
        )
        trajectory.save_kernel(kernel_path)
        return _summary_from_dict(trajectory.to_dict(), work.level, model_name)

    except Exception as e:
        print(f"[Worker] ERROR ({model_name}, lvl {work.level}, p {work.problem_id}): {e}")
        traceback.print_exc()
        return None


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _summary_from_dict(d: dict, level: int, model_name: str) -> dict:
    fr = d.get("final_result") or {}
    return {
        "model": model_name,
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

@pydra.main(base=SweepConfig)
def main(cfg: SweepConfig):
    cfg_path = cfg.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(REPO_TOP_DIR, cfg_path)
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

    # Build (model, level, problem) matrix
    levels = run_cfg["levels"]
    subset = set(run_cfg.get("problem_subset") or [])

    work_items: list[WorkItem] = []
    num_gpus = par_cfg["num_gpu_devices"]
    for level in levels:
        ds = construct_kernelbench_dataset(
            level=level,
            source=run_cfg["dataset_src"],
            dataset_name=run_cfg.get("dataset_name", "ScalingIntelligence/KernelBench"),
            variant=run_cfg.get("variant", "original"),
        )
        all_pids = ds.get_problem_ids()
        pids = [p for p in all_pids if (not subset) or (p in subset)]
        for m_idx, _model in enumerate(sweep["models"]):
            for i, pid in enumerate(pids):
                work_items.append(
                    WorkItem(
                        model_idx=m_idx,
                        level=level,
                        problem_id=int(pid),
                        device_id=(m_idx * 1000 + i) % num_gpus,
                    )
                )

    print(f"[run_sweep] {len(work_items)} work items across "
          f"{len(sweep['models'])} models, levels={levels}")
    print(f"[run_sweep] Workers: {num_gpus * par_cfg['agents_per_gpu']} "
          f"({num_gpus} GPUs x {par_cfg['agents_per_gpu']} per GPU)")

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

    # Per-GPU perf locks
    perf_locks = (
        [manager.Lock() for _ in range(num_gpus)]
        if par_cfg.get("perf_lock_per_gpu", True)
        else [None] * num_gpus
    )

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
                ): w
                for w in work_items
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
