"""Per-GPU evaluation server.

Long-lived process pinned to one GPU. Drains a Manager-backed FIFO queue,
runs each request to completion, ships the result back via a per-call
response queue.

CUDA-context recovery
---------------------
CUDA "illegal memory access" and a handful of related errors are
asynchronous and destroy the entire torch CUDA context for the process.
Once one fires, every subsequent allocation or kernel launch in this
process will fail with the same error, even on perfectly valid candidate
kernels. There is no torch API to reset the context.

Recovery: when we detect a context-fatal error, we ship the error
response back to the requesting agent, log loudly, and exit. The
supervisor in run_sweep.py respawns a fresh server that picks up the
next request from this GPU's queue. Only the bad candidate fails; the
rest of the queue gets a clean process.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)


# Errors that destroy the CUDA context and require process restart.
# Match against the lower-cased exception message.
_CUDA_FATAL_PATTERNS = (
    "illegal memory access",
    "unspecified launch failure",
    "an illegal instruction was encountered",
    "device-side assert",
    "misaligned address",
    "operation not supported on global/shared",
    "invalidaddressspace",
    "cudaerrorillegaladdress",
    "cudaerrorillegal",
    "cudaerrorlaunchfailure",
    "cudaerrorinvalidaddressspace",
    "an unspecified internal error",
)


def _is_cuda_context_fatal(text: str) -> bool:
    s = (text or "").lower()
    return any(p in s for p in _CUDA_FATAL_PATTERNS)


def run_eval_server(
    gpu_id: int,
    request_q,
    ready_event=None,
) -> None:
    """Long-lived loop. Pin to one GPU, drain the queue, exit on None sentinel.

    `request_q` items are: (request_id: str, kind: str, args: dict, response_q).
    Sending `None` to the queue triggers a clean shutdown.
    """
    # Must set CUDA_VISIBLE_DEVICES BEFORE importing torch so the runtime
    # only sees this one GPU (it then becomes cuda:0 from inside the server).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("KB_EVAL_SERVER_GPU", str(gpu_id))
    # Match the agent allocator config so cross-process pressure drops
    # cleanly via empty_cache instead of fragmenting.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    try:
        import torch  # noqa: F401
    except Exception as e:
        logger.error("[eval_server gpu=%d] could not import torch: %s", gpu_id, e)
        if ready_event is not None:
            ready_event.set()
        return

    if not torch.cuda.is_available():
        logger.error("[eval_server gpu=%d] CUDA not available; aborting.", gpu_id)
        if ready_event is not None:
            ready_event.set()
        return

    # Belt-and-suspenders pinning. CUDA_VISIBLE_DEVICES filtering only takes
    # effect if torch hasn't initialized CUDA yet — but with mp.spawn the
    # parent's run_sweep.py is re-imported in the child (torch import + a
    # few helpers), and depending on what those helpers touch, CUDA may
    # already be initialized by the time we get here. If it is, our env
    # var is ignored.
    #
    # We detect both cases via torch.cuda.device_count():
    #   visible == 1  -> env-var pin worked, use cuda:0 (= physical gpu_id)
    #   visible >  1  -> env-var pin failed, use cuda:gpu_id explicitly
    #
    # The crucial part is `server_device` — every ToolContext we build
    # downstream uses this device for tensor allocation, so even when the
    # env var fails, every kernel runs on the right physical GPU.
    visible = torch.cuda.device_count()
    if visible == 1:
        torch.cuda.set_device(0)
        server_device = torch.device("cuda:0")
        physical_gpu = gpu_id
    else:
        torch.cuda.set_device(gpu_id)
        server_device = torch.device(f"cuda:{gpu_id}")
        physical_gpu = gpu_id
        logger.warning(
            "[eval_server gpu=%d] CUDA_VISIBLE_DEVICES=%s did not restrict "
            "device count (saw %d devices); using explicit cuda:%d.",
            gpu_id,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            visible,
            gpu_id,
        )

    try:
        dev_name = torch.cuda.get_device_name(server_device)
    except Exception:
        dev_name = "?"
    logger.info(
        "[eval_server gpu=%d] pinned to physical GPU %d (%s), visible=%d, server_device=%s",
        gpu_id, physical_gpu, dev_name, visible, server_device,
    )

    # Imports that must follow torch (and therefore CUDA_VISIBLE_DEVICES).
    from kernelbench.agent.tools import (
        ToolContext,
        CompileKernelTool,
        RunCorrectnessTool,
        SubmitKernelTool,
        ProfileKernelTool,
        GetGpuSpecsTool,
        DisassembleKernelTool,
        ErtRooflineTool,
    )

    compile_tool = CompileKernelTool()
    correctness_tool = RunCorrectnessTool()
    submit_tool = SubmitKernelTool()
    profile_tool = ProfileKernelTool()
    specs_tool = GetGpuSpecsTool()
    disassemble_tool = DisassembleKernelTool()
    ert_tool = ErtRooflineTool()

    if ready_event is not None:
        ready_event.set()

    logger.info("[eval_server gpu=%d] ready, waiting for requests", gpu_id)
    n_served = 0

    while True:
        try:
            item = request_q.get()
        except (KeyboardInterrupt, SystemExit):
            break

        if item is None:
            logger.info("[eval_server gpu=%d] shutdown signal received "
                        "(served %d requests)", gpu_id, n_served)
            break

        try:
            request_id, kind, args, response_q = item
        except Exception as e:
            logger.warning("[eval_server gpu=%d] bad queue item shape: %s", gpu_id, e)
            continue

        try:
            ctx = _build_server_context(args, server_device)
            if kind == "submit":
                result = submit_tool.execute(ctx, kernel_code=args["kernel_code"])
            elif kind == "compile":
                result = compile_tool.execute(ctx, kernel_code=args["kernel_code"])
            elif kind == "correctness":
                result = correctness_tool.execute(ctx, kernel_code=args["kernel_code"])
            elif kind == "specs":
                result = specs_tool.execute(ctx)
            elif kind == "disassemble":
                result = disassemble_tool.execute(ctx, kernel_code=args["kernel_code"])
            elif kind == "ert":
                result = ert_tool.execute(ctx)
            elif kind == "profile":
                # Restore the agent's prior profile summary so deltas work
                # correctly — see ProfileKernelTool.execute.
                ctx._last_profile_summary = args.get("previous_summary")
                result = profile_tool.execute(ctx, kernel_code=args["kernel_code"])
            else:
                raise ValueError(f"unknown eval kind: {kind!r}")

            payload = {
                "tool_name": result.tool_name,
                "success": result.success,
                "output": result.output,
                "metadata": result.metadata,
            }
            # For profile, also ship back the new summary so the agent can
            # diff against it on the next call.
            if kind == "profile":
                payload["new_profile_summary"] = ctx._last_profile_summary

            response_q.put((request_id, "ok", payload))
            n_served += 1

            # Tools generally CATCH CUDA errors and return ToolResult with
            # success=False, so a fatal CUDA event reaches us through the
            # output text rather than as a raised exception. If the output
            # mentions an illegal memory access (or similar), the context
            # is poisoned and every subsequent kernel on this server will
            # see the same error. Exit so the supervisor respawns us.
            if not result.success and _is_cuda_context_fatal(result.output):
                logger.error(
                    "[eval_server gpu=%d] CUDA context-fatal error detected "
                    "in tool output. Exiting for respawn (served %d requests).",
                    gpu_id, n_served,
                )
                # Brief sleep to let the response_q put flush across IPC.
                time.sleep(0.5)
                sys.exit(1)

        except Exception as e:
            tb = traceback.format_exc()
            logger.warning("[eval_server gpu=%d] request %s failed: %s",
                           gpu_id, request_id, e)
            try:
                response_q.put((request_id, "error",
                                {"error": f"{type(e).__name__}: {e}", "traceback": tb}))
            except Exception:
                # The agent that owned the response queue may have died;
                # drop the response and keep serving.
                pass

            # Same check on the raised path.
            if _is_cuda_context_fatal(str(e)) or _is_cuda_context_fatal(tb):
                logger.error(
                    "[eval_server gpu=%d] CUDA context-fatal exception. "
                    "Exiting for respawn (served %d requests).",
                    gpu_id, n_served,
                )
                time.sleep(0.5)
                sys.exit(1)

    logger.info("[eval_server gpu=%d] exiting", gpu_id)


def _build_server_context(args: dict[str, Any], device) -> "Any":
    """Reconstruct a ToolContext from a serialized request payload.

    `device` is the eval server's pinned physical GPU (a torch.device).
    Every tool's tensor allocation runs on this device, so work goes to
    the right physical GPU even when the CUDA_VISIBLE_DEVICES env-var
    pin is silently ignored (which happens whenever the spawn child
    inherits an already-CUDA-initialized torch state).
    """
    from kernelbench.agent.tools import ToolContext

    return ToolContext(
        ref_arch_src=args["ref_arch_src"],
        backend=args.get("backend", "cuda"),
        precision=args.get("precision", "fp32"),
        device=device,
        build_dir=args.get("build_dir"),
        num_correct_trials=int(args.get("num_correct_trials", 5)),
        num_perf_trials=int(args.get("num_perf_trials", 100)),
        timing_method=args.get("timing_method", "cuda_event"),
        verbose=bool(args.get("verbose", False)),
    )
