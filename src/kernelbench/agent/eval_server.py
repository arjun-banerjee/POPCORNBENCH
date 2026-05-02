"""Per-GPU evaluation server.

A long-lived process pinned to one GPU via CUDA_VISIBLE_DEVICES that drains
a Manager-backed FIFO request queue and serves submit_kernel work items in
arrival order.

Architectural rationale
-----------------------
PopcornBench's bottleneck is `submit_kernel`: the full correctness +
perf-timing eval can take minutes on L3/L4 problems with multi-second
reference runtimes. With agents pinned to GPUs and oversubscribed
(agents_per_gpu > 1), many agents end up serializing on a single GPU's
perf lock. Lock fairness primitives (fcntl.flock, mp.Manager().Lock) do not
solve this — long evals legitimately starve others past `worker_timeout_s`.

The fix is to decouple eval lifetime from agent lifetime:

  agents → push (kernel, args) to the GPU's request queue → block on result
                                                                |
                                            (LLM calls run in parallel here)
                                                                |
  one eval-server per GPU → pop, run, push result, repeat (forever)

If an agent's worker_timeout fires while its eval is queued or running, the
agent process dies; the eval still completes (just no one reads the result),
so there are no dangling locks, no orphaned compile caches, no cleanup races.
The queue depth is observable via Manager().Queue().qsize().
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import traceback
from typing import Any

logger = logging.getLogger(__name__)


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
            ctx = _build_server_context(args)
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

    logger.info("[eval_server gpu=%d] exiting", gpu_id)


def _build_server_context(args: dict[str, Any]) -> "Any":
    """Reconstruct a ToolContext from a serialized request payload.

    Imported lazily so this module can be imported without torch present
    (the publisher does this when scanning code for the prompts page).
    """
    import torch
    from kernelbench.agent.tools import ToolContext

    return ToolContext(
        ref_arch_src=args["ref_arch_src"],
        backend=args.get("backend", "cuda"),
        precision=args.get("precision", "fp32"),
        device=torch.device("cuda:0"),  # CUDA_VISIBLE_DEVICES pin makes this correct
        build_dir=args.get("build_dir"),
        num_correct_trials=int(args.get("num_correct_trials", 5)),
        num_perf_trials=int(args.get("num_perf_trials", 100)),
        timing_method=args.get("timing_method", "cuda_event"),
        verbose=bool(args.get("verbose", False)),
    )
