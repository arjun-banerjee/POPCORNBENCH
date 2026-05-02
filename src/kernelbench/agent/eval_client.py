"""Client side of the FIFO eval-queue RPC.

A small helper used by SubmitKernelTool (and optionally ProfileKernelTool)
to push a request onto a per-GPU Manager queue and block on the response.

Each agent worker is given a single response Queue proxy, allocated by the
main process before the worker is spawned. All of that worker's RPC calls
land on that one queue and are demultiplexed by request_id. We can't pass
the Manager itself (not picklable for security reasons), only its proxies.
"""

from __future__ import annotations

import logging
import queue as _queue
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class EvalRPCClient:
    """Thin RPC client owned by an agent process.

    Holds two Manager queue proxies: the shared per-GPU request queue and
    a per-agent response queue. Both are allocated by the main process and
    passed in via run_one's args.
    """

    def __init__(self, request_q, response_q, *, default_timeout_s: int = 3600):
        self._request_q = request_q
        self._response_q = response_q
        self._default_timeout_s = default_timeout_s

    def submit_kernel(self, ctx, kernel_code: str):
        args = {
            "kernel_code": kernel_code,
            "ref_arch_src": ctx.ref_arch_src,
            "backend": ctx.backend,
            "precision": ctx.precision,
            "build_dir": ctx.build_dir,
            "num_correct_trials": ctx.num_correct_trials,
            "num_perf_trials": ctx.num_perf_trials,
            "timing_method": ctx.timing_method,
            "verbose": ctx.verbose,
        }
        return self._call("submit", args)

    def profile_kernel(self, ctx, kernel_code: str):
        args = {
            "kernel_code": kernel_code,
            "ref_arch_src": ctx.ref_arch_src,
            "backend": ctx.backend,
            "precision": ctx.precision,
            "build_dir": ctx.build_dir,
            "verbose": ctx.verbose,
            "previous_summary": ctx._last_profile_summary,
        }
        result, new_summary = self._call("profile", args, return_aux=True)
        # Update the agent's local context so the NEXT profile gets a delta
        ctx._last_profile_summary = new_summary
        return result

    def _call(self, kind: str, args: dict, *, return_aux: bool = False):
        from kernelbench.agent.tools import ToolResult

        rid = uuid.uuid4().hex

        try:
            self._request_q.put((rid, kind, args, self._response_q))
        except Exception as e:
            logger.error("[eval_client] enqueue failed for %s: %s", kind, e)
            return ToolResult(
                tool_name=f"{kind}_kernel",
                success=False,
                output=f"{kind}_kernel FAILED: enqueue error: {e}",
                metadata={"error": str(e)},
            )

        # Drain the response queue until we see OUR id. Defensive: under
        # normal operation each agent has its own response queue and never
        # sees a stranger's reply, but on a sweep restart a stale entry
        # could linger. Skip those instead of treating them as ours.
        deadline_s = self._default_timeout_s
        while True:
            try:
                payload = self._response_q.get(timeout=deadline_s)
            except _queue.Empty:
                err = f"{kind}_kernel FAILED: eval server did not respond within {self._default_timeout_s}s"
                logger.warning("[eval_client] %s", err)
                result = ToolResult(
                    tool_name=f"{kind}_kernel",
                    success=False,
                    output=err,
                    metadata={"error": "eval_rpc_timeout"},
                )
                if return_aux:
                    return result, None
                return result

            try:
                rid_resp, status, data = payload
            except Exception as e:
                return _err_result(kind, f"malformed response: {e}", return_aux)

            if rid_resp == rid:
                break
            logger.warning("[eval_client] discarding stale response %s "
                           "(waiting for %s)", rid_resp, rid)

        if status == "error":
            err_text = data.get("error", "unknown error")
            tb = data.get("traceback", "")
            result = ToolResult(
                tool_name=f"{kind}_kernel",
                success=False,
                output=f"{kind}_kernel FAILED: {err_text}",
                metadata={"error": err_text, "traceback": tb},
            )
            if return_aux:
                return result, None
            return result

        # Success path: data is the tool's payload dict
        result = ToolResult(
            tool_name=data["tool_name"],
            success=data["success"],
            output=data["output"],
            metadata=data.get("metadata", {}) or {},
        )
        if return_aux:
            return result, data.get("new_profile_summary")
        return result


def _err_result(kind: str, msg: str, return_aux: bool):
    from kernelbench.agent.tools import ToolResult
    result = ToolResult(
        tool_name=f"{kind}_kernel",
        success=False,
        output=f"{kind}_kernel FAILED: {msg}",
        metadata={"error": msg},
    )
    if return_aux:
        return result, None
    return result
