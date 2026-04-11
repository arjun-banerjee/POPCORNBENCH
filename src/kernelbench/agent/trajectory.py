"""
Trajectory dataclasses for multi-turn kernel agent runs.

A trajectory captures the full history of one agent run on one problem:
  - Every LLM turn (prompt → response)
  - Every tool call within each turn (name, args, result)
  - The final KernelExecResult
  - Run-level metadata (model, backend, timestamps, caps)

Serialization is JSON-only, consistent with the existing runs/ storage pattern.
Non-serializable values (exceptions, torch.dtype, etc.) are coerced to strings.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from kernelbench.eval import KernelExecResult


# ---------------------------------------------------------------------------
# ToolCall — one invocation of a tool within a turn
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    tool_name: str
    args: Dict[str, Any]          # arguments passed (kernel_code, etc.)
    result_text: str               # human/LLM-readable output
    success: bool                  # did the tool succeed without error?
    metadata: Dict[str, Any] = field(default_factory=dict)  # structured data for logging


# ---------------------------------------------------------------------------
# KernelTurn — one complete LLM response + all tool calls within it
# ---------------------------------------------------------------------------

@dataclass
class KernelTurn:
    """One turn = one LLM response + all tool calls made in that response."""
    turn_id: int
    # Full conversation messages sent to the model this turn (list of role/content dicts)
    messages_in: List[Dict[str, str]]
    # Raw LLM response text
    response: str
    # Tool calls parsed from this response and executed
    tool_calls: List[ToolCall] = field(default_factory=list)
    # Feedback message appended to conversation after this turn's tools ran
    feedback_to_model: str = ""
    # Wall-clock time for the LLM call (seconds)
    llm_latency_s: float = 0.0
    # Whether this turn contained a final submission
    is_final: bool = False
    # The extracted kernel code at submission time (if any)
    submitted_kernel: Optional[str] = None


# ---------------------------------------------------------------------------
# KernelTrajectory — full run on one problem
# ---------------------------------------------------------------------------

@dataclass
class KernelTrajectory:
    """Full trajectory for one agent run on one problem."""
    # Problem identity
    problem_id: int
    level: int
    problem_name: str

    # Run config
    run_name: str
    model_name: str
    backend: str
    precision: str
    max_turns: int
    max_tool_calls: int
    tools_enabled: List[str]       # names of tools available this run

    # Turns (appended as the agent runs)
    turns: List[KernelTurn] = field(default_factory=list)

    # Final result (set after submit_kernel or when turns exhausted)
    final_result: Optional[KernelExecResult] = None

    # Timestamps (ISO-8601 strings)
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    finished_at: Optional[str] = None

    # Summary stats (filled in by finish())
    total_turns: int = 0
    total_tool_calls: int = 0
    outcome: str = "unknown"  # "correct", "incorrect", "compile_fail", "timeout", "error"

    # ---------------------------------------------------------------------------

    def add_turn(self, turn: KernelTurn) -> None:
        self.turns.append(turn)
        self.total_turns = len(self.turns)
        self.total_tool_calls = sum(len(t.tool_calls) for t in self.turns)

    def finish(self, result: Optional[KernelExecResult]) -> None:
        """Call when the agent run is complete."""
        self.final_result = result
        self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.total_turns = len(self.turns)
        self.total_tool_calls = sum(len(t.tool_calls) for t in self.turns)

        if result is None:
            self.outcome = "error"
        elif not result.compiled:
            self.outcome = "compile_fail"
        elif not result.correctness:
            self.outcome = "incorrect"
        else:
            self.outcome = "correct"

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        def _coerce(obj):
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_coerce(v) for v in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        d: Dict[str, Any] = {
            "problem_id": self.problem_id,
            "level": self.level,
            "problem_name": self.problem_name,
            "run_name": self.run_name,
            "model_name": self.model_name,
            "backend": self.backend,
            "precision": self.precision,
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "tools_enabled": self.tools_enabled,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "outcome": self.outcome,
            "final_result": _coerce(self.final_result.model_dump() if self.final_result else None),
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "messages_in": t.messages_in,
                    "response": t.response,
                    "feedback_to_model": t.feedback_to_model,
                    "llm_latency_s": t.llm_latency_s,
                    "is_final": t.is_final,
                    "submitted_kernel": t.submitted_kernel,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "args": _coerce(tc.args),
                            "result_text": tc.result_text,
                            "success": tc.success,
                            "metadata": _coerce(tc.metadata),
                        }
                        for tc in t.tool_calls
                    ],
                }
                for t in self.turns
            ],
        }
        return d

    def save(self, path: str) -> None:
        """Save trajectory as JSON to the given path."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "KernelTrajectory":
        """Load a previously saved trajectory (read-only dict form)."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)  # returns raw dict; callers can inspect freely
