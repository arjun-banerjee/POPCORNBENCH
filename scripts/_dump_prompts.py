"""Dump every prompt the agent feeds to the LLM as JSON on stdout.

Used by publish_to_gh_pages.py to build the static prompts page. Kept as a
separate script so the publisher can subprocess it (avoids importing torch
into the publisher process).
"""

from __future__ import annotations

import json
import sys

from kernelbench.agent.prompt_templates import (
    build_system_prompt,
    build_problem_message,
    build_turn_warning_message,
)
from kernelbench.agent.tools import get_tools, TOOL_REGISTRY


def main() -> int:
    # Canonical context: a multi-turn cuda agent with the default tool set.
    default_tool_names = [
        "compile_kernel",
        "run_correctness",
        "get_gpu_specs",
        "static_check",
        "submit_kernel",
    ]
    all_tool_names = list(TOOL_REGISTRY.keys())

    system_default = build_system_prompt(
        max_turns=10,
        max_tool_calls=30,
        backend="cuda",
        tool_names=default_tool_names,
    )
    system_all = build_system_prompt(
        max_turns=10,
        max_tool_calls=30,
        backend="cuda",
        tool_names=all_tool_names,
    )
    problem_msg = build_problem_message(
        ref_arch_src="<reference PyTorch module source goes here>",
        backend="cuda",
        precision="fp32",
    )
    turn_warning = build_turn_warning_message(
        turns_remaining=2,
        tool_calls_remaining=5,
        has_profiling_tools=True,
    )

    tools_payload = []
    for name, tool in TOOL_REGISTRY.items():
        tools_payload.append({
            "name": name,
            "description": (tool.description or "").strip(),
            "input_schema": tool.input_schema,
            "in_default_set": name in default_tool_names,
        })

    out = {
        "system_default": system_default,
        "system_all_tools": system_all,
        "problem_message": problem_msg,
        "turn_warning": turn_warning,
        "tools": tools_payload,
    }
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
