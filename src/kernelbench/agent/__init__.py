"""
KernelBench Agent — multi-turn, tool-using kernel optimization agent.

Public API:
    from kernelbench.agent import KernelAgent, get_tools, ToolContext
    from kernelbench.agent.trajectory import KernelTrajectory, KernelTurn, ToolCall
    from kernelbench.agent.tools import TOOL_REGISTRY, ALL_TOOLS
    from kernelbench.agent.nsight_parser import parse_nsight_metrics, ProfileSummary
"""

from .agent import KernelAgent, parse_tool_calls
from .tools import get_tools, ToolContext, TOOL_REGISTRY, ALL_TOOLS
from .trajectory import KernelTrajectory, KernelTurn, ToolCall

__all__ = [
    "KernelAgent",
    "parse_tool_calls",
    "get_tools",
    "ToolContext",
    "TOOL_REGISTRY",
    "ALL_TOOLS",
    "KernelTrajectory",
    "KernelTurn",
    "ToolCall",
]
