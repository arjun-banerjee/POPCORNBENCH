"""
SASS disassembly → human-readable DisassemblySummary for the agent.

Takes raw output from sass.disassemble_so() and converts it into a
structured summary the agent can read and reason about.

Key analysis:
- Per-kernel register counts and shared/local memory from cuobjdump -res-usage
- SASS instruction mix breakdown (memory, compute, control, tensor-core, etc.)
- Register pressure assessment against GPU limits from gpu_specs.py
- Occupancy-limiting-factor diagnosis
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from kernelbench.sass import DisassemblyResult


# ---------------------------------------------------------------------------
# SASS instruction classification
# ---------------------------------------------------------------------------

# Opcode prefix → category mapping.  SASS opcodes vary across GPU
# architectures but the major prefixes are stable enough for classification.
_INSTRUCTION_CATEGORIES: list[tuple[str, list[str]]] = [
    ("memory_global",  ["LDG", "STG", "ATOMG", "RED"]),
    ("memory_shared",  ["LDS", "STS", "ATOMS", "LDSM"]),
    ("memory_local",   ["LDL", "STL"]),
    ("memory_const",   ["LDC"]),
    ("memory_texture", ["TEX", "TLD", "TXQ"]),
    ("compute_fp32",   ["FADD", "FMUL", "FFMA", "FMNMX", "FSET", "FSETP",
                         "MUFU", "FCHK", "FCMP", "FSWZADD"]),
    ("compute_fp16",   ["HADD2", "HMUL2", "HFMA2", "HSET2", "HSETP2",
                         "HMNMX2"]),
    ("compute_fp64",   ["DADD", "DMUL", "DFMA", "DMNMX", "DSET", "DSETP"]),
    ("compute_int",    ["IADD", "IADD3", "IMAD", "IMUL", "IMNMX",
                         "ISETP", "ISET", "ICMP", "LOP3", "LOP", "SHL",
                         "SHR", "SHF", "POPC", "FLO", "BREV", "BFE",
                         "BFI", "BMSK", "PRMT", "LEA"]),
    ("tensor_core",    ["HMMA", "IMMA", "DMMA", "BMMA", "QMMA",
                         "HGMMA", "IGMMA"]),
    ("conversion",     ["I2F", "I2I", "F2I", "F2F", "I2IP", "F2FP"]),
    ("predicate",      ["PSETP", "PLOP3", "P2R", "R2P", "CSET", "CSETP"]),
    ("control_flow",   ["BRA", "JMP", "JMXU", "BRX", "JMX", "CALL",
                         "RET", "EXIT", "BREAK", "CONT", "BSSY", "BSYNC",
                         "YIELD", "NANOSLEEP", "BAR", "WARPSYNC",
                         "DEPBAR", "SSY", "CAL", "PRET", "PBK"]),
    ("move",           ["MOV", "MOV32I", "MOVM", "SHFL", "S2R", "CS2R",
                         "S2UR", "R2UR", "UR2R", "UBMSK", "ULEAP",
                         "UIADD3", "UIMAD", "ULOP3", "UPOPC", "UFLO",
                         "UMOV", "UPRMT", "USEL", "USGXT", "USHF",
                         "USHL", "USHR", "VOTEU"]),
    ("misc",           ["NOP", "CS2R", "VOTE", "MATCH", "REDUX",
                         "ELECT"]),
]

_OPCODE_TO_CATEGORY: Dict[str, str] = {}
for cat, opcodes in _INSTRUCTION_CATEGORIES:
    for op in opcodes:
        _OPCODE_TO_CATEGORY[op] = cat


def _classify_opcode(opcode: str) -> str:
    """Classify a SASS opcode into a high-level category."""
    upper = opcode.upper().split(".")[0]
    return _OPCODE_TO_CATEGORY.get(upper, "other")


# ---------------------------------------------------------------------------
# Resource-usage parser (cuobjdump -res-usage output)
# ---------------------------------------------------------------------------

@dataclass
class KernelResourceUsage:
    """Per-kernel resource usage from cuobjdump -res-usage."""
    name: str
    registers: Optional[int] = None
    stack_frame: Optional[int] = None       # bytes
    spill_stores: Optional[int] = None      # bytes
    spill_loads: Optional[int] = None       # bytes
    shared_memory: Optional[int] = None     # bytes (static)
    local_memory: Optional[int] = None      # bytes
    constant_memory: Optional[int] = None   # bytes


_RES_USAGE_FUNC_RE = re.compile(
    r"Function\s*:\s*(.+)", re.IGNORECASE
)
_RES_USAGE_LINE_RE = re.compile(
    r"(REG|STACK|SHARED|LOCAL|CONSTANT|SPILL_STORES?|SPILL_LOADS?)"
    r"\s*[:=]\s*(\d+)",
    re.IGNORECASE,
)


def parse_res_usage(res_usage_text: str) -> list[KernelResourceUsage]:
    """Parse ``cuobjdump -res-usage`` output into structured data."""
    kernels: list[KernelResourceUsage] = []
    current: Optional[KernelResourceUsage] = None

    for line in res_usage_text.splitlines():
        m = _RES_USAGE_FUNC_RE.search(line)
        if m:
            current = KernelResourceUsage(name=m.group(1).strip())
            kernels.append(current)
            continue

        if current is None:
            continue

        for rm in _RES_USAGE_LINE_RE.finditer(line):
            key = rm.group(1).upper()
            val = int(rm.group(2))
            if key == "REG":
                current.registers = val
            elif key == "STACK":
                current.stack_frame = val
            elif key == "SHARED":
                current.shared_memory = val
            elif key == "LOCAL":
                current.local_memory = val
            elif key == "CONSTANT":
                current.constant_memory = val
            elif "SPILL" in key and "STORE" in key:
                current.spill_stores = val
            elif "SPILL" in key and "LOAD" in key:
                current.spill_loads = val

    return kernels


# ---------------------------------------------------------------------------
# SASS instruction-mix parser
# ---------------------------------------------------------------------------

_SASS_OPCODE_RE = re.compile(
    r"^\s*\/\*[^*]*\*\/\s+(@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)",
)


def parse_instruction_mix(sass_text: str) -> Dict[str, int]:
    """
    Parse SASS disassembly and return instruction counts by category.

    Returns:
        Dict mapping category names to instruction counts,
        plus a "total" key.
    """
    counts: Counter[str] = Counter()
    total = 0

    for line in sass_text.splitlines():
        m = _SASS_OPCODE_RE.match(line)
        if m:
            opcode = m.group(2).split(".")[0]
            cat = _classify_opcode(opcode)
            counts[cat] += 1
            total += 1

    result = dict(sorted(counts.items(), key=lambda x: -x[1]))
    result["total"] = total
    return result


# ---------------------------------------------------------------------------
# DisassemblySummary
# ---------------------------------------------------------------------------

@dataclass
class DisassemblySummary:
    """Structured summary of a disassembly run, formatted for LLM consumption."""
    kernel_resources: list[KernelResourceUsage] = field(default_factory=list)
    instruction_mix: Dict[str, int] = field(default_factory=dict)
    max_registers: Optional[int] = None
    gpu_max_regs_per_thread: Optional[int] = None
    has_register_spills: bool = False
    has_tensor_core_ops: bool = False
    errors: list[str] = field(default_factory=list)

    def format_for_llm(self) -> str:
        lines = ["=== Kernel Disassembly Summary ==="]

        if self.kernel_resources:
            lines.append("")
            lines.append("--- Per-Kernel Resource Usage ---")
            for kr in self.kernel_resources:
                short_name = _shorten_kernel_name(kr.name)
                lines.append(f"  Kernel: {short_name}")
                if kr.registers is not None:
                    reg_warn = ""
                    if self.gpu_max_regs_per_thread and kr.registers > self.gpu_max_regs_per_thread * 0.8:
                        reg_warn = "  ⚠ HIGH register pressure"
                    lines.append(f"    Registers:      {kr.registers}{reg_warn}")
                if kr.shared_memory is not None:
                    lines.append(f"    Shared memory:  {kr.shared_memory} bytes")
                if kr.local_memory is not None and kr.local_memory > 0:
                    lines.append(f"    Local memory:   {kr.local_memory} bytes")
                if kr.stack_frame is not None and kr.stack_frame > 0:
                    lines.append(f"    Stack frame:    {kr.stack_frame} bytes")
                if kr.spill_stores and kr.spill_stores > 0:
                    lines.append(f"    Spill stores:   {kr.spill_stores} bytes  ⚠ REGISTER SPILLS")
                if kr.spill_loads and kr.spill_loads > 0:
                    lines.append(f"    Spill loads:    {kr.spill_loads} bytes  ⚠ REGISTER SPILLS")

        if self.instruction_mix:
            lines.append("")
            lines.append("--- Instruction Mix ---")
            total = self.instruction_mix.get("total", 1)
            for cat, count in self.instruction_mix.items():
                if cat == "total":
                    continue
                pct = 100.0 * count / total if total > 0 else 0
                lines.append(f"    {cat:<20s} {count:>6d}  ({pct:5.1f}%)")
            lines.append(f"    {'TOTAL':<20s} {total:>6d}")

        if self.has_tensor_core_ops:
            lines.append("")
            lines.append("Tensor core instructions detected (HMMA/IMMA/etc).")

        if self.has_register_spills:
            lines.append("")
            lines.append("⚠ Register spills detected — kernel is spilling registers")
            lines.append("  to local memory. Consider reducing register usage by:")
            lines.append("  - Reducing live variables per thread")
            lines.append("  - Using __launch_bounds__ to hint the compiler")
            lines.append("  - Trading register reuse for recomputation")

        if self.max_registers is not None:
            lines.append("")
            peak_str = f" / {self.gpu_max_regs_per_thread}" if self.gpu_max_regs_per_thread else ""
            lines.append(f"Max registers across kernels: {self.max_registers}{peak_str}")

        if self.errors:
            lines.append("")
            lines.append("--- Warnings ---")
            for e in self.errors:
                lines.append(f"  {e}")

        if len(lines) == 1:
            lines.append("No disassembly data was available.")

        return "\n".join(lines)


def _shorten_kernel_name(name: str) -> str:
    """Shorten C++ mangled kernel names for readability."""
    if len(name) > 100:
        if "(" in name:
            name = name[:name.index("(")]
        if len(name) > 100:
            name = name[:97] + "..."
    return name


# ---------------------------------------------------------------------------
# Main parser entry point
# ---------------------------------------------------------------------------

def parse_disassembly(
    result: DisassemblyResult,
    device_name: str = "",
) -> DisassemblySummary:
    """
    Convert a DisassemblyResult → DisassemblySummary.

    Args:
        result: Output from sass.disassemble_so() or sass.disassemble_kernelbench_model().
        device_name: torch.cuda.get_device_name() string for GPU limit lookup.

    Returns:
        DisassemblySummary with all computable fields filled in.
    """
    from kernelbench.prompts.hardware.gpu_specs import GPU_SPEC_INFO
    from kernelbench.agent.nsight_parser import _DEVICE_NAME_TO_SPEC_KEY

    summary = DisassemblySummary(errors=list(result.errors))

    # GPU register limit
    spec_key = None
    for substr, key in _DEVICE_NAME_TO_SPEC_KEY:
        if substr in device_name:
            spec_key = key
            break
    if spec_key and spec_key in GPU_SPEC_INFO:
        max_regs_str = GPU_SPEC_INFO[spec_key].get("Maximum number of registers per thread", "")
        try:
            summary.gpu_max_regs_per_thread = int(re.search(r"\d+", max_regs_str).group())
        except (AttributeError, ValueError):
            pass

    # Parse resource usage
    if result.res_usage:
        summary.kernel_resources = parse_res_usage(result.res_usage)
        reg_values = [kr.registers for kr in summary.kernel_resources if kr.registers is not None]
        if reg_values:
            summary.max_registers = max(reg_values)
        summary.has_register_spills = any(
            (kr.spill_stores and kr.spill_stores > 0) or (kr.spill_loads and kr.spill_loads > 0)
            for kr in summary.kernel_resources
        )

    # Parse instruction mix from cuobjdump SASS (preferred) or nvdisasm SASS
    sass_text = result.sass or result.nvdisasm_sass
    if sass_text:
        summary.instruction_mix = parse_instruction_mix(sass_text)
        summary.has_tensor_core_ops = summary.instruction_mix.get("tensor_core", 0) > 0

    return summary
