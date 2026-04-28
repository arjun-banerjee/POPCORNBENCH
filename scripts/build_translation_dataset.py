"""
Populate KernelBench/level{L}/_translation_sources/{backend}/ with source-DSL
implementations of benchmark problems, used as inputs to translation prompts
(prompt_option=translation, source_backend=...).

Two modes:

  mode=copy_level9_cuda
      Copy the validated CUDA implementations of the 10 level9 graph kernels
      from src/kernelbench/prompts/model_new_level9_*.py into
      KernelBench/level9/_translation_sources/cuda/<problem_filename>.

  mode=torch_compile_triton level=<L>
      For each problem in level L, run torch.compile (Inductor) to emit a
      Triton kernel and write it to
      KernelBench/level{L}/_translation_sources/triton/<problem_filename>.
      Requires a CUDA device.

Both modes write a manifest.json next to the copied kernels listing
{problem_id, problem_name, source_path}.

Usage:
  uv run python scripts/build_translation_dataset.py mode=copy_level9_cuda
  uv run python scripts/build_translation_dataset.py mode=torch_compile_triton level=1 dataset_src=local
"""
import json
import os
import shutil
import sys

import pydra
from pydra import Config

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Hand-curated mapping for level9: prompts/model_new_level9_<slug>.py
# corresponds to KernelBench/level9/<problem_name>. The slug is the
# snake_case form of the problem name (without leading number).
LEVEL9_CUDA_MAP = {
    "graph_edge_softmax_csr": "1_GraphEdgeSoftmaxCSR.py",
    "csr_spmm_message_passing": "2_CSRSpMMMessagePassing.py",
    "edge_softmax_multihead_csr": "3_EdgeSoftmaxMultiHeadCSR.py",
    "segment_topk_csr": "4_SegmentTopKCSR.py",
    "sampled_dense_dense_matmul_edges": "5_SampledDenseDenseMatmulEdges.py",
    "degree_normalized_aggregation": "6_DegreeNormalizedAggregation.py",
    "coo_scatter_add_node_features": "7_COOScatterAddNodeFeatures.py",
    "csr_max_aggregation": "8_CSRMaxAggregation.py",
    "csr_multihead_spmm": "9_CSRMultiHeadSpMM.py",
    "csr_fused_attention_value": "10_CSRFusedAttentionValue.py",
}


class BuildConfig(Config):
    def __init__(self):
        self.mode = "copy_level9_cuda"  # or "torch_compile_triton"
        self.level = 9
        self.dataset_src = "local"
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.problem_ids = None  # comma-separated list to limit work; None = all
        self.overwrite = False  # re-emit even if target file already exists
        self.verbose = True

    def __repr__(self):
        return f"BuildConfig({self.to_dict()})"


def _problem_ids_filter(config: BuildConfig):
    if not config.problem_ids:
        return None
    if isinstance(config.problem_ids, (list, tuple)):
        return {int(x) for x in config.problem_ids}
    return {int(x) for x in str(config.problem_ids).split(",") if x.strip()}


def _write_manifest(out_dir: str, entries: list):
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"entries": entries}, f, indent=2)
    return manifest_path


def copy_level9_cuda(config: BuildConfig):
    prompts_dir = os.path.join(REPO_TOP_DIR, "src", "kernelbench", "prompts")
    problems_dir = os.path.join(REPO_TOP_DIR, "KernelBench", "level9")
    out_dir = os.path.join(problems_dir, "_translation_sources", "cuda")
    os.makedirs(out_dir, exist_ok=True)

    # Build problem_name -> problem_id from the level9 directory listing
    problem_files = sorted(
        f for f in os.listdir(problems_dir)
        if f.endswith(".py") and f[0].isdigit()
    )
    name_to_id = {}
    for pf in problem_files:
        # filenames are <id>_<Name>.py
        prefix = pf.split("_", 1)[0]
        try:
            name_to_id[pf] = int(prefix)
        except ValueError:
            continue

    entries = []
    for slug, problem_filename in LEVEL9_CUDA_MAP.items():
        src = os.path.join(prompts_dir, f"model_new_level9_{slug}.py")
        if not os.path.exists(src):
            print(f"[skip] missing prompt source for {slug}: {src}")
            continue
        if problem_filename not in name_to_id:
            print(f"[skip] no level9 problem named {problem_filename}")
            continue
        dst = os.path.join(out_dir, problem_filename)
        if os.path.exists(dst) and not config.overwrite:
            if config.verbose:
                print(f"[keep] {dst} already exists (set overwrite=True to replace)")
        else:
            shutil.copyfile(src, dst)
            if config.verbose:
                print(f"[copy] {src} -> {dst}")
        entries.append(
            {
                "problem_id": name_to_id[problem_filename],
                "problem_name": problem_filename,
                "source_path": os.path.relpath(dst, REPO_TOP_DIR),
                "source_backend": "cuda",
                "provenance": "src/kernelbench/prompts/model_new_level9_*.py",
            }
        )
    manifest_path = _write_manifest(out_dir, entries)
    print(f"[done] wrote {len(entries)} entries; manifest: {manifest_path}")


def torch_compile_triton(config: BuildConfig):
    # Lazy imports because these require a CUDA-equipped install
    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.compile_annotations import build_annotated_context

    level = int(config.level)
    out_dir = os.path.join(
        REPO_TOP_DIR, "KernelBench", f"level{level}", "_translation_sources", "triton"
    )
    os.makedirs(out_dir, exist_ok=True)

    dataset = construct_kernelbench_dataset(
        level=level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
    )
    pid_filter = _problem_ids_filter(config)
    entries = []
    for problem in dataset:
        pid = problem.problem_id
        if pid_filter is not None and pid not in pid_filter:
            continue
        dst = os.path.join(out_dir, problem.name)
        if os.path.exists(dst) and not config.overwrite:
            if config.verbose:
                print(f"[keep] {dst}")
            entries.append(
                {
                    "problem_id": pid,
                    "problem_name": problem.name,
                    "source_path": os.path.relpath(dst, REPO_TOP_DIR),
                    "source_backend": "triton",
                    "provenance": "torch.compile (Inductor)",
                }
            )
            continue
        try:
            ann = build_annotated_context(problem.code)
        except Exception as e:
            print(f"[fail] level={level} pid={pid} ({problem.name}): {e}")
            continue
        triton_kernels = ann.get("triton_kernels", "").strip()
        if not triton_kernels:
            print(f"[fail] no triton kernels emitted for pid={pid}")
            continue
        # Wrap the inductor-emitted Triton in a minimal Module-style file. The
        # call_fn from inductor already orchestrates the kernels; we expose it
        # as ModelNew.forward.
        call_fn = ann.get("call_fn", "").strip()
        contents = (
            "# Auto-generated by scripts/build_translation_dataset.py via "
            "torch.compile (Inductor).\n"
            "# Do not edit; re-run the script to refresh.\n\n"
            f"{triton_kernels}\n\n"
            f"{call_fn}\n"
        )
        with open(dst, "w") as f:
            f.write(contents)
        if config.verbose:
            print(f"[emit] {dst}")
        entries.append(
            {
                "problem_id": pid,
                "problem_name": problem.name,
                "source_path": os.path.relpath(dst, REPO_TOP_DIR),
                "source_backend": "triton",
                "provenance": "torch.compile (Inductor)",
            }
        )
    manifest_path = _write_manifest(out_dir, entries)
    print(f"[done] wrote {len(entries)} entries; manifest: {manifest_path}")


@pydra.main(base=BuildConfig)
def main(config: BuildConfig):
    print(f"[build_translation_dataset] {config}")
    mode = str(config.mode).lower()
    if mode == "copy_level9_cuda":
        copy_level9_cuda(config)
    elif mode == "torch_compile_triton":
        torch_compile_triton(config)
    else:
        sys.exit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
