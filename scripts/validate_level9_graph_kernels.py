import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LEVEL9 = ROOT / "KernelBench" / "level9"
PROMPTS = ROOT / "src" / "kernelbench" / "prompts"


KERNELS = [
    {
        "name": "graph_edge_softmax_csr",
        "ref": LEVEL9 / "1_GraphEdgeSoftmaxCSR.py",
        "new": PROMPTS / "model_new_level9_graph_edge_softmax_csr.py",
    },
    {
        "name": "csr_spmm_message_passing",
        "ref": LEVEL9 / "2_CSRSpMMMessagePassing.py",
        "new": PROMPTS / "model_new_level9_csr_spmm_message_passing.py",
    },
    {
        "name": "edge_softmax_multihead_csr",
        "ref": LEVEL9 / "3_EdgeSoftmaxMultiHeadCSR.py",
        "new": PROMPTS / "model_new_level9_edge_softmax_multihead_csr.py",
    },
    {
        "name": "segment_topk_csr",
        "ref": LEVEL9 / "4_SegmentTopKCSR.py",
        "new": PROMPTS / "model_new_level9_segment_topk_csr.py",
    },
    {
        "name": "sampled_dense_dense_matmul_edges",
        "ref": LEVEL9 / "5_SampledDenseDenseMatmulEdges.py",
        "new": PROMPTS / "model_new_level9_sampled_dense_dense_matmul_edges.py",
    },
    {
        "name": "degree_normalized_aggregation",
        "ref": LEVEL9 / "6_DegreeNormalizedAggregation.py",
        "new": PROMPTS / "model_new_level9_degree_normalized_aggregation.py",
    },
    {
        "name": "coo_scatter_add_node_features",
        "ref": LEVEL9 / "7_COOScatterAddNodeFeatures.py",
        "new": PROMPTS / "model_new_level9_coo_scatter_add_node_features.py",
    },
    {
        "name": "csr_max_aggregation",
        "ref": LEVEL9 / "8_CSRMaxAggregation.py",
        "new": PROMPTS / "model_new_level9_csr_max_aggregation.py",
    },
    {
        "name": "csr_multihead_spmm",
        "ref": LEVEL9 / "9_CSRMultiHeadSpMM.py",
        "new": PROMPTS / "model_new_level9_csr_multihead_spmm.py",
    },
    {
        "name": "csr_fused_attention_value",
        "ref": LEVEL9 / "10_CSRFusedAttentionValue.py",
        "new": PROMPTS / "model_new_level9_csr_fused_attention_value.py",
    },
]


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_outputs(ref_out, new_out, atol=1e-6, rtol=1e-6):
    if isinstance(ref_out, torch.Tensor):
        return torch.allclose(ref_out, new_out, atol=atol, rtol=rtol), float((ref_out - new_out).abs().max().item())
    if isinstance(ref_out, tuple):
        max_diff = 0.0
        for ref_item, new_item in zip(ref_out, new_out):
            ok, diff = compare_outputs(ref_item, new_item, atol=atol, rtol=rtol)
            if not ok:
                return False, diff
            max_diff = max(max_diff, diff)
        return True, max_diff
    raise TypeError(f"Unsupported output type: {type(ref_out)}")


def run_case(ref_mod, new_mod, inputs, atol=1e-6, rtol=1e-6):
    ref_model = ref_mod.Model(*ref_mod.get_init_inputs()).cuda()
    new_model = new_mod.ModelNew(*ref_mod.get_init_inputs()).cuda()

    gpu_inputs = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            gpu_inputs.append(value.cuda())
        else:
            gpu_inputs.append(value)

    with torch.no_grad():
        ref_out = ref_model(*gpu_inputs)
        new_out = new_model(*gpu_inputs)
        torch.cuda.synchronize()
    return compare_outputs(ref_out, new_out, atol=atol, rtol=rtol)


def handcrafted_cases(name: str):
    if name == "graph_edge_softmax_csr":
        return [[
            torch.tensor([0, 2, 5, 6], dtype=torch.int32),
            torch.tensor([1.0, 2.0, -1.0, 0.0, 3.0, 4.0], dtype=torch.float32),
        ]]
    if name == "csr_spmm_message_passing":
        return [[
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 0], dtype=torch.int32),
            torch.tensor([2.0, -1.0, 0.5], dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        ]]
    if name == "edge_softmax_multihead_csr":
        return [[
            torch.tensor([0, 2, 4], dtype=torch.int32),
            torch.tensor([[1.0, 2.0], [3.0, 0.0], [0.5, -0.5], [1.5, 4.0]], dtype=torch.float32),
        ]]
    if name == "segment_topk_csr":
        return [[
            torch.tensor([0, 3, 5], dtype=torch.int32),
            torch.tensor([0.2, 3.0, 1.0, 5.0, 4.0], dtype=torch.float32),
        ]]
    if name == "sampled_dense_dense_matmul_edges":
        return [[
            torch.tensor([0, 1, 0], dtype=torch.int32),
            torch.tensor([1, 0, 1], dtype=torch.int32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        ]]
    if name == "degree_normalized_aggregation":
        return [[
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 0], dtype=torch.int32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([2.0, 3.0], dtype=torch.float32),
        ]]
    if name == "coo_scatter_add_node_features":
        return [[
            torch.tensor([1, 0, 1], dtype=torch.int32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
        ]]
    if name == "csr_max_aggregation":
        return [[
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 0], dtype=torch.int32),
            torch.tensor([[1.0, 5.0], [3.0, 4.0]], dtype=torch.float32),
        ]]
    if name == "csr_multihead_spmm":
        return [[
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 0], dtype=torch.int32),
            torch.tensor([[1.0, 2.0], [0.5, -1.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ], dtype=torch.float32),
        ]]
    if name == "csr_fused_attention_value":
        return [[
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 0], dtype=torch.int32),
            torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        ]]
    return []


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    summary = {}

    for kernel in KERNELS:
        ref_mod = load_module(kernel["ref"], f"ref_{kernel['name']}")
        new_mod = load_module(kernel["new"], f"new_{kernel['name']}")
        results = []

        for seed in [0, 1, 2]:
            torch.manual_seed(seed)
            inputs = ref_mod.get_inputs()
            ok, diff = run_case(ref_mod, new_mod, inputs, atol=1e-5, rtol=1e-5)
            results.append({"type": f"random_seed_{seed}", "ok": ok, "max_abs_diff": diff})

        for idx, inputs in enumerate(handcrafted_cases(kernel["name"])):
            ok, diff = run_case(ref_mod, new_mod, inputs, atol=1e-6, rtol=1e-6)
            results.append({"type": f"handcrafted_{idx}", "ok": ok, "max_abs_diff": diff})

        summary[kernel["name"]] = results

    for name, results in summary.items():
        print(name)
        for item in results:
            print(f"  {item['type']}: ok={item['ok']} max_abs_diff={item['max_abs_diff']}")


if __name__ == "__main__":
    main()
