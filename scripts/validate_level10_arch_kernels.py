import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LEVEL10 = ROOT / "KernelBench" / "level10"
PROMPTS = ROOT / "src" / "kernelbench" / "prompts"


KERNELS = [
    {
        "name": "deepseek_mla_lora_expansion",
        "ref": LEVEL10 / "1_DeepSeekMLALoRAExpansion.py",
        "new": PROMPTS / "model_new_level10_deepseek_mla_lora_expansion.py",
    },
    {
        "name": "deepseek_moe_grounded_top2_routing",
        "ref": LEVEL10 / "2_DeepSeekMoEGroundedTop2Routing.py",
        "new": PROMPTS / "model_new_level10_deepseek_moe_grounded_top2_routing.py",
    },
    {
        "name": "gqa_kv_head_expansion_attention",
        "ref": LEVEL10 / "3_GQAKVHeadExpansionAttention.py",
        "new": PROMPTS / "model_new_level10_gqa_kv_head_expansion_attention.py",
    },
    {
        "name": "fp8_scaled_attention",
        "ref": LEVEL10 / "4_FP8ScaledAttention.py",
        "new": PROMPTS / "model_new_level10_fp8_scaled_attention.py",
    },
    {
        "name": "gated_delta_net_linear_attention",
        "ref": LEVEL10 / "5_GatedDeltaNetLinearAttention.py",
        "new": PROMPTS / "model_new_level10_gated_delta_net_linear_attention.py",
    },
    {
        "name": "kimi_delta_attention_channelwise",
        "ref": LEVEL10 / "6_KimiDeltaAttentionChannelwise.py",
        "new": PROMPTS / "model_new_level10_kimi_delta_attention_channelwise.py",
    },
    {
        "name": "rope_kv_cache_update",
        "ref": LEVEL10 / "7_RoPEKVCacheUpdate.py",
        "new": PROMPTS / "model_new_level10_rope_kv_cache_update.py",
    },
    {
        "name": "deepseek_moe_dispatch_permute",
        "ref": LEVEL10 / "8_DeepSeekMoEDispatchPermute.py",
        "new": PROMPTS / "model_new_level10_deepseek_moe_dispatch_permute.py",
    },
    {
        "name": "deepseek_moe_combine_scatter",
        "ref": LEVEL10 / "9_DeepSeekMoECombineScatter.py",
        "new": PROMPTS / "model_new_level10_deepseek_moe_combine_scatter.py",
    },
    {
        "name": "fused_mla_attention",
        "ref": LEVEL10 / "10_FusedMLAAttention.py",
        "new": PROMPTS / "model_new_level10_fused_mla_attention.py",
    },
]

TOLERANCES = {
    "deepseek_mla_lora_expansion": (3e-4, 3e-4),
    "fused_mla_attention": (3e-4, 3e-4),
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_outputs(ref_out, new_out, atol=1e-5, rtol=1e-5):
    if isinstance(ref_out, torch.Tensor):
        ok = torch.allclose(ref_out, new_out, atol=atol, rtol=rtol)
        diff = float((ref_out - new_out).abs().max().item())
        return ok, diff
    if isinstance(ref_out, tuple):
        max_diff = 0.0
        for ref_item, new_item in zip(ref_out, new_out):
            ok, diff = compare_outputs(ref_item, new_item, atol=atol, rtol=rtol)
            if not ok:
                return False, diff
            max_diff = max(max_diff, diff)
        return True, max_diff
    raise TypeError(f"Unsupported output type: {type(ref_out)}")


def run_case(ref_mod, new_mod, inputs, atol=1e-5, rtol=1e-5):
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
    if name == "deepseek_mla_lora_expansion":
        return [[
            torch.tensor([[[1.0, 2.0], [0.5, -1.0]]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 0.0, 1.0, 0.0], [1.0, 3.0, 1.0, 3.0]], dtype=torch.float32),
        ]]
    if name == "deepseek_moe_grounded_top2_routing":
        return [[
            torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            torch.tensor([[0.1, 0.2, -0.5], [0.0, 1.0, -1.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
            0.25,
        ]]
    if name == "gqa_kv_head_expansion_attention":
        return [[
            torch.randn(1, 4, 4, 4, dtype=torch.float32),
            torch.randn(1, 4, 2, 4, dtype=torch.float32),
            torch.randn(1, 4, 2, 4, dtype=torch.float32),
        ]]
    if name == "fp8_scaled_attention":
        q = torch.randint(-4, 5, (1, 4, 2, 4), dtype=torch.int8)
        k = torch.randint(-4, 5, (1, 4, 2, 4), dtype=torch.int8)
        v = torch.randint(-4, 5, (1, 4, 2, 4), dtype=torch.int8)
        scale = torch.tensor([0.25, 0.5], dtype=torch.float32)
        return [[q, k, v, scale, scale, scale]]
    if name == "gated_delta_net_linear_attention":
        return [[
            torch.randn(1, 6, 2, 4, dtype=torch.float32),
            torch.randn(1, 6, 2, 4, dtype=torch.float32),
            torch.randn(1, 6, 2, 4, dtype=torch.float32),
            torch.sigmoid(torch.randn(1, 6, 2, dtype=torch.float32)),
            torch.sigmoid(torch.randn(1, 6, 2, dtype=torch.float32)),
        ]]
    if name == "kimi_delta_attention_channelwise":
        return [[
            torch.randn(1, 6, 2, 4, dtype=torch.float32),
            torch.randn(1, 6, 2, 4, dtype=torch.float32),
            torch.randn(1, 6, 2, 4, dtype=torch.float32),
            torch.sigmoid(torch.randn(1, 6, 2, 4, dtype=torch.float32)),
        ]]
    if name == "rope_kv_cache_update":
        return [[
            torch.randn(1, 3, 2, 4, dtype=torch.float32),
            torch.randn(1, 3, 2, 4, dtype=torch.float32),
            torch.ones(8, 2, dtype=torch.float32),
            torch.zeros(8, 2, dtype=torch.float32),
            torch.zeros(1, 8, 2, 4, dtype=torch.float32),
            torch.zeros(1, 8, 2, 4, dtype=torch.float32),
            torch.tensor([[1, 3, 5]], dtype=torch.int32),
        ]]
    if name == "deepseek_moe_dispatch_permute":
        return [[
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
            torch.tensor([0, 1, 0, 1], dtype=torch.int32),
            torch.tensor([0, 0, 1, 1], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int32),
        ]]
    if name == "deepseek_moe_combine_scatter":
        return [[
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]], dtype=torch.float32),
            torch.tensor([0, 0, 1, 1], dtype=torch.int32),
            torch.tensor([0.25, 0.75, 0.6, 0.4], dtype=torch.float32),
            2,
        ]]
    if name == "fused_mla_attention":
        return [[
            torch.randn(1, 5, 2, 4, dtype=torch.float32),
            torch.randn(1, 5, 3, dtype=torch.float32),
            torch.randn(3, 2, 4, dtype=torch.float32),
            torch.randn(3, 2, 4, dtype=torch.float32),
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
            atol, rtol = TOLERANCES.get(kernel["name"], (2e-5, 2e-5))
            ok, diff = run_case(ref_mod, new_mod, inputs, atol=atol, rtol=rtol)
            results.append({"type": f"random_seed_{seed}", "ok": ok, "max_abs_diff": diff})

        for idx, inputs in enumerate(handcrafted_cases(kernel["name"])):
            atol, rtol = TOLERANCES.get(kernel["name"], (1e-5, 1e-5))
            ok, diff = run_case(ref_mod, new_mod, inputs, atol=atol, rtol=rtol)
            results.append({"type": f"handcrafted_{idx}", "ok": ok, "max_abs_diff": diff})

        summary[kernel["name"]] = results

    for name, results in summary.items():
        print(name)
        for item in results:
            print(f"  {item['type']}: ok={item['ok']} max_abs_diff={item['max_abs_diff']}")


if __name__ == "__main__":
    main()
