# Architecture and Long-Context reference problems

These modules implement compute-intensive kernels inspired by **modern
transformer and state-space architectures**. They focus on isolated kernel
motifs rather than full blocks so the resulting training data emphasizes
reusable implementation patterns: low-rank KV expansion, expert routing,
grouped-query attention, scaled low-precision attention, and recurrent
delta-style linear attention.

Each file is a self-contained PyTorch reference (`class Model`, `get_inputs()`,
`get_init_inputs()`). They are local-only reference problems in the same style
as `level9`.

## Suggested mapping to architecture domains

| File | Domain | Typical use |
|------|--------|-------------|
| `1_DeepSeekMLALoRAExpansion.py` | MLA / compressed KV cache | Reconstruct K/V tensors from low-rank latent projections |
| `2_DeepSeekMoEGroundedTop2Routing.py` | MoE routing | Grounded top-2 expert selection with similarity-biased router logits |
| `3_GQAKVHeadExpansionAttention.py` | Grouped-query attention | Causal attention with KV-head expansion across query-head groups |
| `4_FP8ScaledAttention.py` | Low-precision attention | Scaled quantized Q/K/V attention with explicit dequantization semantics |
| `5_GatedDeltaNetLinearAttention.py` | Linear attention / recurrent state | Scalar-gated delta-state update with query readout |
| `6_KimiDeltaAttentionChannelwise.py` | Linear attention / recurrent state | Channel-wise gated delta-state update with per-channel decay |
