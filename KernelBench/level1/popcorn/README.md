# Graph and Sparse Learning reference problems

These modules implement compute-intensive kernels from **graph neural networks**,
**sparse attention on graphs**, and **irregular reduction workloads**. They focus
on segmented operations over CSR adjacency, edge-wise normalization, message
passing, scatter/gather, and graph-specific reductions.

Each file is a self-contained PyTorch reference (`class Model`, `get_inputs()`,
`get_init_inputs()`). They are not wired into the default KernelBench
HuggingFace dataset; use `ref_origin=kernelbench dataset_src=local level=9`
or `ref_origin=local` against files in this directory.

## Suggested mapping to graph domains

| File | Domain | Typical use |
|------|--------|-------------|
| `1_GraphEdgeSoftmaxCSR.py` | Graph attention | Row-wise softmax over CSR edge scores for GAT-style attention |
| `2_CSRSpMMMessagePassing.py` | Message passing | CSR sparse aggregation of source node features into destination nodes |
| `3_EdgeSoftmaxMultiHeadCSR.py` | Graph attention | Multi-head row-wise softmax over CSR edge logits |
| `4_SegmentTopKCSR.py` | Graph sampling | Per-node top-k neighbor score selection within CSR segments |
| `5_SampledDenseDenseMatmulEdges.py` | Link prediction | Edge-only dot products between source and destination embeddings |
| `6_DegreeNormalizedAggregation.py` | Spectral GNNs | Degree-normalized neighbor aggregation in the GCN style |
| `7_COOScatterAddNodeFeatures.py` | Message passing | Atomic scatter-add of edge features into destination node buffers |
| `8_CSRMaxAggregation.py` | Pooling GNNs | Feature-wise max aggregation over CSR neighborhoods |
| `9_CSRMultiHeadSpMM.py` | Graph attention | Multi-head weighted sparse aggregation over CSR edges |
| `10_CSRFusedAttentionValue.py` | Graph attention | Fused sparse softmax + weighted value aggregation |
