# Computational Biology reference problems

These modules implement compute-intensive kernels found in **structural biology** and **bioinformatics** models such as AlphaFold2, ESMFold, protein structure predictors, and molecular dynamics simulators. They exercise patterns like triangular attention, invariant point attention, outer-product updates, pairwise distance geometry, equivariant layers, and sequence-alignment-style dynamic programming.

Each file is a self-contained PyTorch reference (`class Model`, `get_inputs()`, `get_init_inputs()`). They are **not** wired into the default KernelBench HuggingFace dataset; use `ref_origin=local` and point to this directory.

## Suggested mapping to biology domains

| File | Domain | Typical use |
|------|--------|-------------|
| `1_TriangularAttention.py` | Structure prediction | AlphaFold2 pair-representation triangular self-attention |
| `2_TriangularMultiplicativeUpdateOutgoing.py` | Structure prediction | AlphaFold2 outgoing edges multiplicative update |
| `3_TriangularMultiplicativeUpdateIncoming.py` | Structure prediction | AlphaFold2 incoming edges multiplicative update |
| `4_OuterProductMean.py` | Structure prediction | MSA → pair representation outer-product mean |
| `5_MSARowAttention.py` | Structure prediction | Row-wise gated self-attention over MSA |
| `6_MSAColumnAttention.py` | Structure prediction | Column-wise gated self-attention over MSA |
| `7_InvariantPointAttention.py` | Structure prediction | SE(3)-aware attention over residue frames (IPA) |
| `8_PairwiseDistanceMatrix.py` | Molecular dynamics | All-pairs Euclidean distance for 3-D point clouds |
| `9_RotaryPositionEmbeddingBio.py` | Protein language models | RoPE applied to protein residue sequences |
| `10_SmithWatermanDPScore.py` | Sequence alignment | Differentiable local-alignment dynamic programming |
| `11_RadialBasisFunctionExpansion.py` | Molecular GNNs | Expand interatomic distances into RBF features |
| `12_SE3InvariantLinear.py` | Equivariant models | Linear layer that respects SE(3) invariance |
| `13_ContactMapPrediction.py` | Protein structure | Predict residue–residue contacts from embeddings |
| `14_VoxelGridPooling.py` | 3-D molecular modeling | Discretize 3-D point cloud into voxel grid & pool |
| `15_EvoformerBlock.py` | Structure prediction | Full Evoformer block (MSA + pair stack) |
| `16_AxialAttention.py` | Bioimage analysis | Row-then-column factored attention for 2-D grids |
