import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


coo_scatter_add_cpp_source = """
torch::Tensor coo_scatter_add_cuda(
    torch::Tensor dst_idx,
    torch::Tensor edge_feat,
    int64_t num_nodes
);
"""


coo_scatter_add_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coo_scatter_add_kernel(
    const int* dst_idx,
    const float* edge_feat,
    float* out,
    int num_edges,
    int feat_dim
) {
    int edge = blockIdx.x;
    int feat = threadIdx.x;
    if (edge >= num_edges || feat >= feat_dim) {
        return;
    }

    int dst = dst_idx[edge];
    atomicAdd(&out[dst * feat_dim + feat], edge_feat[edge * feat_dim + feat]);
}

torch::Tensor coo_scatter_add_cuda(
    torch::Tensor dst_idx,
    torch::Tensor edge_feat,
    int64_t num_nodes
) {
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA");
    TORCH_CHECK(edge_feat.is_cuda(), "edge_feat must be CUDA");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
    TORCH_CHECK(edge_feat.scalar_type() == torch::kFloat32, "edge_feat must be float32");
    TORCH_CHECK(dst_idx.dim() == 1, "dst_idx must be 1D");
    TORCH_CHECK(edge_feat.dim() == 2, "edge_feat must be 2D");
    TORCH_CHECK(dst_idx.is_contiguous(), "dst_idx must be contiguous");
    TORCH_CHECK(edge_feat.is_contiguous(), "edge_feat must be contiguous");

    int num_edges = dst_idx.size(0);
    int feat_dim = edge_feat.size(1);
    TORCH_CHECK(feat_dim <= 1024, "feat_dim must be <= 1024");
    auto out = torch::zeros({num_nodes, feat_dim}, edge_feat.options());

    dim3 grid(num_edges);
    dim3 block(feat_dim);
    coo_scatter_add_kernel<<<grid, block>>>(
        dst_idx.data_ptr<int>(),
        edge_feat.data_ptr<float>(),
        out.data_ptr<float>(),
        num_edges,
        feat_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "coo_scatter_add_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


coo_scatter_add_ext = load_inline(
    name="level9_coo_scatter_add_cuda",
    cpp_sources=coo_scatter_add_cpp_source,
    cuda_sources=coo_scatter_add_cuda_source,
    functions=["coo_scatter_add_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.ext = coo_scatter_add_ext

    def forward(self, dst_idx, edge_feat):
        dst_idx = dst_idx.to(dtype=torch.int32).contiguous()
        edge_feat = edge_feat.to(dtype=torch.float32).contiguous()
        return self.ext.coo_scatter_add_cuda(dst_idx, edge_feat, self.num_nodes)
