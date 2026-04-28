import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


csr_multihead_spmm_cpp_source = """
torch::Tensor csr_multihead_spmm_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor edge_weight,
    torch::Tensor node_feat
);
"""


csr_multihead_spmm_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void csr_multihead_spmm_kernel(
    const int* row_ptr,
    const int* col_idx,
    const float* edge_weight,
    const float* node_feat,
    float* out,
    int num_nodes,
    int num_heads,
    int head_dim
) {
    int dst = blockIdx.x;
    int feat = threadIdx.x;
    int head = threadIdx.y;
    if (dst >= num_nodes || feat >= head_dim || head >= num_heads) {
        return;
    }

    int start = row_ptr[dst];
    int end = row_ptr[dst + 1];
    float acc = 0.0f;
    for (int edge = start; edge < end; ++edge) {
        int src = col_idx[edge];
        acc += edge_weight[edge * num_heads + head] * node_feat[(src * num_heads + head) * head_dim + feat];
    }
    out[(dst * num_heads + head) * head_dim + feat] = acc;
}

torch::Tensor csr_multihead_spmm_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor edge_weight,
    torch::Tensor node_feat
) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be CUDA");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be CUDA");
    TORCH_CHECK(edge_weight.is_cuda(), "edge_weight must be CUDA");
    TORCH_CHECK(node_feat.is_cuda(), "node_feat must be CUDA");
    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_idx.scalar_type() == torch::kInt32, "col_idx must be int32");
    TORCH_CHECK(edge_weight.scalar_type() == torch::kFloat32, "edge_weight must be float32");
    TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32, "node_feat must be float32");
    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be 1D");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must be 1D");
    TORCH_CHECK(edge_weight.dim() == 2, "edge_weight must be 2D");
    TORCH_CHECK(node_feat.dim() == 3, "node_feat must be 3D");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(col_idx.is_contiguous(), "col_idx must be contiguous");
    TORCH_CHECK(edge_weight.is_contiguous(), "edge_weight must be contiguous");
    TORCH_CHECK(node_feat.is_contiguous(), "node_feat must be contiguous");

    int num_nodes = row_ptr.size(0) - 1;
    int num_heads = node_feat.size(1);
    int head_dim = node_feat.size(2);
    TORCH_CHECK(num_heads * head_dim <= 1024, "num_heads * head_dim must be <= 1024");
    auto out = torch::zeros({num_nodes, num_heads, head_dim}, node_feat.options());

    dim3 grid(num_nodes);
    dim3 block(head_dim, num_heads);
    csr_multihead_spmm_kernel<<<grid, block>>>(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        edge_weight.data_ptr<float>(),
        node_feat.data_ptr<float>(),
        out.data_ptr<float>(),
        num_nodes,
        num_heads,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "csr_multihead_spmm_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


csr_multihead_spmm_ext = load_inline(
    name="level9_csr_multihead_spmm_cuda",
    cpp_sources=csr_multihead_spmm_cpp_source,
    cuda_sources=csr_multihead_spmm_cuda_source,
    functions=["csr_multihead_spmm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = csr_multihead_spmm_ext

    def forward(self, row_ptr, col_idx, edge_weight, node_feat):
        row_ptr = row_ptr.to(dtype=torch.int32).contiguous()
        col_idx = col_idx.to(dtype=torch.int32).contiguous()
        edge_weight = edge_weight.to(dtype=torch.float32).contiguous()
        node_feat = node_feat.to(dtype=torch.float32).contiguous()
        return self.ext.csr_multihead_spmm_cuda(row_ptr, col_idx, edge_weight, node_feat)
