import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


csr_fused_attention_value_cpp_source = """
torch::Tensor csr_fused_attention_value_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor edge_scores,
    torch::Tensor node_value
);
"""


csr_fused_attention_value_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void csr_fused_attention_value_kernel(
    const int* row_ptr,
    const int* col_idx,
    const float* edge_scores,
    const float* node_value,
    float* out,
    int num_nodes,
    int feat_dim
) {
    int dst = blockIdx.x;
    int feat = threadIdx.x;
    if (dst >= num_nodes || feat >= feat_dim) {
        return;
    }

    int start = row_ptr[dst];
    int end = row_ptr[dst + 1];
    if (end <= start) {
        return;
    }

    __shared__ float row_max;
    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        float max_val = -FLT_MAX;
        for (int edge = start; edge < end; ++edge) {
            max_val = fmaxf(max_val, edge_scores[edge]);
        }
        float sum_val = 0.0f;
        for (int edge = start; edge < end; ++edge) {
            sum_val += __expf(edge_scores[edge] - max_val);
        }
        row_max = max_val;
        row_sum = sum_val;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int edge = start; edge < end; ++edge) {
        int src = col_idx[edge];
        float weight = __expf(edge_scores[edge] - row_max) / row_sum;
        acc += weight * node_value[src * feat_dim + feat];
    }
    out[dst * feat_dim + feat] = acc;
}

torch::Tensor csr_fused_attention_value_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor edge_scores,
    torch::Tensor node_value
) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be CUDA");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be CUDA");
    TORCH_CHECK(edge_scores.is_cuda(), "edge_scores must be CUDA");
    TORCH_CHECK(node_value.is_cuda(), "node_value must be CUDA");
    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_idx.scalar_type() == torch::kInt32, "col_idx must be int32");
    TORCH_CHECK(edge_scores.scalar_type() == torch::kFloat32, "edge_scores must be float32");
    TORCH_CHECK(node_value.scalar_type() == torch::kFloat32, "node_value must be float32");
    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be 1D");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must be 1D");
    TORCH_CHECK(edge_scores.dim() == 1, "edge_scores must be 1D");
    TORCH_CHECK(node_value.dim() == 2, "node_value must be 2D");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(col_idx.is_contiguous(), "col_idx must be contiguous");
    TORCH_CHECK(edge_scores.is_contiguous(), "edge_scores must be contiguous");
    TORCH_CHECK(node_value.is_contiguous(), "node_value must be contiguous");

    int num_nodes = row_ptr.size(0) - 1;
    int feat_dim = node_value.size(1);
    TORCH_CHECK(feat_dim <= 1024, "feat_dim must be <= 1024");
    auto out = torch::zeros({num_nodes, feat_dim}, node_value.options());

    dim3 grid(num_nodes);
    dim3 block(feat_dim);
    csr_fused_attention_value_kernel<<<grid, block>>>(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        edge_scores.data_ptr<float>(),
        node_value.data_ptr<float>(),
        out.data_ptr<float>(),
        num_nodes,
        feat_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "csr_fused_attention_value_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


csr_fused_attention_value_ext = load_inline(
    name="level9_csr_fused_attention_value_cuda",
    cpp_sources=csr_fused_attention_value_cpp_source,
    cuda_sources=csr_fused_attention_value_cuda_source,
    functions=["csr_fused_attention_value_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = csr_fused_attention_value_ext

    def forward(self, row_ptr, col_idx, edge_scores, node_value):
        row_ptr = row_ptr.to(dtype=torch.int32).contiguous()
        col_idx = col_idx.to(dtype=torch.int32).contiguous()
        edge_scores = edge_scores.to(dtype=torch.float32).contiguous()
        node_value = node_value.to(dtype=torch.float32).contiguous()
        return self.ext.csr_fused_attention_value_cuda(row_ptr, col_idx, edge_scores, node_value)
