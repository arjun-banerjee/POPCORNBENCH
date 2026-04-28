import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


degree_norm_cpp_source = """
torch::Tensor degree_normalized_aggregation_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor node_feat,
    torch::Tensor degrees
);
"""


degree_norm_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void degree_normalized_aggregation_kernel(
    const int* row_ptr,
    const int* col_idx,
    const float* node_feat,
    const float* degrees,
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
    float dst_degree = degrees[dst];
    float acc = 0.0f;
    for (int edge = start; edge < end; ++edge) {
        int src = col_idx[edge];
        float norm = rsqrtf(dst_degree * degrees[src]);
        acc += norm * node_feat[src * feat_dim + feat];
    }
    out[dst * feat_dim + feat] = acc;
}

torch::Tensor degree_normalized_aggregation_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor node_feat,
    torch::Tensor degrees
) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be CUDA");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be CUDA");
    TORCH_CHECK(node_feat.is_cuda(), "node_feat must be CUDA");
    TORCH_CHECK(degrees.is_cuda(), "degrees must be CUDA");
    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_idx.scalar_type() == torch::kInt32, "col_idx must be int32");
    TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32, "node_feat must be float32");
    TORCH_CHECK(degrees.scalar_type() == torch::kFloat32, "degrees must be float32");
    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be 1D");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must be 1D");
    TORCH_CHECK(node_feat.dim() == 2, "node_feat must be 2D");
    TORCH_CHECK(degrees.dim() == 1, "degrees must be 1D");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(col_idx.is_contiguous(), "col_idx must be contiguous");
    TORCH_CHECK(node_feat.is_contiguous(), "node_feat must be contiguous");
    TORCH_CHECK(degrees.is_contiguous(), "degrees must be contiguous");

    int num_nodes = row_ptr.size(0) - 1;
    int feat_dim = node_feat.size(1);
    TORCH_CHECK(feat_dim <= 1024, "feat_dim must be <= 1024 for this kernel");
    auto out = torch::zeros({num_nodes, feat_dim}, node_feat.options());

    dim3 grid(num_nodes);
    dim3 block(feat_dim);
    degree_normalized_aggregation_kernel<<<grid, block>>>(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        node_feat.data_ptr<float>(),
        degrees.data_ptr<float>(),
        out.data_ptr<float>(),
        num_nodes,
        feat_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "degree_normalized_aggregation_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


degree_norm_ext = load_inline(
    name="level9_degree_normalized_aggregation_cuda",
    cpp_sources=degree_norm_cpp_source,
    cuda_sources=degree_norm_cuda_source,
    functions=["degree_normalized_aggregation_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = degree_norm_ext

    def forward(self, row_ptr, col_idx, node_feat, degrees):
        row_ptr = row_ptr.to(dtype=torch.int32).contiguous()
        col_idx = col_idx.to(dtype=torch.int32).contiguous()
        node_feat = node_feat.to(dtype=torch.float32).contiguous()
        degrees = degrees.to(dtype=torch.float32).contiguous()
        return self.ext.degree_normalized_aggregation_cuda(row_ptr, col_idx, node_feat, degrees)
