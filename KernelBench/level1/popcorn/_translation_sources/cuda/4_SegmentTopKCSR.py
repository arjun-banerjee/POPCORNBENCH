import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


segment_topk_cpp_source = """
std::vector<torch::Tensor> segment_topk_cuda(
    torch::Tensor row_ptr,
    torch::Tensor edge_scores,
    int64_t k
);
"""


segment_topk_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <vector>

template <int K>
__global__ void segment_topk_kernel(
    const int* row_ptr,
    const float* edge_scores,
    float* topk_vals,
    int64_t* topk_idx,
    int num_nodes
) {
    int row = blockIdx.x;
    if (row >= num_nodes || threadIdx.x != 0) {
        return;
    }

    float best_vals[K];
    int best_idx[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        best_vals[i] = -INFINITY;
        best_idx[i] = -1;
    }

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int edge = start; edge < end; ++edge) {
        float value = edge_scores[edge];
        int insert_pos = -1;
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            if (value > best_vals[i]) {
                insert_pos = i;
                break;
            }
        }
        if (insert_pos >= 0) {
            for (int j = K - 1; j > insert_pos; --j) {
                best_vals[j] = best_vals[j - 1];
                best_idx[j] = best_idx[j - 1];
            }
            best_vals[insert_pos] = value;
            best_idx[insert_pos] = edge;
        }
    }

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        topk_vals[row * K + i] = best_vals[i];
        topk_idx[row * K + i] = static_cast<int64_t>(best_idx[i]);
    }
}

std::vector<torch::Tensor> segment_topk_cuda(
    torch::Tensor row_ptr,
    torch::Tensor edge_scores,
    int64_t k
) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be CUDA");
    TORCH_CHECK(edge_scores.is_cuda(), "edge_scores must be CUDA");
    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(edge_scores.scalar_type() == torch::kFloat32, "edge_scores must be float32");
    TORCH_CHECK(k == 4, "This kernel currently supports k=4 only");
    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be 1D");
    TORCH_CHECK(edge_scores.dim() == 1, "edge_scores must be 1D");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(edge_scores.is_contiguous(), "edge_scores must be contiguous");

    int num_nodes = row_ptr.size(0) - 1;
    auto topk_vals = torch::full({num_nodes, k}, -INFINITY, edge_scores.options());
    auto topk_idx = torch::full({num_nodes, k}, -1, torch::TensorOptions().device(edge_scores.device()).dtype(torch::kInt64));

    dim3 grid(num_nodes);
    dim3 block(1);
    segment_topk_kernel<4><<<grid, block>>>(
        row_ptr.data_ptr<int>(),
        edge_scores.data_ptr<float>(),
        topk_vals.data_ptr<float>(),
        topk_idx.data_ptr<int64_t>(),
        num_nodes
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "segment_topk_kernel launch failed: ", cudaGetErrorString(err));

    return {topk_vals, topk_idx};
}
"""


segment_topk_ext = load_inline(
    name="level9_segment_topk_cuda",
    cpp_sources=segment_topk_cpp_source,
    cuda_sources=segment_topk_cuda_source,
    functions=["segment_topk_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.ext = segment_topk_ext

    def forward(self, row_ptr, edge_scores):
        row_ptr = row_ptr.to(dtype=torch.int32).contiguous()
        edge_scores = edge_scores.to(dtype=torch.float32).contiguous()
        return tuple(self.ext.segment_topk_cuda(row_ptr, edge_scores, self.k))
