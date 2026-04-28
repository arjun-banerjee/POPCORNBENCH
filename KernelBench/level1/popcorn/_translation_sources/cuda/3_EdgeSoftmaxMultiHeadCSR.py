import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


edge_softmax_mh_cpp_source = """
torch::Tensor edge_softmax_multihead_cuda(
    torch::Tensor row_ptr,
    torch::Tensor edge_scores
);
"""


edge_softmax_mh_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

constexpr int LANES = 32;

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void edge_softmax_multihead_kernel(
    const int* row_ptr,
    const float* edge_scores,
    float* out,
    int num_nodes,
    int num_heads
) {
    int row = blockIdx.x;
    int head = threadIdx.y;
    int lane = threadIdx.x;
    if (row >= num_nodes || head >= num_heads) {
        return;
    }

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    if (end <= start) {
        return;
    }

    float max_val = -FLT_MAX;
    for (int edge = start + lane; edge < end; edge += blockDim.x) {
        float value = edge_scores[edge * num_heads + head];
        max_val = value > max_val ? value : max_val;
    }
    max_val = warp_reduce_max(max_val);
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    float sum = 0.0f;
    for (int edge = start + lane; edge < end; edge += blockDim.x) {
        sum += __expf(edge_scores[edge * num_heads + head] - max_val);
    }
    sum = warp_reduce_sum(sum);
    sum = __shfl_sync(0xffffffff, sum, 0);

    for (int edge = start + lane; edge < end; edge += blockDim.x) {
        out[edge * num_heads + head] = __expf(edge_scores[edge * num_heads + head] - max_val) / sum;
    }
}

torch::Tensor edge_softmax_multihead_cuda(
    torch::Tensor row_ptr,
    torch::Tensor edge_scores
) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be CUDA");
    TORCH_CHECK(edge_scores.is_cuda(), "edge_scores must be CUDA");
    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(edge_scores.scalar_type() == torch::kFloat32, "edge_scores must be float32");
    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be 1D");
    TORCH_CHECK(edge_scores.dim() == 2, "edge_scores must be 2D");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(edge_scores.is_contiguous(), "edge_scores must be contiguous");

    int num_nodes = row_ptr.size(0) - 1;
    int num_heads = edge_scores.size(1);
    TORCH_CHECK(num_heads <= 8, "num_heads must be <= 8 for this kernel launch configuration");
    auto out = torch::zeros_like(edge_scores);

    dim3 grid(num_nodes);
    dim3 block(LANES, num_heads);
    edge_softmax_multihead_kernel<<<grid, block>>>(
        row_ptr.data_ptr<int>(),
        edge_scores.data_ptr<float>(),
        out.data_ptr<float>(),
        num_nodes,
        num_heads
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "edge_softmax_multihead_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


edge_softmax_mh_ext = load_inline(
    name="level9_edge_softmax_multihead_cuda",
    cpp_sources=edge_softmax_mh_cpp_source,
    cuda_sources=edge_softmax_mh_cuda_source,
    functions=["edge_softmax_multihead_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = edge_softmax_mh_ext

    def forward(self, row_ptr, edge_scores):
        row_ptr = row_ptr.to(dtype=torch.int32).contiguous()
        edge_scores = edge_scores.to(dtype=torch.float32).contiguous()
        return self.ext.edge_softmax_multihead_cuda(row_ptr, edge_scores)
