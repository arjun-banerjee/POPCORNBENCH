import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


graph_edge_softmax_cpp_source = """
torch::Tensor graph_edge_softmax_cuda(torch::Tensor row_ptr, torch::Tensor edge_scores);
"""


graph_edge_softmax_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

namespace {

constexpr int THREADS = 128;
constexpr int NUM_WARPS = THREADS / 32;

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

__inline__ __device__ float block_reduce_max(float val) {
    __shared__ float shared[NUM_WARPS];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? shared[lane] : -FLT_MAX;
    if (warp == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[NUM_WARPS];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? shared[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void graph_edge_softmax_kernel(
    const int* row_ptr,
    const float* edge_scores,
    float* out,
    int num_nodes
) {
    int row = blockIdx.x;
    if (row >= num_nodes) {
        return;
    }

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    int degree = end - start;
    if (degree <= 0) {
        return;
    }

    float local_max = -FLT_MAX;
    for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        float value = edge_scores[start + idx];
        local_max = value > local_max ? value : local_max;
    }
    float row_max = block_reduce_max(local_max);
    __shared__ float shared_row_max;
    if (threadIdx.x == 0) {
        shared_row_max = row_max;
    }
    __syncthreads();
    row_max = shared_row_max;

    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        local_sum += __expf(edge_scores[start + idx] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);
    __shared__ float shared_row_sum;
    if (threadIdx.x == 0) {
        shared_row_sum = row_sum;
    }
    __syncthreads();
    row_sum = shared_row_sum;

    for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        out[start + idx] = __expf(edge_scores[start + idx] - row_max) / row_sum;
    }
}

}  // namespace

torch::Tensor graph_edge_softmax_cuda(torch::Tensor row_ptr, torch::Tensor edge_scores) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(edge_scores.is_cuda(), "edge_scores must be a CUDA tensor");
    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(edge_scores.scalar_type() == torch::kFloat32, "edge_scores must be float32");
    TORCH_CHECK(row_ptr.dim() == 1, "row_ptr must be 1D");
    TORCH_CHECK(edge_scores.dim() == 1, "edge_scores must be 1D");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(edge_scores.is_contiguous(), "edge_scores must be contiguous");

    int num_nodes = row_ptr.size(0) - 1;
    auto out = torch::zeros_like(edge_scores);

    dim3 grid(num_nodes);
    dim3 block(THREADS);
    graph_edge_softmax_kernel<<<grid, block>>>(
        row_ptr.data_ptr<int>(),
        edge_scores.data_ptr<float>(),
        out.data_ptr<float>(),
        num_nodes
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "graph_edge_softmax_kernel launch failed: ", cudaGetErrorString(err));

    return out;
}
"""


graph_edge_softmax = load_inline(
    name="graph_edge_softmax_csr",
    cpp_sources=graph_edge_softmax_cpp_source,
    cuda_sources=graph_edge_softmax_cuda_source,
    functions=["graph_edge_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_cuda_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = graph_edge_softmax

    def forward(self, row_ptr, edge_scores):
        row_ptr = row_ptr.to(dtype=torch.int32).contiguous()
        edge_scores = edge_scores.to(dtype=torch.float32).contiguous()
        return self.kernel.graph_edge_softmax_cuda(row_ptr, edge_scores)
