import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


moe_combine_cpp_source = """
torch::Tensor moe_combine_scatter_cuda(
    torch::Tensor expert_hidden,
    torch::Tensor token_idx,
    torch::Tensor gates,
    int64_t num_tokens
);
"""


moe_combine_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void moe_combine_scatter_kernel(
    const float* expert_hidden,
    const int* token_idx,
    const float* gates,
    float* out,
    int num_rows,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * hidden_dim;
    if (idx >= total) {
        return;
    }

    int feat = idx % hidden_dim;
    int row = idx / hidden_dim;
    int token = token_idx[row];
    atomicAdd(&out[token * hidden_dim + feat], gates[row] * expert_hidden[row * hidden_dim + feat]);
}

torch::Tensor moe_combine_scatter_cuda(
    torch::Tensor expert_hidden,
    torch::Tensor token_idx,
    torch::Tensor gates,
    int64_t num_tokens
) {
    TORCH_CHECK(expert_hidden.is_cuda() && token_idx.is_cuda() && gates.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(expert_hidden.scalar_type() == torch::kFloat32, "expert_hidden must be float32");
    TORCH_CHECK(token_idx.scalar_type() == torch::kInt32, "token_idx must be int32");
    TORCH_CHECK(gates.scalar_type() == torch::kFloat32, "gates must be float32");

    int num_rows = expert_hidden.size(0);
    int hidden_dim = expert_hidden.size(1);
    auto out = torch::zeros({num_tokens, hidden_dim}, expert_hidden.options());
    int block = 256;
    int total = num_rows * hidden_dim;
    moe_combine_scatter_kernel<<<(total + block - 1) / block, block>>>(
        expert_hidden.data_ptr<float>(),
        token_idx.data_ptr<int>(),
        gates.data_ptr<float>(),
        out.data_ptr<float>(),
        num_rows,
        hidden_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_combine_scatter_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


moe_combine_ext = load_inline(
    name="level10_deepseek_moe_combine_scatter_cuda",
    cpp_sources=moe_combine_cpp_source,
    cuda_sources=moe_combine_cuda_source,
    functions=["moe_combine_scatter_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = moe_combine_ext

    def forward(self, expert_hidden, token_idx, gates, num_tokens):
        return self.ext.moe_combine_scatter_cuda(
            expert_hidden.to(torch.float32).contiguous(),
            token_idx.to(torch.int32).contiguous(),
            gates.to(torch.float32).contiguous(),
            int(num_tokens),
        )
