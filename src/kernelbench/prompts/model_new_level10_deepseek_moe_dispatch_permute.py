import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


moe_dispatch_cpp_source = """
torch::Tensor moe_dispatch_permute_cuda(
    torch::Tensor token_hidden,
    torch::Tensor expert_idx,
    torch::Tensor slot_idx,
    torch::Tensor expert_offsets
);
"""


moe_dispatch_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void moe_dispatch_permute_kernel(
    const float* token_hidden,
    const int* expert_idx,
    const int* slot_idx,
    const int* expert_offsets,
    float* out,
    int num_tokens,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_dim;
    if (idx >= total) {
        return;
    }

    int feat = idx % hidden_dim;
    int token = idx / hidden_dim;
    int expert = expert_idx[token];
    int row = expert_offsets[expert] + slot_idx[token];
    out[row * hidden_dim + feat] = token_hidden[token * hidden_dim + feat];
}

torch::Tensor moe_dispatch_permute_cuda(
    torch::Tensor token_hidden,
    torch::Tensor expert_idx,
    torch::Tensor slot_idx,
    torch::Tensor expert_offsets
) {
    TORCH_CHECK(token_hidden.is_cuda(), "token_hidden must be CUDA");
    TORCH_CHECK(expert_idx.is_cuda() && slot_idx.is_cuda() && expert_offsets.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(token_hidden.scalar_type() == torch::kFloat32, "token_hidden must be float32");
    TORCH_CHECK(expert_idx.scalar_type() == torch::kInt32, "expert_idx must be int32");
    TORCH_CHECK(slot_idx.scalar_type() == torch::kInt32, "slot_idx must be int32");
    TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "expert_offsets must be int32");

    int rows = expert_offsets[expert_offsets.size(0) - 1].item<int>();
    int hidden_dim = token_hidden.size(1);
    int num_tokens = token_hidden.size(0);
    auto out = torch::zeros({rows, hidden_dim}, token_hidden.options());
    int block = 256;
    int total = num_tokens * hidden_dim;
    moe_dispatch_permute_kernel<<<(total + block - 1) / block, block>>>(
        token_hidden.data_ptr<float>(),
        expert_idx.data_ptr<int>(),
        slot_idx.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        out.data_ptr<float>(),
        num_tokens,
        hidden_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_dispatch_permute_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


moe_dispatch_ext = load_inline(
    name="level10_deepseek_moe_dispatch_permute_cuda",
    cpp_sources=moe_dispatch_cpp_source,
    cuda_sources=moe_dispatch_cuda_source,
    functions=["moe_dispatch_permute_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = moe_dispatch_ext

    def forward(self, token_hidden, expert_idx, slot_idx, expert_offsets):
        return self.ext.moe_dispatch_permute_cuda(
            token_hidden.to(torch.float32).contiguous(),
            expert_idx.to(torch.int32).contiguous(),
            slot_idx.to(torch.int32).contiguous(),
            expert_offsets.to(torch.int32).contiguous(),
        )
