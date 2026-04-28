import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


mla_lora_cpp_source = """
torch::Tensor mla_lora_down_cuda(torch::Tensor hidden, torch::Tensor w_down);
torch::Tensor mla_lora_up_cuda(torch::Tensor latent, torch::Tensor w_up_k, torch::Tensor w_up_v);
"""


mla_lora_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mla_lora_down_kernel(
    const float* hidden,
    const float* w_down,
    float* latent,
    int rows,
    int model_dim,
    int rank
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || col >= rank) {
        return;
    }

    double acc = 0.0;
    for (int k = 0; k < model_dim; ++k) {
        acc += hidden[row * model_dim + k] * w_down[k * rank + col];
    }
    latent[row * rank + col] = static_cast<float>(acc);
}

__global__ void mla_lora_up_kernel(
    const float* latent,
    const float* w_up_k,
    const float* w_up_v,
    float* packed,
    int rows,
    int rank,
    int out_dim
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || col >= (2 * out_dim)) {
        return;
    }

    const float* w = col < out_dim ? w_up_k : w_up_v;
    int out_col = col < out_dim ? col : (col - out_dim);
    double acc = 0.0;
    for (int r = 0; r < rank; ++r) {
        acc += latent[row * rank + r] * w[r * out_dim + out_col];
    }
    packed[row * (2 * out_dim) + col] = static_cast<float>(acc);
}

torch::Tensor mla_lora_down_cuda(torch::Tensor hidden, torch::Tensor w_down) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(w_down.is_cuda(), "w_down must be CUDA");
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32, "hidden must be float32");
    TORCH_CHECK(w_down.scalar_type() == torch::kFloat32, "w_down must be float32");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D");
    TORCH_CHECK(w_down.dim() == 2, "w_down must be 2D");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(w_down.is_contiguous(), "w_down must be contiguous");

    int rows = hidden.size(0);
    int model_dim = hidden.size(1);
    int rank = w_down.size(1);
    TORCH_CHECK(w_down.size(0) == model_dim, "w_down shape mismatch");

    auto latent = torch::zeros({rows, rank}, hidden.options());
    dim3 block(128);
    dim3 grid(rows, (rank + block.x - 1) / block.x);
    mla_lora_down_kernel<<<grid, block>>>(
        hidden.data_ptr<float>(),
        w_down.data_ptr<float>(),
        latent.data_ptr<float>(),
        rows,
        model_dim,
        rank
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mla_lora_down_kernel launch failed: ", cudaGetErrorString(err));
    return latent;
}

torch::Tensor mla_lora_up_cuda(torch::Tensor latent, torch::Tensor w_up_k, torch::Tensor w_up_v) {
    TORCH_CHECK(latent.is_cuda(), "latent must be CUDA");
    TORCH_CHECK(w_up_k.is_cuda(), "w_up_k must be CUDA");
    TORCH_CHECK(w_up_v.is_cuda(), "w_up_v must be CUDA");
    TORCH_CHECK(latent.scalar_type() == torch::kFloat32, "latent must be float32");
    TORCH_CHECK(w_up_k.scalar_type() == torch::kFloat32, "w_up_k must be float32");
    TORCH_CHECK(w_up_v.scalar_type() == torch::kFloat32, "w_up_v must be float32");
    TORCH_CHECK(latent.dim() == 2, "latent must be 2D");
    TORCH_CHECK(w_up_k.dim() == 2, "w_up_k must be 2D");
    TORCH_CHECK(w_up_v.dim() == 2, "w_up_v must be 2D");
    TORCH_CHECK(latent.is_contiguous(), "latent must be contiguous");
    TORCH_CHECK(w_up_k.is_contiguous(), "w_up_k must be contiguous");
    TORCH_CHECK(w_up_v.is_contiguous(), "w_up_v must be contiguous");

    int rows = latent.size(0);
    int rank = latent.size(1);
    int out_dim = w_up_k.size(1);
    TORCH_CHECK(w_up_k.size(0) == rank, "w_up_k shape mismatch");
    TORCH_CHECK(w_up_v.size(0) == rank, "w_up_v shape mismatch");
    TORCH_CHECK(w_up_v.size(1) == out_dim, "w_up_v shape mismatch");

    auto packed = torch::zeros({rows, 2 * out_dim}, latent.options());
    dim3 block(128);
    dim3 grid(rows, ((2 * out_dim) + block.x - 1) / block.x);
    mla_lora_up_kernel<<<grid, block>>>(
        latent.data_ptr<float>(),
        w_up_k.data_ptr<float>(),
        w_up_v.data_ptr<float>(),
        packed.data_ptr<float>(),
        rows,
        rank,
        out_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mla_lora_up_kernel launch failed: ", cudaGetErrorString(err));
    return packed;
}
"""


mla_lora_ext = load_inline(
    name="level10_deepseek_mla_lora_expansion_cuda",
    cpp_sources=mla_lora_cpp_source,
    cuda_sources=mla_lora_cuda_source,
    functions=["mla_lora_down_cuda", "mla_lora_up_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = mla_lora_ext

    def forward(self, hidden, w_down, w_up_k, w_up_v):
        hidden = hidden.to(dtype=torch.float32).contiguous()
        w_down = w_down.to(dtype=torch.float32).contiguous()
        w_up_k = w_up_k.to(dtype=torch.float32).contiguous()
        w_up_v = w_up_v.to(dtype=torch.float32).contiguous()

        batch_size, seq_len, model_dim = hidden.shape
        rows = batch_size * seq_len
        latent = self.ext.mla_lora_down_cuda(hidden.view(rows, model_dim), w_down)
        packed = self.ext.mla_lora_up_cuda(latent, w_up_k, w_up_v)

        kv_heads = 4
        head_dim = w_up_k.shape[1] // kv_heads
        packed = packed.view(batch_size, seq_len, 2, kv_heads, head_dim)
        return packed
