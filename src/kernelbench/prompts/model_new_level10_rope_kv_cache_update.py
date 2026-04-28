import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


rope_kv_cpp_source = """
torch::Tensor rope_kv_cache_update_cuda(
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor cache_k,
    torch::Tensor cache_v,
    torch::Tensor positions
);
"""


rope_kv_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rope_rotate_write_k_kernel(
    const float* k_new,
    const float* cos,
    const float* sin,
    const int* positions,
    float* out_k,
    int batch_size,
    int update_len,
    int cache_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * update_len * num_heads * (head_dim / 2);
    if (idx >= total) {
        return;
    }

    int half = head_dim / 2;
    int pair = idx % half;
    int tmp = idx / half;
    int head = tmp % num_heads;
    tmp /= num_heads;
    int t = tmp % update_len;
    int b = tmp / update_len;

    int pos = positions[b * update_len + t];
    int src_base = (((b * update_len + t) * num_heads + head) * head_dim);
    int dst_base = (((b * cache_len + pos) * num_heads + head) * head_dim);
    float c = cos[pos * half + pair];
    float s = sin[pos * half + pair];
    float even = k_new[src_base + 2 * pair];
    float odd = k_new[src_base + 2 * pair + 1];
    out_k[dst_base + 2 * pair] = even * c - odd * s;
    out_k[dst_base + 2 * pair + 1] = even * s + odd * c;
}

__global__ void cache_v_write_kernel(
    const float* v_new,
    const int* positions,
    float* out_v,
    int batch_size,
    int update_len,
    int cache_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * update_len * num_heads * head_dim;
    if (idx >= total) {
        return;
    }

    int feat = idx % head_dim;
    int tmp = idx / head_dim;
    int head = tmp % num_heads;
    tmp /= num_heads;
    int t = tmp % update_len;
    int b = tmp / update_len;

    int pos = positions[b * update_len + t];
    int src_idx = (((b * update_len + t) * num_heads + head) * head_dim) + feat;
    int dst_idx = (((b * cache_len + pos) * num_heads + head) * head_dim) + feat;
    out_v[dst_idx] = v_new[src_idx];
}

torch::Tensor rope_kv_cache_update_cuda(
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor cache_k,
    torch::Tensor cache_v,
    torch::Tensor positions
) {
    TORCH_CHECK(k_new.is_cuda() && v_new.is_cuda(), "k_new and v_new must be CUDA");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "cos and sin must be CUDA");
    TORCH_CHECK(cache_k.is_cuda() && cache_v.is_cuda(), "cache_k and cache_v must be CUDA");
    TORCH_CHECK(positions.is_cuda(), "positions must be CUDA");
    TORCH_CHECK(k_new.scalar_type() == torch::kFloat32, "k_new must be float32");
    TORCH_CHECK(v_new.scalar_type() == torch::kFloat32, "v_new must be float32");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32, "cos and sin must be float32");
    TORCH_CHECK(cache_k.scalar_type() == torch::kFloat32 && cache_v.scalar_type() == torch::kFloat32, "cache tensors must be float32");
    TORCH_CHECK(positions.scalar_type() == torch::kInt32, "positions must be int32");

    auto out_k = cache_k.clone();
    auto out_v = cache_v.clone();
    int batch_size = k_new.size(0);
    int update_len = k_new.size(1);
    int cache_len = cache_k.size(1);
    int num_heads = k_new.size(2);
    int head_dim = k_new.size(3);

    int total_k = batch_size * update_len * num_heads * (head_dim / 2);
    int total_v = batch_size * update_len * num_heads * head_dim;
    int block = 256;
    rope_rotate_write_k_kernel<<<(total_k + block - 1) / block, block>>>(
        k_new.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        positions.data_ptr<int>(),
        out_k.data_ptr<float>(),
        batch_size,
        update_len,
        cache_len,
        num_heads,
        head_dim
    );
    cache_v_write_kernel<<<(total_v + block - 1) / block, block>>>(
        v_new.data_ptr<float>(),
        positions.data_ptr<int>(),
        out_v.data_ptr<float>(),
        batch_size,
        update_len,
        cache_len,
        num_heads,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rope_kv cache kernels launch failed: ", cudaGetErrorString(err));
    return torch::stack({out_k, out_v}, 0);
}
"""


rope_kv_ext = load_inline(
    name="level10_rope_kv_cache_update_cuda",
    cpp_sources=rope_kv_cpp_source,
    cuda_sources=rope_kv_cuda_source,
    functions=["rope_kv_cache_update_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = rope_kv_ext

    def forward(self, k_new, v_new, cos, sin, cache_k, cache_v, positions):
        return self.ext.rope_kv_cache_update_cuda(
            k_new.to(torch.float32).contiguous(),
            v_new.to(torch.float32).contiguous(),
            cos.to(torch.float32).contiguous(),
            sin.to(torch.float32).contiguous(),
            cache_k.to(torch.float32).contiguous(),
            cache_v.to(torch.float32).contiguous(),
            positions.to(torch.int32).contiguous(),
        )
