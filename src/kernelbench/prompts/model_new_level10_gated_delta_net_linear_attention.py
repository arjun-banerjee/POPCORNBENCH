import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


gated_delta_cpp_source = """
torch::Tensor gated_delta_net_linear_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta
);
"""


gated_delta_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int MAX_D = 32;

__global__ void gated_delta_net_linear_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int row = blockIdx.x;
    int feat = threadIdx.x;
    int h = row % num_heads;
    int b = row / num_heads;
    if (b >= batch_size || feat >= head_dim) {
        return;
    }

    __shared__ float state[MAX_D * MAX_D];
    for (int idx = feat; idx < head_dim * head_dim; idx += blockDim.x) {
        state[idx] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < seq_len; ++t) {
        float gate_t = gate[(b * seq_len + t) * num_heads + h];
        float beta_t = beta[(b * seq_len + t) * num_heads + h];

        for (int idx = feat; idx < head_dim * head_dim; idx += blockDim.x) {
            int d = idx / head_dim;
            int m = idx % head_dim;
            int kv_base = (((b * seq_len + t) * num_heads + h) * head_dim);
            state[idx] = gate_t * state[idx] + beta_t * k[kv_base + d] * v[kv_base + m];
        }
        __syncthreads();

        float acc = 0.0f;
        int q_base = (((b * seq_len + t) * num_heads + h) * head_dim);
        for (int d = 0; d < head_dim; ++d) {
            acc += q[q_base + d] * state[d * head_dim + feat];
        }
        int out_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + feat;
        out[out_idx] = acc;
        __syncthreads();
    }
}

torch::Tensor gated_delta_net_linear_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(beta.is_cuda(), "beta must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(gate.scalar_type() == torch::kFloat32, "gate must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "beta must be float32");
    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(k.dim() == 4, "k must be 4D");
    TORCH_CHECK(v.dim() == 4, "v must be 4D");
    TORCH_CHECK(gate.dim() == 3, "gate must be 3D");
    TORCH_CHECK(beta.dim() == 3, "beta must be 3D");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");

    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int head_dim = q.size(3);
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");
    TORCH_CHECK(gate.size(0) == batch_size && gate.size(1) == seq_len && gate.size(2) == num_heads, "gate shape mismatch");
    TORCH_CHECK(beta.size(0) == batch_size && beta.size(1) == seq_len && beta.size(2) == num_heads, "beta shape mismatch");
    TORCH_CHECK(head_dim <= MAX_D, "head_dim must be <= 32");

    auto out = torch::zeros_like(q);
    gated_delta_net_linear_attention_kernel<<<batch_size * num_heads, head_dim>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        gate.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gated_delta_net_linear_attention_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


gated_delta_ext = load_inline(
    name="level10_gated_delta_net_linear_attention_cuda",
    cpp_sources=gated_delta_cpp_source,
    cuda_sources=gated_delta_cuda_source,
    functions=["gated_delta_net_linear_attention_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = gated_delta_ext

    def forward(self, q, k, v, gate, beta):
        q = q.to(dtype=torch.float32).contiguous()
        k = k.to(dtype=torch.float32).contiguous()
        v = v.to(dtype=torch.float32).contiguous()
        gate = gate.to(dtype=torch.float32).contiguous()
        beta = beta.to(dtype=torch.float32).contiguous()
        return self.ext.gated_delta_net_linear_attention_cuda(q, k, v, gate, beta)
