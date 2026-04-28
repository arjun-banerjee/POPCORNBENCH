import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


kimi_delta_cpp_source = """
torch::Tensor kimi_delta_attention_channelwise_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate
);
"""


kimi_delta_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int MAX_D = 32;

__global__ void kimi_delta_attention_channelwise_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
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
        int kv_base = (((b * seq_len + t) * num_heads + h) * head_dim);
        int gate_base = (((b * seq_len + t) * num_heads + h) * head_dim);

        for (int idx = feat; idx < head_dim * head_dim; idx += blockDim.x) {
            int d = idx / head_dim;
            int m = idx % head_dim;
            state[idx] = gate[gate_base + m] * state[idx] + k[kv_base + d] * v[kv_base + m];
        }
        __syncthreads();

        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            acc += q[kv_base + d] * state[d * head_dim + feat];
        }
        int out_idx = kv_base + feat;
        out[out_idx] = acc;
        __syncthreads();
    }
}

torch::Tensor kimi_delta_attention_channelwise_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(gate.scalar_type() == torch::kFloat32, "gate must be float32");
    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(k.dim() == 4, "k must be 4D");
    TORCH_CHECK(v.dim() == 4, "v must be 4D");
    TORCH_CHECK(gate.dim() == 4, "gate must be 4D");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");

    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int head_dim = q.size(3);
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");
    TORCH_CHECK(gate.sizes() == q.sizes(), "gate shape mismatch");
    TORCH_CHECK(head_dim <= MAX_D, "head_dim must be <= 32");

    auto out = torch::zeros_like(q);
    kimi_delta_attention_channelwise_kernel<<<batch_size * num_heads, head_dim>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        gate.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kimi_delta_attention_channelwise_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


kimi_delta_ext = load_inline(
    name="level10_kimi_delta_attention_channelwise_cuda",
    cpp_sources=kimi_delta_cpp_source,
    cuda_sources=kimi_delta_cuda_source,
    functions=["kimi_delta_attention_channelwise_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = kimi_delta_ext

    def forward(self, q, k, v, gate):
        q = q.to(dtype=torch.float32).contiguous()
        k = k.to(dtype=torch.float32).contiguous()
        v = v.to(dtype=torch.float32).contiguous()
        gate = gate.to(dtype=torch.float32).contiguous()
        return self.ext.kimi_delta_attention_channelwise_cuda(q, k, v, gate)
