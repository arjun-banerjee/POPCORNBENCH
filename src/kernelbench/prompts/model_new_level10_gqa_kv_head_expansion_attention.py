import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


gqa_cpp_source = """
torch::Tensor gqa_kv_expansion_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
);
"""


gqa_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void gqa_kv_expansion_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int row = blockIdx.x;
    int feat = threadIdx.x;
    int t = row % seq_len;
    int tmp = row / seq_len;
    int hq = tmp % num_q_heads;
    int b = tmp / num_q_heads;
    if (b >= batch_size || feat >= head_dim) {
        return;
    }

    int group_size = num_q_heads / num_kv_heads;
    int hkv = hq / group_size;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float row_max;
    __shared__ float row_sum;
    if (feat == 0) {
        float max_val = -FLT_MAX;
        for (int s = 0; s <= t; ++s) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int q_idx = (((b * seq_len + t) * num_q_heads + hq) * head_dim) + d;
                int k_idx = (((b * seq_len + s) * num_kv_heads + hkv) * head_dim) + d;
                score += q[q_idx] * k[k_idx];
            }
            score *= scale;
            if (score > max_val) {
                max_val = score;
            }
        }

        float sum_val = 0.0f;
        for (int s = 0; s <= t; ++s) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int q_idx = (((b * seq_len + t) * num_q_heads + hq) * head_dim) + d;
                int k_idx = (((b * seq_len + s) * num_kv_heads + hkv) * head_dim) + d;
                score += q[q_idx] * k[k_idx];
            }
            sum_val += __expf(score * scale - max_val);
        }
        row_max = max_val;
        row_sum = sum_val;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int s = 0; s <= t; ++s) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            int q_idx = (((b * seq_len + t) * num_q_heads + hq) * head_dim) + d;
            int k_idx = (((b * seq_len + s) * num_kv_heads + hkv) * head_dim) + d;
            score += q[q_idx] * k[k_idx];
        }
        float weight = __expf(score * scale - row_max) / row_sum;
        int v_idx = (((b * seq_len + s) * num_kv_heads + hkv) * head_dim) + feat;
        acc += weight * v[v_idx];
    }

    int out_idx = (((b * seq_len + t) * num_q_heads + hq) * head_dim) + feat;
    out[out_idx] = acc;
}

torch::Tensor gqa_kv_expansion_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(k.dim() == 4, "k must be 4D");
    TORCH_CHECK(v.dim() == 4, "v must be 4D");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_q_heads = q.size(2);
    int head_dim = q.size(3);
    int num_kv_heads = k.size(2);
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "batch mismatch");
    TORCH_CHECK(k.size(1) == seq_len && v.size(1) == seq_len, "seq mismatch");
    TORCH_CHECK(k.size(2) == num_kv_heads && v.size(2) == num_kv_heads, "kv head mismatch");
    TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim, "head dim mismatch");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(head_dim <= 256, "head_dim must be <= 256");

    auto out = torch::zeros_like(q);
    int rows = batch_size * seq_len * num_q_heads;
    gqa_kv_expansion_attention_kernel<<<rows, head_dim>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gqa_kv_expansion_attention_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


gqa_ext = load_inline(
    name="level10_gqa_kv_head_expansion_attention_cuda",
    cpp_sources=gqa_cpp_source,
    cuda_sources=gqa_cuda_source,
    functions=["gqa_kv_expansion_attention_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = gqa_ext

    def forward(self, q, k, v):
        q = q.to(dtype=torch.float32).contiguous()
        k = k.to(dtype=torch.float32).contiguous()
        v = v.to(dtype=torch.float32).contiguous()
        return self.ext.gqa_kv_expansion_attention_cuda(q, k, v)
