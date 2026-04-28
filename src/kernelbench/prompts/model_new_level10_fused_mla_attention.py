import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


fused_mla_cpp_source = """
torch::Tensor fused_mla_attention_cuda(
    torch::Tensor q,
    torch::Tensor kv_latent,
    torch::Tensor w_up_k,
    torch::Tensor w_up_v
);
"""


fused_mla_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void fused_mla_attention_kernel(
    const float* q,
    const float* kv_latent,
    const float* w_up_k,
    const float* w_up_v,
    float* out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rank
) {
    int row = blockIdx.x;
    int feat = threadIdx.x;
    int t = row % seq_len;
    int tmp = row / seq_len;
    int h = tmp % num_heads;
    int b = tmp / num_heads;
    if (b >= batch_size || feat >= head_dim) {
        return;
    }

    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float row_max;
    __shared__ float row_sum;
    if (feat == 0) {
        float max_val = -FLT_MAX;
        for (int s = 0; s <= t; ++s) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float k_val = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    int lat_idx = ((b * seq_len + s) * rank) + r;
                    int wk_idx = ((r * num_heads + h) * head_dim) + d;
                    k_val += kv_latent[lat_idx] * w_up_k[wk_idx];
                }
                int q_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + d;
                score += q[q_idx] * k_val;
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
                float k_val = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    int lat_idx = ((b * seq_len + s) * rank) + r;
                    int wk_idx = ((r * num_heads + h) * head_dim) + d;
                    k_val += kv_latent[lat_idx] * w_up_k[wk_idx];
                }
                int q_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + d;
                score += q[q_idx] * k_val;
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
            float k_val = 0.0f;
            for (int r = 0; r < rank; ++r) {
                int lat_idx = ((b * seq_len + s) * rank) + r;
                int wk_idx = ((r * num_heads + h) * head_dim) + d;
                k_val += kv_latent[lat_idx] * w_up_k[wk_idx];
            }
            int q_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + d;
            score += q[q_idx] * k_val;
        }
        float weight = __expf(score * scale - row_max) / row_sum;

        float v_val = 0.0f;
        for (int r = 0; r < rank; ++r) {
            int lat_idx = ((b * seq_len + s) * rank) + r;
            int wv_idx = ((r * num_heads + h) * head_dim) + feat;
            v_val += kv_latent[lat_idx] * w_up_v[wv_idx];
        }
        acc += weight * v_val;
    }
    int out_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + feat;
    out[out_idx] = acc;
}

torch::Tensor fused_mla_attention_cuda(
    torch::Tensor q,
    torch::Tensor kv_latent,
    torch::Tensor w_up_k,
    torch::Tensor w_up_v
) {
    TORCH_CHECK(q.is_cuda() && kv_latent.is_cuda() && w_up_k.is_cuda() && w_up_v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(kv_latent.scalar_type() == torch::kFloat32, "kv_latent must be float32");
    TORCH_CHECK(w_up_k.scalar_type() == torch::kFloat32 && w_up_v.scalar_type() == torch::kFloat32, "weights must be float32");
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int num_heads = q.size(2);
    int head_dim = q.size(3);
    int rank = kv_latent.size(2);
    TORCH_CHECK(head_dim <= 256, "head_dim must be <= 256");

    auto out = torch::zeros_like(q);
    int rows = batch_size * seq_len * num_heads;
    fused_mla_attention_kernel<<<rows, head_dim>>>(
        q.data_ptr<float>(),
        kv_latent.data_ptr<float>(),
        w_up_k.data_ptr<float>(),
        w_up_v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        rank
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_mla_attention_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


fused_mla_ext = load_inline(
    name="level10_fused_mla_attention_cuda",
    cpp_sources=fused_mla_cpp_source,
    cuda_sources=fused_mla_cuda_source,
    functions=["fused_mla_attention_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = fused_mla_ext

    def forward(self, q, kv_latent, w_up_k, w_up_v):
        return self.ext.fused_mla_attention_cuda(
            q.to(torch.float32).contiguous(),
            kv_latent.to(torch.float32).contiguous(),
            w_up_k.to(torch.float32).contiguous(),
            w_up_v.to(torch.float32).contiguous(),
        )
