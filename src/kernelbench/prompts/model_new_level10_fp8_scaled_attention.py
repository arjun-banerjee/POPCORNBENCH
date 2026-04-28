import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


fp8_scaled_cpp_source = """
torch::Tensor fp8_scaled_attention_cuda(
    torch::Tensor q_q,
    torch::Tensor k_q,
    torch::Tensor v_q,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale
);
"""


fp8_scaled_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void fp8_scaled_attention_kernel(
    const int8_t* q_q,
    const int8_t* k_q,
    const int8_t* v_q,
    const float* q_scale,
    const float* k_scale,
    const float* v_scale,
    float* out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
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
    const float q_s = q_scale[h];
    const float k_s = k_scale[h];
    const float v_s = v_scale[h];

    __shared__ float row_max;
    __shared__ float row_sum;
    if (feat == 0) {
        float max_val = -FLT_MAX;
        for (int s = 0; s <= t; ++s) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int q_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + d;
                int k_idx = (((b * seq_len + s) * num_heads + h) * head_dim) + d;
                score += (static_cast<float>(q_q[q_idx]) * q_s) * (static_cast<float>(k_q[k_idx]) * k_s);
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
                int q_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + d;
                int k_idx = (((b * seq_len + s) * num_heads + h) * head_dim) + d;
                score += (static_cast<float>(q_q[q_idx]) * q_s) * (static_cast<float>(k_q[k_idx]) * k_s);
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
            int q_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + d;
            int k_idx = (((b * seq_len + s) * num_heads + h) * head_dim) + d;
            score += (static_cast<float>(q_q[q_idx]) * q_s) * (static_cast<float>(k_q[k_idx]) * k_s);
        }
        float weight = __expf(score * scale - row_max) / row_sum;
        int v_idx = (((b * seq_len + s) * num_heads + h) * head_dim) + feat;
        acc += weight * (static_cast<float>(v_q[v_idx]) * v_s);
    }

    int out_idx = (((b * seq_len + t) * num_heads + h) * head_dim) + feat;
    out[out_idx] = acc;
}

torch::Tensor fp8_scaled_attention_cuda(
    torch::Tensor q_q,
    torch::Tensor k_q,
    torch::Tensor v_q,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale
) {
    TORCH_CHECK(q_q.is_cuda(), "q_q must be CUDA");
    TORCH_CHECK(k_q.is_cuda(), "k_q must be CUDA");
    TORCH_CHECK(v_q.is_cuda(), "v_q must be CUDA");
    TORCH_CHECK(q_scale.is_cuda(), "q_scale must be CUDA");
    TORCH_CHECK(k_scale.is_cuda(), "k_scale must be CUDA");
    TORCH_CHECK(v_scale.is_cuda(), "v_scale must be CUDA");
    TORCH_CHECK(q_q.scalar_type() == torch::kInt8, "q_q must be int8");
    TORCH_CHECK(k_q.scalar_type() == torch::kInt8, "k_q must be int8");
    TORCH_CHECK(v_q.scalar_type() == torch::kInt8, "v_q must be int8");
    TORCH_CHECK(q_scale.scalar_type() == torch::kFloat32, "q_scale must be float32");
    TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32, "k_scale must be float32");
    TORCH_CHECK(v_scale.scalar_type() == torch::kFloat32, "v_scale must be float32");
    TORCH_CHECK(q_q.dim() == 4, "q_q must be 4D");
    TORCH_CHECK(k_q.dim() == 4, "k_q must be 4D");
    TORCH_CHECK(v_q.dim() == 4, "v_q must be 4D");
    TORCH_CHECK(q_q.is_contiguous(), "q_q must be contiguous");
    TORCH_CHECK(k_q.is_contiguous(), "k_q must be contiguous");
    TORCH_CHECK(v_q.is_contiguous(), "v_q must be contiguous");
    TORCH_CHECK(q_scale.is_contiguous(), "q_scale must be contiguous");
    TORCH_CHECK(k_scale.is_contiguous(), "k_scale must be contiguous");
    TORCH_CHECK(v_scale.is_contiguous(), "v_scale must be contiguous");

    int batch_size = q_q.size(0);
    int seq_len = q_q.size(1);
    int num_heads = q_q.size(2);
    int head_dim = q_q.size(3);
    TORCH_CHECK(k_q.sizes() == q_q.sizes(), "k_q shape mismatch");
    TORCH_CHECK(v_q.sizes() == q_q.sizes(), "v_q shape mismatch");
    TORCH_CHECK(q_scale.numel() == num_heads, "q_scale shape mismatch");
    TORCH_CHECK(k_scale.numel() == num_heads, "k_scale shape mismatch");
    TORCH_CHECK(v_scale.numel() == num_heads, "v_scale shape mismatch");
    TORCH_CHECK(head_dim <= 256, "head_dim must be <= 256");

    auto out = torch::zeros({batch_size, seq_len, num_heads, head_dim}, q_scale.options());
    int rows = batch_size * seq_len * num_heads;
    fp8_scaled_attention_kernel<<<rows, head_dim>>>(
        q_q.data_ptr<int8_t>(),
        k_q.data_ptr<int8_t>(),
        v_q.data_ptr<int8_t>(),
        q_scale.data_ptr<float>(),
        k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fp8_scaled_attention_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


fp8_scaled_ext = load_inline(
    name="level10_fp8_scaled_attention_cuda",
    cpp_sources=fp8_scaled_cpp_source,
    cuda_sources=fp8_scaled_cuda_source,
    functions=["fp8_scaled_attention_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = fp8_scaled_ext

    def forward(self, q_q, k_q, v_q, q_scale, k_scale, v_scale):
        q_q = q_q.to(dtype=torch.int8).contiguous()
        k_q = k_q.to(dtype=torch.int8).contiguous()
        v_q = v_q.to(dtype=torch.int8).contiguous()
        q_scale = q_scale.to(dtype=torch.float32).contiguous()
        k_scale = k_scale.to(dtype=torch.float32).contiguous()
        v_scale = v_scale.to(dtype=torch.float32).contiguous()
        return self.ext.fp8_scaled_attention_cuda(q_q, k_q, v_q, q_scale, k_scale, v_scale)
