import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


moe_grounded_cpp_source = """
torch::Tensor moe_grounded_top2_cuda(
    torch::Tensor token_hidden,
    torch::Tensor router_logits,
    torch::Tensor expert_ground,
    double alpha
);
"""


moe_grounded_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

constexpr int MAX_EXPERTS = 64;

__global__ void moe_grounded_top2_kernel(
    const float* token_hidden,
    const float* router_logits,
    const float* expert_ground,
    float* out,
    int num_tokens,
    int hidden_dim,
    int num_experts,
    float alpha
) {
    int token = blockIdx.x;
    int expert = threadIdx.x;
    if (token >= num_tokens || expert >= num_experts) {
        return;
    }

    __shared__ float scores[MAX_EXPERTS];
    float dot = 0.0f;
    for (int d = 0; d < hidden_dim; ++d) {
        dot += token_hidden[token * hidden_dim + d] * expert_ground[expert * hidden_dim + d];
    }
    scores[expert] = router_logits[token * num_experts + expert] + alpha * dot;
    __syncthreads();

    if (expert == 0) {
        float best_val0 = -FLT_MAX;
        float best_val1 = -FLT_MAX;
        int best_idx0 = 0;
        int best_idx1 = 1;
        for (int e = 0; e < num_experts; ++e) {
            float score = scores[e];
            if (score > best_val0) {
                best_val1 = best_val0;
                best_idx1 = best_idx0;
                best_val0 = score;
                best_idx0 = e;
            } else if (score > best_val1) {
                best_val1 = score;
                best_idx1 = e;
            }
        }

        float max_val = best_val0 > best_val1 ? best_val0 : best_val1;
        float w0 = __expf(best_val0 - max_val);
        float w1 = __expf(best_val1 - max_val);
        float inv = 1.0f / (w0 + w1);

        out[token * 4 + 0] = static_cast<float>(best_idx0);
        out[token * 4 + 1] = w0 * inv;
        out[token * 4 + 2] = static_cast<float>(best_idx1);
        out[token * 4 + 3] = w1 * inv;
    }
}

torch::Tensor moe_grounded_top2_cuda(
    torch::Tensor token_hidden,
    torch::Tensor router_logits,
    torch::Tensor expert_ground,
    double alpha
) {
    TORCH_CHECK(token_hidden.is_cuda(), "token_hidden must be CUDA");
    TORCH_CHECK(router_logits.is_cuda(), "router_logits must be CUDA");
    TORCH_CHECK(expert_ground.is_cuda(), "expert_ground must be CUDA");
    TORCH_CHECK(token_hidden.scalar_type() == torch::kFloat32, "token_hidden must be float32");
    TORCH_CHECK(router_logits.scalar_type() == torch::kFloat32, "router_logits must be float32");
    TORCH_CHECK(expert_ground.scalar_type() == torch::kFloat32, "expert_ground must be float32");
    TORCH_CHECK(token_hidden.dim() == 2, "token_hidden must be 2D");
    TORCH_CHECK(router_logits.dim() == 2, "router_logits must be 2D");
    TORCH_CHECK(expert_ground.dim() == 2, "expert_ground must be 2D");
    TORCH_CHECK(token_hidden.is_contiguous(), "token_hidden must be contiguous");
    TORCH_CHECK(router_logits.is_contiguous(), "router_logits must be contiguous");
    TORCH_CHECK(expert_ground.is_contiguous(), "expert_ground must be contiguous");

    int num_tokens = token_hidden.size(0);
    int hidden_dim = token_hidden.size(1);
    int num_experts = router_logits.size(1);
    TORCH_CHECK(expert_ground.size(0) == num_experts, "expert count mismatch");
    TORCH_CHECK(expert_ground.size(1) == hidden_dim, "expert_ground shape mismatch");
    TORCH_CHECK(num_experts <= MAX_EXPERTS, "num_experts must be <= 64");

    auto out = torch::zeros({num_tokens, 2, 2}, token_hidden.options());
    moe_grounded_top2_kernel<<<num_tokens, num_experts>>>(
        token_hidden.data_ptr<float>(),
        router_logits.data_ptr<float>(),
        expert_ground.data_ptr<float>(),
        out.data_ptr<float>(),
        num_tokens,
        hidden_dim,
        num_experts,
        static_cast<float>(alpha)
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_grounded_top2_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


moe_grounded_ext = load_inline(
    name="level10_deepseek_moe_grounded_top2_cuda",
    cpp_sources=moe_grounded_cpp_source,
    cuda_sources=moe_grounded_cuda_source,
    functions=["moe_grounded_top2_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = moe_grounded_ext

    def forward(self, token_hidden, router_logits, expert_ground, alpha):
        token_hidden = token_hidden.to(dtype=torch.float32).contiguous()
        router_logits = router_logits.to(dtype=torch.float32).contiguous()
        expert_ground = expert_ground.to(dtype=torch.float32).contiguous()
        return self.ext.moe_grounded_top2_cuda(token_hidden, router_logits, expert_ground, float(alpha))
