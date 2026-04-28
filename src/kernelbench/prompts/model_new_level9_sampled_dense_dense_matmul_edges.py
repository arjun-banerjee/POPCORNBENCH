import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


sddmm_cpp_source = """
torch::Tensor sampled_sddmm_cuda(
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    torch::Tensor src_feat,
    torch::Tensor dst_feat
);
"""


sddmm_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sampled_sddmm_kernel(
    const int* src_idx,
    const int* dst_idx,
    const float* src_feat,
    const float* dst_feat,
    float* out,
    int num_edges,
    int feat_dim
) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= num_edges) {
        return;
    }

    int src = src_idx[edge];
    int dst = dst_idx[edge];
    float acc = 0.0f;
    for (int f = 0; f < feat_dim; ++f) {
        acc += src_feat[src * feat_dim + f] * dst_feat[dst * feat_dim + f];
    }
    out[edge] = acc;
}

torch::Tensor sampled_sddmm_cuda(
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    torch::Tensor src_feat,
    torch::Tensor dst_feat
) {
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA");
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA");
    TORCH_CHECK(src_feat.is_cuda(), "src_feat must be CUDA");
    TORCH_CHECK(dst_feat.is_cuda(), "dst_feat must be CUDA");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
    TORCH_CHECK(src_feat.scalar_type() == torch::kFloat32, "src_feat must be float32");
    TORCH_CHECK(dst_feat.scalar_type() == torch::kFloat32, "dst_feat must be float32");
    TORCH_CHECK(src_idx.dim() == 1, "src_idx must be 1D");
    TORCH_CHECK(dst_idx.dim() == 1, "dst_idx must be 1D");
    TORCH_CHECK(src_feat.dim() == 2, "src_feat must be 2D");
    TORCH_CHECK(dst_feat.dim() == 2, "dst_feat must be 2D");
    TORCH_CHECK(src_idx.is_contiguous(), "src_idx must be contiguous");
    TORCH_CHECK(dst_idx.is_contiguous(), "dst_idx must be contiguous");
    TORCH_CHECK(src_feat.is_contiguous(), "src_feat must be contiguous");
    TORCH_CHECK(dst_feat.is_contiguous(), "dst_feat must be contiguous");

    int num_edges = src_idx.size(0);
    int feat_dim = src_feat.size(1);
    auto out = torch::zeros({num_edges}, src_feat.options());

    int threads = 256;
    int blocks = (num_edges + threads - 1) / threads;
    sampled_sddmm_kernel<<<blocks, threads>>>(
        src_idx.data_ptr<int>(),
        dst_idx.data_ptr<int>(),
        src_feat.data_ptr<float>(),
        dst_feat.data_ptr<float>(),
        out.data_ptr<float>(),
        num_edges,
        feat_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sampled_sddmm_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""


sddmm_ext = load_inline(
    name="level9_sampled_sddmm_cuda",
    cpp_sources=sddmm_cpp_source,
    cuda_sources=sddmm_cuda_source,
    functions=["sampled_sddmm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = sddmm_ext

    def forward(self, src_idx, dst_idx, src_feat, dst_feat):
        src_idx = src_idx.to(dtype=torch.int32).contiguous()
        dst_idx = dst_idx.to(dtype=torch.int32).contiguous()
        src_feat = src_feat.to(dtype=torch.float32).contiguous()
        dst_feat = dst_feat.to(dtype=torch.float32).contiguous()
        return self.ext.sampled_sddmm_cuda(src_idx, dst_idx, src_feat, dst_feat)
