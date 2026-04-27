import torch
import torch.nn as nn
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


@nki.jit
def nki_add_kernel(a_tensor, b_tensor, out_tensor):
    """NKI kernel for element-wise addition on AWS Trainium/Inferentia."""
    i_p = nl.arange(128)[:, None]
    i_f = nl.arange(a_tensor.shape[1])[None, :]

    a_tile = nl.load(a_tensor[i_p, i_f])
    b_tile = nl.load(b_tensor[i_p, i_f])
    out_tile = nl.add(a_tile, b_tile)
    nl.store(out_tensor[i_p, i_f], value=out_tile)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        out = torch.empty_like(a)
        nki_add_kernel(a, b, out)
        return out
