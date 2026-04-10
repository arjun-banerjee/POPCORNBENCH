"""
Three-stage MLP on packed activations: h1 = relu(h0 @ W1), h2 = relu(h1 @ W2), out = h2 @ W3.
Dimensions are fixed; weights are trainable parameters with default init.
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

B = 16
H0 = 48
H1 = 64
H2 = 40
H3 = 24


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(H0, H1, bias=False)
        self.W2 = nn.Linear(H1, H2, bias=False)
        self.W3 = nn.Linear(H2, H3, bias=False)

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.W1(h0))
        h2 = torch.relu(self.W2(h1))
        return self.W3(h2)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(14)
    return [torch.randn(B, H0, device=dev, generator=g)]


def get_init_inputs():
    return []
