import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton
import torch_xla2


def add_kernel(x_ref, y_ref, o_ref):
    """Pallas kernel for element-wise addition."""
    o_ref[...] = x_ref[...] + y_ref[...]


def pallas_add(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Launch the Pallas element-wise add kernel."""
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        grid=(a.shape[0],),
        in_specs=[
            pl.BlockSpec((1, a.shape[1]), lambda i: (i, 0)),
            pl.BlockSpec((1, a.shape[1]), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((1, a.shape[1]), lambda i: (i, 0)),
    )(a, b)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Convert PyTorch tensors to JAX arrays, run Pallas kernel, convert back
        a_jax = jnp.array(a.cpu().numpy())
        b_jax = jnp.array(b.cpu().numpy())
        out_jax = pallas_add(a_jax, b_jax)
        return torch.from_numpy(jax.device_get(out_jax)).to(a.device)
