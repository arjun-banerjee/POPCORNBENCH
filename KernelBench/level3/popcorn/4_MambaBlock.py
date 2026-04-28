import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Full Mamba block: RMSNorm -> linear expansion into two branches -> causal depthwise Conv1d ->
    SiLU -> selective scan (input-dependent SSM) -> elementwise gate -> linear projection -> residual.
    This is the complete Mamba architecture block as described in Gu & Dao (2023), used in Mamba,
    Caduceus (bidirectional DNA), and various protein/genomic language models.

    Args:
        d_model (int): Model dimension.
        d_state (int): SSM state dimension (N).
        d_conv (int): Causal convolution kernel size.
        expand (int): Expansion factor for inner dimension.
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super(Model, self).__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.activation = nn.SiLU()

        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.proj_B = nn.Linear(self.d_inner, d_state, bias=False)
        self.proj_C = nn.Linear(self.d_inner, d_state, bias=False)
        self.proj_delta = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[..., :L]
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.activation(x_branch)

        N = self.d_state
        delta = torch.nn.functional.softplus(self.proj_delta(x_branch))
        B_t = self.proj_B(x_branch)
        C_t = self.proj_C(x_branch)

        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = delta.unsqueeze(-1) * B_t.unsqueeze(2)
        x_db = x_branch.unsqueeze(-1) * dB

        h = torch.zeros(B, self.d_inner, N, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + x_db[:, t]
            y_t = (h * C_t[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)

        z = self.activation(z)
        y = y * z
        y = self.out_proj(y)
        y = y + residual
        return y

batch_size = 16
d_model = 256
d_state = 16
d_conv = 4
expand = 2
seq_len = 2048

def get_inputs():
    return [torch.randn(batch_size, seq_len, d_model)]

def get_init_inputs():
    return [d_model, d_state, d_conv, expand]