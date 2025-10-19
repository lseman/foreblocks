import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Lite RMSNorm over last dim."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _make_activation(name: str):
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


def _make_norm(kind: str, d: int):
    kind = kind.lower()
    if kind == "layer":
        return nn.LayerNorm(d)
    if kind == "rms":
        return RMSNorm(d)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {kind}")


class GRN(nn.Module):
    """
    Gated Residual Network (TFT-style), improved:
      - Fused GLU (nn.GLU) for clarity & speed
      - Optional context with safe broadcasting
      - Selectable activation: silu/gelu/elu
      - Selectable norm: layer/rms/none
      - Residual scaling to stabilize deep stacks
      - TorchScript-friendly, works on any shape [..., D]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int | None = None,
        output_size: int | None = None,
        dropout: float = 0.0,
        context_size: int | None = None,
        *,
        activation: str = "silu",      # "silu" | "gelu" | "elu"
        norm: str = "layer",           # "layer" | "rms" | "none"
        residual_scale: float = 1.0,   # scale residual before adding
        bias: bool = True,             # biases in linear layers
    ):
        super().__init__()

        hidden_size = hidden_size or input_size
        output_size = output_size or input_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.residual_scale = residual_scale

        # First projection
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)

        # Optional context projection; no bias to keep it purely additive
        if context_size is not None:
            self.ctx = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.ctx = None

        # Second projection → 2*H for GLU split
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.glu = nn.GLU(dim=-1)  # GLU([a, b]) = a * σ(b)

        # Output projection to match desired output_size
        self.out = nn.Linear(hidden_size, output_size, bias=bias)

        # Residual path projection if needed
        self.skip = nn.Linear(input_size, output_size, bias=False) if input_size != output_size else nn.Identity()

        # Regularization & norms
        self.drop = nn.Dropout(dropout)
        self.act = _make_activation(activation)
        self.norm = _make_norm(norm, output_size)

        # Init: kaiming for hidden, xavier for output tends to work well
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        if self.fc1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc1.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        if self.fc2.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc2.bias, -bound, bound)

        nn.init.xavier_uniform_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

        if isinstance(self.skip, nn.Linear):
            nn.init.xavier_uniform_(self.skip.weight)

        if self.ctx is not None:
            nn.init.xavier_uniform_(self.ctx.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [..., input_size]
        context (optional): broadcastable to x, last dim == context_size
            Accepts shapes:
              - [..., context_size]      (same prefix as x)
              - [*, context_size]        (will broadcast across time/batch dims)
              - [context_size]           (single global context)
        returns: [..., output_size]
        """
        residual = x

        h = self.fc1(x)  # [..., H]

        if self.ctx is not None and context is not None:
            # Make context broadcastable to h's prefix dims
            # Expect last dim == context_size
            if context.dim() == 1:  # [C]
                ctx_proj = self.ctx(context).view(*([1] * (h.dim() - 1)), -1)
            else:
                # Align trailing dim; rely on PyTorch broadcasting for the rest
                ctx_proj = self.ctx(context)
                # If needed, unsqueeze leading dims to match
                while ctx_proj.dim() < h.dim():
                    ctx_proj = ctx_proj.unsqueeze(0)
            h = h + ctx_proj  # broadcast add

        h = self.act(h)
        h = self.glu(self.fc2(h))   # [..., H]
        h = self.drop(h)
        h = self.out(h)             # [..., O]

        # Residual + norm
        res = self.skip(residual) if not isinstance(self.skip, nn.Identity) else residual
        y = h + self.residual_scale * res
        y = self.norm(y)
        return y
