"""
Lightweight state-space (S4D-lite) search-space operation.

Fills the ``"ssm"`` operation family that the candidate sampler already
special-cases as a placeholder (see ``search/candidate_config.py``) but that
had no member operation. Rather than a full selective-scan Mamba block (a
heavier dependency and a different complexity class than the rest of the
search space), this implements a real diagonal state-space model in the S4D
style: a per-channel, per-state exponential-decay recurrence, unrolled as a
truncated causal depthwise convolution, with Mamba-style input/output gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.norms import RMSNorm


class SSMOp(nn.Module):
    """Diagonal state-space (S4D-style) block with gated input/output.

    For each output channel ``d`` and state ``s``, the block realises the
    linear recurrence ``h[s,t] = decay[d,s] * h[s,t-1] + B[d,s] * u[t]`` and
    reads out ``y[d,t] = C[d,s] . h[s,t] + D[d] * u[t]``. Because the
    recurrence is diagonal and time-invariant, it has a closed-form causal
    convolution kernel of length ``kernel_len`` (truncated impulse response),
    so the whole block is a single depthwise causal convolution rather than a
    sequential scan.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        state_dim: int = 8,
        kernel_len: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.state_dim = max(1, state_dim)
        self.kernel_len = max(4, min(kernel_len or seq_length, seq_length, 64))

        # Gated input projection (Mamba-style): split into the SSM input and
        # a parallel gate applied to the SSM output before the output proj.
        self.in_proj = nn.Linear(input_dim, latent_dim * 2, bias=False)

        # Diagonal state-space parameters. ``log_decay`` is passed through a
        # sigmoid so the per-state decay stays in (0, 1) — a stable, purely
        # real S4D diagonal (no complex eigenvalues, matching the rest of
        # this codebase's real-valued ops).
        self.log_decay = nn.Parameter(
            torch.randn(latent_dim, self.state_dim) * 0.5 - 2.0
        )
        self.B = nn.Parameter(torch.randn(latent_dim, self.state_dim) * 0.5)
        self.C = nn.Parameter(torch.randn(latent_dim, self.state_dim) * 0.5)
        self.D = nn.Parameter(torch.ones(latent_dim))

        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.norm = RMSNorm(latent_dim)
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def _causal_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the ``[latent_dim, kernel_len]`` truncated impulse response."""
        decay = torch.sigmoid(self.log_decay).to(device=device, dtype=dtype)
        powers = torch.arange(self.kernel_len, device=device, dtype=dtype)
        # decay_pow[d, s, k] = decay[d, s] ** k
        decay_pow = decay.unsqueeze(-1) ** powers.view(1, 1, -1)
        coeff = self.B * self.C  # [latent_dim, state_dim]
        kernel = torch.einsum("ds,dsk->dk", coeff, decay_pow)  # [latent_dim, kernel_len]
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        proj = self.in_proj(x)
        u, gate = proj.chunk(2, dim=-1)  # each [B, L, latent_dim]

        kernel = self._causal_kernel(u.device, u.dtype)  # [latent_dim, K]
        # F.conv1d cross-correlates, so flip the kernel to realise
        # y[t] = sum_k kernel[k] * u[t - k] under left-padding by K - 1.
        weight = kernel.flip(-1).unsqueeze(1)  # [latent_dim, 1, K]

        u_t = u.transpose(1, 2)  # [B, latent_dim, L]
        u_t = F.pad(u_t, (self.kernel_len - 1, 0))
        ssm_out = F.conv1d(u_t, weight, groups=self.latent_dim)  # [B, latent_dim, L]
        ssm_out = ssm_out.transpose(1, 2) + self.D * u  # skip connection

        gated = F.silu(gate) * ssm_out
        out = self.out_proj(gated)
        out = self.dropout(out)
        return self.norm(out + residual)
