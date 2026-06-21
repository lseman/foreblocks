"""Mamba2-style SSM block with **diagonal A** and optional **chunked scan**."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ops.mamba import (
    chunked_ssd_forward,
    dt_prep,
    fused_out,
    mamba2_split_conv1d_scan_combined,
)
from foreblocks.ops.mamba.fused_mamba2 import fused_mamba2_forward, FUSED_MAMBA2_TRITON_AVAILABLE
from foreblocks.sequence.mamba.conv import CausalDepthwiseConv1d
from foreblocks.sequence.mamba.norms import RMSNormWeightOnly
from foreblocks.sequence.mamba.utils import conv_step, fused_out_2d


class Mamba2Block(nn.Module):
    """Mamba2-style block (diagonal A, chunked scan).

    Parameters
    ----------
    d_model : int
        Input / output dimension.
    d_state : int, default 16
        State dimension *per head* (``N`` in the Mamba2 paper).
    d_conv : int, default 4
        Width of the causal depthwise convolution.
    head_dim : int, default 64
        Dimension per attention head.  ``num_heads = d_inner // head_dim``.
    n_groups : int, default 1
        Number of B / C groups.  Heads in the same group share B and C.
    chunk_size : int, default 64
        Chunk size for the SSD scan.  Set to ``None`` to disable chunking.
    dt_min / dt_max : float
        Clamp range for the softplus(dt) output.
    A_init_range : tuple[float, float], default (1, 16)
        Uniform range for initial ``A_log`` values.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 128,
        d_conv: int = 4,
        head_dim: int = 64,
        num_heads: int | None = None,
        n_groups: int = 1,
        chunk_size: int | None = 256,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_min: float | None = None,
        dt_init_max: float | None = None,
        dt_init_floor: float = 1e-4,
        dt_limit: tuple[float, float] | None = None,
        A_init_range: tuple[float, float] = (1, 16),
        conv_init: float | None = None,
        use_conv_bias: bool = True,
        use_bias: bool = False,
        norm_eps: float = 1e-5,
        use_fused_path: bool = True,
        use_triton_ssd: bool = True,
        use_pre_norm: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner or 2 * d_model  # expand=2, same as fla
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_groups = n_groups
        if num_heads is not None:
            if self.d_inner % num_heads != 0:
                raise ValueError("d_inner must be divisible by num_heads")
            self.num_heads = num_heads
            self.head_dim = self.d_inner // num_heads
        else:
            if self.d_inner % head_dim != 0:
                raise ValueError("d_inner must be divisible by head_dim")
            self.head_dim = head_dim
            self.num_heads = self.d_inner // head_dim
        self.chunk_size = chunk_size
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_min = dt_init_min if dt_init_min is not None else dt_min
        self.dt_init_max = dt_init_max if dt_init_max is not None else dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_limit = dt_limit if dt_limit is not None else (dt_min, dt_max)
        self.A_init_range = A_init_range
        self.norm_eps = norm_eps
        self.use_fused_path = use_fused_path  # TODO: fused kernel has no backward + CUDA issues; default False
        self.use_triton_ssd = use_triton_ssd

        if self.num_heads % n_groups != 0:
            raise ValueError("num_heads must be divisible by n_groups")

        # ── in projection ────────────────────────────────────────────
        # Output layout: [z, u/B/C, dt]
        # z: d_inner
        # u + B + C (after conv): d_inner + 2 * n_groups * d_state
        # dt: nheads (direct, no low-rank bottleneck)
        self.conv_dim = self.d_inner + 2 * n_groups * d_state
        total_out = self.d_inner + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(d_model, total_out, bias=use_bias)

        # ── dt bias (inverse softplus init, Mamba2-style) ────────────
        dt_init = torch.exp(
            torch.rand(self.num_heads)
            * (math.log(self.dt_init_max) - math.log(self.dt_init_min))
            + math.log(self.dt_init_min)
        )
        dt_init = torch.clamp(dt_init, min=dt_init_floor)
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ── A_log (per-head scalar, diagonal A, Mamba2-style) ────────
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # ── D skip (per head × head_dim) ─────────────────────────────
        self.Dskip = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.Dskip._no_weight_decay = True

        # ── conv1d on [u, B, C] ──────────────────────────────────────
        self.conv = CausalDepthwiseConv1d(
            self.conv_dim,
            kernel_size=d_conv,
            bias=use_conv_bias,
            conv_init=conv_init,
        )

        # ── RMSNorm + out ────────────────────────────────────────────
        self.norm = RMSNormWeightOnly(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)

        self.residual_proj = (
            nn.Identity()
            if d_model == self.d_inner
            else nn.Linear(d_model, self.d_inner, bias=False)
        )
        self.pre_norm = nn.LayerNorm(d_model) if use_pre_norm else nn.Identity()
        self.activation = activation if activation else "silu"

        self.reset_parameters()

    # ── parameter initialisation ──────────────────────────────────────

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_normed = self.pre_norm(x)
        p = self.in_proj(x_normed)
        residual_inner = self.residual_proj(x_normed) if self.residual_proj is not None else x_normed

        # ── Fused path: single Triton kernel (conv + dt + SSM + out) ──
        # Only for inference — fused kernel has no backward pass
        if self.use_fused_path and not self.training and FUSED_MAMBA2_TRITON_AVAILABLE:
            try:
                return self._forward_fused(x_normed, p, residual_inner)
            except RuntimeError:
                pass  # fall through to split path

        # ── Standard path ──────────────────────────────────────────
        z, conv_input, dt_raw = torch.split(
            p, [self.d_inner, self.conv_dim, self.num_heads], dim=-1
        )
        if attention_mask is not None:
            conv_input = self._apply_attention_mask(conv_input, attention_mask)
        conv_out = self.conv(conv_input)
        # SiLU activation on conv output before splitting — matches official Mamba2
        conv_out = F.silu(conv_out)
        if attention_mask is not None:
            conv_out = self._apply_attention_mask(conv_out, attention_mask)
        u, Braw, Craw = self._split_conv_out(conv_out)

        dt = dt_prep(
            dt_raw,
            self.dt_bias,
            dt_min=self.dt_limit[0],
            dt_max=self.dt_limit[1],
        )
        A = -torch.exp(self.A_log)  # [H] — scalar per head

        # ── SSM scan ─────────────────────────────────────────────────
        chunk_size = self.chunk_size or u.shape[1]
        y = chunked_ssd_forward(
            u=u.reshape(u.shape[0], u.shape[1], self.num_heads, self.head_dim),
            dt=dt,
            A=A,
            B=Braw,
            C=Craw,
            D=self.Dskip,
            chunk_size=chunk_size,
            use_triton=self.use_triton_ssd,
        ).reshape(u.shape[0], u.shape[1], self.d_inner)

        # ── norm + gate + out (RMSNormGated: rms_norm(y)*silu(z)) ──────
        y = fused_out(y, z, self.norm.weight, eps=self.norm_eps)
        return self.out_proj(y)

        z, conv_input, dt_raw = torch.split(
            p, [self.d_inner, self.conv_dim, self.num_heads], dim=-1
        )
        if attention_mask is not None:
            conv_input = self._apply_attention_mask(conv_input, attention_mask)
        conv_out = self.conv(conv_input)
        # SiLU activation on conv output before splitting — matches official Mamba2
        conv_out = F.silu(conv_out)
        if attention_mask is not None:
            conv_out = self._apply_attention_mask(conv_out, attention_mask)
        u, Braw, Craw = self._split_conv_out(conv_out)

        dt = dt_prep(
            dt_raw,
            self.dt_bias,
            dt_min=self.dt_limit[0],
            dt_max=self.dt_limit[1],
        )
        A = -torch.exp(self.A_log)  # [H] — scalar per head

        # ── SSM scan ─────────────────────────────────────────────────
        chunk_size = self.chunk_size or u.shape[1]
        y = chunked_ssd_forward(
            u=u.reshape(u.shape[0], u.shape[1], self.num_heads, self.head_dim),
            dt=dt,
            A=A,
            B=Braw,
            C=Craw,
            D=self.Dskip,
            chunk_size=chunk_size,
            use_triton=self.use_triton_ssd,
        ).reshape(u.shape[0], u.shape[1], self.d_inner)

        # ── norm + gate + out (RMSNormGated: rms_norm(y)*silu(z)) ──────
        y = fused_out(y, z, self.norm.weight, eps=self.norm_eps)
        return self.out_proj(y)

    # ── fused forward path ────────────────────────────────────────

    def _forward_fused(
        self,
        x_normed: torch.Tensor,
        p: torch.Tensor,
        residual_inner: torch.Tensor,
    ) -> torch.Tensor:
        """Single-kernel fused path: conv + dt + SSM + out projection."""
        # Split projected tensor
        z, conv_input, dt_raw = torch.split(
            p, [self.d_inner, self.conv_dim, self.num_heads], dim=-1
        )

        # Conv + SiLU (for extracting u, B, C)
        weight = self.conv.conv.weight.view(self.conv_dim, self.d_conv).contiguous()
        bias = (
            self.conv.conv.bias.contiguous()
            if self.conv.conv.bias is not None
            else None
        )
        conv_out = F.silu(self.conv(conv_input))

        # Extract u, B, C from conv_out
        u, Bflat, Cflat = torch.split(
            conv_out,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )

        # Reshape u to [B, T, H, P]
        B, T, _ = conv_out.shape
        u_heads = u.reshape(B, T, self.num_heads, self.head_dim)

        # Expand B, C by groups → [B, T, H, N]
        Braw = Bflat.reshape(B, T, self.n_groups, self.d_state).repeat_interleave(
            self.num_heads // self.n_groups, dim=2
        )
        Craw = Cflat.reshape(B, T, self.n_groups, self.d_state).repeat_interleave(
            self.num_heads // self.n_groups, dim=2
        )

        # dt_hidden: use dt_raw from in_proj (no projection layer)
        dt_hidden = dt_raw.contiguous()

        A = -torch.exp(self.A_log)  # [H]
        chunk_size = self.chunk_size or T

        # Transpose conv_weight to [K, conv_dim] for kernel efficiency
        conv_weight_t = weight.t().contiguous()

        return fused_mamba2_forward(
            residual_inner=residual_inner,
            conv_input=conv_input,
            conv_weight=conv_weight_t,
            conv_bias=bias,
            dt_proj_weight=None,
            dt_bias=self.dt_bias,
            A=A,
            D=self.Dskip,
            norm_weight=self.norm.weight,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            u=u_heads,
            z=z,
            dt_hidden=dt_hidden,
            B=Braw,
            C=Craw,
            chunk_size=chunk_size,
            dt_min=self.dt_limit[0],
            dt_max=self.dt_limit[1],
            norm_eps=self.norm_eps,
            num_heads=self.num_heads,
        )

    # ── helpers ───────────────────────────────────────────────────────

    def _split_conv_out(self, conv_out: torch.Tensor):
        """Unpack convolved [u, B, C] activations."""
        D = self.d_inner
        N = self.d_state
        ng = self.n_groups

        u, Bflat, Cflat = torch.split(conv_out, [D, ng * N, ng * N], dim=-1)
        if conv_out.ndim == 3:
            B, T, _ = conv_out.shape
            Braw = Bflat.reshape(B, T, ng, N).repeat_interleave(
                self.num_heads // ng, dim=2
            )
            Craw = Cflat.reshape(B, T, ng, N).repeat_interleave(
                self.num_heads // ng, dim=2
            )
        elif conv_out.ndim == 2:
            B = conv_out.shape[0]
            Braw = Bflat.reshape(B, ng, N).repeat_interleave(
                self.num_heads // ng, dim=1
            )
            Craw = Cflat.reshape(B, ng, N).repeat_interleave(
                self.num_heads // ng, dim=1
            )
        else:
            raise ValueError("conv output must have shape [B, D] or [B, T, D]")
        return u, Braw, Craw

    @staticmethod
    def _apply_attention_mask(
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B, T]")
        if attention_mask.shape != x.shape[:2]:
            raise ValueError("attention_mask shape must match [B, T]")
        return x * attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)

    # ── state management (for autoregressive decoding) ────────────────

    def make_state(self, batch: int, device=None, dtype=None) -> dict:
        return {
            "conv": torch.zeros(
                batch,
                self.conv_dim,
                self.d_conv - 1,
                device=device,
                dtype=dtype,
            ),
            "ssm": torch.zeros(
                batch,
                self.num_heads,
                self.head_dim,
                self.d_state,
                device=device,
                dtype=dtype,
            ),
        }

    def step(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        """Single-token autoregressive step (no chunking)."""
        B = x.shape[0]
        x_normed = self.pre_norm(x)
        p = self.in_proj(x_normed)
        z, conv_input, dt_raw = torch.split(
            p, [self.d_inner, self.conv_dim, self.num_heads], dim=-1
        )

        weight = self.conv.conv.weight.view(self.conv_dim, self.d_conv)
        conv_out, state["conv"] = conv_step(
            conv_input.squeeze(1), state["conv"], weight, self.conv.conv.bias
        )
        # SiLU activation on conv output — matches official Mamba2
        conv_out = F.silu(conv_out)
        u, Bpar, Cpar = self._split_conv_out(conv_out)

        dt = F.softplus(dt_raw + self.dt_bias).clamp(*self.dt_limit)
        A = -torch.exp(self.A_log)  # [H]

        h = state["ssm"].to(dtype=torch.float32)
        u_heads = u.reshape(B, self.num_heads, self.head_dim)
        abar = torch.exp(
            dt.squeeze(1).unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        h = abar * h + dt.squeeze(1).unsqueeze(-1).unsqueeze(-1) * Bpar.unsqueeze(
            -2
        ) * u_heads.unsqueeze(-1)
        y = (Cpar.unsqueeze(-2) * h).sum(dim=-1) + self.Dskip * u_heads
        y = y.reshape(B, self.d_inner)
        state["ssm"] = h.detach()

        y_normed = fused_out_2d(
            y, z.squeeze(1), self.norm.weight, eps=self.norm_eps
        )
        return self.out_proj(y_normed)
