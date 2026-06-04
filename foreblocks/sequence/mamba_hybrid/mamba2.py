"""Mamba2-style SSM block with **diagonal A** and optional **chunked scan**."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ops.mamba import chunked_ssd_forward, dt_prep, fused_out, mamba2_split_conv1d_scan_combined
from foreblocks.sequence.mamba_hybrid.conv import CausalDepthwiseConv1d
from foreblocks.sequence.mamba_hybrid.norms import RMSNormWeightOnly
from foreblocks.sequence.mamba_hybrid.utils import auto_dt_rank, conv_step, fused_out_2d, inverse_softplus


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
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        head_dim: int = 64,
        num_heads: int | None = None,
        n_groups: int = 1,
        chunk_size: int | None = 256,
        dt_min: float = 1e-4,
        dt_max: float = 1.0,
        dt_init_min: float | None = None,
        dt_init_max: float | None = None,
        dt_init_floor: float = 1e-4,
        dt_limit: tuple[float, float] | None = None,
        A_init_range: tuple[float, float] = (1, 16),
        conv_init: float | None = None,
        use_conv_bias: bool = True,
        use_bias: bool = False,
        norm_eps: float = 1e-6,
        use_fused_path: bool = True,
        use_triton_ssd: bool = True,
        use_pre_norm: bool = True,
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
        self.use_fused_path = use_fused_path
        self.use_triton_ssd = use_triton_ssd

        if self.num_heads % n_groups != 0:
            raise ValueError("num_heads must be divisible by n_groups")

        # ── in projection ────────────────────────────────────────────
        # Output layout: [z, u, B, C, dt]
        # z: d_inner
        # u + B + C (after conv): d_inner + 2 * n_groups * d_state
        # dt: low-rank dt hidden state
        self.conv_dim = self.d_inner + 2 * n_groups * d_state
        dt_rank = dt_rank or auto_dt_rank(d_model)
        total_out = self.d_inner + self.conv_dim + dt_rank
        self.in_proj = nn.Linear(d_model, total_out, bias=use_bias)

        # ── dt projection ────────────────────────────────────────────
        self.dt_rank = dt_rank
        self.dt_proj = nn.Linear(dt_rank, self.num_heads, bias=False)

        # ── dt bias (inverse softplus init) ──────────────────────────
        dt_init = torch.exp(
            torch.rand(self.num_heads)
            * (math.log(self.dt_init_max) - math.log(self.dt_init_min))
            + math.log(self.dt_init_min)
        )
        dt_init = torch.clamp(dt_init, min=dt_init_floor)
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ── A_log (per-head scalar, diagonal A) ──────────────────────
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

        self.reset_parameters()

    # ── parameter initialisation ──────────────────────────────────────

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x

        p = self.in_proj(self.pre_norm(x))
        residual_inner = self.residual_proj(residual)
        if self.use_fused_path:
            weight = self.conv.conv.weight.view(self.conv_dim, self.d_conv).contiguous()
            bias = self.conv.conv.bias.contiguous() if self.conv.conv.bias is not None else None
            return mamba2_split_conv1d_scan_combined(
                p,
                residual_inner,
                conv_weight=weight,
                conv_bias=bias,
                dt_proj_weight=self.dt_proj.weight,
                dt_bias=self.dt_bias,
                A_log=self.A_log,
                Dskip=self.Dskip,
                norm_weight=self.norm.weight,
                out_proj_weight=self.out_proj.weight,
                out_proj_bias=self.out_proj.bias,
                d_inner=self.d_inner,
                conv_dim=self.conv_dim,
                dt_rank=self.dt_rank,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                n_groups=self.n_groups,
                d_state=self.d_state,
                chunk_size=self.chunk_size or x.shape[1],
                dt_limit=self.dt_limit,
                norm_eps=self.norm_eps,
                attention_mask=attention_mask,
                # Triton SSD forward is safe in training: its autograd backward
                # (vectorised chunked, in ssd.py) recomputes from saved inputs and
                # is independent of which forward ran.
                use_triton_ssd=self.use_triton_ssd,
            )

        z, conv_input, dt_hidden = torch.split(
            p, [self.d_inner, self.conv_dim, self.dt_rank], dim=-1
        )
        if attention_mask is not None:
            conv_input = self._apply_attention_mask(conv_input, attention_mask)
        conv_out = self.conv(conv_input)
        if attention_mask is not None:
            conv_out = self._apply_attention_mask(conv_out, attention_mask)
        u, Braw, Craw = self._split_conv_out(conv_out)

        dt_raw = self.dt_proj(dt_hidden)
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

        # ── norm + gate + out ────────────────────────────────────────
        y = fused_out(y, z, residual_inner, self.norm.weight, eps=self.norm_eps)
        return self.out_proj(y)

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
                batch, self.conv_dim, self.d_conv - 1, device=device, dtype=dtype,
            ),
            "ssm": torch.zeros(
                batch, self.num_heads, self.head_dim, self.d_state,
                device=device, dtype=dtype,
            ),
        }

    def step(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        """Single-token autoregressive step (no chunking)."""
        B = x.shape[0]
        p = self.in_proj(self.pre_norm(x))
        z, conv_input, dt_hidden = torch.split(
            p, [self.d_inner, self.conv_dim, self.dt_rank], dim=-1
        )

        weight = self.conv.conv.weight.view(self.conv_dim, self.d_conv)
        conv_out, state["conv"] = conv_step(
            conv_input, state["conv"], weight, self.conv.conv.bias
        )
        u, Bpar, Cpar = self._split_conv_out(conv_out)

        dt_raw = self.dt_proj(dt_hidden)
        dt = F.softplus(dt_raw + self.dt_bias).clamp(*self.dt_limit)
        A = -torch.exp(self.A_log)  # [H]

        h = state["ssm"].to(dtype=torch.float32)
        u_heads = u.reshape(B, self.num_heads, self.head_dim)
        abar = torch.exp(
            dt.unsqueeze(-1).unsqueeze(-1)
            * A.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        h = (
            abar * h
            + dt.unsqueeze(-1).unsqueeze(-1)
            * Bpar.unsqueeze(-2)
            * u_heads.unsqueeze(-1)
        )
        y = (Cpar.unsqueeze(-2) * h).sum(dim=-1) + self.Dskip * u_heads
        y = y.reshape(B, self.d_inner)
        state["ssm"] = h.detach()

        residual_inner = self.residual_proj(x)
        y_normed = fused_out_2d(y, z, residual_inner, self.norm.weight, eps=self.norm_eps)
        return self.out_proj(y_normed)
