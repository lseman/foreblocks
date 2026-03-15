from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import (
    causal_depthwise_conv1d,
    dt_prep,
    fused_out,
    grouped_ssd_scan,
    grouped_ssd_scan_reference,
    selective_scan,
)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, d_inner: int, kernel_size: int):
        super().__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=kernel_size,
            groups=d_inner,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        weight = self.conv.weight.view(self.d_inner, self.kernel_size).contiguous()
        bias = self.conv.bias.contiguous() if self.conv.bias is not None else None
        x = causal_depthwise_conv1d(x, weight, bias)
        return x.transpose(1, 2)


class RMSNormWeightOnly(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        raise RuntimeError("Use fused_out(...) instead.")


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        window_size: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _sliding_mask(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rows = torch.arange(seqlen, device=device)
        cols = torch.arange(seqlen, device=device)
        rel = rows[:, None] - cols[None, :]
        allowed = (rel >= 0) & (rel < self.window_size)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
        mask.masked_fill_(allowed, 0.0)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seqlen, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = self._sliding_mask(seqlen, x.device, q.dtype)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(batch_size, seqlen, self.d_model)
        return self.out_proj(y)


def _auto_dt_rank(d_model: int) -> int:
    return max(4, math.ceil(d_model / 16))


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class HybridMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        dt_min: float = 1e-4,
        dt_max: float = 1.0,
        use_cuda_scan: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner or 2 * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or _auto_dt_rank(d_model)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_cuda_scan = use_cuda_scan

        total_out = 2 * self.d_inner + self.dt_rank + 2 * self.d_inner * self.d_state
        self.in_proj = nn.Linear(d_model, total_out, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner, kernel_size=d_conv)

        self.A_log = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        self.Dskip = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))

        self.norm = RMSNormWeightOnly(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.residual_proj = (
            nn.Identity()
            if self.d_model == self.d_inner
            else nn.Linear(self.d_model, self.d_inner, bias=False)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
        with torch.no_grad():
            base = torch.arange(1, self.d_state + 1, device=self.A_log.device, dtype=self.A_log.dtype)
            self.A_log.copy_(base.log().unsqueeze(0).expand(self.d_inner, -1))
            self.Dskip.fill_(1.0)
            dt = torch.rand(self.d_inner, device=self.dt_bias.device, dtype=self.dt_bias.dtype)
            dt = dt * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
            self.dt_bias.copy_(_inverse_softplus(dt.exp()))

    def _split_proj(self, p: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size, seqlen, _ = p.shape
        D = self.d_inner
        N = self.d_state

        z, u, dt_hidden, Bflat, Cflat = torch.split(
            p,
            [D, D, self.dt_rank, D * N, D * N],
            dim=-1,
        )
        Bpar = Bflat.reshape(batch_size, seqlen, D, N)
        Cpar = Cflat.reshape(batch_size, seqlen, D, N)
        return z, u, dt_hidden, Bpar, Cpar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        p = self.in_proj(x)
        z, u, dt_hidden, Bpar, Cpar = self._split_proj(p)
        u = self.conv(u)

        dt_raw = self.dt_proj(dt_hidden)
        dt = dt_prep(dt_raw, self.dt_bias, dt_min=self.dt_min, dt_max=self.dt_max)
        A = -torch.exp(self.A_log)

        y = selective_scan(
            u=u,
            dt=dt,
            A=A,
            Bpar=Bpar,
            Cpar=Cpar,
            Dskip=self.Dskip,
            use_cuda_kernel=self.use_cuda_scan,
        )

        residual_inner = self.residual_proj(residual)
        y = fused_out(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y)


class StructuredStateSpaceDualityBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        num_heads: int = 8,
        dt_min: float = 1e-4,
        dt_max: float = 1.0,
        use_gated_delta: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner or 2 * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or _auto_dt_rank(d_model)
        self.num_heads = num_heads
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_gated_delta = use_gated_delta
        if self.d_inner % num_heads != 0:
            raise ValueError("d_inner must be divisible by num_heads")

        self.head_dim = self.d_inner // self.num_heads
        extra = self.num_heads if use_gated_delta else 0
        total_out = 2 * self.d_inner + self.dt_rank + 2 * self.num_heads * self.d_state + extra
        self.in_proj = nn.Linear(d_model, total_out, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.num_heads, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner, kernel_size=d_conv)

        self.A_log = nn.Parameter(torch.empty(self.num_heads, self.d_state))
        self.Dskip = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))

        self.norm = RMSNormWeightOnly(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.residual_proj = (
            nn.Identity()
            if self.d_model == self.d_inner
            else nn.Linear(self.d_model, self.d_inner, bias=False)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
        with torch.no_grad():
            base = torch.arange(1, self.d_state + 1, device=self.A_log.device, dtype=self.A_log.dtype)
            self.A_log.copy_(base.log().unsqueeze(0).expand(self.num_heads, -1))
            self.Dskip.fill_(1.0)
            dt = torch.rand(self.num_heads, device=self.dt_bias.device, dtype=self.dt_bias.dtype)
            dt = dt * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
            self.dt_bias.copy_(_inverse_softplus(dt.exp()))

    def _split_proj(self, p: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size, seqlen, _ = p.shape
        h = self.num_heads
        n = self.d_state

        split_sizes = [self.d_inner, self.d_inner, self.dt_rank, h * n, h * n]
        if self.use_gated_delta:
            split_sizes.append(h)
        parts = torch.split(p, split_sizes, dim=-1)

        z, u, dt_hidden, Bflat, Cflat = parts[:5]
        delta_gate = parts[5] if self.use_gated_delta else None
        Bpar = Bflat.reshape(batch_size, seqlen, h, n)
        Cpar = Cflat.reshape(batch_size, seqlen, h, n)
        return z, u, dt_hidden, Bpar, Cpar, delta_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, seqlen, _ = x.shape

        p = self.in_proj(x)
        z, u, dt_hidden, Bpar, Cpar, delta_gate = self._split_proj(p)
        u = self.conv(u)

        dt_raw = self.dt_proj(dt_hidden)
        dt = dt_prep(dt_raw, self.dt_bias, dt_min=self.dt_min, dt_max=self.dt_max)
        A = -torch.exp(self.A_log)

        u_heads = u.reshape(batch_size, seqlen, self.num_heads, self.head_dim)
        y = grouped_ssd_scan(
            u=u_heads,
            dt=dt,
            A=A,
            Bpar=Bpar,
            Cpar=Cpar,
            Dskip=self.Dskip,
            delta_gate=delta_gate,
        ).reshape(batch_size, seqlen, self.d_inner)

        residual_inner = self.residual_proj(residual)
        y = fused_out(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y)


class HybridMamba2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        num_heads: int = 8,
        window_size: int = 128,
        attn_dropout: float = 0.0,
        use_gated_delta: bool = False,
        use_cuda_scan: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.ssm_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.mix_norm = nn.LayerNorm(d_model)
        del use_cuda_scan
        self.ssm = StructuredStateSpaceDualityBranch(
            d_model=d_model,
            d_inner=d_inner,
            d_state=d_state,
            d_conv=d_conv,
            dt_rank=dt_rank,
            num_heads=num_heads,
            use_gated_delta=use_gated_delta,
        )
        self.attn = SlidingWindowAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            dropout=attn_dropout,
        )
        self.mix_gate = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.mix_gate.weight)
        nn.init.zeros_(self.mix_gate.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ssm_out = self.ssm(self.ssm_norm(x))
        attn_out = self.attn(self.attn_norm(x))
        gate = torch.sigmoid(self.mix_gate(self.mix_norm(x)))
        mixed = gate * ssm_out + (1.0 - gate) * attn_out
        return self.out_proj(mixed)


class TinyHybridMambaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                HybridMambaBlock(
                    d_model=d_model,
                    d_inner=2 * d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                    use_cuda_scan=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.final_norm(x)
        return self.lm_head(x)


class TinyHybridMamba2LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        num_heads: int = 8,
        window_size: int = 128,
        attn_every_n: int = 2,
        tie_embeddings: bool = True,
        use_gated_delta: bool = False,
        use_cuda_scan: bool = True,
    ):
        super().__init__()
        if attn_every_n < 1:
            raise ValueError("attn_every_n must be >= 1")

        self.embed = nn.Embedding(vocab_size, d_model)
        blocks = []
        for idx in range(n_layers):
            if idx % attn_every_n == 0:
                block = HybridMamba2Block(
                    d_model=d_model,
                    d_inner=2 * d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                    num_heads=num_heads,
                    window_size=window_size,
                    use_gated_delta=use_gated_delta,
                    use_cuda_scan=use_cuda_scan,
                )
            else:
                block = HybridMambaBlock(
                    d_model=d_model,
                    d_inner=2 * d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                    use_cuda_scan=use_cuda_scan,
                )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.final_norm(x)
        return self.lm_head(x)
