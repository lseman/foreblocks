from __future__ import annotations

import torch
import torch.nn as nn

from .attention import SlidingWindowAttention
from .ssd import StructuredStateSpaceDualityBranch


class HybridMamba2Block(nn.Module):
    """Hybrid block that fuses an SSD branch with sliding-window attention."""

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        num_heads: int = 8,
        n_kv_heads: int | None = None,
        window_size: int = 128,
        attn_dropout: float = 0.0,
        use_gated_delta: bool = False,
        use_cuda_scan: bool = True,
        rope_base: int = 10_000,
        max_seq_len: int = 8192,
        n_sink_tokens: int = 0,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        attn_logit_softcap: float | None = None,
        layer_scale_init: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        del use_cuda_scan

        self.ssm_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.mix_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)

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
            n_kv_heads=n_kv_heads,
            window_size=window_size,
            dropout=attn_dropout,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
            n_sink_tokens=n_sink_tokens,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            logit_softcap=attn_logit_softcap,
        )
        self.mix_gate = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_scale = (
            nn.Parameter(torch.full((d_model,), layer_scale_init))
            if layer_scale_init is not None
            else None
        )

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
        out = self.out_proj(self.out_norm(mixed))
        if self.layer_scale is not None:
            out = out * self.layer_scale
        return out

    def make_state(self, batch: int, device=None, dtype=None) -> dict:
        """Return a fresh recurrent state for *batch* sequences."""
        state = self.ssm.make_state(batch, device=device, dtype=dtype)
        nkv = self.attn.n_kv_heads
        ws = self.attn.window_size
        hd = self.attn.head_dim
        n_sink = self.attn.n_sink_tokens
        state["kv_k"] = torch.zeros(batch, nkv, ws, hd, device=device, dtype=dtype)
        state["kv_v"] = torch.zeros(batch, nkv, ws, hd, device=device, dtype=dtype)
        state["kv_abs"] = torch.full((ws,), -1, device=device, dtype=torch.long)
        if n_sink > 0:
            state["sink_k"] = torch.zeros(
                batch, nkv, n_sink, hd, device=device, dtype=dtype
            )
            state["sink_v"] = torch.zeros(
                batch, nkv, n_sink, hd, device=device, dtype=dtype
            )
        state["kv_pos"] = 0
        state["kv_len"] = 0
        state["abs_pos"] = 0
        return state

    def _attn_step(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        B = x.shape[0]
        attn = self.attn

        q = attn.q_proj(x).reshape(B, attn.num_heads, attn.head_dim)
        k = attn.k_proj(x).reshape(B, attn.n_kv_heads, attn.head_dim)
        v = attn.v_proj(x).reshape(B, attn.n_kv_heads, attn.head_dim)

        q = attn.q_norm(q)
        k = attn.k_norm(k)
        abs_pos = state["abs_pos"]
        q, k = attn.rope.apply_at_pos(q, k, abs_pos)
        state["abs_pos"] = abs_pos + 1

        if attn.n_sink_tokens > 0 and abs_pos < attn.n_sink_tokens:
            state["sink_k"][:, :, abs_pos] = k.detach()
            state["sink_v"][:, :, abs_pos] = v.detach()

        buf = state["kv_pos"]
        state["kv_k"][:, :, buf] = k.detach()
        state["kv_v"][:, :, buf] = v.detach()
        state["kv_abs"][buf] = abs_pos
        state["kv_pos"] = (buf + 1) % attn.window_size
        state["kv_len"] = min(state["kv_len"] + 1, attn.window_size)

        valid = state["kv_len"]
        recent_k = (
            state["kv_k"] if valid == attn.window_size else state["kv_k"][:, :, :valid]
        )
        recent_v = (
            state["kv_v"] if valid == attn.window_size else state["kv_v"][:, :, :valid]
        )
        recent_abs = (
            state["kv_abs"] if valid == attn.window_size else state["kv_abs"][:valid]
        )

        if attn.n_sink_tokens > 0:
            local_start = max(0, abs_pos - attn.window_size + 1)
            recent_mask = (recent_abs >= local_start) & (
                recent_abs >= attn.n_sink_tokens
            )
            sink_len = min(abs_pos + 1, attn.n_sink_tokens)
            sink_k = state["sink_k"][:, :, :sink_len]
            sink_v = state["sink_v"][:, :, :sink_len]
            k_ctx = torch.cat([sink_k, recent_k[:, :, recent_mask]], dim=2)
            v_ctx = torch.cat([sink_v, recent_v[:, :, recent_mask]], dim=2)
        else:
            k_ctx = recent_k
            v_ctx = recent_v

        if attn.n_rep > 1:
            k_ctx = k_ctx.repeat_interleave(attn.n_rep, dim=1)
            v_ctx = v_ctx.repeat_interleave(attn.n_rep, dim=1)

        y = attn._attention(q.unsqueeze(2), k_ctx, v_ctx)
        y = y.squeeze(2).reshape(B, attn.d_model)
        return attn.out_proj(y)

    def step(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        """Single-token recurrent forward."""
        ssm_out = self.ssm.step(self.ssm_norm(x), state)
        attn_out = self._attn_step(self.attn_norm(x), state)
        gate = torch.sigmoid(self.mix_gate(self.mix_norm(x)))
        mixed = gate * ssm_out + (1.0 - gate) * attn_out
        out = self.out_proj(self.out_norm(mixed))
        if self.layer_scale is not None:
            out = out * self.layer_scale
        return out
