# transformer_moe_dmoe.py
# -----------------------------------------------------------------------------
# Enhanced dMoE FeedForward (fast path)
#  - Router: softmax over Top-K only
#  - Dispatcher: vectorized capacity prune
#  - bf16 autocast for experts + router
#  - Optional pre-stacked expert weights reuse per step
#  - Safe aux handling | optional MoELogger integration
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional import: adjust to your package path
try:
    from .moe_logging import MoELogger  # type: ignore
except Exception:
    MoELogger = None  # type: ignore

# Optional grouped kernel path (can accept prepacked B* if your wrapper supports)
try:
    # Prefer a wrapper that accepts B12_cat_prepacked / B3_cat_prepacked
    from .kernels import grouped_mlp_swiglu  # type: ignore
except Exception:
    grouped_mlp_swiglu = None  # fallback uses Python loop


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def maybe_compile(mod):
    """Wrap with torch.compile if available, else return the module."""
    try:
        return torch.compile(mod, dynamic=True)
    except Exception:
        return mod


def _autocast_bf16_enabled() -> bool:
    return torch.cuda.is_available()


# -----------------------------------------------------------------------------
# Optimized Top-K routing (softmax over Top-K only)
# -----------------------------------------------------------------------------
@torch.jit.script
def optimized_topk_routing(logits: torch.Tensor, k: int):
    """
    Returns (top_p, top_i) for K experts per token.
    - Take topk on raw logits (no full softmax over E).
    - Normalize only within the chosen K.
    """
    if k == 1:
        top_v, top_i = torch.max(logits, dim=-1, keepdim=True)
        top_p = torch.ones_like(top_v)
        return top_p, top_i
    else:
        top_v, top_i = torch.topk(logits, k, dim=-1, sorted=False)
        m = top_v.max(dim=-1, keepdim=True).values
        expv = torch.exp(top_v - m)
        top_p = expv / (expv.sum(dim=-1, keepdim=True) + 1e-12)
        return top_p, top_i


# -----------------------------------------------------------------------------
# Vectorized dropless dispatcher (with capacity pruning)
# -----------------------------------------------------------------------------
class DroplessPackedDispatcher:
    """
    Packs tokens by expert, keeps at most per_expert_cap items per expert.
    - Fully vectorized capacity pruning (no Python per-expert loop).
    - Returns packed_x / packed_w and (experts_seq, tokens_seq, offsets).
    """

    def __init__(self, num_experts: int, top_k: int, capacity_factor: float = 1.25):
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)

        # Reusable buffers (grown as needed)
        self._buffer_size = 0
        self._sort_buffer: Optional[torch.Tensor] = None
        self._offsets_buffer: Optional[torch.Tensor] = None

    def _ensure_buffers(self, size: int, device: torch.device):
        if self._sort_buffer is None or self._buffer_size < size:
            self._buffer_size = size
            self._sort_buffer = torch.empty(size, dtype=torch.long, device=device)
            self._offsets_buffer = torch.empty(self.num_experts + 1, dtype=torch.long, device=device)

    @torch.no_grad()
    def pack(
        self,
        x_flat: torch.Tensor,      # [T, D]
        topk_p: torch.Tensor,      # [T, K]
        topk_i: torch.Tensor,      # [T, K]
        capacity_factor: Optional[float] = None,
    ):
        device = x_flat.device
        T, D = x_flat.shape
        K = topk_i.shape[1]
        S = T * K
        if S == 0:
            empty = x_flat.new_zeros((0, D))
            empty_long = x_flat.new_zeros((0,), dtype=torch.long)
            empty_offsets = x_flat.new_zeros((self.num_experts + 1,), dtype=torch.long)
            return empty, empty, empty_long, empty_long, empty_offsets, 0

        self._ensure_buffers(S, device)

        experts = topk_i.reshape(-1)           # [S]
        weights = topk_p.reshape(-1)           # [S]
        tokens = torch.arange(T, device=device, dtype=torch.long).repeat_interleave(K)  # [S]

        # Sort by expert (stable)
        torch.argsort(experts, stable=True, out=self._sort_buffer[:S])
        sort_idx = self._sort_buffer[:S]
        experts_sorted = experts[sort_idx]     # [S]
        weights_sorted = weights[sort_idx]     # [S]
        tokens_sorted = tokens[sort_idx]       # [S]

        # Offsets by bincount
        counts = torch.bincount(experts_sorted, minlength=self.num_experts)  # [E]
        self._offsets_buffer.zero_()
        torch.cumsum(counts, 0, out=self._offsets_buffer[1:])
        offsets = self._offsets_buffer.clone()  # [E+1]

        # Pack inputs and weights
        packed_x = x_flat.index_select(0, tokens_sorted)      # [S, D]
        packed_w = weights_sorted.unsqueeze(1)                 # [S, 1]

        # Vectorized capacity prune
        total_capacity = math.ceil(T * K * (capacity_factor or self.capacity_factor))
        per_expert_cap = max(1, math.ceil(total_capacity / self.num_experts))

        # rank within expert block: rank = arange(S) - offsets[experts_sorted]
        idx_in_expert = torch.arange(S, device=device) - offsets[experts_sorted]
        kept_mask = idx_in_expert < per_expert_cap
        kept = kept_mask.nonzero(as_tuple=True)[0]
        dropped = int(S - kept.numel())

        if dropped > 0:
            experts_sorted = experts_sorted[kept]
            tokens_sorted = tokens_sorted[kept]
            packed_x = packed_x[kept]
            packed_w = packed_w[kept]
            # recompute offsets after pruning
            counts = torch.bincount(experts_sorted, minlength=self.num_experts)
            offsets = torch.cumsum(
                torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts]), dim=0
            )

        return packed_x, packed_w, experts_sorted, tokens_sorted, offsets, dropped

    @staticmethod
    def scatter_back(out_accum: torch.Tensor, packed_y: torch.Tensor, tokens_seq: torch.Tensor):
        out_accum.index_add_(0, tokens_seq, packed_y)
        return out_accum


# -----------------------------------------------------------------------------
# Router (noisy top-k variant)
# -----------------------------------------------------------------------------
class NoisyTopKRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        clamp_range: Tuple[float, float] = (-1e4, 1e4),
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None
        self.jitter = float(jitter)
        self.clamp_min, self.clamp_max = clamp_range
        self._init_router()

        # Exposed metrics (detached copies for hooks/logging)
        self.last_gate_logits: Optional[torch.Tensor] = None
        self.last_topk_idx: Optional[torch.Tensor] = None
        self.last_tokens_dropped: int = 0
        self.last_aux_loss: float = 0.0
        self.last_latency_ms: float = 0.0
        self.last_meta: Optional[Dict[str, Any]] = None

    def _init_router(self):
        with torch.no_grad():
            std = 0.02
            bound = std * math.sqrt(3)
            self.router.weight.uniform_(-bound, bound)
            if self.router.bias is not None:
                self.router.bias.zero_()

    def forward(self, x: torch.Tensor, return_raw_logits: bool = False):
        if x.numel() == 0:
            shp = (*x.shape[:-1], self.num_experts)
            z = x.new_zeros(shp)
            return (z, z, z) if return_raw_logits else (z, z)

        if self.training and self.input_dropout is not None:
            x = self.input_dropout(x)

        raw = self.router(x)
        if self.training and self.jitter > 0:
            logits = (raw + torch.randn_like(raw) * self.jitter).clamp_(self.clamp_min, self.clamp_max)
        else:
            logits = raw.clamp_(self.clamp_min, self.clamp_max)

        # For speed, don’t compute full softmax here; downstream uses Top-K only.
        if return_raw_logits:
            # Dummy placeholders for compatibility with previous code paths
            return raw, logits, logits  # (raw, logits, "probs_like")
        else:
            return logits, logits


# -----------------------------------------------------------------------------
# Experts
# -----------------------------------------------------------------------------
class MoE_SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, expert_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = float(dropout)
        self.expert_dropout_p = float(expert_dropout)
        self._needs_dropout = dropout > 0
        self._needs_edrop = expert_dropout > 0
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.w12(x)
        g, u = gu.split(self.d_ff, dim=-1)
        h = F.silu(g) * u
        if self.training and self._needs_dropout:
            h = F.dropout(h, p=self.dropout_p, training=True)
        if self.training and self._needs_edrop:
            h = F.dropout(h, p=self.expert_dropout_p, training=True)
        return self.w3(h)


class MoE_FFNExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu", expert_dropout: float = 0.0):
        super().__init__()
        self.dropout_p = float(dropout)
        self.expert_dropout_p = float(expert_dropout)
        self._needs_dropout = dropout > 0
        self._needs_edrop = expert_dropout > 0
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = getattr(F, activation.lower(), F.gelu)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        if self.training and self._needs_dropout:
            x = F.dropout(x, p=self.dropout_p, training=True)
        if self.training and self._needs_edrop:
            x = F.dropout(x, p=self.expert_dropout_p, training=True)
        return self.fc2(x)


# -----------------------------------------------------------------------------
# dMoE FFN (fast path)
# -----------------------------------------------------------------------------
class MoEFeedForwardDMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        activation: str = "gelu",
        load_balance_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        moe_capacity_factor: float = 1.25,
        router_type: str = "noisy_topk",
        use_bias: bool = False,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        expert_dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        # Optional logging
        moe_logger: Optional["MoELogger"] = None,
        step_getter: Optional[Callable[[], int]] = None,
        log_latency: bool = False,
    ):
        super().__init__()
        print(
            f"MoEFeedForwardDMoE: E={num_experts}, K={top_k}, swiglu={use_swiglu}, "
            f"cap={moe_capacity_factor}, router={router_type}"
        )
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = int(num_experts)
        self.top_k = int(min(top_k, num_experts))
        self.load_balance_weight = float(load_balance_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.moe_capacity_factor = float(moe_capacity_factor)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.moe_logger = moe_logger
        self.step_getter = step_getter
        self.log_latency = bool(log_latency)

        self.input_norm = nn.LayerNorm(d_model)

        # Router
        if router_type == "noisy_topk":
            self.router = NoisyTopKRouter(
                d_model=d_model,
                num_experts=num_experts,
                input_dropout=input_dropout,
                jitter=jitter,
                use_bias=use_bias,
            )
        else:
            self.router = nn.Linear(d_model, num_experts, bias=use_bias)
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        # Experts
        expert_cls = MoE_SwiGLUExpert if use_swiglu else MoE_FFNExpert
        expert_kwargs = dict(d_model=d_model, d_ff=d_ff, dropout=dropout, expert_dropout=expert_dropout)
        if not use_swiglu:
            expert_kwargs["activation"] = activation
        self.experts = nn.ModuleList([expert_cls(**expert_kwargs) for _ in range(num_experts)])

        # Optional compile
        self.router = maybe_compile(self.router)
        self.experts = nn.ModuleList([maybe_compile(e) for e in self.experts])

        # Dispatcher
        self.dispatcher = DroplessPackedDispatcher(num_experts, self.top_k, moe_capacity_factor)

        # Metrics / buffers
        self.register_buffer("_eps", torch.tensor(1e-8))
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("momentum", torch.tensor(0.999))
        self.aux_loss = torch.tensor(0.0)

        # Prepacked expert weights cache (optional fast path)
        self._packed_step = -1
        self._B12_cat = None  # [E, D, 2H] or [E, K, N] depending on wrapper
        self._B3_cat = None   # [E, H, D]

    # ---------- Prepack expert weights (bf16) and reuse per step ----------
    @torch.no_grad()
    def _refresh_packed_weights(self, step: int):
        if step == self._packed_step:
            return
        w12_list, w3_list = [], []
        for e in self.experts:
            if hasattr(e, "w12") and hasattr(e, "w3"):
                w12 = e.w12.weight.t().contiguous()  # [D, 2H]
                w3 = e.w3.weight.t().contiguous()    # [H, D]
            else:
                # Fallback naming (e.g., gate_up_proj/down_proj style)
                w12 = e.gate_up_proj.weight.t().contiguous()  # type: ignore
                w3 = e.down_proj.weight.t().contiguous()      # type: ignore
            w12_list.append(w12)
            w3_list.append(w3)

        # Move to bf16 to match autocast; keep accumulation fp32 in kernels
        self._B12_cat = torch.stack([w.to(dtype=torch.bfloat16) for w in w12_list], dim=0)
        self._B3_cat = torch.stack([w.to(dtype=torch.bfloat16) for w in w3_list], dim=0)
        self._packed_step = step

    def invalidate_packed_weights(self):
        self._packed_step = -1
        self._B12_cat = None
        self._B3_cat = None

    # ---------- Helpers ----------
    def _compute_router_logits(self, x_flat: torch.Tensor) -> torch.Tensor:
        out = None
        try:
            out = self.router(x_flat, return_raw_logits=True)  # preferred
        except TypeError:
            out = self.router(x_flat)
        if isinstance(out, tuple):
            if len(out) >= 2:
                logits = out[1]
            else:
                logits = out[0]
        else:
            logits = out
        return logits

    def _update_expert_usage_ema(self, top_p: torch.Tensor, top_i: torch.Tensor):
        with torch.no_grad():
            cur = torch.zeros(self.num_experts, device=top_p.device)
            cur.scatter_add_(0, top_i.reshape(-1), top_p.reshape(-1))
            cur = cur / max(top_i.size(0), 1)
            decay = float(self.momentum)
            self.expert_usage.mul_(decay).add_(cur, alpha=1.0 - decay)

    def _maybe_start_timer(self):
        if not self.log_latency:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _maybe_stop_timer(self, t0):
        if not self.log_latency or t0 is None:
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    # ---------- Forward ----------
    def forward(
        self,
        x: torch.Tensor,                    # [..., T, D] or [B, T, D]
        return_aux_loss: bool = True,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ):
        # Pre-norm
        x_norm = self.input_norm(x)
        x_flat = x_norm.reshape(-1, self.d_model)

        # bf16 autocast region
        use_amp = _autocast_bf16_enabled()
        t0 = self._maybe_start_timer()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = self._compute_router_logits(x_flat)                 # [T, E]
            Toks = x_flat.size(0)
            top_p, top_i = optimized_topk_routing(logits, self.top_k)    # [T, K], [T, K]

            if self.training:
                self._update_expert_usage_ema(top_p, top_i)

            # Pack (vectorized capacity prune)
            packed_x, packed_w, experts_seq, tokens_seq, offsets, dropped = self.dispatcher.pack(
                x_flat, top_p, top_i, capacity_factor=self.moe_capacity_factor
            )

            if packed_x.numel() == 0:
                out = torch.zeros_like(x_flat).reshape_as(x_norm)
                aux = x.new_zeros(())
                self.aux_loss = aux.detach()
                latency_ms = self._maybe_stop_timer(t0)
                self._populate_router_last(logits, top_i, dropped, float(aux), latency_ms, meta)
                self._maybe_log_router(logits, top_i, dropped, float(aux), latency_ms, meta)
                return (out, aux) if return_aux_loss else out

            # Expert compute
            if grouped_mlp_swiglu is not None:
                # Optional prepacked expert weights (refresh once per step)
                step = int(self.step_getter()) if self.step_getter is not None else -1
                try:
                    self._refresh_packed_weights(step)
                    packed_y = grouped_mlp_swiglu(
                        packed_x,
                        offsets,
                        self.experts,
                        dropout_p=getattr(self.experts[0], "dropout_p", 0.0),
                        training=self.training,
                        out_dtype=packed_x.dtype,
                        B12_cat_prepacked=self._B12_cat,  # may be ignored if wrapper doesn't accept
                        B3_cat_prepacked=self._B3_cat,
                    )
                except TypeError:
                    # Backward compatible wrapper without prepacked args
                    packed_y = grouped_mlp_swiglu(
                        packed_x,
                        offsets,
                        self.experts,
                        dropout_p=getattr(self.experts[0], "dropout_p", 0.0),
                        training=self.training,
                        out_dtype=packed_x.dtype,
                    )
            else:
                # Fallback: loop (optionally checkpoint)
                packed_y = torch.empty_like(packed_x)
                for e in range(self.num_experts):
                    s = int(offsets[e].item())
                    t = int(offsets[e + 1].item())
                    if t <= s:
                        continue
                    seg = packed_x[s:t]
                    if self.use_gradient_checkpointing and self.training:
                        y = torch.utils.checkpoint.checkpoint(self.experts[e], seg, use_reentrant=False)
                    else:
                        y = self.experts[e](seg)
                    packed_y[s:t] = y

            # Weight by gate prob and scatter back
            packed_y.mul_(packed_w)
            out_flat = torch.zeros_like(x_flat)
            self.dispatcher.scatter_back(out_flat, packed_y, tokens_seq)
            out = out_flat.reshape_as(x_norm)

        # Aux (compute possibly outside autocast for numerical stability)
        if return_aux_loss:
            aux = self._compute_aux_loss(logits, Toks, experts_seq)
            self.aux_loss = aux.detach()
        else:
            aux = x.new_zeros(())
            self.aux_loss = aux.detach()

        latency_ms = self._maybe_stop_timer(t0)
        self._populate_router_last(logits, top_i, dropped, float(aux), latency_ms, meta)
        self._maybe_log_router(logits, top_i, dropped, float(aux), latency_ms, meta)

        return (out, aux) if return_aux_loss else out

    # ---------- Aux losses ----------
    def _compute_aux_loss(self, logits: torch.Tensor, num_tokens: int, experts_seq: torch.Tensor):
        aux = logits.new_zeros(())

        # Z-loss
        if self.z_loss_weight > 0:
            log_z = torch.logsumexp(logits.view(-1, self.num_experts), dim=-1)
            aux = aux + self.z_loss_weight * 0.5 * (log_z ** 2).mean()

        # Load-balance loss (uses post-capacity f and average p over E)
        if self.load_balance_weight > 0:
            # Full softmax here only for loss (off hot path)
            probs = F.softmax(logits, dim=-1)
            with torch.no_grad():
                counts = torch.bincount(experts_seq, minlength=self.num_experts).float()
                denom = max(1, (num_tokens * self.top_k))
                f = counts / denom
            p = probs.mean(dim=0)
            aux = aux + self.load_balance_weight * self.num_experts * (f * p).pow(2).sum()

        return aux

    # ---------- Stats ----------
    @torch.no_grad()
    def get_expert_stats(self) -> Dict[str, torch.Tensor]:
        eps = float(self._eps)
        usage = self.expert_usage.clamp_min(eps)
        entropy = -(usage * usage.log()).sum()
        return {
            "expert_usage": self.expert_usage.clone(),
            "usage_entropy": entropy,
            "max_usage": self.expert_usage.max(),
            "min_usage": self.expert_usage.min(),
        }

    def reset_stats(self):
        self.expert_usage.zero_()
        self.aux_loss = torch.zeros((), device=self.expert_usage.device).detach()
        self.invalidate_packed_weights()

    # ---------- Logging helpers ----------
    def _populate_router_last(
        self,
        logits: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens_dropped: int,
        aux_loss: float,
        latency_ms: float,
        meta: Optional[Dict[str, Any]],
    ):
        r = self.router
        try:
            r.last_gate_logits = logits.detach()
            r.last_topk_idx = topk_idx.detach()
            r.last_tokens_dropped = int(tokens_dropped)
            r.last_aux_loss = float(aux_loss)
            r.last_latency_ms = float(latency_ms)
            r.last_meta = meta
        except Exception:
            pass

    def _maybe_log_router(
        self,
        logits: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens_dropped: int,
        aux_loss: float,
        latency_ms: float,
        meta: Optional[Dict[str, Any]],
    ):
        if self.moe_logger is None or MoELogger is None:
            return
        step = int(self.step_getter()) if self.step_getter is not None else 0
        try:
            self.moe_logger.log_router(
                step=step,
                gate_logits=logits.detach().cpu(),
                topk_idx=topk_idx.detach().cpu(),
                capacity_dropped=int(tokens_dropped),
                aux_loss=float(aux_loss),
                latency_ms=float(latency_ms),
                meta=meta,
            )
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Standard FFN (non-MoE)
# -----------------------------------------------------------------------------
class _StandardFeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_ff, dropout=0.0, use_swiglu=True, activation="gelu"):
        super().__init__()
        self.use_swiglu = bool(use_swiglu)
        self.dropout_p = float(dropout)
        if self.use_swiglu:
            swiglu_dim = int(dim_ff * 4 // 3)
            self.w1 = nn.Linear(d_model, swiglu_dim, bias=False)
            self.w2 = nn.Linear(d_model, swiglu_dim, bias=False)
            self.w3 = nn.Linear(swiglu_dim, d_model, bias=False)
            nn.init.xavier_uniform_(self.w1.weight)
            nn.init.xavier_uniform_(self.w2.weight)
            nn.init.xavier_uniform_(self.w3.weight)
        else:
            self.fc1 = nn.Linear(d_model, dim_ff, bias=False)
            self.fc2 = nn.Linear(dim_ff, d_model, bias=False)
            self.activation = getattr(F, activation.lower(), F.gelu)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        if self.use_swiglu:
            u = self.w1(x)
            v = self.w2(x)
            h = F.silu(u) * v
            if self.training and self.dropout_p > 0:
                h = F.dropout(h, p=self.dropout_p, training=True)
            return self.w3(h)
        else:
            y = self.activation(self.fc1(x))
            if self.training and self.dropout_p > 0:
                y = F.dropout(y, p=self.dropout_p, training=True)
            return self.fc2(y)


# -----------------------------------------------------------------------------
# Public wrapper
# -----------------------------------------------------------------------------
class FeedForwardBlock(nn.Module):
    """
    Unified FFN wrapper.
    - use_moe=True -> MoEFeedForwardDMoE (fast path)
    - use_moe=False -> standard FFN (with optional SwiGLU)
    """
    def __init__(
        self,
        d_model: int,
        dim_ff: int,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        activation: str = "gelu",
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        load_balance_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        moe_capacity_factor: float = 1.25,
        router_type: str = "noisy_topk",
        use_bias: bool = False,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        expert_dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        # Optional logging passthrough:
        moe_logger: Optional["MoELogger"] = None,
        step_getter: Optional[Callable[[], int]] = None,
        log_latency: bool = False,
    ):
        super().__init__()
        self.use_moe = bool(use_moe)
        if self.use_moe:
            self.block = MoEFeedForwardDMoE(
                d_model=d_model,
                d_ff=dim_ff,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                use_swiglu=use_swiglu,
                activation=activation,
                load_balance_weight=load_balance_weight,
                z_loss_weight=z_loss_weight,
                moe_capacity_factor=moe_capacity_factor,
                router_type=router_type,
                use_bias=use_bias,
                input_dropout=input_dropout,
                jitter=jitter,
                expert_dropout=expert_dropout,
                use_gradient_checkpointing=use_gradient_checkpointing,
                moe_logger=moe_logger,
                step_getter=step_getter,
                log_latency=log_latency,
            )
            self._supports_aux = True
        else:
            self.block = _StandardFeedForwardBlock(
                d_model=d_model,
                dim_ff=dim_ff,
                dropout=dropout,
                use_swiglu=use_swiglu,
                activation=activation,
            )
            self._supports_aux = False

    def forward(self, x, return_aux_loss: bool = False, **kwargs):
        if self.use_moe:
            out, aux = self.block(x, return_aux_loss=True, **kwargs)
            return (out, aux) if return_aux_loss else out
        else:
            y = self.block(x)
            return (y, x.new_zeros(())) if return_aux_loss else y
