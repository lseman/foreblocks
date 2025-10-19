# transformer_moe_dmoe.py
# -----------------------------------------------------------------------------
# Enhanced dMoE FeedForward with safe aux handling (no graph reuse across steps)
# - Router unpacking robust to compiled/uncompiled + varied tuple arities
# - Z-loss computed from local logits (no stored raw logits with grad)
# - Stored metrics (aux_loss) are detached to avoid cross-step graph reuse
# - Capacity-aware dropless dispatcher
# - Optional grouped_swiglu kernel path
# -----------------------------------------------------------------------------

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# If you have a custom kernel, import it here; else keep the fallback path below.
try:
    from .kernels import grouped_mlp_swiglu  # type: ignore
except Exception:
    grouped_mlp_swiglu = None  # Fallback will use Python loop


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def maybe_compile(mod):
    """Wrap with torch.compile if available, otherwise return the module."""
    try:
        return torch.compile(mod, dynamic=True)
    except Exception:
        return mod


# -----------------------------------------------------------------------------
# Optimized Top-K (Scripted for speed)
# -----------------------------------------------------------------------------
@torch.jit.script
def optimized_topk_routing(logits: torch.Tensor, k: int):
    probs = F.softmax(logits, dim=-1)
    if k == 1:
        top_p, top_i = torch.max(probs, dim=-1, keepdim=True)
    else:
        top_p, top_i = torch.topk(probs, k, dim=-1, sorted=False)
    return top_p, top_i


# -----------------------------------------------------------------------------
# Enhanced Dispatcher with Capacity Drop
# -----------------------------------------------------------------------------
class DroplessPackedDispatcher:
    """
    Sort-pack with buffer reuse + capacity-based random drop.
    """
    def __init__(self, num_experts: int, top_k: int, capacity_factor: float = 1.25):
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self._buffer_size = 0
        self._sort_buffer: Optional[torch.Tensor] = None
        self._counts_buffer: Optional[torch.Tensor] = None
        self._offsets_buffer: Optional[torch.Tensor] = None

    def _ensure_buffers(self, size: int, device: torch.device):
        if self._sort_buffer is None or self._buffer_size < size:
            self._buffer_size = size
            self._sort_buffer = torch.empty(size, dtype=torch.long, device=device)
            self._counts_buffer = torch.zeros(self.num_experts, dtype=torch.long, device=device)
            self._offsets_buffer = torch.zeros(self.num_experts + 1, dtype=torch.long, device=device)

    @torch.no_grad()
    def pack(
        self,
        x_flat: torch.Tensor,
        topk_p: torch.Tensor,
        topk_i: torch.Tensor,
        capacity_factor: Optional[float] = None,
    ):
        device = x_flat.device
        T, D = x_flat.shape
        K = topk_i.shape[1]
        S = T * K
        capacity_factor = capacity_factor or self.capacity_factor

        if S == 0:
            empty = x_flat.new_zeros((0, D))
            empty_idx = x_flat.new_zeros(0, dtype=torch.long)
            empty_offsets = x_flat.new_zeros(self.num_experts + 1, dtype=torch.long)
            return empty, empty, empty_idx, empty_idx, empty_offsets

        self._ensure_buffers(S, device)
        experts = topk_i.reshape(-1)
        weights = topk_p.reshape(-1)
        tokens = torch.arange(T, device=device, dtype=torch.long).unsqueeze(1).expand(T, K).reshape(-1)

        # sort by expert (stable)
        torch.argsort(experts, stable=True, out=self._sort_buffer[:S])
        sort_idx = self._sort_buffer[:S]
        experts_sorted = experts[sort_idx]
        weights_sorted = weights[sort_idx]
        tokens_sorted = tokens[sort_idx]

        counts = torch.bincount(experts_sorted, minlength=self.num_experts)
        self._offsets_buffer.zero_()
        torch.cumsum(counts, 0, out=self._offsets_buffer[1:])
        offsets = self._offsets_buffer.clone()

        packed_x = x_flat.index_select(0, tokens_sorted)
        packed_w = weights_sorted.unsqueeze(1)

        # Capacity drop: per-expert random subsample if overload
        total_capacity = math.ceil(T * K * capacity_factor)
        per_expert_cap = max(1, math.ceil(total_capacity / self.num_experts))
        kept_mask = torch.ones(S, dtype=torch.bool, device=device)
        for e in range(self.num_experts):
            s = int(offsets[e].item())
            t = int(offsets[e + 1].item())
            num = t - s
            if num > per_expert_cap:
                local_keep = torch.randperm(num, device=device)[:per_expert_cap]
                local_mask = torch.zeros(num, dtype=torch.bool, device=device)
                local_mask[local_keep] = True
                kept_mask[s:t] = local_mask

        kept = kept_mask.nonzero(as_tuple=True)[0]
        if len(kept) < S:
            packed_x = packed_x[kept]
            packed_w = packed_w[kept]
            experts_sorted = experts_sorted[kept]
            tokens_sorted = tokens_sorted[kept]
            # Recompute offsets post-drop
            counts = torch.bincount(experts_sorted, minlength=self.num_experts)
            offsets = torch.cumsum(
                torch.cat([torch.tensor([0], device=device, dtype=torch.long), counts]),
                dim=0
            )

        return packed_x, packed_w, experts_sorted, tokens_sorted, offsets

    @staticmethod
    def scatter_back(
        out_accum: torch.Tensor,
        packed_y: torch.Tensor,
        tokens_seq: torch.Tensor,
    ):
        out_accum.index_add_(0, tokens_seq, packed_y)
        return out_accum


# -----------------------------------------------------------------------------
# Router (Noisy Top-K)
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
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None
        self.jitter = jitter
        self.clamp_min, self.clamp_max = clamp_range
        self._init_router()

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

        # stable softmax later; for now provide logits + probs compatibility
        m = logits.max(dim=-1, keepdim=True).values
        probs = torch.exp(logits - m)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

        if return_raw_logits:
            return raw, logits, probs
        else:
            return logits, probs


# -----------------------------------------------------------------------------
# Experts
# -----------------------------------------------------------------------------
class MoE_SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, expert_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.expert_dropout_p = expert_dropout
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
        self.dropout_p = dropout
        self.expert_dropout_p = expert_dropout
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
# Enhanced dMoE Layer (safe aux + robust router unpack)
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
    ):
        super().__init__()
        print(
            f"MoEFeedForwardDMoE: num_experts={num_experts}, top_k={top_k}, "
            f"use_swiglu={use_swiglu}, capacity_factor={moe_capacity_factor}, router_type={router_type}"
        )
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.load_balance_weight = load_balance_weight
        self.z_loss_weight = z_loss_weight
        self.moe_capacity_factor = moe_capacity_factor
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.input_norm = nn.LayerNorm(d_model)

        # Router
        if router_type == "noisy_topk":
            self.router = NoisyTopKRouter(
                d_model=d_model, num_experts=num_experts,
                input_dropout=input_dropout, jitter=jitter, use_bias=use_bias
            )
        else:
            self.router = nn.Linear(d_model, num_experts, bias=use_bias)
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        # Experts
        expert_cls = MoE_SwiGLUExpert if use_swiglu else MoE_FFNExpert
        expert_kwargs = {
            'd_model': d_model,
            'd_ff': d_ff,
            'dropout': dropout,
            'expert_dropout': expert_dropout
        }
        if not use_swiglu:
            expert_kwargs['activation'] = activation

        self.experts = nn.ModuleList([expert_cls(**expert_kwargs) for _ in range(num_experts)])

        # Optional compile for speed
        self.router = maybe_compile(self.router)
        self.experts = nn.ModuleList([maybe_compile(e) for e in self.experts])

        # Dispatcher with capacity
        self.dispatcher = DroplessPackedDispatcher(num_experts, self.top_k, moe_capacity_factor)

        # Stats / buffers
        self.register_buffer("_eps", torch.tensor(1e-8))
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("momentum", torch.tensor(0.999))

        # metrics-only store; always detached copies
        self.aux_loss = torch.tensor(0.0)

    def _compute_router_logits(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Robustly obtain router logits as a single Tensor, regardless of whether:
        - the router is a NoisyTopKRouter,
        - it has been torch.compile'd (which can obscure isinstance checks and kwargs),
        - it returns (raw, logits, probs), (logits, probs), or just logits.
        """
        out = None

        # Try to request raw logits explicitly (works on uncompiled router).
        try:
            out = self.router(x_flat, return_raw_logits=True)  # preferred
        except TypeError:
            # Compiled wrappers often drop/rename kwargs; call without it.
            out = self.router(x_flat)

        # Normalize outputs to a single logits tensor
        if isinstance(out, tuple):
            if len(out) == 3:
                _raw_unused, logits, _probs = out
            elif len(out) == 2:
                logits, _probs = out
            elif len(out) == 1:
                logits = out[0]
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

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True):
        # Pre-norm
        x_norm = self.input_norm(x)
        x_flat = x_norm.reshape(-1, self.d_model)

        # Router
        logits = self._compute_router_logits(x_flat)
        T = x_flat.size(0)
        top_p, top_i = optimized_topk_routing(logits, self.top_k)

        # Track usage
        if self.training:
            self._update_expert_usage_ema(top_p, top_i)

        # Pack with capacity drop
        packed_x, packed_w, experts_seq, tokens_seq, offsets = self.dispatcher.pack(
            x_flat, top_p, top_i, capacity_factor=self.moe_capacity_factor
        )

        # If nothing dispatched (e.g., empty batch)
        if packed_x.numel() == 0:
            out = torch.zeros_like(x_flat).reshape_as(x_norm)
            if return_aux_loss:
                self.aux_loss = x.new_zeros(()).detach()
                return out, self.aux_loss
            else:
                self.aux_loss = x.new_zeros(()).detach()
                return out

        # Expert compute
        if grouped_mlp_swiglu is not None:
            packed_y = grouped_mlp_swiglu(
                packed_x, offsets, self.experts,
                dropout_p=getattr(self.experts[0], "dropout_p", 0.0),
                training=self.training,
                out_dtype=packed_x.dtype,
            )
        else:
            packed_y = torch.empty_like(packed_x)
            for e in range(self.num_experts):
                s = int(offsets[e].item()); t = int(offsets[e + 1].item())
                if t <= s:
                    continue
                seg = packed_x[s:t]
                if self.use_gradient_checkpointing and self.training:
                    y = torch.utils.checkpoint.checkpoint(self.experts[e], seg, use_reentrant=False)
                else:
                    y = self.experts[e](seg)
                packed_y[s:t] = y

        # Gating
        packed_y.mul_(packed_w)

        # Scatter back
        out_flat = torch.zeros_like(x_flat)
        self.dispatcher.scatter_back(out_flat, packed_y, tokens_seq)
        out = out_flat.reshape_as(x_norm)

        # Aux (with grad for current step; detached copy stored for metrics)
        if return_aux_loss:
            aux = self._compute_aux_loss(logits, T, experts_seq)
            self.aux_loss = aux.detach()
            return out, aux
        else:
            self.aux_loss = x.new_zeros(()).detach()
            return out

    def _compute_aux_loss(self, logits: torch.Tensor, num_tokens: int, experts_seq: torch.Tensor):
        aux = logits.new_zeros(())

        # Z-loss based on current logits (no reliance on stored tensors)
        if self.z_loss_weight > 0:
            log_z = torch.logsumexp(logits.view(-1, self.num_experts), dim=-1)
            aux = aux + self.z_loss_weight * 0.5 * (log_z ** 2).mean()

        # Load-balance (quadratic on f*p)
        if self.load_balance_weight > 0:
            probs = F.softmax(logits, dim=-1)
            with torch.no_grad():
                counts = torch.bincount(experts_seq, minlength=self.num_experts).float()
                f = counts / max(1, (num_tokens * self.top_k))  # post-capacity fraction
            p = probs.mean(dim=0)
            aux = aux + self.load_balance_weight * self.num_experts * (f * p).pow(2).sum()

        return aux

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
        # keep metrics placeholder on the same device, detached
        self.aux_loss = torch.zeros((), device=self.expert_usage.device).detach()


# -----------------------------------------------------------------------------
# Non-MoE FFN
# -----------------------------------------------------------------------------
class _StandardFeedForwardBlock(nn.Module):
    """Standard FFN with optional SwiGLU."""
    def __init__(self, d_model, dim_ff, dropout=0.0, use_swiglu=True, activation="gelu"):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout_p = dropout
        if use_swiglu:
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
# Public Wrapper
# -----------------------------------------------------------------------------
class FeedForwardBlock(nn.Module):
    """
    Unified FFN wrapper. When use_moe=True, employs enhanced dMoE.
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
    ):
        super().__init__()
        self.use_moe = use_moe
        if use_moe:
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

    def forward(self, x, return_aux_loss: bool = False):
        if self.use_moe:
            out, aux = self.block(x, return_aux_loss=True)
            return (out, aux) if return_aux_loss else out
        else:
            y = self.block(x)
            return (y, x.new_zeros(())) if return_aux_loss else y
