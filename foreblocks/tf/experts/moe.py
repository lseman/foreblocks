# transformer_moe_dmoe.py
# -----------------------------------------------------------------------------
# Enhanced dMoE FeedForward (fast path)
#  - Router: softmax over Top-K only
#  - Routing mode: token_choice (default) or expert_choice
#  - Dispatcher: vectorized capacity prune
#  - bf16 autocast for experts + router (CUDA only)
#  - Optional pre-stacked expert weights reuse per step
#  - Safe aux handling | optional MoELogger integration
# -----------------------------------------------------------------------------

from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._tensor import Tensor

# Optional import: adjust to your package path
try:
    from .moe_logging import MoELogger  # type: ignore
except Exception:
    MoELogger = None  # type: ignore
from .routers import (
    AdaptiveNoisyTopKRouter,
    ContinuousTopKRouter,
    HashTopKRouter,
    LinearRouter,
    NoisyTopKRouter,
    Router,
    StraightThroughTopKRouter,
)
from .dispatchers import DroplessPackedDispatcher, ExpertChoiceDispatcher

# Optional grouped kernel path (can accept prepacked B* if your wrapper supports)
try:
    # Prefer a wrapper that accepts B12_cat_prepacked / B3_cat_prepacked
    from ..compute.kernels import fused_router_topk, grouped_mlp_swiglu  # type: ignore
except Exception:
    grouped_mlp_swiglu = None  # fallback uses Python loop
    fused_router_topk = None  # type: ignore


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def maybe_compile(
    mod: nn.Module, enabled: bool = True, dynamic: bool = True
) -> nn.Module:
    """
    Wrap with torch.compile if available and enabled, else return the module.
    Kept tiny/defensive to avoid surprising failures.
    """
    if not enabled:
        return mod
    try:
        return torch.compile(mod, dynamic=dynamic)
    except Exception:
        return mod


def _autocast_bf16_enabled(device_type: str) -> bool:
    """
    Decide whether to enable bf16 autocast for the given device_type.
    For now: only CUDA; CPU autocast in bf16 is still not universally stable.
    """
    if device_type != "cuda":
        return False
    return torch.cuda.is_available()


def _supports_grouped_prepacked() -> bool:
    """
    Detect once whether grouped_mlp_swiglu supports prepacked expert args.
    Avoids exception-driven probing on every forward.
    """
    if grouped_mlp_swiglu is None:
        return False
    try:
        sig = inspect.signature(grouped_mlp_swiglu)
        return ("B12_cat_prepacked" in sig.parameters) and (
            "B3_cat_prepacked" in sig.parameters
        )
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Optimized Top-K routing (softmax over Top-K only)
# -----------------------------------------------------------------------------
@torch.jit.script
def optimized_topk_routing(logits: torch.Tensor, k: int):
    """
    Returns (top_p, top_i) for K experts per token.
    - Take topk on raw logits (no full softmax over E).
    - Normalize only within the chosen K.

    logits: [T, E]
    top_p:  [T, K]  (probabilities within top-k)
    top_i:  [T, K]  (expert indices)
    """
    if k == 1:
        top_v, top_i = torch.max(logits, dim=-1, keepdim=True)
        # When K=1, the top expert gets prob=1
        top_p = torch.ones_like(top_v)
        return top_p, top_i
    else:
        top_v, top_i = torch.topk(logits, k, dim=-1, sorted=False)
        m = top_v.max(dim=-1, keepdim=True).values
        expv = torch.exp(top_v - m)
        top_p = expv / (expv.sum(dim=-1, keepdim=True) + 1e-12)
        return top_p, top_i


# -----------------------------------------------------------------------------
# Experts
# -----------------------------------------------------------------------------
class MoE_SwiGLUExpert(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.0, expert_dropout: float = 0.0
    ):
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
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        expert_dropout: float = 0.0,
    ):
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


class MTPHead(nn.Module):
    """Simple multi-token/multi-horizon prediction head bank."""

    def __init__(self, d_model: int, n_extra: int = 3, init_scale: float = 0.02):
        super().__init__()
        self.n_extra = int(n_extra)
        self.extra_heads = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(self.n_extra)]
        )
        std = float(init_scale) / math.sqrt(max(self.n_extra + 1, 1))
        for h in self.extra_heads:
            nn.init.normal_(h.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, D] -> [N, H, D]
        if self.n_extra <= 0:
            return x.new_zeros((x.size(0), 0, x.size(-1)))
        preds = [h(x) for h in self.extra_heads]
        return torch.stack(preds, dim=1)


# -----------------------------------------------------------------------------
# dMoE FFN (fast path)
# -----------------------------------------------------------------------------
class MoEFeedForwardDMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        num_shared: int = 1,
        top_k: int = 2,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        activation: str = "gelu",
        load_balance_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        moe_capacity_factor: float = 1.25,
        router_type: str = "noisy_topk",
        router_temperature: float = 1.0,
        router_perturb_noise: float = 0.0,
        router_hash_num_hashes: int = 2,
        router_hash_num_buckets: int = 64,
        router_hash_bucket_size: int = 8,
        router_hash_seed: int = 17,
        routing_mode: str = "token_choice",
        expert_choice_tokens_per_expert: Optional[int] = None,
        use_bias: bool = False,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        expert_dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        d_ff_shared: Optional[int] = None,
        shared_scale_init: float = 1.0,
        shared_combine: str = "add",
        adaptive_k_head_dim: int = 32,
        adaptive_k_tau: float = 1.0,
        adaptive_k_baseline_momentum: float = 0.99,
        adaptive_k_sparsity_lambda: float = 0.0,
        # Optional logging
        moe_logger: Optional["MoELogger"] = None,
        step_getter: Optional[Callable[[], int]] = None,
        log_latency: bool = False,
        # Optional compile toggles
        compile_router: bool = True,
        compile_experts: bool = True,
        # Optional grouped-kernel controls
        use_grouped_kernel: bool = True,
        use_fused_router_topk: bool = True,
        triton_use_fp16_acc: bool = False,
        triton_use_shared_b: bool = False,
        router_expert_bias_init: float = 0.0,
        router_bias_update_rate: float = 0.01,
        router_bias_clip: float = 2.0,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.0,
        mtp_init_scale: float = 0.02,
    ):
        super().__init__()
        num_experts = int(num_experts)
        num_shared = int(num_shared)
        if num_shared < 0:
            raise ValueError("num_shared must be >= 0")
        if num_shared >= num_experts:
            raise ValueError("num_shared must be < num_experts")
        num_routed = num_experts - num_shared
        if num_routed < 1:
            raise ValueError("num_experts - num_shared must be >= 1")
        top_k = int(min(top_k, num_routed))
        shared_combine = str(shared_combine)
        router_type = str(router_type).lower()
        if router_type == "soft_topk":
            router_type = "continuous_topk"
        if router_type == "rs_topk":
            router_type = "relaxed_sort_topk"
        if router_type == "perturb_and_pick":
            router_type = "perturb_and_pick_topk"
        routing_mode = str(routing_mode).lower()
        if shared_combine not in ("add", "concat"):
            raise ValueError("shared_combine must be 'add' or 'concat'")
        allowed_router_types = {
            "noisy_topk",
            "adaptive_noisy_topk",
            "linear",
            "st_topk",
            "continuous_topk",
            "relaxed_sort_topk",
            "perturb_and_pick_topk",
            "hash_topk",
            "multi_hash_topk",
        }
        if router_type not in allowed_router_types:
            raise ValueError(
                "router_type must be one of "
                f"{sorted(allowed_router_types)}, got '{router_type}'"
            )
        if routing_mode not in ("token_choice", "expert_choice"):
            raise ValueError("routing_mode must be 'token_choice' or 'expert_choice'")
        if routing_mode == "expert_choice" and router_type == "adaptive_noisy_topk":
            raise ValueError(
                "routing_mode='expert_choice' does not support "
                "router_type='adaptive_noisy_topk'"
            )
        if router_temperature <= 0:
            raise ValueError("router_temperature must be > 0")
        if router_perturb_noise < 0:
            raise ValueError("router_perturb_noise must be >= 0")
        if router_hash_num_hashes < 1:
            raise ValueError("router_hash_num_hashes must be >= 1")
        if router_hash_num_buckets < 2:
            raise ValueError("router_hash_num_buckets must be >= 2")
        if router_hash_bucket_size < 1:
            raise ValueError("router_hash_bucket_size must be >= 1")
        if (
            expert_choice_tokens_per_expert is not None
            and int(expert_choice_tokens_per_expert) < 1
        ):
            raise ValueError("expert_choice_tokens_per_expert must be >= 1 when set")
        ec_tokens_desc = (
            str(int(expert_choice_tokens_per_expert))
            if expert_choice_tokens_per_expert is not None
            else "auto"
        )
        print(
            f"MoEFeedForwardDMoE: E={num_experts} (shared={num_shared}, routed={num_routed}), "
            f"K={top_k}, routing={routing_mode}, ec_tokens={ec_tokens_desc}, "
            f"swiglu={use_swiglu}, shared={shared_combine}, "
            f"cap={moe_capacity_factor}, router={router_type}, "
            f"aux_lb=off, bias_lr={router_bias_update_rate}"
        )
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_shared = num_shared
        self.num_routed_experts = num_routed
        self.top_k = top_k
        self.routing_mode = routing_mode
        self.expert_choice_tokens_per_expert = (
            int(expert_choice_tokens_per_expert)
            if expert_choice_tokens_per_expert is not None
            else None
        )
        self.router_type = router_type
        self.shared_combine = shared_combine
        self.load_balance_weight = float(load_balance_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.moe_capacity_factor = float(moe_capacity_factor)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.adaptive_k_baseline_momentum = float(adaptive_k_baseline_momentum)
        self.adaptive_k_sparsity_lambda = float(adaptive_k_sparsity_lambda)
        self.router_bias_update_rate = float(router_bias_update_rate)
        self.router_bias_clip = float(router_bias_clip)
        self.mtp_loss_weight = float(mtp_loss_weight)

        self.moe_logger = moe_logger
        self.step_getter = step_getter
        self.log_latency = bool(log_latency)
        self.use_grouped_kernel = bool(use_grouped_kernel)
        self.use_fused_router_topk = bool(use_fused_router_topk)
        self.triton_use_fp16_acc = bool(triton_use_fp16_acc)
        self.triton_use_shared_b = bool(triton_use_shared_b)
        self._grouped_kernel_prepacked = _supports_grouped_prepacked()

        self.input_norm = nn.LayerNorm(d_model)

        # Router
        router: Router
        if router_type == "adaptive_noisy_topk":
            router = AdaptiveNoisyTopKRouter(
                d_model=d_model,
                num_experts=num_routed,
                max_k=top_k,
                k_head_dim=adaptive_k_head_dim,
                k_tau=adaptive_k_tau,
                input_dropout=input_dropout,
                jitter=jitter,
                use_bias=use_bias,
                expert_bias_init=router_expert_bias_init,
            )
        elif router_type == "st_topk":
            router = StraightThroughTopKRouter(
                d_model=d_model,
                num_experts=num_routed,
                top_k=top_k,
                temperature=router_temperature,
                input_dropout=input_dropout,
                jitter=jitter,
                use_bias=use_bias,
                expert_bias_init=router_expert_bias_init,
            )
        elif router_type in ("continuous_topk", "relaxed_sort_topk", "perturb_and_pick_topk"):
            # relaxed_sort_topk and perturb_and_pick_topk use the same continuous
            # top-k implementation with configurable perturbation.
            perturb_std = (
                router_perturb_noise
                if router_type != "relaxed_sort_topk"
                else max(router_perturb_noise, 1e-2)
            )
            router = ContinuousTopKRouter(
                d_model=d_model,
                num_experts=num_routed,
                top_k=top_k,
                temperature=router_temperature,
                perturb_std=perturb_std,
                input_dropout=input_dropout,
                jitter=jitter,
                use_bias=use_bias,
                expert_bias_init=router_expert_bias_init,
            )
        elif router_type in ("hash_topk", "multi_hash_topk"):
            num_hashes = 1 if router_type == "hash_topk" else max(2, router_hash_num_hashes)
            router = HashTopKRouter(
                d_model=d_model,
                num_experts=num_routed,
                top_k=top_k,
                num_hashes=num_hashes,
                num_buckets=router_hash_num_buckets,
                bucket_size=router_hash_bucket_size,
                hash_seed=router_hash_seed,
                input_dropout=input_dropout,
                jitter=jitter,
                use_bias=use_bias,
                expert_bias_init=router_expert_bias_init,
            )
        elif router_type == "linear":
            router = LinearRouter(
                d_model=d_model,
                num_experts=num_routed,
                use_bias=use_bias,
            )
        elif router_type == "noisy_topk":
            router = NoisyTopKRouter(
                d_model=d_model,
                num_experts=num_routed,
                input_dropout=input_dropout,
                jitter=jitter,
                use_bias=use_bias,
                expert_bias_init=router_expert_bias_init,
            )
        else:
            raise ValueError(f"unsupported router_type '{router_type}'")
        self.router = maybe_compile(router, enabled=compile_router)

        # Experts
        expert_cls = MoE_SwiGLUExpert if use_swiglu else MoE_FFNExpert
        expert_kwargs = dict(
            d_model=d_model, d_ff=d_ff, dropout=dropout, expert_dropout=expert_dropout
        )
        if not use_swiglu:
            expert_kwargs["activation"] = activation
        if d_ff_shared is None:
            shared_ff = int(d_ff // 2)
        else:
            shared_ff = int(d_ff_shared)
        if num_shared > 0 and shared_ff < 1:
            raise ValueError("d_ff_shared must be >= 1")
        shared_kwargs = dict(
            d_model=d_model,
            d_ff=shared_ff,
            dropout=dropout,
            expert_dropout=expert_dropout,
        )
        if not use_swiglu:
            shared_kwargs["activation"] = activation

        shared_experts = [expert_cls(**shared_kwargs) for _ in range(num_shared)]
        experts = [expert_cls(**expert_kwargs) for _ in range(num_routed)]
        if compile_experts:
            shared_experts = [maybe_compile(e) for e in shared_experts]
            experts = [maybe_compile(e) for e in experts]
        self.shared_experts = nn.ModuleList(shared_experts)
        self.experts = nn.ModuleList(experts)
        if num_shared > 0:
            self.shared_scale = nn.Parameter(
                torch.full((num_shared,), float(shared_scale_init))
            )
            if self.shared_combine == "concat":
                in_dim = d_model * (1 + num_shared)
                self.shared_proj = nn.Linear(in_dim, d_model, bias=False)
                nn.init.xavier_uniform_(self.shared_proj.weight)
            else:
                self.shared_proj = None
        else:
            self.shared_scale = None
            self.shared_proj = None

        # Optional MTP head bank (off by default).
        self.mtp_num_heads = int(mtp_num_heads)
        self.mtp_heads = (
            MTPHead(
                d_model=d_model, n_extra=self.mtp_num_heads, init_scale=mtp_init_scale
            )
            if self.mtp_num_heads > 0
            else None
        )

        # Dispatcher
        self.dispatcher = DroplessPackedDispatcher(
            num_routed, self.top_k, moe_capacity_factor
        )
        self.expert_choice_dispatcher = ExpertChoiceDispatcher(
            num_experts=num_routed,
            top_k=self.top_k,
            capacity_factor=self.moe_capacity_factor,
            tokens_per_expert=self.expert_choice_tokens_per_expert,
        )

        # Metrics / buffers
        self.register_buffer("_eps", torch.tensor(1e-8))
        self.register_buffer("expert_usage", torch.zeros(num_routed))
        self.register_buffer("momentum", torch.tensor(0.999))
        self.aux_loss = torch.tensor(0.0)

        # Prepacked expert weights cache (optional fast path)
        self._packed_step = -1
        self._B12_cat = None  # [E, D, 2H] or [E, K, N] depending on wrapper
        self._B3_cat = None  # [E, H, D]
        self._last_num_assignments = 0
        self.last_per_token_k: Optional[torch.Tensor] = None
        self.last_log_prob_k: Optional[torch.Tensor] = None
        self.register_buffer("adaptive_k_baseline", torch.tensor(0.0))
        self.last_mtp_loss = torch.tensor(0.0)

    # ---------- Prepack expert weights (bf16) and reuse per step ----------
    @torch.no_grad()
    def _refresh_packed_weights(self, step: int):
        if step == self._packed_step:
            return
        w12_list, w3_list = [], []
        for e in self.experts:
            if hasattr(e, "w12") and hasattr(e, "w3"):
                w12 = e.w12.weight.t().contiguous()  # [D, 2H]
                w3 = e.w3.weight.t().contiguous()  # [H, D]
            else:
                # Fallback naming (e.g., gate_up_proj/down_proj style)
                w12 = e.gate_up_proj.weight.t().contiguous()  # type: ignore
                w3 = e.down_proj.weight.t().contiguous()  # type: ignore
            w12_list.append(w12)
            w3_list.append(w3)

        # Move to bf16 to match autocast; keep accumulation fp32 in kernels
        self._B12_cat = torch.stack(
            [w.to(dtype=torch.bfloat16) for w in w12_list], dim=0
        )
        self._B3_cat = torch.stack([w.to(dtype=torch.bfloat16) for w in w3_list], dim=0)
        self._packed_step = step

    def invalidate_packed_weights(self):
        self._packed_step = -1
        self._B12_cat = None
        self._B3_cat = None

    # ---------- Helpers ----------
    def _compute_router_outputs(
        self, x_flat: torch.Tensor, tau: Optional[float] = None
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Unified router output parser.
        Returns:
          logits, per_token_k, k_logits, k_probs, precomputed_top_p, precomputed_top_i
        """
        out = None
        try:
            out = self.router(x_flat, return_raw_logits=True, tau=tau)
        except TypeError:
            try:
                out = self.router(x_flat, return_raw_logits=True)
            except TypeError:
                out = self.router(x_flat)
        per_token_k = None
        k_logits = None
        k_probs = None
        top_p = None
        top_i = None
        if isinstance(out, tuple):
            if len(out) >= 2:
                logits = out[1]
            else:
                logits = out[0]
            if len(out) >= 5:
                k_logits = out[3]
                per_token_k = out[4]
            if len(out) >= 6:
                k_probs = out[5]
            if len(out) >= 8:
                top_p = out[6]
                top_i = out[7]
        else:
            logits = out
        return logits, per_token_k, k_logits, k_probs, top_p, top_i

    def _adaptive_topk_routing(
        self, logits: torch.Tensor, per_token_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sorted top-k so we can take first K entries per token
        top_v, top_i = torch.topk(logits, self.top_k, dim=-1, sorted=True)
        ranks = torch.arange(self.top_k, device=logits.device).unsqueeze(0)
        mask = ranks < per_token_k.unsqueeze(1)
        masked_top_v = top_v.masked_fill(~mask, float("-inf"))
        m = masked_top_v.max(dim=-1, keepdim=True).values
        expv = torch.exp(masked_top_v - m) * mask
        top_p = expv / (expv.sum(dim=-1, keepdim=True) + 1e-12)
        return top_p, top_i

    def _update_expert_usage_ema(self, top_p: torch.Tensor, top_i: torch.Tensor):
        with torch.no_grad():
            cur = torch.zeros(self.num_routed_experts, device=top_p.device)
            cur.scatter_add_(0, top_i.reshape(-1), top_p.reshape(-1))
            cur = cur / max(top_i.size(0), 1)
            decay = float(self.momentum)
            self.expert_usage.mul_(decay).add_(cur, alpha=1.0 - decay)

    def _update_expert_usage_ema_from_assignments(self, experts_seq: torch.Tensor):
        with torch.no_grad():
            if experts_seq.numel() == 0:
                return
            cur = torch.bincount(experts_seq, minlength=self.num_routed_experts).to(
                dtype=self.expert_usage.dtype
            )
            cur = cur / cur.sum().clamp_min(1.0)
            decay = float(self.momentum)
            self.expert_usage.mul_(decay).add_(cur, alpha=1.0 - decay)

    def _get_router_expert_bias(self) -> Optional[torch.Tensor]:
        r = self.router
        if hasattr(r, "expert_bias"):
            return getattr(r, "expert_bias")
        # torch.compile wrappers often keep original module here
        orig = getattr(r, "_orig_mod", None)
        if orig is not None and hasattr(orig, "expert_bias"):
            return getattr(orig, "expert_bias")
        return None

    def _maybe_update_router_bias(self):
        if (not self.training) or self.router_bias_update_rate <= 0:
            return
        bias = self._get_router_expert_bias()
        if bias is None:
            return
        with torch.no_grad():
            usage = self.expert_usage
            usage = usage / usage.sum().clamp_min(1e-12)
            target = torch.full_like(usage, 1.0 / self.num_routed_experts)
            bias.add_(self.router_bias_update_rate * (target - usage))
            if self.router_bias_clip > 0:
                bias.clamp_(-self.router_bias_clip, self.router_bias_clip)

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

    def _reshape_mtp_targets(
        self, mtp_targets: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        if mtp_targets.dim() == 3:
            # [N, H, D]
            tgt = mtp_targets
        elif mtp_targets.dim() == 4:
            # [B, T, H, D] -> [B*T, H, D]
            tgt = mtp_targets.reshape(-1, mtp_targets.size(-2), mtp_targets.size(-1))
        else:
            raise ValueError(
                f"mtp_targets must be [N,H,D] or [B,T,H,D], got {tuple(mtp_targets.shape)}"
            )

        if tgt.size(0) != num_tokens:
            raise ValueError(
                f"mtp_targets token count {tgt.size(0)} does not match routed tokens {num_tokens}"
            )
        if tgt.size(-1) != self.d_model:
            raise ValueError(
                f"mtp_targets feature dim {tgt.size(-1)} does not match d_model {self.d_model}"
            )
        return tgt

    def _compute_mtp_loss(
        self, out_flat: torch.Tensor, mtp_targets: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if (
            (not self.training)
            or self.mtp_heads is None
            or self.mtp_loss_weight <= 0
            or mtp_targets is None
        ):
            return out_flat.new_zeros(())

        mtp_preds = self.mtp_heads(out_flat)  # [N, H, D]
        if mtp_preds.size(1) == 0:
            return out_flat.new_zeros(())

        target = self._reshape_mtp_targets(mtp_targets, out_flat.size(0)).to(
            device=mtp_preds.device, dtype=mtp_preds.dtype
        )
        if target.size(1) < mtp_preds.size(1):
            raise ValueError(
                f"mtp_targets horizon {target.size(1)} < mtp_num_heads {mtp_preds.size(1)}"
            )
        target = target[:, : mtp_preds.size(1), :]
        return self.mtp_loss_weight * F.mse_loss(mtp_preds, target)

    # ---------- Forward ----------
    def forward(
        self,
        x: torch.Tensor,  # [..., T, D] or [B, T, D]
        return_aux_loss: bool = True,
        tau: float = 1.0,
        downstream_loss: Optional[torch.Tensor] = None,
        mtp_targets: Optional[torch.Tensor] = None,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ):
        # Pre-norm
        x_norm = self.input_norm(x)
        x_flat = x_norm.reshape(-1, self.d_model)
        Toks = x_flat.size(0)

        device_type = x_flat.device.type
        use_amp = _autocast_bf16_enabled(device_type)

        t0 = self._maybe_start_timer()
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=use_amp
        ):
            # Shared experts (always-on), computed for all tokens
            if self.num_shared > 0:
                shared_out = torch.zeros_like(x_flat)
                shared_list = []
                for s_idx, s_exp in enumerate(self.shared_experts):
                    y = s_exp(x_flat)
                    scale = self.shared_scale[s_idx]
                    y = y * scale
                    shared_out.add_(y)
                    if self.shared_combine == "concat":
                        shared_list.append(y)
                shared_cat = (
                    torch.cat(shared_list, dim=-1)
                    if self.shared_combine == "concat"
                    else None
                )
            else:
                shared_out = None
                shared_cat = None

            logits, per_token_k, _k_logits, _k_probs, _router_top_p, _router_top_i = (
                self._compute_router_outputs(x_flat, tau=tau)
            )  # [T, E]
            if self.routing_mode == "expert_choice":
                self.last_per_token_k = None
                self.last_log_prob_k = None
                top_i = torch.topk(
                    logits,
                    k=min(self.top_k, self.num_routed_experts),
                    dim=-1,
                    sorted=False,
                ).indices
                packed_x, packed_w, experts_seq, tokens_seq, offsets, dropped = (
                    self.expert_choice_dispatcher.pack(x_flat, logits)
                )
                if self.training:
                    self._update_expert_usage_ema_from_assignments(experts_seq)
                    self._maybe_update_router_bias()
            else:
                if _router_top_p is not None and _router_top_i is not None:
                    top_p, top_i = _router_top_p, _router_top_i
                    self.last_per_token_k = None
                    self.last_log_prob_k = None
                elif per_token_k is None:
                    can_use_fused_topk = (
                        (fused_router_topk is not None)
                        and self.use_fused_router_topk
                        and (not self.training)
                        and logits.is_cuda
                        and (self.top_k in (1, 2))
                    )
                    if can_use_fused_topk:
                        top_p, top_i = fused_router_topk(logits, self.top_k)
                    else:
                        top_p, top_i = optimized_topk_routing(logits, self.top_k)
                    self.last_per_token_k = None
                    self.last_log_prob_k = None
                else:
                    if _k_logits is None:
                        raise RuntimeError(
                            "router returned per_token_k without k_logits"
                        )
                    top_p, top_i = self._adaptive_topk_routing(logits, per_token_k)
                    # Cache K stats for REINFORCE-style loss
                    log_probs = F.log_softmax(_k_logits, dim=-1)
                    log_prob_k = log_probs.gather(
                        -1, (per_token_k - 1).unsqueeze(-1)
                    ).squeeze(-1)
                    self.last_per_token_k = per_token_k.detach()
                    self.last_log_prob_k = log_prob_k

                if self.training:
                    self._update_expert_usage_ema(top_p, top_i)
                    self._maybe_update_router_bias()

                # Pack (vectorized capacity prune)
                packed_x, packed_w, experts_seq, tokens_seq, offsets, dropped = (
                    self.dispatcher.pack(
                        x_flat, top_p, top_i, capacity_factor=self.moe_capacity_factor
                    )
                )
            self._last_num_assignments = int(experts_seq.numel())

            if packed_x.numel() == 0:
                out = torch.zeros_like(x_flat)
                if self.shared_combine == "concat" and shared_cat is not None:
                    out = self.shared_proj(torch.cat([out, shared_cat], dim=-1))
                elif shared_out is not None:
                    out = out + shared_out
                out = out.reshape_as(x_norm)
                aux = x.new_zeros(())
                self.aux_loss = aux.detach()
                self.last_mtp_loss = aux.detach()
                latency_ms = self._maybe_stop_timer(t0)
                self._populate_router_last(
                    logits,
                    top_i,
                    dropped,
                    float(aux),
                    latency_ms,
                    meta,
                    per_token_k=per_token_k,
                )
                self._maybe_log_router(
                    logits,
                    top_i,
                    dropped,
                    float(aux),
                    latency_ms,
                    meta,
                    per_token_k=per_token_k,
                )
                return (out, aux) if return_aux_loss else out

            # Expert compute
            if self.use_grouped_kernel and grouped_mlp_swiglu is not None:
                # Optional prepacked expert weights (refresh once per step)
                step = int(self.step_getter()) if self.step_getter is not None else -1
                use_prepacked = self._grouped_kernel_prepacked and (not self.training)
                if use_prepacked:
                    self._refresh_packed_weights(step)
                    packed_y = grouped_mlp_swiglu(
                        packed_x,
                        offsets,
                        self.experts,
                        dropout_p=getattr(self.experts[0], "dropout_p", 0.0),
                        training=self.training,
                        out_dtype=packed_x.dtype,
                        use_fp16_acc=self.triton_use_fp16_acc,
                        use_shared_b=self.triton_use_shared_b,
                        allow_triton_training=True,
                        B12_cat_prepacked=self._B12_cat,  # may be ignored if wrapper doesn't accept
                        B3_cat_prepacked=self._B3_cat,
                    )
                else:
                    # Backward-compatible grouped kernel wrapper (no prepacked weights)
                    packed_y = grouped_mlp_swiglu(
                        packed_x,
                        offsets,
                        self.experts,
                        dropout_p=getattr(self.experts[0], "dropout_p", 0.0),
                        training=self.training,
                        out_dtype=packed_x.dtype,
                        use_fp16_acc=self.triton_use_fp16_acc,
                        use_shared_b=self.triton_use_shared_b,
                        allow_triton_training=True,
                    )
            else:
                # Fallback: loop (optionally checkpoint)
                packed_y = torch.empty_like(packed_x)
                for e_idx in range(self.num_routed_experts):
                    s = int(offsets[e_idx].item())
                    t = int(offsets[e_idx + 1].item())
                    if t <= s:
                        continue
                    seg = packed_x[s:t]
                    if self.use_gradient_checkpointing and self.training:
                        y = torch.utils.checkpoint.checkpoint(
                            self.experts[e_idx], seg, use_reentrant=False
                        )
                    else:
                        y = self.experts[e_idx](seg)
                    packed_y[s:t] = y

            # Weight by gate prob and scatter back
            packed_y.mul_(packed_w)
            out_flat = torch.zeros_like(x_flat)
            self.dispatcher.scatter_back(out_flat, packed_y, tokens_seq)
            if self.shared_combine == "concat" and shared_cat is not None:
                out_flat = self.shared_proj(torch.cat([out_flat, shared_cat], dim=-1))
            elif shared_out is not None:
                out_flat = out_flat + shared_out
            out = out_flat.reshape_as(x_norm)

        # Aux (compute possibly outside autocast for numerical stability)
        if return_aux_loss:
            aux = self._compute_aux_loss(
                logits, Toks, experts_seq, num_assignments=self._last_num_assignments
            )
            mtp_loss = self._compute_mtp_loss(out_flat, mtp_targets)
            aux = aux + mtp_loss
            self.last_mtp_loss = mtp_loss.detach()
            if (
                self.training
                and downstream_loss is not None
                and self.last_log_prob_k is not None
                and self.last_per_token_k is not None
                and self.adaptive_k_sparsity_lambda != 0.0
            ):
                reward = -downstream_loss + self.adaptive_k_sparsity_lambda * (
                    self.top_k - self.last_per_token_k.float().mean()
                )
                baseline = self.adaptive_k_baseline
                with torch.no_grad():
                    r_mean: Tensor = reward.detach().mean()
                    self.adaptive_k_baseline.mul_(
                        self.adaptive_k_baseline_momentum
                    ).add_(r_mean, alpha=1.0 - self.adaptive_k_baseline_momentum)
                advantage = (reward - baseline).detach()
                if advantage.dim() == 0:
                    advantage = advantage.expand_as(self.last_log_prob_k)
                reinforce_loss = (advantage * self.last_log_prob_k).mean()
                aux = aux + reinforce_loss
            self.aux_loss = aux.detach()
        else:
            aux = x.new_zeros(())
            self.aux_loss = aux.detach()
            self.last_mtp_loss = aux.detach()

        latency_ms = self._maybe_stop_timer(t0)
        self._populate_router_last(
            logits,
            top_i,
            dropped,
            float(aux),
            latency_ms,
            meta,
            per_token_k=per_token_k,
        )
        self._maybe_log_router(
            logits,
            top_i,
            dropped,
            float(aux),
            latency_ms,
            meta,
            per_token_k=per_token_k,
        )

        return (out, aux) if return_aux_loss else out

    # ---------- Aux losses ----------
    def _compute_aux_loss(
        self,
        logits: torch.Tensor,
        num_tokens: int,
        experts_seq: torch.Tensor,
        num_assignments: Optional[int] = None,
    ):
        aux = logits.new_zeros(())

        # Z-loss
        if self.z_loss_weight > 0:
            log_z = torch.logsumexp(logits.view(-1, self.num_routed_experts), dim=-1)
            aux = aux + self.z_loss_weight * 0.5 * (log_z**2).mean()

        # Classic load-balance auxiliary loss intentionally removed.
        # Expert utilization is handled via router expert-bias adaptation.

        return aux

    def compute_adaptive_k_reinforce_loss(
        self, reward: torch.Tensor, detach_reward: bool = True
    ) -> torch.Tensor:
        if self.last_log_prob_k is None:
            return reward.new_zeros(())
        if detach_reward:
            reward_t = reward.detach()
        else:
            reward_t = reward
        baseline = self.adaptive_k_baseline
        # Update baseline with mean reward
        with torch.no_grad():
            r_mean = reward_t.mean()
            self.adaptive_k_baseline.mul_(self.adaptive_k_baseline_momentum).add_(
                r_mean, alpha=1.0 - self.adaptive_k_baseline_momentum
            )
        advantage = reward_t - baseline
        # If reward is scalar, broadcast over tokens
        if advantage.dim() == 0:
            advantage = advantage.expand_as(self.last_log_prob_k)
        return (advantage * self.last_log_prob_k).mean()

    def get_last_per_token_k(self) -> Optional[torch.Tensor]:
        return self.last_per_token_k

    # ---------- Stats ----------
    @torch.no_grad()
    def get_expert_stats(self) -> Dict[str, torch.Tensor]:
        eps = float(self._eps)
        usage = self.expert_usage.clamp_min(eps)
        entropy = -(usage * usage.log()).sum()
        target = torch.full_like(usage, 1.0 / max(self.num_routed_experts, 1))
        uniform_mse = (usage - target).pow(2).mean()
        bias = self._get_router_expert_bias()
        if bias is None:
            bias = torch.zeros_like(usage)
        return {
            "expert_usage": self.expert_usage.clone(),
            "usage_entropy": entropy,
            "max_usage": self.expert_usage.max(),
            "min_usage": self.expert_usage.min(),
            "usage_uniform_mse": uniform_mse,
            "router_expert_bias": bias.detach().clone(),
        }

    @torch.no_grad()
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
        per_token_k: Optional[torch.Tensor] = None,
    ):
        r = self.router
        try:
            # Not all router variants necessarily have these fields,
            # so we guard with try/except.
            r.last_gate_logits = logits.detach()
            r.last_topk_idx = topk_idx.detach()
            if per_token_k is not None:
                r.last_per_token_k = per_token_k.detach()
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
        per_token_k: Optional[torch.Tensor] = None,
    ):
        if self.moe_logger is None or MoELogger is None:
            return
        step = int(self.step_getter()) if self.step_getter is not None else 0
        try:
            self.moe_logger.log_router(
                step=step,
                gate_logits=logits.detach().cpu(),
                topk_idx=topk_idx.detach().cpu(),
                per_token_k=per_token_k.detach().cpu()
                if per_token_k is not None
                else None,
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
    def __init__(
        self, d_model, dim_ff, dropout=0.0, use_swiglu=True, activation="gelu"
    ):
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
    - routing_mode='token_choice' (default) or 'expert_choice'
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
        num_shared: int = 1,
        top_k: int = 2,
        load_balance_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        moe_capacity_factor: float = 1.25,
        router_type: str = "noisy_topk",
        router_temperature: float = 1.0,
        router_perturb_noise: float = 0.0,
        router_hash_num_hashes: int = 2,
        router_hash_num_buckets: int = 64,
        router_hash_bucket_size: int = 8,
        router_hash_seed: int = 17,
        routing_mode: str = "token_choice",
        expert_choice_tokens_per_expert: Optional[int] = None,
        use_bias: bool = False,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        expert_dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        d_ff_shared: Optional[int] = None,
        shared_scale_init: float = 1.0,
        shared_combine: str = "add",
        adaptive_k_head_dim: int = 32,
        adaptive_k_tau: float = 1.0,
        adaptive_k_baseline_momentum: float = 0.99,
        adaptive_k_sparsity_lambda: float = 0.0,
        # Optional logging passthrough:
        moe_logger: Optional["MoELogger"] = None,
        step_getter: Optional[Callable[[], int]] = None,
        log_latency: bool = False,
        # Optional compile flags (forwarded to MoEFeedForwardDMoE)
        compile_router: bool = True,
        compile_experts: bool = True,
        # Optional grouped-kernel flags (forwarded to MoEFeedForwardDMoE)
        use_grouped_kernel: bool = True,
        use_fused_router_topk: bool = True,
        triton_use_fp16_acc: bool = False,
        triton_use_shared_b: bool = False,
        router_expert_bias_init: float = 0.0,
        router_bias_update_rate: float = 0.01,
        router_bias_clip: float = 2.0,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.0,
        mtp_init_scale: float = 0.02,
    ):
        super().__init__()
        self.use_moe = bool(use_moe)
        if self.use_moe:
            self.block = MoEFeedForwardDMoE(
                d_model=d_model,
                d_ff=dim_ff,
                num_experts=num_experts,
                num_shared=num_shared,
                top_k=top_k,
                dropout=dropout,
                use_swiglu=use_swiglu,
                activation=activation,
                load_balance_weight=load_balance_weight,
                z_loss_weight=z_loss_weight,
                moe_capacity_factor=moe_capacity_factor,
                router_type=router_type,
                router_temperature=router_temperature,
                router_perturb_noise=router_perturb_noise,
                router_hash_num_hashes=router_hash_num_hashes,
                router_hash_num_buckets=router_hash_num_buckets,
                router_hash_bucket_size=router_hash_bucket_size,
                router_hash_seed=router_hash_seed,
                routing_mode=routing_mode,
                expert_choice_tokens_per_expert=expert_choice_tokens_per_expert,
                use_bias=use_bias,
                input_dropout=input_dropout,
                jitter=jitter,
                expert_dropout=expert_dropout,
                use_gradient_checkpointing=use_gradient_checkpointing,
                d_ff_shared=d_ff_shared,
                shared_scale_init=shared_scale_init,
                shared_combine=shared_combine,
                adaptive_k_head_dim=adaptive_k_head_dim,
                adaptive_k_tau=adaptive_k_tau,
                adaptive_k_baseline_momentum=adaptive_k_baseline_momentum,
                adaptive_k_sparsity_lambda=adaptive_k_sparsity_lambda,
                moe_logger=moe_logger,
                step_getter=step_getter,
                log_latency=log_latency,
                compile_router=compile_router,
                compile_experts=compile_experts,
                use_grouped_kernel=use_grouped_kernel,
                use_fused_router_topk=use_fused_router_topk,
                triton_use_fp16_acc=triton_use_fp16_acc,
                triton_use_shared_b=triton_use_shared_b,
                router_expert_bias_init=router_expert_bias_init,
                router_bias_update_rate=router_bias_update_rate,
                router_bias_clip=router_bias_clip,
                mtp_num_heads=mtp_num_heads,
                mtp_loss_weight=mtp_loss_weight,
                mtp_init_scale=mtp_init_scale,
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

    def compute_adaptive_k_reinforce_loss(
        self, reward: torch.Tensor, detach_reward: bool = True
    ) -> torch.Tensor:
        if not self.use_moe:
            return reward.new_zeros(())
        if not hasattr(self.block, "compute_adaptive_k_reinforce_loss"):
            return reward.new_zeros(())
        return self.block.compute_adaptive_k_reinforce_loss(
            reward, detach_reward=detach_reward
        )

    def get_last_per_token_k(self) -> Optional[torch.Tensor]:
        if not self.use_moe:
            return None
        if not hasattr(self.block, "get_last_per_token_k"):
            return None
        return self.block.get_last_per_token_k()
