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

import math
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._tensor import Tensor

from foreblocks.modules.moe.experts.dispatchers import (
    DroplessPackedDispatcher,
    ExpertChoiceDispatcher,
)
from foreblocks.modules.moe.experts.expert_blocks import (
    MoE_FFNExpert,
    MoE_SwiGLUExpert,
    MTPHead,
)
from foreblocks.modules.moe.experts.moe_utils import (
    autocast_bf16_enabled as _autocast_bf16_enabled,
)
from foreblocks.modules.moe.experts.moe_utils import (
    eager_topk_routing,
    maybe_compile,
    optimized_topk_routing,
)
from foreblocks.modules.moe.experts.moe_utils import (
    should_fallback_router_topk as _should_fallback_router_topk,
)
from foreblocks.modules.moe.experts.moe_utils import (
    supports_grouped_prepacked as _supports_grouped_prepacked,
)
from foreblocks.modules.moe.experts.routers import (
    AdaptiveNoisyTopKRouter,
    AuxiliaryTokenRouter,
    ContinuousTopKRouter,
    HashTopKRouter,
    LinearRouter,
    NoisyTopKRouter,
    Router,
    SoftDenseRouter,
    StraightThroughTopKRouter,
)

# Optional import: adjust to your package path
try:
    from foreblocks.modules.moe.experts.moe_logging import MoELogger  # type: ignore
except Exception:
    MoELogger = None  # type: ignore

# Optional grouped kernel path (can accept prepacked B* if your wrapper supports)
try:
    # Prefer a wrapper that accepts B12_cat_prepacked / B3_cat_prepacked
    from foreblocks.modules.moe.kernels.kernels import (
        fused_router_topk,  # type: ignore
        grouped_mlp_swiglu,
    )
except Exception:
    grouped_mlp_swiglu = None  # fallback uses Python loop
    fused_router_topk = None  # type: ignore


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
        expert_choice_tokens_per_expert: int | None = None,
        use_bias: bool = False,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        expert_dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        d_ff_shared: int | None = None,
        shared_scale_init: float = 1.0,
        shared_combine: str = "add",
        adaptive_k_head_dim: int = 32,
        adaptive_k_tau: float = 1.0,
        adaptive_k_baseline_momentum: float = 0.99,
        adaptive_k_sparsity_lambda: float = 0.0,
        # Optional logging
        moe_logger: MoELogger | None = None,
        step_getter: Callable[[], int] | None = None,
        log_latency: bool = False,
        # Optional compile toggles
        compile_router: bool = True,
        compile_experts: bool = True,
        # Optional grouped-kernel controls
        use_grouped_kernel: bool = True,
        use_fused_router_topk: bool = True,
        triton_use_fp16_acc: bool = False,
        triton_use_shared_b: bool = False,
        router_aux_num_aux: int = 1,
        router_expert_bias_init: float = 0.0,
        router_bias_update_rate: float = 0.01,
        router_bias_clip: float = 2.0,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.0,
        mtp_init_scale: float = 0.02,
        moe_use_latent: bool = False,
        moe_latent_dim: int | None = None,
        moe_latent_d_ff: int | None = None,
        # ── SOTA: load balancing, soft capacity, entropy ─────────────────────
        moe_aux_loss_weight: float = 1e-2,
        moe_num_groups: int = 1,
        moe_soft_capacity: bool = False,
        moe_entropy_reg_weight: float = 0.0,
        moe_capacity_min: float = 0.5,
        moe_capacity_max: float = 2.0,
        # ── Auxiliary-loss-free load balancing via per-expert router bias ──
        moe_router_bias_balance: bool = True,
        moe_router_bias_warmup_steps: int = 0,
        moe_router_bias_min_usage: float = 0.01,
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
            "soft_dense",
            "auxiliary_token",
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
        latent_enabled = bool(moe_use_latent)
        if latent_enabled:
            latent_dim = (
                int(moe_latent_dim)
                if moe_latent_dim is not None
                else max(1, int(d_model // 4))
            )
            if latent_dim < 1:
                raise ValueError(
                    "moe_latent_dim must be >= 1 when latent MoE is enabled"
                )
            if latent_dim >= int(d_model):
                raise ValueError(
                    "moe_latent_dim must be < d_model when latent MoE is enabled"
                )
            latent_d_ff = (
                int(moe_latent_d_ff)
                if moe_latent_d_ff is not None
                else max(
                    1, int(round(float(d_ff) * float(latent_dim) / float(d_model)))
                )
            )
            if latent_d_ff < 1:
                raise ValueError(
                    "moe_latent_d_ff must be >= 1 when latent MoE is enabled"
                )
        else:
            latent_dim = int(d_model)
            latent_d_ff = int(d_ff)
        # ── SOTA validation ───────────────────────────────────────────────
        moe_num_groups = int(moe_num_groups)
        if moe_num_groups < 1:
            raise ValueError("moe_num_groups must be >= 1")
        if num_routed % moe_num_groups != 0:
            raise ValueError(
                f"num_routed ({num_routed}) must be divisible by moe_num_groups ({moe_num_groups})"
            )
        moe_aux_loss_weight = float(moe_aux_loss_weight)
        moe_entropy_reg_weight = float(moe_entropy_reg_weight)
        ec_tokens_desc = (
            str(int(expert_choice_tokens_per_expert))
            if expert_choice_tokens_per_expert is not None
            else "auto"
        )
        aux_status = f"lb_w={moe_aux_loss_weight}" if moe_aux_loss_weight > 0 else "off"
        soft_cap = "elastic" if moe_soft_capacity else f"hard({moe_capacity_factor})"
        bias_bal_status = (
            f"on(lr={router_bias_update_rate},clip={router_bias_clip})"
            if bool(moe_router_bias_balance) and router_bias_update_rate > 0
            else "off"
        )
        print(
            f"MoEFeedForwardDMoE: E={num_experts} (shared={num_shared}, routed={num_routed}), "
            f"K={top_k}, routing={routing_mode}, ec_tokens={ec_tokens_desc}, "
            f"swiglu={use_swiglu}, shared={shared_combine}, "
            f"cap={soft_cap}, groups={moe_num_groups}, router={router_type}, "
            f"aux_lb={aux_status}, z_loss={z_loss_weight}, "
            f"entropy_reg={moe_entropy_reg_weight}, bias_balance={bias_bal_status}, "
            f"latent={latent_enabled} latent_dim={latent_dim if latent_enabled else 'off'}"
        )
        self.d_model = d_model
        self.d_ff = d_ff
        self.moe_use_latent = latent_enabled
        self.moe_latent_dim = int(latent_dim)
        self.moe_latent_d_ff = int(latent_d_ff)
        self.routed_d_model = int(latent_dim)
        self.routed_d_ff = int(latent_d_ff)
        self.num_experts = num_experts
        self.num_shared = num_shared
        self.num_routed_experts = num_routed
        self.use_swiglu = bool(use_swiglu)
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
        # ── SOTA params ───────────────────────────────────────────────────
        self.moe_aux_loss_weight = moe_aux_loss_weight
        self.moe_num_groups = moe_num_groups
        self.moe_soft_capacity = bool(moe_soft_capacity)
        self.moe_entropy_reg_weight = moe_entropy_reg_weight
        self.moe_capacity_min = float(moe_capacity_min)
        self.moe_capacity_max = float(moe_capacity_max)
        # ── Router bias balance (aux-loss-free load balancing) ────────────
        self.moe_router_bias_balance = bool(moe_router_bias_balance)
        self.moe_router_bias_warmup_steps = int(moe_router_bias_warmup_steps)
        self.moe_router_bias_min_usage = float(moe_router_bias_min_usage)
        self._current_step = 0

        self.moe_logger = moe_logger
        self.step_getter = step_getter
        self.log_latency = bool(log_latency)
        self.use_grouped_kernel = bool(use_grouped_kernel)
        self.use_fused_router_topk = bool(use_fused_router_topk)
        self._force_eager_router_topk = False
        self.triton_use_fp16_acc = bool(triton_use_fp16_acc)
        self.triton_use_shared_b = bool(triton_use_shared_b)
        self._grouped_kernel_prepacked = _supports_grouped_prepacked()

        self.input_norm = nn.LayerNorm(d_model)
        if self.moe_use_latent:
            self.latent_down_proj = nn.Linear(d_model, self.routed_d_model, bias=False)
            self.latent_up_proj = nn.Linear(self.routed_d_model, d_model, bias=False)
            nn.init.xavier_uniform_(self.latent_down_proj.weight)
            nn.init.xavier_uniform_(self.latent_up_proj.weight)
        else:
            self.latent_down_proj = None
            self.latent_up_proj = None

        # Router
        router: Router
        if router_type == "adaptive_noisy_topk":
            router = AdaptiveNoisyTopKRouter(
                d_model=self.routed_d_model,
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
                d_model=self.routed_d_model,
                num_experts=num_routed,
                top_k=top_k,
                temperature=router_temperature,
                input_dropout=input_dropout,
                jitter=jitter,
                use_bias=use_bias,
                expert_bias_init=router_expert_bias_init,
            )
        elif router_type in (
            "continuous_topk",
            "relaxed_sort_topk",
            "perturb_and_pick_topk",
        ):
            # relaxed_sort_topk and perturb_and_pick_topk use the same continuous
            # top-k implementation with configurable perturbation.
            perturb_std = (
                router_perturb_noise
                if router_type != "relaxed_sort_topk"
                else max(router_perturb_noise, 1e-2)
            )
            router = ContinuousTopKRouter(
                d_model=self.routed_d_model,
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
            num_hashes = (
                1 if router_type == "hash_topk" else max(2, router_hash_num_hashes)
            )
            router = HashTopKRouter(
                d_model=self.routed_d_model,
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
                d_model=self.routed_d_model,
                num_experts=num_routed,
                use_bias=use_bias,
            )
        elif router_type == "noisy_topk":
            router = NoisyTopKRouter(
                d_model=self.routed_d_model,
                num_experts=num_routed,
                input_dropout=input_dropout,
                jitter=jitter,
                use_bias=use_bias,
                expert_bias_init=router_expert_bias_init,
            )
        elif router_type == "soft_dense":
            router = SoftDenseRouter(
                d_model=self.routed_d_model,
                num_experts=num_routed,
                temperature=router_temperature,
                input_dropout=input_dropout,
                use_bias=use_bias,
                expert_bias_init=router_expert_bias_init,
            )
        elif router_type == "auxiliary_token":
            router = AuxiliaryTokenRouter(
                d_model=self.routed_d_model,
                num_experts=num_routed,
                num_aux=router_aux_num_aux,
                temperature=router_temperature,
                input_dropout=input_dropout,
            )
        else:
            raise ValueError(f"unsupported router_type '{router_type}'")
        self.router = maybe_compile(router, enabled=compile_router)

        # Experts
        expert_cls = MoE_SwiGLUExpert if use_swiglu else MoE_FFNExpert
        expert_kwargs = dict(
            d_model=self.routed_d_model,
            d_ff=self.routed_d_ff,
            dropout=dropout,
            expert_dropout=expert_dropout,
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
        # Expert group index: [num_routed] -> group id per expert
        if moe_num_groups > 1 and num_routed % moe_num_groups == 0:
            group_size = num_routed // moe_num_groups
            self.register_buffer(
                "_expert_group",
                torch.cat(
                    [
                        torch.full((group_size,), g, dtype=torch.long)
                        for g in range(moe_num_groups)
                    ]
                ),
            )
        else:
            self.register_buffer(
                "_expert_group", torch.zeros(num_routed, dtype=torch.long)
            )
        self.aux_loss = torch.tensor(0.0)

        # Prepacked expert weights cache (optional fast path)
        self._packed_step = -1
        self._B12_cat = None  # [E, D, 2H] or [E, K, N] depending on wrapper
        self._B3_cat = None  # [E, H, D]
        self._last_num_assignments = 0
        self.last_per_token_k: torch.Tensor | None = None
        self.last_log_prob_k: torch.Tensor | None = None
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
        self, x_flat: torch.Tensor, tau: float | None = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        float,
    ]:
        """
        Unified router output parser.
        Always calls the router with return_raw_logits=True and extracts
        named fields from the RouterOutput dataclass.  Falls back to the
        old tuple path only when torch.compile wraps the module and the
        output deviates from the contract.

        Returns:
          logits, per_token_k, k_logits, k_probs, precomputed_top_p, precomputed_top_i, router_entropy
        """
        from foreblocks.modules.moe.experts.routers import RouterOutput

        out = None
        try:
            out = self.router(x_flat, return_raw_logits=True, tau=tau)
        except TypeError:
            try:
                out = self.router(x_flat, return_raw_logits=True)
            except TypeError:
                out = self.router(x_flat)

        # --- Prefer named fields from RouterOutput ---
        router_entropy = 0.0
        if hasattr(out, "logits"):
            logits = out.logits
            per_token_k = out.per_token_k
            k_logits = out.k_logits
            k_probs = out.k_probs
            top_p = out.top_p
            top_i = out.top_i
            router_entropy = out.router_entropy
        elif isinstance(out, tuple) and len(out) >= 9:
            (
                logits,
                raw,
                sparse,
                k_logits,
                per_token_k,
                k_probs,
                top_p,
                top_i,
                router_entropy,
            ) = out
        elif isinstance(out, tuple) and len(out) >= 8:
            logits, raw, sparse, k_logits, per_token_k, k_probs, top_p, top_i = out
        elif isinstance(out, tuple):
            # Legacy: some compiled wrappers return 2-tuple (logits, sparse)
            logits = out[0]
            per_token_k = k_logits = k_probs = top_p = top_i = None
        else:
            logits = out
            per_token_k = k_logits = k_probs = top_p = top_i = None
        return logits, per_token_k, k_logits, k_probs, top_p, top_i, router_entropy

    def _adaptive_topk_routing(
        self, logits: torch.Tensor, per_token_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _get_router_expert_bias(self) -> torch.Tensor | None:
        r = self.router
        if hasattr(r, "expert_bias"):
            return getattr(r, "expert_bias")
        # torch.compile wrappers often keep original module here
        orig = getattr(r, "_orig_mod", None)
        if orig is not None and hasattr(orig, "expert_bias"):
            return getattr(orig, "expert_bias")
        return None

    def _maybe_update_router_bias(self):
        """Update per-expert router bias toward uniform based on usage EMA.

        This is an auxiliary-loss-free load balancing mechanism: when an expert
        is under-utilized relative to uniform, its bias increases (making the
        router more likely to pick it next time).  When over-utilized, bias
        decreases.  The bias is clamped to [-clip, +clip] to prevent runaway.

        Works independently of ``moe_aux_loss_weight`` — you can disable the
        aux loss entirely and still get balanced load via bias terms alone.
        """
        if not self.moe_router_bias_balance:
            return
        if (not self.training) or self.router_bias_update_rate <= 0:
            return

        # Warmup: skip bias updates until we've seen enough steps
        if self.moe_router_bias_warmup_steps > 0:
            step = (
                int(self.step_getter())
                if self.step_getter is not None
                else self._current_step
            )
            if step < self.moe_router_bias_warmup_steps:
                self._current_step = step
                return
            self._current_step = step

        bias = self._get_router_expert_bias()
        if bias is None:
            return

        with torch.no_grad():
            usage = self.expert_usage.clone()
            # Skip if usage is too sparse (still in early/noise phase)
            if usage.max() < 1e-6 or usage.sum() < 1e-12:
                return
            # Only update experts below the min_usage threshold
            mask = usage < self.moe_router_bias_min_usage
            if not mask.any():
                return
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
        self, out_flat: torch.Tensor, mtp_targets: torch.Tensor | None
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
        downstream_loss: torch.Tensor | None = None,
        mtp_targets: torch.Tensor | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ):
        # Pre-norm
        x_norm = self.input_norm(x)
        x_flat = x_norm.reshape(-1, self.d_model)
        if self.moe_use_latent:
            routed_x = self.latent_down_proj(x_flat)
        else:
            routed_x = x_flat
        Toks = x_flat.size(0)

        device_type = routed_x.device.type
        use_amp = _autocast_bf16_enabled(device_type)

        t0 = self._maybe_start_timer()
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=use_amp
        ):
            # Shared experts (always-on), computed for all tokens
            if self.num_shared > 0:
                # Batched path: pack all-tokens repeated num_shared times into
                # a single grouped_mlp_swiglu call, avoiding sequential launches.
                # Requires SwiGLU experts, add-combine, and the grouped kernel.
                _can_batch_shared = (
                    self.num_shared > 1
                    and self.use_swiglu
                    and self.use_grouped_kernel
                    and grouped_mlp_swiglu is not None
                    and x_flat.is_cuda
                    and self.shared_combine == "add"
                )
                if _can_batch_shared:
                    T_flat = x_flat.size(0)
                    packed_sh = x_flat.repeat(self.num_shared, 1)  # [num_shared*T, D]
                    offsets_sh = (
                        torch.arange(
                            self.num_shared + 1,
                            device=x_flat.device,
                            dtype=torch.int32,
                        )
                        * T_flat
                    )
                    # Unwrap torch.compile wrappers so weight extraction works
                    sh_experts = [
                        getattr(e, "_orig_mod", e) for e in self.shared_experts
                    ]
                    sh_dp = getattr(sh_experts[0], "dropout_p", 0.0)
                    packed_sh_y = grouped_mlp_swiglu(
                        packed_sh,
                        offsets_sh,
                        sh_experts,
                        dropout_p=sh_dp,
                        training=self.training,
                        out_dtype=x_flat.dtype,
                        use_fp16_acc=self.triton_use_fp16_acc,
                        use_shared_b=self.triton_use_shared_b,
                        allow_triton_training=True,
                    )
                    shared_chunks = packed_sh_y.view(
                        self.num_shared, T_flat, self.d_model
                    )  # [num_shared, T, D]
                    shared_out = (
                        shared_chunks * self.shared_scale.view(self.num_shared, 1, 1)
                    ).sum(0)  # [T, D]
                    shared_cat = None
                else:
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

            (
                logits,
                per_token_k,
                _k_logits,
                _k_probs,
                _router_top_p,
                _router_top_i,
                router_entropy,
            ) = self._compute_router_outputs(routed_x, tau=tau)  # [T, E]
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
                    self.expert_choice_dispatcher.pack(routed_x, logits)
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
                        and (not self._force_eager_router_topk)
                        and (not self.training)
                        and logits.is_cuda
                        and (self.top_k in (1, 2))
                    )
                    if can_use_fused_topk:
                        try:
                            top_p, top_i = fused_router_topk(logits, self.top_k)
                        except RuntimeError as exc:
                            if not _should_fallback_router_topk(exc):
                                raise
                            self._force_eager_router_topk = True
                            self.use_fused_router_topk = False
                            top_p, top_i = eager_topk_routing(logits, self.top_k)
                    elif self._force_eager_router_topk:
                        top_p, top_i = eager_topk_routing(logits, self.top_k)
                    else:
                        try:
                            top_p, top_i = optimized_topk_routing(logits, self.top_k)
                        except RuntimeError as exc:
                            if not _should_fallback_router_topk(exc):
                                raise
                            self._force_eager_router_topk = True
                            self.use_fused_router_topk = False
                            top_p, top_i = eager_topk_routing(logits, self.top_k)
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
                        routed_x,
                        top_p,
                        top_i,
                        capacity_factor=self.moe_capacity_factor,
                        soft_capacity=self.moe_soft_capacity,
                        expert_usage=self.expert_usage,
                        soft_capacity_min=self.moe_capacity_min,
                        soft_capacity_max=self.moe_capacity_max,
                    )
                )
            self._last_num_assignments = int(experts_seq.numel())

            if packed_x.numel() == 0:
                out_flat = x_flat.new_zeros((Toks, self.d_model))
                if self.shared_combine == "concat" and shared_cat is not None:
                    out_flat = self.shared_proj(
                        torch.cat([out_flat, shared_cat], dim=-1)
                    )
                elif shared_out is not None:
                    out_flat = out_flat + shared_out
                out = out_flat.reshape_as(x_norm)
                aux = x.new_zeros(())
                self.aux_loss = aux.detach()
                self.last_mtp_loss = aux.detach()
                latency_ms = self._maybe_stop_timer(t0)
                aux_scalar = float(aux.detach())
                self._populate_router_last(
                    logits,
                    top_i,
                    dropped,
                    aux_scalar,
                    latency_ms,
                    meta,
                    per_token_k=per_token_k,
                    router_entropy=router_entropy,
                )
                self._maybe_log_router(
                    logits,
                    top_i,
                    dropped,
                    aux_scalar,
                    latency_ms,
                    meta,
                    per_token_k=per_token_k,
                    router_entropy=router_entropy,
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
            routed_out = routed_x.new_zeros((Toks, self.routed_d_model))
            self.dispatcher.scatter_back(routed_out, packed_y, tokens_seq)
            if self.moe_use_latent:
                out_flat = self.latent_up_proj(routed_out)
            else:
                out_flat = routed_out
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
        aux_scalar = float(aux.detach())
        self._populate_router_last(
            logits,
            top_i,
            dropped,
            aux_scalar,
            latency_ms,
            meta,
            per_token_k=per_token_k,
            router_entropy=router_entropy,
        )
        self._maybe_log_router(
            logits,
            top_i,
            dropped,
            aux_scalar,
            latency_ms,
            meta,
            per_token_k=per_token_k,
            router_entropy=router_entropy,
        )

        return (out, aux) if return_aux_loss else out

    # ---------- Aux losses ----------
    def _compute_aux_loss(
        self,
        logits: torch.Tensor,
        num_tokens: int,
        experts_seq: torch.Tensor,
        num_assignments: int | None = None,
    ):
        aux = logits.new_zeros(())
        num_routed = self.num_routed_experts

        # ── 1. Z-loss (Qwen2.5 / DeepSeek style) ────────────────────────
        if self.z_loss_weight > 0:
            log_z = torch.logsumexp(logits.view(-1, num_routed), dim=-1)
            aux = aux + self.z_loss_weight * 0.5 * (log_z**2).mean()

        # ── 2. Standard load-balancing auxiliary loss (GShard / Mixtral) ─
        if self.moe_aux_loss_weight > 0:
            # Gate probs over experts: [T, E]
            gate_probs = F.softmax(logits, dim=-1)  # [T, E]
            # Fraction of tokens routed to each expert (from top-k assignments)
            expert_freq = torch.zeros(
                num_routed, device=logits.device, dtype=logits.dtype
            )
            if experts_seq.numel() > 0:
                expert_freq = torch.bincount(experts_seq, minlength=num_routed).to(
                    dtype=logits.dtype
                )
            expert_freq = expert_freq / max(num_tokens * self.top_k, 1)  # [E]
            # Average gate probability per expert: [E]
            avg_gate = gate_probs.mean(dim=0)  # [E]
            if self.moe_num_groups > 1:
                # Group-wise: balance within each group
                group_size = num_routed // self.moe_num_groups
                group_aux = 0.0
                for g in range(self.moe_num_groups):
                    start = g * group_size
                    end = start + group_size
                    freq_g = expert_freq[start:end]
                    gate_g = avg_gate[start:end]
                    group_aux = group_aux + (freq_g * gate_g).sum()
                aux = aux + self.moe_aux_loss_weight * group_aux
            else:
                # Standard (global) load-balancing loss
                aux = aux + self.moe_aux_loss_weight * (expert_freq * avg_gate).sum()

        # ── 3. Router entropy regularization (encourages diverse routing) ─
        if self.moe_entropy_reg_weight > 0:
            gate_probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            token_entropy = -(gate_probs * log_probs).sum(dim=-1).mean()  # scalar
            max_entropy = math.log(num_routed)
            # Encourage high entropy: penalty = max_entropy - actual_entropy
            aux = aux - self.moe_entropy_reg_weight * (max_entropy - token_entropy)

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

    def get_last_per_token_k(self) -> torch.Tensor | None:
        return self.last_per_token_k

    # ---------- Stats ----------
    @torch.no_grad()
    def get_expert_stats(self) -> dict[str, torch.Tensor]:
        eps = float(self._eps)
        usage = self.expert_usage.clamp_min(eps)
        entropy = -(usage * usage.log()).sum()
        target = torch.full_like(usage, 1.0 / max(self.num_routed_experts, 1))
        uniform_mse = (usage - target).pow(2).mean()
        bias = self._get_router_expert_bias()
        if bias is None:
            bias = torch.zeros_like(usage)
        stats: dict[str, torch.Tensor] = {
            "expert_usage": self.expert_usage.clone(),
            "usage_entropy": entropy,
            "max_usage": self.expert_usage.max(),
            "min_usage": self.expert_usage.min(),
            "usage_uniform_mse": uniform_mse,
            "router_expert_bias": bias.detach().clone(),
        }
        # ── Group-wise stats (Qwen2.5-MoE style) ────────────────────────
        if self.moe_num_groups > 1:
            group_usage = torch.zeros(
                self.moe_num_groups, device=usage.device, dtype=usage.dtype
            )
            group_size = self.num_routed_experts // self.moe_num_groups
            for g in range(self.moe_num_groups):
                start = g * group_size
                end = start + group_size
                group_usage[g] = usage[start:end].mean()
            stats["group_usage"] = group_usage
            group_target = torch.full_like(group_usage, 1.0 / self.moe_num_groups)
            stats["group_uniform_mse"] = (group_usage - group_target).pow(2).mean()
        return stats

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
        meta: dict[str, Any] | None,
        per_token_k: torch.Tensor | None = None,
        router_entropy: float = 0.0,
    ):
        r = self.router
        try:
            r.last_gate_logits = logits.detach()
            r.last_topk_idx = topk_idx.detach()
            if per_token_k is not None:
                r.last_per_token_k = per_token_k.detach()
            r.last_tokens_dropped = int(tokens_dropped)
            r.last_aux_loss = float(aux_loss)
            r.last_latency_ms = float(latency_ms)
            r.last_meta = meta
            r.last_router_entropy = float(router_entropy)
        except Exception:
            pass

    def _maybe_log_router(
        self,
        logits: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens_dropped: int,
        aux_loss: float,
        latency_ms: float,
        meta: dict[str, Any] | None,
        per_token_k: torch.Tensor | None = None,
        router_entropy: float = 0.0,
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
            # Store router entropy for reporting
            if hasattr(self.moe_logger, "buff"):
                self.moe_logger.buff.setdefault("router_entropy", []).append(
                    float(router_entropy)
                )
        except Exception:
            pass
