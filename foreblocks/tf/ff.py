from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F


if TYPE_CHECKING:
    from .experts.moe_logging import MoELogger

try:
    from .compute.kernels import grouped_mlp_swiglu
except Exception:
    grouped_mlp_swiglu = None


class _StandardFeedForwardBlock(nn.Module):
    def __init__(
        self,
        d_model,
        dim_ff,
        dropout=0.0,
        use_swiglu=True,
        activation="gelu",
        use_triton: bool = True,
        triton_use_fp16_acc: bool = False,
        triton_use_shared_b: bool = False,
    ):
        super().__init__()
        self.use_swiglu = bool(use_swiglu)
        self.dropout_p = float(dropout)
        self.use_triton = bool(use_triton)
        self.triton_use_fp16_acc = bool(triton_use_fp16_acc)
        self.triton_use_shared_b = bool(triton_use_shared_b)
        if self.use_swiglu:
            swiglu_dim = int(dim_ff * 4 // 3)
            self.swiglu_dim = int(swiglu_dim)
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

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        return (
            self.use_swiglu
            and self.use_triton
            and grouped_mlp_swiglu is not None
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )

    def _forward_swiglu_triton(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_packed = x.reshape(-1, x_shape[-1]).contiguous()
        S = int(x_packed.size(0))

        offsets = torch.tensor([0, S], device=x_packed.device, dtype=torch.int32)
        B12 = torch.cat(
            [self.w1.weight.t().contiguous(), self.w2.weight.t().contiguous()], dim=-1
        ).unsqueeze(0)
        B3 = self.w3.weight.t().contiguous().unsqueeze(0)

        y_packed = grouped_mlp_swiglu(
            x_packed,
            offsets,
            experts=[],
            dropout_p=self.dropout_p,
            training=self.training,
            out_dtype=x_packed.dtype,
            use_fp16_acc=self.triton_use_fp16_acc,
            use_shared_b=self.triton_use_shared_b,
            allow_triton_training=True,
            B12_cat_prepacked=B12,
            B3_cat_prepacked=B3,
        )
        return y_packed.reshape(*x_shape[:-1], y_packed.size(-1))

    def forward(self, x):
        if self.use_swiglu:
            if self._can_use_triton(x):
                try:
                    return self._forward_swiglu_triton(x)
                except Exception:
                    pass
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
        moe_logger: MoELogger | None = None,
        step_getter: Callable[[], int] | None = None,
        log_latency: bool = False,
        compile_router: bool = True,
        compile_experts: bool = True,
        use_grouped_kernel: bool = True,
        use_fused_router_topk: bool = True,
        use_triton: bool = True,
        triton_use_fp16_acc: bool = False,
        triton_use_shared_b: bool = False,
        router_expert_bias_init: float = 0.0,
        router_bias_update_rate: float = 0.01,
        router_bias_clip: float = 2.0,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.0,
        mtp_init_scale: float = 0.02,
        moe_use_latent: bool = False,
        moe_latent_dim: int | None = None,
        moe_latent_d_ff: int | None = None,
    ):
        super().__init__()
        self.use_moe = bool(use_moe)
        if self.use_moe:
            from .experts.moe import MoEFeedForwardDMoE

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
                moe_use_latent=moe_use_latent,
                moe_latent_dim=moe_latent_dim,
                moe_latent_d_ff=moe_latent_d_ff,
            )
            self._supports_aux = True
        else:
            self.block = _StandardFeedForwardBlock(
                d_model=d_model,
                dim_ff=dim_ff,
                dropout=dropout,
                use_swiglu=use_swiglu,
                activation=activation,
                use_triton=use_triton,
                triton_use_fp16_acc=triton_use_fp16_acc,
                triton_use_shared_b=triton_use_shared_b,
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

    def get_last_per_token_k(self) -> torch.Tensor | None:
        if not self.use_moe:
            return None
        if not hasattr(self.block, "get_last_per_token_k"):
            return None
        return self.block.get_last_per_token_k()
