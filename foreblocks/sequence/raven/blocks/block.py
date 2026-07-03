"""foreblocks.sequence.raven.blocks.block.

Transformer-style Raven block with FLA-backed attention and MLP.

Wraps a Raven or FLA attention layer followed by a RavenMLP, with optional
attnres residual projections and fused norm paths. Designed as the building
block for Raven decoder stacks with hybrid attention support.

Core API:
- RavenBlock: transformer-style block with attention + MLP sub-layers

"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from foreblocks.ops.raven import Attention, Cache, RavenMLP, RMSNorm, fused_attnres
from foreblocks.sequence.raven.blocks.raven import Raven


class RavenBlock(nn.Module):
    """Transformer-style Raven block with local structure and FLA-backed ops."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        norm_cls = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.attn_norm = norm_cls(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn["layers"]:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn["num_heads"],
                num_kv_heads=config.attn["num_kv_heads"],
                qkv_bias=config.attn["qkv_bias"],
                window_size=config.attn["window_size"],
                rope_theta=config.attn["rope_theta"],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            self.attn = Raven(
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                num_slots=config.num_slots,
                feature_map=config.feature_map,
                use_output_gate=config.use_output_gate,
                gate_fn=config.hidden_act,
                decay_type=config.decay_type,
                topk=config.topk,
                bias_rmm=config.bias_rmm,
                add_gumbel_noise=config.add_gumbel_noise,
                router_score=config.router_score,
                router_type=config.router_type,
                use_rope=config.use_rope,
                rope_theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
                gate_logit_normalizer=config.gate_logit_normalizer,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
            )
        self.mlp_norm = norm_cls(config.hidden_size, eps=config.norm_eps)
        self.mlp = RavenMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            block_size = config.attnres_block_size
            self.attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.attn_res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.attnres_is_attn_boundary = (2 * layer_idx) % block_size == 0
            self.attnres_is_mlp_boundary = (2 * layer_idx + 1) % block_size == 0
            self.attn_res_proj._is_attnres_proj = True
            self.mlp_res_proj._is_attnres_proj = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        attnres_states: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[
        torch.FloatTensor, torch.Tensor | None, Cache | None, list[torch.Tensor] | None
    ]:
        if self.use_attnres:
            prefix_sum = hidden_states
            if attnres_states is None:
                hidden_states = self.attn_norm(prefix_sum)
                attnres_states = [prefix_sum]
                prefix_sum = None
            else:
                residuals = [*attnres_states, prefix_sum]
                if self.attnres_is_attn_boundary:
                    attnres_states = residuals
                    prefix_sum = None
                hidden_states = fused_attnres(
                    query=self.attn_res_proj.weight,
                    residuals=residuals,
                    rms_weight=self.attn_res_norm.weight,
                    output_rms_weight=self.attn_norm.weight,
                    rms_eps=self.attn_res_norm.eps,
                )
        else:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)

        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        if self.use_attnres:
            prefix_sum = (
                hidden_states if prefix_sum is None else prefix_sum + hidden_states
            )
            residuals = [*attnres_states, prefix_sum]
            if self.attnres_is_mlp_boundary:
                attnres_states = residuals
                prefix_sum = None
            hidden_states = fused_attnres(
                query=self.mlp_res_proj.weight,
                residuals=residuals,
                rms_weight=self.mlp_res_norm.weight,
                output_rms_weight=self.mlp_norm.weight,
                rms_eps=self.mlp_res_norm.eps,
            )
        elif self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)

        hidden_states = self.mlp(hidden_states, **kwargs)
        if self.use_attnres:
            hidden_states = (
                hidden_states if prefix_sum is None else prefix_sum + hidden_states
            )
        else:
            hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values, attnres_states


__all__ = ["RavenBlock"]
