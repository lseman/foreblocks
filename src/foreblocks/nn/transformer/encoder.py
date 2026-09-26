"""Transformer encoder stack with patching, variate mixing, and depth routing."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.nn.embeddings.patching import PatchInfo, PatchTokenizer
from foreblocks.nn.transformer.base import BaseTransformer, StackTrace
from foreblocks.nn.transformer.config import Role, TransformerConfig
from foreblocks.nn.transformer.layers.encoder import TransformerEncoderLayer
from foreblocks.nn.transformer.layers.mixing import MixingTransformer
from foreblocks.nn.transformer.runtime.contiguous import (
    contiguous_quantile_loss,
    prepare_contiguous_forecast,
    project_contiguous_quantiles,
)
from foreblocks.nn.transformer.runtime.execution import ModelLayerInvokeStrategy
from foreblocks.nn.transformer.runtime.forward import (
    LayerResult,
    execute_encoder_layer,
    prepare_encoder_input,
)
from foreblocks.nn.transformer.runtime.outputs import TransformerEncoderOutput
from foreblocks.nn.transformer.runtime.routing import (
    gateskip_active_mask_from_padding,
    gather_padding_mask,
    gather_sequence_tokens,
    gather_square_mask,
)
from foreblocks.studio.node_spec import node


@node(
    type_id="transformer_encoder",
    name="Transformer Encoder",
    category="Encoder",
    outputs=["encoder"],
    color="bg-gradient-to-br from-green-700 to-green-800",
)
class TransformerEncoder(BaseTransformer):
    """Encoder stack over ``[B, T, C]`` inputs.

    Example::

        encoder = TransformerEncoder(input_size=4, d_model=64, n_heads=4,
                                     patching="none", residual="gateskip")
        memory = encoder(x).last_hidden_state
    """

    role: ClassVar[Role] = "encoder"

    def _build_role_modules(self) -> None:
        config = self.config
        d_model = config.d_model
        self.patcher = (
            PatchTokenizer(
                d_model,
                config.patch_len,
                config.patch_stride,
                pad_end=config.patch_pad_end,
            )
            if config.patching == "shared"
            else None
        )
        channel = config.patching == "channel"
        self.channel_patch_embed = (
            nn.Linear(config.patch_len, d_model) if channel else None
        )
        self.channel_fuse = (
            nn.Linear(config.input_size * d_model, d_model)
            if channel and config.channel_fuse == "linear"
            else None
        )
        variate = config.variate_attention
        self.variate_output_fuse = (
            nn.Linear(config.input_size * d_model, d_model)
            if variate and config.variate_fuse == "linear"
            else None
        )
        self.missing_token = nn.Parameter(torch.empty(d_model)) if variate else None
        if self.missing_token is not None:
            nn.init.normal_(self.missing_token, mean=0.0, std=config.init_std)
        self.contiguous_patch_head = (
            nn.Linear(d_model, config.patch_len * len(config.quantiles))
            if config.contiguous_decoding
            else None
        )
        # Padding mask and patch geometry of the last forward, for decoders
        # that attend to this encoder's (patched) memory.
        self.last_memory_key_padding_mask: torch.Tensor | None = None
        self.last_patch_info: PatchInfo | None = None

    def _make_layer(self, config: TransformerConfig) -> nn.Module:
        if config.variate_attention:
            return MixingTransformer(config)
        return TransformerEncoderLayer(config)

    # ---- Channel patching -----------------------------------------------------
    def _channel_patchify(self, src: torch.Tensor) -> tuple[torch.Tensor, PatchInfo]:
        """Embed each channel's patches, then fuse channels per patch."""
        config = self.config
        batch_size, length, channels = src.shape
        x = src.transpose(1, 2).contiguous()  # [B,C,T]
        pad = _end_padding(length, config.patch_len, config.patch_stride)
        if config.patch_pad_end and pad > 0:
            x = F.pad(x, (0, pad))
        patches = x.unfold(2, config.patch_len, config.patch_stride)  # [B,C,Np,P]
        num_patches = patches.size(2)
        tokens = self.channel_patch_embed(patches)  # [B,C,Np,D]
        if config.channel_fuse == "mean":
            tokens = tokens.mean(dim=1)
        else:
            tokens = tokens.permute(0, 2, 1, 3).reshape(
                batch_size, num_patches, channels * config.d_model
            )
            tokens = self.channel_fuse(tokens)
        info = PatchInfo(
            T_orig=length,
            T_pad=x.size(-1),
            n_patches=num_patches,
            patch_len=config.patch_len,
            stride=config.patch_stride,
        )
        return tokens, info

    # ---- Variate mixing -------------------------------------------------------
    def _fuse_variate_hidden(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        fuse = self.config.variate_fuse
        if fuse == "none":
            return hidden_states
        if fuse == "mean":
            if padding_mask is None:
                return hidden_states.mean(dim=1)
            valid = (~padding_mask).unsqueeze(-1).to(dtype=hidden_states.dtype)
            return (hidden_states * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        batch_size, num_variates, token_length, d_model = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(
            batch_size, token_length, num_variates * d_model
        )
        return self.variate_output_fuse(hidden_states)

    def _fuse_variate_mask(
        self, padding_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        if padding_mask is None or self.config.variate_fuse == "none":
            return padding_mask
        return padding_mask.all(dim=1)

    def _forward_variate_stack(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
        *,
        output_hidden_states: bool,
        output_attentions: bool,
        return_dict: bool,
        preserve_variate_axis: bool,
    ) -> torch.Tensor | TransformerEncoderOutput:
        use_ckpt = self._use_layer_checkpointing()
        trace = StackTrace.start(x, output_hidden_states)
        variate_attentions: list[torch.Tensor] = []

        for index in range(self.config.num_layers):
            layer = self._resolve_layer(index)
            if use_ckpt and not output_attentions:

                def run_layer(
                    value: torch.Tensor, mixing_layer: nn.Module = layer
                ) -> torch.Tensor:
                    return mixing_layer(
                        value, src_key_padding_mask, sequence_mask=src_mask
                    )[0]

                x = self._run_with_checkpoint(run_layer, x, use_checkpoint=True)
                sequence_weights = variate_weights = None
            else:
                x, sequence_weights, variate_weights = layer(
                    x,
                    src_key_padding_mask,
                    sequence_mask=src_mask,
                    need_weights=output_attentions,
                )
            trace.record(
                index, LayerResult(hidden_states=x, self_attention=sequence_weights)
            )
            if variate_weights is not None:
                variate_attentions.append(variate_weights)

        x = self._finish_stack(x, trace, None)
        hidden_states = trace.hidden_states
        if preserve_variate_axis:
            fused_mask, fused_output = src_key_padding_mask, x
        else:
            fused_mask = self._fuse_variate_mask(src_key_padding_mask)
            fused_output = self._fuse_variate_hidden(x, src_key_padding_mask)
            if hidden_states is not None:
                hidden_states = [
                    self._fuse_variate_hidden(state, src_key_padding_mask)
                    for state in hidden_states
                ]
        self.last_memory_key_padding_mask = fused_mask

        if not return_dict:
            return fused_output
        return TransformerEncoderOutput(
            last_hidden_state=fused_output,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            aux_loss=self.aux_loss,
            padding_mask=fused_mask,
            attentions=tuple(trace.attentions) if trace.attentions else None,
            variate_attentions=(
                tuple(variate_attentions) if variate_attentions else None
            ),
        )

    # ---- Contiguous multi-patch forecasting ----------------------------------
    def forecast_contiguous(
        self,
        target: torch.Tensor,
        horizon: int = 0,
        *,
        past_only_covariates: torch.Tensor | None = None,
        past_future_covariates: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        past_only_mask: torch.Tensor | None = None,
        past_future_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast an entire horizon from masked future patches in one pass.

        Inputs use Foreblocks' time-major convention. ``target`` and
        ``past_only_covariates`` are ``[B, context, C]``. Known-future
        covariates are ``[B, context + horizon, C_known]`` and remain visible
        over the horizon. The result is ``[B, horizon, C_target, Q]``.
        """
        if self.contiguous_patch_head is None:
            raise RuntimeError("set contiguous_decoding=True to forecast contiguously")
        config = self.config
        prepared = prepare_contiguous_forecast(
            target,
            horizon,
            input_size=config.input_size,
            patch_length=config.patch_len,
            past_only_covariates=past_only_covariates,
            past_future_covariates=past_future_covariates,
            target_mask=target_mask,
            past_only_mask=past_only_mask,
            past_future_mask=past_future_mask,
            context_mask=context_mask,
        )
        encoded = self(
            prepared.values,
            src_key_padding_mask=prepared.attention_padding,
            value_mask=prepared.value_mask,
            preserve_variate_axis=True,
            return_dict=False,
        )
        return project_contiguous_quantiles(
            encoded,
            self.contiguous_patch_head,
            prepared,
            patch_length=config.patch_len,
            num_quantiles=len(config.quantiles),
        )

    def contiguous_quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pinball loss for ``[B,H,C,Q]`` contiguous forecasts."""
        return contiguous_quantile_loss(
            predictions, targets, self.config.quantiles, mask
        )

    # ---- Forward ---------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,  # [B, T, C]
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,  # [B,T] bool
        time_features: torch.Tensor | None = None,  # [B, T, F_tf]
        active_mask: torch.Tensor | None = None,  # [B,T] bool
        value_mask: torch.Tensor | None = None,
        preserve_variate_axis: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> torch.Tensor | TransformerEncoderOutput:
        """Encode ``src``.

        ``active_mask`` marks tokens eligible for GateSkip gating and
        Mixture-of-Depths routing; it defaults to the non-padded tokens.
        ``value_mask`` marks missing values (variate mixing only).
        """
        self._begin_forward()
        prepared = prepare_encoder_input(
            self, src, src_key_padding_mask, active_mask, value_mask=value_mask
        )
        x = prepared.hidden_states
        src_key_padding_mask = prepared.padding_mask
        active_mask = prepared.active_mask
        if active_mask is None:
            active_mask = gateskip_active_mask_from_padding(src_key_padding_mask)
        self.last_memory_key_padding_mask = src_key_padding_mask
        self.last_patch_info = prepared.patch_info

        x = self._add_input_positions(x)
        x = self._add_time_encoding(x, time_features)
        x = self._input_dropout(x)

        if self.config.variate_attention:
            return self._forward_variate_stack(
                x,
                src_mask,
                src_key_padding_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
                preserve_variate_axis=preserve_variate_axis,
            )
        if preserve_variate_axis:
            raise ValueError("preserve_variate_axis requires variate_attention=True")

        attention_residual_state = self._init_attention_residual_state(x)
        streams = self._init_streams(x)
        invoke = ModelLayerInvokeStrategy(
            owner=self, use_checkpoint=self._use_layer_checkpointing()
        )
        budget = self._gate_budget()

        def run_layer(index: int, hidden: torch.Tensor) -> LayerResult:
            nonlocal streams
            result = execute_encoder_layer(
                self,
                invoke,
                layer_index=index,
                hidden_states=hidden,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                budget=budget,
                streams=streams,
                attention_residual_state=attention_residual_state,
                active_mask=active_mask,
                output_attentions=output_attentions,
            )
            streams = result.streams
            return result

        def run_routed(index, hidden, layer, routed_indices, routed_slots):
            nonlocal streams
            x_routed = gather_sequence_tokens(hidden, routed_indices)
            x_routed_out, streams = invoke.run_encoder_layer(
                layer,
                x_routed,
                src_mask=gather_square_mask(src_mask, routed_indices),
                src_key_padding_mask=gather_padding_mask(
                    src_key_padding_mask, routed_indices, routed_slots
                ),
                gate_budget=budget,
                streams=streams,
                attention_residual_state=attention_residual_state,
                active_mask=routed_slots,
            )
            return x_routed, x_routed_out

        trace = StackTrace.start(x, output_hidden_states)
        x = self._run_layer_stack(
            x, active_mask, trace, run_layer=run_layer, run_routed=run_routed
        )
        x = self._finish_stack(x, trace, attention_residual_state)
        if not return_dict:
            return x
        return TransformerEncoderOutput(
            last_hidden_state=x,
            hidden_states=(
                tuple(trace.hidden_states) if trace.hidden_states is not None else None
            ),
            aux_loss=self.aux_loss,
            padding_mask=src_key_padding_mask,
            router_states=tuple(trace.router_states) if trace.router_states else None,
            attentions=tuple(trace.attentions) if trace.attentions else None,
        )

    def _add_time_encoding(
        self, x: torch.Tensor, time_features: torch.Tensor | None
    ) -> torch.Tensor:
        # Timestep-space inputs only: patch tokens have no per-step features.
        if (
            self.config.patching != "none"
            or self.time_encoder is None
            or time_features is None
        ):
            return x
        time_emb = self.time_encoder(time_features)  # [B, T, D]
        variate = self.config.variate_attention
        expected = (x.shape[0], x.shape[2]) if variate else x.shape[:2]
        if time_emb.shape[:2] != expected:
            raise ValueError(
                f"time features shape {tuple(time_emb.shape)} does not match "
                f"input {tuple(x.shape)}"
            )
        return x + (time_emb[:, None] if variate else time_emb)


def _end_padding(length: int, patch_len: int, stride: int) -> int:
    """Padding that lets strided patches cover every step."""
    if length <= 0:
        return 0
    if length < patch_len:
        return patch_len - length
    num_patches = math.ceil((length - patch_len) / stride) + 1
    return max(0, (num_patches - 1) * stride + patch_len - length)


__all__ = ["TransformerEncoder"]
