"""Transformer decoder stack with incremental caching and generation."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn

from foreblocks.nn.attention.cache.kv import StaticKVCache
from foreblocks.nn.transformer.base import BaseTransformer, StackTrace
from foreblocks.nn.transformer.config import GenerationConfig, Role, TransformerConfig
from foreblocks.nn.transformer.layers.decoder import TransformerDecoderLayer
from foreblocks.nn.transformer.runtime.cache import DecoderCacheManager
from foreblocks.nn.transformer.runtime.decoding import GenerationEngine
from foreblocks.nn.transformer.runtime.execution import ModelLayerInvokeStrategy
from foreblocks.nn.transformer.runtime.forward import (
    LayerResult,
    build_decoder_output,
    execute_decoder_layer,
    prepare_decoder_state,
    validate_memory_padding_mask,
)
from foreblocks.nn.transformer.runtime.mtp import build_decoder_mtp_targets
from foreblocks.nn.transformer.runtime.routing import (
    gateskip_active_mask_from_padding,
    gather_padding_mask,
    gather_query_mask,
    gather_sequence_tokens,
    gather_square_mask,
)
from foreblocks.nn.transformer.runtime.state import (
    AttentionCacheState,
    DecoderLayerState,
    DecoderState,
)
from foreblocks.studio.node_spec import node


@node(
    type_id="transformer_decoder",
    name="Transformer Decoder",
    category="Decoder",
    outputs=["decoder"],
    color="bg-gradient-to-br from-purple-700 to-purple-800",
)
class TransformerDecoder(BaseTransformer):
    """Decoder stack with cross-attention to encoder memory.

    Example::

        decoder = TransformerDecoder(input_size=2, output_size=2, d_model=64)
        forecast = decoder(tgt, memory).last_hidden_state
        generated = decoder.generate(tgt[:, :1], memory, max_new_tokens=12)
    """

    role: ClassVar[Role] = "decoder"

    def _build_role_modules(self) -> None:
        config = self.config
        self.output_projection = (
            nn.Identity()
            if config.output_size == config.d_model
            else nn.Linear(config.d_model, config.output_size)
        )
        # Plain helpers holding a reference to this decoder, not submodules.
        self._cache_manager = DecoderCacheManager(self)
        self._generation_engine = GenerationEngine(self, self._cache_manager)

    def init_static_cache(
        self,
        batch_size: int,
        *,
        max_cache_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict:
        reference = next(self.parameters())
        cache_device = reference.device if device is None else torch.device(device)
        cache_dtype = reference.dtype if dtype is None else dtype
        capacity = (
            self.config.max_seq_len if max_cache_len is None else int(max_cache_len)
        )
        layer_states: list[DecoderLayerState] = []
        for layer_idx in range(self.config.num_layers):
            attention = self._resolve_layer(layer_idx)._self_attn()
            cache = StaticKVCache(
                batch_size=batch_size,
                num_heads=attention.n_kv_heads,
                max_cache_len=capacity,
                head_dim=attention.head_dim,
                device=cache_device,
                dtype=cache_dtype,
            )
            layer_states.append(
                DecoderLayerState(
                    self_attn=AttentionCacheState(static_cache=cache),
                    cross_attn=AttentionCacheState(),
                )
            )
        return DecoderState(layers=layer_states, cache_implementation="static")

    def _make_layer(self, config: TransformerConfig) -> nn.Module:
        return TransformerDecoderLayer(config)

    def _wants_static_cache(self) -> bool:
        """Static caches are explicit, or implied by ``auto`` under torch.compile."""
        kv_cache = self.config.kv_cache
        if kv_cache == "static":
            return True
        compiler = getattr(torch, "compiler", None)
        return bool(
            kv_cache == "auto"
            and compiler is not None
            and hasattr(compiler, "is_compiling")
            and compiler.is_compiling()
        )

    def _informer_padding_mask(
        self, batch_size: int, length: int, device: torch.device
    ) -> torch.Tensor | None:
        label_len = self.config.label_len
        if not self.config.informer or label_len <= 0 or label_len >= length:
            return None
        mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        mask[:, label_len:] = True
        return mask

    def _infer_mtp_num_heads(self) -> int:
        n = 0
        for i in range(self.config.num_layers):
            layer = self._get_layer(i)
            ff = getattr(layer, "feed_forward", None)
            block = getattr(ff, "block", None) if ff is not None else None
            n_i = int(getattr(block, "mtp_num_heads", 0) or 0)
            n = max(n, n_i)
        return n

    def forward(
        self,
        tgt: torch.Tensor,  # [B, T_tgt, C_tgt]
        memory: torch.Tensor,  # [B, T_src_or_Np, D]
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        incremental_state: dict | DecoderState | None = None,
        return_incremental_state: bool = False,
        time_features: torch.Tensor | None = None,
        mtp_targets: torch.Tensor | None = None,  # [B,T,F] or [B,T,H,D]
        active_mask: torch.Tensor | None = None,  # [B,T] bool
        position_offset: torch.Tensor | int | None = None,
        cache_position: torch.Tensor | None = None,
        cache_update_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ):
        """Decode ``tgt`` against encoder ``memory``.

        With ``return_incremental_state=True`` the KV caches are returned for
        continued decoding (see :meth:`prefill` and :meth:`decode`).
        ``active_mask`` marks tokens eligible for GateSkip gating and
        Mixture-of-Depths routing; it defaults to the non-padded tokens.
        """
        config = self.config
        batch_size, length, _ = tgt.shape
        device = tgt.device
        self._begin_forward()

        if (
            incremental_state is None
            and return_incremental_state
            and self._wants_static_cache()
        ):
            incremental_state = self.init_static_cache(
                batch_size=batch_size, device=device, dtype=tgt.dtype
            )
        prepared_state = prepare_decoder_state(
            incremental_state,
            num_layers=config.num_layers,
            batch_size=batch_size,
            sequence_length=length,
            device=device,
            cache_position=cache_position,
            cache_update_mask=cache_update_mask,
            position_offset=position_offset,
        )
        incremental_state = prepared_state.state
        layer_states = prepared_state.layer_states
        if incremental_state is not None and config.residual in {"mod", "mhc"}:
            raise RuntimeError(
                f"residual={config.residual!r} does not support KV-cached "
                "decoding; decode full sequences instead"
            )
        validate_memory_padding_mask(memory, memory_key_padding_mask)

        x = self.input_adapter(tgt)  # [B, T, D]
        x = self._add_input_positions(x, prepared_state.position_offset)
        if self.time_encoder is not None and time_features is not None:
            time_emb = self.time_encoder(time_features)  # [B, T, D]
            if time_emb.shape[:2] == x.shape[:2]:
                x = x + time_emb
        x = self._input_dropout(x)

        attention_residual_state = self._init_attention_residual_state(x)
        layer_mtp_targets = self._layer_mtp_targets(mtp_targets, batch_size, length)
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(length, device, dtype=x.dtype)
        # All real positions are active by default. Derive this only from the
        # caller's padding, not from the automatic Informer horizon masking.
        if active_mask is None:
            active_mask = gateskip_active_mask_from_padding(tgt_key_padding_mask)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._informer_padding_mask(
                batch_size, length, device
            )

        streams = self._init_streams(x)
        invoke = ModelLayerInvokeStrategy(
            owner=self, use_checkpoint=self._use_layer_checkpointing()
        )
        budget = self._gate_budget()

        def run_layer(index: int, hidden: torch.Tensor) -> LayerResult:
            nonlocal streams
            result = execute_decoder_layer(
                self,
                invoke,
                layer_index=index,
                hidden_states=hidden,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                layer_state=layer_states[index],
                previous_state=layer_states[index - 1] if index > 0 else None,
                budget=budget,
                streams=streams,
                mtp_targets=layer_mtp_targets,
                attention_residual_state=attention_residual_state,
                active_mask=active_mask,
                output_attentions=output_attentions,
            )
            layer_states[index], streams = result.layer_state, result.streams
            return result

        def run_routed(index, hidden, layer, routed_indices, routed_slots):
            nonlocal streams
            x_routed = gather_sequence_tokens(hidden, routed_indices)
            x_routed_out, layer_states[index], streams = invoke.run_decoder_layer(
                layer,
                x_routed,
                memory=memory,
                tgt_mask=gather_square_mask(tgt_mask, routed_indices),
                memory_mask=gather_query_mask(memory_mask, routed_indices),
                tgt_key_padding_mask=gather_padding_mask(
                    tgt_key_padding_mask, routed_indices, routed_slots
                ),
                memory_key_padding_mask=memory_key_padding_mask,
                layer_state=layer_states[index],
                prev_layer_state=layer_states[index - 1] if index > 0 else None,
                gate_budget=budget,
                streams=streams,
                mtp_targets=(
                    gather_sequence_tokens(layer_mtp_targets, routed_indices)
                    if layer_mtp_targets is not None
                    else None
                ),
                attention_residual_state=attention_residual_state,
                active_mask=routed_slots,
            )
            return x_routed, x_routed_out

        trace = StackTrace.start(x, output_hidden_states)
        x = self._run_layer_stack(
            x, active_mask, trace, run_layer=run_layer, run_routed=run_routed
        )
        x = self._finish_stack(x, trace, attention_residual_state)
        out = self.output_projection(x)  # [B, T, output_size]

        if return_incremental_state:
            if incremental_state is None:
                incremental_state = DecoderState.from_mapping(
                    None, num_layers=config.num_layers
                )
            incremental_state["layers"] = layer_states
        if return_dict:
            return build_decoder_output(
                out,
                hidden_states=trace.hidden_states,
                state=incremental_state,
                aux_loss=self.aux_loss,
                router_states=trace.router_states,
                attentions=trace.attentions,
                cross_attentions=trace.cross_attentions,
            )
        if return_incremental_state:
            return out, incremental_state
        return out

    def _layer_mtp_targets(
        self, mtp_targets: torch.Tensor | None, batch_size: int, length: int
    ) -> torch.Tensor | None:
        if not self.training or mtp_targets is None:
            return None
        num_heads = self._infer_mtp_num_heads()
        if num_heads <= 0:
            return None
        return build_decoder_mtp_targets(
            mtp_targets,
            batch_size=batch_size,
            sequence_length=length,
            d_model=self.config.d_model,
            num_heads=num_heads,
            input_adapter=self.input_adapter,
        )

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: dict | None = None,
        time_features: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        cache_update_mask: torch.Tensor | None = None,
    ):
        if tgt.dim() != 3 or tgt.size(1) <= 0:
            raise ValueError(
                f"forward_one_step expects tgt [B,T,C] with T>0, got {tuple(tgt.shape)}"
            )

        incremental_state = DecoderState.coerce(
            incremental_state, num_layers=self.config.num_layers
        )
        decoded_len = incremental_state.decoded_length if incremental_state else 0
        has_kv_cache = decoded_len > 0

        step_tgt = tgt[:, -1:, :] if has_kv_cache else tgt
        step_time_features = (
            time_features[:, -1:, :]
            if (has_kv_cache and time_features is not None)
            else time_features
        )
        has_static_cache = bool(
            incremental_state
            and incremental_state.layers
            and isinstance(
                incremental_state.layers[0].self_attention.cache, StaticKVCache
            )
        )
        if cache_position is None and not has_static_cache:
            cache_position = torch.arange(
                decoded_len,
                decoded_len + step_tgt.size(1),
                device=step_tgt.device,
                dtype=torch.long,
            )
        call_state = incremental_state
        if call_state is None and not self._wants_static_cache():
            call_state = {}
        out, next_state = self.forward(
            step_tgt,
            memory,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            incremental_state=call_state,
            return_incremental_state=True,
            time_features=step_time_features,
            position_offset=decoded_len,
            cache_position=cache_position,
            cache_update_mask=cache_update_mask,
            return_dict=False,
        )
        next_state["_decoded_len"] = decoded_len + step_tgt.size(1)
        return out, next_state

    def prefill(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        **kwargs,
    ):
        kwargs.pop("incremental_state", None)
        kwargs.pop("return_incremental_state", None)
        return self.forward(
            tgt,
            memory,
            incremental_state=None,
            return_incremental_state=True,
            return_dict=False,
            **kwargs,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: dict,
        **kwargs,
    ):
        if incremental_state is None:
            raise ValueError("decode requires an initialized incremental_state")
        if tgt.size(1) > 1:
            return self.forward_multi_step(
                tgt, memory, incremental_state=incremental_state, **kwargs
            )
        return self.forward_one_step(
            tgt,
            memory,
            incremental_state=incremental_state,
            **kwargs,
        )

    def forward_multi_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: dict,
        **kwargs,
    ):
        if tgt.dim() != 3 or tgt.size(1) < 1:
            raise ValueError("forward_multi_step expects tgt [B,T,C] with T>0")
        output, state = self.forward(
            tgt,
            memory,
            incremental_state=incremental_state,
            return_incremental_state=True,
            return_dict=False,
            **kwargs,
        )
        state["_decoded_len"] = int(state.get("_decoded_len", 0)) + tgt.size(1)
        return output, state

    def reorder_incremental_state(
        self, incremental_state: dict, beam_idx: torch.LongTensor
    ) -> dict:
        return self._cache_manager.reorder(incremental_state, beam_idx)

    def cache_state_dict(self, incremental_state: dict) -> dict:
        return self._cache_manager.state_dict(incremental_state)

    def load_cache_state_dict(self, state: dict, *, device=None) -> dict:
        return self._cache_manager.load_state_dict(state, device=device)

    def offload_cache(self, incremental_state: dict) -> dict:
        return self._cache_manager.offload(incremental_state)

    def save_cache(self, incremental_state: dict, path) -> None:
        self._cache_manager.save(incremental_state, path)

    def load_cache(self, path, *, device=None) -> dict:
        return self._cache_manager.load(path, device=device)

    def speculative_decode(
        self,
        draft_tokens: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: dict,
        *,
        verifier_fn=None,
        **kwargs,
    ):
        return self._generation_engine.speculative_decode(
            draft_tokens,
            memory,
            incremental_state,
            verifier_fn=verifier_fn,
            **kwargs,
        )

    def compile_prefill(self, **compile_options):
        return self._generation_engine.compile_prefill(**compile_options)

    def compile_decode(self, **compile_options):
        return self._generation_engine.compile_decode(**compile_options)

    @torch.no_grad()
    def generate(
        self,
        initial_tgt: torch.Tensor,
        memory: torch.Tensor,
        max_new_tokens: int | None = None,
        *,
        generation_config: GenerationConfig | None = None,
        incremental_state: dict | None = None,
        feedback_fn=None,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ):
        return self._generation_engine.generate(
            initial_tgt,
            memory,
            max_new_tokens,
            generation_config=generation_config,
            incremental_state=incremental_state,
            feedback_fn=feedback_fn,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def beam_search(
        self,
        initial_tgt: torch.Tensor,
        memory: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        proposal_fn,
    ):
        return self._generation_engine.beam_search(
            initial_tgt, memory, max_new_tokens, num_beams, proposal_fn
        )


__all__ = ["TransformerDecoder"]
