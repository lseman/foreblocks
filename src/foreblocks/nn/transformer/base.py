"""Shared transformer stack construction and execution.

``BaseTransformer`` owns embeddings, layer construction, Mixture-of-Depths
routing, and the forward steps shared by the encoder and decoder stacks.
Settings are read from ``self.config``; live collaborators are constructor
arguments."""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.nn.embeddings import (
    InformerTimeEmbedding,
    LearnablePositionalEncoding,
    PositionalEncoding,
)
from foreblocks.nn.normalization import RMSNorm, create_norm_layer
from foreblocks.nn.residual.attention_residual import AttentionResidual
from foreblocks.nn.residual.hyper_connections import mhc_init_streams
from foreblocks.nn.routing.gateskip import BudgetScheduler
from foreblocks.nn.routing.mod import (
    LayerDropoutSchedule,
    MoDBudgetScheduler,
    MoDRouter,
    mod_capacity,
    mod_routed_indices,
    mod_router_aux_loss,
    mod_topk_mask,
)
from foreblocks.nn.transformer.config import Role, TransformerConfig
from foreblocks.nn.transformer.runtime.forward import (
    LayerResult,
    positions_from_offset,
)
from foreblocks.nn.transformer.runtime.residual_state import (
    AttentionResidualState,
    attention_residual_values,
    init_attention_residual_state,
)
from foreblocks.nn.transformer.runtime.routing import run_mod_layer


def _build_positional_encoder(config: TransformerConfig) -> nn.Module | None:
    """Input-level position encoding; RoPE and ALiBi live inside attention."""
    if config.position == "learnable":
        return LearnablePositionalEncoding(
            config.d_model,
            max_len=config.max_seq_len,
            dropout=config.dropout,
            scale_strategy="fixed",
            scale_value=config.position_scale,
            use_layer_norm=False,
        )
    if config.position == "sinusoidal":
        return PositionalEncoding(
            config.d_model, max_len=config.max_seq_len, scale=config.position_scale
        )
    return None


@dataclass
class StackTrace:
    """Diagnostics collected while one forward pass runs the layer stack."""

    hidden_states: list[torch.Tensor] | None
    used_indices: list[int] = field(default_factory=list)
    router_states: list[object] = field(default_factory=list)
    attentions: list[torch.Tensor] = field(default_factory=list)
    cross_attentions: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def start(cls, x: torch.Tensor, output_hidden_states: bool) -> StackTrace:
        return cls(hidden_states=[x] if output_hidden_states else None)

    def record(self, index: int, result: LayerResult) -> None:
        self.used_indices.append(index)
        if self.hidden_states is not None:
            self.hidden_states.append(result.hidden_states)
        if result.router_state is not None:
            self.router_states.append(result.router_state)
        if result.self_attention is not None:
            self.attentions.append(result.self_attention)
        if result.cross_attention is not None:
            self.cross_attentions.append(result.cross_attention)


class BaseTransformer(nn.Module, ABC):
    """Common stack behavior for :class:`TransformerEncoder` and decoder.

    Args:
        config: Base settings; defaults to ``TransformerConfig()``.
        pos_encoder: Replaces the input-level position encoding.
        gate_scheduler: Anneals the GateSkip budget during training.
        mod_scheduler: Per-layer Mixture-of-Depths keep rates.
        dropout_schedule: Per-layer dropout rates.
        **overrides: Any ``TransformerConfig`` field, applied over ``config``.
    """

    Config: ClassVar[type[TransformerConfig]] = TransformerConfig
    role: ClassVar[Role]

    def __init__(
        self,
        config: TransformerConfig | None = None,
        *,
        pos_encoder: nn.Module | None = None,
        gate_scheduler: BudgetScheduler | None = None,
        mod_scheduler: MoDBudgetScheduler | None = None,
        dropout_schedule: LayerDropoutSchedule | None = None,
        **overrides: Any,
    ):
        super().__init__()
        config = TransformerConfig.resolve(config, **overrides)
        config.validate_for(self.role)
        self.config = config
        self.gate_scheduler = gate_scheduler
        self.mod_scheduler = mod_scheduler
        self.dropout_schedule = dropout_schedule

        # Variate mixing keeps each input channel as its own token stream, so
        # its shared scalar adapter consumes one channel at a time.
        adapter_input_size = 1 if config.variate_attention else config.input_size
        self.input_adapter = nn.Linear(adapter_input_size, config.d_model)
        self.pos_encoder = (
            pos_encoder
            if pos_encoder is not None
            else _build_positional_encoder(config)
        )
        self.time_encoder = (
            InformerTimeEmbedding(config.d_model) if config.time_encoding else None
        )
        self.register_buffer("_causal_mask", torch.empty(0, 0), persistent=False)

        self.shared_layer, self.layers = self._build_layers()
        self.final_norm = (
            create_norm_layer(config.norm, config.d_model, config.norm_eps)
            if config.final_norm
            else nn.Identity()
        )
        self.mod_routers = (
            nn.ModuleList(
                MoDRouter(d_model=config.d_model, mode="token", hidden=0, init_bias=2.0)
                for _ in range(config.num_layers)
            )
            if config.residual == "mod"
            else None
        )
        self.output_attention_residual = (
            AttentionResidual(config.d_model)
            if config.residual == "attention"
            else None
        )
        self._build_role_modules()

        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        self.mod_aux_loss: float | torch.Tensor = 0.0
        self._layer_aux_losses: list[torch.Tensor] = []
        self._materialize_configured_attention_backends()
        self.apply(self._init_weights)
        self._apply_depth_scaled_initialization()
        self._print_init_summary()

    # ---- Sequence-module interface ------------------------------------------------
    # Composers such as ForecastingModel duck-type any encoder/decoder by these
    # sizes (RNN backbones expose the same names).
    @property
    def input_size(self) -> int:
        return self.config.input_size

    @property
    def output_size(self) -> int:
        return self.config.output_size

    @property
    def d_model(self) -> int:
        return self.config.d_model

    # ---- Construction ------------------------------------------------------------
    @abstractmethod
    def _make_layer(self, config: TransformerConfig) -> nn.Module: ...

    def _build_role_modules(self) -> None:
        """Create encoder/decoder modules; runs before weight initialization."""

    def _build_layers(self) -> tuple[nn.Module | None, nn.ModuleList | None]:
        config = self.config
        schedule = self.dropout_schedule
        dropouts = [
            schedule.get_dropout(index) if schedule is not None else None
            for index in range(config.num_layers)
        ]
        if config.share_layers:
            if any(value != dropouts[0] for value in dropouts[1:]):
                raise ValueError(
                    "share_layers requires a constant layer dropout schedule"
                )
            # One module serves every depth; it materializes each depth's
            # backend and switches between them at run time.
            return self._make_layer(config.for_layer(0, dropout=dropouts[0])), None
        layers = nn.ModuleList(
            self._make_layer(config.for_layer(index, dropout=dropouts[index]))
            for index in range(config.num_layers)
        )
        return None, layers

    def _init_weights(self, m: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def _apply_depth_scaled_initialization(self) -> None:
        config = self.config
        if not config.depth_scaled_init:
            return
        residual_std = config.init_std / math.sqrt(2.0 * config.num_layers)
        residual_suffixes = ("out_proj", "o_proj", "w3", "fc2")
        with torch.no_grad():
            for name, module in self.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                leaf_name = name.rsplit(".", 1)[-1]
                if leaf_name in residual_suffixes or name.endswith("out_proj.2"):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)

    def _print_init_summary(self) -> None:
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        config = self.config
        lines = [
            f"[{type(self).__name__}]",
            f"  shape:     d_model={config.d_model} n_heads={config.n_heads} "
            f"layers={config.num_layers} max_seq_len={config.max_seq_len} "
            f"dropout={config.dropout}",
            f"  attention: {config.attention} pattern={config.attention_pattern} "
            f"(layers={', '.join(self._configured_attention_types())})",
            f"  norm:      {config.norm}/{config.norm_placement} "
            f"final_norm={config.final_norm}",
            f"  residual:  {config.residual}",
        ]
        if config.use_moe:
            lines.append(
                f"  moe:       experts={config.moe_experts} top_k={config.moe_top_k}"
            )
        if self.role == "encoder" and config.patching != "none":
            lines.append(
                f"  patching:  {config.patching} len={config.patch_len} "
                f"stride={config.patch_stride}"
            )
        lines.append(
            f"  runtime:   share_layers={config.share_layers} "
            f"gradient_checkpointing={config.gradient_checkpointing}"
        )
        print("\n".join(lines))

    # ---- Layers ------------------------------------------------------------------
    def _get_layer(self, idx: int) -> nn.Module:
        return self.shared_layer if self.layers is None else self.layers[idx]

    def _resolve_layer(self, layer_idx: int) -> nn.Module:
        layer = self._get_layer(layer_idx)
        if self.shared_layer is not None and hasattr(layer, "set_layer_attention_type"):
            layer.set_layer_attention_type(self.config.layer_attention(layer_idx))
        return layer

    def _configured_attention_types(self) -> list[str]:
        return sorted(
            {self.config.layer_attention(i) for i in range(self.config.num_layers)}
        )

    def _materialize_configured_attention_backends(self) -> None:
        if self.shared_layer is not None:
            materialize = getattr(self.shared_layer, "materialize_attention_type", None)
            if callable(materialize):
                for attention_type in self._configured_attention_types():
                    materialize(attention_type)
            return
        for index in range(self.config.num_layers):
            materialize = getattr(
                self._get_layer(index), "materialize_attention_type", None
            )
            if callable(materialize):
                materialize(self.config.layer_attention(index))

    def _generate_causal_mask(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        size = int(size)
        if size <= 0:
            return torch.empty(0, 0, device=device, dtype=dtype)
        mask = self._causal_mask
        if mask.numel() == 0 or mask.device != device or mask.size(0) < size:
            mask = torch.triu(
                torch.full((size, size), float("-inf"), device=device),
                diagonal=1,
            )
            self._causal_mask = mask
        else:
            mask = mask[:size, :size]
        return mask if mask.dtype == dtype else mask.to(dtype=dtype)

    # ---- Forward steps shared by encoder and decoder ------------------------------
    def _begin_forward(self) -> None:
        self.mod_aux_loss = 0.0
        self._layer_aux_losses.clear()

    def _add_input_positions(
        self, x: torch.Tensor, position_offset: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        """Apply the input-level position encoding, if any.

        A 4D ``[B, V, T, D]`` input (variate mixing) is encoded per variate.
        ``position_offset`` (scalar or ``[B]``) starts positions mid-sequence,
        as in cached decoding.
        """
        if self.pos_encoder is None:
            return x
        if x.ndim == 4:
            batch_size, num_variates, token_length, d_model = x.shape
            flat = x.reshape(batch_size * num_variates, token_length, d_model)
            return self.pos_encoder(flat).reshape(x.shape)
        if position_offset is None:
            return self.pos_encoder(x)
        positions = positions_from_offset(
            position_offset,
            batch_size=x.shape[0],
            length=x.shape[1],
            device=x.device,
        )
        return self.pos_encoder(x, pos=positions)

    def _input_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.config.dropout > 0:
            return F.dropout(x, p=self.config.dropout, training=True)
        return x

    def _use_layer_checkpointing(self) -> bool:
        # The config rejects checkpointing with mHC and attention residuals,
        # whose streams and state are not threaded through checkpoints.
        return self.training and self.config.gradient_checkpointing

    def _init_streams(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.config.residual != "mhc":
            return None
        return mhc_init_streams(x, self.config.mhc_streams)

    def _init_attention_residual_state(
        self, x: torch.Tensor
    ) -> AttentionResidualState | None:
        if self.config.residual != "attention":
            return None
        return init_attention_residual_state(
            x,
            self.config.attention_residual_mode,
            self.config.attention_residual_block_size,
        )

    def _gate_budget(self) -> float | None:
        if self.training and self.gate_scheduler is not None:
            return self.gate_scheduler.get_budget()
        return self.config.gate_budget

    def _run_layer_stack(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor | None,
        trace: StackTrace,
        *,
        run_layer: Callable[[int, torch.Tensor], LayerResult],
        run_routed: Callable[
            [int, torch.Tensor, nn.Module, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> torch.Tensor:
        """Run every layer, through Mixture-of-Depths routing when enabled.

        ``run_layer(index, x)`` executes a layer on the full sequence.
        ``run_routed(index, x, layer, indices, slots)`` gathers the routed
        tokens of ``x``, executes the layer on them, and returns
        ``(routed_input, routed_output)``.
        """
        for index in range(self.config.num_layers):
            if self.config.residual == "mod":
                x, used = run_mod_layer(
                    self,
                    index,
                    x,
                    active_mask,
                    trace.hidden_states,
                    trace.router_states,
                    functools.partial(run_routed, index, x),
                )
                if used:
                    trace.used_indices.append(index)
                continue
            result = run_layer(index, x)
            x = result.hidden_states
            trace.record(index, result)
        return x

    def _finish_stack(
        self,
        x: torch.Tensor,
        trace: StackTrace,
        attention_residual_state: AttentionResidualState | None,
    ) -> torch.Tensor:
        """Aggregate auxiliary losses, step schedulers, and normalize outputs."""
        self._aggregate_aux_loss(trace.used_indices)
        if self.training and self.gate_scheduler is not None:
            self.gate_scheduler.step()
        if self.training and self.mod_scheduler is not None:
            self.mod_scheduler.step()
        if attention_residual_state is not None and self.output_attention_residual:
            x = self.output_attention_residual(
                attention_residual_values(attention_residual_state)
            )
        x = self.final_norm(x)
        if trace.hidden_states is not None:
            trace.hidden_states[-1] = x
        return x

    # ---- Auxiliary losses and routing ------------------------------------------
    def _record_layer_aux_loss(self, loss: torch.Tensor) -> None:
        # Snapshot each invocation before a shared layer is executed again.
        # This runs outside checkpoint closures, so backward cannot append twice.
        self._layer_aux_losses.append(loss)

    def _aggregate_aux_loss(self, used_indices: list[int]) -> None:
        total = self.aux_loss.new_zeros(())
        for loss in self._layer_aux_losses:
            total = total + loss
        self.aux_loss = (
            total / max(len(used_indices), 1) * self.config.moe_aux_weight
            + self.mod_aux_loss
        )
        self._layer_aux_losses.clear()

    @staticmethod
    def _run_with_checkpoint(fn, *inputs: torch.Tensor, use_checkpoint: bool):
        if not use_checkpoint:
            return fn(*inputs)
        return torch.utils.checkpoint.checkpoint(fn, *inputs, use_reentrant=False)

    def _prepare_layer_routing(
        self,
        layer_idx: int,
        x: torch.Tensor,
        active_mask: torch.Tensor | None,
    ) -> tuple[
        nn.Module,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        layer = self._resolve_layer(layer_idx)
        if self.mod_routers is None:
            return layer, None, None, None, None

        router_logits = self.mod_routers[layer_idx](x).squeeze(-1)
        keep_rate = (
            1.0
            if self.mod_scheduler is None
            else float(self.mod_scheduler.get_keep_rate(layer_idx))
        )
        keep_mask = mod_topk_mask(router_logits, keep_rate, active_mask=active_mask)
        weight = self.config.mod_aux_weight
        if self.training and weight > 0:
            self.mod_aux_loss = self.mod_aux_loss + weight * mod_router_aux_loss(
                router_logits, keep_mask, active_mask=active_mask
            )

        capacity = mod_capacity(keep_mask)
        if capacity <= 0:
            return layer, router_logits, keep_mask, None, None
        indices, slot_mask = mod_routed_indices(keep_mask, capacity=capacity)
        return layer, router_logits, keep_mask, indices, slot_mask


__all__ = ["BaseTransformer", "StackTrace"]
