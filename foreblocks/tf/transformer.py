from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import InformerTimeEmbedding, PositionalEncoding
from .transformer_att import MultiAttention
from .transformer_aux import *
from .transformer_moe import *


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_len=5000):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_size, max_len=max_len
        )

    def forward(self, x):
        # x: [B, T, input_size]
        x = self.input_projection(x)  # [B, T, hidden_size]
        x = self.positional_encoding(x)  # [B, T, hidden_size]
        return x


class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        dim_feedforward: int = 2048,
        use_swiglu: bool = True,
        norm_strategy: str = "pre_norm",
        use_adaptive_ln: str = "layer",
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dropout_p = dropout
        self._is_pre_norm = norm_strategy == "pre_norm"
        self._needs_dropout = dropout > 0.0
        self.use_moe = use_moe

        # Feedforward block (can be MoE or standard)
        self.feed_forward = FeedForwardBlock(
            d_model=d_model,
            dim_ff=dim_feedforward,
            dropout=dropout,
            use_swiglu=use_swiglu,
            activation=activation,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=moe_capacity_factor,
            expert_dropout=dropout,
        )

        # Normalization layers (created dynamically in subclasses)
        self.norm_layers = nn.ModuleList()

    def create_norm(self, d_model: int, use_adaptive_ln: str, eps: float) -> nn.Module:
        return create_norm_layer(use_adaptive_ln, d_model, eps)

    def apply_dropout_residual(
        self, x: torch.Tensor, residual: torch.Tensor, training: bool
    ):
        if self._needs_dropout and training:
            x = F.dropout(x, p=self.dropout_p, training=True)
        return residual + x

    # Pre-norm (correct): LayerNorm(x) → FF → Dropout → Residual
    # Post-norm (incorrect): x → FF → Dropout → Residual → LayerNorm
    def forward_feedforward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Optimized feedforward with reduced memory allocations"""
        residual = x
        x_normed = self.norm_layers[-1](x) if self._is_pre_norm else x

        if self.use_moe:
            ff_out, aux_loss = self.feed_forward(x_normed, return_aux_loss=True)
            self.aux_loss = aux_loss
        else:
            ff_out = self.feed_forward(x_normed)
            # Avoid repeated assignment to 0.0
            if self.aux_loss != 0.0:
                self.aux_loss = 0.0

        if self._is_pre_norm:
            return self.apply_dropout_residual(ff_out, residual, training)
        else:
            # Post-norm: apply dropout + residual, then normalization
            ff_with_residual = self.apply_dropout_residual(ff_out, residual, training)
            return self.norm_layers[-1](ff_with_residual)


class TransformerEncoderLayer(BaseTransformerLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        freq_modes: int = 16,
        use_swiglu: bool = True,
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_adaptive_ln: str = "layer",
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
    ):
        super().__init__(
            d_model,
            dropout,
            activation,
            dim_feedforward,
            use_swiglu,
            norm_strategy,
            use_adaptive_ln,
            use_moe,
            num_experts,
            top_k,
            moe_capacity_factor,
            layer_norm_eps,
        )

        self.self_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
        )

        # Create norm1 (attention) and norm2 (ffn)
        self.norm_layers.append(
            self.create_norm(d_model, use_adaptive_ln, layer_norm_eps)
        )
        self.norm_layers.append(
            self.create_norm(d_model, use_adaptive_ln, layer_norm_eps)
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        training = self.training
        residual = src

        if self._is_pre_norm:
            src_normed = self.norm_layers[0](src)
            attn_out, _, _ = self.self_attn(
                src_normed,
                src_normed,
                src_normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = self.apply_dropout_residual(attn_out, residual, training)
            src = self.forward_feedforward(src, training)
        else:
            attn_out, _, _ = self.self_attn(
                src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )
            # Use helper method for consistency
            src_with_residual = self.apply_dropout_residual(
                attn_out, residual, training
            )
            src = self.norm_layers[0](src_with_residual)
            src = self.forward_feedforward(src, training)

        return src


class TransformerDecoderLayer(BaseTransformerLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        use_swiglu: bool = True,
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_adaptive_ln: str = "layer",
        informer_like: bool = False,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
    ):
        super().__init__(
            d_model,
            dropout,
            activation,
            dim_feedforward,
            use_swiglu,
            norm_strategy,
            use_adaptive_ln,
            use_moe,
            num_experts,
            top_k,
            moe_capacity_factor,
            layer_norm_eps,
        )

        self.self_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            cross_attention=False,
        )
        self.cross_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            cross_attention=True,
        )
        self._is_causal = not informer_like

        self.norm_layers.extend(
            [
                self.create_norm(d_model, use_adaptive_ln, layer_norm_eps),  # norm1
                self.create_norm(d_model, use_adaptive_ln, layer_norm_eps),  # norm2
                self.create_norm(d_model, use_adaptive_ln, layer_norm_eps),  # norm3
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        training = self.training
        self_attn_state = (
            None if incremental_state is None else incremental_state.get("self_attn")
        )
        cross_attn_state = (
            None if incremental_state is None else incremental_state.get("cross_attn")
        )

        if self._is_pre_norm:
            # Self-attention
            residual = tgt
            tgt_normed = self.norm_layers[0](tgt)
            self_attn_out, _, updated_self_state = self.self_attn(
                tgt_normed,
                tgt_normed,
                tgt_normed,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=self._is_causal,
                layer_state=self_attn_state,
            )
            tgt = self.apply_dropout_residual(self_attn_out, residual, training)

            # Cross-attention
            residual = tgt
            tgt_normed = self.norm_layers[1](tgt)
            cross_attn_out, _, updated_cross_state = self.cross_attn(
                tgt_normed,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                layer_state=cross_attn_state,
            )
            tgt = self.apply_dropout_residual(cross_attn_out, residual, training)
            tgt = self.forward_feedforward(tgt, training)
        else:
            # Self-attention
            self_attn_out, _, updated_self_state = self.self_attn(
                tgt,
                tgt,
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=self._is_causal,
                layer_state=self_attn_state,
            )
            # Use helper method for consistency
            tgt_with_residual = self.apply_dropout_residual(
                self_attn_out, tgt, training
            )
            tgt = self.norm_layers[0](tgt_with_residual)

            # Cross-attention
            cross_attn_out, _, updated_cross_state = self.cross_attn(
                tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                layer_state=cross_attn_state,
            )
            # Use helper method for consistency
            tgt_with_residual = self.apply_dropout_residual(
                cross_attn_out, tgt, training
            )
            tgt = self.norm_layers[1](tgt_with_residual)

            tgt = self.forward_feedforward(tgt, training)

        # Update state
        if incremental_state is not None:
            incremental_state["self_attn"] = updated_self_state
            incremental_state["cross_attn"] = updated_cross_state
            return tgt, incremental_state
        return tgt, None


class BaseTransformer(nn.Module, ABC):
    """Base class for Transformer encoder and decoder with shared functionality."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_adaptive_ln: str = "layer",
        max_seq_len: int = 5000,
        pos_encoding_scale: float = 1.0,
        pos_encoder: Optional[nn.Module] = None,
        use_gradient_checkpointing: bool = False,
        share_layers: bool = False,
        use_final_norm: bool = True,
        use_swiglu: bool = True,
        freq_modes: int = 32,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        **kwargs  # For subclass-specific parameters
    ):
        super().__init__()

        # Store common parameters
        self.d_model = d_model
        self.num_layers = num_layers
        self._use_gradient_checkpointing = use_gradient_checkpointing
        self._use_final_norm = use_final_norm
        self._needs_dropout = dropout > 0.0
        self.dropout_p = dropout
        self.aux_loss = 0.0

        # Common layer arguments
        self.layer_args = (
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            att_type,
            layer_norm_eps,
            norm_strategy,
            use_adaptive_ln,
            use_swiglu,
            freq_modes,
            use_moe,
            num_experts,
            top_k,
            moe_capacity_factor,
        )

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = pos_encoder or PositionalEncoding(
            d_model, dropout, max_seq_len, pos_encoding_scale
        )

        # Layer creation (shared or individual)
        if share_layers:
            self.shared_layer = self._make_layer(*self.layer_args, **kwargs)
            self.layers = None
        else:
            self.layers = nn.ModuleList(
                [
                    self._make_layer(*self.layer_args, **kwargs)
                    for _ in range(num_layers)
                ]
            )
            self.shared_layer = None

        # Final normalization
        self.final_norm = (
            create_norm_layer(use_adaptive_ln, d_model, layer_norm_eps)
            if use_final_norm
            else nn.Identity()
        )
        self._final_norm_is_identity = isinstance(self.final_norm, nn.Identity)

        # Initialize weights
        self.apply(self._init_weights)

    @abstractmethod
    def _make_layer(self, *args, **kwargs):
        """Create a transformer layer. Must be implemented by subclasses."""
        pass

    def _init_weights(self, module: nn.Module):
        """Optimized and extensible Transformer weight initialization."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "padding_idx") and m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()

    def _apply_input_processing(self, x, additional_features=None):
        """Apply input projection, positional encoding, and optional features."""
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        if additional_features is not None:
            x += additional_features

        if self.training and self._needs_dropout:
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x

    def _apply_final_norm(self, x):
        """Apply final normalization if needed."""
        if not self._final_norm_is_identity:
            x = self.final_norm(x)
        return x

    def _get_layer(self, layer_idx):
        """Get layer by index (shared or individual)."""
        return self.shared_layer if self.layers is None else self.layers[layer_idx]

    def get_aux_loss(self):
        """Get auxiliary loss from MoE layers."""
        return getattr(self, "aux_loss", 0.0)


class TransformerEncoder(BaseTransformer):
    """Transformer Encoder with shared base functionality."""

    def __init__(self, input_size: int, **kwargs):
        # Extract encoder-specific parameters
        self.freq_modes = kwargs.get("freq_modes", 32)

        super().__init__(input_size, **kwargs)

        # Encoder-specific components
        self.time_encoder = InformerTimeEmbedding(self.d_model)

    def _make_layer(self, *args, **kwargs):
        """Create transformer encoder layer."""
        (
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            att_type,
            layer_norm_eps,
            norm_strategy,
            use_adaptive_ln,
            use_swiglu,
            freq_modes,
            use_moe,
            num_experts,
            top_k,
            moe_capacity_factor,
        ) = args

        return TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type=att_type,
            freq_modes=freq_modes,
            use_swiglu=use_swiglu,
            layer_norm_eps=layer_norm_eps,
            norm_strategy=norm_strategy,
            use_adaptive_ln=use_adaptive_ln,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_capacity_factor=moe_capacity_factor,
        )

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, time_features=None
    ):
        # Process input with time features
        time_emb = (
            self.time_encoder(time_features) if time_features is not None else None
        )
        src = self._apply_input_processing(src, time_emb)

        # Pass through layers
        aux_loss = 0.0
        for i in range(self.num_layers):
            layer = self._get_layer(i)

            if self.training and self._use_gradient_checkpointing:
                src = torch.utils.checkpoint.checkpoint(
                    layer, src, src_mask, src_key_padding_mask, use_reentrant=False
                )
            else:
                src = layer(src, src_mask, src_key_padding_mask)

            if hasattr(layer, "aux_loss"):
                aux_loss += layer.aux_loss

        # Apply final normalization
        src = self._apply_final_norm(src)

        self.aux_loss = aux_loss
        return src


class TransformerDecoder(BaseTransformer):
    """Transformer Decoder with shared base functionality."""

    def __init__(
        self, input_size: int, output_size: int, informer_like: bool = False, **kwargs
    ):
        self.output_size = output_size
        self.informer_like = informer_like

        super().__init__(input_size, informer_like=informer_like, **kwargs)

        # Decoder-specific output projection
        self.output_projection = (
            nn.Linear(self.d_model, output_size)
            if output_size != self.d_model
            else nn.Identity()
        )
        self._output_proj_is_identity = isinstance(self.output_projection, nn.Identity)

    def _make_layer(self, *args, **kwargs):
        """Create transformer decoder layer."""
        (
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            att_type,
            layer_norm_eps,
            norm_strategy,
            use_adaptive_ln,
            use_swiglu,
            freq_modes,
            use_moe,
            num_experts,
            top_k,
            moe_capacity_factor,
        ) = args

        informer_like = kwargs.get("informer_like", False)

        return TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type=att_type,
            layer_norm_eps=layer_norm_eps,
            norm_strategy=norm_strategy,
            use_adaptive_ln=use_adaptive_ln,
            informer_like=informer_like,
            use_swiglu=use_swiglu,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_capacity_factor=moe_capacity_factor,
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        incremental_state: Optional[Dict] = None,
        return_incremental_state: bool = False,
    ):
        # Process input
        tgt = self._apply_input_processing(tgt)

        # Handle incremental state for inference
        layer_states = (
            incremental_state.get("layers", [None] * self.num_layers)
            if incremental_state
            else [None] * self.num_layers
        )

        # Pass through layers
        aux_loss = 0.0
        for i in range(self.num_layers):
            layer = self._get_layer(i)

            if self.training and self._use_gradient_checkpointing:
                tgt = torch.utils.checkpoint.checkpoint(
                    layer,
                    tgt,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    use_reentrant=False,
                )
                layer_states[i] = None
            else:
                tgt, layer_states[i] = layer(
                    tgt,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    layer_states[i],
                )

            if hasattr(layer, "aux_loss"):
                aux_loss += layer.aux_loss

        # Apply final normalization
        tgt = self._apply_final_norm(tgt)

        self.aux_loss = aux_loss

        # Update incremental state
        if incremental_state is not None:
            incremental_state["layers"] = layer_states

        # Apply output projection
        if not self._output_proj_is_identity:
            tgt = self.output_projection(tgt)

        return (tgt, incremental_state) if return_incremental_state else tgt

    def forward_one_step(self, tgt, memory, incremental_state=None):
        """Forward pass for one step during inference."""
        incremental_state = incremental_state or {}
        return self.forward(
            tgt,
            memory,
            incremental_state=incremental_state,
            return_incremental_state=True,
        )
