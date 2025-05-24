import math

from typing import List, Optional, Tuple
from typing import Dict, Any

import torch
import torch.nn as nn


from .embeddings import PositionalEncoding
from .transformer_att import MultiAttention

from .transformer_aux import FeedForwardBlock


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


def init_transformer_weights(module, d_model: Optional[int] = None):
    """Custom initialization for transformer weights with optional d_model scaling"""

    if isinstance(module, nn.Linear):
        std = 0.02 if d_model is None else 1.0 / math.sqrt(d_model)
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "padding_idx") and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
        torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif hasattr(module, "w1") and hasattr(module, "w2") and hasattr(module, "w3"):
        # Likely a SwiGLU block
        for w in [module.w1, module.w2, module.w3]:
            torch.nn.init.normal_(w.weight, mean=0.0, std=0.01)
            if w.bias is not None:
                torch.nn.init.zeros_(w.bias)


class AdaptiveRMSNorm(nn.Module):
    """Adaptive RMSNorm with learnable scaling and optional bias"""

    def __init__(self, d_model, eps=1e-5, use_bias=False):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))  # Learnable scale
        self.use_bias = use_bias
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(d_model))
        print("[Normalization] Adaptive RMSNorm")

    def forward(self, x):
        # RMS norm: no mean subtraction
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        out = self.alpha * (x / rms)
        if self.use_bias:
            out += self.beta
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(d_model={self.alpha.shape[0]}, eps={self.eps})"
        )


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm with learnable scaling and optional bias"""

    def __init__(self, d_model, eps=1e-5, use_bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.alpha = nn.Parameter(torch.ones(d_model))  # Per-dim scaling
        self.use_bias = use_bias
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(d_model))
        print("[Normalization] Adaptive LayerNorm")

    def forward(self, x):
        out = self.alpha * self.norm(x)
        if self.use_bias:
            out += self.beta
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(normalized_shape={self.norm.normalized_shape}, eps={self.norm.eps})"


class TransformerEncoderLayer(nn.Module):
    """
    Optimized Transformer Encoder Layer with:
    - Simplified initialization with smart defaults
    - Unified forward pass logic
    - Removed redundant parameters and checks
    - Better error handling
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",  # Simplified from att_type/freq_att
        freq_modes: int = 16,
        use_swiglu: bool = True,
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",  # pre_norm or post_norm
        use_adaptive_ln: str = "layer",  # layer, rms, adaptive_layer, adaptive_rms
        # MoE parameters
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
    ):
        super().__init__()

        self.norm_strategy = norm_strategy
        self.use_moe = use_moe

        # Self-attention with unified parameters
        self.self_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
        )

        # Feed-forward network
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
            expert_dropout=dropout,  # Use same dropout
        )

        # Normalization layers
        self.norm1 = self._create_norm_layer(use_adaptive_ln, d_model, layer_norm_eps)
        self.norm2 = self._create_norm_layer(use_adaptive_ln, d_model, layer_norm_eps)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # MoE auxiliary loss
        self.aux_loss = 0.0

    def _create_norm_layer(self, norm_type: str, d_model: int, eps: float) -> nn.Module:
        """Create appropriate normalization layer"""
        if norm_type == "layer":
            return nn.LayerNorm(d_model, eps=eps)
        elif norm_type == "rms":
            return AdaptiveRMSNorm(d_model, eps=eps)
        elif norm_type == "adaptive_layer":
            return AdaptiveLayerNorm(d_model, eps=eps)
        elif norm_type == "adaptive_rms":
            return AdaptiveRMSNorm(d_model, eps=eps)
        else:
            return nn.LayerNorm(d_model, eps=eps)  # Default fallback

    def _apply_attention(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply self-attention with error handling"""
        attn_out, _, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        return self.dropout1(attn_out)

    def _apply_feedforward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Apply feed-forward with MoE support"""
        if self.use_moe:
            ff_out, aux_loss = self.feed_forward(x, return_aux_loss=True)
            return self.dropout2(ff_out), aux_loss
        else:
            ff_out = self.feed_forward(x)
            return self.dropout2(ff_out), 0.0

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unified forward pass handling both pre-norm and post-norm strategies
        """
        self.aux_loss = 0.0  # Reset auxiliary loss

        if self.norm_strategy == "pre_norm":
            # Pre-norm: Norm -> SubLayer -> Add
            # Self-attention sublayer
            attn_out = self._apply_attention(
                self.norm1(src), src_mask, src_key_padding_mask
            )
            src = src + attn_out

            # Feed-forward sublayer
            ff_out, self.aux_loss = self._apply_feedforward(self.norm2(src))
            src = src + ff_out

        else:  # post_norm
            # Post-norm: SubLayer -> Add -> Norm
            # Self-attention sublayer
            attn_out = self._apply_attention(src, src_mask, src_key_padding_mask)
            src = self.norm1(src + attn_out)

            # Feed-forward sublayer
            ff_out, self.aux_loss = self._apply_feedforward(src)
            src = self.norm2(src + ff_out)

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Optimized Transformer Decoder Layer with:
    - Unified forward pass logic
    - Cleaner parameter interface
    - Better incremental state handling
    - Removed redundant NaN checks
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",  # Unified attention type
        use_swiglu: bool = True,
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",  # pre_norm or post_norm
        use_adaptive_ln: str = "layer",  # layer, rms, adaptive_layer, adaptive_rms
        informer_like: bool = False,  # Controls causal masking
        # MoE parameters
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
    ):
        super().__init__()

        self.norm_strategy = norm_strategy
        self.use_moe = use_moe
        self.informer_like = informer_like

        # Self-attention (decoder self-attention)
        self.self_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            cross_attention=False,
        )

        # Cross-attention (decoder-encoder attention)
        self.cross_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            cross_attention=True,
        )

        # Feed-forward network
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

        # Normalization layers
        self.norm1 = self._create_norm_layer(use_adaptive_ln, d_model, layer_norm_eps)
        self.norm2 = self._create_norm_layer(use_adaptive_ln, d_model, layer_norm_eps)
        self.norm3 = self._create_norm_layer(use_adaptive_ln, d_model, layer_norm_eps)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # MoE auxiliary loss
        self.aux_loss = 0.0

    def _create_norm_layer(self, norm_type: str, d_model: int, eps: float) -> nn.Module:
        """Create appropriate normalization layer"""
        if norm_type == "layer":
            return nn.LayerNorm(d_model, eps=eps)
        elif norm_type == "rms":
            return AdaptiveRMSNorm(d_model, eps=eps)
        elif norm_type == "adaptive_layer":
            return AdaptiveLayerNorm(d_model, eps=eps)
        elif norm_type == "adaptive_rms":
            return AdaptiveRMSNorm(d_model, eps=eps)
        else:
            return nn.LayerNorm(d_model, eps=eps)

    def _apply_self_attention(
        self,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply self-attention with incremental state support"""
        is_causal = not self.informer_like

        self_attn_state = (
            None if incremental_state is None else incremental_state.get("self_attn")
        )

        attn_out, _, updated_state = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=is_causal,
            layer_state=self_attn_state,
        )

        return self.dropout1(attn_out), updated_state

    def _apply_cross_attention(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply cross-attention with incremental state support"""
        cross_attn_state = (
            None if incremental_state is None else incremental_state.get("cross_attn")
        )

        attn_out, _, updated_state = self.cross_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            layer_state=cross_attn_state,
        )

        return self.dropout2(attn_out), updated_state

    def _apply_feedforward(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Apply feed-forward with MoE support"""
        if self.use_moe:
            ff_out, aux_loss = self.feed_forward(tgt, return_aux_loss=True)
            return self.dropout3(ff_out), aux_loss
        else:
            ff_out = self.feed_forward(tgt)
            return self.dropout3(ff_out), 0.0

    def _update_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Any]],
        self_attn_state: Optional[Dict[str, Any]],
        cross_attn_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Update incremental state with new attention states"""
        if incremental_state is not None:
            if self_attn_state is not None:
                incremental_state["self_attn"] = self_attn_state
            if cross_attn_state is not None:
                incremental_state["cross_attn"] = cross_attn_state
        return incremental_state

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Unified forward pass handling both pre-norm and post-norm strategies
        """
        self.aux_loss = 0.0  # Reset auxiliary loss

        if self.norm_strategy == "pre_norm":
            # Pre-norm: Norm -> SubLayer -> Add

            # Self-attention sublayer
            self_attn_out, updated_self_state = self._apply_self_attention(
                self.norm1(tgt), tgt_mask, tgt_key_padding_mask, incremental_state
            )
            tgt = tgt + self_attn_out

            # Cross-attention sublayer
            cross_attn_out, updated_cross_state = self._apply_cross_attention(
                self.norm2(tgt),
                memory,
                memory_mask,
                memory_key_padding_mask,
                incremental_state,
            )
            tgt = tgt + cross_attn_out

            # Feed-forward sublayer
            ff_out, self.aux_loss = self._apply_feedforward(self.norm3(tgt))
            tgt = tgt + ff_out

        else:  # post_norm
            # Post-norm: SubLayer -> Add -> Norm

            # Self-attention sublayer
            self_attn_out, updated_self_state = self._apply_self_attention(
                tgt, tgt_mask, tgt_key_padding_mask, incremental_state
            )
            tgt = self.norm1(tgt + self_attn_out)

            # Cross-attention sublayer
            cross_attn_out, updated_cross_state = self._apply_cross_attention(
                tgt, memory, memory_mask, memory_key_padding_mask, incremental_state
            )
            tgt = self.norm2(tgt + cross_attn_out)

            # Feed-forward sublayer
            ff_out, self.aux_loss = self._apply_feedforward(tgt)
            tgt = self.norm3(tgt + ff_out)

        # Update incremental state
        updated_incremental_state = self._update_incremental_state(
            incremental_state, updated_self_state, updated_cross_state
        )

        return tgt, updated_incremental_state


class TransformerEncoder(nn.Module):
    """
    Optimized Transformer Encoder with:
    - Simplified parameter interface
    - Better memory management
    - Unified layer creation
    - Optional final normalization
    - Cleaner architecture
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",  # Unified attention parameter
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_adaptive_ln: str = "layer",  # layer, rms, adaptive_layer, adaptive_rms
        # Positional encoding
        max_seq_len: int = 5000,
        pos_encoding_scale: float = 1.0,
        pos_encoder: Optional[nn.Module] = None,
        # Optimization options
        use_gradient_checkpointing: bool = False,
        share_layers: bool = False,  # Memory optimization
        use_final_norm: bool = True,  # Final layer norm
        # Advanced options
        use_swiglu: bool = True,
        freq_modes: int = 32,  # For frequency attention
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
    ):
        super().__init__()

        # Core attributes
        self.d_model = d_model
        self.input_size = input_size
        self.num_layers = num_layers
        self.norm_strategy = norm_strategy
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_final_norm = use_final_norm
        self.is_transformer = True

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        if pos_encoder is not None:
            self.pos_encoder = pos_encoder
        else:
            self.pos_encoder = PositionalEncoding(
                d_model, dropout=dropout, max_len=max_seq_len, scale=pos_encoding_scale
            )

        # Input dropout
        self.dropout = nn.Dropout(dropout)

        # Create transformer layers
        if share_layers:
            # Single shared layer (memory efficient)
            self.shared_layer = self._create_encoder_layer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                att_type,
                layer_norm_eps,
                use_adaptive_ln,
                norm_strategy,
                use_swiglu,
                freq_modes,
                use_moe,
                num_experts,
                top_k,
                moe_capacity_factor,
            )
            self.layers = None
        else:
            # Individual layers
            self.layers = nn.ModuleList(
                [
                    self._create_encoder_layer(
                        d_model,
                        nhead,
                        dim_feedforward,
                        dropout,
                        activation,
                        att_type,
                        layer_norm_eps,
                        use_adaptive_ln,
                        norm_strategy,
                        use_swiglu,
                        freq_modes,
                        use_moe,
                        num_experts,
                        top_k,
                        moe_capacity_factor,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.shared_layer = None

        # Final normalization (especially useful for pre-norm)
        if use_final_norm:
            self.final_norm = self._create_norm_layer(
                use_adaptive_ln, d_model, layer_norm_eps
            )
        else:
            self.final_norm = nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

        print(
            f"[Encoder] Transformer Encoder: {num_layers} layers, "
            f"{att_type} attention, {norm_strategy} norm"
        )

    def _create_encoder_layer(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        attention_type,
        layer_norm_eps,
        use_adaptive_ln,
        norm_strategy,
        use_swiglu,
        freq_modes,
        use_moe,
        num_experts,
        top_k,
        moe_capacity_factor,
    ):
        """Create a single encoder layer with all configurations"""
        return TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type=attention_type,
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

    def _create_norm_layer(self, norm_type: str, d_model: int, eps: float) -> nn.Module:
        """Create appropriate normalization layer"""
        if norm_type == "layer":
            return nn.LayerNorm(d_model, eps=eps)
        elif norm_type == "rms":
            return AdaptiveRMSNorm(d_model, eps=eps)
        elif norm_type == "adaptive_layer":
            return AdaptiveLayerNorm(d_model, eps=eps)
        elif norm_type == "adaptive_rms":
            return AdaptiveRMSNorm(d_model, eps=eps)
        else:
            return nn.LayerNorm(d_model, eps=eps)

    def _init_weights(self, module):
        """Initialize transformer weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer encoder.

        Args:
            src: Source sequence [batch_size, seq_len, input_size]
            src_mask: Optional attention mask [seq_len, seq_len]
            src_key_padding_mask: Optional padding mask [batch_size, seq_len]

        Returns:
            Encoded output [batch_size, seq_len, d_model]
        """
        # Project input to model dimension
        src = self.input_projection(src)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Apply input dropout
        src = self.dropout(src)

        # Apply transformer layers
        total_aux_loss = 0.0

        for i in range(self.num_layers):
            # Get layer (shared or individual)
            layer = self.shared_layer if self.layers is None else self.layers[i]

            if self.training and self.use_gradient_checkpointing:
                # Gradient checkpointing for memory efficiency
                src = torch.utils.checkpoint.checkpoint(
                    layer, src, src_mask, src_key_padding_mask, use_reentrant=False
                )
            else:
                src = layer(src, src_mask, src_key_padding_mask)

            # Accumulate auxiliary losses (for MoE)
            if hasattr(layer, "aux_loss"):
                total_aux_loss += layer.aux_loss

        # Apply final normalization
        src = self.final_norm(src)

        # Store total auxiliary loss
        self.aux_loss = total_aux_loss

        return src

    def get_aux_loss(self) -> float:
        """Get total auxiliary loss from all layers"""
        return getattr(self, "aux_loss", 0.0)


class TransformerDecoder(nn.Module):
    """
    Optimized Transformer Decoder matching the encoder structure
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
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
        # Positional encoding
        max_seq_len: int = 5000,
        pos_encoding_scale: float = 1.0,
        pos_encoder: Optional[nn.Module] = None,
        # Optimization options
        use_gradient_checkpointing: bool = False,
        share_layers: bool = False,
        use_final_norm: bool = True,
        informer_like: bool = False,
        # Advanced options
        use_swiglu: bool = True,
        freq_modes: int = 32,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
    ):
        super().__init__()

        self.d_model = d_model
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_final_norm = use_final_norm

        self.input_projection = nn.Linear(input_size, d_model)

        # Output projection (if needed)
        if output_size != d_model:
            self.output_projection = nn.Linear(d_model, output_size)
        else:
            self.output_projection = nn.Identity()

        # Positional encoding
        if pos_encoder is not None:
            self.pos_encoder = pos_encoder
        else:
            self.pos_encoder = PositionalEncoding(
                d_model, dropout=dropout, max_len=max_seq_len, scale=pos_encoding_scale
            )

        self.dropout = nn.Dropout(dropout)

        # Create decoder layers
        if share_layers:
            self.shared_layer = TransformerDecoderLayer(
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
            self.layers = None
        else:
            self.layers = nn.ModuleList(
                [
                    TransformerDecoderLayer(
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
                    for _ in range(num_layers)
                ]
            )
            self.shared_layer = None

        # Final normalization
        if use_final_norm:
            self.final_norm = self._create_norm_layer(
                use_adaptive_ln, d_model, layer_norm_eps
            )
        else:
            self.final_norm = nn.Identity()

        self.apply(self._init_weights)

        print(
            f"[Decoder] Transformer Decoder: {num_layers} layers, "
            f"{att_type} attention, {'Informer-like' if informer_like else 'Standard'}, {norm_strategy} norm"
        )

    def _create_norm_layer(self, norm_type: str, d_model: int, eps: float) -> nn.Module:
        """Create appropriate normalization layer"""
        if norm_type == "layer":
            return nn.LayerNorm(d_model, eps=eps)
        elif norm_type == "rms":
            return AdaptiveRMSNorm(d_model, eps=eps)
        elif norm_type == "adaptive_layer":
            return AdaptiveLayerNorm(d_model, eps=eps)
        elif norm_type == "adaptive_rms":
            return AdaptiveRMSNorm(d_model, eps=eps)
        else:
            return nn.LayerNorm(d_model, eps=eps)

    def _init_weights(self, module):
        """Initialize transformer weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[dict] = None,
        return_incremental_state: bool = False,
    ):
        """Forward pass with optional incremental decoding"""

        # Project output to model dimension if needed
        tgt = self.input_projection(tgt)

        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)

        # Initialize incremental state
        if incremental_state is None:
            layer_states = [None] * self.num_layers
        else:
            layer_states = incremental_state.get("layers", [None] * self.num_layers)

        total_aux_loss = 0.0

        # Apply decoder layers
        for i in range(self.num_layers):
            layer = self.shared_layer if self.layers is None else self.layers[i]

            if self.training and self.use_gradient_checkpointing:
                # Note: Checkpointing with incremental state is complex
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
                total_aux_loss += layer.aux_loss

        # Apply final normalization
        tgt = self.final_norm(tgt)

        self.aux_loss = total_aux_loss

        # Update incremental state
        if incremental_state is not None:
            incremental_state["layers"] = layer_states

        tgt = self.output_projection(tgt)

        if return_incremental_state:
            return tgt, incremental_state
        else:
            return tgt

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: Optional[dict] = None,
    ):
        """Single step forward for autoregressive generation"""
        if incremental_state is None:
            incremental_state = {}

        return self.forward(tgt, memory, incremental_state=incremental_state, return_incremental_state=True)
