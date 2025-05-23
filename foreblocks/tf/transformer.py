import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .embeddings import PositionalEncoding, RotaryEmbedding
from .transformer_att import XFormerAttention

from .transformer_aux import MoEFeedForward, FeedForwardBlock
from .fed import FrequencyAttention


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
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        att_type="prob_sparse",
        use_swiglu=True,
        use_flash_attn=True,
        layer_norm_eps=1e-5,
        norm_strategy="pre_norm",
        freq_att=False,
        freq_modes=16,
        seq_len=None,
        use_adaptive_ln="rms",
        use_moe=False,
        num_experts=5,
        top_k=5,
        moe_capacity_factor=0.1,
    ):
        super().__init__()

        self.norm_strategy = norm_strategy
        self.use_moe = use_moe

        self.self_attn = XFormerAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            use_flash_attn=use_flash_attn,
            attention_type="frequency" if freq_att else att_type,
            freq_modes=freq_modes,
        )

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
            expert_dropout=0.1,
        )

        norm_cls = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if use_adaptive_ln == "rms":
            norm_cls = AdaptiveRMSNorm
        elif use_adaptive_ln == "layer":
            norm_cls = AdaptiveLayerNorm
        self.norm1 = norm_cls(d_model, eps=layer_norm_eps)
        self.norm2 = norm_cls(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.aux_loss = 0.0

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_strategy == "pre_norm":
            # === Pre-norm Self-Attention ===
            normed_src = self.norm1(src)
            src2, _, _ = self.self_attn(
                normed_src,
                normed_src,
                normed_src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self.dropout1(src2)

            # === Pre-norm Feedforward ===
            normed_src = self.norm2(src)

            # check for nan in normed_src
            if torch.isnan(normed_src).any():
                raise RuntimeError("NaN detected in normed_src before feedforward")
            if self.use_moe:
                src2, self.aux_loss = self.feed_forward(
                    normed_src, return_aux_loss=True
                )
            else:
                src2 = self.feed_forward(normed_src)
            # check for nan in src2
            if torch.isnan(src2).any():
                raise RuntimeError("NaN detected in src2 after feedforward")
            src = src + self.dropout2(src2)
            return src

        else:
            # === Post-norm Self-Attention ===
            src2, _, _ = self.self_attn(
                src,
                src,
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = self.norm1(src + self.dropout1(src2))

            # === Post-norm Feedforward ===
            if self.use_moe:
                src2, self.aux_loss = self.feed_forward(src, return_aux_loss=True)
            else:
                src2 = self.feed_forward(src)
            src = self.norm2(src + self.dropout2(src2))
            return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        informer_like=False,
        att_type="prob_sparse",
        use_swiglu=True,
        use_flash_attn=True,
        layer_norm_eps=1e-5,
        norm_strategy="pre_norm",
        use_adaptive_ln="rms",
        use_moe=False,
        num_experts=10,
        top_k=5,
        moe_capacity_factor=0.1,
    ):
        super().__init__()

        self.norm_strategy = norm_strategy
        self.use_moe = use_moe
        self.informer_like = informer_like

        self.self_attn = XFormerAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            use_flash_attn=use_flash_attn,
            attention_type=att_type,
        )

        self.cross_attn = XFormerAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            cross_attention=True,
            use_flash_attn=use_flash_attn,
            attention_type=att_type,
        )

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
            expert_dropout=0.1,
        )

        norm_cls = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if use_adaptive_ln == "rms":
            norm_cls = AdaptiveRMSNorm
        elif use_adaptive_ln == "layer":
            norm_cls = AdaptiveLayerNorm
        self.norm1 = norm_cls(d_model, eps=layer_norm_eps)
        self.norm2 = norm_cls(d_model, eps=layer_norm_eps)
        self.norm3 = norm_cls(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.aux_loss = 0.0

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        incremental_state=None,
    ):
        is_causal = not self.informer_like

        self_attn_state = (
            None if incremental_state is None else incremental_state.get("self_attn")
        )
        cross_attn_state = (
            None if incremental_state is None else incremental_state.get("cross_attn")
        )

        if self.norm_strategy == "pre_norm":
            # === Pre-norm Self-Attention ===
            normed_tgt = self.norm1(tgt)
            tgt2, _, updated_self = self.self_attn(
                normed_tgt,
                normed_tgt,
                normed_tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=is_causal,
                layer_state=self_attn_state,
            )
            tgt = tgt + self.dropout1(tgt2)

            # === Pre-norm Cross-Attention ===
            normed_tgt = self.norm2(tgt)
            if torch.isnan(normed_tgt).any():
                raise RuntimeError("NaN detected before cross-attention residual")

            # check for nan in memory
            if torch.isnan(memory).any():
                raise RuntimeError("NaN detected in memory before cross-attention")

            tgt2, _, updated_cross = self.cross_attn(
                normed_tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                layer_state=cross_attn_state,
            )
            if torch.isnan(tgt2).any():
                raise RuntimeError(
                    "NaN detected after cross-attention and before residual"
                )

            tgt = tgt + self.dropout2(tgt2)

            if torch.isnan(tgt).any():
                raise RuntimeError("NaN detected after cross-attention residual")

            # === Pre-norm Feedforward ===
            normed_tgt = self.norm3(tgt)

            if self.use_moe:
                tgt2, self.aux_loss = self.feed_forward(
                    normed_tgt, return_aux_loss=True
                )
            else:
                tgt2 = self.feed_forward(normed_tgt)

            tgt = tgt + self.dropout3(tgt2)

            if incremental_state is not None:
                incremental_state["self_attn"] = updated_self
                incremental_state["cross_attn"] = updated_cross

            return tgt, incremental_state

        else:
            # === Post-norm Self-Attention ===
            tgt2, _, updated_self = self.self_attn(
                tgt,
                tgt,
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=is_causal,
            )
            tgt = self.norm1(tgt + self.dropout1(tgt2))

            # === Post-norm Cross-Attention ===
            tgt2, _, updated_cross = self.cross_attn(
                tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = self.norm2(tgt + self.dropout2(tgt2))

            # === Post-norm Feedforward ===
            if self.use_moe:
                tgt2, self.aux_loss = self.feed_forward(tgt, return_aux_loss=True)
            else:
                tgt2 = self.feed_forward(tgt)
            tgt = self.norm3(tgt + self.dropout3(tgt2))

            if incremental_state is not None:
                incremental_state["self_attn"] = updated_self
                incremental_state["cross_attn"] = updated_cross

            return tgt, incremental_state


class TransformerEncoder(nn.Module):
    """
    Enhanced Transformer Encoder with modern improvements:
    - Optimized layer stacking
    - Optional embedding sharing
    - Configurable normalization strategy
    - Memory-efficient attention mechanisms
    - Gradient checkpointing support for training larger models
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        hidden_size: Optional[int] = None,
        att_type: str = "prob_sparse",
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_gradient_checkpointing: bool = False,
        pos_encoding_scale: float = 1.0,
        max_seq_len: int = 5000,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
        use_moe: bool = False,
        use_adaptive_ln: str = "rms",
        pos_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        print("[Encoder] Transformer Encoder initialized.")

        # Set up model dimensions
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.norm_strategy = norm_strategy
        self.d_model = d_model
        self.input_size = input_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.is_transformer = True

        # Input projection and embedding
        self.input_projection = nn.Linear(self.input_size, d_model)

        # Positional encoding
        self.pos_encoder = (
            pos_encoder
            if pos_encoder
            else PositionalEncoding(
                d_model, dropout=dropout, max_len=max_seq_len, scale=pos_encoding_scale
            )
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    att_type=att_type,
                    use_swiglu=use_swiglu,
                    use_flash_attn=use_flash_attn,
                    layer_norm_eps=layer_norm_eps,
                    norm_strategy=norm_strategy,
                    use_moe=use_moe,
                    use_adaptive_ln=use_adaptive_ln,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization (for pre-norm architecture)
        self.apply(init_transformer_weights)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer encoder.

        Args:
            src: Source sequence [batch_size, seq_len, hidden_size]
            src_mask: Optional attention mask [seq_len, seq_len]
            src_key_padding_mask: Optional padding mask [batch_size, seq_len]

        Returns:
            Encoded output [batch_size, seq_len, d_model]
        """
        # Project input to model dimension
        src = self.input_projection(src)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Apply dropout
        src = self.dropout(src)

        # Apply transformer layers with optional gradient checkpointing
        for layer in self.layers:
            if self.training and self.use_gradient_checkpointing:
                src = torch.utils.checkpoint.checkpoint(
                    layer, src, src_mask, src_key_padding_mask
                )
            else:
                src = layer(
                    src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
                )

        # check for nan
        return src


class TransformerDecoder(nn.Module):
    """
    Enhanced Transformer Decoder with modern improvements:
    - Optimized layer stacking
    - Optional embedding sharing
    - Support for incremental/autoregressive decoding
    - Configurable normalization strategy
    - Memory-efficient attention mechanisms
    - Gradient checkpointing support for training larger models
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        hidden_size: Optional[int] = None,
        informer_like: bool = False,
        att_type: str = "prob_sparse",
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_gradient_checkpointing: bool = False,
        pos_encoding_scale: float = 1.0,
        max_seq_len: int = 5000,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
        use_moe: bool = False,
        use_adaptive_ln: str = "rms",
        pos_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        print("[Decoder] Transformer Decoder initialized.")

        # Set up model dimensions
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.d_model = d_model
        self.output_size = output_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.is_transformer = True

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = (
            pos_encoder
            if pos_encoder
            else PositionalEncoding(
                d_model, dropout=dropout, max_len=max_seq_len, scale=pos_encoding_scale
            )
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Transformer decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    informer_like=informer_like,
                    att_type=att_type,
                    use_swiglu=use_swiglu,
                    use_flash_attn=use_flash_attn,
                    layer_norm_eps=layer_norm_eps,
                    norm_strategy=norm_strategy,
                    use_moe=use_moe,
                    use_adaptive_ln=use_adaptive_ln,
                )
                for _ in range(num_layers)
            ]
        )

        # Improved output projection
        if output_size == 1:
            self.output_projection = nn.Linear(d_model, output_size)
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, output_size),
            )

        # Cache for incremental decoding
        self.incremental_state = None
        self.apply(init_transformer_weights)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer decoder.

        Args:
            tgt: Target sequence [batch_size, tgt_len, hidden_size]
            memory: Memory from encoder [batch_size, src_len, d_model]
            tgt_mask: Optional attention mask [tgt_len, tgt_len]
            memory_mask: Optional cross-attention mask [tgt_len, src_len]
            tgt_key_padding_mask: Optional padding mask [batch_size, tgt_len]
            memory_key_padding_mask: Optional padding mask [batch_size, src_len]
            incremental_state: Optional state for incremental decoding

        Returns:
            Decoded output [batch_size, tgt_len, output_size]
        """
        # Project input to model dimension
        tgt = self.input_projection(tgt)

        # Add positional encoding
        tgt = self.pos_encoder(tgt)

        # Apply dropout
        tgt = self.dropout(tgt)

        # Initialize incremental state if not provided
        if incremental_state is None:
            incremental_state = [None] * len(self.layers)

        # Apply transformer layers with optional gradient checkpointing
        for idx, layer in enumerate(self.layers):
            if self.training and self.use_gradient_checkpointing:
                tgt, _ = torch.utils.checkpoint.checkpoint(
                    layer,
                    tgt,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                )
            else:
                tgt, _ = layer(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

        # Project to output size
        return self.dropout(self.output_projection(tgt))

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: Optional[List[dict]] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Optimized single-step forward pass for autoregressive generation.
        """
        if incremental_state is None:
            incremental_state = [None] * len(self.layers)

        # Project input
        tgt = self.input_projection(tgt)

        # Position embedding for the current token
        if tgt.size(1) == 1:
            pos = incremental_state[0].get("position", 0) if incremental_state[0] else 0
            pos_enc = self.pos_encoder.pe[:, pos : pos + 1]  # safer: getattr fallback
            tgt = tgt + pos_enc

            # Update position
            for i in range(len(self.layers)):
                if incremental_state[i] is None:
                    incremental_state[i] = {"position": pos + 1}
                else:
                    incremental_state[i]["position"] = pos + 1
        else:
            tgt = self.pos_encoder(tgt)

        tgt = self.dropout(tgt)

        # Decoder layers (assumes they handle caching if enabled)
        for i, layer in enumerate(self.layers):
            tgt, updated_state = layer(
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
                incremental_state=incremental_state[i],
            )
            incremental_state[i] = updated_state

        output = self.output_projection(tgt)
        return output, incremental_state
