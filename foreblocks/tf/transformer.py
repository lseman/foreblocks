from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import (
    InformerTimeEmbedding,
    PositionalEncoding,
    RoPEPositionalEncoding,
)
from .transformer_att import MultiAttention
from .transformer_aux import *
from .transformer_moe import *


class NormWrapper(nn.Module):
    """Normalization + residual + dropout, supports pre- or post-norm."""
    def __init__(self, norm: nn.Module, strategy: str = "pre_norm", dropout_p: float = 0.0):
        super().__init__()
        assert strategy in {"pre_norm", "post_norm"}
        self.norm = norm
        self.strategy = strategy
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, fn) -> torch.Tensor:
        if self.strategy == "pre_norm":
            out = fn(self.norm(x))
            if self.dropout_p > 0:
                out = F.dropout(out, p=self.dropout_p, training=self.training)
            return x + out
        else:
            out = fn(x)
            if self.dropout_p > 0:
                out = F.dropout(out, p=self.dropout_p, training=self.training)
            out = x + out
            return self.norm(out)


class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        dim_feedforward: int = 2048,
        use_swiglu: bool = True,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.use_moe = use_moe
        self.norm_strategy = norm_strategy

        # Feedforward (MoE or MLP)
        self.feed_forward = FeedForwardBlock(
            d_model=d_model,
            dim_ff=dim_feedforward,
            dropout=dropout,
            use_swiglu=use_swiglu,
            activation=activation,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            # capacity_factor=moe_capacity_factor,
            # expert_dropout=dropout,
        )

        self.norm_wrappers = nn.ModuleList()
        self.register_buffer("_zero_aux", torch.tensor(0.0), persistent=False)
        self.aux_loss = self._zero_aux

    def _make_norm(self, d_model: int, custom_norm: str, eps: float) -> nn.Module:
        return create_norm_layer(custom_norm, d_model, eps)

    def forward_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        def ff_fn(x_normed):
            if self.use_moe:
                out, aux = self.feed_forward(x_normed, return_aux_loss=True)
                # Keep aux as tensor on correct device/dtype
                self.aux_loss = aux if torch.is_tensor(aux) else x_normed.new_tensor(aux)
                return out
            else:
                self.aux_loss = x_normed.new_zeros(())
                return self.feed_forward(x_normed)

        return self.norm_wrappers[-1](x, ff_fn)


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
        custom_norm: str = "rms",
        use_moe: bool = False,
        num_experts: int = 10,
        top_k: int = 4,
        moe_capacity_factor: float = 1.25,
    ):
        super().__init__(
            d_model,
            dropout,
            activation,
            dim_feedforward,
            use_swiglu,
            norm_strategy,
            custom_norm,
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

        norm1 = self._make_norm(d_model, custom_norm, layer_norm_eps)
        norm2 = self._make_norm(d_model, custom_norm, layer_norm_eps)
        self.norm_wrappers.extend(
            [
                NormWrapper(norm1, norm_strategy, dropout_p=dropout),
                NormWrapper(norm2, norm_strategy, dropout_p=dropout),
            ]
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def attn_fn(x_normed):
            attn_out, _, _ = self.self_attn(
                x_normed,
                x_normed,
                x_normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            return attn_out

        src = self.norm_wrappers[0](src, attn_fn)
        src = self.forward_feedforward(src)
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
        freq_modes: int = 32,
        use_swiglu: bool = True,
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        informer_like: bool = False,
        use_moe: bool = False,
        num_experts: int = 10,
        top_k: int = 5,
        moe_capacity_factor: float = 1.25,
    ):
        super().__init__(
            d_model=d_model,
            dropout=dropout,
            activation=activation,
            dim_feedforward=dim_feedforward,
            use_swiglu=use_swiglu,
            norm_strategy=norm_strategy,
            custom_norm=custom_norm,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_capacity_factor=moe_capacity_factor,
            layer_norm_eps=layer_norm_eps,
        )

        self.self_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
            cross_attention=False,
        )

        self.cross_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
            cross_attention=True,
        )

        self._is_causal = not informer_like

        # 3 NormWrappers: self-attn, cross-attn, FF
        for _ in range(3):
            norm = self._make_norm(d_model, custom_norm, layer_norm_eps)
            self.norm_wrappers.append(
                NormWrapper(norm, norm_strategy, dropout_p=dropout)
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
        # Wrap state dict for mutable updates
        state = {
            "self": incremental_state.get("self_attn") if incremental_state else None,
            "cross": incremental_state.get("cross_attn") if incremental_state else None,
        }

        def self_attn_fn(x_normed):
            out, _, updated = self.self_attn(
                x_normed,
                x_normed,
                x_normed,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=self._is_causal,
                layer_state=state["self"],
            )
            state["self"] = updated
            return out

        def cross_attn_fn(x_normed):
            out, _, updated = self.cross_attn(
                x_normed,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                layer_state=state["cross"],
            )
            state["cross"] = updated
            return out

        tgt = self.norm_wrappers[0](tgt, self_attn_fn)
        tgt = self.norm_wrappers[1](tgt, cross_attn_fn)
        tgt = self.forward_feedforward(tgt)

        if incremental_state is not None:
            incremental_state["self_attn"] = state["self"]
            incremental_state["cross_attn"] = state["cross"]
            return tgt, incremental_state

        return tgt, None


######################################################


class BaseTransformer(nn.Module, ABC):
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
        custom_norm: str = "rms",
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
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self._use_gradient_checkpointing = use_gradient_checkpointing
        self._use_final_norm = use_final_norm
        self.dropout_p = dropout
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = pos_encoder or PositionalEncoding(
            d_model, max_len=max_seq_len, scale=pos_encoding_scale
        )

        # Layer creation
        layer_kwargs = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type=att_type,
            layer_norm_eps=layer_norm_eps,
            norm_strategy=norm_strategy,
            custom_norm=custom_norm,
            use_swiglu=use_swiglu,
            freq_modes=freq_modes,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            # moe_capacity_factor=moe_capacity_factor,
            **kwargs,
        )

        if share_layers:
            self.shared_layer = self._make_layer(**layer_kwargs)
            self.layers = None
        else:
            self.layers = nn.ModuleList(
                [self._make_layer(**layer_kwargs) for _ in range(num_layers)]
            )
            self.shared_layer = None

        # Final normalization
        self.final_norm = (
            create_norm_layer(custom_norm, d_model, layer_norm_eps)
            if use_final_norm
            else nn.Identity()
        )

        # Initialize weights
        self.apply(self._init_weights)

    @abstractmethod
    def _make_layer(self, **kwargs): ...

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "padding_idx", None) is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def _apply_input_processing(self, x, additional_features=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        if additional_features is not None:
            x = x + additional_features
        if self.training and self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=True)
        return x

    def _apply_final_norm(self, x):
        return self.final_norm(x)

    def _get_layer(self, idx: int) -> nn.Module:
        return self.shared_layer if self.layers is None else self.layers[idx]

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def get_aux_loss(self):
        """Return accumulated auxiliary loss (e.g., from MoE)."""
        return getattr(self, "aux_loss", None) or torch.tensor(
            0.0, device=self.input_projection.weight.device
        )


class TransformerEncoder(BaseTransformer):
    def __init__(self, input_size: int, **kwargs):
        self.freq_modes = kwargs.get("freq_modes", 32)
        super().__init__(input_size, **kwargs)
        self.input_size = input_size
        self.time_encoder = InformerTimeEmbedding(self.d_model)

    def _make_layer(self, **kwargs):
        return TransformerEncoderLayer(**kwargs)

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, time_features=None
    ):
        # Input validation
        B, T, C = src.shape
        if C != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {C}"
            )
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds maximum {self.max_seq_len}"
            )

        # Apply time embedding if available
        time_emb = (
            self.time_encoder(time_features) if time_features is not None else None
        )
        src = self._apply_input_processing(src, time_emb)

        # Process through layers
        aux_loss = src.new_zeros(())
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            if self.training and self._use_gradient_checkpointing:
                # Checkpoint for memory efficiency during training
                def fn(_src):
                    return layer(_src, src_mask, src_key_padding_mask)
                src = torch.utils.checkpoint.checkpoint(fn, src, use_reentrant=False)
            else:
                src = layer(src, src_mask, src_key_padding_mask)

            # Accumulate auxiliary losses
            layer_aux = getattr(layer, "aux_loss", src.new_zeros(()))
            if not torch.is_tensor(layer_aux):
                layer_aux = src.new_tensor(layer_aux)
            aux_loss = aux_loss + layer_aux

        # Average aux loss across layers to prevent scaling with depth
        self.aux_loss = aux_loss / self.num_layers if self.num_layers > 0 else aux_loss
        return self._apply_final_norm(src)


class TransformerDecoder(BaseTransformer):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        informer_like: bool = False,
        use_time_encoding: bool = False,
        **kwargs
    ):
        self.output_size = output_size
        self.informer_like = informer_like
        self.use_time_encoding = use_time_encoding
        super().__init__(input_size, informer_like=informer_like, **kwargs)
        
        # Optional time encoding for decoder
        if use_time_encoding:
            self.time_encoder = InformerTimeEmbedding(self.d_model)
        
        # Output projection
        self.output_projection = (
            nn.Identity()
            if output_size == self.d_model
            else nn.Linear(self.d_model, output_size)
        )

    def _make_layer(self, **kwargs):
        return TransformerDecoderLayer(**kwargs)

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
        time_features=None,
    ):
        # Auto-generate causal mask if not provided and not informer_like
        if tgt_mask is None and not self.informer_like:
            T = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(T, tgt.device)

        # Apply time embedding if available
        time_emb = None
        if self.use_time_encoding and time_features is not None:
            time_emb = self.time_encoder(time_features)
        
        tgt = self._apply_input_processing(tgt, time_emb)

        # Manage layer states for incremental decoding
        layer_states = (
            incremental_state.get("layers", [None] * self.num_layers)
            if incremental_state
            else [None] * self.num_layers
        )

        # Process through layers
        aux_loss = tgt.new_zeros(())
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            if self.training and self._use_gradient_checkpointing:
                # During training with checkpointing, drop incremental_state
                # (recompute from scratch for backward pass)
                def fn(_tgt, _mem):
                    out, _ = layer(
                        _tgt,
                        _mem,
                        tgt_mask,
                        memory_mask,
                        tgt_key_padding_mask,
                        memory_key_padding_mask,
                        incremental_state=None,
                    )
                    return out
                tgt = torch.utils.checkpoint.checkpoint(fn, tgt, memory, use_reentrant=False)
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

            # Accumulate auxiliary losses
            layer_aux = getattr(layer, "aux_loss", tgt.new_zeros(()))
            if not torch.is_tensor(layer_aux):
                layer_aux = tgt.new_tensor(layer_aux)
            aux_loss = aux_loss + layer_aux

        # Average aux loss across layers
        self.aux_loss = aux_loss / self.num_layers if self.num_layers > 0 else aux_loss
        tgt = self._apply_final_norm(tgt)

        # Update incremental state
        if incremental_state is not None:
            incremental_state["layers"] = layer_states

        # Project to output size
        tgt = self.output_projection(tgt)
        return (tgt, incremental_state) if return_incremental_state else tgt

    def forward_one_step(self, tgt, memory, incremental_state=None, time_features=None):
        """Convenience method for single-step autoregressive generation."""
        return self.forward(
            tgt,
            memory,
            incremental_state=incremental_state or {},
            return_incremental_state=True,
            time_features=time_features,
        )
