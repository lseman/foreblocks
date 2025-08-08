from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import (InformerTimeEmbedding, PositionalEncoding,
                         RoPEPositionalEncoding)
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


class NormWrapper(nn.Module):
    """Wraps normalization with residual + dropout, pre- or post-norm."""

    def __init__(
        self, norm: nn.Module, strategy: str = "pre_norm", dropout_p: float = 0.0
    ):
        super().__init__()
        assert strategy in {"pre_norm", "post_norm"}
        self.norm = norm
        self.strategy = strategy
        self.dropout_p = dropout_p

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, fn, training: bool
    ) -> torch.Tensor:
        if self.strategy == "pre_norm":
            x_norm = self.norm(x)
            out = fn(x_norm)
            if training and self.dropout_p > 0:
                out = F.dropout(out, p=self.dropout_p, training=True)
            return residual + out
        else:
            out = fn(x)
            if training and self.dropout_p > 0:
                out = F.dropout(out, p=self.dropout_p, training=True)
            out = residual + out
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
        use_adaptive_ln: str = "rms",
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.training_dropout = dropout > 0.0
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
            capacity_factor=moe_capacity_factor,
            expert_dropout=dropout,
        )

        # Norm stubs for attention and feedforward — subclasses fill in
        self.norm_wrappers = nn.ModuleList()
        self.aux_loss = 0.0

    def _make_norm(self, d_model: int, use_adaptive_ln: str, eps: float) -> nn.Module:
        return create_norm_layer(use_adaptive_ln, d_model, eps)

    def forward_feedforward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        def ff_fn(x_normed):
            if self.use_moe:
                out, aux = self.feed_forward(x_normed, return_aux_loss=True)
                self.aux_loss = aux
                return out
            else:
                self.aux_loss = 0.0
                return self.feed_forward(x_normed)

        return self.norm_wrappers[-1](x, x, ff_fn, training)


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
        use_adaptive_ln: str = "rms",
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

        norm1 = self._make_norm(d_model, use_adaptive_ln, layer_norm_eps)
        norm2 = self._make_norm(d_model, use_adaptive_ln, layer_norm_eps)
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
        training = self.training

        def attn_fn(x_normed):
            attn_out, _, _ = self.self_attn(
                x_normed,
                x_normed,
                x_normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            return attn_out

        src = self.norm_wrappers[0](src, src, attn_fn, training)
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
        freq_modes: int = 32,  # ✅ now supported
        use_swiglu: bool = True,
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_adaptive_ln: str = "rms",
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
            use_adaptive_ln=use_adaptive_ln,
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
            freq_modes=freq_modes,  # ✅ propagated
            cross_attention=False,
        )

        self.cross_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,  # ✅ propagated
            cross_attention=True,
        )

        self._is_causal = not informer_like

        # Add 3 NormWrappers (self-attn, cross-attn, FF)
        for _ in range(3):
            norm = self._make_norm(d_model, use_adaptive_ln, layer_norm_eps)
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
        training = self.training

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

        # Apply layers with norm wrappers
        tgt = self.norm_wrappers[0](tgt, tgt, self_attn_fn, training)
        tgt = self.norm_wrappers[1](tgt, tgt, cross_attn_fn, training)
        tgt = self.forward_feedforward(tgt, training)

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
        use_adaptive_ln: str = "rms",
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

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = pos_encoder or RoPEPositionalEncoding(
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
            use_adaptive_ln=use_adaptive_ln,
            use_swiglu=use_swiglu,
            freq_modes=freq_modes,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_capacity_factor=moe_capacity_factor,
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
            create_norm_layer(use_adaptive_ln, d_model, layer_norm_eps)
            if use_final_norm
            else nn.Identity()
        )

        self.apply(self._init_weights)

    @abstractmethod
    def _make_layer(self, **kwargs): ...

    def _init_weights(self, module: nn.Module):
        for m in module.modules():
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
            x += additional_features
        if self.training and self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=True)
        return x

    def _apply_final_norm(self, x):
        return self.final_norm(x)

    def _get_layer(self, idx: int) -> nn.Module:
        return self.shared_layer if self.layers is None else self.layers[idx]

    def get_aux_loss(self):
        return getattr(self, "aux_loss", 0.0)


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
        time_emb = (
            self.time_encoder(time_features) if time_features is not None else None
        )
        src = self._apply_input_processing(src, time_emb)

        aux_loss = 0.0
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            if self.training and self._use_gradient_checkpointing:
                src = torch.utils.checkpoint.checkpoint(
                    layer, src, src_mask, src_key_padding_mask, use_reentrant=False
                )
            else:
                src = layer(src, src_mask, src_key_padding_mask)

            aux_loss += getattr(layer, "aux_loss", 0.0)

        self.aux_loss = aux_loss
        return self._apply_final_norm(src)


class TransformerDecoder(BaseTransformer):
    def __init__(
        self, input_size: int, output_size: int, informer_like: bool = False, **kwargs
    ):
        self.output_size = output_size
        self.informer_like = informer_like
        super().__init__(input_size, informer_like=informer_like, **kwargs)
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
    ):
        tgt = self._apply_input_processing(tgt)

        layer_states = (
            incremental_state.get("layers", [None] * self.num_layers)
            if incremental_state
            else [None] * self.num_layers
        )

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
            aux_loss += getattr(layer, "aux_loss", 0.0)

        self.aux_loss = aux_loss
        tgt = self._apply_final_norm(tgt)

        if incremental_state is not None:
            incremental_state["layers"] = layer_states

        tgt = self.output_projection(tgt)
        return (tgt, incremental_state) if return_incremental_state else tgt

    def forward_one_step(self, tgt, memory, incremental_state=None):
        return self.forward(
            tgt,
            memory,
            incremental_state=incremental_state or {},
            return_incremental_state=True,
        )
