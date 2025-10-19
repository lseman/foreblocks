from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.node_spec import node

from .embeddings import InformerTimeEmbedding, PositionalEncoding
from .moe import *
from .multi_att import MultiAttention
from .norms import *


class NormWrapper(nn.Module):
    """Simplified normalization wrapper with residual connection."""
    def __init__(
        self, 
        d_model: int,
        norm_type: str = "rms",
        strategy: str = "pre_norm",
        dropout: float = 0.0,
        eps: float = 1e-5
    ):
        super().__init__()
        self.norm = create_norm_layer(norm_type, d_model, eps)
        self.strategy = strategy
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        """Apply norm + sublayer + residual based on strategy."""
        if self.strategy == "pre_norm":
            return x + self.dropout(sublayer(self.norm(x)))
        else:  # post_norm
            return self.norm(x + self.dropout(sublayer(x)))


class BaseTransformerLayer(nn.Module):
    """Base layer with feedforward and aux loss tracking."""
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.use_moe = use_moe
        
        self.feed_forward = FeedForwardBlock(
            d_model=d_model,
            dim_ff=dim_feedforward,
            dropout=dropout,
            use_swiglu=use_swiglu,
            activation=activation,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # Simplified aux loss tracking (scalar for broadcasting)
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def _reset_aux_loss(self):
        """Reset aux loss to zero."""
        self.aux_loss.zero_()

    def _update_aux_loss(self, new_loss):
        """Add to aux loss (handles tensor/scalar)."""
        if torch.is_tensor(new_loss):
            self.aux_loss += new_loss
        elif new_loss != 0:
            self.aux_loss += torch.tensor(new_loss, device=self.aux_loss.device)


class TransformerEncoderLayer(BaseTransformerLayer):
    """Encoder layer: self-attention + feedforward."""
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
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__(
            d_model, dim_feedforward, dropout, activation,
            use_swiglu, use_moe, num_experts, top_k
        )

        self.self_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
        )

        # Two norm wrappers: attention + feedforward
        self.attn_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)
        self.ff_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._reset_aux_loss()
        
        # Self-attention block
        def attn_fn(x):
            out, _, _ = self.self_attn(x, x, x, src_mask, src_key_padding_mask)
            return out
        
        src = self.attn_norm(src, attn_fn)
        
        # Feedforward block
        def ff_fn(x):
            if self.use_moe:
                out, aux = self.feed_forward(x, return_aux_loss=True)
                self._update_aux_loss(aux)
                return out
            return self.feed_forward(x)
        
        src = self.ff_norm(src, ff_fn)
        return src


class TransformerDecoderLayer(BaseTransformerLayer):
    """Decoder layer: self-attention + cross-attention + feedforward."""
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
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        informer_like: bool = False,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__(
            d_model, dim_feedforward, dropout, activation,
            use_swiglu, use_moe, num_experts, top_k
        )

        self.self_attn = MultiAttention(
            d_model=d_model, n_heads=nhead, dropout=dropout,
            attention_type=att_type, freq_modes=freq_modes, cross_attention=False,
        )
        self.cross_attn = MultiAttention(
            d_model=d_model, n_heads=nhead, dropout=dropout,
            attention_type=att_type, freq_modes=freq_modes, cross_attention=True,
        )
        
        self.is_causal = not informer_like

        # Three norm wrappers
        self.self_attn_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)
        self.cross_attn_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)
        self.ff_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)

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
        self._reset_aux_loss()
        
        # Manage incremental state
        state = {
            "self": incremental_state.get("self_attn") if incremental_state else None,
            "cross": incremental_state.get("cross_attn") if incremental_state else None,
        }

        # Self-attention
        def self_attn_fn(x):
            out, _, updated = self.self_attn(
                x, x, x, tgt_mask, tgt_key_padding_mask,
                is_causal=self.is_causal, layer_state=state["self"]
            )
            state["self"] = updated
            return out

        tgt = self.self_attn_norm(tgt, self_attn_fn)

        # Cross-attention
        def cross_attn_fn(x):
            out, _, updated = self.cross_attn(
                x, memory, memory, memory_mask, memory_key_padding_mask,
                layer_state=state["cross"]
            )
            state["cross"] = updated
            return out

        tgt = self.cross_attn_norm(tgt, cross_attn_fn)

        # Feedforward
        def ff_fn(x):
            if self.use_moe:
                out, aux = self.feed_forward(x, return_aux_loss=True)
                self._update_aux_loss(aux)
                return out
            return self.feed_forward(x)

        tgt = self.ff_norm(tgt, ff_fn)

        # Update incremental state if needed
        if incremental_state is not None:
            incremental_state.update({"self_attn": state["self"], "cross_attn": state["cross"]})
            return tgt, incremental_state

        return tgt, None


class BaseTransformer(nn.Module, ABC):
    """Base transformer with unified input processing."""
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
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
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
        # Patching options
        use_patch: bool = False,
        patch_len: int = 16,
        patch_stride: int = 8,
        channel_independent: bool = False,
        ci_aggregate: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Patching config
        self.use_patch = use_patch
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.channel_independent = channel_independent
        self.ci_aggregate = ci_aggregate

        # Input processing setup
        self._setup_input_processing(input_size, pos_encoder, pos_encoding_scale)

        # Layer creation
        layer_kwargs = {
            "d_model": d_model, "nhead": nhead, "dim_feedforward": dim_feedforward,
            "dropout": dropout, "activation": activation, "att_type": att_type,
            "norm_strategy": norm_strategy, "custom_norm": custom_norm,
            "layer_norm_eps": layer_norm_eps, "use_swiglu": use_swiglu,
            "freq_modes": freq_modes, "use_moe": use_moe,
            "num_experts": num_experts, "top_k": top_k, **kwargs,
        }

        if share_layers:
            self.shared_layer = self._make_layer(**layer_kwargs)
            self.layers = None
        else:
            self.layers = nn.ModuleList([self._make_layer(**layer_kwargs) for _ in range(num_layers)])
            self.shared_layer = None

        # Final norm
        self.final_norm = (
            create_norm_layer(custom_norm, d_model, layer_norm_eps)
            if use_final_norm else nn.Identity()
        )

        # Aux loss tracking (scalar tensor for proper broadcasting)
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        
        self.apply(self._init_weights)

    def _setup_input_processing(self, input_size: int, pos_encoder: Optional[nn.Module], scale: float):
        """Unified setup for input projection and positional encoding."""
        if self.use_patch:
            from .embeddings import CIPatchEmbedding, PatchEmbedding
            
            if self.channel_independent:
                self.input_adapter = CIPatchEmbedding(
                    in_channels=input_size, embed_dim=self.d_model,
                    patch_len=self.patch_len, patch_stride=self.patch_stride,
                )
                # CI aggregation projection
                if self.ci_aggregate in ("linear", "concat"):
                    self.ci_proj = nn.Linear(input_size * self.d_model, self.d_model)
                else:
                    self.ci_proj = None
            else:
                self.input_adapter = PatchEmbedding(
                    in_channels=input_size, embed_dim=self.d_model,
                    patch_len=self.patch_len, patch_stride=self.patch_stride,
                )
                self.ci_proj = None
            
            # Calculate max tokens after patching
            max_tokens = self.input_adapter.output_len(self.max_seq_len)
            if max_tokens <= 0:
                raise ValueError(
                    f"max_seq_len={self.max_seq_len} too small for patch_len={self.patch_len}"
                )
        else:
            self.input_adapter = nn.Linear(input_size, self.d_model)
            self.ci_proj = None
            max_tokens = self.max_seq_len

        # Positional encoding
        self.pos_encoder = pos_encoder or PositionalEncoding(
            self.d_model, max_len=max_tokens, scale=scale
        )

    @abstractmethod
    def _make_layer(self, **kwargs): ...

    def _init_weights(self, m: nn.Module):
        """Initialize weights following best practices."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _get_layer(self, idx: int) -> nn.Module:
        return self.shared_layer if self.layers is None else self.layers[idx]

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        return torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)

    def _aggregate_aux_loss(self, layers_used: int):
        """Aggregate aux loss from all layers."""
        total_aux = torch.tensor(0.0, device=self.aux_loss.device, dtype=self.aux_loss.dtype)
        for i in range(layers_used):
            layer = self._get_layer(i)
            if hasattr(layer, 'aux_loss'):
                total_aux = total_aux + layer.aux_loss
        self.aux_loss = total_aux / max(layers_used, 1)

@node(
    type_id="transformer_encoder",
    name="Transformer Encoder",
    category="Encoder",
    color="bg-gradient-to-br from-green-700 to-green-800",
)
class TransformerEncoder(BaseTransformer):
    """Transformer encoder with optional channel-independent patching."""
    def __init__(self, input_size: int, use_time_encoding: bool = True, **kwargs):
        self.use_time_encoding = use_time_encoding
        super().__init__(input_size, **kwargs)
        self.input_size = input_size
        
        if use_time_encoding:
            self.time_encoder = InformerTimeEmbedding(self.d_model)
        else:
            self.time_encoder = None

    def _make_layer(self, **kwargs):
        # Remove decoder-specific kwargs if present
        kwargs.pop("informer_like", None)
        return TransformerEncoderLayer(**kwargs)

    def _process_standard_input(self, x: torch.Tensor, time_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Process input for standard (non-CI) path."""
        # Project and add positional encoding
        x = self.input_adapter(x) if self.use_patch else self.input_adapter(x)
        x = self.pos_encoder(x)
        
        # Add time features if provided
        if self.time_encoder is not None and time_features is not None:
            time_emb = self.time_encoder(time_features)
            if time_emb.shape[:2] != x.shape[:2]:
                raise ValueError(f"Time features shape {time_emb.shape} incompatible with input {x.shape}")
            x = x + time_emb
        
        # Dropout
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)
        
        return x

    def _process_ci_input(self, x: torch.Tensor) -> torch.Tensor:
        """Process input for channel-independent path."""
        B, T, C = x.shape
        
        # Patch embedding: [B, T_p, C, D]
        x = self.input_adapter(x)
        B, T_p, C, D = x.shape
        
        # Flatten channels into batch: [B*C, T_p, D]
        x = x.reshape(B * C, T_p, D)
        x = self.pos_encoder(x)
        
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)
        
        return x, (B, C, T_p, D)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = src.shape
        
        if C != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {C}")
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max {self.max_seq_len}")

        # Channel-independent patching path
        if self.use_patch and self.channel_independent:
            x, (B, C, T_p, D) = self._process_ci_input(src)
            
            # Process through layers (shared weights across channels)
            for i in range(self.num_layers):
                layer = self._get_layer(i)
                if self.training and self.use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        lambda _x: layer(_x, None, None), x, use_reentrant=False
                    )
                else:
                    x = layer(x, None, None)
            
            # Unflatten and aggregate: [B*C, T_p, D] -> [B, T_p, D]
            x = x.view(B, C, T_p, D).transpose(1, 2)  # [B, T_p, C, D]
            
            if self.ci_aggregate == "mean":
                x = x.mean(dim=2)
            elif self.ci_aggregate in ("concat", "linear"):
                x = x.reshape(B, T_p, C * D)
                x = self.ci_proj(x)
            else:
                raise ValueError(f"Unknown ci_aggregate: {self.ci_aggregate}")
            
            self._aggregate_aux_loss(self.num_layers)
            return self.final_norm(x)

        # Standard path
        x = self._process_standard_input(src, time_features)
        
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            if self.training and self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    lambda _x: layer(_x, src_mask, src_key_padding_mask),
                    x, use_reentrant=False
                )
            else:
                x = layer(x, src_mask, src_key_padding_mask)
        
        self._aggregate_aux_loss(self.num_layers)
        return self.final_norm(x)

@node(
    type_id="transformer_decoder",
    name="Transformer Decoder",
    category="Decoder",
    color="bg-gradient-to-br from-purple-700 to-purple-800",
)
class TransformerDecoder(BaseTransformer):
    """Transformer decoder with Informer-like support."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        label_len: int = 0,
        informer_like: bool = False,
        use_time_encoding: bool = True,
        **kwargs,
    ):
        self.output_size = output_size
        self.label_len = label_len  # FIX: Now properly initialized
        self.informer_like = informer_like
        self.use_time_encoding = use_time_encoding
        
        super().__init__(input_size, informer_like=informer_like, **kwargs)
        
        if use_time_encoding:
            self.time_encoder = InformerTimeEmbedding(self.d_model)
        else:
            self.time_encoder = None
        
        # Output projection
        self.output_projection = (
            nn.Identity() if output_size == self.d_model
            else nn.Linear(self.d_model, output_size)
        )

    def _make_layer(self, **kwargs):
        return TransformerDecoderLayer(**kwargs)

    def _create_informer_padding_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Create padding mask for Informer-style decoding (mask future placeholders)."""
        if not self.informer_like or self.label_len >= T:
            return None
        
        label_len = max(0, min(self.label_len, T))
        # Mask positions after label_len (True = masked)
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        mask[:, label_len:] = True
        return mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict] = None,
        return_incremental_state: bool = False,
        time_features: Optional[torch.Tensor] = None,
    ):
        B, T, _ = tgt.shape
        device = tgt.device

        # Always use causal mask unless provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(T, device)

        # Input processing
        tgt = self.input_adapter(tgt)
        tgt = self.pos_encoder(tgt)
        
        if self.time_encoder is not None and time_features is not None:
            time_emb = self.time_encoder(time_features)
            if time_emb.shape[:2] == tgt.shape[:2]:
                tgt = tgt + time_emb
        
        if self.training and self.dropout > 0:
            tgt = F.dropout(tgt, p=self.dropout, training=True)

        # Informer-style padding mask for zero placeholders
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._create_informer_padding_mask(B, T, device)

        # Incremental state management
        layer_states = (
            incremental_state.get("layers", [None] * self.num_layers)
            if incremental_state else [None] * self.num_layers
        )

        # Process through layers
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            
            if self.training and self.use_gradient_checkpointing:
                tgt = torch.utils.checkpoint.checkpoint(
                    lambda _t: layer(_t, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, None)[0],
                    tgt, use_reentrant=False
                )
                layer_states[i] = None
            else:
                tgt, layer_states[i] = layer(
                    tgt, memory, tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask, layer_states[i]
                )

        self._aggregate_aux_loss(self.num_layers)
        
        # Final norm and projection
        tgt = self.final_norm(tgt)
        out = self.output_projection(tgt)

        if return_incremental_state:
            if incremental_state is None:
                incremental_state = {}
            incremental_state["layers"] = layer_states
            return out, incremental_state
        
        return out

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: Optional[Dict] = None,
        time_features: Optional[torch.Tensor] = None,
    ):
        """Single-step autoregressive generation."""
        return self.forward(
            tgt, memory,
            incremental_state=incremental_state or {},
            return_incremental_state=True,
            time_features=time_features,
        )
