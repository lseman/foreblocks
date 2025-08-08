import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class LinearSelfAttention(nn.Module):
    """Improved linear self-attention with consistent behavior"""

    def __init__(self, dim, heads=4, dropout=0.0, causal=False):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.causal = causal

        # Ensure head_dim is valid
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"

        # Fused QKV projection
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout

        # Layer norm for stability
        self.norm = RMSNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads

        # Apply normalization first
        x_norm = self.norm(x)

        # More efficient reshape pattern
        qkv = self.to_qkv(x_norm).view(B, T, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, T, head_dim]

        if self.causal:
            # Standard causal attention for autoregressive behavior
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Create causal mask
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores.masked_fill_(mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout_p)

            out = torch.matmul(attn_weights, v)
        else:
            # Linear attention with numerical stability (non-causal)
            # Apply softmax to keys for stability
            k_norm = F.softmax(k * self.scale, dim=-2)

            # Compute context and output
            context = torch.einsum("bhtd,bhtv->bhdv", k_norm, v)
            out = torch.einsum("bhtd,bhdv->bhtv", q, context)

        # Reshape back and apply residual connection
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        if self.training and self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p)

        # Residual connection
        return x + out


class LightweightTransformerEncoder(nn.Module):
    """Improved transformer encoder with better RNN compatibility"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_layers=2,
        dropout=0.1,
        nhead=4,
        max_seq_len=512,
        causal=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.causal = causal

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, latent_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": LinearSelfAttention(
                            latent_dim, heads=nhead, dropout=dropout, causal=causal
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(latent_dim, latent_dim * 4, bias=False),
                            nn.SiLU(),
                            nn.Linear(latent_dim * 4, latent_dim, bias=False),
                        ),
                        "norm1": RMSNorm(latent_dim),
                        "norm2": RMSNorm(latent_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(latent_dim)
        self.dropout_p = dropout

        # State projection for RNN compatibility
        self.state_proj = nn.Linear(
            latent_dim, latent_dim * 2, bias=False
        )  # Project to (h, c)

    def forward(self, x, hidden_state=None):
        B, T, _ = x.shape

        x = self.input_proj(x)

        # Add positional encoding with interpolation for longer sequences
        if T <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :T]
        else:
            # Interpolate positional embeddings for longer sequences
            pos_embed_interp = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            x = x + pos_embed_interp

        # Process through transformer layers
        for layer in self.layers:
            # Self-attention with residual (handled inside LinearSelfAttention)
            x = layer["self_attn"](x)

            # FFN with residual
            ffn_input = layer["norm1"](x)
            ffn_out = layer["ffn"](ffn_input)
            if self.training and self.dropout_p > 0:
                ffn_out = F.dropout(ffn_out, p=self.dropout_p)
            x = layer["norm2"](x + ffn_out)

        x = self.final_norm(x)

        # Create RNN-compatible outputs
        # Context is the last timestep
        context = x[:, -1:, :]  # [B, 1, D]

        # Create hidden state compatible with RNNs
        # Use mean pooling of the sequence for global representation
        pooled = x.mean(dim=1)  # [B, D]
        state_proj = self.state_proj(pooled)  # [B, 2*D]
        h_state, c_state = state_proj.chunk(2, dim=-1)  # Each [B, D]

        # Reshape to match RNN state format [num_layers, B, D]
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1)
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1)

        return x, context, (h_state, c_state)


class LightweightTransformerDecoder(nn.Module):
    """Improved transformer decoder with better compatibility"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_layers=2,
        dropout=0.1,
        nhead=4,
        max_seq_len=512,
        causal=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.causal = causal

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, latent_dim) * 0.02)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": LinearSelfAttention(
                            latent_dim, heads=nhead, dropout=dropout, causal=causal
                        ),
                        "cross_attn": nn.MultiheadAttention(
                            latent_dim,
                            nhead,
                            dropout=dropout,
                            batch_first=True,
                            bias=False,
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(latent_dim, latent_dim * 4, bias=False),
                            nn.SiLU(),
                            nn.Linear(latent_dim * 4, latent_dim, bias=False),
                        ),
                        "norm1": RMSNorm(latent_dim),
                        "norm2": RMSNorm(latent_dim),
                        "norm3": RMSNorm(latent_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(latent_dim)
        self.dropout_p = dropout
        self.state_proj = nn.Linear(latent_dim, latent_dim * 2, bias=False)

    def _prepare_memory(self, memory_or_hidden):
        """Robustly prepare memory from various input formats"""
        if memory_or_hidden is None:
            return None

        if isinstance(memory_or_hidden, tuple):
            if len(memory_or_hidden) == 2:  # (h, c) format
                h, c = memory_or_hidden
                if h.dim() == 3:
                    # Handle both [layers, batch, dim] and [batch, layers, dim]
                    if h.size(0) == self.num_layers:
                        memory = h.transpose(0, 1)  # [batch, layers, dim]
                    else:
                        memory = h  # Already [batch, seq, dim]
                else:
                    memory = h.unsqueeze(1)  # Add sequence dimension
            else:
                memory = memory_or_hidden[0]
        else:
            # Single tensor
            if memory_or_hidden.dim() == 3:
                if memory_or_hidden.size(0) == self.num_layers:
                    memory = memory_or_hidden.transpose(0, 1)
                else:
                    memory = memory_or_hidden
            else:
                memory = memory_or_hidden.unsqueeze(1)

        return memory

    def forward(self, tgt, memory_or_hidden, hidden_state=None):
        tgt = self.input_proj(tgt)
        seq_len = tgt.size(1)

        # Add positional encoding
        if seq_len <= self.pos_embed.size(1):
            tgt = tgt + self.pos_embed[:, :seq_len]
        else:
            pos_embed_interp = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            tgt = tgt + pos_embed_interp

        # Prepare memory
        memory = self._prepare_memory(memory_or_hidden)

        # Process through decoder layers
        for layer in self.layers:
            # Self-attention (causal)
            tgt = layer["self_attn"](tgt)

            # Cross-attention with robust error handling
            if memory is not None:
                try:
                    attn_input = layer["norm1"](tgt)
                    cross_out, _ = layer["cross_attn"](attn_input, memory, memory)
                    if self.training and self.dropout_p > 0:
                        cross_out = F.dropout(cross_out, p=self.dropout_p)
                    tgt = layer["norm2"](tgt + cross_out)
                except (RuntimeError, ValueError):
                    # Fallback: skip cross-attention if shapes don't match
                    tgt = layer["norm2"](tgt)
            else:
                tgt = layer["norm2"](tgt)

            # FFN
            ffn_input = layer["norm2"](tgt) if memory is None else tgt
            ffn_out = layer["ffn"](ffn_input)
            if self.training and self.dropout_p > 0:
                ffn_out = F.dropout(ffn_out, p=self.dropout_p)
            tgt = layer["norm3"](tgt + ffn_out)

        tgt = self.final_norm(tgt)

        # Create RNN-compatible state
        last_token = tgt[:, -1]  # [B, D]
        state_proj = self.state_proj(last_token)  # [B, 2*D]
        h_state, c_state = state_proj.chunk(2, dim=-1)

        # Reshape to RNN format
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1)
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1)

        return tgt, (h_state, c_state)


class ArchitectureNormalizer(nn.Module):
    """Normalizes outputs from different architectures for compatibility"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Projection layers to ensure consistent output dimensions
        self.rnn_proj = nn.Linear(latent_dim, latent_dim)
        self.transformer_proj = nn.Linear(latent_dim, latent_dim)

        # State normalization
        self.state_norm = nn.LayerNorm(latent_dim)

    def normalize_state(self, state, arch_type: str):
        """FIX: Better state normalization with type checking"""
        if state is None:
            return None, None

        if arch_type == "lstm":
            if isinstance(state, tuple) and len(state) == 2:
                h, c = state
                return self.state_norm(h), self.state_norm(c)
            else:
                # Handle malformed LSTM state
                h = state if not isinstance(state, tuple) else state[0]
                c = torch.zeros_like(h)
                return self.state_norm(h), self.state_norm(c)

        elif arch_type == "gru":
            h = state if not isinstance(state, tuple) else state[0]
            c = torch.zeros_like(h)
            return self.state_norm(h), self.state_norm(c)

        elif arch_type == "transformer":
            if isinstance(state, tuple) and len(state) == 2:
                h, c = state
                return self.state_norm(h), self.state_norm(c)
            else:
                h = state if not isinstance(state, tuple) else state[0]
                c = torch.zeros_like(h) if h is not None else None
                return self.state_norm(h) if h is not None else None, (
                    self.state_norm(c) if c is not None else None
                )

    def normalize_output(self, output: torch.Tensor, arch_type: str) -> torch.Tensor:
        """Apply architecture-specific normalization"""
        if arch_type in ["lstm", "gru"]:
            return self.rnn_proj(output)
        elif arch_type == "transformer":
            return self.transformer_proj(output)
        else:
            return output


class MixedEncoder(nn.Module):
    """Improved mixed encoder with better compatibility"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.temperature = temperature

        # Individual encoders
        self.lstm = nn.LSTM(
            input_dim,
            latent_dim,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_dim,
            latent_dim,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )
        self.transformer = LightweightTransformerEncoder(
            input_dim=input_dim, latent_dim=latent_dim, num_layers=2, dropout=dropout
        )

        self.encoders = nn.ModuleList([self.lstm, self.gru, self.transformer])
        # Normalization and compatibility
        self.normalizer = ArchitectureNormalizer(latent_dim)

        # Architecture selection
        num_options = len(self.encoders)
        init = torch.zeros(num_options)
        init[random.randint(0, num_options - 1)] = 1.0
        self.register_parameter("alphas", nn.Parameter(init))

        # Context projection to ensure consistent format
        self.context_proj = nn.Linear(latent_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly"""
        for rnn in [self.lstm, self.gru]:
            for name, param in rnn.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.data.fill_(0)
                    if "lstm" in str(rnn.__class__).lower() and "bias_ih" in name:
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with improved compatibility"""
        weights = F.gumbel_softmax(self.alphas, tau=self.temperature, hard=False, dim=0)

        # Run all encoders
        lstm_out, lstm_state = self.lstm(x)
        gru_out, gru_state = self.gru(x)
        trans_out, trans_ctx, trans_state = self.transformer(x)

        # Normalize outputs for compatibility
        lstm_out_norm = self.normalizer.normalize_output(lstm_out, "lstm")
        gru_out_norm = self.normalizer.normalize_output(gru_out, "gru")
        trans_out_norm = self.normalizer.normalize_output(trans_out, "transformer")

        # Normalize states
        lstm_state_norm = self.normalizer.normalize_state(lstm_state, "lstm")
        gru_state_norm = self.normalizer.normalize_state(gru_state, "gru")
        trans_state_norm = self.normalizer.normalize_state(trans_state, "transformer")

        # Weighted combination of normalized outputs
        output = (
            weights[0] * lstm_out_norm
            + weights[1] * gru_out_norm
            + weights[2] * trans_out_norm
        )

        # Create consistent context (last timestep)
        lstm_ctx = lstm_out[:, -1:, :]
        gru_ctx = gru_out[:, -1:, :]
        # trans_ctx is already provided by transformer

        context = weights[0] * lstm_ctx + weights[1] * gru_ctx + weights[2] * trans_ctx
        context = self.context_proj(context)

        # Blend states properly
        h_blended = (
            weights[0] * lstm_state_norm[0]
            + weights[1] * gru_state_norm[0]
            + weights[2] * trans_state_norm[0]
        )
        c_blended = (
            weights[0] * lstm_state_norm[1]
            + weights[1] * gru_state_norm[1]
            + weights[2] * trans_state_norm[1]
        )

        return output, context, (h_blended, c_blended)

    def get_alphas(self) -> torch.Tensor:
        """Get normalized architecture weights"""
        return F.softmax(self.alphas, dim=0)

    def set_temperature(self, temp: float):
        """Set temperature for sampling"""
        self.temperature = temp


class AttentionBridge(nn.Module):
    """Unified attention bridge that adapts to different architectures"""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = min(num_heads, d_model)

        # Ensure divisibility
        while d_model % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.head_dim = d_model // self.num_heads
        self.scale = self.head_dim**-0.5

        # Unified attention components
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Input adaptation layers
        self.decoder_adapter = nn.Linear(d_model, d_model)
        self.encoder_adapter = nn.Linear(d_model, d_model)

        # Gating mechanism
        self.gate = nn.Linear(d_model * 2, d_model)  # Input: [decoder, attended]

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unified attention that works with both sequence and context inputs
        """
        B, L_dec, D = decoder_hidden.shape

        # Adapt decoder input
        decoder_adapted = self.decoder_adapter(decoder_hidden)

        # Determine encoder input (prefer full sequence over context)
        if encoder_output is not None:
            encoder_input = encoder_output
            L_enc = encoder_output.size(1)
        elif encoder_context is not None:
            # Expand context to create a pseudo-sequence
            encoder_input = encoder_context.expand(B, L_dec, D)
            L_enc = L_dec
        else:
            # No encoder input - return adapted decoder
            return decoder_adapted

        # Adapt encoder input
        encoder_adapted = self.encoder_adapter(encoder_input)

        # Compute attention
        q = self.q_proj(decoder_adapted)
        k = self.k_proj(encoder_adapted)
        v = self.v_proj(encoder_adapted)

        # Reshape for multi-head attention
        if self.num_heads > 1:
            q = q.view(B, L_dec, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, L_enc, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, L_enc, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attended = torch.matmul(attn_weights, v)
            attended = attended.transpose(1, 2).contiguous().view(B, L_dec, D)
        else:
            # Single-head attention (more efficient for small models)
            scores = torch.sum(q * k, dim=-1, keepdim=True) * self.scale
            attn_weights = F.softmax(scores, dim=1)
            attended = attn_weights * v

        attended = self.out_proj(attended)

        # Gated combination
        combined_input = torch.cat([decoder_hidden, attended], dim=-1)
        gate_weights = torch.sigmoid(self.gate(combined_input))

        output = gate_weights * attended + (1 - gate_weights) * decoder_hidden
        return output


class MixedDecoder(nn.Module):
    """Improved mixed decoder with better architecture compatibility"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_attention_bridge: bool = True,
        attention_layers: int = 1,  # For backward compatibility
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.temperature = temperature
        self.use_attention_bridge = use_attention_bridge
        self.attention_layers = attention_layers

        # Create decoders
        self.lstm = nn.LSTM(
            input_dim,
            latent_dim,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_dim,
            latent_dim,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )
        self.transformer = LightweightTransformerDecoder(
            input_dim=input_dim, latent_dim=latent_dim, num_layers=2, dropout=dropout
        )

        # Compatibility attributes
        self.decoders = nn.ModuleList([self.lstm, self.gru, self.transformer])
        self.decoder_names = ["lstm", "gru", "transformer"]
        self.rnn_names = ["lstm", "gru", "transformer"]

        # Architecture normalization
        self.normalizer = ArchitectureNormalizer(latent_dim)

        # Architecture selection parameters
        num_decoder_options = len(self.decoders)
        decoder_init = torch.zeros(num_decoder_options)
        decoder_init[random.randint(0, num_decoder_options - 1)] = 1.0
        self.register_parameter("alphas", nn.Parameter(decoder_init))

        # Unified attention bridge
        if use_attention_bridge:
            self.attention_bridge = AttentionBridge(
                latent_dim, num_heads=4, dropout=dropout
            )

            # Optional: attention choice parameters for backward compatibility
            # This might be what was causing the UnboundLocalError
            attention_init = torch.zeros(2)  # [use_attention, no_attention]
            attention_init[0] = 1.0  # Default to using attention
            self.register_parameter("attention_alphas", nn.Parameter(attention_init))

        self._init_weights()

    def _init_weights(self):
        """Initialize RNN weights"""
        for rnn in [self.lstm, self.gru]:
            for name, param in rnn.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.data.fill_(0)
                    if "lstm" in str(rnn.__class__).lower() and "bias_ih" in name:
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with improved compatibility"""
        batch_size = tgt.size(0)

        # FIX: More robust hidden state initialization
        if hidden_state is None:
            device, dtype = tgt.device, tgt.dtype
            h_0 = torch.zeros(
                2, batch_size, self.latent_dim, device=device, dtype=dtype
            )
            c_0 = torch.zeros(
                2, batch_size, self.latent_dim, device=device, dtype=dtype
            )
            lstm_state = (h_0.contiguous(), c_0.contiguous())
            gru_state = h_0.contiguous()
            trans_state = (h_0.contiguous(), c_0.contiguous())
        else:
            # FIX: Better state format handling
            if isinstance(hidden_state, tuple) and len(hidden_state) == 2:
                h, c = hidden_state
                # Ensure contiguous and proper device/dtype
                h = h.contiguous().to(device=tgt.device, dtype=tgt.dtype)
                c = c.contiguous().to(device=tgt.device, dtype=tgt.dtype)
                lstm_state = (h, c)
                gru_state = h
                trans_state = (h, c)
            else:
                h = hidden_state.contiguous().to(device=tgt.device, dtype=tgt.dtype)
                c = torch.zeros_like(h)
                lstm_state = (h, c)
                gru_state = h
                trans_state = (h, c)

        # Run all decoders
        lstm_out, lstm_new_state = self.lstm(tgt, lstm_state)
        gru_out, gru_new_state = self.gru(tgt, gru_state)

        # Handle transformer memory input more robustly
        if memory is not None:
            trans_out, trans_new_state = self.transformer(tgt, memory, trans_state)
        else:
            # Fallback: use encoder context as memory
            fallback_memory = (
                encoder_context
                if encoder_context is not None
                else torch.zeros_like(tgt[:, :1, :])
            )
            trans_out, trans_new_state = self.transformer(
                tgt, fallback_memory, trans_state
            )

        # Normalize outputs
        lstm_out_norm = self.normalizer.normalize_output(lstm_out, "lstm")
        gru_out_norm = self.normalizer.normalize_output(gru_out, "gru")
        trans_out_norm = self.normalizer.normalize_output(trans_out, "transformer")

        # Get architecture weights
        decoder_weights = F.gumbel_softmax(
            self.alphas, tau=self.temperature, hard=False, dim=0
        )

        # Apply attention if enabled
        if self.use_attention_bridge and (
            encoder_output is not None or encoder_context is not None
        ):
            # Determine whether to use attention based on parameters
            if hasattr(self, "attention_alphas"):
                attention_weights = F.gumbel_softmax(
                    self.attention_alphas, tau=self.temperature, hard=False, dim=0
                )
                use_attention_prob = attention_weights[
                    0
                ]  # Probability of using attention
            else:
                use_attention_prob = 1.0  # Always use attention if bridge is enabled

            # Apply attention to outputs
            lstm_attended = self.attention_bridge(
                lstm_out_norm, encoder_output, encoder_context
            )
            gru_attended = self.attention_bridge(
                gru_out_norm, encoder_output, encoder_context
            )
            trans_attended = self.attention_bridge(
                trans_out_norm, encoder_output, encoder_context
            )

            # Mix attended and non-attended outputs based on attention choice
            if hasattr(self, "attention_alphas"):
                lstm_final = (
                    use_attention_prob * lstm_attended
                    + (1 - use_attention_prob) * lstm_out_norm
                )
                gru_final = (
                    use_attention_prob * gru_attended
                    + (1 - use_attention_prob) * gru_out_norm
                )
                trans_final = (
                    use_attention_prob * trans_attended
                    + (1 - use_attention_prob) * trans_out_norm
                )
            else:
                lstm_final = lstm_attended
                gru_final = gru_attended
                trans_final = trans_attended

            # Weighted combination
            output = (
                decoder_weights[0] * lstm_final
                + decoder_weights[1] * gru_final
                + decoder_weights[2] * trans_final
            )
        else:
            # No attention - direct combination
            output = (
                decoder_weights[0] * lstm_out_norm
                + decoder_weights[1] * gru_out_norm
                + decoder_weights[2] * trans_out_norm
            )

        # Normalize and blend states
        lstm_state_norm = self.normalizer.normalize_state(lstm_new_state, "lstm")
        gru_state_norm = self.normalizer.normalize_state(gru_new_state, "gru")
        trans_state_norm = self.normalizer.normalize_state(
            trans_new_state, "transformer"
        )

        # Blend states
        h_blended = (
            decoder_weights[0] * lstm_state_norm[0]
            + decoder_weights[1] * gru_state_norm[0]
            + decoder_weights[2] * trans_state_norm[0]
        )
        c_blended = (
            decoder_weights[0] * lstm_state_norm[1]
            + decoder_weights[1] * gru_state_norm[1]
            + decoder_weights[2] * trans_state_norm[1]
        )

        return output, (h_blended, c_blended)

    def get_alphas(self) -> torch.Tensor:
        """Get architecture parameters for compatibility"""
        decoder_alphas = F.softmax(self.alphas, dim=0)

        if self.use_attention_bridge and hasattr(self, "attention_alphas"):
            attention_alphas = F.softmax(self.attention_alphas, dim=0)
            return torch.cat([decoder_alphas, attention_alphas])

        return decoder_alphas

    def set_temperature(self, temp: float):
        """Set temperature for sampling"""
        self.temperature = temp


# Helper function for backward compatibility
def blend_states(w, states):
    """Improved state blending with proper normalization"""
    normalizer = ArchitectureNormalizer(
        states[0][0].size(-1) if isinstance(states[0], tuple) else states[0].size(-1)
    )

    normalized_states = []
    arch_types = ["lstm", "gru", "transformer"]

    for i, state in enumerate(states):
        arch_type = arch_types[i] if i < len(arch_types) else "lstm"
        normalized_state = normalizer.normalize_state(state, arch_type)
        normalized_states.append(normalized_state)

    # Blend normalized states
    h = sum(w[i] * normalized_states[i][0] for i in range(len(normalized_states)))
    c = sum(w[i] * normalized_states[i][1] for i in range(len(normalized_states)))
    return (h, c)


class ArchitectureConverter:
    """Utility class for converting between mixed and fixed architectures"""

    @staticmethod
    def get_best_architecture(alphas: torch.Tensor) -> str:
        """Get the best architecture from alpha weights"""
        arch_names = ["lstm", "gru", "transformer"]
        best_idx = torch.argmax(
            alphas[:3]
        ).item()  # Only consider first 3 (decoder types)
        return arch_names[best_idx]

    @staticmethod
    def ensure_proper_state_format(
        state,
        rnn_type: str,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
    ):
        """Ensure hidden state has proper format for the given RNN type"""
        if state is None:
            # Create zero states
            h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            c = torch.zeros(num_layers, batch_size, hidden_size, device=device)

            if rnn_type == "gru":
                return h
            elif rnn_type == "lstm":
                return (h, c)
            else:  # transformer
                return (h, c)

        if rnn_type == "gru":
            if isinstance(state, tuple):
                h = state[0]
            else:
                h = state

            # Ensure correct dimensions
            if h.dim() == 2:
                h = h.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
            elif h.dim() == 3:
                h = h.contiguous()

            return h

        elif rnn_type == "lstm":
            if isinstance(state, tuple):
                h, c = state
            else:
                h = state
                c = torch.zeros_like(h)

            # Ensure correct dimensions
            if h.dim() == 2:
                h = h.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
                c = c.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
            elif h.dim() == 3:
                h = h.contiguous()
                c = c.contiguous()

            return (h, c)
        else:  # transformer
            if isinstance(state, tuple):
                h, c = state
            else:
                h = state
                c = torch.zeros_like(h)

            # Ensure correct dimensions
            if h.dim() == 2:
                h = h.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
                c = c.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
            elif h.dim() == 3:
                h = h.contiguous()
                c = c.contiguous()

            return (h, c)

    @staticmethod
    def get_attention_choice(mixed_decoder) -> str:
        """Get the best attention choice from mixed decoder"""
        if (
            not hasattr(mixed_decoder, "use_attention_bridge")
            or not mixed_decoder.use_attention_bridge
        ):
            return "no_attention"

        if hasattr(mixed_decoder, "attention_alphas"):
            attention_weights = F.softmax(mixed_decoder.attention_alphas, dim=0)
            if len(attention_weights) >= 2:
                # Assume format is [use_attention, no_attention]
                return (
                    "attention"
                    if attention_weights[0] > attention_weights[1]
                    else "no_attention"
                )

        return "attention"  # Default to using attention

    @staticmethod
    def fix_mixed_weights(mixed_model, temperature: float = 0.01):
        """Fix mixed model to use best architecture by setting alphas"""
        with torch.no_grad():
            # Get current best architecture
            alphas = mixed_model.get_alphas()
            best_idx = torch.argmax(alphas[:3])

            # Create one-hot alphas
            new_alphas = torch.zeros_like(mixed_model.alphas)
            new_alphas[best_idx] = 1.0
            mixed_model.alphas.data = new_alphas

            # Set very low temperature for sharp selection
            mixed_model.set_temperature(temperature)

            # Fix attention alphas if they exist
            if hasattr(mixed_model, "attention_alphas"):
                attention_best = torch.argmax(mixed_model.attention_alphas)
                new_attention_alphas = torch.zeros_like(mixed_model.attention_alphas)
                new_attention_alphas[attention_best] = 1.0
                mixed_model.attention_alphas.data = new_attention_alphas

    @staticmethod
    def derive_architecture_safely(mixed_model, device=None):
        """Safely derive architecture from mixed model with proper error handling"""
        if device is None:
            device = next(mixed_model.parameters()).device

        try:
            # Handle encoder
            if hasattr(mixed_model, "encoder") or hasattr(
                mixed_model, "forecast_encoder"
            ):
                encoder = getattr(mixed_model, "encoder", None) or getattr(
                    mixed_model, "forecast_encoder", None
                )
                if encoder and hasattr(encoder, "get_alphas"):
                    encoder_alphas = encoder.get_alphas()
                    best_encoder_type = ArchitectureConverter.get_best_architecture(
                        encoder_alphas
                    )

                    # Create fixed encoder
                    fixed_encoder = ArchitectureConverter.create_fixed_encoder(encoder)

                    # Replace in model
                    if hasattr(mixed_model, "encoder"):
                        mixed_model.encoder = fixed_encoder
                    if hasattr(mixed_model, "forecast_encoder"):
                        mixed_model.forecast_encoder = fixed_encoder

            # Handle decoder
            if hasattr(mixed_model, "decoder") or hasattr(
                mixed_model, "forecast_decoder"
            ):
                decoder = getattr(mixed_model, "decoder", None) or getattr(
                    mixed_model, "forecast_decoder", None
                )
                if decoder and hasattr(decoder, "get_alphas"):
                    decoder_alphas = decoder.get_alphas()
                    best_decoder_type = ArchitectureConverter.get_best_architecture(
                        decoder_alphas
                    )
                    attention_choice = ArchitectureConverter.get_attention_choice(
                        decoder
                    )

                    # Create fixed decoder
                    fixed_decoder = ArchitectureConverter.create_fixed_decoder(
                        decoder, use_attention_bridge=(attention_choice == "attention")
                    )

                    # Replace in model
                    if hasattr(mixed_model, "decoder"):
                        mixed_model.decoder = fixed_decoder
                    if hasattr(mixed_model, "forecast_decoder"):
                        mixed_model.forecast_decoder = fixed_decoder

            return mixed_model

        except Exception as e:
            print(f"Warning: Could not derive architecture safely: {e}")
            print("Falling back to fixing weights only...")

            # Fallback: just fix the weights
            if hasattr(mixed_model, "encoder") or hasattr(
                mixed_model, "forecast_encoder"
            ):
                encoder = getattr(mixed_model, "encoder", None) or getattr(
                    mixed_model, "forecast_encoder", None
                )
                if encoder:
                    ArchitectureConverter.fix_mixed_weights(encoder)

            if hasattr(mixed_model, "decoder") or hasattr(
                mixed_model, "forecast_decoder"
            ):
                decoder = getattr(mixed_model, "decoder", None) or getattr(
                    mixed_model, "forecast_decoder", None
                )
                if decoder:
                    ArchitectureConverter.fix_mixed_weights(decoder)

            return mixed_model

    @staticmethod
    def create_fixed_encoder(mixed_encoder, **kwargs) -> "FixedEncoder":
        """Create a FixedEncoder from a MixedEncoder"""
        best_type = ArchitectureConverter.get_best_architecture(
            mixed_encoder.get_alphas()
        )

        # Create fixed encoder
        fixed_encoder = FixedEncoder(
            rnn_type=best_type,
            input_dim=mixed_encoder.input_dim,
            latent_dim=mixed_encoder.latent_dim,
            **kwargs,
        )

        # Transfer weights
        ArchitectureConverter._transfer_encoder_weights(
            mixed_encoder, fixed_encoder, best_type
        )
        return fixed_encoder

    @staticmethod
    def create_fixed_decoder(mixed_decoder, **kwargs) -> "FixedDecoder":
        """Create a FixedDecoder from a MixedDecoder"""
        best_type = ArchitectureConverter.get_best_architecture(
            mixed_decoder.get_alphas()
        )

        # Create fixed decoder
        fixed_decoder = FixedDecoder(
            rnn_type=best_type,
            input_dim=mixed_decoder.input_dim,
            latent_dim=mixed_decoder.latent_dim,
            use_attention_bridge=kwargs.get(
                "use_attention_bridge", mixed_decoder.use_attention_bridge
            ),
            **{k: v for k, v in kwargs.items() if k != "use_attention_bridge"},
        )

        # Transfer weights
        ArchitectureConverter._transfer_decoder_weights(
            mixed_decoder, fixed_decoder, best_type
        )
        return fixed_decoder

    @staticmethod
    def _transfer_encoder_weights(mixed_encoder, fixed_encoder, arch_type: str):
        """Transfer weights from mixed to fixed encoder"""
        try:
            # Transfer the specific architecture weights
            if arch_type == "lstm":
                source_rnn = mixed_encoder.lstm
            elif arch_type == "gru":
                source_rnn = mixed_encoder.gru
            elif arch_type == "transformer":
                source_rnn = mixed_encoder.transformer
            else:
                raise ValueError(f"Unknown architecture type: {arch_type}")

            # Copy state dict
            fixed_encoder.rnn.load_state_dict(source_rnn.state_dict())
        except Exception as e:
            print(f"Warning: Could not transfer encoder weights: {e}")

    @staticmethod
    def _transfer_decoder_weights(mixed_decoder, fixed_decoder, arch_type: str):
        """Transfer weights from mixed to fixed decoder"""
        try:
            # Transfer the specific architecture weights
            if arch_type == "lstm":
                source_rnn = mixed_decoder.lstm
            elif arch_type == "gru":
                source_rnn = mixed_decoder.gru
            elif arch_type == "transformer":
                source_rnn = mixed_decoder.transformer
            else:
                raise ValueError(f"Unknown architecture type: {arch_type}")

            # Copy state dict
            fixed_decoder.rnn.load_state_dict(source_rnn.state_dict())

            # Transfer attention bridge weights if present
            if (
                fixed_decoder.use_attention_bridge
                and hasattr(mixed_decoder, "attention_bridge")
                and hasattr(fixed_decoder, "attention_bridge")
            ):
                try:
                    fixed_decoder.attention_bridge.load_state_dict(
                        mixed_decoder.attention_bridge.state_dict()
                    )
                except Exception as e:
                    print(f"Warning: Could not transfer attention bridge weights: {e}")

        except Exception as e:
            print(f"Warning: Could not transfer decoder weights: {e}")


class FixedEncoder(nn.Module):
    """Simple fixed encoder wrapper for deployment"""

    def __init__(
        self,
        rnn=None,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        rnn_or_type = rnn if rnn is not None else rnn_type
        if rnn_or_type is None:
            raise ValueError("Either 'rnn' or 'rnn_type' must be provided")

        if isinstance(rnn_or_type, str):
            self.rnn_type = rnn_or_type.lower()
            self.latent_dim = latent_dim

            if self.rnn_type == "lstm":
                self.rnn = nn.LSTM(
                    input_dim,
                    latent_dim,
                    num_layers,
                    dropout=dropout if dropout > 0 else 0,
                    batch_first=True,
                )
            elif self.rnn_type == "gru":
                self.rnn = nn.GRU(
                    input_dim,
                    latent_dim,
                    num_layers,
                    dropout=dropout if dropout > 0 else 0,
                    batch_first=True,
                )
            elif self.rnn_type == "transformer":
                self.rnn = LightweightTransformerEncoder(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            self.rnn = rnn_or_type
            if isinstance(self.rnn, nn.LSTM):
                self.rnn_type = "lstm"
                self.latent_dim = self.rnn.hidden_size
            elif isinstance(self.rnn, nn.GRU):
                self.rnn_type = "gru"
                self.latent_dim = self.rnn.hidden_size
            elif hasattr(self.rnn, "latent_dim"):
                self.rnn_type = "transformer"
                self.latent_dim = self.rnn.latent_dim
            else:
                self.rnn_type = "unknown"
                self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass optimized for single architecture"""
        if self.rnn_type == "transformer":
            return self.rnn(x)  # Returns (output, context, state)
        else:
            output, state = self.rnn(x)
            context = output[:, -1:, :]  # Last timestep

            # Ensure state format is consistent
            if isinstance(self.rnn, nn.GRU):
                # GRU returns [num_layers, batch, hidden_size]
                # Convert to tuple format: (h, c) where c is zeros
                h = state
                c = torch.zeros_like(h)
                state = (h, c)
            elif isinstance(self.rnn, nn.LSTM):
                # LSTM already returns (h, c) in correct format
                pass

            return output, context, state

    def get_alphas(self) -> torch.Tensor:
        """Return one-hot encoding for the fixed architecture"""
        device = next(self.parameters()).device
        if self.rnn_type == "lstm":
            return torch.tensor([1.0, 0.0, 0.0], device=device)
        elif self.rnn_type == "gru":
            return torch.tensor([0.0, 1.0, 0.0], device=device)
        else:  # transformer
            return torch.tensor([0.0, 0.0, 1.0], device=device)

    def set_temperature(self, temp: float):
        """No-op for fixed encoder"""


class FixedDecoder(nn.Module):
    """Simple fixed decoder wrapper for deployment"""

    def __init__(
        self,
        rnn=None,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_attention_bridge: bool = False,
        attention_layers: int = 1,
    ):
        super().__init__()
        self.use_attention_bridge = use_attention_bridge

        rnn_or_type = rnn if rnn is not None else rnn_type
        if rnn_or_type is None:
            raise ValueError("Either 'rnn' or 'rnn_type' must be provided")

        if isinstance(rnn_or_type, str):
            self.rnn_type = rnn_or_type.lower()
            self.latent_dim = latent_dim

            if self.rnn_type == "lstm":
                self.rnn = nn.LSTM(
                    input_dim,
                    latent_dim,
                    num_layers,
                    dropout=dropout if dropout > 0 else 0,
                    batch_first=True,
                )
            elif self.rnn_type == "gru":
                self.rnn = nn.GRU(
                    input_dim,
                    latent_dim,
                    num_layers,
                    dropout=dropout if dropout > 0 else 0,
                    batch_first=True,
                )
            elif self.rnn_type == "transformer":
                self.rnn = LightweightTransformerDecoder(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            self.rnn = rnn_or_type
            if isinstance(self.rnn, nn.LSTM):
                self.rnn_type = "lstm"
                self.latent_dim = self.rnn.hidden_size
            elif isinstance(self.rnn, nn.GRU):
                self.rnn_type = "gru"
                self.latent_dim = self.rnn.hidden_size
            elif hasattr(self.rnn, "latent_dim"):
                self.rnn_type = "transformer"
                self.latent_dim = self.rnn.latent_dim
            else:
                self.rnn_type = "unknown"
                self.latent_dim = latent_dim

        # Simple attention bridge for fixed decoder
        if use_attention_bridge:
            self.attention_bridge = AttentionBridge(
                latent_dim, num_heads=4, dropout=dropout
            )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: torch.Tensor = None,
    ) -> tuple:
        """Forward pass optimized for single architecture"""
        batch_size = tgt.size(0)

        # Get RNN parameters
        num_layers = getattr(self.rnn, "num_layers", 1)
        hidden_size = getattr(self.rnn, "hidden_size", self.latent_dim)

        # Ensure proper hidden state format
        hidden_state = ArchitectureConverter.ensure_proper_state_format(
            hidden_state, self.rnn_type, num_layers, batch_size, hidden_size, tgt.device
        )

        # Forward pass
        if self.rnn_type == "transformer":
            output, new_state = self.rnn(tgt, memory, hidden_state)
        else:
            output, new_state = self.rnn(tgt, hidden_state)

        # Apply attention if enabled
        if self.use_attention_bridge and hasattr(self, "attention_bridge"):
            attention_source = encoder_output if encoder_output is not None else memory
            if attention_source is not None:
                output = self.attention_bridge(output, attention_source)

        return output, new_state

    def get_alphas(self) -> torch.Tensor:
        """Return one-hot encoding for the fixed architecture"""
        device = next(self.parameters()).device
        if self.rnn_type == "lstm":
            return torch.tensor([1.0, 0.0, 0.0], device=device)
        elif self.rnn_type == "gru":
            return torch.tensor([0.0, 1.0, 0.0], device=device)
        else:  # transformer
            return torch.tensor([0.0, 0.0, 1.0], device=device)

    def set_temperature(self, temp: float):
        """No-op for fixed decoder"""


class RotaryPositionalEncoding(nn.Module):
    """Streamlined rotary positional encoding with efficient caching"""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute and cache frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute embeddings for common sequence lengths
        self._init_cached_embeddings(max_seq_len)

    def _init_cached_embeddings(self, max_len: int):
        """Pre-compute embeddings for efficiency"""
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cached_cos", emb.cos(), persistent=False)
        self.register_buffer("cached_sin", emb.sin(), persistent=False)

    def _compute_embeddings(self, seq_len: int, device: torch.device) -> tuple:
        """Compute embeddings on-the-fly for longer sequences"""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, seq_len: int, device: torch.device = None) -> tuple:
        """Generate cos and sin embeddings with efficient caching"""
        if device is None:
            device = self.inv_freq.device

        if seq_len <= self.cached_cos.size(0):
            # Use cached embeddings
            cos = self.cached_cos[:seq_len]
            sin = self.cached_sin[:seq_len]

            # Only move to device if necessary
            if cos.device != device:
                cos = cos.to(device)
                sin = sin.to(device)

            return cos, sin
        else:
            # Compute on-the-fly for longer sequences
            return self._compute_embeddings(seq_len, device)

    def apply_rotary_pos_emb(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary positional embedding efficiently"""
        seq_len = x.size(-2)
        head_dim = x.size(-1)

        # Handle odd head dimensions gracefully
        if head_dim % 2 != 0:
            raise ValueError(
                f"Head dimension {head_dim} must be even for rotary embeddings"
            )

        half_dim = head_dim // 2

        # Ensure dimensions match and reshape for broadcasting
        cos = cos[:seq_len, :half_dim].view(1, 1, seq_len, half_dim)
        sin = sin[:seq_len, :half_dim].view(1, 1, seq_len, half_dim)

        # Split and apply rotation in one operation
        x_even, x_odd = x.chunk(2, dim=-1)
        return torch.cat(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        )

    def extend_cache(self, new_max_len: int):
        """Extend cached embeddings to new maximum length"""
        if new_max_len > self.max_seq_len:
            self.max_seq_len = new_max_len
            self._init_cached_embeddings(new_max_len)

    def get_embeddings_for_length(
        self, seq_len: int, device: torch.device = None
    ) -> tuple:
        """Convenient method to get embeddings for specific length"""
        return self.forward(seq_len, device)


class PositionalEncoding(nn.Module):
    """Optimized positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model

        # Ensure d_model is even for proper sin/cos pairing
        if d_model % 2 != 0:
            raise ValueError(f"d_model {d_model} must be even for positional encoding")

        # Vectorized computation of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # More numerically stable computation
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(base) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as non-persistent buffer (won't be saved in checkpoints)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding with automatic device placement"""
        seq_len = x.size(1)

        # Handle sequences longer than max_len gracefully
        if seq_len > self.pe.size(1):
            # Extend PE on-the-fly for longer sequences
            pe_extended = self._compute_extended_pe(seq_len, x.device, x.dtype)
            return x + pe_extended[:, :seq_len]

        # Use cached PE
        pe_slice = self.pe[:, :seq_len]

        # Ensure correct device and dtype
        if pe_slice.device != x.device or pe_slice.dtype != x.dtype:
            pe_slice = pe_slice.to(device=x.device, dtype=x.dtype)

        return x + pe_slice

    def _compute_extended_pe(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Compute positional encoding for sequences longer than max_len"""
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, seq_len, dtype=dtype, device=device).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=dtype, device=device)
            * -(math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def extend_max_len(self, new_max_len: int):
        """Extend the maximum length of cached positional encodings"""
        if new_max_len > self.pe.size(1):
            device = self.pe.device
            dtype = self.pe.dtype
            new_pe = self._compute_extended_pe(new_max_len, device, dtype)
            self.register_buffer("pe", new_pe, persistent=False)

