import copy
import math
import warnings
from typing import Dict, List, Optional, Tuple

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


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        mid = dim * expand
        self.w1 = nn.Linear(dim, mid, bias=False)
        self.w2 = nn.Linear(dim, mid, bias=False)
        self.w3 = nn.Linear(mid, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LinearSelfAttention(nn.Module):
    """Improved linear self-attention with consistent behavior"""

    def __init__(
        self,
        dim,
        heads=4,
        dropout=0.0,
        causal=False,
        rope_base: float = 500000.0,
        rope_max_seq_len: int = 1024,
    ):
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

        # Always enable RoPE; odd dimensions are handled via partial rotation.
        self.rotary_emb = RotaryPositionalEncoding(
            self.head_dim,
            max_seq_len=rope_max_seq_len,
            base=rope_base,
        )

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.size(-2)
        cos, sin = self.rotary_emb.get_embeddings_for_length(seq_len, q.device)
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q = self.rotary_emb.apply_rotary_pos_emb(q, cos, sin)
        k = self.rotary_emb.apply_rotary_pos_emb(k, cos, sin)
        return q, k

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        # Non-negative kernel feature map for linear attention.
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads

        # More efficient reshape pattern
        qkv = self.to_qkv(x).view(B, T, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, T, head_dim]
        q, k = self._apply_rotary(q, k)

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
            # Kernelized linear attention: phi(q) * (phi(k)^T v) / (phi(q) * sum(phi(k))).
            q_feat = self._feature_map(q * self.scale)
            k_feat = self._feature_map(k)

            kv = torch.einsum("bhtd,bhtv->bhdv", k_feat, v)
            k_sum = k_feat.sum(dim=2)
            denom = torch.einsum("bhtd,bhd->bht", q_feat, k_sum).clamp_min(1e-6)
            out = torch.einsum("bhtd,bhdv->bhtv", q_feat, kv) / denom.unsqueeze(-1)

        # Reshape back and project
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        if self.training and self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p)

        return out


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
        rope_base: float = 500000.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.causal = causal

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        # Transformer layers  # position handled by RoPE inside LinearSelfAttention
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": LinearSelfAttention(
                            latent_dim,
                            heads=nhead,
                            dropout=dropout,
                            causal=causal,
                            rope_base=rope_base,
                            rope_max_seq_len=max_seq_len,
                        ),
                        "ffn": SwiGLUFFN(latent_dim, expand=4),
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

        # Process through transformer layers
        for layer in self.layers:
            # Pre-norm self-attention + residual
            attn_out = layer["self_attn"](layer["norm1"](x))
            if self.training and self.dropout_p > 0:
                attn_out = F.dropout(attn_out, p=self.dropout_p)
            x = x + attn_out

            # Pre-norm FFN + residual
            ffn_out = layer["ffn"](layer["norm2"](x))
            if self.training and self.dropout_p > 0:
                ffn_out = F.dropout(ffn_out, p=self.dropout_p)
            x = x + ffn_out

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
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

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
        rope_base: float = 500000.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.causal = causal

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)
        # position handled by RoPE inside LinearSelfAttention

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": LinearSelfAttention(
                            latent_dim,
                            heads=nhead,
                            dropout=dropout,
                            causal=causal,
                            rope_base=rope_base,
                            rope_max_seq_len=max_seq_len,
                        ),
                        "cross_attn": nn.MultiheadAttention(
                            latent_dim,
                            nhead,
                            dropout=dropout,
                            batch_first=True,
                            bias=False,
                        ),
                        "ffn": SwiGLUFFN(latent_dim, expand=4),
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

        # Prepare memory
        memory = self._prepare_memory(memory_or_hidden)

        # Process through decoder layers
        for layer in self.layers:
            # Pre-norm self-attention + residual
            self_attn_out = layer["self_attn"](layer["norm1"](tgt))
            if self.training and self.dropout_p > 0:
                self_attn_out = F.dropout(self_attn_out, p=self.dropout_p)
            tgt = tgt + self_attn_out

            # Cross-attention (pre-norm via norm2 + residual)
            if memory is not None:
                try:
                    cross_out, _ = layer["cross_attn"](
                        layer["norm2"](tgt), memory, memory
                    )
                    if self.training and self.dropout_p > 0:
                        cross_out = F.dropout(cross_out, p=self.dropout_p)
                    tgt = tgt + cross_out
                except (RuntimeError, ValueError):
                    pass  # skip cross-attention if shapes don't match

            # FFN (pre-norm via norm3 + residual)
            ffn_out = layer["ffn"](layer["norm3"](tgt))
            if self.training and self.dropout_p > 0:
                ffn_out = F.dropout(ffn_out, p=self.dropout_p)
            tgt = tgt + ffn_out

        tgt = self.final_norm(tgt)

        # Create RNN-compatible state
        last_token = tgt[:, -1]  # [B, D]
        state_proj = self.state_proj(last_token)  # [B, 2*D]
        h_state, c_state = state_proj.chunk(2, dim=-1)

        # Reshape to RNN format
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

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


class SearchableDecomposition(nn.Module):
    """Searchable decomposition front-end for trend/seasonal mixing."""

    def __init__(self, c_in: int, kernel_size: int = 25):
        super().__init__()
        self.c_in = int(c_in)
        self.kernel_size = int(max(3, kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        # [none, moving_avg_trend, seasonal_residual, learnable_filter]
        self.alpha_logits = nn.Parameter(torch.randn(4) * 0.01)
        pad = self.kernel_size // 2
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        self.learnable_filter = nn.Conv1d(
            self.c_in,
            self.c_in,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.c_in,
            bias=False,
        )
        nn.init.dirac_(self.learnable_filter.weight)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if x.dim() != 3:
            return x

        tau = max(float(temperature), 1e-3)
        weights = F.softmax(self.alpha_logits / tau, dim=0)

        x_t = x.transpose(1, 2)
        trend = self.avg_pool(x_t).transpose(1, 2)
        seasonal = x - trend
        filtered = self.learnable_filter(x_t).transpose(1, 2)

        return (
            weights[0] * x
            + weights[1] * trend
            + weights[2] * seasonal
            + weights[3] * filtered
        )

    def get_alphas(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=0)


class SequenceStateAdapter:
    """Shared hidden-state adapter for mixed/fixed encoder-decoder blocks."""

    @staticmethod
    def _extract_tensor_dtype(
        state, fallback: torch.dtype = torch.float32
    ) -> torch.dtype:
        if (
            isinstance(state, tuple)
            and len(state) > 0
            and isinstance(state[0], torch.Tensor)
        ):
            return state[0].dtype
        if isinstance(state, torch.Tensor):
            return state.dtype
        return fallback

    @staticmethod
    def _coerce_pair_state(
        state,
        *,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = torch.zeros(
                num_layers, batch_size, hidden_size, device=device, dtype=dtype
            )
            c = torch.zeros(
                num_layers, batch_size, hidden_size, device=device, dtype=dtype
            )
            return h, c

        if isinstance(state, tuple) and len(state) == 2:
            h, c = state
        else:
            h = state
            c = torch.zeros_like(h)

        h = h.to(device=device, dtype=dtype).contiguous()
        c = c.to(device=device, dtype=dtype).contiguous()

        if h.dim() == 2:
            h = h.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
            c = c.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        return h, c

    @staticmethod
    def ensure_rnn_state(
        state,
        rnn_type: str,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ):
        state_dtype = dtype or SequenceStateAdapter._extract_tensor_dtype(state)
        h, c = SequenceStateAdapter._coerce_pair_state(
            state,
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
            dtype=state_dtype,
        )

        if rnn_type == "gru":
            return h
        return (h, c)

    @staticmethod
    def split_mixed_decoder_states(
        hidden_state,
        *,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        pair_state = SequenceStateAdapter.ensure_rnn_state(
            hidden_state,
            rnn_type="lstm",
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
        )
        h, c = pair_state
        lstm_state = (h, c)
        gru_state = h
        trans_state = (h, c)
        return lstm_state, gru_state, trans_state


class BaseMixedSequenceBlock(nn.Module):
    """Mother class for mixed encoder/decoder blocks."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        num_layers: int = 2,
        num_options: int = 3,
        single_path_search: bool = True,
        arch_path_keep_prob: float = 0.85,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.temperature = temperature
        self.num_layers = num_layers
        self.single_path_search = single_path_search
        self.arch_path_keep_prob = float(min(max(arch_path_keep_prob, 0.0), 1.0))
        self._warned_no_grad_sampling = False

        self.lstm = nn.LSTM(
            input_dim,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_dim,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )

        init = 0.01 * torch.randn(num_options)
        self.register_parameter("alphas", nn.Parameter(init))
        layer_offsets = 0.01 * torch.randn(num_layers, num_options)
        self.register_parameter("layer_alpha_offsets", nn.Parameter(layer_offsets))
        self._init_rnn_weights()

    def _init_rnn_weights(self):
        """Initialize shared RNN weights."""
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

    def _get_layer_arch_logits(self) -> torch.Tensor:
        return self.alphas.unsqueeze(0) + self.layer_alpha_offsets

    def _should_use_stochastic_arch_sampling(self) -> bool:
        """Only sample stochastically when training with gradients enabled."""
        if not self.training:
            return False
        if torch.is_grad_enabled():
            return True
        if self.single_path_search and not self._warned_no_grad_sampling:
            warnings.warn(
                "Single-path architecture sampling requested while model is in train() "
                "but gradients are disabled. Falling back to deterministic softmax. "
                "Call eval() for inference to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_no_grad_sampling = True
        return False

    @staticmethod
    def _sample_straight_through_gumbel(
        logits: torch.Tensor, tau: float, dim: int = -1
    ) -> torch.Tensor:
        soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=dim)
        hard_idx = soft.argmax(dim=dim, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(dim, hard_idx, 1.0)
        return hard - soft.detach() + soft

    def _get_arch_weights(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Get differentiable per-layer architecture weights."""
        logits = self._get_layer_arch_logits()
        tau = max(float(self.temperature), 1e-3)
        if self._should_use_stochastic_arch_sampling():
            if self.single_path_search:
                weights = self._sample_straight_through_gumbel(logits, tau=tau, dim=-1)
            else:
                weights = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            weights = F.softmax(logits / tau, dim=-1)

        if (
            self.training
            and self.single_path_search
            and torch.is_grad_enabled()
            and self.arch_path_keep_prob < 1.0
        ):
            keep = torch.full_like(weights, self.arch_path_keep_prob)
            mask = torch.bernoulli(keep)
            if mask.dim() == 2:
                zero_rows = mask.sum(dim=-1, keepdim=True) == 0
                if zero_rows.any():
                    top_idx = weights.argmax(dim=-1, keepdim=True)
                    for r in torch.where(zero_rows.squeeze(-1))[0].tolist():
                        mask[r, top_idx[r, 0]] = 1.0
            elif mask.dim() == 1 and mask.sum().item() == 0:
                mask[weights.argmax(dim=-1)] = 1.0

            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if layer_idx is None:
            return weights

        layer_idx = max(0, min(int(layer_idx), weights.size(0) - 1))
        return weights[layer_idx]

    def _get_output_arch_weights(self) -> torch.Tensor:
        """Use top-layer weights to mix sequence outputs."""
        return self._get_arch_weights(layer_idx=self.num_layers - 1)

    def get_layer_alphas(self) -> torch.Tensor:
        return F.softmax(self._get_layer_arch_logits(), dim=-1)

    def get_alphas(self) -> torch.Tensor:
        # Aggregate view used by existing logging/finalization paths.
        return self.get_layer_alphas().mean(dim=0)

    def set_temperature(self, temp: float):
        self.temperature = max(float(temp), 1e-3)

    def orthogonal_regularization(self) -> torch.Tensor:
        """Orthogonal regularization over recurrent hidden-to-hidden matrices."""
        reg = None
        for rnn in [self.lstm, self.gru]:
            for name, param in rnn.named_parameters():
                if "weight_hh" not in name or param.dim() != 2:
                    continue

                rows, cols = param.shape
                if cols <= 0:
                    continue

                if rows % cols == 0:
                    gates = rows // cols
                    chunks = param.view(gates, cols, cols)
                    for mat in chunks:
                        gram = mat @ mat.t()
                        eye = torch.eye(cols, device=mat.device, dtype=mat.dtype)
                        term = (gram - eye).pow(2).mean()
                        reg = term if reg is None else reg + term
                else:
                    gram = param @ param.t()
                    eye = torch.eye(rows, device=param.device, dtype=param.dtype)
                    term = (gram - eye).pow(2).mean()
                    reg = term if reg is None else reg + term

        if reg is None:
            ref = self.alphas
            return ref.new_zeros(())
        return reg


class BaseFixedSequenceBlock(nn.Module):
    """Mother class for fixed encoder/decoder blocks."""

    def __init__(
        self,
        rnn=None,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        transformer_factory=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        rnn_or_type = rnn if rnn is not None else rnn_type
        if rnn_or_type is None:
            raise ValueError("Either 'rnn' or 'rnn_type' must be provided")

        if isinstance(rnn_or_type, str):
            self.rnn_type = rnn_or_type.lower()
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
                if transformer_factory is None:
                    raise ValueError(
                        "transformer_factory must be provided for transformer type"
                    )
                self.rnn = transformer_factory(
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

    def get_alphas(self) -> torch.Tensor:
        """Return one-hot encoding for the fixed architecture."""
        device = next(self.parameters()).device
        if self.rnn_type == "lstm":
            return torch.tensor([1.0, 0.0, 0.0], device=device)
        elif self.rnn_type == "gru":
            return torch.tensor([0.0, 1.0, 0.0], device=device)
        return torch.tensor([0.0, 0.0, 1.0], device=device)

    def set_temperature(self, temp: float):
        """No-op for fixed architecture."""


class MixedEncoder(BaseMixedSequenceBlock):
    """Improved mixed encoder with better compatibility"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        single_path_search: bool = True,
        arch_path_keep_prob: float = 0.85,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            dropout=dropout,
            temperature=temperature,
            num_layers=2,
            num_options=3,
            single_path_search=single_path_search,
            arch_path_keep_prob=arch_path_keep_prob,
        )
        self.searchable_decomp = SearchableDecomposition(c_in=input_dim)

        self.transformer = LightweightTransformerEncoder(
            input_dim=input_dim, latent_dim=latent_dim, num_layers=2, dropout=dropout
        )

        self.encoders = nn.ModuleList([self.lstm, self.gru, self.transformer])
        self.encoder_names = ["lstm", "gru", "transformer"]
        # Normalization and compatibility
        self.normalizer = ArchitectureNormalizer(latent_dim)

        # Context projection to ensure consistent format
        self.context_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with improved compatibility"""
        x = self.searchable_decomp(x, temperature=self.temperature)
        layer_weights = self._get_arch_weights()
        output_weights = self._get_output_arch_weights()

        single_path_active = self.training and self.single_path_search
        selected_output_idx = int(torch.argmax(output_weights.detach()).item())
        selected_layer_idx = torch.argmax(layer_weights.detach(), dim=-1)

        if single_path_active:
            required_arches = {selected_output_idx}
            required_arches.update(int(idx.item()) for idx in selected_layer_idx)
        else:
            required_arches = {0, 1, 2}

        output_by_arch = {}
        context_by_arch = {}
        state_by_arch = {}

        if 0 in required_arches:
            lstm_out, lstm_state = self.lstm(x)
            output_by_arch[0] = self.normalizer.normalize_output(lstm_out, "lstm")
            context_by_arch[0] = lstm_out[:, -1:, :]
            state_by_arch[0] = self.normalizer.normalize_state(lstm_state, "lstm")

        if 1 in required_arches:
            gru_out, gru_state = self.gru(x)
            output_by_arch[1] = self.normalizer.normalize_output(gru_out, "gru")
            context_by_arch[1] = gru_out[:, -1:, :]
            state_by_arch[1] = self.normalizer.normalize_state(gru_state, "gru")

        if 2 in required_arches:
            trans_out, trans_ctx, trans_state = self.transformer(x)
            output_by_arch[2] = self.normalizer.normalize_output(
                trans_out, "transformer"
            )
            context_by_arch[2] = trans_ctx
            state_by_arch[2] = self.normalizer.normalize_state(
                trans_state, "transformer"
            )

        if single_path_active:
            selected_output_weight = output_weights[selected_output_idx]
            output = selected_output_weight * output_by_arch[selected_output_idx]
            context = selected_output_weight * context_by_arch[selected_output_idx]
        else:
            output = sum(output_weights[i] * output_by_arch[i] for i in range(3))
            context = sum(output_weights[i] * context_by_arch[i] for i in range(3))

        context = self.context_proj(context)

        if single_path_active:
            h_blended = torch.zeros(
                self.num_layers,
                x.size(0),
                self.latent_dim,
                device=x.device,
                dtype=output.dtype,
            )
            c_blended = torch.zeros_like(h_blended)
            for layer_idx in range(self.num_layers):
                arch_idx = int(selected_layer_idx[layer_idx].item())
                layer_weight = layer_weights[layer_idx, arch_idx]
                h_blended[layer_idx] = (
                    layer_weight * state_by_arch[arch_idx][0][layer_idx]
                )
                c_blended[layer_idx] = (
                    layer_weight * state_by_arch[arch_idx][1][layer_idx]
                )
        else:
            layer_weights_expanded = layer_weights.unsqueeze(-1).unsqueeze(
                -1
            )  # [L, 3, 1, 1]
            h_stack = torch.stack(
                [state_by_arch[0][0], state_by_arch[1][0], state_by_arch[2][0]], dim=1
            )  # [L, 3, B, D]
            c_stack = torch.stack(
                [state_by_arch[0][1], state_by_arch[1][1], state_by_arch[2][1]], dim=1
            )  # [L, 3, B, D]
            h_blended = (layer_weights_expanded * h_stack).sum(dim=1)
            c_blended = (layer_weights_expanded * c_stack).sum(dim=1)

        return output, context, (h_blended, c_blended)

    def get_alphas(self) -> torch.Tensor:
        """Expose architecture + decomposition weights for monitoring."""
        base = super().get_alphas()
        if hasattr(self, "searchable_decomp"):
            return torch.cat([base, self.searchable_decomp.get_alphas()])
        return base


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

        # Reshape for unified multi-head computation (works for num_heads == 1 too).
        q = q.view(B, L_dec, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_enc, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_enc, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention over encoder length.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(B, L_dec, D)

        attended = self.out_proj(attended)

        # Gated combination
        combined_input = torch.cat([decoder_hidden, attended], dim=-1)
        gate_weights = torch.sigmoid(self.gate(combined_input))

        output = gate_weights * attended + (1 - gate_weights) * decoder_hidden
        return output


class LinearAttentionBridge(nn.Module):
    """Lightweight cross linear-attention bridge."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = min(max(1, int(num_heads)), self.d_model)
        while self.d_model % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout_p = float(max(dropout, 0.0))

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_output is None and encoder_context is None:
            return decoder_hidden

        source = encoder_output if encoder_output is not None else encoder_context
        if source.dim() == 2:
            source = source.unsqueeze(1)

        B, L_dec, D = decoder_hidden.shape
        L_enc = source.size(1)

        q = self.q_proj(decoder_hidden)
        k = self.k_proj(source)
        v = self.v_proj(source)

        q = q.view(B, L_dec, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_enc, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_enc, self.num_heads, self.head_dim).transpose(1, 2)

        q_feat = self._feature_map(q * self.scale)
        k_feat = self._feature_map(k)
        kv = torch.einsum("bhtd,bhtv->bhdv", k_feat, v)
        k_sum = k_feat.sum(dim=2)
        denom = torch.einsum("bhtd,bhd->bht", q_feat, k_sum).clamp_min(1e-6)
        out = torch.einsum("bhtd,bhdv->bhtv", q_feat, kv) / denom.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, L_dec, D)
        out = self.out_proj(out)
        if self.training and self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p)
        return out


class LearnedPoolingBridge(nn.Module):
    """Compress encoder sequences into a fixed-size decoder memory."""

    def __init__(
        self, dim: int, num_queries: int = 8, num_heads: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = max(1, int(num_queries))
        self.num_heads = min(max(1, int(num_heads)), dim)
        while dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.queries = nn.Parameter(torch.randn(1, self.num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.norm = RMSNorm(dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        if encoder_output is None:
            raise ValueError(
                "encoder_output must not be None for LearnedPoolingBridge."
            )

        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)

        batch_size = encoder_output.size(0)
        queries = self.queries.expand(batch_size, -1, -1)
        memory, _ = self.attn(queries, encoder_output, encoder_output)
        return self.norm(memory)


class MixedDecoder(BaseMixedSequenceBlock):
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
        use_learned_memory_pooling: bool = True,
        memory_num_queries: int = 8,
        single_path_search: bool = True,
        arch_path_keep_prob: float = 0.85,
        attention_temperature_mult: float = 0.7,
        min_attention_temperature: float = 0.25,
        memory_query_options: Optional[List[int]] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            dropout=dropout,
            temperature=temperature,
            num_layers=2,
            num_options=3,
            single_path_search=single_path_search,
            arch_path_keep_prob=arch_path_keep_prob,
        )

        self.use_attention_bridge = use_attention_bridge
        self.attention_layers = attention_layers
        self.use_learned_memory_pooling = use_learned_memory_pooling
        self.attention_temperature_mult = float(max(attention_temperature_mult, 1e-3))
        self.min_attention_temperature = float(max(min_attention_temperature, 1e-3))
        self.searchable_decomp = SearchableDecomposition(c_in=input_dim)
        self.default_memory_num_queries = int(max(1, memory_num_queries))

        self.transformer = LightweightTransformerDecoder(
            input_dim=input_dim, latent_dim=latent_dim, num_layers=2, dropout=dropout
        )

        # Compatibility attributes
        self.decoders = nn.ModuleList([self.lstm, self.gru, self.transformer])
        self.decoder_names = ["lstm", "gru", "transformer"]
        self.rnn_names = ["lstm", "gru", "transformer"]

        # Architecture normalization
        self.normalizer = ArchitectureNormalizer(latent_dim)
        self.memory_query_options = (
            list(memory_query_options)
            if memory_query_options is not None
            else [4, 8, 16]
        )
        self.memory_query_options = [
            int(max(1, q)) for q in self.memory_query_options if int(max(1, q)) > 0
        ]
        if not self.memory_query_options:
            self.memory_query_options = [4, 8, 16]
        if self.default_memory_num_queries not in self.memory_query_options:
            self.memory_query_options = sorted(
                set(self.memory_query_options + [self.default_memory_num_queries])
            )
        self.default_memory_query_idx = self.memory_query_options.index(
            self.default_memory_num_queries
        )

        if use_learned_memory_pooling:
            self.memory_pool_bridges = nn.ModuleList(
                [
                    LearnedPoolingBridge(
                        dim=latent_dim,
                        num_queries=num_queries,
                        num_heads=4,
                        dropout=dropout,
                    )
                    for num_queries in self.memory_query_options
                ]
            )
            memory_query_init = 0.01 * torch.randn(len(self.memory_query_options))
            self.register_parameter(
                "memory_query_alphas", nn.Parameter(memory_query_init)
            )
        else:
            self.memory_pool_bridges = None

        # Unified attention bridge
        if use_attention_bridge:
            self.attention_bridge = AttentionBridge(
                latent_dim, num_heads=4, dropout=dropout
            )
            self.linear_attention_bridge = LinearAttentionBridge(
                latent_dim, num_heads=4, dropout=dropout
            )

            # Attention choice parameters: [none, sdp_attention, linear_attention]
            attention_init = 0.01 * torch.randn(3)
            self.register_parameter("attention_alphas", nn.Parameter(attention_init))

    def _get_decoder_weights(self) -> torch.Tensor:
        """Get differentiable decoder-mixture weights with deterministic eval behavior."""
        return self._get_output_arch_weights()

    def _get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention-choice mixture weights when the bridge is enabled."""
        if not (self.use_attention_bridge and hasattr(self, "attention_alphas")):
            return None

        tau = max(
            float(self.temperature) * self.attention_temperature_mult,
            self.min_attention_temperature,
        )
        if self._should_use_stochastic_arch_sampling():
            if self.single_path_search:
                return self._sample_straight_through_gumbel(
                    self.attention_alphas, tau=tau, dim=0
                )
            return F.gumbel_softmax(self.attention_alphas, tau=tau, hard=False, dim=0)
        return F.softmax(self.attention_alphas / tau, dim=0)

    def _get_memory_query_weights(self) -> Optional[torch.Tensor]:
        if not (
            self.use_learned_memory_pooling
            and hasattr(self, "memory_query_alphas")
            and self.memory_pool_bridges is not None
        ):
            return None

        tau = max(float(self.temperature), 1e-3)
        if self._should_use_stochastic_arch_sampling():
            if self.single_path_search:
                return self._sample_straight_through_gumbel(
                    self.memory_query_alphas, tau=tau, dim=0
                )
            return F.gumbel_softmax(
                self.memory_query_alphas, tau=tau, hard=False, dim=0
            )
        return F.softmax(self.memory_query_alphas / tau, dim=0)

    def _build_shared_memory(
        self,
        memory: Optional[torch.Tensor],
        encoder_output: Optional[torch.Tensor],
        encoder_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        source = encoder_output
        if source is None:
            source = memory
        if source is None:
            source = encoder_context
        if source is None:
            return None

        if isinstance(source, tuple):
            source = source[0]

        if source.dim() == 2:
            source = source.unsqueeze(1)

        if self.memory_pool_bridges is not None:
            memory_weights = self._get_memory_query_weights()
            if memory_weights is None:
                pooled = self.memory_pool_bridges[self.default_memory_query_idx](source)
                return self._resize_memory_queries(
                    pooled, self.default_memory_num_queries
                )

            if self.training and self.single_path_search:
                chosen = int(torch.argmax(memory_weights.detach()).item())
                pooled = self.memory_pool_bridges[chosen](source)
                pooled = self._resize_memory_queries(
                    pooled, self.default_memory_num_queries
                )
                return memory_weights[chosen] * pooled

            pooled_outputs = [
                self._resize_memory_queries(
                    bridge(source), self.default_memory_num_queries
                )
                for bridge in self.memory_pool_bridges
            ]
            return sum(w * out for w, out in zip(memory_weights, pooled_outputs))

        return source

    @staticmethod
    def _resize_memory_queries(memory: torch.Tensor, target_queries: int) -> torch.Tensor:
        """Resize [B, Q, D] memory tokens to a common Q for weighted mixing."""
        if memory.dim() != 3:
            return memory
        q = int(memory.size(1))
        target = int(max(1, target_queries))
        if q == target:
            return memory
        return F.interpolate(
            memory.transpose(1, 2),
            size=target,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with improved compatibility"""
        tgt = self.searchable_decomp(tgt, temperature=self.temperature)
        batch_size = tgt.size(0)
        single_path_active = self.training and self.single_path_search

        lstm_state, gru_state, trans_state = (
            SequenceStateAdapter.split_mixed_decoder_states(
                hidden_state,
                num_layers=self.num_layers,
                batch_size=batch_size,
                hidden_size=self.latent_dim,
                device=tgt.device,
                dtype=tgt.dtype,
            )
        )

        shared_memory = self._build_shared_memory(
            memory, encoder_output, encoder_context
        )
        transformer_memory = (
            shared_memory
            if shared_memory is not None
            else (
                encoder_context
                if encoder_context is not None
                else torch.zeros_like(tgt[:, :1, :])
            )
        )
        attention_source = (
            shared_memory if shared_memory is not None else encoder_output
        )
        if attention_source is None:
            attention_source = encoder_context

        # Get architecture weights and selected paths
        decoder_layer_weights = self._get_arch_weights()
        decoder_weights = self._get_decoder_weights()
        selected_output_idx = int(torch.argmax(decoder_weights.detach()).item())
        selected_layer_idx = torch.argmax(decoder_layer_weights.detach(), dim=-1)

        if single_path_active:
            required_arches = {selected_output_idx}
            required_arches.update(int(idx.item()) for idx in selected_layer_idx)
        else:
            required_arches = {0, 1, 2}

        output_by_arch = {}
        state_by_arch = {}

        if 0 in required_arches:
            lstm_out, lstm_new_state = self.lstm(tgt, lstm_state)
            output_by_arch[0] = self.normalizer.normalize_output(lstm_out, "lstm")
            state_by_arch[0] = self.normalizer.normalize_state(lstm_new_state, "lstm")

        if 1 in required_arches:
            gru_out, gru_new_state = self.gru(tgt, gru_state)
            output_by_arch[1] = self.normalizer.normalize_output(gru_out, "gru")
            state_by_arch[1] = self.normalizer.normalize_state(gru_new_state, "gru")

        if 2 in required_arches:
            trans_out, trans_new_state = self.transformer(
                tgt, transformer_memory, trans_state
            )
            output_by_arch[2] = self.normalizer.normalize_output(
                trans_out, "transformer"
            )
            state_by_arch[2] = self.normalizer.normalize_state(
                trans_new_state, "transformer"
            )

        can_attend = self.use_attention_bridge and attention_source is not None
        attention_weights = self._get_attention_weights() if can_attend else None

        output_final_by_arch = {}
        for arch_idx in required_arches:
            base_out = output_by_arch[arch_idx]
            if can_attend and attention_weights is not None:
                if attention_weights.numel() == 2:
                    # Backward compatibility with old [use_attention, no_attention].
                    none_weight = attention_weights[1]
                    sdp_weight = attention_weights[0]
                    linear_weight = attention_weights.new_tensor(0.0)
                else:
                    # Current order: [none, sdp_attention, linear_attention].
                    none_weight = attention_weights[0]
                    sdp_weight = attention_weights[1]
                    linear_weight = (
                        attention_weights[2]
                        if attention_weights.numel() > 2
                        else attention_weights.new_tensor(0.0)
                    )

                if single_path_active:
                    selected_att_idx = int(torch.argmax(attention_weights.detach()).item())
                    if attention_weights.numel() == 2:
                        if selected_att_idx == 0:
                            attended = self.attention_bridge(
                                base_out,
                                encoder_output=attention_source,
                                encoder_context=encoder_context,
                            )
                            output_final_by_arch[arch_idx] = sdp_weight * attended
                        else:
                            output_final_by_arch[arch_idx] = none_weight * base_out
                    else:
                        if selected_att_idx == 0:
                            output_final_by_arch[arch_idx] = none_weight * base_out
                        elif selected_att_idx == 1:
                            attended = self.attention_bridge(
                                base_out,
                                encoder_output=attention_source,
                                encoder_context=encoder_context,
                            )
                            output_final_by_arch[arch_idx] = sdp_weight * attended
                        else:
                            linear_attended = self.linear_attention_bridge(
                                base_out,
                                encoder_output=attention_source,
                                encoder_context=encoder_context,
                            )
                            output_final_by_arch[arch_idx] = (
                                linear_weight * linear_attended
                            )
                else:
                    sdp_attended = self.attention_bridge(
                        base_out,
                        encoder_output=attention_source,
                        encoder_context=encoder_context,
                    )
                    linear_attended = self.linear_attention_bridge(
                        base_out,
                        encoder_output=attention_source,
                        encoder_context=encoder_context,
                    )
                    output_final_by_arch[arch_idx] = (
                        none_weight * base_out
                        + sdp_weight * sdp_attended
                        + linear_weight * linear_attended
                    )
            elif can_attend:
                output_final_by_arch[arch_idx] = self.attention_bridge(
                    base_out,
                    encoder_output=attention_source,
                    encoder_context=encoder_context,
                )
            else:
                output_final_by_arch[arch_idx] = base_out

        if single_path_active:
            selected_output_weight = decoder_weights[selected_output_idx]
            output = selected_output_weight * output_final_by_arch[selected_output_idx]
        else:
            output = sum(decoder_weights[i] * output_final_by_arch[i] for i in range(3))

        if single_path_active:
            h_blended = torch.zeros(
                self.num_layers,
                batch_size,
                self.latent_dim,
                device=tgt.device,
                dtype=output.dtype,
            )
            c_blended = torch.zeros_like(h_blended)
            for layer_idx in range(self.num_layers):
                arch_idx = int(selected_layer_idx[layer_idx].item())
                layer_weight = decoder_layer_weights[layer_idx, arch_idx]
                h_blended[layer_idx] = (
                    layer_weight * state_by_arch[arch_idx][0][layer_idx]
                )
                c_blended[layer_idx] = (
                    layer_weight * state_by_arch[arch_idx][1][layer_idx]
                )
        else:
            decoder_layer_weights_expanded = decoder_layer_weights.unsqueeze(
                -1
            ).unsqueeze(-1)  # [L, 3, 1, 1]
            h_stack = torch.stack(
                [state_by_arch[0][0], state_by_arch[1][0], state_by_arch[2][0]], dim=1
            )  # [L, 3, B, D]
            c_stack = torch.stack(
                [state_by_arch[0][1], state_by_arch[1][1], state_by_arch[2][1]], dim=1
            )  # [L, 3, B, D]
            h_blended = (decoder_layer_weights_expanded * h_stack).sum(dim=1)
            c_blended = (decoder_layer_weights_expanded * c_stack).sum(dim=1)

        return output, (h_blended, c_blended)

    def get_alphas(self) -> torch.Tensor:
        """Get architecture parameters for compatibility"""
        parts = [super().get_alphas()]

        if hasattr(self, "searchable_decomp"):
            parts.append(self.searchable_decomp.get_alphas())

        if self.use_learned_memory_pooling and hasattr(self, "memory_query_alphas"):
            parts.append(F.softmax(self.memory_query_alphas, dim=0))

        if self.use_attention_bridge and hasattr(self, "attention_alphas"):
            parts.append(F.softmax(self.attention_alphas, dim=0))

        return torch.cat(parts)


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
        return SequenceStateAdapter.ensure_rnn_state(
            state,
            rnn_type=rnn_type,
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
        )

    @staticmethod
    def fix_mixed_weights(mixed_model, temperature: float = 0.01):
        """Fix mixed model to use best architecture by setting alphas"""
        with torch.no_grad():
            # Get current best architecture
            alphas = mixed_model.get_alphas()
            best_idx = torch.argmax(alphas[:3])

            # Set logits so softmax/gumbel-softmax is effectively one-hot.
            new_alphas = torch.full_like(mixed_model.alphas, -10.0)
            new_alphas[best_idx] = 10.0
            mixed_model.alphas.copy_(new_alphas)
            if hasattr(mixed_model, "layer_alpha_offsets"):
                mixed_model.layer_alpha_offsets.zero_()

            # Set very low temperature for sharp selection
            mixed_model.set_temperature(temperature)

            # Fix attention alphas if they exist
            if hasattr(mixed_model, "attention_alphas"):
                attention_best = torch.argmax(mixed_model.attention_alphas)
                new_attention_alphas = torch.full_like(
                    mixed_model.attention_alphas, -10.0
                )
                new_attention_alphas[attention_best] = 10.0
                mixed_model.attention_alphas.copy_(new_attention_alphas)

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
        attention_variant = kwargs.get("attention_variant", "sdp")

        # Create fixed decoder
        fixed_decoder = FixedDecoder(
            rnn_type=best_type,
            input_dim=mixed_decoder.input_dim,
            latent_dim=mixed_decoder.latent_dim,
            use_attention_bridge=kwargs.get(
                "use_attention_bridge", mixed_decoder.use_attention_bridge
            ),
            attention_variant=attention_variant,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in {"use_attention_bridge", "attention_variant"}
            },
        )

        # Transfer weights
        ArchitectureConverter._transfer_decoder_weights(
            mixed_decoder, fixed_decoder, best_type, attention_variant=attention_variant
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

            # Preserve mixed encoder post-processing to avoid transfer-time behavior drift.
            if hasattr(mixed_encoder, "normalizer"):
                fixed_encoder.normalizer = ArchitectureNormalizer(
                    mixed_encoder.latent_dim
                ).to(next(fixed_encoder.parameters()).device)
                fixed_encoder.normalizer.load_state_dict(
                    mixed_encoder.normalizer.state_dict()
                )
            if hasattr(mixed_encoder, "context_proj"):
                fixed_encoder.context_proj = nn.Linear(
                    mixed_encoder.latent_dim,
                    mixed_encoder.latent_dim,
                ).to(next(fixed_encoder.parameters()).device)
                fixed_encoder.context_proj.load_state_dict(
                    mixed_encoder.context_proj.state_dict()
                )
            if hasattr(mixed_encoder, "searchable_decomp"):
                fixed_encoder.searchable_decomp = copy.deepcopy(
                    mixed_encoder.searchable_decomp
                ).to(next(fixed_encoder.parameters()).device)
                if hasattr(fixed_encoder.searchable_decomp, "alpha_logits"):
                    with torch.no_grad():
                        logits = fixed_encoder.searchable_decomp.alpha_logits
                        hard = torch.full_like(logits, -10.0)
                        hard[int(torch.argmax(logits).item())] = 10.0
                        logits.copy_(hard)
                    fixed_encoder.searchable_decomp.alpha_logits.requires_grad_(False)
        except Exception as e:
            print(f"Warning: Could not transfer encoder weights: {e}")

    @staticmethod
    def _transfer_decoder_weights(
        mixed_decoder, fixed_decoder, arch_type: str, attention_variant: str = "sdp"
    ):
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

            if hasattr(mixed_decoder, "normalizer"):
                fixed_decoder.normalizer = ArchitectureNormalizer(
                    mixed_decoder.latent_dim
                ).to(next(fixed_decoder.parameters()).device)
                fixed_decoder.normalizer.load_state_dict(
                    mixed_decoder.normalizer.state_dict()
                )
            if hasattr(mixed_decoder, "searchable_decomp"):
                fixed_decoder.searchable_decomp = copy.deepcopy(
                    mixed_decoder.searchable_decomp
                ).to(next(fixed_decoder.parameters()).device)
                if hasattr(fixed_decoder.searchable_decomp, "alpha_logits"):
                    with torch.no_grad():
                        logits = fixed_decoder.searchable_decomp.alpha_logits
                        hard = torch.full_like(logits, -10.0)
                        hard[int(torch.argmax(logits).item())] = 10.0
                        logits.copy_(hard)
                    fixed_decoder.searchable_decomp.alpha_logits.requires_grad_(False)

            # Transfer attention bridge weights if present
            if (
                fixed_decoder.use_attention_bridge
                and hasattr(mixed_decoder, "attention_bridge")
                and hasattr(fixed_decoder, "attention_bridge")
            ):
                try:
                    if attention_variant == "linear" and hasattr(
                        mixed_decoder, "linear_attention_bridge"
                    ):
                        fixed_decoder.attention_bridge.load_state_dict(
                            mixed_decoder.linear_attention_bridge.state_dict()
                        )
                    else:
                        fixed_decoder.attention_bridge.load_state_dict(
                            mixed_decoder.attention_bridge.state_dict()
                        )
                except Exception as e:
                    print(f"Warning: Could not transfer attention bridge weights: {e}")

        except Exception as e:
            print(f"Warning: Could not transfer decoder weights: {e}")


class FixedEncoder(BaseFixedSequenceBlock):
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
        super().__init__(
            rnn=rnn,
            rnn_type=rnn_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            transformer_factory=LightweightTransformerEncoder,
        )
        # Optional modules copied from MixedEncoder by ArchitectureConverter.
        self.normalizer = None
        self.context_proj = None
        self.searchable_decomp = None

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass optimized for single architecture"""
        if self.searchable_decomp is not None:
            x = self.searchable_decomp(x, temperature=0.01)

        if self.rnn_type == "transformer":
            output, context, state = self.rnn(x)
        else:
            raw_output, state = self.rnn(x)
            context = raw_output[:, -1:, :]  # Last timestep
            output = raw_output

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

        if self.normalizer is not None:
            output = self.normalizer.normalize_output(output, self.rnn_type)
            state = self.normalizer.normalize_state(state, self.rnn_type)
        if self.context_proj is not None:
            context = self.context_proj(context)

        return output, context, state


class FixedDecoder(BaseFixedSequenceBlock):
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
        attention_variant: str = "sdp",
    ):
        self.use_attention_bridge = use_attention_bridge
        self.attention_variant = str(attention_variant).lower()

        super().__init__(
            rnn=rnn,
            rnn_type=rnn_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            transformer_factory=LightweightTransformerDecoder,
        )

        # Simple attention bridge for fixed decoder
        if use_attention_bridge:
            if self.attention_variant == "linear":
                self.attention_bridge = LinearAttentionBridge(
                    latent_dim, num_heads=4, dropout=dropout
                )
            else:
                self.attention_bridge = AttentionBridge(
                    latent_dim, num_heads=4, dropout=dropout
                )
        self.normalizer = None
        self.searchable_decomp = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: torch.Tensor = None,
    ) -> tuple:
        """Forward pass optimized for single architecture"""
        if self.searchable_decomp is not None:
            tgt = self.searchable_decomp(tgt, temperature=0.01)

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
            raw_output, new_state = self.rnn(tgt, memory, hidden_state)
        else:
            raw_output, new_state = self.rnn(tgt, hidden_state)

        output = raw_output
        if self.normalizer is not None:
            output = self.normalizer.normalize_output(output, self.rnn_type)
            normalized_state = self.normalizer.normalize_state(new_state, self.rnn_type)
            if normalized_state is not None:
                new_state = normalized_state

        # Apply attention if enabled
        if self.use_attention_bridge and hasattr(self, "attention_bridge"):
            attention_source = encoder_output if encoder_output is not None else memory
            if attention_source is not None:
                output = self.attention_bridge(output, attention_source)

        return output, new_state


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding with odd-dim support and per-device dynamic caches."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 500000.0):
        super().__init__()
        self.dim = int(dim)
        self.rotary_dim = int(self.dim - (self.dim % 2))
        self.max_seq_len = int(max_seq_len) * 2
        self.base = float(base)

        if self.rotary_dim > 0:
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
            )
        else:
            inv_freq = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._device_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._init_cache(self.max_seq_len)

    def _cache_key(self, device: torch.device) -> str:
        if device.type == "cuda":
            index = 0 if device.index is None else int(device.index)
            return f"cuda:{index}"
        return device.type

    def _compute_chunk(
        self, start: int, end: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == 0:
            empty = torch.zeros(end - start, 0, device=device, dtype=self.inv_freq.dtype)
            return empty, empty

        t = torch.arange(start, end, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

    def _init_cache(self, max_len: int) -> None:
        cos, sin = self._compute_chunk(0, int(max_len), self.inv_freq.device)
        self.register_buffer("cached_cos", cos, persistent=False)
        self.register_buffer("cached_sin", sin, persistent=False)
        key = self._cache_key(self.inv_freq.device)
        self._device_cache[key] = {"cos": self.cached_cos, "sin": self.cached_sin}

    def _ensure_cache(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self._cache_key(device)

        if key not in self._device_cache:
            base_cos = self.cached_cos.to(device)
            base_sin = self.cached_sin.to(device)
            self._device_cache[key] = {"cos": base_cos, "sin": base_sin}

        cache = self._device_cache[key]
        cos = cache["cos"]
        sin = cache["sin"]
        cached_len = int(cos.size(0))

        if seq_len > cached_len:
            new_len = max(seq_len, cached_len * 2)
            add_cos, add_sin = self._compute_chunk(cached_len, new_len, device)
            cos = torch.cat([cos, add_cos], dim=0)
            sin = torch.cat([sin, add_sin], dim=0)
            cache["cos"] = cos
            cache["sin"] = sin

        return cache["cos"][:seq_len], cache["sin"][:seq_len]

    def forward(self, seq_len: int, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.inv_freq.device
        return self._ensure_cache(int(seq_len), device)

    def apply_rotary_pos_emb(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply RoPE to the largest even sub-dimension and passthrough any tail dims."""
        seq_len = x.size(-2)
        head_dim = x.size(-1)
        rotary_dim = min(self.rotary_dim, head_dim - (head_dim % 2))
        if rotary_dim <= 0:
            return x

        half = rotary_dim // 2
        cos = cos[:seq_len, :half].view(1, 1, seq_len, half).to(dtype=x.dtype)
        sin = sin[:seq_len, :half].view(1, 1, seq_len, half).to(dtype=x.dtype)

        x_rot = x[..., :rotary_dim]
        x_even = x_rot[..., :half]
        x_odd = x_rot[..., half:rotary_dim]
        rotated = torch.cat(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        )

        if rotary_dim < head_dim:
            return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
        return rotated

    def get_embeddings_for_length(
        self, seq_len: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
