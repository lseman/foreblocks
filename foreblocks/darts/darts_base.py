import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.size()
        H = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(B, T, H, D // H).transpose(1, 2) for t in qkv
        ]  # [B, H, T, D//H]

        k = k.softmax(dim=-2)  # across sequence
        context = torch.einsum("bhtd,bhtv->bhdv", k, v)  # context: [B, H, D//H, D//H]
        out = torch.einsum("bhtd,bhdv->bhtv", q, context)  # attention result
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.dropout(self.out_proj(out))


class LightweightTransformerEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=2, dropout=0.1, nhead=4):
        super().__init__()
        self.latent_dim = latent_dim  # ← this was missing

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": LinearSelfAttention(
                            latent_dim, heads=nhead, dropout=dropout
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(latent_dim, latent_dim * 4),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(latent_dim * 4, latent_dim),
                        ),
                        "norm1": nn.LayerNorm(latent_dim),
                        "norm2": nn.LayerNorm(latent_dim),
                        "drop": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.state_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, hidden_state=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            attn_out = layer["attn"](x)
            x = layer["norm1"](x + layer["drop"](attn_out))
            ff_out = layer["ffn"](x)
            x = layer["norm2"](x + layer["drop"](ff_out))

        ctx = x[:, -1:, :]
        h_state = (
            self.state_proj(ctx.squeeze(1))
            .unsqueeze(0)
            .expand(len(self.layers), -1, -1)
            .contiguous()
        )
        c_state = h_state.clone()
        return x, ctx, (h_state, c_state)


class LightweightTransformerDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=2, dropout=0.1, nhead=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_decoder = nn.Parameter(torch.randn(1, 512, latent_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": LinearSelfAttention(
                            latent_dim, heads=nhead, dropout=dropout
                        ),
                        "cross_attn": nn.MultiheadAttention(
                            latent_dim, nhead, dropout=dropout, batch_first=True
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(latent_dim, latent_dim * 4),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(latent_dim * 4, latent_dim),
                        ),
                        "norm1": nn.LayerNorm(latent_dim),
                        "norm2": nn.LayerNorm(latent_dim),
                        "norm3": nn.LayerNorm(latent_dim),
                        "drop": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.state_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, tgt, memory_or_hidden, hidden_state=None):
        # Handle RNN-style state or Transformer memory
        if isinstance(memory_or_hidden, tuple):
            memory = memory_or_hidden[0].transpose(0, 1)  # [B, T, D]
        else:
            memory = memory_or_hidden

        tgt = self.input_proj(tgt)
        seq_len = tgt.size(1)
        if seq_len <= self.pos_decoder.size(1):
            tgt = tgt + self.pos_decoder[:, :seq_len, :]

        for layer in self.layers:
            tgt2 = layer["self_attn"](tgt)
            tgt = layer["norm1"](tgt + layer["drop"](tgt2))

            tgt2, _ = layer["cross_attn"](tgt, memory, memory)
            tgt = layer["norm2"](tgt + layer["drop"](tgt2))

            tgt2 = layer["ffn"](tgt)
            tgt = layer["norm3"](tgt + layer["drop"](tgt2))

        last = tgt[:, -1, :]
        h_state = (
            self.state_proj(last)
            .unsqueeze(0)
            .expand(len(self.layers), -1, -1)
            .contiguous()
        )
        c_state = h_state.clone()
        return tgt, (h_state, c_state)


class BaseRNN(nn.Module):
    """Streamlined base class for encoder/decoder implementations"""

    def __init__(
        self, input_dim: int, latent_dim: int, num_layers: int = 2, dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Cache RNN names and mappings for efficiency
        self.rnn_names = ["lstm", "gru", "transformer"]
        self._type_cache = {}
        self._alpha_cache = {}

    def _create_rnn(
        self,
        rnn_type: str,
        input_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float,
        is_decoder: bool = False,
    ) -> nn.Module:
        """Create RNN with optimized parameter handling"""
        rnn_type = rnn_type.lower()

        if rnn_type == "transformer":
            return self._create_transformer(
                input_dim, latent_dim, num_layers, dropout, is_decoder
            )
        elif rnn_type in ["lstm", "gru"]:
            return self._create_recurrent_rnn(
                rnn_type, input_dim, latent_dim, num_layers, dropout
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

    def _create_transformer(
        self,
        input_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float,
        is_decoder: bool,
    ) -> nn.Module:
        """Create transformer encoder or decoder"""
        wrapper_class = (
            LightweightTransformerDecoder
            if is_decoder
            else LightweightTransformerEncoder
        )
        return wrapper_class(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def _create_recurrent_rnn(
        self,
        rnn_type: str,
        input_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        """Create LSTM or GRU with proper initialization"""
        rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU

        # Optimize dropout handling
        effective_dropout = dropout if num_layers > 1 else 0.0

        # Create RNN with appropriate parameters
        if rnn_type == "lstm":
            rnn = rnn_class(
                input_size=input_dim,
                hidden_size=latent_dim,
                num_layers=num_layers,
                dropout=effective_dropout,
                batch_first=True,
                proj_size=0,  # LSTM-specific parameter
            )
        else:  # GRU
            rnn = rnn_class(
                input_size=input_dim,
                hidden_size=latent_dim,
                num_layers=num_layers,
                dropout=effective_dropout,
                batch_first=True,
            )

        # Initialize weights for better convergence
        self._init_rnn_weights(rnn, rnn_type)
        return rnn

    def _init_rnn_weights(self, rnn: nn.Module, rnn_type: str):
        """Initialize RNN weights for better training stability"""
        for name, param in rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

                # Set forget gate bias to 1 for LSTM
                if rnn_type == "lstm" and "bias_ih" in name:
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)

    def _detect_rnn_type(self, rnn: nn.Module) -> str:
        """Detect RNN type with caching"""
        rnn_id = id(rnn)
        if rnn_id not in self._type_cache:
            class_name = type(rnn).__name__.lower()

            if "transformer" in class_name:
                rnn_type = "transformer"
            elif "lstm" in class_name:
                rnn_type = "lstm"
            elif "gru" in class_name:
                rnn_type = "gru"
            else:
                rnn_type = class_name

            self._type_cache[rnn_id] = rnn_type

        return self._type_cache[rnn_id]

    def _extract_rnn_properties(self, rnn: nn.Module) -> tuple:
        """Extract RNN properties with fallback options"""
        # Try multiple attribute names for robustness
        latent_dim = (
            getattr(rnn, "hidden_size", None)
            or getattr(rnn, "latent_dim", None)
            or getattr(rnn, "d_model", None)
        )

        if latent_dim is None:
            raise ValueError(f"Cannot extract latent_dim from {type(rnn).__name__}")

        num_layers = getattr(rnn, "num_layers", 1)
        return latent_dim, num_layers

    def _get_alpha_for_type(self, rnn_type: str, device: torch.device) -> torch.Tensor:
        """Get cached one-hot encoding for RNN type"""
        cache_key = (rnn_type, device)

        if cache_key not in self._alpha_cache:
            alpha_map = {
                "lstm": [1.0, 0.0, 0.0],
                "gru": [0.0, 1.0, 0.0],
                "transformer": [0.0, 0.0, 1.0],
            }

            alpha = torch.tensor(
                alpha_map.get(rnn_type, [0.0, 0.0, 1.0]),
                device=device,
                dtype=torch.float32,
            )
            self._alpha_cache[cache_key] = alpha

        return self._alpha_cache[cache_key]

    def get_rnn_info(self, rnn: nn.Module) -> dict[str, any]:
        """Get comprehensive RNN information for debugging"""
        rnn_type = self._detect_rnn_type(rnn)
        latent_dim, num_layers = self._extract_rnn_properties(rnn)

        return {
            "type": rnn_type,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "parameters": sum(p.numel() for p in rnn.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in rnn.parameters() if p.requires_grad
            ),
        }

    def clear_caches(self):
        """Clear all internal caches"""
        self._type_cache.clear()
        self._alpha_cache.clear()

    def set_dropout(self, dropout: float):
        """Update dropout rate for all components"""
        self.dropout = dropout
        # Update dropout in existing RNN modules if needed
        for module in self.modules():
            if hasattr(module, "dropout") and isinstance(module.dropout, float):
                module.dropout = dropout


class FixedEncoder(BaseRNN):
    """Streamlined single encoder with fixed architecture"""

    def __init__(
        self,
        rnn: nn.Module = None,
        *,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(input_dim, latent_dim, num_layers, dropout)

        # Initialize RNN from existing module or create new one
        if rnn is not None:
            self._init_from_existing_rnn(rnn)
        else:
            self._init_from_parameters(
                rnn_type, input_dim, latent_dim, num_layers, dropout
            )

        # Cache for alpha tensor
        self._cached_alpha = None

    def _init_from_existing_rnn(self, rnn: nn.Module):
        """Initialize from existing RNN module"""
        self.rnn = rnn
        self.rnn_type = self._detect_rnn_type(rnn)
        self.latent_dim, self.num_layers = self._extract_rnn_properties(rnn)

    def _init_from_parameters(
        self,
        rnn_type: str,
        input_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """Initialize by creating new RNN"""
        if rnn_type is None:
            raise ValueError("Either 'rnn' or 'rnn_type' must be provided")

        self.rnn_type = rnn_type.lower()
        self.rnn = self._create_rnn(
            self.rnn_type, input_dim, latent_dim, num_layers, dropout
        )

    def _process_transformer_output(self, x: torch.Tensor) -> tuple:
        """Process transformer output"""
        return self.rnn(x)  # Returns (output, ctx, state)

    def _process_recurrent_output(self, x: torch.Tensor) -> tuple:
        """Process LSTM/GRU output"""
        h, state = self.rnn(x)
        # Extract last timestep as context efficiently
        ctx = h.narrow(1, h.size(1) - 1, 1)
        return h, ctx, state

    def forward(self, x: torch.Tensor) -> tuple:
        """Optimized forward pass with type-specific processing"""
        if self.rnn_type == "transformer":
            return self._process_transformer_output(x)
        else:
            return self._process_recurrent_output(x)

    def get_alphas(self) -> torch.Tensor:
        """Get cached alpha tensor for this encoder type"""
        device = next(self.parameters()).device

        # Return cached alpha if valid, otherwise compute and cache
        if self._cached_alpha is None or self._cached_alpha.device != device:
            self._cached_alpha = self._get_alpha_for_type(self.rnn_type, device)

        return self._cached_alpha

    def get_entropy_loss(self) -> torch.Tensor:
        """Return zero entropy loss for fixed encoder"""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_encoder_info(self) -> dict[str, any]:
        """Get comprehensive encoder information"""
        base_info = self.get_rnn_info(self.rnn)
        base_info.update(
            {
                "encoder_type": "fixed",
                "architecture": self.rnn_type,
                "is_learnable": False,
            }
        )
        return base_info

    def set_temperature(self, temp: float):
        """Temperature has no effect on fixed encoder"""
        pass  # No-op for fixed encoder

    def clear_cache(self):
        """Clear cached alpha tensor"""
        self._cached_alpha = None


class MixedEncoder(BaseRNN):
    """Streamlined mixed encoder with learnable weights"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__(input_dim, latent_dim, num_layers=2, dropout=dropout)

        self.temperature = temperature
        self.seq_len = seq_len

        # Create encoders for each RNN type
        self.encoders = nn.ModuleList(
            [
                self._create_rnn(
                    rnn_type, input_dim, latent_dim, self.num_layers, dropout
                )
                for rnn_type in self.rnn_names
            ]
        )
        self.encoder_names = self.rnn_names

        # Initialize architecture parameters
        self.alphas = nn.Parameter(torch.zeros(len(self.encoders)))

        # Cache for inference optimization
        self._inference_cache = None

    def _get_encoder_weights(self) -> torch.Tensor:
        """Get encoder weights with temperature scaling"""
        return F.softmax(self.alphas / self.temperature, dim=0)

    def _process_single_encoder(
        self, encoder: nn.Module, rnn_type: str, x: torch.Tensor
    ) -> tuple:
        """Process single encoder and return standardized output format"""
        if rnn_type == "transformer":
            return encoder(x)  # Returns (h, ctx, state)
        else:
            h, state = encoder(x)
            ctx = h.narrow(1, h.size(1) - 1, 1)  # Last timestep as context
            return h, ctx, state

    def _handle_inference_caching(self, weights: torch.Tensor, x: torch.Tensor):
        """Handle inference with caching for dominant encoder"""
        max_weight = weights.max().item()
        if max_weight <= 0.9:
            return None  # No dominant encoder

        max_idx = weights.argmax().item()

        # Use cached encoder if available
        if self._inference_cache is not None and self._inference_cache[0] == max_idx:
            encoder = self._inference_cache[1]
        else:
            encoder = self.encoders[max_idx]
            self._inference_cache = (max_idx, encoder)

        rnn_type = self.rnn_names[max_idx]
        return self._process_single_encoder(encoder, rnn_type, x)

    def _get_active_encoders(self, weights: torch.Tensor, threshold: float = 1e-3):
        """Get active encoder indices and normalized weights"""
        active_indices = (weights > threshold).nonzero(as_tuple=False).squeeze(-1)

        if len(active_indices) == 0:
            # Fallback to first encoder
            active_indices = torch.tensor([0], device=weights.device)

        active_weights = weights[active_indices]
        active_weights = active_weights / active_weights.sum()  # Normalize

        return active_indices, active_weights

    def _combine_encoder_outputs(
        self, outputs: list, contexts: list, states: list, active_weights: torch.Tensor
    ) -> tuple:
        """Efficiently combine outputs from multiple encoders"""
        # Stack and weight outputs
        weighted_output = torch.stack(outputs, dim=0)
        weighted_output = torch.sum(
            active_weights.view(-1, 1, 1, 1) * weighted_output, dim=0
        )

        # Stack and weight contexts
        weighted_context = torch.stack(contexts, dim=0)
        weighted_context = torch.sum(
            active_weights.view(-1, 1, 1, 1) * weighted_context, dim=0
        )

        # Use state from encoder with highest weight
        max_weight_idx = active_weights.argmax().item()
        final_state = states[max_weight_idx]

        return weighted_output, weighted_context, final_state

    def forward(self, x: torch.Tensor) -> tuple:
        """Optimized forward with caching and efficient combination"""
        weights = self._get_encoder_weights()

        # Try inference caching for dominant encoder
        if not self.training:
            cached_result = self._handle_inference_caching(weights, x)
            if cached_result is not None:
                return cached_result

        # Get active encoders
        active_indices, active_weights = self._get_active_encoders(weights)

        # Process active encoders
        outputs, contexts, states = [], [], []

        for i in active_indices:
            encoder = self.encoders[i]
            rnn_type = self.rnn_names[i]

            h, ctx, state = self._process_single_encoder(encoder, rnn_type, x)

            outputs.append(h)
            contexts.append(ctx)
            states.append(state)

        # Combine outputs
        return self._combine_encoder_outputs(outputs, contexts, states, active_weights)

    def get_alphas(self) -> torch.Tensor:
        """Get normalized architecture weights"""
        return F.softmax(self.alphas, dim=0)

    def get_entropy_loss(self) -> torch.Tensor:
        """Compute entropy loss for regularization"""
        probs = self.get_alphas()
        log_probs = torch.log(torch.clamp(probs, min=1e-8, max=1.0))
        entropy = -(probs * log_probs).sum() * 0.01
        return torch.clamp(entropy, min=0.0, max=1.0)

    def set_temperature(self, temp: float):
        """Set temperature and clear cache"""
        self.temperature = temp
        self._inference_cache = None

    def clear_cache(self):
        """Clear inference cache"""
        self._inference_cache = None

    def get_encoder_weights(self) -> dict[str, float]:
        """Get current encoder weights for analysis"""
        weights = self._get_encoder_weights()
        return {
            name: weight.item() for name, weight in zip(self.encoder_names, weights)
        }


class BaseDecoder(BaseRNN):
    """Streamlined base decoder with efficient attention handling"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        temperature: float = 1.0,
        use_attention_bridge: bool = True,
        attention_heads: int = 8,
        attention_layers: int = 1,
        use_temporal_bias: bool = True,
        use_rotary: bool = False,
        attention_d_model: int = None,
    ):
        super().__init__(input_dim, latent_dim, num_layers, dropout)

        self.temperature = temperature
        self.use_attention_bridge = use_attention_bridge
        self.attention_d_model = attention_d_model or latent_dim

        # Initialize attention bridge if enabled
        if use_attention_bridge:
            self._init_attention_components(
                attention_heads,
                attention_layers,
                dropout,
                use_temporal_bias,
                use_rotary,
            )
        else:
            print("⚠️ CrossAttentionBridge disabled")

        # Cache for attention optimization
        self._attention_cache = None

    def _init_attention_components(
        self,
        attention_heads: int,
        attention_layers: int,
        dropout: float,
        use_temporal_bias: bool,
        use_rotary: bool,
    ):
        """Initialize attention bridge components"""
        # Optimize number of heads based on model size
        optimal_heads = min(attention_heads, max(1, self.attention_d_model // 32))

        # Create attention bridges
        self.attention_bridges = nn.ModuleList(
            [
                CrossAttentionBridge(
                    d_model=self.attention_d_model,
                    num_heads=optimal_heads,
                    dropout=dropout,
                    use_temporal_bias=use_temporal_bias,
                    use_rotary=use_rotary,
                )
                for _ in range(attention_layers)
            ]
        )

        # Initialize attention weights (favor no attention initially)
        self.attention_alphas = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(attention_layers),  # Attention layers
                    torch.tensor([1.0]),  # No attention (higher initial weight)
                ]
            )
        )

        print(
            f"✅ CrossAttentionBridge enabled with {optimal_heads} heads and {attention_layers} layers"
        )

    def _prepare_hidden_state(
        self, hidden_state, batch_size: int, device: torch.device
    ):
        """Prepare hidden state with efficient memory handling"""
        if hidden_state is None:
            # Create initial hidden states
            h_0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.latent_dim,
                device=device,
                dtype=torch.float32,
            )
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0), h_0

        if isinstance(hidden_state, tuple):
            h, c = hidden_state
            # Ensure proper dimensions
            if h.dim() != 3:
                h = h.unsqueeze(0).expand(self.num_layers, -1, -1)
            if c.dim() != 3:
                c = c.unsqueeze(0).expand(self.num_layers, -1, -1)
            return (h.contiguous(), c.contiguous()), h
        else:
            # Handle single tensor (GRU case)
            if hidden_state.dim() != 3:
                h = hidden_state.unsqueeze(0).expand(self.num_layers, -1, -1)
            else:
                h = hidden_state
            c = torch.zeros_like(h)
            return (h.contiguous(), c.contiguous()), h

    def _ensure_sequence_dims(
        self, tensor: torch.Tensor, rnn_type: str
    ) -> torch.Tensor:
        """Ensure proper sequence dimensions for RNN types"""
        if rnn_type in ["lstm", "gru"] and tensor.dim() == 2:
            return tensor.unsqueeze(1)
        return tensor

    def _get_attention_weights(self) -> torch.Tensor:
        """Get attention weights with temperature scaling"""
        return F.softmax(self.attention_alphas / self.temperature, dim=0)

    def _apply_single_attention(
        self,
        decoder_output: torch.Tensor,
        encoder_output: torch.Tensor,
        bridge_idx: int,
    ) -> torch.Tensor:
        """Apply single attention bridge with error handling"""
        try:
            if bridge_idx == len(self.attention_bridges):
                return decoder_output  # No attention

            bridge = self.attention_bridges[bridge_idx]
            attended_output, _ = bridge(decoder_output, encoder_output)
            return attended_output
        except Exception as e:
            print(f"Attention bridge {bridge_idx} failed: {e}")
            return decoder_output

    def _apply_attention_bridge(
        self,
        decoder_output: torch.Tensor,
        encoder_output: torch.Tensor,
        rnn_type: str = "transformer",
    ) -> torch.Tensor:
        """Apply attention bridge with caching and efficient combination"""
        if (
            not self.use_attention_bridge
            or not hasattr(self, "attention_bridges")
            or encoder_output is None
        ):
            return decoder_output

        try:
            # Ensure proper sequence dimensions
            decoder_output = self._ensure_sequence_dims(decoder_output, rnn_type)
            encoder_output = self._ensure_sequence_dims(encoder_output, rnn_type)

            attention_weights = self._get_attention_weights()

            # Aggressive caching for inference with dominant weight
            if not self.training:
                max_weight = attention_weights.max().item()
                if max_weight > 0.95:
                    max_idx = attention_weights.argmax().item()

                    # Use cached function if available
                    if (
                        self._attention_cache is not None
                        and self._attention_cache[0] == max_idx
                    ):
                        return self._attention_cache[1](decoder_output, encoder_output)
                    else:
                        # Cache the dominant attention function
                        if max_idx == len(self.attention_bridges):
                            func = lambda x, y: x  # No attention
                        else:
                            func = lambda x, y: self._apply_single_attention(
                                x, y, max_idx
                            )

                        self._attention_cache = (max_idx, func)
                        return func(decoder_output, encoder_output)

            # Weighted combination of all active attention mechanisms
            attended_outputs = []
            active_weights = []

            # Process each attention option
            for i in range(len(attention_weights)):
                weight = attention_weights[i]
                if weight.item() > 1e-3:  # Skip very small weights
                    attended_output = self._apply_single_attention(
                        decoder_output, encoder_output, i
                    )
                    attended_outputs.append(attended_output)
                    active_weights.append(weight)

            # Return weighted combination
            if not attended_outputs:
                return decoder_output

            if len(attended_outputs) == 1:
                return attended_outputs[0]

            # Efficient weighted sum
            total_weight = sum(active_weights)
            if total_weight < 1e-8:
                return decoder_output

            stacked_outputs = torch.stack(attended_outputs, dim=0)
            norm_weights = torch.stack(active_weights) / total_weight
            return torch.sum(norm_weights.view(-1, 1, 1, 1) * stacked_outputs, dim=0)

        except Exception as e:
            print(f"Attention bridge application failed: {e}")
            return decoder_output

    def _process_decoder_output(
        self,
        decoder: nn.Module,
        rnn_type: str,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state,
        batch_size: int,
        device: torch.device,
    ):
        """Process decoder output based on RNN type"""
        if rnn_type == "lstm":
            lstm_state, _ = self._prepare_hidden_state(hidden_state, batch_size, device)
            return decoder(tgt, lstm_state)
        elif rnn_type == "gru":
            _, gru_state = self._prepare_hidden_state(hidden_state, batch_size, device)
            return decoder(tgt, gru_state)
        else:  # transformer
            return decoder(tgt, memory, hidden_state)

    def _get_attention_entropy_loss(self) -> torch.Tensor:
        """Compute attention entropy loss for regularization"""
        if self.use_attention_bridge and hasattr(self, "attention_alphas"):
            attn_probs = F.softmax(self.attention_alphas / self.temperature, dim=0)
            log_probs = torch.log(torch.clamp(attn_probs, min=1e-8, max=1.0))
            attn_entropy = -(attn_probs * log_probs).sum() * 0.01
            return torch.clamp(attn_entropy, min=0.0, max=1.0)

        return torch.tensor(0.0, device=next(self.parameters()).device)

    def set_temperature(self, temp: float):
        """Set temperature and clear cache"""
        self.temperature = temp
        self._attention_cache = None

    def clear_attention_cache(self):
        """Clear attention cache"""
        self._attention_cache = None


class FixedDecoder(BaseDecoder):
    """Streamlined fixed decoder with efficient caching"""

    def __init__(
        self,
        rnn: nn.Module = None,
        *,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        temperature: float = 1.0,
        use_attention_bridge: bool = True,
        attention_heads: int = 8,
        attention_layers: int = 1,
        use_temporal_bias: bool = True,
        use_rotary: bool = False,
        attention_d_model: int = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            temperature=temperature,
            use_attention_bridge=use_attention_bridge,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            use_temporal_bias=use_temporal_bias,
            use_rotary=use_rotary,
            attention_d_model=attention_d_model,
        )

        # Initialize RNN from existing module or create new one
        if rnn is not None:
            self._init_from_existing_rnn(rnn)
        else:
            self._init_from_parameters(
                rnn_type, input_dim, latent_dim, num_layers, dropout
            )

        # Cache for alpha computation
        self._cached_alphas = None

    def _init_from_existing_rnn(self, rnn: nn.Module):
        """Initialize from existing RNN module"""
        self.rnn = rnn
        self.rnn_type = self._detect_rnn_type(rnn)
        self.latent_dim, self.num_layers = self._extract_rnn_properties(rnn)

    def _init_from_parameters(
        self,
        rnn_type: str,
        input_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """Initialize by creating new RNN"""
        if rnn_type is None:
            raise ValueError("Either 'rnn' or 'rnn_type' must be provided")

        self.rnn_type = rnn_type.lower()
        self.rnn = self._create_rnn(
            self.rnn_type,
            input_dim,
            latent_dim,
            num_layers,
            dropout,
            is_decoder=True,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor = None,
        hidden_state=None,
        encoder_output: torch.Tensor = None,
    ) -> tuple:
        """Optimized forward pass with attention bridge integration"""
        batch_size = tgt.size(0)
        device = tgt.device

        # Determine attention source
        attention_memory = encoder_output if encoder_output is not None else memory

        # Process through RNN/Transformer decoder
        output, new_state = self._process_decoder_output(
            self.rnn, self.rnn_type, tgt, memory, hidden_state, batch_size, device
        )

        # Apply cross-attention bridge if available
        if attention_memory is not None:
            output = self._apply_attention_bridge(
                output, attention_memory, self.rnn_type
            )

        return output, new_state

    def get_alphas(self) -> torch.Tensor:
        """Get cached alpha tensor combining decoder and attention alphas"""
        if self._cached_alphas is None:
            self._cached_alphas = self._compute_combined_alphas()

        return self._cached_alphas

    def _compute_combined_alphas(self) -> torch.Tensor:
        """Compute combined decoder and attention alphas"""
        device = next(self.parameters()).device

        # Get decoder alphas (one-hot for fixed type)
        decoder_alphas = self._get_alpha_for_type(self.rnn_type, device)

        # Add attention alphas if attention bridge is enabled
        if self.use_attention_bridge and hasattr(self, "attention_alphas"):
            attention_alphas = F.softmax(self.attention_alphas, dim=0)
            return torch.cat([decoder_alphas, attention_alphas])
        else:
            return decoder_alphas

    def get_entropy_loss(self) -> torch.Tensor:
        """Get entropy loss from attention bridge only (decoder is fixed)"""
        return self._get_attention_entropy_loss()

    def get_decoder_info(self) -> dict[str, any]:
        """Get comprehensive decoder information"""
        base_info = self.get_rnn_info(self.rnn)
        base_info.update(
            {
                "decoder_type": "fixed",
                "architecture": self.rnn_type,
                "is_learnable": False,
                "attention_bridge_enabled": self.use_attention_bridge,
                "attention_layers": (
                    len(self.attention_bridges)
                    if hasattr(self, "attention_bridges")
                    else 0
                ),
            }
        )
        return base_info

    def set_temperature(self, temp: float):
        """Set temperature for attention bridge (decoder architecture is fixed)"""
        self.temperature = temp
        self._cached_alphas = None  # Invalidate cache

        # Set temperature for attention bridge if it exists
        if hasattr(self, "attention_bridges"):
            for bridge in self.attention_bridges:
                if hasattr(bridge, "set_temperature"):
                    bridge.set_temperature(temp)

    def clear_cache(self):
        """Clear all cached tensors"""
        self._cached_alphas = None
        self.clear_attention_cache()

    def get_decoder_weights(self) -> dict[str, float]:
        """Get current decoder weights (will be one-hot for fixed decoder)"""
        alphas = self.get_alphas()

        weights = {"decoder_type": self.rnn_type}

        # Add attention weights if available
        if self.use_attention_bridge and hasattr(self, "attention_alphas"):
            decoder_alpha_size = 1  # Fixed decoder has single alpha
            attention_alphas = alphas[decoder_alpha_size:]

            attention_names = [
                f"attention_layer_{i}" for i in range(len(attention_alphas) - 1)
            ] + ["no_attention"]
            attention_weights = {
                name: weight.item()
                for name, weight in zip(attention_names, attention_alphas)
            }
            weights["attention_bridge"] = attention_weights

        return weights


class MixedDecoder(BaseDecoder):
    """Streamlined mixed decoder with efficient weight handling"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_attention_bridge: bool = True,
        attention_heads: int = 8,
        attention_layers: int = 1,
        use_temporal_bias: bool = True,
        use_rotary: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=2,
            dropout=dropout,
            temperature=temperature,
            use_attention_bridge=use_attention_bridge,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            use_temporal_bias=use_temporal_bias,
            use_rotary=use_rotary,
        )
        self.seq_len = seq_len

        # Create all decoder types
        self.decoders = nn.ModuleList(
            [
                self._create_rnn(
                    decoder_type,
                    input_dim,
                    latent_dim,
                    self.num_layers,
                    dropout,
                    is_decoder=True,
                )
                for decoder_type in self.rnn_names
            ]
        )
        self.decoder_names = self.rnn_names

        # Architecture parameters
        self.alphas = nn.Parameter(torch.zeros(len(self.decoders)))

        # Caching for inference optimization
        self._decoder_cache = None

    def _get_active_decoders(self, weights: torch.Tensor, threshold: float = 1e-3):
        """Get indices and weights of active decoders"""
        active_indices = (weights > threshold).nonzero(as_tuple=False).squeeze(-1)
        if len(active_indices) == 0:
            # Fallback to first decoder (usually LSTM)
            active_indices = torch.tensor([0], device=weights.device)

        active_weights = weights[active_indices]
        active_weights = active_weights / active_weights.sum()  # Normalize
        return active_indices, active_weights

    def _process_single_decoder(
        self,
        decoder_idx: int,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state,
        batch_size: int,
        device: torch.device,
    ):
        """Process a single decoder and return output and state"""
        decoder = self.decoders[decoder_idx]
        rnn_type = self.rnn_names[decoder_idx]

        return self._process_decoder_output(
            decoder, rnn_type, tgt, memory, hidden_state, batch_size, device
        )

    def _handle_inference_caching(
        self,
        weights: torch.Tensor,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state,
        batch_size: int,
        device: torch.device,
    ):
        """Handle inference with caching for dominant decoder"""
        max_weight = weights.max().item()
        if max_weight <= 0.9:
            return None, None  # No dominant decoder, use normal path

        max_idx = weights.argmax().item()

        # Use cached decoder if available
        if self._decoder_cache is not None and self._decoder_cache[0] == max_idx:
            decoder = self._decoder_cache[1]
            rnn_type = self._decoder_cache[2]
        else:
            decoder = self.decoders[max_idx]
            rnn_type = self.rnn_names[max_idx]
            self._decoder_cache = (max_idx, decoder, rnn_type)

        return self._process_decoder_output(
            decoder, rnn_type, tgt, memory, hidden_state, batch_size, device
        )

    def _combine_decoder_outputs(
        self,
        outputs: List[torch.Tensor],
        states: List,
        active_weights: torch.Tensor,
        active_indices: torch.Tensor,
    ):
        """Efficiently combine outputs from multiple decoders"""
        # Stack and combine outputs
        stacked_outputs = torch.stack(outputs, dim=0)
        combined_output = torch.sum(
            active_weights.view(-1, 1, 1, 1) * stacked_outputs, dim=0
        )

        # Use state from decoder with highest weight
        max_idx = active_weights.argmax().item()
        dominant_state = states[max_idx]
        dominant_rnn_type = self.rnn_names[active_indices[max_idx]]

        return combined_output, dominant_state, dominant_rnn_type

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: Optional[torch.Tensor] = None,
    ):
        """Optimized forward with efficient caching and combination"""
        batch_size = tgt.size(0)
        device = tgt.device
        weights = F.softmax(self.alphas / self.temperature, dim=0)

        attention_memory = encoder_output if encoder_output is not None else memory

        # Try inference caching for dominant decoder
        if not self.training:
            cached_output, cached_state = self._handle_inference_caching(
                weights, tgt, memory, hidden_state, batch_size, device
            )
            if cached_output is not None:
                if attention_memory is not None:
                    cached_output = self._apply_attention_bridge(
                        cached_output, attention_memory, self._decoder_cache[2]
                    )
                return cached_output, cached_state

        # Get active decoders
        active_indices, active_weights = self._get_active_decoders(weights)

        # Process active decoders
        outputs, states = [], []
        for i in active_indices:
            output, state = self._process_single_decoder(
                i.item(), tgt, memory, hidden_state, batch_size, device
            )
            outputs.append(output)
            states.append(state)

        # Combine outputs
        combined_output, dominant_state, dominant_rnn_type = (
            self._combine_decoder_outputs(
                outputs, states, active_weights, active_indices
            )
        )

        # Apply attention bridge if needed
        if attention_memory is not None:
            combined_output = self._apply_attention_bridge(
                combined_output, attention_memory, dominant_rnn_type
            )

        return combined_output, dominant_state

    def get_alphas(self) -> torch.Tensor:
        """Get all architecture parameters including attention alphas"""
        decoder_alphas = F.softmax(self.alphas, dim=0)

        if self.use_attention_bridge and hasattr(self, "attention_alphas"):
            attention_alphas = F.softmax(self.attention_alphas, dim=0)
            return torch.cat([decoder_alphas, attention_alphas])

        return decoder_alphas

    def get_entropy_loss(self) -> torch.Tensor:
        """Compute entropy loss for regularization"""
        # Decoder entropy
        probs = F.softmax(self.alphas / self.temperature, dim=0)
        log_probs = torch.log(torch.clamp(probs, min=1e-8, max=1.0))
        entropy = -(probs * log_probs).sum() * 0.01

        # Add attention entropy if available
        if hasattr(self, "_get_attention_entropy_loss"):
            attention_entropy = self._get_attention_entropy_loss()
            entropy += attention_entropy

        return torch.clamp(entropy, min=0.0, max=1.0)

    def set_temperature(self, temp: float):
        """Set temperature and clear cache"""
        self.temperature = temp
        self._decoder_cache = None

        # Set temperature for attention bridge if it exists
        if hasattr(self, "attention_bridge"):
            self.attention_bridge.set_temperature(temp)

    def clear_cache(self):
        """Clear decoder cache"""
        self._decoder_cache = None

    def get_decoder_weights(self) -> Dict[str, float]:
        """Get current decoder weights for analysis"""
        weights = F.softmax(self.alphas / self.temperature, dim=0)
        return {
            name: weight.item() for name, weight in zip(self.decoder_names, weights)
        }


class CrossAttentionBridge(nn.Module):
    """Streamlined RNN-compatible cross-attention bridge"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_temporal_bias: bool = True,
        use_rotary: bool = False,
        max_seq_len: int = 512,
        rnn_integration_mode: str = "adaptive",
    ):
        super().__init__()

        # Store config and validate heads
        self.d_model = d_model
        self.num_heads = self._find_valid_heads(d_model, num_heads)
        self.head_dim = d_model // self.num_heads
        self.scale = self.head_dim**-0.5
        self.rnn_integration_mode = rnn_integration_mode
        self.use_temporal_bias = use_temporal_bias
        self.use_rotary = use_rotary

        # Core attention components
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Optional components
        if use_temporal_bias:
            self._init_temporal_bias(min(max_seq_len, 128))

        if use_rotary:
            self.rotary_emb = RotaryPositionalEncoding(self.head_dim, max_seq_len)

        # Cache for dimension projections
        self._projection_cache = {}

        self._init_weights()

    def _find_valid_heads(self, d_model: int, num_heads: int) -> int:
        """Find valid number of heads that divides d_model"""
        num_heads = min(num_heads, d_model)
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        return max(1, num_heads)

    def _init_temporal_bias(self, max_len: int):
        """Create temporal bias with reduced memory footprint"""
        bias = torch.zeros(self.num_heads, max_len, max_len)
        positions = torch.arange(max_len).float()
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

        temporal_weight = torch.exp(-distance_matrix * 0.1)
        recency_bias = -distance_matrix * 0.05
        bias[:] = temporal_weight + recency_bias

        self.register_buffer("temporal_bias", bias.unsqueeze(0))

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1 / math.sqrt(3))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def _get_projection(
        self, from_dim: int, to_dim: int, device: torch.device, dtype: torch.dtype
    ) -> nn.Linear:
        """Get or create cached projection layer"""
        key = (from_dim, to_dim)
        if key not in self._projection_cache:
            self._projection_cache[key] = nn.Linear(
                from_dim, to_dim, device=device, dtype=dtype
            )
        return self._projection_cache[key]

    def _ensure_dims(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Ensure tensor has target dimension"""
        if tensor.size(-1) == target_dim:
            return tensor

        proj = self._get_projection(
            tensor.size(-1), target_dim, tensor.device, tensor.dtype
        )
        return proj(tensor)

    def _compute_qkv(self, query_input: torch.Tensor, kv_input: torch.Tensor) -> tuple:
        """Compute Q, K, V tensors efficiently"""
        B, L_q = query_input.shape[:2]
        L_kv = kv_input.shape[1]

        # Compute Q from decoder, K,V from encoder
        q = F.linear(query_input, self.qkv_proj.weight[: self.d_model])
        k = F.linear(kv_input, self.qkv_proj.weight[self.d_model : 2 * self.d_model])
        v = F.linear(kv_input, self.qkv_proj.weight[2 * self.d_model :])

        # Reshape for multi-head attention
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        return q, k, v

    def _apply_temporal_bias(
        self, attn_scores: torch.Tensor, L_q: int, L_kv: int
    ) -> torch.Tensor:
        """Apply temporal bias with interpolation if needed"""
        if not self.use_temporal_bias or not hasattr(self, "temporal_bias"):
            return attn_scores

        bias_size = self.temporal_bias.size(-1)

        if L_q <= bias_size and L_kv <= bias_size:
            # Direct slicing for small sequences
            bias_slice = self.temporal_bias[:, : self.num_heads, :L_q, :L_kv]
            attn_scores += bias_slice
        else:
            # Interpolate for larger sequences
            bias = F.interpolate(
                self.temporal_bias.squeeze(0),
                size=(L_q, L_kv),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)
            attn_scores += bias[:, : self.num_heads]

        return attn_scores

    def _apply_masks(
        self,
        attn_scores: torch.Tensor,
        encoder_mask: torch.Tensor = None,
        decoder_mask: torch.Tensor = None,
        L_q: int = None,
        L_kv: int = None,
    ) -> torch.Tensor:
        """Apply encoder and decoder masks efficiently"""
        # Encoder mask
        if encoder_mask is not None:
            if encoder_mask.dim() == 2:
                mask = encoder_mask.view(
                    encoder_mask.size(0), 1, 1, encoder_mask.size(1)
                )
            elif encoder_mask.dim() == 3:
                mask = encoder_mask.unsqueeze(1)
            else:
                mask = encoder_mask

            if mask.size(-1) == L_kv and mask.size(-2) == 1 and L_q > 1:
                mask = mask.expand(-1, -1, L_q, -1)

            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Decoder mask
        if decoder_mask is not None:
            if decoder_mask.dim() == 2:
                mask = decoder_mask.view(
                    decoder_mask.size(0), 1, decoder_mask.size(1), 1
                )
            else:
                mask = decoder_mask

            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        return attn_scores

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor = None,
        return_attention: bool = False,
        decoder_mask: torch.Tensor = None,
        use_causal_mask: bool = False,
    ) -> tuple:
        """Main forward pass with cross-attention"""

        B, L_q, _ = decoder_hidden.shape
        _, L_kv, _ = encoder_output.shape

        # Early exit for empty sequences
        if B == 0 or L_q == 0 or L_kv == 0:
            return decoder_hidden, None

        # Store original for residual
        residual = decoder_hidden

        try:
            # Ensure dimension compatibility and normalize
            query_input = self.norm(self._ensure_dims(decoder_hidden, self.d_model))
            kv_input = self.norm(self._ensure_dims(encoder_output, self.d_model))

            # Compute Q, K, V
            q, k, v = self._compute_qkv(query_input, kv_input)

            # Apply rotary embeddings if enabled
            if self.use_rotary:
                cos_emb, sin_emb = self.rotary_emb(max(L_q, L_kv), q.device)
                q = self.rotary_emb.apply_rotary_pos_emb(q, cos_emb, sin_emb)
                k = self.rotary_emb.apply_rotary_pos_emb(k, cos_emb, sin_emb)

            # Scaled dot-product attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply temporal bias and masks
            attn_scores = self._apply_temporal_bias(attn_scores, L_q, L_kv)
            attn_scores = self._apply_masks(
                attn_scores, encoder_mask, decoder_mask, L_q, L_kv
            )

            # Attention weights and apply to values
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attended = torch.matmul(attn_weights, v)
            attended = attended.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

            # Output projection and gating
            attended = self.out_proj(attended)
            residual = self._ensure_dims(residual, self.d_model)

            gate = torch.sigmoid(self.gate_proj(attended))
            output = gate * attended + (1 - gate) * residual

            return output, attn_weights.mean(dim=1) if return_attention else None

        except Exception as e:
            print(f"CrossAttentionBridge failed: {e}")
            print(
                f"Shapes - decoder: {decoder_hidden.shape}, encoder: {encoder_output.shape}"
            )
            return self._ensure_dims(residual, self.d_model), None


class RotaryPositionalEncoding(nn.Module):
    """Streamlined rotary positional encoding with efficient caching"""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute and cache frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute embeddings for common sequence lengths
        self._init_cached_embeddings(max_seq_len)

    def _init_cached_embeddings(self, max_len: int):
        """Pre-compute embeddings for efficiency"""
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cached_cos", emb.cos())
        self.register_buffer("cached_sin", emb.sin())

    def _compute_embeddings(self, seq_len: int, device: torch.device) -> tuple:
        """Compute embeddings on-the-fly for longer sequences"""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, seq_len: int, device: torch.device) -> tuple:
        """Generate cos and sin embeddings with efficient caching"""
        if seq_len <= self.cached_cos.size(0):
            # Use cached embeddings
            return (
                self.cached_cos[:seq_len].to(device),
                self.cached_sin[:seq_len].to(device),
            )
        else:
            # Compute on-the-fly for longer sequences
            return self._compute_embeddings(seq_len, device)

    def apply_rotary_pos_emb(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary positional embedding efficiently"""
        seq_len = x.size(-2)
        head_dim = x.size(-1)

        # Ensure dimensions match and reshape for broadcasting
        cos = cos[:seq_len, : head_dim // 2].view(1, 1, seq_len, head_dim // 2)
        sin = sin[:seq_len, : head_dim // 2].view(1, 1, seq_len, head_dim // 2)

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
        if device is None:
            device = self.inv_freq.device
        return self.forward(seq_len, device)


class PositionalEncoding(nn.Module):
    """Optimized positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model

        # Vectorized computation of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(base) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding with automatic device placement"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device, x.dtype)

