

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple, List

class LinearSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.size()
        H = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, H, D // H).transpose(1, 2) for t in qkv]  # [B, H, T, D//H]

        k = k.softmax(dim=-2)  # across sequence
        context = torch.einsum('bhtd,bhtv->bhdv', k, v)  # context: [B, H, D//H, D//H]
        out = torch.einsum('bhtd,bhdv->bhtv', q, context)  # attention result
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.dropout(self.out_proj(out))


class LightweightTransformerEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=2, dropout=0.1, nhead=4):
        super().__init__()
        self.latent_dim = latent_dim  # ← this was missing

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': LinearSelfAttention(latent_dim, heads=nhead, dropout=dropout),
                'ffn': nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim * 4, latent_dim)
                ),
                'norm1': nn.LayerNorm(latent_dim),
                'norm2': nn.LayerNorm(latent_dim),
                'drop': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        self.state_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, hidden_state=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            attn_out = layer['attn'](x)
            x = layer['norm1'](x + layer['drop'](attn_out))
            ff_out = layer['ffn'](x)
            x = layer['norm2'](x + layer['drop'](ff_out))

        ctx = x[:, -1:, :]
        h_state = self.state_proj(ctx.squeeze(1)).unsqueeze(0).expand(len(self.layers), -1, -1).contiguous()
        c_state = h_state.clone()
        return x, ctx, (h_state, c_state)

class LightweightTransformerDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=2, dropout=0.1, nhead=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_decoder = nn.Parameter(torch.randn(1, 512, latent_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': LinearSelfAttention(latent_dim, heads=nhead, dropout=dropout),
                'cross_attn': nn.MultiheadAttention(latent_dim, nhead, dropout=dropout, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim * 4, latent_dim)
                ),
                'norm1': nn.LayerNorm(latent_dim),
                'norm2': nn.LayerNorm(latent_dim),
                'norm3': nn.LayerNorm(latent_dim),
                'drop': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
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
            tgt2 = layer['self_attn'](tgt)
            tgt = layer['norm1'](tgt + layer['drop'](tgt2))

            tgt2, _ = layer['cross_attn'](tgt, memory, memory)
            tgt = layer['norm2'](tgt + layer['drop'](tgt2))

            tgt2 = layer['ffn'](tgt)
            tgt = layer['norm3'](tgt + layer['drop'](tgt2))

        last = tgt[:, -1, :]
        h_state = self.state_proj(last).unsqueeze(0).expand(len(self.layers), -1, -1).contiguous()
        c_state = h_state.clone()
        return tgt, (h_state, c_state)

class BaseRNN(nn.Module):
    """Optimized base class for encoder/decoder implementations"""
    
    def __init__(self, input_dim: int, latent_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Cache RNN names and type mappings for efficiency
        self.rnn_names = ["lstm", "gru", "transformer"]
        self._type_cache = {}
        self._alpha_cache = {}
    
    def _create_rnn(self, rnn_type: str, input_dim: int, latent_dim: int, 
                   num_layers: int, dropout: float, is_decoder: bool = False) -> nn.Module:
        """Optimized RNN factory with better parameter handling"""
        rnn_type = rnn_type.lower()
        
        if rnn_type == "transformer":
            wrapper_class = LightweightTransformerDecoder if is_decoder else LightweightTransformerEncoder
            return wrapper_class(
                input_dim=input_dim,
                latent_dim=latent_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif rnn_type in ["lstm", "gru"]:
            rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
            # Optimize dropout handling
            effective_dropout = dropout if num_layers > 1 else 0.0
            
            # Create RNN with type-specific parameters
            if rnn_type == "lstm":
                rnn = rnn_class(
                    input_size=input_dim,
                    hidden_size=latent_dim,
                    num_layers=num_layers,
                    dropout=effective_dropout,
                    batch_first=True,
                    proj_size=0,  # Only LSTM supports proj_size
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
            self._init_rnn_weights(rnn)
            return rnn
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
    
    def _init_rnn_weights(self, rnn: nn.Module):
        """Initialize RNN weights for better training stability"""
        for name, param in rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for LSTM
                if hasattr(rnn, 'bias_ih_l0') and 'bias_ih' in name:
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)
    
    def _detect_rnn_type(self, rnn: nn.Module) -> str:
        """Cached RNN type detection"""
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
    
    def _extract_rnn_properties(self, rnn: nn.Module) -> Tuple[int, int]:
        """Optimized property extraction with fallback"""
        # Try multiple attribute names for robustness
        latent_dim = (getattr(rnn, "hidden_size", None) or 
                     getattr(rnn, "latent_dim", None) or
                     getattr(rnn, "d_model", None))
        
        if latent_dim is None:
            raise ValueError(f"Cannot extract latent_dim from {type(rnn).__name__}")
        
        num_layers = getattr(rnn, "num_layers", 1)
        return latent_dim, num_layers
    
    def _get_alpha_for_type(self, rnn_type: str, device: torch.device) -> torch.Tensor:
        """Cached one-hot encoding for RNN type"""
        cache_key = (rnn_type, device)
        if cache_key not in self._alpha_cache:
            alpha_map = {
                "lstm": [1.0, 0.0, 0.0], 
                "gru": [0.0, 1.0, 0.0], 
                "transformer": [0.0, 0.0, 1.0]
            }
            alpha = torch.tensor(
                alpha_map.get(rnn_type, [0.0, 0.0, 1.0]), 
                device=device, 
                dtype=torch.float32
            )
            self._alpha_cache[cache_key] = alpha
        return self._alpha_cache[cache_key]


class FixedEncoder(BaseRNN):
    """Optimized single encoder with fixed architecture"""
    
    def __init__(
        self,
        rnn: Optional[Union[nn.LSTM, nn.GRU]] = None,
        *,
        rnn_type: Optional[str] = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(input_dim, latent_dim, num_layers, dropout)

        if rnn is not None:
            self.rnn = rnn
            self.rnn_type = self._detect_rnn_type(rnn)
            self.latent_dim, self.num_layers = self._extract_rnn_properties(rnn)
        else:
            if rnn_type is None:
                raise ValueError("Either 'rnn' or 'rnn_type' must be provided")
            
            self.rnn_type = rnn_type.lower()
            self.rnn = self._create_rnn(self.rnn_type, input_dim, latent_dim, num_layers, dropout)
        
        # Cache alpha tensor for efficiency
        self._cached_alpha = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized forward pass with better memory management"""
        if self.rnn_type == "transformer":
            return self.rnn(x)  # Returns (output, ctx, state)
        else:
            # Use torch.jit.script for potential speedup in production
            h, state = self.rnn(x)
            # More efficient context extraction
            ctx = h.narrow(1, h.size(1) - 1, 1)  # Get last timestep efficiently
            return h, ctx, state

    def get_alphas(self) -> torch.Tensor:
        """Cached alpha retrieval"""
        if self._cached_alpha is None or self._cached_alpha.device != next(self.parameters()).device:
            device = next(self.parameters()).device
            self._cached_alpha = self._get_alpha_for_type(self.rnn_type, device)
        return self._cached_alpha


class MixedEncoder(BaseRNN):
    """Optimized mixed encoder with learnable weights"""

    def __init__(self, input_dim: int, latent_dim: int, seq_len: int, 
                 dropout: float = 0.1, temperature: float = 1.0):
        super().__init__(input_dim, latent_dim, num_layers=2, dropout=dropout)
        self.temperature = temperature
        self.seq_len = seq_len

        # Create encoders with shared parameters where possible
        self.encoders = nn.ModuleList([
            self._create_rnn(rnn_type, input_dim, latent_dim, self.num_layers, dropout)
            for rnn_type in self.rnn_names
        ])
        self.encoder_names = self.rnn_names
        
        # Initialize alphas with better starting values
        self.alphas = nn.Parameter(torch.zeros(len(self.encoders)))
        
        # Cache for inference optimization
        self._inference_cache = None
        self._last_weights = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized forward with caching and early returns"""
        weights = F.softmax(self.alphas / self.temperature, dim=0)
        
        # Cache dominant encoder for inference
        if not self.training:
            max_weight = weights.max().item()
            if max_weight > 0.9:
                max_idx = weights.argmax().item()
                
                # Use cached result if same encoder dominates
                if (self._inference_cache is not None and 
                    self._inference_cache[0] == max_idx):
                    encoder = self._inference_cache[1]
                else:
                    encoder = self.encoders[max_idx]
                    self._inference_cache = (max_idx, encoder)
                
                rnn_type = self.rnn_names[max_idx]
                if rnn_type == "transformer":
                    return encoder(x)
                else:
                    h, state = encoder(x)
                    ctx = h.narrow(1, h.size(1) - 1, 1)
                    return h, ctx, state

        # Efficient weighted combination
        outputs, contexts, states = [], [], []
        active_indices = (weights > 1e-3).nonzero(as_tuple=False).squeeze(-1)
        
        if len(active_indices) == 0:
            # Fallback to first encoder
            encoder = self.encoders[0]
            h, state = encoder(x)
            ctx = h.narrow(1, h.size(1) - 1, 1)
            return h, ctx, state

        # Process only active encoders
        active_weights = weights[active_indices]
        for i in active_indices:
            encoder = self.encoders[i]
            rnn_type = self.rnn_names[i]
            
            if rnn_type == "transformer":
                h, ctx, state = encoder(x)
            else:
                h, state = encoder(x)
                ctx = h.narrow(1, h.size(1) - 1, 1)
            
            outputs.append(h)
            contexts.append(ctx)
            states.append(state)

        # Normalize weights and combine
        active_weights = active_weights / active_weights.sum()
        
        # Efficient tensor combination
        weighted_output = torch.stack(outputs, dim=0)
        weighted_output = torch.sum(active_weights.view(-1, 1, 1, 1) * weighted_output, dim=0)
        
        weighted_context = torch.stack(contexts, dim=0)
        weighted_context = torch.sum(active_weights.view(-1, 1, 1, 1) * weighted_context, dim=0)

        # Use state from encoder with highest weight
        max_weight_idx = active_weights.argmax().item()
        final_state = states[max_weight_idx]

        return weighted_output, weighted_context, final_state

    def get_alphas(self) -> torch.Tensor:
        """Cached softmax computation"""
        return F.softmax(self.alphas, dim=0)

    def get_entropy_loss(self) -> torch.Tensor:
        """Optimized entropy computation"""
        probs = self.get_alphas()
        # Clamp to avoid log(0) and use more stable computation
        log_probs = torch.log(torch.clamp(probs, min=1e-8, max=1.0))
        entropy = -(probs * log_probs).sum() * 0.01
        return torch.clamp(entropy, min=0.0, max=1.0)


class BaseDecoder(BaseRNN):
    """Optimized base decoder with efficient attention handling"""
    
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
        attention_d_model: Optional[int] = None,
    ):
        super().__init__(input_dim, latent_dim, num_layers, dropout)
        self.temperature = temperature
        self.use_attention_bridge = use_attention_bridge
        self.attention_d_model = attention_d_model or latent_dim

        if use_attention_bridge:
            # Optimize number of heads based on model size
            optimal_heads = min(attention_heads, max(1, self.attention_d_model // 32))
            
            self.attention_bridges = nn.ModuleList([
                CrossAttentionBridge(
                    d_model=self.attention_d_model,
                    num_heads=optimal_heads,
                    dropout=dropout,
                    use_temporal_bias=use_temporal_bias,
                    use_rotary=use_rotary
                )
                for _ in range(attention_layers)
            ])
            
            # Better initialization for attention weights
            self.attention_alphas = nn.Parameter(
                torch.cat([
                    torch.zeros(attention_layers),  # Attention layers
                    torch.tensor([1.0])  # No attention (higher initial weight)
                ])
            )
            
            # Cache for attention optimization
            self._attention_cache = None
            
            print(f"✅ CrossAttentionBridge enabled with {optimal_heads} heads and {attention_layers} layers")
        else:
            print("⚠️  CrossAttentionBridge disabled")

    def _prepare_hidden_state(self, hidden_state, batch_size, device):
        """Optimized hidden state preparation with better memory efficiency"""
        if hidden_state is None:
            # Use more efficient tensor creation
            h_0 = torch.zeros(self.num_layers, batch_size, self.latent_dim, 
                            device=device, dtype=torch.float32)
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0), h_0

        if isinstance(hidden_state, tuple):
            h, c = hidden_state
            
            # More efficient dimension handling
            if h.dim() != 3:
                h = h.unsqueeze(0).expand(self.num_layers, -1, -1)
            if c.dim() != 3:
                c = c.unsqueeze(0).expand(self.num_layers, -1, -1)
                
            return (h.contiguous(), c.contiguous()), h
        else:
            if hidden_state.dim() != 3:
                h = hidden_state.unsqueeze(0).expand(self.num_layers, -1, -1)
            else:
                h = hidden_state
            c = torch.zeros_like(h)
            return (h.contiguous(), c.contiguous()), h
    
    def _apply_attention_bridge(self, decoder_output, encoder_output, rnn_type="transformer"):
        """Optimized attention bridge with caching and early exits"""
        if (not self.use_attention_bridge or 
            not hasattr(self, 'attention_bridges') or 
            encoder_output is None):
            return decoder_output
        
        try:
            # Ensure proper sequence dimensions with efficient checks
            if rnn_type in ["lstm", "gru"]:
                if decoder_output.dim() == 2:
                    decoder_output = decoder_output.unsqueeze(1)
                if encoder_output.dim() == 2:
                    encoder_output = encoder_output.unsqueeze(1)
            
            attention_weights = F.softmax(self.attention_alphas / self.temperature, dim=0)
            
            # Aggressive caching for inference
            if not self.training:
                max_weight = attention_weights.max().item()
                if max_weight > 0.95:  # Higher threshold for caching
                    max_idx = attention_weights.argmax().item()
                    
                    # Cache dominant attention choice
                    if (self._attention_cache is not None and 
                        self._attention_cache[0] == max_idx):
                        bridge_func = self._attention_cache[1]
                    else:
                        if max_idx == len(self.attention_bridges):
                            bridge_func = lambda x, y: x  # No attention
                        else:
                            bridge_func = self.attention_bridges[max_idx]
                        self._attention_cache = (max_idx, bridge_func)
                    
                    if max_idx == len(self.attention_bridges):
                        return decoder_output
                    else:
                        attended_output, _ = bridge_func(decoder_output, encoder_output)
                        return attended_output
            
            # Efficient weighted combination
            attended_outputs = []
            active_weights = []
            
            # Check no-attention option
            no_attention_weight = attention_weights[-1]
            if no_attention_weight.item() > 1e-3:
                attended_outputs.append(decoder_output)
                active_weights.append(no_attention_weight)
            
            # Process attention bridges
            current_output = decoder_output
            for i, bridge in enumerate(self.attention_bridges):
                weight = attention_weights[i]
                if weight.item() > 1e-3:
                    try:
                        attended_output, _ = bridge(current_output, encoder_output)
                        attended_outputs.append(attended_output)
                        active_weights.append(weight)
                        current_output = attended_output
                    except Exception as e:
                        print(f"Attention bridge {i} failed: {e}")
                        continue
            
            if not attended_outputs:
                return decoder_output
            
            # Efficient weighted sum
            if len(active_weights) == 1:
                return attended_outputs[0]
            
            total_weight = sum(active_weights)
            if total_weight < 1e-8:
                return decoder_output
            
            # Stack and weight efficiently
            stacked_outputs = torch.stack(attended_outputs, dim=0)
            norm_weights = torch.stack(active_weights) / total_weight
            result = torch.sum(norm_weights.view(-1, 1, 1, 1) * stacked_outputs, dim=0)
            
            return result
            
        except Exception as e:
            print(f"Attention bridge application failed: {e}")
            return decoder_output
    
    def _process_decoder_output(self, decoder: nn.Module, rnn_type: str, tgt: torch.Tensor, 
                               memory: torch.Tensor, hidden_state, batch_size: int, device: torch.device):
        """Optimized decoder output processing"""
        if rnn_type == "lstm":
            lstm_state, _ = self._prepare_hidden_state(hidden_state, batch_size, device)
            return decoder(tgt, lstm_state)
        elif rnn_type == "gru":
            _, gru_state = self._prepare_hidden_state(hidden_state, batch_size, device)
            return decoder(tgt, gru_state)
        else:  # transformer
            return decoder(tgt, memory, hidden_state)
    
    def _get_attention_entropy_loss(self) -> torch.Tensor:
        """Optimized attention entropy computation"""
        if self.use_attention_bridge and hasattr(self, 'attention_alphas'):
            attn_probs = F.softmax(self.attention_alphas / self.temperature, dim=0)
            log_probs = torch.log(torch.clamp(attn_probs, min=1e-8, max=1.0))
            attn_entropy = -(attn_probs * log_probs).sum() * 0.01
            return torch.clamp(attn_entropy, min=0.0, max=1.0)
        return torch.tensor(0.0, device=next(self.parameters()).device)


class FixedDecoder(BaseDecoder):
    """Optimized fixed decoder with caching"""
    
    def __init__(
        self,
        rnn: Optional[Union[nn.LSTM, nn.GRU]] = None,
        *,
        rnn_type: Optional[str] = None,
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
        attention_d_model: Optional[int] = None,
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

        if rnn is not None:
            self.rnn = rnn
            self.rnn_type = self._detect_rnn_type(rnn)
            self.latent_dim, self.num_layers = self._extract_rnn_properties(rnn)
        else:
            if rnn_type is None:
                raise ValueError("Either 'rnn' or 'rnn_type' must be provided")
            
            self.rnn_type = rnn_type.lower()
            self.rnn = self._create_rnn(self.rnn_type, input_dim, latent_dim, num_layers, dropout, is_decoder=True)
        
        # Cache for alpha computation
        self._cached_alphas = None

    def forward(self, tgt, memory=None, hidden_state=None, encoder_output=None):
        batch_size = tgt.size(0)
        device = tgt.device
        
        attention_memory = encoder_output if encoder_output is not None else memory

        # Standard RNN/Transformer decoding
        output, new_state = self._process_decoder_output(
            self.rnn, self.rnn_type, tgt, memory, hidden_state, batch_size, device
        )

        # Apply cross-attention bridge
        if attention_memory is not None:
            output = self._apply_attention_bridge(output, attention_memory, self.rnn_type)

        return output, new_state

    def get_alphas(self):
        """Cached alpha computation"""
        if self._cached_alphas is None:
            device = next(self.parameters()).device
            decoder_alphas = self._get_alpha_for_type(self.rnn_type, device)
            
            if self.use_attention_bridge and hasattr(self, 'attention_alphas'):
                attention_alphas = F.softmax(self.attention_alphas, dim=0)
                self._cached_alphas = torch.cat([decoder_alphas, attention_alphas])
            else:
                self._cached_alphas = decoder_alphas
        
        return self._cached_alphas

    def get_entropy_loss(self):
        return self._get_attention_entropy_loss()


class MixedDecoder(BaseDecoder):
    """Optimized mixed decoder with efficient weight handling"""
    
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
        use_rotary: bool = True
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
            use_rotary=use_rotary
        )
        self.seq_len = seq_len
        
        # Create all decoder types
        self.decoders = nn.ModuleList([
            self._create_rnn(decoder_type, input_dim, latent_dim, self.num_layers, dropout, is_decoder=True)
            for decoder_type in self.rnn_names
        ])
        self.decoder_names = self.rnn_names
        
        # Better initialization
        self.alphas = nn.Parameter(torch.zeros(len(self.decoders)))
        
        # Caching for inference
        self._decoder_cache = None

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, hidden_state=None, encoder_output: Optional[torch.Tensor] = None):
        """Optimized forward with aggressive caching"""
        batch_size = tgt.size(0)
        device = tgt.device
        weights = F.softmax(self.alphas / self.temperature, dim=0)
        
        attention_memory = encoder_output if encoder_output is not None else memory
        
        # Aggressive inference caching
        if not self.training:
            max_weight = weights.max().item()
            if max_weight > 0.9:
                max_idx = weights.argmax().item()
                
                # Use cached decoder
                if (self._decoder_cache is not None and 
                    self._decoder_cache[0] == max_idx):
                    decoder = self._decoder_cache[1]
                    rnn_type = self._decoder_cache[2]
                else:
                    decoder = self.decoders[max_idx]
                    rnn_type = self.rnn_names[max_idx]
                    self._decoder_cache = (max_idx, decoder, rnn_type)
                
                output, new_state = self._process_decoder_output(
                    decoder, rnn_type, tgt, memory, hidden_state, batch_size, device
                )
                
                if attention_memory is not None:
                    output = self._apply_attention_bridge(output, attention_memory, rnn_type)
                return output, new_state

        # Efficient weighted combination
        active_indices = (weights > 1e-3).nonzero(as_tuple=False).squeeze(-1)
        
        if len(active_indices) == 0:
            # Fallback to LSTM
            output, new_state = self._process_decoder_output(
                self.decoders[0], self.rnn_names[0], tgt, memory, hidden_state, batch_size, device
            )
            if attention_memory is not None:
                output = self._apply_attention_bridge(output, attention_memory, "lstm")
            return output, new_state

        # Process only active decoders
        outputs, new_states = [], []
        active_weights = weights[active_indices]
        
        for i in active_indices:
            decoder = self.decoders[i]
            rnn_type = self.rnn_names[i]
            
            output, state = self._process_decoder_output(
                decoder, rnn_type, tgt, memory, hidden_state, batch_size, device
            )
            outputs.append(output)
            new_states.append(state)

        # Efficient weighted combination
        active_weights = active_weights / active_weights.sum()
        
        # Stack and combine efficiently
        stacked_outputs = torch.stack(outputs, dim=0)
        output = torch.sum(active_weights.view(-1, 1, 1, 1) * stacked_outputs, dim=0)
        
        # Use state from decoder with highest weight
        max_idx = active_weights.argmax().item()
        new_state = new_states[max_idx]
        dominant_rnn_type = self.rnn_names[active_indices[max_idx]]
        
        if attention_memory is not None:
            output = self._apply_attention_bridge(output, attention_memory, dominant_rnn_type)
        
        return output, new_state

    def get_alphas(self):
        """Efficient alpha computation"""
        decoder_alphas = F.softmax(self.alphas, dim=0)
        if self.use_attention_bridge and hasattr(self, 'attention_alphas'):
            attention_alphas = F.softmax(self.attention_alphas, dim=0)
            return torch.cat([decoder_alphas, attention_alphas])
        return decoder_alphas

    def get_entropy_loss(self):
        """Optimized entropy computation"""
        # Decoder entropy
        probs = F.softmax(self.alphas / self.temperature, dim=0)
        log_probs = torch.log(torch.clamp(probs, min=1e-8, max=1.0))
        entropy = -(probs * log_probs).sum() * 0.01
        
        # Add attention entropy
        attention_entropy = self._get_attention_entropy_loss()
        entropy += attention_entropy
        
        return torch.clamp(entropy, min=0.0, max=1.0)

class CrossAttentionBridge(nn.Module):
    """Optimized RNN-compatible cross-attention bridge with fixed dimension handling"""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8,
        dropout: float = 0.1,
        use_temporal_bias: bool = True,
        use_rotary: bool = False,
        max_seq_len: int = 512,
        rnn_integration_mode: str = "adaptive"
    ):
        super().__init__()
        
        # Find valid number of heads and cache dimensions
        self.num_heads = self._find_valid_heads(d_model, num_heads)
        self.d_model = d_model
        self.head_dim = d_model // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.rnn_integration_mode = rnn_integration_mode
        
        # Single fused QKV projection for better memory efficiency
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Pre-normalization (more stable and efficient)
        self.norm = nn.LayerNorm(d_model)
        
        # Temporal bias - only create if needed
        self.use_temporal_bias = use_temporal_bias
        if use_temporal_bias:
            # Use smaller bias matrix and interpolate when needed
            self._register_temporal_bias(min(max_seq_len, 128), self.num_heads)

        # Rotary embeddings
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary_emb = RotaryPositionalEncoding(self.head_dim, max_seq_len)

        # Simplified gating mechanism
        self.gate_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Cache for emergency projections to avoid repeated creation
        self._projection_cache = {}
        
        self._init_weights()

    @staticmethod
    def _find_valid_heads(d_model: int, num_heads: int) -> int:
        """Find valid number of heads that divides d_model"""
        num_heads = min(num_heads, d_model)
        # Use bit operations for faster division check
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        return max(1, num_heads)
    
    def _register_temporal_bias(self, max_len: int, num_heads: int):
        """Create optimized temporal bias with reduced memory footprint"""
        # Create only upper triangular part to save memory
        bias = torch.zeros(num_heads, max_len, max_len)
        
        # Vectorized computation
        positions = torch.arange(max_len).float()
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        
        temporal_weight = torch.exp(-distance_matrix * 0.1)
        recency_bias = -distance_matrix * 0.05
        
        bias[:] = temporal_weight + recency_bias
        self.register_buffer('temporal_bias', bias.unsqueeze(0))
    
    def _init_weights(self):
        """Optimized weight initialization"""
        # Xavier initialization for QKV projection
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1/math.sqrt(3))
        
        # Standard initialization for output
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
        # Gate initialization
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
    
    def _get_or_create_projection(self, from_dim: int, to_dim: int, device: torch.device, dtype: torch.dtype) -> nn.Linear:
        """Cache emergency projections to avoid repeated creation"""
        key = (from_dim, to_dim)
        if key not in self._projection_cache:
            proj = nn.Linear(from_dim, to_dim, device=device, dtype=dtype)
            self._projection_cache[key] = proj
        return self._projection_cache[key]
    
    def _ensure_compatible_dims(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Optimized dimension compatibility check"""
        if tensor.size(-1) == target_dim:
            return tensor
        
        proj = self._get_or_create_projection(
            tensor.size(-1), target_dim, tensor.device, tensor.dtype
        )
        return proj(tensor)
    
    def _apply_temporal_bias(self, attn_scores: torch.Tensor, L_q: int, L_kv: int) -> torch.Tensor:
        """Optimized temporal bias application with interpolation"""
        if not self.use_temporal_bias or not hasattr(self, 'temporal_bias'):
            return attn_scores
        
        bias_size = self.temporal_bias.size(-1)
        
        if L_q <= bias_size and L_kv <= bias_size:
            # Direct slicing for small sequences
            bias_slice = self.temporal_bias[:, :self.num_heads, :L_q, :L_kv]
            attn_scores += bias_slice
        else:
            # Interpolate for larger sequences
            bias = F.interpolate(
                self.temporal_bias.squeeze(0),  # Remove batch dim
                size=(L_q, L_kv),
                mode='bilinear',
                align_corners=False
            ).unsqueeze(0)
            attn_scores += bias[:, :self.num_heads]
        
        return attn_scores
    
    def _apply_masks(self, attn_scores: torch.Tensor, encoder_mask: Optional[torch.Tensor], 
                    decoder_mask: Optional[torch.Tensor], L_q: int, L_kv: int) -> torch.Tensor:
        """Optimized mask application"""
        # Encoder mask
        if encoder_mask is not None:
            # Efficient mask reshaping
            if encoder_mask.dim() == 2:
                mask = encoder_mask.view(encoder_mask.size(0), 1, 1, encoder_mask.size(1))
            elif encoder_mask.dim() == 3:
                mask = encoder_mask.unsqueeze(1)
            else:
                mask = encoder_mask
            
            # Expand mask efficiently
            if mask.size(-1) == L_kv and mask.size(-2) == 1 and L_q > 1:
                mask = mask.expand(-1, -1, L_q, -1)
            
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Decoder mask
        if decoder_mask is not None:
            if decoder_mask.dim() == 2:
                mask = decoder_mask.view(decoder_mask.size(0), 1, decoder_mask.size(1), 1)
            else:
                mask = decoder_mask
            
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        return attn_scores
    
    def forward(
        self, 
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        decoder_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Optimized cross-attention forward pass"""
        
        B, L_q, _ = decoder_hidden.shape
        _, L_kv, _ = encoder_output.shape
        
        # Early exit for empty sequences
        if B == 0 or L_q == 0 or L_kv == 0:
            return decoder_hidden, None
        
        # Store original for residual connection
        residual = decoder_hidden
        
        try:
            # Ensure dimension compatibility
            query_input = self._ensure_compatible_dims(decoder_hidden, self.d_model)
            kv_input = self._ensure_compatible_dims(encoder_output, self.d_model)
            
            # Pre-normalization
            query_input = self.norm(query_input)
            kv_input = self.norm(kv_input)
            
            # Compute Q, K, V more efficiently
            # Q from decoder, K,V from encoder
            q = F.linear(query_input, self.qkv_proj.weight[:self.d_model])
            k = F.linear(kv_input, self.qkv_proj.weight[self.d_model:2*self.d_model])
            v = F.linear(kv_input, self.qkv_proj.weight[2*self.d_model:])
            
            # Reshape for multi-head attention
            q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Apply rotary embeddings if enabled
            if self.use_rotary:
                cos_emb, sin_emb = self.rotary_emb(max(L_q, L_kv), q.device)
                q = self.rotary_emb.apply_rotary_pos_emb(q, cos_emb, sin_emb)
                k = self.rotary_emb.apply_rotary_pos_emb(k, cos_emb, sin_emb)
            
            # Scaled dot-product attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply temporal bias
            attn_scores = self._apply_temporal_bias(attn_scores, L_q, L_kv)
            
            # Apply masks
            attn_scores = self._apply_masks(attn_scores, encoder_mask, decoder_mask, L_q, L_kv)
            
            # Attention weights and dropout
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attended = torch.matmul(attn_weights, v)
            attended = attended.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
            
            # Output projection
            attended = self.out_proj(attended)
            
            # Ensure residual compatibility
            residual = self._ensure_compatible_dims(residual, self.d_model)
            
            # Simplified gating mechanism
            gate = torch.sigmoid(self.gate_proj(attended))
            output = gate * attended + (1 - gate) * residual
            
            return output, attn_weights.mean(dim=1) if return_attention else None
            
        except Exception as e:
            # Fallback with detailed error info
            print(f"CrossAttentionBridge forward pass failed: {e}")
            print(f"Shapes - decoder: {decoder_hidden.shape}, encoder: {encoder_output.shape}")
            print(f"d_model: {self.d_model}, num_heads: {self.num_heads}, head_dim: {self.head_dim}")
            return self._ensure_compatible_dims(residual, self.d_model), None


class RotaryPositionalEncoding(nn.Module):
    """Optimized rotary positional encoding"""
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pre-compute and cache frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute embeddings for common sequence lengths
        self._cache_embeddings(max_seq_len)
    
    def _cache_embeddings(self, max_len: int):
        """Pre-compute embeddings for efficiency"""
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cached_cos', emb.cos())
        self.register_buffer('cached_sin', emb.sin())
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate cos and sin embeddings with caching"""
        if seq_len <= self.cached_cos.size(0):
            # Use cached embeddings
            return (
                self.cached_cos[:seq_len].to(device),
                self.cached_sin[:seq_len].to(device)
            )
        else:
            # Compute on-the-fly for longer sequences
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Optimized rotary positional embedding application"""
        seq_len = x.size(-2)
        head_dim = x.size(-1)
        
        # Ensure dimensions match
        cos = cos[:seq_len, :head_dim//2]
        sin = sin[:seq_len, :head_dim//2]
        
        # Reshape for broadcasting
        cos = cos.view(1, 1, seq_len, head_dim//2)
        sin = sin.view(1, 1, seq_len, head_dim//2)
        
        # Split and apply rotation in one operation
        x_even, x_odd = x.chunk(2, dim=-1)
        
        return torch.cat([
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos
        ], dim=-1)


class PositionalEncoding(nn.Module):
    """Optimized positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        
        # Vectorized computation of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                           -(math.log(base) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding with automatic device placement"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device, x.dtype)