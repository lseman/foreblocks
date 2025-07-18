import copy
from typing import Callable, Dict, Optional, Tuple, Union, List
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizationConfig:
    """Configuration for manual quantization"""
    def __init__(self, 
                 bit_width: int = 8,
                 symmetric: bool = True,
                 per_channel: bool = False,
                 observer_type: str = "minmax"):
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.observer_type = observer_type
        
        # Calculate quantization parameters
        self.qmin = -(2 ** (bit_width - 1)) if symmetric else 0
        self.qmax = 2 ** (bit_width - 1) - 1 if symmetric else 2 ** bit_width - 1


class QuantizationObserver(nn.Module):
    """Manual quantization observer to collect statistics"""
    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('num_batches', torch.tensor(0))
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled and self.training:
            self._update_stats(x)
        return x
    
    def _update_stats(self, x: torch.Tensor):
        """Update min/max statistics"""
        if self.config.per_channel:
            # Per-channel quantization (for Conv2d/Linear weights)
            if x.dim() >= 2:
                dims = list(range(x.dim()))
                dims.remove(0)  # Keep channel dimension
                current_min = torch.min(x, dim=dims, keepdim=True)[0].flatten()
                current_max = torch.max(x, dim=dims, keepdim=True)[0].flatten()
            else:
                current_min = torch.min(x)
                current_max = torch.max(x)
        else:
            # Per-tensor quantization
            current_min = torch.min(x)
            current_max = torch.max(x)
        
        # Update running statistics
        self.min_val = torch.min(self.min_val, current_min)
        self.max_val = torch.max(self.max_val, current_max)
        self.num_batches += 1
    
    def calculate_scale_zero_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantization scale and zero point"""
        if self.config.symmetric:
            # Symmetric quantization
            abs_max = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
            scale = abs_max / (2 ** (self.config.bit_width - 1) - 1)
            zero_point = torch.zeros_like(scale, dtype=torch.long)
        else:
            # Asymmetric quantization
            scale = (self.max_val - self.min_val) / (self.config.qmax - self.config.qmin)
            zero_point = self.config.qmin - torch.round(self.min_val / scale)
            zero_point = torch.clamp(zero_point, self.config.qmin, self.config.qmax).long()
        
        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        return scale, zero_point
class QuantizedLinear(nn.Module):
    """Fixed static quantized Linear layer with proper device handling"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config: 'QuantizationConfig' = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        
        # Quantized weights and bias - these need to be on the same device
        self.register_buffer('weight_int', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0, dtype=torch.long))
        
        if bias:
            self.register_buffer('bias_int', torch.zeros(out_features, dtype=torch.int32))
            self.register_buffer('bias_scale', torch.tensor(1.0))
        else:
            self.register_buffer('bias_int', None)
            self.register_buffer('bias_scale', None)
        
        # Input quantization parameters
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('input_zero_point', torch.tensor(0, dtype=torch.long))
        
        # Output quantization parameters
        self.register_buffer('output_scale', torch.tensor(1.0))
        self.register_buffer('output_zero_point', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure all tensors are on the same device as input
        device = x.device
        
        # Move quantization parameters to input device if needed
        input_scale = self.input_scale.to(device)
        input_zero_point = self.input_zero_point.to(device)
        weight_int = self.weight_int.to(device)
        bias_int = self.bias_int.to(device) if self.bias_int is not None else None
        output_scale = self.output_scale.to(device)
        output_zero_point = self.output_zero_point.to(device)
        
        # Quantize input
        x_int = torch.round(x / input_scale) + input_zero_point
        x_int = torch.clamp(x_int, self.config.qmin, self.config.qmax).to(torch.int8)
        
        # Quantized linear operation - ensure all tensors are on same device
        output_int = F.linear(
            x_int.float(), 
            weight_int.float(), 
            bias_int.float() if bias_int is not None else None
        )
        
        # Dequantize output
        output = (output_int - output_zero_point) * output_scale
        
        return output
    
    def to(self, device_or_dtype):
        """Override to method to ensure all buffers move together"""
        result = super().to(device_or_dtype)
        return result
    
    @classmethod
    def from_float(cls, float_module: nn.Linear, config: 'QuantizationConfig' = None):
        """Convert float linear layer to quantized version"""
        
        config = config or QuantizationConfig()
        quantized_module = cls(float_module.in_features, float_module.out_features, 
                             float_module.bias is not None, config)
        
        # Get device from float module
        device = next(float_module.parameters()).device
        
        # Quantize weights
        weight_observer = QuantizationObserver(config)
        weight_observer._update_stats(float_module.weight)
        weight_scale, weight_zero_point = weight_observer.calculate_scale_zero_point()
        
        weight_int = torch.round(float_module.weight / weight_scale) + weight_zero_point
        weight_int = torch.clamp(weight_int, config.qmin, config.qmax).to(torch.int8)
        
        # Ensure all tensors are on the same device
        quantized_module.weight_int.copy_(weight_int.to(device))
        quantized_module.weight_scale.copy_(weight_scale.to(device))
        quantized_module.weight_zero_point.copy_(weight_zero_point.to(device))
        
        # Quantize bias if present
        if float_module.bias is not None:
            bias_scale = weight_scale  # Typically same as weight scale
            bias_int = torch.round(float_module.bias / bias_scale).to(torch.int32)
            quantized_module.bias_int.copy_(bias_int.to(device))
            quantized_module.bias_scale.copy_(bias_scale.to(device))
        
        # Move entire module to device
        quantized_module = quantized_module.to(device)
        
        return quantized_module


class DynamicQuantizedLinear(nn.Module):
    """Fixed dynamic quantized Linear layer with proper device handling"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config: 'QuantizationConfig' = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        
        # Statically quantized weights
        self.register_buffer('weight_int', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0, dtype=torch.long))
        
        # Float bias (not quantized in dynamic quantization)
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure all tensors are on the same device as input
        device = x.device
        
        # Move quantization parameters to input device if needed
        weight_int = self.weight_int.to(device)
        weight_scale = self.weight_scale.to(device)
        bias = self.bias.to(device) if self.bias is not None else None
        
        # Dynamically quantize input activations
        from .quantization import QuantizationObserver
        x_observer = QuantizationObserver(self.config)
        x_observer._update_stats(x)
        input_scale, input_zero_point = x_observer.calculate_scale_zero_point()
        
        # Move observer outputs to correct device
        input_scale = input_scale.to(device)
        input_zero_point = input_zero_point.to(device)
        
        # Quantize input
        x_int = torch.round(x / input_scale) + input_zero_point
        x_int = torch.clamp(x_int, self.config.qmin, self.config.qmax).to(torch.int8)
        
        # Quantized linear operation
        output = F.linear(x_int.float(), weight_int.float(), bias)
        
        # Scale output back
        output = output * input_scale * weight_scale
        
        return output
    
    @classmethod
    def from_float(cls, float_module: nn.Linear, config: 'QuantizationConfig' = None):
        
        config = config or QuantizationConfig()
        quantized_module = cls(float_module.in_features, float_module.out_features, 
                             float_module.bias is not None, config)
        
        # Get device from float module
        device = next(float_module.parameters()).device
        
        # Quantize weights statically
        weight_observer = QuantizationObserver(config)
        weight_observer._update_stats(float_module.weight)
        weight_scale, weight_zero_point = weight_observer.calculate_scale_zero_point()
        
        weight_int = torch.round(float_module.weight / weight_scale) + weight_zero_point
        weight_int = torch.clamp(weight_int, config.qmin, config.qmax).to(torch.int8)
        
        # Ensure all tensors are on the same device
        quantized_module.weight_int.copy_(weight_int.to(device))
        quantized_module.weight_scale.copy_(weight_scale.to(device))
        quantized_module.weight_zero_point.copy_(weight_zero_point.to(device))
        
        # Keep bias as float
        if float_module.bias is not None:
            quantized_module.bias.copy_(float_module.bias.to(device))
        
        # Move entire module to device
        quantized_module = quantized_module.to(device)
        
        return quantized_module



class FakeQuantize(nn.Module):
    """Fake quantization for QAT"""
    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.observer = QuantizationObserver(config)
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0, dtype=torch.long))
        self.fake_quant_enabled = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update observer
        x = self.observer(x)
        
        if self.fake_quant_enabled:
            # Calculate current scale and zero point
            scale, zero_point = self.observer.calculate_scale_zero_point()
            
            # Apply fake quantization
            x = self._fake_quantize(x, scale, zero_point)
        
        return x
    
    def _fake_quantize(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization"""
        scale = scale.to(x.device)
        zero_point = zero_point.to(x.device)
        
        # Quantize
        x_int = torch.round(x / scale) + zero_point
        
        # Clamp to quantization range
        x_int = torch.clamp(x_int, self.config.qmin, self.config.qmax)
        
        # Dequantize
        x_fake_quant = (x_int - zero_point) * scale
        
        return x_fake_quant
    
    def calculate_qparams(self):
        """Calculate and store quantization parameters"""
        self.scale, self.zero_point = self.observer.calculate_scale_zero_point()
        
    def disable_observer(self):
        """Disable observer for inference"""
        self.observer.enabled = False
        
    def disable_fake_quant(self):
        """Disable fake quantization"""
        self.fake_quant_enabled = False


class StaticQuantizedLinear(QuantizedLinear):
    """Alias for QuantizedLinear to distinguish from dynamic quantization"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config: QuantizationConfig = None):
        super().__init__(in_features, out_features, bias, config)




class ManualQuantStub(nn.Module):
    """Manual quantization stub"""
    def __init__(self, config: QuantizationConfig = None):
        super().__init__()
        self.config = config or QuantizationConfig()
        self.fake_quant = FakeQuantize(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fake_quant(x)


class ManualDeQuantStub(nn.Module):
    """Manual dequantization stub"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # No-op for manual implementation


import copy
from typing import Callable, Dict, Optional, Tuple, Union, List
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseForecastingModel(nn.Module):
    """
    Base forecasting model with pure forecasting functionality.
    This class handles the core forecasting logic without distillation or quantization.
    Clean, lightweight, and focused on prediction tasks.
    """

    VALID_STRATEGIES = ["seq2seq", "autoregressive", "direct", "transformer_seq2seq"]
    VALID_MODEL_TYPES = ["lstm", "transformer", "informer-like"]

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        target_len: int = 5,
        forecasting_strategy: str = "seq2seq",
        model_type: str = "lstm",
        # Processing modules
        input_preprocessor: nn.Module = None,
        output_postprocessor: nn.Module = None,
        input_normalization: nn.Module = None,
        output_normalization: nn.Module = None,
        output_block: nn.Module = None,
        # Architecture options
        attention_module: nn.Module = None,
        output_size: int = None,
        hidden_size: int = 64,
        input_skip_connection: bool = False,
        # Multi-encoder
        multi_encoder_decoder: bool = False,
        input_processor_output_size: int = 16,
        # Training
        teacher_forcing_ratio: float = 0.5,
        scheduled_sampling_fn: Callable = None,
        # Time embeddings
        time_feature_embedding_enc: nn.Module = None,
        time_feature_embedding_dec: nn.Module = None,
    ):
        super().__init__()

        # Validate inputs
        assert forecasting_strategy in self.VALID_STRATEGIES, f"Invalid strategy: {forecasting_strategy}"
        assert model_type in self.VALID_MODEL_TYPES, f"Invalid model type: {model_type}"

        # Core parameters
        self.strategy = forecasting_strategy
        self.model_type = model_type
        self.target_len = target_len
        self.pred_len = target_len  # Compatibility
        self.hidden_size = hidden_size
        self.multi_encoder_decoder = multi_encoder_decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduled_sampling_fn = scheduled_sampling_fn
        self.input_skip_connection = input_skip_connection

        # Processing modules (use Identity as default)
        self.input_preprocessor = input_preprocessor or nn.Identity()
        self.output_postprocessor = output_postprocessor or nn.Identity()
        self.input_normalization = input_normalization or nn.Identity()
        self.output_normalization = output_normalization or nn.Identity()
        self.output_block = output_block or nn.Identity()

        # Time embeddings
        self.time_feature_embedding_enc = time_feature_embedding_enc
        self.time_feature_embedding_dec = time_feature_embedding_dec

        # Setup architecture
        self._setup_architecture(encoder, decoder, input_processor_output_size)

        # Sizes
        self.input_size = getattr(encoder, "input_size", None) if encoder else None
        self.output_size = output_size or (getattr(decoder, "output_size", None) if decoder else None)
        self.label_len = getattr(decoder, "output_size", None) if decoder else None

        # Attention
        self.use_attention = attention_module is not None
        self.attention_module = attention_module

        # Setup output layers
        self._setup_output_layers()

        # Decoder input projection
        if encoder and self.output_size:
            encoder_dim = getattr(encoder, "hidden_size", self.hidden_size)
            self.init_decoder_input_layer = nn.Linear(encoder_dim, self.output_size)

        self._kl = None

    def _setup_architecture(self, encoder, decoder, input_processor_output_size):
        """Setup encoder/decoder architecture"""
        if self.multi_encoder_decoder:
            self.encoder = nn.ModuleList([copy.deepcopy(encoder) for _ in range(input_processor_output_size)])
            self.decoder = nn.ModuleList([copy.deepcopy(decoder) for _ in range(input_processor_output_size)])
            self.decoder_aggregator = nn.Linear(input_processor_output_size, 1, bias=False)
        else:
            self.encoder = encoder
            self.decoder = decoder

    def _setup_output_layers(self):
        """Setup output projection layers"""
        if not self.encoder or not self.decoder:
            self.output_layer = nn.Identity()
            self.project_output = nn.Identity()
            return

        # Get dimensions
        encoder_list = isinstance(self.encoder, nn.ModuleList)
        encoder_hidden = getattr(self.encoder[0] if encoder_list else self.encoder, "hidden_size", self.hidden_size)
        decoder_hidden = getattr(self.decoder[0] if encoder_list else self.decoder, "hidden_size", self.hidden_size)
        decoder_output = getattr(self.decoder[0] if encoder_list else self.decoder, "output_size", self.output_size)

        # Output layer
        output_dim = decoder_output + encoder_hidden if self.use_attention else decoder_output
        self.output_layer = nn.Linear(decoder_hidden * 2, self.output_size)

        # Project output if needed
        self.project_output = (nn.Linear(self.input_size, self.output_size) 
                              if self.input_size and self.input_size != self.output_size 
                              else nn.Identity())

    # ==================== FORWARD METHODS ====================

    def forward(self, src: torch.Tensor, targets: torch.Tensor = None, 
                time_features: torch.Tensor = None, epoch: int = None) -> torch.Tensor:
        """Main forward pass for pure forecasting"""
        processed_src = self._preprocess_input(src)

        # Route to strategy
        strategy_map = {
            "direct": self._forward_direct,
            "autoregressive": self._forward_autoregressive,
            "seq2seq": self._forward_seq2seq,
            "transformer_seq2seq": self._forward_seq2seq
        }
        
        output = strategy_map[self.strategy](processed_src, targets, time_features, epoch)
        return output

    def _preprocess_input(self, src: torch.Tensor) -> torch.Tensor:
        """Apply input preprocessing with optional skip connection"""
        processed = self.input_preprocessor(src)
        if self.input_skip_connection:
            processed = processed + src
        return self.input_normalization(processed)

    def _forward_direct(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Direct forecasting - single forward pass"""
        output = self.decoder(src)
        return self.output_postprocessor(self.output_normalization(output))

    def _forward_autoregressive(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Autoregressive forecasting"""
        outputs = []
        decoder_input = src[:, -1:, :]
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            output = self.output_normalization(self.decoder(decoder_input))
            outputs.append(output)

            if t < self.target_len - 1:
                decoder_input = targets[:, t:t+1, :] if use_teacher_forcing and targets is not None else output

        return self.output_postprocessor(torch.cat(outputs, dim=1))

    def _forward_seq2seq(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Seq2seq forward for all model types"""
        if self.multi_encoder_decoder:
            return self._forward_multi_encoder_decoder(src, targets, epoch)

        # Route by model type
        forward_map = {
            "informer-like": self._forward_informer_style,
            "transformer": self._forward_transformer_style
        }
        
        return forward_map.get(self.model_type, self._forward_rnn_style)(src, targets, time_features, epoch)

    def _forward_rnn_style(self, src, targets=None, time_features=None, epoch=None):
        """RNN/LSTM seq2seq"""
        batch_size = src.size(0)
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
        self._kl = kl_div

        decoder_input = src[:, -1:, :]  # Last timestep as initial input
        outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Apply attention if configured
            if self.use_attention:
                context, _ = self.attention_module(decoder_hidden, encoder_outputs)
                decoder_output = self.output_layer(torch.cat([decoder_output, context], dim=-1))

            # Post-process
            decoder_output = self.output_normalization(self.output_block(decoder_output))
            outputs.append(decoder_output.squeeze(1) if decoder_output.dim() == 3 else decoder_output)

            # Next input
            if t < self.target_len - 1:
                if use_teacher_forcing and targets is not None:
                    decoder_input = targets[:, t:t+1, :]
                else:
                    decoder_input = decoder_output.unsqueeze(1) if decoder_output.dim() == 2 else decoder_output

        return self.output_postprocessor(torch.stack(outputs, dim=1))

    def _forward_transformer_style(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Transformer with autoregressive decoding"""
        batch_size = src.size(0)
        memory = self.encoder(src)
        next_input = src[:, -self.label_len:][:, -1:, :]
        outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.pred_len):
            out = self.decoder(next_input, memory)
            pred_t = self.output_layer(out)
            outputs.append(pred_t)

            if t < self.pred_len - 1:
                if use_teacher_forcing and targets is not None:
                    next_input = targets[:, t:t+1, :]
                else:
                    next_input = pred_t
                    # Handle input/output size mismatch
                    if self.input_size != self.output_size:
                        pad_size = self.input_size - self.output_size
                        padding = torch.zeros(batch_size, 1, pad_size, device=src.device)
                        next_input = torch.cat([next_input, padding], dim=-1)

        return torch.cat(outputs, dim=1)

    def _forward_informer_style(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Informer-style parallel decoding"""
        batch_size = src.size(0)

        # Encode
        enc_result = self.encoder(src, time_features=time_features)
        enc_out = enc_result[0] if isinstance(enc_result, tuple) else enc_result

        # Decode
        start_token = src[:, -1:, :]
        dec_input = start_token.expand(batch_size, self.pred_len, -1)
        
        dec_result = self.decoder(dec_input, enc_out)
        out = dec_result[0] if isinstance(dec_result, tuple) else dec_result

        return self.output_layer(out)

    def _forward_multi_encoder_decoder(self, src: torch.Tensor, targets=None, epoch=None) -> torch.Tensor:
        """Multi encoder-decoder processing"""
        batch_size, seq_len, input_size = src.shape
        feature_outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for i in range(input_size):
            feature_input = src[:, :, i:i+1]
            encoder_outputs, encoder_hidden = self.encoder[i](feature_input)
            decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
            self._kl = kl_div

            decoder_input = torch.zeros(batch_size, 1, self.output_size, device=src.device)
            feature_output = []

            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder[i](decoder_input, decoder_hidden)

                if self.use_attention:
                    query = self._get_attention_query(decoder_output, decoder_hidden)
                    context, _ = self.attention_module(query, encoder_outputs)
                    decoder_output = self.output_layer(torch.cat([decoder_output, context], dim=-1))
                else:
                    decoder_output = self.output_layer(decoder_output)

                decoder_output = self.output_normalization(self.output_block(decoder_output))
                feature_output.append(decoder_output.squeeze(1) if decoder_output.dim() == 3 else decoder_output)

                if t < self.target_len - 1:
                    if use_teacher_forcing and targets is not None:
                        decoder_input = targets[:, t:t+1, :]
                    else:
                        decoder_input = decoder_output.unsqueeze(1) if decoder_output.dim() == 2 else decoder_output

            feature_outputs.append(torch.stack(feature_output, dim=1))

        # Aggregate features
        feature_outputs = torch.stack(feature_outputs, dim=-1)
        aggregated = self.decoder_aggregator(feature_outputs).squeeze(-1)
        return self.output_postprocessor(aggregated)

    def _should_use_teacher_forcing(self, targets=None, epoch=None, fallback_device="cpu") -> bool:
        """Determine whether to use teacher forcing safely during FX tracing"""
        # If in eval mode or no targets, disable teacher forcing
        if (not self.training) or (targets is None):
            return False

        # Get scheduled sampling ratio
        ratio = (
            self.scheduled_sampling_fn(epoch)
            if self.scheduled_sampling_fn and epoch is not None
            else self.teacher_forcing_ratio
        )

        # Get a safe device for torch.rand (use targets.device if available)
        device = getattr(targets, "device", torch.device(fallback_device))

        # Important: FX tracing cannot handle torch.rand with dynamic device, so fallback to CPU
        if torch.fx._symbolic_trace.is_fx_tracing():
            return False  # Disable random during tracing to keep deterministic

        return torch.rand((1,), device=device).item() < ratio

    def _process_encoder_hidden(self, encoder_hidden) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process encoder hidden state, handling VAE and bidirectional cases"""
        # VAE case: (z, mu, logvar)
        if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
            z, mu, logvar = encoder_hidden
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return (z,), kl_div
        
        return self._prepare_decoder_hidden(encoder_hidden), None

    def _prepare_decoder_hidden(self, encoder_hidden):
        """Prepare encoder hidden for decoder"""
        if not getattr(self.encoder, "bidirectional", False):
            return encoder_hidden

        # Handle bidirectional case
        if isinstance(encoder_hidden, tuple):  # LSTM
            h_n, c_n = encoder_hidden
            return (self._combine_bidirectional(h_n), self._combine_bidirectional(c_n))
        else:  # GRU/RNN
            return self._combine_bidirectional(encoder_hidden)

    def _combine_bidirectional(self, hidden):
        """Combine bidirectional hidden states"""
        # Only combine if we actually have bidirectional layers
        # For bidirectional RNNs, hidden.size(0) = num_layers * 2
        num_layers = hidden.size(0) // 2
        hidden = hidden.view(num_layers, 2, *hidden.shape[1:]).sum(dim=1)
        return hidden

    def _get_attention_query(self, decoder_output, decoder_hidden):
        """Extract attention query from decoder state"""
        if getattr(self.decoder, "is_transformer", False):
            return decoder_hidden.permute(1, 0, 2)
        else:
            return decoder_hidden[0][-1] if isinstance(decoder_hidden, tuple) else decoder_hidden[-1]

    # ==================== UTILITY METHODS ====================

    def get_kl(self) -> Optional[torch.Tensor]:
        """Get KL divergence for VAE loss"""
        return self._kl

    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Get model size information"""
        param_size = sum(p.numel() for p in self.parameters())
        buffer_size = sum(b.numel() for b in self.buffers())
        
        # Base model is always FP32
        size_mb = (param_size + buffer_size) * 4 / (1024 * 1024)  # 4 bytes per param
        
        return {
            "parameters": param_size,
            "buffers": buffer_size,
            "total_elements": param_size + buffer_size,
            "size_mb": size_mb,
            "is_quantized": False
        }

    def benchmark_inference(self, input_tensor: torch.Tensor, num_runs: int = 100, warmup_runs: int = 10):
        """Benchmark inference speed with automatic device handling"""
        import time
        
        self.eval()
        
        # Automatically move input to same device as model
        model_device = next(self.parameters()).device
        if input_tensor.device != model_device:
            print(f"Moving input tensor from {input_tensor.device} to {model_device}")
            input_tensor = input_tensor.to(model_device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self(input_tensor)
        
        # Benchmark
        if torch.cuda.is_available() and model_device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self(input_tensor)
        
        if torch.cuda.is_available() and model_device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_samples_per_sec": 1.0 / avg_time,
            "device": str(model_device)
        }

    def attribute_forward(self, src: torch.Tensor, time_features: Optional[torch.Tensor] = None,
                         targets: Optional[torch.Tensor] = None, epoch: Optional[int] = None,
                         output_idx: Optional[int] = None) -> torch.Tensor:
        """Captum-compatible forward pass for attribution"""
        self.train()  # Enable backward for RNN
        self._disable_dropout()
        
        src = src.requires_grad_()
        out = self.forward(src, targets=targets, time_features=time_features, epoch=epoch)
        
        return out[..., output_idx] if output_idx is not None else out

    def _disable_dropout(self):
        """Disable dropout layers while keeping model in train() mode"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0


# ==================== FORECASTING MODEL WITH DISTILLATION ====================

class ForecastingModel(BaseForecastingModel):
    """
    Forecasting model with knowledge distillation support.
    Inherits all forecasting functionality and adds distillation capabilities.
    """

    VALID_DISTILLATION_MODES = ["none", "output", "feature", "attention", "comprehensive"]

    def __init__(
        self,
        # Knowledge distillation options
        distillation_mode: str = "none",
        teacher_model: nn.Module = None,
        distillation_temperature: float = 4.0,
        distillation_alpha: float = 0.7,
        feature_distillation_layers: List[str] = None,
        attention_distillation_layers: List[str] = None,
        # All other parameters pass through to parent
        **kwargs
    ):
        # Validate distillation mode
        assert distillation_mode in self.VALID_DISTILLATION_MODES, f"Invalid distillation mode: {distillation_mode}"
        
        # Initialize parent class
        super().__init__(**kwargs)

        # Knowledge distillation parameters
        self.distillation_mode = distillation_mode
        self.teacher_model = teacher_model
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha
        self.feature_distillation_layers = feature_distillation_layers or []
        self.attention_distillation_layers = attention_distillation_layers or []
        self.feature_hooks = {}
        self.attention_hooks = {}
        self.teacher_features = {}
        self.teacher_attentions = {}
        self.student_features = {}
        self.student_attentions = {}

        # Initialize knowledge distillation if needed
        if distillation_mode != "none":
            self._setup_distillation()

    # ==================== DISTILLATION METHODS ====================

    def _setup_distillation(self):
        """Setup knowledge distillation configuration"""
        if self.distillation_mode == "none" or self.teacher_model is None:
            return
        
        # Set teacher model to evaluation mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Setup feature distillation hooks
        if self.distillation_mode in ["feature", "comprehensive"]:
            self._setup_feature_hooks()
        
        # Setup attention distillation hooks
        if self.distillation_mode in ["attention", "comprehensive"]:
            self._setup_attention_hooks()
    
    def _setup_feature_hooks(self):
        """Setup hooks for feature distillation"""
        def create_feature_hook(name, storage_dict):
            def hook(module, input, output):
                storage_dict[name] = output
            return hook
        
        # Register hooks for teacher model
        for layer_name in self.feature_distillation_layers:
            if hasattr(self.teacher_model, layer_name):
                layer = getattr(self.teacher_model, layer_name)
                hook = layer.register_forward_hook(
                    create_feature_hook(layer_name, self.teacher_features)
                )
                self.feature_hooks[f"teacher_{layer_name}"] = hook
        
        # Register hooks for student model (self)
        for layer_name in self.feature_distillation_layers:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                hook = layer.register_forward_hook(
                    create_feature_hook(layer_name, self.student_features)
                )
                self.feature_hooks[f"student_{layer_name}"] = hook
    
    def _setup_attention_hooks(self):
        """Setup hooks for attention distillation"""
        def create_attention_hook(name, storage_dict):
            def hook(module, input, output):
                # For attention modules, we typically want the attention weights
                if hasattr(module, 'attention_weights'):
                    storage_dict[name] = module.attention_weights
                else:
                    storage_dict[name] = output
            return hook
        
        # Register hooks for teacher model
        for layer_name in self.attention_distillation_layers:
            if hasattr(self.teacher_model, layer_name):
                layer = getattr(self.teacher_model, layer_name)
                hook = layer.register_forward_hook(
                    create_attention_hook(layer_name, self.teacher_attentions)
                )
                self.attention_hooks[f"teacher_{layer_name}"] = hook
        
        # Register hooks for student model (self)
        for layer_name in self.attention_distillation_layers:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                hook = layer.register_forward_hook(
                    create_attention_hook(layer_name, self.student_attentions)
                )
                self.attention_hooks[f"student_{layer_name}"] = hook

    def _compute_feature_distillation_loss(self) -> torch.Tensor:
        """Compute feature-level distillation loss"""
        feature_loss = 0.0
        num_layers = 0

        for layer_name in self.feature_distillation_layers:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                teacher_feat = self.teacher_features[layer_name]
                student_feat = self.student_features[layer_name]

                # ✅ Unwrap if tuple/list
                if isinstance(teacher_feat, (tuple, list)):
                    teacher_feat = teacher_feat[0]
                if isinstance(student_feat, (tuple, list)):
                    student_feat = student_feat[0]

                # ✅ Skip non-tensors safely
                if not isinstance(teacher_feat, torch.Tensor) or not isinstance(student_feat, torch.Tensor):
                    continue

                # ✅ Align dimensions safely
                if teacher_feat.shape != student_feat.shape:
                    # Instead of creating new projection every time, either skip OR reuse a predefined one
                    student_feat = self._align_feature_dimensions(student_feat, teacher_feat.shape)

                feature_loss += F.mse_loss(student_feat, teacher_feat)
                num_layers += 1

        return feature_loss / max(num_layers, 1)

    def _compute_output_distillation_loss(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """Compute output-level distillation loss using KL divergence with temperature scaling"""
        # For regression tasks, use MSE instead of KL divergence
        if student_output.dtype == torch.float32 and teacher_output.dtype == torch.float32:
            return F.mse_loss(student_output, teacher_output)
        
        # For classification tasks, use KL divergence with temperature scaling
        student_soft = F.softmax(student_output / self.distillation_temperature, dim=-1)
        teacher_soft = F.softmax(teacher_output / self.distillation_temperature, dim=-1)
        
        kl_loss = F.kl_div(
            F.log_softmax(student_output / self.distillation_temperature, dim=-1),
            teacher_soft,
            reduction='batchmean'
        )
        
        return kl_loss * (self.distillation_temperature ** 2)
    
    def _compute_feature_distillation_loss(self) -> torch.Tensor:
        """Compute feature-level distillation loss"""
        feature_loss = 0.0
        num_layers = 0
        
        for layer_name in self.feature_distillation_layers:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                teacher_feat = self.teacher_features[layer_name]
                student_feat = self.student_features[layer_name]
                
                # Align dimensions if necessary
                if teacher_feat.shape != student_feat.shape:
                    student_feat = self._align_feature_dimensions(student_feat, teacher_feat.shape)
                
                # MSE loss between features
                layer_loss = F.mse_loss(student_feat, teacher_feat)
                feature_loss += layer_loss
                num_layers += 1
        
        return feature_loss / max(num_layers, 1)
    
    def _compute_attention_distillation_loss(self) -> torch.Tensor:
        """Compute attention-level distillation loss"""
        attention_loss = 0.0
        num_layers = 0
        
        for layer_name in self.attention_distillation_layers:
            if layer_name in self.teacher_attentions and layer_name in self.student_attentions:
                teacher_att = self.teacher_attentions[layer_name]
                student_att = self.student_attentions[layer_name]
                
                # Align dimensions if necessary
                if teacher_att.shape != student_att.shape:
                    student_att = self._align_attention_dimensions(student_att, teacher_att.shape)
                
                # MSE loss between attention maps
                layer_loss = F.mse_loss(student_att, teacher_att)
                attention_loss += layer_loss
                num_layers += 1
        
        return attention_loss / max(num_layers, 1)
    
    def _align_feature_dimensions(self, student_feat: torch.Tensor, teacher_shape: torch.Size) -> torch.Tensor:
        """Align student feature dimensions to match teacher"""
        if student_feat.shape[-1] != teacher_shape[-1]:
            # Use linear projection to match feature dimensions
            projection = nn.Linear(student_feat.shape[-1], teacher_shape[-1]).to(student_feat.device)
            student_feat = projection(student_feat)
        
        # Handle sequence length differences
        if len(student_feat.shape) > 1 and student_feat.shape[1] != teacher_shape[1]:
            if student_feat.shape[1] < teacher_shape[1]:
                # Pad or interpolate
                student_feat = F.interpolate(
                    student_feat.transpose(1, 2), 
                    size=teacher_shape[1], 
                    mode='linear'
                ).transpose(1, 2)
            else:
                # Truncate
                student_feat = student_feat[:, :teacher_shape[1], :]
        
        return student_feat
    
    def _align_attention_dimensions(self, student_att: torch.Tensor, teacher_shape: torch.Size) -> torch.Tensor:
        """Align student attention dimensions to match teacher"""
        if student_att.shape != teacher_shape:
            # For attention maps, typically we need to handle head differences
            if len(student_att.shape) == 4:  # [batch, heads, seq, seq]
                if student_att.shape[1] != teacher_shape[1]:  # Different number of heads
                    # Average or interpolate heads
                    if student_att.shape[1] < teacher_shape[1]:
                        # Repeat heads
                        repeat_factor = teacher_shape[1] // student_att.shape[1]
                        student_att = student_att.repeat(1, repeat_factor, 1, 1)
                    else:
                        # Average heads
                        group_size = student_att.shape[1] // teacher_shape[1]
                        student_att = student_att.view(
                            student_att.shape[0], teacher_shape[1], group_size, 
                            student_att.shape[2], student_att.shape[3]
                        ).mean(dim=2)
                
                # Handle sequence length differences
                if student_att.shape[2] != teacher_shape[2] or student_att.shape[3] != teacher_shape[3]:
                    student_att = F.interpolate(
                        student_att.view(-1, 1, student_att.shape[2], student_att.shape[3]),
                        size=(teacher_shape[2], teacher_shape[3]),
                        mode='bilinear'
                    ).view(student_att.shape[0], student_att.shape[1], teacher_shape[2], teacher_shape[3])
        
        return student_att
    
    def _combine_distillation_losses(self, loss_components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine different distillation losses with appropriate weights"""
        total_loss = 0.0
        
        # Task loss (ground truth)
        if 'task_loss' in loss_components:
            total_loss += (1.0 - self.distillation_alpha) * loss_components['task_loss']
        
        # Distillation losses
        distillation_loss = 0.0
        num_distill_losses = 0
        
        if 'output_distillation' in loss_components:
            distillation_loss += loss_components['output_distillation']
            num_distill_losses += 1
        
        if 'feature_distillation' in loss_components:
            distillation_loss += 0.5 * loss_components['feature_distillation']  # Lower weight for features
            num_distill_losses += 1
        
        if 'attention_distillation' in loss_components:
            distillation_loss += 0.3 * loss_components['attention_distillation']  # Lower weight for attention
            num_distill_losses += 1
        
        if num_distill_losses > 0:
            total_loss += self.distillation_alpha * (distillation_loss / num_distill_losses)
        
        return total_loss

    def set_teacher_model(self, teacher_model: nn.Module):
        """Set or update the teacher model for distillation"""
        self.teacher_model = teacher_model
        if teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            # Re-setup distillation if mode is not none
            if self.distillation_mode != "none":
                self._setup_distillation()
    
    def enable_distillation(self, mode: str = "output", teacher_model: nn.Module = None):
        """Enable knowledge distillation with specified mode"""
        assert mode in self.VALID_DISTILLATION_MODES, f"Invalid distillation mode: {mode}"
        
        self.distillation_mode = mode
        if teacher_model is not None:
            self.set_teacher_model(teacher_model)
        
        if mode != "none":
            self._setup_distillation()
    
    def disable_distillation(self):
        """Disable knowledge distillation"""
        self.distillation_mode = "none"
        self.teacher_model = None
        
        # Remove hooks
        for hook in self.feature_hooks.values():
            hook.remove()
        for hook in self.attention_hooks.values():
            hook.remove()
        
        self.feature_hooks.clear()
        self.attention_hooks.clear()
    
    def get_distillation_info(self) -> Dict[str, Union[str, int, float]]:
        """Get information about current distillation setup"""
        return {
            "distillation_enabled": True,
            "distillation_mode": self.distillation_mode,
            "has_teacher": self.teacher_model is not None,
            "temperature": self.distillation_temperature,
            "alpha": self.distillation_alpha,
            "feature_layers": len(self.feature_distillation_layers),
            "attention_layers": len(self.attention_distillation_layers),
            "active_feature_hooks": len(self.feature_hooks),
            "active_attention_hooks": len(self.attention_hooks)
        }

    # ==================== OVERRIDDEN METHODS ====================

    def forward(self, src: torch.Tensor, targets: torch.Tensor = None, 
                time_features: torch.Tensor = None, epoch: int = None, 
                return_teacher_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Enhanced forward pass with distillation support"""
        # Clear previous feature/attention storages
        self.student_features.clear()
        self.student_attentions.clear()
        
        # Get teacher outputs if in distillation mode
        teacher_output = None
        if self.distillation_mode != "none" and self.teacher_model is not None:
            self.teacher_features.clear()
            self.teacher_attentions.clear()
            with torch.no_grad():
                teacher_output = self.teacher_model(src, targets, time_features, epoch)
        
        # Call parent forward method
        output = super().forward(src, targets, time_features, epoch)
        
        if return_teacher_outputs and teacher_output is not None:
            return output, teacher_output
        return output

    def benchmark_inference(self, input_tensor: torch.Tensor, num_runs: int = 100, warmup_runs: int = 10):
        """Enhanced benchmark with distillation info"""
        result = super().benchmark_inference(input_tensor, num_runs, warmup_runs)
        
        # Add distillation info
        result.update({
            "distillation_mode": self.distillation_mode,
            "has_teacher": self.teacher_model is not None,
        })
        
        return result


# ==================== QUANTIZED FORECASTING MODEL ====================

class QuantizedForecastingModel(ForecastingModel):
    """
    Quantized forecasting model that inherits from ForecastingModel and adds quantization capabilities.
    This class extends the distillation-enabled model with quantization-specific features.
    """

    VALID_QUANTIZATION_MODES = ["none", "ptq", "qat", "dynamic", "static"]

    def __init__(
        self,
        # Quantization options
        quantization_mode: str = "none",
        bit_width: int = 8,
        symmetric_quantization: bool = True,
        per_channel_quantization: bool = False,
        # All other parameters pass through to parent
        **kwargs
    ):
        # Validate quantization mode
        assert quantization_mode in self.VALID_QUANTIZATION_MODES, f"Invalid quantization mode: {quantization_mode}. Valid modes: {self.VALID_QUANTIZATION_MODES}"
        
        # Initialize parent class (inherits both base forecasting and distillation)
        super().__init__(**kwargs)

        # Quantization parameters
        self.quantization_mode = quantization_mode
        self.bit_width = bit_width
        self.symmetric_quantization = symmetric_quantization
        self.per_channel_quantization = per_channel_quantization
        self.is_quantized = False
        self.quantization_config = QuantizationConfig(
            bit_width=bit_width,
            symmetric=symmetric_quantization,
            per_channel=per_channel_quantization
        )

        # Manual quantization stubs
        if quantization_mode in ["ptq", "qat", "static"]:
            self.quant = ManualQuantStub(self.quantization_config)
            self.dequant = ManualDeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

        # Initialize quantization if needed
        if quantization_mode != "none":
            self._setup_quantization()

    # ==================== QUANTIZATION METHODS ====================

    def _setup_quantization(self):
        """Setup manual quantization configuration"""
        if self.quantization_mode == "none":
            return
        
        # Initialize quantization observers for all linear layers
        self._add_quantization_observers()

    def _add_quantization_observers(self):
        """Add quantization observers to all linear layers"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Add fake quantization to linear layers for QAT
                if self.quantization_mode == "qat":
                    fake_quant = FakeQuantize(self.quantization_config)
                    setattr(module, 'weight_fake_quant', fake_quant)
                    if module.bias is not None:
                        bias_fake_quant = FakeQuantize(self.quantization_config)
                        setattr(module, 'bias_fake_quant', bias_fake_quant)

    def prepare_for_quantization(self, calibration_data: Optional[torch.utils.data.DataLoader] = None):
        """
        Prepare model for quantization.
        
        Args:
            calibration_data: DataLoader for calibration (PTQ only)
        """
        if self.quantization_mode == "none":
            return self

        if self.quantization_mode == "dynamic":
            # Dynamic quantization - convert immediately
            return self._apply_dynamic_quantization()
        
        elif self.quantization_mode == "ptq":
            # Post-training quantization
            return self._apply_post_training_quantization(calibration_data)
        
        elif self.quantization_mode == "qat":
            # Quantization-aware training
            return self._prepare_quantization_aware_training()
        
        elif self.quantization_mode == "static":
            # Static quantization (similar to PTQ but without calibration)
            return self._apply_static_quantization(calibration_data)

    def _apply_dynamic_quantization(self):
        """Apply dynamic quantization to linear layers"""
        quantized_model = copy.deepcopy(self)
        
        # Replace linear layers with dynamic quantized versions
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Create dynamic quantized version
                dynamic_quantized_linear = DynamicQuantizedLinear.from_float(module, self.quantization_config)
                
                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(quantized_model.named_modules())[parent_name]
                    setattr(parent_module, child_name, dynamic_quantized_linear)
                else:
                    setattr(quantized_model, child_name, dynamic_quantized_linear)
        
        quantized_model.is_quantized = True
        return quantized_model

    def _apply_static_quantization(self, calibration_data: Optional[torch.utils.data.DataLoader] = None):
        """Apply static quantization to linear layers"""
        quantized_model = copy.deepcopy(self)
        
        # Replace linear layers with static quantized versions
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Create static quantized version
                static_quantized_linear = StaticQuantizedLinear.from_float(module, self.quantization_config)
                
                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(quantized_model.named_modules())[parent_name]
                    setattr(parent_module, child_name, static_quantized_linear)
                else:
                    setattr(quantized_model, child_name, static_quantized_linear)
        
        quantized_model.is_quantized = True
        return quantized_model

    def _apply_post_training_quantization(self, calibration_data: Optional[torch.utils.data.DataLoader] = None):
        """Apply post-training quantization"""
        # First, add observers to collect statistics
        self._add_quantization_observers()
        
        # Calibrate if data is provided
        if calibration_data is not None:
            self._calibrate_model(calibration_data)
        
        # Convert to quantized model
        return self._convert_to_quantized()

    def _calibrate_model(self, calibration_data: torch.utils.data.DataLoader):
        """Calibrate model using calibration data"""
        self.eval()
        
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch
                    targets = None
                
                # Forward pass to collect statistics
                self.forward(inputs, targets)

    def _prepare_quantization_aware_training(self):
        """Prepare model for quantization-aware training"""
        # Add fake quantization to all linear layers
        self._add_quantization_observers()
        
        # Enable fake quantization
        for module in self.modules():
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quant.fake_quant_enabled = True
        
        return self

    def _convert_to_quantized(self):
        """Convert model to static quantized version"""
        quantized_model = copy.deepcopy(self)
        
        # Replace linear layers with static quantized versions
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Create static quantized version
                quantized_linear = StaticQuantizedLinear.from_float(module, self.quantization_config)
                
                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(quantized_model.named_modules())[parent_name]
                    setattr(parent_module, child_name, quantized_linear)
                else:
                    setattr(quantized_model, child_name, quantized_linear)
        
        quantized_model.is_quantized = True
        return quantized_model

    def finalize_quantization(self):
        """Finalize quantization after QAT training"""
        if self.quantization_mode == "qat":
            # Calculate final quantization parameters
            for module in self.modules():
                if hasattr(module, 'weight_fake_quant'):
                    module.weight_fake_quant.calculate_qparams()
                    module.weight_fake_quant.disable_observer()
                    module.weight_fake_quant.disable_fake_quant()
            
            # Convert to fully quantized model
            return self._convert_to_quantized()
        return self

    def set_quantization_mode(self, mode: str):
        """Set quantization mode"""
        assert mode in self.VALID_QUANTIZATION_MODES, f"Invalid quantization mode: {mode}"
        self.quantization_mode = mode
        
        if mode == "none":
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
        elif mode in ["ptq", "qat", "static"]:
            self.quant = ManualQuantStub(self.quantization_config)
            self.dequant = ManualDeQuantStub()
        else:
            # For dynamic, quantization happens inside the layers
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

    def get_quantization_info(self) -> Dict[str, Union[str, int, float, bool]]:
        """Get information about current quantization setup"""
        return {
            "quantization_enabled": True,
            "quantization_mode": self.quantization_mode,
            "bit_width": self.bit_width,
            "symmetric": self.symmetric_quantization,
            "per_channel": self.per_channel_quantization,
            "is_quantized": self.is_quantized,
            "num_quantizable_layers": sum(1 for m in self.modules() if isinstance(m, nn.Linear))
        }

    # ==================== OVERRIDDEN METHODS ====================

    def forward(self, src: torch.Tensor, targets: torch.Tensor = None, 
                time_features: torch.Tensor = None, epoch: int = None, 
                return_teacher_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Main forward pass with quantization and distillation support"""
        # Apply quantization stub if needed
        if self.quantization_mode in ["ptq", "qat", "static"]:
            src = self.quant(src)
        
        # Call parent forward method (includes distillation)
        output = super().forward(src, targets, time_features, epoch, return_teacher_outputs)
        
        # Apply dequantization stub if needed
        if self.quantization_mode in ["ptq", "qat", "static"]:
            if isinstance(output, tuple):
                output = (self.dequant(output[0]), output[1])
            else:
                output = self.dequant(output)
        
        return output

    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Get model size information with quantization awareness"""
        param_size = sum(p.numel() for p in self.parameters())
        buffer_size = sum(b.numel() for b in self.buffers())
        
        # Estimate memory usage (in MB)
        if self.is_quantized:
            # INT8 quantized model
            size_mb = (param_size + buffer_size) * 1 / (1024 * 1024)  # 1 byte per param
        else:
            # FP32 model
            size_mb = (param_size + buffer_size) * 4 / (1024 * 1024)  # 4 bytes per param
        
        return {
            "parameters": param_size,
            "buffers": buffer_size,
            "total_elements": param_size + buffer_size,
            "size_mb": size_mb,
            "is_quantized": self.is_quantized
        }

    def benchmark_inference(self, input_tensor: torch.Tensor, num_runs: int = 100, warmup_runs: int = 10):
        """Enhanced benchmark with quantization and distillation info"""
        result = super().benchmark_inference(input_tensor, num_runs, warmup_runs)
        
        # Update result with quantization info
        result.update({
            "quantization_mode": self.quantization_mode,
            "is_quantized": self.is_quantized,
        })
        
        return result
