import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizationConfig:
    """Configuration for manual quantization"""

    def __init__(
        self,
        bit_width: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        observer_type: str = "minmax",
    ):
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.observer_type = observer_type

        # Calculate quantization parameters
        self.qmin = -(2 ** (bit_width - 1)) if symmetric else 0
        self.qmax = 2 ** (bit_width - 1) - 1 if symmetric else 2**bit_width - 1


class QuantizationObserver(nn.Module):
    """Manual quantization observer to collect statistics"""

    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches", torch.tensor(0))
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
            scale = (self.max_val - self.min_val) / (
                self.config.qmax - self.config.qmin
            )
            zero_point = self.config.qmin - torch.round(self.min_val / scale)
            zero_point = torch.clamp(
                zero_point, self.config.qmin, self.config.qmax
            ).long()

        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        return scale, zero_point


class QuantizedLinear(nn.Module):
    """Fixed static quantized Linear layer with proper device handling"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: "QuantizationConfig" = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()

        # Quantized weights and bias - these need to be on the same device
        self.register_buffer(
            "weight_int", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("weight_zero_point", torch.tensor(0, dtype=torch.long))

        if bias:
            self.register_buffer(
                "bias_int", torch.zeros(out_features, dtype=torch.int32)
            )
            self.register_buffer("bias_scale", torch.tensor(1.0))
        else:
            self.register_buffer("bias_int", None)
            self.register_buffer("bias_scale", None)

        # Input quantization parameters
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("input_zero_point", torch.tensor(0, dtype=torch.long))

        # Output quantization parameters
        self.register_buffer("output_scale", torch.tensor(1.0))
        self.register_buffer("output_zero_point", torch.tensor(0, dtype=torch.long))

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
            bias_int.float() if bias_int is not None else None,
        )

        # Dequantize output
        output = (output_int - output_zero_point) * output_scale

        return output

    def to(self, device_or_dtype):
        """Override to method to ensure all buffers move together"""
        result = super().to(device_or_dtype)
        return result

    @classmethod
    def from_float(cls, float_module: nn.Linear, config: "QuantizationConfig" = None):
        """Convert float linear layer to quantized version"""

        config = config or QuantizationConfig()
        quantized_module = cls(
            float_module.in_features,
            float_module.out_features,
            float_module.bias is not None,
            config,
        )

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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: "QuantizationConfig" = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()

        # Statically quantized weights
        self.register_buffer(
            "weight_int", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("weight_zero_point", torch.tensor(0, dtype=torch.long))

        # Float bias (not quantized in dynamic quantization)
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)

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
    def from_float(cls, float_module: nn.Linear, config: "QuantizationConfig" = None):

        config = config or QuantizationConfig()
        quantized_module = cls(
            float_module.in_features,
            float_module.out_features,
            float_module.bias is not None,
            config,
        )

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
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.long))
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

    def _fake_quantize(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantizationConfig = None,
    ):
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
