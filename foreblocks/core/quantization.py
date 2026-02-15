import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Quantization Core & Obs
# =========================

class QuantizationConfig:
    """
    Simple config for manual (fake/static/dynamic) quantization.
    - bit_width: 8 by default
    - symmetric: if True -> qmin = -(2^(b-1)), qmax = 2^(b-1)-1, zp ~ 0
                 if False -> qmin = 0, qmax = 2^b-1, zp in [qmin,qmax]
    - per_channel: if True, weight quant params are per out_channel (dim=0)
    """
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
        self.observer_type = observer_type  # reserved for future use

        self.qmin = -(2 ** (bit_width - 1)) if symmetric else 0
        self.qmax =  (2 ** (bit_width - 1) - 1) if symmetric else (2 ** bit_width - 1)


class QuantizationObserver(nn.Module):
    """
    Min/max observer (per-tensor or per-outchannel for weights).
    For Linear/Conv weights, "channel" is out_features/out_channels at dim=0.

    Usage:
      obs = QuantizationObserver(cfg)
      obs._update_stats(tensor)  # or call obs(tensor) while in training mode
      scale, zp = obs.calculate_scale_zero_point()
    """
    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        # Lazily initialized buffers to match shape (scalar or [C])
        self.register_buffer("min_val", None, persistent=True)   # Tensor or None
        self.register_buffer("max_val", None, persistent=True)   # Tensor or None
        self.register_buffer("num_batches", torch.zeros((), dtype=torch.long), persistent=True)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled and self.training:
            self._update_stats(x)
        return x

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor):
        if self.config.per_channel and x.dim() >= 2:
            # Reduce over all dims except channel 0
            reduce_dims = tuple(range(1, x.dim()))
            cur_min = torch.amin(x, dim=reduce_dims)  # [C]
            cur_max = torch.amax(x, dim=reduce_dims)  # [C]
        else:
            cur_min = torch.amin(x)                   # scalar
            cur_max = torch.amax(x)                   # scalar

        if self.min_val is None or self.max_val is None:
            self.min_val = cur_min.detach().clone()
            self.max_val = cur_max.detach().clone()
        else:
            self.min_val = torch.minimum(self.min_val, cur_min)
            self.max_val = torch.maximum(self.max_val, cur_max)

        self.num_batches += 1

    @torch.no_grad()
    def calculate_scale_zero_point(self):
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Observer has no data. Call _update_stats(...) before calculating qparams.")

        cfg = self.config
        rng = self.max_val - self.min_val
        # guard zero ranges
        rng = torch.where(rng == 0, torch.ones_like(rng), rng)

        if cfg.symmetric:
            abs_max = torch.maximum(self.max_val.abs(), self.min_val.abs())
            abs_max = torch.where(abs_max == 0, torch.ones_like(abs_max), abs_max)
            scale = abs_max / (2 ** (cfg.bit_width - 1) - 1)
            zero_point = torch.zeros_like(scale, dtype=torch.long)
        else:
            scale = rng / (cfg.qmax - cfg.qmin)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero_point = cfg.qmin - torch.round(self.min_val / scale)
            zero_point = torch.clamp(zero_point, cfg.qmin, cfg.qmax).to(torch.long)

        return scale, zero_point


# =========================
# Static (PTQ) Quant Linear
# =========================

class QuantizedLinear(nn.Module):
    """
    Post-Training Quantized linear layer (static activations/weights).
    - Weights stored as int8 with per-tensor or per-channel (out_features) qparams.
    - Activations use fixed qparams (must be set from calibration).
    - Bias stored as int32 with scale = input_scale * weight_scale.
    - Performs: y = ( (x_int - x_zp) @ (w_int - w_zp)^T ) * (x_s * w_s) + bias_int * (x_s * w_s)
      (uses float accumulation for simplicity)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantizationConfig = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()

        w_dtype = torch.int8
        # int8 weights
        self.register_buffer("weight_int", torch.zeros(out_features, in_features, dtype=w_dtype))
        # scales/zps: scalar () or per-channel [out]
        s_shape = (out_features,) if self.config.per_channel else ()
        self.register_buffer("weight_scale", torch.ones(s_shape))
        self.register_buffer("weight_zero_point", torch.zeros(s_shape, dtype=torch.long))

        # bias int32 (static)
        if bias:
            self.register_buffer("bias_int", torch.zeros(out_features, dtype=torch.int32))
            self.register_buffer("bias_scale", torch.ones(s_shape))  # should equal input_scale * weight_scale
        else:
            self.register_buffer("bias_int", None)
            self.register_buffer("bias_scale", None)

        # Activation qparams (set via calibration)
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("input_zero_point", torch.tensor(0, dtype=torch.long))

    def set_input_qparams(self, scale: torch.Tensor, zero_point: torch.Tensor):
        """Set calibrated activation qparams (per-tensor)."""
        self.input_scale.copy_(scale.detach())
        self.input_zero_point.copy_(zero_point.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device

        # ----- Quantize activations -----
        a_s  = self.input_scale.to(dev)
        a_zp = self.input_zero_point.to(dev)
        x_int = torch.clamp(torch.round(x / a_s) + a_zp, self.config.qmin, self.config.qmax).to(torch.int8)

        # ----- Center integers -----
        Wq   = self.weight_int.to(dev)
        w_s  = self.weight_scale.to(dev)
        w_zp = self.weight_zero_point.to(dev)

        x_c = (x_int.to(torch.int32) - a_zp.to(torch.int32))
        if self.config.per_channel:
            w_c = (Wq.to(torch.int32) - w_zp.view(-1, 1).to(torch.int32))  # [out,in]
        else:
            w_c = (Wq.to(torch.int32) - w_zp.to(torch.int32))

        # ----- Matmul (float accum for simplicity) -----
        out_acc = F.linear(x_c.float(), w_c.float(), None)  # [B,out]

        # ----- Dequantize core matmul -----
        if self.config.per_channel:
            out = out_acc * (a_s * w_s).view(1, -1)
        else:
            out = out_acc * (a_s * w_s)

        # ----- Add bias in real units -----
        if self.bias_int is not None:
            b_int = self.bias_int.to(dev).float()
            b_s   = self.bias_scale.to(dev)
            if self.config.per_channel:
                out = out + b_int * b_s.view(1, -1)
            else:
                out = out + b_int * b_s

        return out

    @classmethod
    def from_float(
        cls,
        float_module: nn.Linear,
        config: QuantizationConfig = None,
        act_observer: QuantizationObserver = None,
    ):
        """
        Convert fp32 Linear to static quantized Linear.
        Requires a calibrated activation observer for input qparams.
        Weight quantization is symmetric by default (zp≈0).
        """
        cfg = config or QuantizationConfig()
        q = cls(float_module.in_features, float_module.out_features, float_module.bias is not None, cfg)
        dev = next(float_module.parameters()).device

        # --- Weight PTQ (symmetric per-tensor/per-channel) ---
        w_obs = QuantizationObserver(
            QuantizationConfig(bit_width=cfg.bit_width, symmetric=True, per_channel=cfg.per_channel)
        )
        w_obs._update_stats(float_module.weight.data)
        w_scale, w_zp = w_obs.calculate_scale_zero_point()

        if cfg.per_channel:
            w_int = torch.round(float_module.weight / w_scale.view(-1, 1)) + w_zp.view(-1, 1)
        else:
            w_int = torch.round(float_module.weight / w_scale) + w_zp

        w_int = torch.clamp(w_int, cfg.qmin, cfg.qmax).to(torch.int8)

        q.weight_int.copy_(w_int.to(dev))
        q.weight_scale.copy_(w_scale.to(dev))
        q.weight_zero_point.copy_(w_zp.to(dev))

        # --- Activation qparams from provided observer (calibrated over data) ---
        if act_observer is None or act_observer.min_val is None:
            raise ValueError("from_float requires a calibrated activation observer to set input scale/zero_point.")
        a_scale, a_zp = act_observer.calculate_scale_zero_point()
        q.input_scale.copy_(a_scale.to(dev))
        q.input_zero_point.copy_(a_zp.to(dev))

        # --- Bias quantization: int32 with scale = a_scale * w_scale ---
        if float_module.bias is not None:
            if cfg.per_channel:
                b_scale = (a_scale * w_scale).to(dev)  # [out]
                b_int = torch.round(float_module.bias.to(dev) / b_scale).to(torch.int32)
            else:
                b_scale = (a_scale * w_scale).to(dev)  # scalar
                b_int = torch.round(float_module.bias.to(dev) / b_scale).to(torch.int32)
            q.bias_int.copy_(b_int)
            q.bias_scale.copy_(b_scale)

        return q


# =========================
# Dynamic Quant Linear
# =========================

class DynamicQuantizedLinear(nn.Module):
    """
    Dynamic quantization:
      - Weights are int8 (PTQ) with fixed qparams.
      - Activations are quantized per forward (per-tensor, asymmetric).
      - Bias remains float and is added after dequant.
      y = ( (x_int - x_zp) @ (w_int - w_zp)^T ) * (a_s * w_s) + bias_float
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantizationConfig = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()

        # Weight int8 + qparams
        self.register_buffer("weight_int", torch.zeros(out_features, in_features, dtype=torch.int8))
        s_shape = (out_features,) if self.config.per_channel else ()
        self.register_buffer("weight_scale", torch.ones(s_shape))
        self.register_buffer("weight_zero_point", torch.zeros(s_shape, dtype=torch.long))

        # Bias kept in float in dynamic quant
        if bias:
            self.register_buffer("bias_float", torch.zeros(out_features))
        else:
            self.register_buffer("bias_float", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device

        # Per-batch activation observer (asymmetric per-tensor)
        x_obs = QuantizationObserver(
            QuantizationConfig(bit_width=self.config.bit_width, symmetric=False, per_channel=False)
        )
        with torch.no_grad():
            x_obs._update_stats(x)
            a_scale, a_zp = x_obs.calculate_scale_zero_point()

        a_scale = a_scale.to(dev)
        a_zp    = a_zp.to(dev)

        x_int = torch.clamp(torch.round(x / a_scale) + a_zp, self.config.qmin, self.config.qmax).to(torch.int8)

        Wq   = self.weight_int.to(dev)
        w_s  = self.weight_scale.to(dev)
        w_zp = self.weight_zero_point.to(dev)

        x_c = (x_int.to(torch.int32) - a_zp.to(torch.int32))
        if self.config.per_channel:
            w_c = (Wq.to(torch.int32) - w_zp.view(-1, 1).to(torch.int32))
        else:
            w_c = (Wq.to(torch.int32) - w_zp.to(torch.int32))

        out_acc = F.linear(x_c.float(), w_c.float(), None)
        if self.config.per_channel:
            out = out_acc * (a_scale * w_s).view(1, -1)
        else:
            out = out_acc * (a_scale * w_s)

        if self.bias_float is not None:
            out = out + self.bias_float.to(dev)

        return out

    @classmethod
    def from_float(cls, float_module: nn.Linear, config: QuantizationConfig = None):
        cfg = config or QuantizationConfig()
        q = cls(float_module.in_features, float_module.out_features, float_module.bias is not None, cfg)
        dev = next(float_module.parameters()).device

        w_obs = QuantizationObserver(
            QuantizationConfig(bit_width=cfg.bit_width, symmetric=True, per_channel=cfg.per_channel)
        )
        w_obs._update_stats(float_module.weight.data)
        w_scale, w_zp = w_obs.calculate_scale_zero_point()

        if cfg.per_channel:
            w_int = torch.round(float_module.weight / w_scale.view(-1, 1)) + w_zp.view(-1, 1)
        else:
            w_int = torch.round(float_module.weight / w_scale) + w_zp

        w_int = torch.clamp(w_int, cfg.qmin, cfg.qmax).to(torch.int8)

        q.weight_int.copy_(w_int.to(dev))
        q.weight_scale.copy_(w_scale.to(dev))
        q.weight_zero_point.copy_(w_zp.to(dev))

        if float_module.bias is not None:
            q.bias_float.copy_(float_module.bias.detach().to(dev))

        return q


# =========================
# Fake Quantize (QAT)
# =========================

class FakeQuantize(nn.Module):
    """
    Module for QAT-style fake quantization.
    - Uses an internal observer to track ranges.
    - If config.per_channel=True, treats the *last* dimension as channel for activations.
      (Common for [B, D] activations from Linear layers.)
    """
    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.observer = QuantizationObserver(config)
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.long))
        self.fake_quant_enabled = True

    def _broadcast_params(self, x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor):
        # per-tensor: scalar scale/zp
        if scale.dim() == 0:
            return scale, zp
        # per-channel: assume last-dim channels
        shape = [1] * x.dim()
        shape[-1] = scale.shape[0]
        return scale.view(*shape), zp.view(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update observer stats during training
        x = self.observer(x)

        if not self.fake_quant_enabled:
            return x

        scale, zp = self.observer.calculate_scale_zero_point()
        scale = scale.to(x.device)
        zp    = zp.to(x.device)

        # Broadcast to activation shape if per-channel
        scale_b, zp_b = self._broadcast_params(x, scale, zp)

        # Quantize -> clamp -> Dequantize (fake-quant)
        x_int = torch.round(x / scale_b) + zp_b
        x_int = torch.clamp(x_int, self.config.qmin, self.config.qmax)
        x_fq  = (x_int - zp_b) * scale_b
        return x_fq

    def calculate_qparams(self):
        self.scale, self.zero_point = self.observer.calculate_scale_zero_point()

    def disable_observer(self):
        self.observer.enabled = False

    def disable_fake_quant(self):
        self.fake_quant_enabled = False


# =========================
# Convenience wrappers
# =========================

class StaticQuantizedLinear(QuantizedLinear):
    """Alias to distinguish static from dynamic in codebases."""
    def __init__(self, in_features, out_features, bias=True, config: QuantizationConfig = None):
        super().__init__(in_features, out_features, bias, config)


class ManualQuantStub(nn.Module):
    """
    A simple activation fake-quant stub for QAT graphs.
    Typically used like:
        self.quant = ManualQuantStub(QuantizationConfig(per_channel=False))
        self.dequant = ManualDeQuantStub()
        x = self.quant(x); y = self.linear(x); y = self.dequant(y)
    """
    def __init__(self, config: QuantizationConfig = None):
        super().__init__()
        self.config = config or QuantizationConfig()
        self.fake_quant = FakeQuantize(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fake_quant(x)


class ManualDeQuantStub(nn.Module):
    """No-op dequant stub to mirror standard FX/QAT APIs."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
