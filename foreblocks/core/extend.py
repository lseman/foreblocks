import copy

# -----------------------------
# Graph + Per-Node Core Wrapper
# -----------------------------
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.quantization import (
    DynamicQuantizedLinear,
    ManualDeQuantStub,
    ManualQuantStub,
    StaticQuantizedLinear,
)

from .model import ForecastingModel


class DistilledForecastingModel(ForecastingModel):
    """
    Forecasting model with knowledge distillation support (output/feature/attention).
    - Robust nested-hook registration (supports dotted layer paths).
    - Cached, trainable adapters for feature-width/head-count mismatches.
    - AMP-safe interpolation and KL paths.
    - Optional alpha/temperature schedules and per-component loss weights.

    Public APIs kept similar to your original version.
    """

    VALID_DISTILLATION_MODES = [
        "none",
        "output",
        "feature",
        "attention",
        "comprehensive",
    ]

    def __init__(
        self,
        distillation_mode: str = "none",
        teacher_model: Optional[nn.Module] = None,
        distillation_temperature: float = 4.0,
        distillation_alpha: float = 0.7,
        feature_distillation_layers: Optional[List[str]] = None,
        attention_distillation_layers: Optional[List[str]] = None,
        # NEW:
        task_type: str = "regression",  # "regression" | "logits"
        alpha_schedule=None,  # Optional[Callable[[int], float]]
        temp_schedule=None,  # Optional[Callable[[int], float]]
        loss_weights: Optional[
            Dict[str, float]
        ] = None,  # keys: "output","feature","attention"
        **kwargs,
    ):
        assert distillation_mode in self.VALID_DISTILLATION_MODES, (
            f"Invalid distillation mode: {distillation_mode}"
        )

        super().__init__(**kwargs)

        self.distillation_mode = distillation_mode
        self.teacher_model = teacher_model
        self.distillation_temperature = float(distillation_temperature)
        self.distillation_alpha = float(distillation_alpha)
        self.feature_distillation_layers = feature_distillation_layers or []
        self.attention_distillation_layers = attention_distillation_layers or []

        # Schedules and task type
        self.task_type = task_type
        self.alpha_schedule = alpha_schedule
        self.temp_schedule = temp_schedule

        # Per-component loss weights
        self.loss_weights = {
            "output": 1.0,
            "feature": 0.5,
            "attention": 0.3,
        }
        if loss_weights:
            self.loss_weights.update(loss_weights)

        # Hook storages
        self.feature_hooks: Dict[str, any] = {}
        self.attention_hooks: Dict[str, any] = {}
        self.teacher_features: Dict[str, torch.Tensor] = {}
        self.teacher_attentions: Dict[str, torch.Tensor] = {}
        self.student_features: Dict[str, torch.Tensor] = {}
        self.student_attentions: Dict[str, torch.Tensor] = {}

        # Cached adapters (registered parameters)
        self._feat_adapters = nn.ModuleDict()  # feature width adapters
        self._head_adapters = nn.ModuleDict()  # attention head adapters

        if self.distillation_mode != "none":
            self._setup_distillation()

    # ==================== UTILITIES ====================

    @staticmethod
    def _get_module_by_name(root: nn.Module, dotted: str) -> Optional[nn.Module]:
        if not dotted:
            return None
        mod = root
        for part in dotted.split("."):
            if not hasattr(mod, part):
                return None
            mod = getattr(mod, part)
        return mod

    def _clear_hooks(self):
        for h in self.feature_hooks.values():
            try:
                h.remove()
            except Exception:
                pass
        for h in self.attention_hooks.values():
            try:
                h.remove()
            except Exception:
                pass
        self.feature_hooks.clear()
        self.attention_hooks.clear()

    # ==================== DISTILLATION SETUP ====================

    def _setup_distillation(self):
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        # Idempotent re-registration
        self._clear_hooks()

        if self.distillation_mode in {"feature", "comprehensive"}:
            self._register_hooks(
                self.feature_distillation_layers,
                self.teacher_features,
                self.student_features,
                self.feature_hooks,
                attention=False,
            )

        if self.distillation_mode in {"attention", "comprehensive"}:
            self._register_hooks(
                self.attention_distillation_layers,
                self.teacher_attentions,
                self.student_attentions,
                self.attention_hooks,
                attention=True,
            )

    def _register_hooks(
        self, layer_names, teacher_store, student_store, hook_store, attention=False
    ):
        def create_hook(name, store, attention_capture: bool):
            def hook(module, inputs, output):
                if attention_capture:
                    att = None
                    if isinstance(output, (tuple, list)) and len(output) >= 2:
                        att = output[1]
                    elif isinstance(output, dict) and "attn" in output:
                        att = output["attn"]
                    elif hasattr(module, "attention_weights"):
                        att = module.attention_weights
                    store[name] = att.detach() if torch.is_tensor(att) else att
                else:
                    feat = output[0] if isinstance(output, (tuple, list)) else output
                    store[name] = feat.detach() if torch.is_tensor(feat) else feat

            return hook

        for name in layer_names:
            t_mod = (
                self._get_module_by_name(self.teacher_model, name)
                if self.teacher_model
                else None
            )
            s_mod = self._get_module_by_name(self, name)
            if t_mod is not None:
                h = t_mod.register_forward_hook(
                    create_hook(name, teacher_store, attention)
                )
                hook_store[f"teacher:{name}"] = h
            if s_mod is not None:
                h = s_mod.register_forward_hook(
                    create_hook(name, student_store, attention)
                )
                hook_store[f"student:{name}"] = h

    # ==================== DISTILLATION LOSSES ====================

    def _compute_output_distillation_loss(
        self, student_out, teacher_out
    ) -> torch.Tensor:
        if self.task_type == "regression":
            return F.mse_loss(student_out, teacher_out)

        # classification/logit distillation
        T = float(self.distillation_temperature)
        # Avoid AMP dtype issues in softmax/kl
        with torch.cuda.amp.autocast(enabled=False):
            ps = F.log_softmax(student_out.float() / T, dim=-1)
            pt = F.softmax(teacher_out.float() / T, dim=-1)
            return F.kl_div(ps, pt, reduction="batchmean") * (T**2)

    def _compute_feature_distillation_loss(self) -> torch.Tensor:
        return self._compute_hooked_loss(
            self.student_features,
            self.teacher_features,
            self.feature_distillation_layers,
            self._align_feature_dimensions,
        )

    def _compute_attention_distillation_loss(self) -> torch.Tensor:
        return self._compute_hooked_loss(
            self.student_attentions,
            self.teacher_attentions,
            self.attention_distillation_layers,
            self._align_attention_dimensions,
        )

    def _compute_hooked_loss(
        self, student_dict, teacher_dict, layer_names, align_fn
    ) -> torch.Tensor:
        total_loss, count = 0.0, 0
        for name in layer_names:
            if name not in student_dict or name not in teacher_dict:
                continue
            s_val, t_val = student_dict[name], teacher_dict[name]
            if s_val is None or t_val is None:
                continue

            # Unwrap tuples/lists -> first tensor
            s_val = s_val[0] if isinstance(s_val, (tuple, list)) else s_val
            t_val = t_val[0] if isinstance(t_val, (tuple, list)) else t_val
            if not (
                isinstance(s_val, torch.Tensor) and isinstance(t_val, torch.Tensor)
            ):
                continue

            if s_val.shape != t_val.shape:
                s_val = align_fn(s_val, t_val.shape, key=name)

            # You can switch to KL for attention maps if desired:
            # if align_fn == self._align_attention_dimensions:
            #     with torch.cuda.amp.autocast(enabled=False):
            #         ps = s_val.float().log_softmax(dim=-1)
            #         pt = t_val.float().softmax(dim=-1)
            #         loss = F.kl_div(ps, pt, reduction="batchmean")
            # else:
            loss = F.mse_loss(s_val, t_val)

            total_loss += loss
            count += 1

        return total_loss / max(count, 1)

    # ==================== DIMENSION ALIGNMENT ====================

    def _align_feature_dimensions(
        self, student_feat: torch.Tensor, target_shape, key=None
    ):
        # Expect (B, T, C) or (B, *, C)
        # Width projection
        if student_feat.shape[-1] != target_shape[-1]:
            proj_key = f"feat:{key or str(target_shape)}"
            if proj_key not in self._feat_adapters:
                self._feat_adapters[proj_key] = nn.Linear(
                    student_feat.shape[-1],
                    target_shape[-1],
                    bias=False,
                    device=student_feat.device,
                    dtype=student_feat.dtype,
                )
            student_feat = self._feat_adapters[proj_key](student_feat)

        # Sequence/time resize if 3D (B,T,C)
        if student_feat.dim() >= 3 and student_feat.shape[1] != target_shape[1]:
            need = target_shape[1]
            with torch.cuda.amp.autocast(enabled=False):
                x32 = student_feat.float().transpose(1, 2)  # (B,C,T)
                x32 = F.interpolate(x32, size=need, mode="linear", align_corners=False)
                student_feat = x32.transpose(1, 2).to(student_feat.dtype)
        return student_feat

    def _align_attention_dimensions(
        self, student_att: torch.Tensor, target_shape, key=None
    ):
        if student_att is None:
            return None
        if student_att.shape == target_shape:
            return student_att

        B, Hs, L1, L2 = student_att.shape
        Bt, Ht, L1t, L2t = target_shape

        att = student_att

        # Head projection Hs -> Ht (learned, works for non-multiples)
        if Hs != Ht:
            proj_key = f"head:{key or str(target_shape)}"
            if proj_key not in self._head_adapters:
                self._head_adapters[proj_key] = nn.Linear(
                    Hs, Ht, bias=False, device=att.device, dtype=att.dtype
                )
            A = att.permute(0, 2, 3, 1).reshape(-1, Hs)  # (B*L1*L2, Hs)
            A = self._head_adapters[proj_key](A)  # (B*L1*L2, Ht)
            att = A.reshape(B, L1, L2, Ht).permute(0, 3, 1, 2)  # (B, Ht, L1, L2)

        # Spatial/temporal resize
        if (L1, L2) != (L1t, L2t):
            with torch.cuda.amp.autocast(enabled=False):
                x32 = att.float().reshape(B * att.shape[1], 1, L1, L2)
                x32 = F.interpolate(
                    x32, size=(L1t, L2t), mode="bilinear", align_corners=False
                )
                att = x32.reshape(B, -1, L1t, L2t).to(att.dtype)

        return att

    # ==================== LOSS COMBINATION ====================

    def _combine_distillation_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Combine task loss and available distillation losses with alpha and per-component weights.
        Expects `losses` to contain at least "task_loss"; optionally:
          - "output_distillation", "feature_distillation", "attention_distillation"
        """
        task = losses.get("task_loss", 0.0)
        distill = 0.0
        if "output_distillation" in losses:
            distill += self.loss_weights["output"] * losses["output_distillation"]
        if "feature_distillation" in losses:
            distill += self.loss_weights["feature"] * losses["feature_distillation"]
        if "attention_distillation" in losses:
            distill += self.loss_weights["attention"] * losses["attention_distillation"]

        alpha = float(self.distillation_alpha)
        return (1 - alpha) * task + alpha * distill

    # ==================== CONTROL ====================

    def set_teacher_model(self, model: nn.Module):
        self.teacher_model = model
        if model is not None:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            if self.distillation_mode != "none":
                self._setup_distillation()

    def enable_distillation(self, mode="output", teacher_model=None):
        assert mode in self.VALID_DISTILLATION_MODES, (
            f"Invalid distillation mode: {mode}"
        )
        self.distillation_mode = mode
        if teacher_model is not None:
            self.set_teacher_model(teacher_model)
        self._setup_distillation()

    def disable_distillation(self):
        self.distillation_mode = "none"
        self._clear_hooks()
        self.teacher_model = None
        self.teacher_features.clear()
        self.teacher_attentions.clear()
        self.student_features.clear()
        self.student_attentions.clear()

    def get_distillation_info(self) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "distillation_enabled": self.distillation_mode != "none",
            "distillation_mode": self.distillation_mode,
            "has_teacher": self.teacher_model is not None,
            "temperature": float(self.distillation_temperature),
            "alpha": float(self.distillation_alpha),
            "feature_layers": len(self.feature_distillation_layers),
            "attention_layers": len(self.attention_distillation_layers),
            "active_feature_hooks": len(self.feature_hooks),
            "active_attention_hooks": len(self.attention_hooks),
            "task_type": self.task_type,
        }

    # ==================== OVERRIDES ====================

    def forward(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        return_teacher_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Update schedules (if provided)
        if self.distillation_mode != "none" and self.teacher_model is not None:
            if self.alpha_schedule is not None and epoch is not None:
                self.distillation_alpha = float(self.alpha_schedule(epoch))
            if self.temp_schedule is not None and epoch is not None:
                self.distillation_temperature = float(self.temp_schedule(epoch))

        # Clear per-forward caches
        self.student_features.clear()
        self.student_attentions.clear()

        teacher_output = None
        if self.distillation_mode != "none" and self.teacher_model is not None:
            self.teacher_features.clear()
            self.teacher_attentions.clear()
            with torch.no_grad():
                teacher_output = self.teacher_model(src, targets, time_features, epoch)

        student_output = super().forward(src, targets, time_features, epoch)
        if return_teacher_outputs and teacher_output is not None:
            return student_output, teacher_output
        return student_output

    def benchmark_inference(
        self, input_tensor: torch.Tensor, num_runs=100, warmup_runs=10
    ):
        result = super().benchmark_inference(input_tensor, num_runs, warmup_runs)
        result.update(
            {
                "distillation_mode": self.distillation_mode,
                "has_teacher": self.teacher_model is not None,
            }
        )
        return result


# ==================== QUANTIZED FORECASTING MODEL ====================


class QuantizedForecastingModel(DistilledForecastingModel):
    """
    Forecasting model with quantization support on top of distillation-enabled model.
    Supports PTQ, QAT, static and dynamic quantization.
    """

    VALID_QUANTIZATION_MODES = ["none", "ptq", "qat", "dynamic", "static"]

    def __init__(
        self,
        quantization_mode: str = "none",
        bit_width: int = 8,
        symmetric_quantization: bool = True,
        per_channel_quantization: bool = False,
        **kwargs,
    ):
        assert quantization_mode in self.VALID_QUANTIZATION_MODES, (
            f"Invalid quantization mode: {quantization_mode}"
        )

        super().__init__(**kwargs)

        # Core quantization config
        self.quantization_mode = quantization_mode
        self.bit_width = bit_width
        self.symmetric_quantization = symmetric_quantization
        self.per_channel_quantization = per_channel_quantization
        self.is_quantized = False

        self.quantization_config = QuantizationConfig(
            bit_width=bit_width,
            symmetric=symmetric_quantization,
            per_channel=per_channel_quantization,
        )

        # Quantization stubs (only needed for static/PTQ/QAT)
        if quantization_mode in {"ptq", "qat", "static"}:
            self.quant = ManualQuantStub(self.quantization_config)
            self.dequant = ManualDeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

        if quantization_mode != "none":
            self._setup_quantization()

    # ==================== SETUP ====================

    def _setup_quantization(self):
        """Setup quantization observers or fake quant depending on mode"""
        self._add_quantization_observers()

    def _add_quantization_observers(self):
        """Attach fake quant observers to Linear layers"""
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear) and self.quantization_mode == "qat":
                module.weight_fake_quant = FakeQuantize(self.quantization_config)
                if module.bias is not None:
                    module.bias_fake_quant = FakeQuantize(self.quantization_config)

    def set_quantization_mode(self, mode: str):
        """Switch quantization mode"""
        assert mode in self.VALID_QUANTIZATION_MODES, f"Invalid mode: {mode}"
        self.quantization_mode = mode

        if mode == "none":
            self.quant, self.dequant = nn.Identity(), nn.Identity()
        elif mode in {"ptq", "qat", "static"}:
            self.quant = ManualQuantStub(self.quantization_config)
            self.dequant = ManualDeQuantStub()
        else:  # dynamic
            self.quant, self.dequant = nn.Identity(), nn.Identity()

    # ==================== PREPARATION MODES ====================

    def prepare_for_quantization(self, calibration_data=None):
        if self.quantization_mode == "none":
            return self
        elif self.quantization_mode == "dynamic":
            return self._apply_dynamic_quantization()
        elif self.quantization_mode == "ptq":
            return self._apply_post_training_quantization(calibration_data)
        elif self.quantization_mode == "qat":
            return self._prepare_qat_training()
        elif self.quantization_mode == "static":
            return self._apply_static_quantization()

    def _apply_dynamic_quantization(self):
        """Dynamically quantize Linear layers"""
        return self._replace_linear_layers(DynamicQuantizedLinear)

    def _apply_static_quantization(self):
        """Apply static quantization after observer setup"""
        return self._replace_linear_layers(StaticQuantizedLinear)

    def _apply_post_training_quantization(self, calibration_data):
        """PTQ = Calibrate then convert"""
        self._add_quantization_observers()
        if calibration_data:
            self._calibrate_model(calibration_data)
        return self._convert_to_quantized_model()

    def _prepare_qat_training(self):
        """Attach fake quant modules and enable them for QAT"""
        self._add_quantization_observers()
        for m in self.modules():
            if hasattr(m, "weight_fake_quant"):
                m.weight_fake_quant.fake_quant_enabled = True
        return self

    def _calibrate_model(self, dataloader):
        """Collect statistics from calibration data"""
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                src, targets = (
                    batch[0],
                    batch[1] if isinstance(batch, (tuple, list)) else (batch, None),
                )
                _ = self.forward(src, targets)

    def finalize_quantization(self):
        """Convert QAT-trained model to fully quantized version"""
        if self.quantization_mode == "qat":
            for m in self.modules():
                if hasattr(m, "weight_fake_quant"):
                    m.weight_fake_quant.calculate_qparams()
                    m.weight_fake_quant.disable_observer()
                    m.weight_fake_quant.disable_fake_quant()
            return self._convert_to_quantized_model()
        return self

    def _convert_to_quantized_model(self):
        """Convert fake-quant-aware model to real quantized model"""
        return self._replace_linear_layers(StaticQuantizedLinear)

    # ==================== MODULE REPLACEMENT ====================

    def _replace_linear_layers(self, QuantLayerClass):
        """Replace all Linear layers with quantized equivalents"""
        model = copy.deepcopy(self)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quant_module = QuantLayerClass.from_float(
                    module, self.quantization_config
                )
                self._assign_module_by_name(model, name, quant_module)
        model.is_quantized = True
        return model

    def _assign_module_by_name(self, model, full_name, new_module):
        """Replace a submodule in the model hierarchy given its full dotted path"""
        path = full_name.split(".")
        parent = model
        for name in path[:-1]:
            parent = getattr(parent, name)
        setattr(parent, path[-1], new_module)

    # ==================== FORWARD / METRICS ====================

    def forward(
        self,
        src,
        targets=None,
        time_features=None,
        epoch=None,
        return_teacher_outputs=False,
    ):
        if self.quantization_mode in {"ptq", "qat", "static"}:
            src = self.quant(src)
        output = super().forward(
            src, targets, time_features, epoch, return_teacher_outputs
        )
        if self.quantization_mode in {"ptq", "qat", "static"}:
            if isinstance(output, tuple):
                output = (self.dequant(output[0]), output[1])
            else:
                output = self.dequant(output)
        return output

    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Estimate model size in MB depending on quantization"""
        param_count = sum(p.numel() for p in self.parameters())
        buffer_count = sum(b.numel() for b in self.buffers())
        element_count = param_count + buffer_count
        element_size = 1 if self.is_quantized else 4
        size_mb = element_count * element_size / (1024**2)
        return {
            "parameters": param_count,
            "buffers": buffer_count,
            "total_elements": element_count,
            "size_mb": size_mb,
            "is_quantized": self.is_quantized,
        }

    def benchmark_inference(
        self, input_tensor: torch.Tensor, num_runs=100, warmup_runs=10
    ):
        result = super().benchmark_inference(input_tensor, num_runs, warmup_runs)
        result.update(
            {
                "quantization_mode": self.quantization_mode,
                "is_quantized": self.is_quantized,
            }
        )
        return result

    def get_quantization_info(self) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "quantization_enabled": self.quantization_mode != "none",
            "quantization_mode": self.quantization_mode,
            "bit_width": self.bit_width,
            "symmetric": self.symmetric_quantization,
            "per_channel": self.per_channel_quantization,
            "is_quantized": self.is_quantized,
            "num_quantizable_layers": sum(
                isinstance(m, nn.Linear) for m in self.modules()
            ),
        }
