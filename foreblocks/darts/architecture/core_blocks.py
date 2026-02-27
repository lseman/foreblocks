import copy
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.tf.norms import RevIN

from .base_blocks import MixedDecoder, MixedEncoder
from .operation_blocks import (
    ConvMixerOp,
    DLinearOp,
    FourierOp,
    GRNOp,
    IdentityOp,
    InvertedAttentionOp,
    MambaOp,
    MLPMixerOp,
    MultiScaleConvOp,
    PatchEmbedOp,
    PyramidConvOp,
    ResidualMLPOp,
    RMSNorm,
    TCNOp,
    TimeConvOp,
    WaveletOp,
)


# --- Searchable Normalization ---
class SearchableNorm(nn.Module):
    """Searchable normalization: RevIN vs InstanceNorm vs passthrough"""

    def __init__(self, num_features):
        super().__init__()
        self.revin = RevIN(num_features)
        self.inst_norm = nn.InstanceNorm1d(num_features, affine=True)

    def forward(self, x, alpha):  # x: [B, L, C]
        w = F.softmax(alpha, dim=0)
        rev = self.revin(x, mode="norm")
        inst = self.inst_norm(x.transpose(1, 2)).transpose(1, 2)
        return w[0] * rev + w[1] * inst + w[2] * x

    def apply_output_denorm(
        self, y: torch.Tensor, alpha: torch.Tensor, selected_norm: Optional[str] = None
    ) -> torch.Tensor:
        """
        Denormalize output when RevIN is selected.
        - If selected_norm is provided (derived/fixed architecture), use hard behavior.
        - During search, use straight-through hard routing to avoid partial denorm blending.
        """
        if selected_norm is not None:
            if str(selected_norm).lower() != "revin":
                return y
            try:
                return self.revin(y, mode="denorm")
            except Exception:
                return y

        w_soft = F.softmax(alpha, dim=0)
        hard = torch.zeros_like(w_soft)
        hard[int(torch.argmax(w_soft).item())] = 1.0
        # Straight-through estimator: forward is hard, backward follows soft weights.
        w = hard - w_soft.detach() + w_soft if self.training else hard
        try:
            rev_denorm = self.revin(y, mode="denorm")
        except Exception:
            return y
        return w[0] * rev_denorm + (1.0 - w[0]) * y


@dataclass
class DARTSConfig:
    """Configuration class for DARTS model"""

    input_dim: int = 3
    hidden_dim: int = 64
    latent_dim: int = 64
    forecast_horizon: int = 24
    seq_length: int = 48
    num_cells: int = 2
    num_nodes: int = 4
    dropout: float = 0.1
    initial_search: bool = False
    selected_ops: Optional[List[str]] = None
    loss_type: str = "huber"
    use_gradient_checkpointing: bool = False
    temperature: float = 1.0
    use_mixed_precision: bool = True
    use_compile: bool = False
    memory_efficient: bool = True
    single_path_search: bool = True
    use_learned_memory_pooling: bool = True
    memory_num_queries: int = 8

    # New optimization parameters
    arch_lr: float = 3e-4
    weight_lr: float = 1e-3
    alpha_l2_reg: float = 1e-3
    edge_normalization: bool = True
    progressive_shrinking: bool = True

    # MixedOp search-stability knobs
    group_temperature_mult: float = 1.5
    min_group_temperature: float = 0.7
    min_op_temperature: float = 0.3
    group_min_prob: float = 0.03
    competition_power: float = 1.25
    adaptive_bias_scale: float = 0.15
    performance_ema_decay: float = 0.95
    pc_ratio: float = 0.25
    use_drnas: bool = True
    drnas_concentration: float = 8.0
    use_fair_darts_hierarchical: bool = True

    @classmethod
    def with_search_profile(
        cls, profile: str = "conservative", **overrides
    ) -> "DARTSConfig":
        """Create config using a named MixedOp search profile."""
        profiles = {
            "conservative": {
                "group_temperature_mult": 1.8,
                "min_group_temperature": 0.9,
                "min_op_temperature": 0.5,
                "group_min_prob": 0.05,
                "competition_power": 1.1,
                "adaptive_bias_scale": 0.10,
                "performance_ema_decay": 0.97,
                "use_drnas": True,
                "drnas_concentration": 10.0,
                "use_fair_darts_hierarchical": True,
            },
            "aggressive": {
                "group_temperature_mult": 1.2,
                "min_group_temperature": 0.6,
                "min_op_temperature": 0.25,
                "group_min_prob": 0.01,
                "competition_power": 1.5,
                "adaptive_bias_scale": 0.20,
                "performance_ema_decay": 0.90,
                "use_drnas": True,
                "drnas_concentration": 6.0,
                "use_fair_darts_hierarchical": True,
            },
        }

        if profile not in profiles:
            valid = ", ".join(sorted(profiles.keys()))
            raise ValueError(f"Unknown search profile '{profile}'. Valid: {valid}")

        params = {**profiles[profile], **overrides}
        return cls(**params)


class MixedOp(nn.Module):
    """Enhanced MixedOp using your existing operators with better search strategy"""

    _efficiency_cache: Dict[Any, Dict[str, float]] = {}
    _flops_warning_emitted: bool = False

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        available_ops: Optional[List[str]] = None,
        drop_prob: float = 0.1,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        num_nodes: int = 4,
        use_hierarchical: bool = True,
        adaptive_sampling: bool = True,
        group_temperature_mult: float = 1.5,
        min_group_temperature: float = 0.7,
        min_op_temperature: float = 0.3,
        group_min_prob: float = 0.03,
        competition_power: float = 1.25,
        adaptive_bias_scale: float = 0.15,
        performance_ema_decay: float = 0.95,
        pc_ratio: float = 0.25,
        use_drnas: bool = True,
        drnas_concentration: float = 8.0,
        use_fair_darts_hierarchical: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.drop_prob = drop_prob
        self.temperature = temperature
        self.group_temperature_mult = group_temperature_mult
        self.min_group_temperature = min_group_temperature
        self.min_op_temperature = min_op_temperature
        self.group_temperature = max(
            temperature * self.group_temperature_mult, self.min_group_temperature
        )
        self.op_temperature = max(temperature, self.min_op_temperature)
        self.use_gumbel = use_gumbel
        self.use_drnas = use_drnas
        self.drnas_concentration = max(drnas_concentration, 1e-3)
        self.use_fair_darts_hierarchical = use_fair_darts_hierarchical
        self.use_hierarchical = use_hierarchical
        self.adaptive_sampling = adaptive_sampling
        self.group_min_prob = group_min_prob
        self.competition_power = competition_power
        self.adaptive_bias_scale = adaptive_bias_scale
        self.performance_ema_decay = performance_ema_decay
        self.pc_ratio = float(min(max(pc_ratio, 0.0), 1.0))

        # Define operation map using your existing operators
        self.op_map = {
            "Identity": lambda: IdentityOp(input_dim, latent_dim),
            "TimeConv": lambda: TimeConvOp(input_dim, latent_dim),
            "ResidualMLP": lambda: ResidualMLPOp(input_dim, latent_dim),
            "Wavelet": lambda: WaveletOp(input_dim, latent_dim),
            "Fourier": lambda: FourierOp(input_dim, latent_dim, seq_length),
            "TCN": lambda: TCNOp(input_dim, latent_dim),
            "ConvMixer": lambda: ConvMixerOp(input_dim, latent_dim),
            "GRN": lambda: GRNOp(input_dim, latent_dim),
            "MultiScaleConv": lambda: MultiScaleConvOp(input_dim, latent_dim),
            "PyramidConv": lambda: PyramidConvOp(input_dim, latent_dim),
            "Mamba": lambda: MambaOp(input_dim, latent_dim),
            "PatchEmbed": lambda: PatchEmbedOp(input_dim, latent_dim, patch_size=16),
            "iTransformerBlock": lambda: InvertedAttentionOp(input_dim, latent_dim),
            "TimeMixer": lambda: MLPMixerOp(input_dim, latent_dim, seq_length),
            "DLinear": lambda: DLinearOp(input_dim, latent_dim),
        }

        # Group operations by complexity/type for hierarchical search
        self.operation_groups = {
            "basic": ["Identity", "ResidualMLP", "DLinear"],
            "temporal": ["TimeConv", "TCN", "ConvMixer", "TimeMixer"],
            "frequency": ["Fourier", "Wavelet"],
            "advanced": [
                "GRN",
                "MultiScaleConv",
                "PyramidConv",
                "Mamba",
                "PatchEmbed",
                "iTransformerBlock",
            ],
        }

        # Initialize operations
        self.available_ops = self._validate_ops(available_ops)

        if use_hierarchical:
            self._init_hierarchical_search()
        else:
            self._init_flat_search()

        # Operation efficiency scores (for regularization)
        self.op_efficiency = {
            "Identity": 1.0,
            "ResidualMLP": 0.8,
            "TimeConv": 0.7,
            "TCN": 0.5,
            "ConvMixer": 0.6,
            "Fourier": 0.4,
            "Wavelet": 0.4,
            "GRN": 0.6,
            "MultiScaleConv": 0.3,
            "PyramidConv": 0.2,
            "Mamba": 0.65,
            "PatchEmbed": 0.7,
            "iTransformerBlock": 0.55,
            "TimeMixer": 0.68,
            "DLinear": 0.9,
        }

        # Adaptive sampling weights
        if adaptive_sampling:
            self.performance_tracker = nn.Parameter(
                torch.zeros(len(self.available_ops)), requires_grad=False
            )
            self.usage_counter = nn.Parameter(
                torch.zeros(len(self.available_ops)), requires_grad=False
            )

        # Fallback operation
        self.fallback_idx = (
            self.available_ops.index("Identity")
            if "Identity" in self.available_ops
            else 0
        )

        # Output projection for dimension mismatch
        self.output_proj = nn.Identity()  # Will be replaced if needed
        self._flops_profiled = False
        self._dynamic_efficiency_profiled = False
        self._profile_flops()

    def _efficiency_cache_key(self) -> Any:
        return (
            tuple(self.available_ops),
            int(self.input_dim),
            int(self.latent_dim),
            int(self.seq_length),
        )

    def _profile_flops(self):
        """Profile operation FLOPs and convert them to normalized efficiency scores."""
        if self._flops_profiled:
            return

        cache_key = self._efficiency_cache_key()
        if cache_key in MixedOp._efficiency_cache:
            self.op_efficiency.update(copy.deepcopy(MixedOp._efficiency_cache[cache_key]))
            self._dynamic_efficiency_profiled = True
            self._flops_profiled = True
            return

        try:
            from fvcore.nn import FlopCountAnalysis
        except Exception:
            if not MixedOp._flops_warning_emitted:
                warnings.warn(
                    "fvcore is not installed; MixedOp FLOPs-aware efficiency profiling "
                    "is disabled and static efficiency priors are used.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                MixedOp._flops_warning_emitted = True
            self._flops_profiled = True
            return

        ref_param = next(self.parameters(), None)
        device = ref_param.device if ref_param is not None else torch.device("cpu")
        dtype = ref_param.dtype if ref_param is not None else torch.float32
        dummy = torch.randn(2, self.seq_length, self.latent_dim, device=device, dtype=dtype)

        inverse_scores: Dict[str, float] = {}
        for op_name, op in zip(self.available_ops, self.ops):
            was_training = op.training
            try:
                op.eval()
                flops = float(FlopCountAnalysis(op, dummy).total())
                if math.isfinite(flops) and flops > 0:
                    inverse_scores[op_name] = 1.0 / (flops / 1e9 + 1e-6)
            except Exception:
                continue
            finally:
                op.train(was_training)

        if inverse_scores:
            vals = list(inverse_scores.values())
            lo, hi = min(vals), max(vals)
            if hi - lo > 1e-12:
                profiled_eff = {
                    name: float((score - lo) / (hi - lo))
                    for name, score in inverse_scores.items()
                }
            else:
                profiled_eff = {name: 1.0 for name in inverse_scores}

            self.op_efficiency.update(profiled_eff)
            MixedOp._efficiency_cache[cache_key] = copy.deepcopy(profiled_eff)
            self._dynamic_efficiency_profiled = True

        self._flops_profiled = True

    def _validate_ops(self, ops):
        """Validate and filter available operations"""
        if not ops:
            return ["Identity", "DLinear", "TimeConv", "ResidualMLP", "TCN"]

        # Filter valid operations
        valid_ops = [op for op in ops if op in self.op_map]

        # Ensure minimum operations
        if len(valid_ops) < 2:
            valid_ops = ["Identity", "TimeConv"]

        return valid_ops

    def _init_hierarchical_search(self):
        """Initialize hierarchical search with group and operation level alphas"""
        # Group level parameters
        active_groups = {}
        for group_name, group_ops in self.operation_groups.items():
            group_valid_ops = [op for op in group_ops if op in self.available_ops]
            if group_valid_ops:
                active_groups[group_name] = group_valid_ops

        self.active_groups = active_groups
        self.group_names = list(active_groups.keys())

        # Group selection parameters
        self.group_alphas = nn.Parameter(torch.randn(len(self.group_names)) * 0.1)

        # Operation modules and parameters for each group
        self.op_alphas = nn.ParameterDict()

        self.op_to_group = {}  # Map operation to group index
        self.group_op_indices = {}  # Map group to operation indices

        self.ops = nn.ModuleList()
        all_ops = []
        for group_idx, (group_name, group_ops) in enumerate(active_groups.items()):
            # Create alpha parameters for this group
            self.op_alphas[group_name] = nn.Parameter(torch.randn(len(group_ops)) * 0.1)

            # Track mappings
            start_idx = len(all_ops)
            for local_idx, op in enumerate(group_ops):
                global_idx = start_idx + local_idx
                self.op_to_group[global_idx] = (group_idx, local_idx)
                self.ops.append(self.op_map[op]())
                all_ops.append(op)

            self.group_op_indices[group_name] = list(
                range(start_idx, start_idx + len(group_ops))
            )

        self.available_ops = all_ops
        self._flops_profiled = False
        if self.adaptive_sampling and hasattr(self, "performance_tracker"):
            if self.performance_tracker.numel() != len(self.available_ops):
                device = self.performance_tracker.device
                self.performance_tracker = nn.Parameter(
                    torch.zeros(len(self.available_ops), device=device),
                    requires_grad=False,
                )
                self.usage_counter = nn.Parameter(
                    torch.zeros(len(self.available_ops), device=device),
                    requires_grad=False,
                )

    def _init_flat_search(self):
        """Initialize flat search space"""
        self.ops = nn.ModuleList([self.op_map[op]() for op in self.available_ops])
        self._alphas = nn.Parameter(
            torch.randn(len(self.ops)) * 0.1
        )  # Use _alphas internally
        self._flops_profiled = False

    def _get_weights(self, top_k: Optional[int] = None):
        """Get operation weights with optional top-k selection"""
        if self.use_hierarchical:
            return self._get_hierarchical_weights(top_k)
        else:
            return self._get_flat_weights(top_k)

    def _get_hierarchical_weights(self, top_k: Optional[int] = None):
        """Get weights for hierarchical search"""
        # Keep group routing smoother than op-level routing to reduce early collapse.
        group_weights = F.softmax(self.group_alphas / self.group_temperature, dim=0)
        if self.group_min_prob > 0:
            group_weights = group_weights.clamp_min(self.group_min_prob)
            group_weights = group_weights / group_weights.sum()

        candidates = []
        for group_name, group_weight in zip(self.group_names, group_weights):
            if group_weight.item() <= 1e-8:
                continue

            op_logits = self.op_alphas[group_name]
            if self.adaptive_sampling and self.training:
                op_indices = self.group_op_indices[group_name]
                perf_bias = (
                    torch.tanh(self.performance_tracker[op_indices])
                    * self.adaptive_bias_scale
                )
                op_logits = op_logits + perf_bias

            if self.use_fair_darts_hierarchical:
                # Fair mode: independent gates, then global normalization across all ops.
                op_scores = self._sample_op_weights(
                    op_logits,
                    independent_sigmoid=True,
                    normalize=False,
                )
                op_scores = op_scores / max(op_scores.numel(), 1)
            else:
                op_scores = self._sample_op_weights(op_logits, independent_sigmoid=False)
                op_scores = op_scores.pow(self.competition_power)
                op_scores = op_scores / op_scores.sum().clamp_min(1e-8)

            for op_idx, op_score in zip(self.group_op_indices[group_name], op_scores):
                candidates.append((op_idx, group_weight * op_score))

        if not candidates:
            return []

        selected_ops = [op_idx for op_idx, _ in candidates]
        weight_tensor = torch.stack([w for _, w in candidates]).clamp_min(1e-8)

        if top_k is not None and len(selected_ops) > top_k:
            top_vals, top_idx = torch.topk(weight_tensor, top_k)
            selected_ops = [selected_ops[i] for i in top_idx.tolist()]
            weight_tensor = top_vals

        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-8)
        return list(zip(selected_ops, weight_tensor))

    def _get_flat_weights(self, top_k: Optional[int] = None):
        """Get weights for flat search"""
        logits = self._alphas
        if self.adaptive_sampling and self.training:
            logits = (
                logits + torch.tanh(self.performance_tracker) * self.adaptive_bias_scale
            )

        weights = self._sample_op_weights(logits)

        weights = weights.pow(self.competition_power)
        weights = weights / weights.sum().clamp_min(1e-8)

        selected_ops = [(i, w) for i, w in enumerate(weights) if w.item() > 1e-6]

        # Apply top-k selection
        if top_k is not None and len(selected_ops) > top_k:
            selected_ops = sorted(
                selected_ops, key=lambda x: x[1].item(), reverse=True
            )[:top_k]

        return selected_ops

    def _sample_op_weights(
        self,
        logits: torch.Tensor,
        independent_sigmoid: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Sample architecture weights with DrNAS Dirichlet or fallback soft relaxation."""
        if independent_sigmoid:
            if self.training and self.use_gumbel:
                u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
                logistic_noise = torch.log(u) - torch.log1p(-u)
                probs = torch.sigmoid((logits + logistic_noise) / self.op_temperature)
            else:
                probs = torch.sigmoid(logits / self.op_temperature)
            probs = probs.clamp_min(1e-8)
            if normalize:
                probs = probs / probs.sum().clamp_min(1e-8)
            return probs

        probs = F.softmax(logits / self.op_temperature, dim=0)
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum().clamp_min(1e-8)
        if self.training and self.use_drnas:
            concentration = probs * self.drnas_concentration
            concentration = concentration.clamp_min(1e-4)
            dist = torch.distributions.Dirichlet(concentration)
            return dist.rsample()

        if self.use_gumbel and self.training:
            return F.gumbel_softmax(logits, tau=self.op_temperature, hard=False)

        return probs

    def _ensure_output_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure output has correct dimensions"""
        if x.shape[-1] != self.latent_dim:
            if isinstance(self.output_proj, nn.Identity):
                self.output_proj = nn.Linear(x.shape[-1], self.latent_dim).to(x.device)
            x = self.output_proj(x)
        return x

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """Enhanced forward with better operation selection"""
        op_weights = self._get_weights(top_k)

        if not op_weights:
            # Fallback to identity or first operation
            fallback_out = self.ops[self.fallback_idx](x)
            return self._ensure_output_dim(fallback_out)

        outputs = []
        total_weight = 0
        efficiency_penalty = 0

        for op_idx, weight in op_weights:
            # DropPath (stochastic depth)
            if self.training and self.drop_prob > 0.0:
                if torch.rand((), device=x.device).item() < self.drop_prob:
                    continue

            op_input = x
            pc_scale = 1.0
            if self.training and 0.0 < self.pc_ratio < 1.0 and x.size(-1) > 1:
                channels = int(x.size(-1))
                active_channels = max(1, int(round(channels * self.pc_ratio)))
                if active_channels < channels:
                    idx = torch.randperm(channels, device=x.device)[:active_channels]
                    channel_mask = torch.zeros(channels, device=x.device, dtype=x.dtype)
                    channel_mask[idx] = 1.0
                    op_input = x * channel_mask.view(1, 1, channels)
                    pc_scale = channels / float(active_channels)

            try:
                out = self.ops[op_idx](op_input)
            except Exception as e:
                op_name = self.available_ops[op_idx]
                raise RuntimeError(
                    f"MixedOp forward failed for operation '{op_name}' (index={op_idx})"
                ) from e

            if pc_scale != 1.0:
                out = out * pc_scale

            out = self._ensure_output_dim(out)
            outputs.append(out * weight)
            total_weight += weight.item()

            # Track efficiency penalty
            op_name = self.available_ops[op_idx]
            if op_name in self.op_efficiency:
                efficiency_penalty += weight.item() * (1 - self.op_efficiency[op_name])

            # Update performance tracker if adaptive sampling
            if self.adaptive_sampling and self.training:
                with torch.no_grad():
                    # Closed-form gradient norm proxy of mean-square output energy:
                    # d mean(out^2) / d out = 2*out/N. Lower norm is more stable.
                    finite_ok = torch.isfinite(out).all()
                    if finite_ok:
                        out_det = out.detach()
                        grad_norm = (
                            2.0 * out_det.norm(p=2) / max(float(out_det.numel()), 1.0)
                        )
                        score = 1.0 / (float(grad_norm.item()) + 1e-6)
                        self.performance_tracker[op_idx].mul_(
                            self.performance_ema_decay
                        ).add_((1 - self.performance_ema_decay) * score)
                    else:
                        self.performance_tracker[op_idx].mul_(
                            self.performance_ema_decay
                        ).add_(-(1 - self.performance_ema_decay))

                    self.usage_counter[op_idx] += 1

        if outputs:
            result = sum(outputs) / max(total_weight, 1e-6)
            # Store efficiency penalty for regularization
            self._last_efficiency_penalty = efficiency_penalty * 0.01
            return result

        # Final fallback
        fallback_out = self.ops[self.fallback_idx](x)
        return self._ensure_output_dim(fallback_out)

    def _hierarchical_probability_vector(self, temperature: float = 1.0) -> torch.Tensor:
        group_temp = max(self.group_temperature * temperature, 1e-6)
        op_temp = max(self.op_temperature * temperature, 1e-6)

        group_weights = F.softmax(self.group_alphas / group_temp, dim=0)
        if self.group_min_prob > 0:
            group_weights = group_weights.clamp_min(self.group_min_prob)
            group_weights = group_weights / group_weights.sum().clamp_min(1e-8)

        scores = []
        for group_name, group_weight in zip(self.group_names, group_weights):
            if self.use_fair_darts_hierarchical:
                op_scores = torch.sigmoid(self.op_alphas[group_name] / op_temp)
                op_scores = op_scores.clamp_min(1e-8)
                op_scores = op_scores / max(op_scores.numel(), 1)
            else:
                op_scores = F.softmax(self.op_alphas[group_name] / op_temp, dim=0)
                op_scores = op_scores.pow(self.competition_power)
                op_scores = op_scores / op_scores.sum().clamp_min(1e-8)
            scores.append(group_weight * op_scores)

        if not scores:
            return torch.zeros(0, device=self.group_alphas.device)

        out = torch.cat(scores, dim=0).clamp_min(1e-8)
        return out / out.sum().clamp_min(1e-8)

    def get_alphas(self, temperature=1.0, detach=True):
        """Get architecture weights after temperature scaling (for reporting)"""
        if self.use_hierarchical:
            out = self._hierarchical_probability_vector(temperature=float(temperature))
            return out.detach() if detach else out

        alphas = self._alphas / (self.op_temperature * temperature)
        out = F.softmax(alphas, dim=0)
        return out.detach() if detach else out

    @property
    def alphas(self):
        """Compatibility property for accessing alphas"""
        if self.use_hierarchical:
            return self.get_alphas()
        else:
            return self._alphas

    @alphas.setter
    def alphas(self, value):
        """Compatibility setter for alphas"""
        if self.use_hierarchical:
            # For hierarchical, we need to distribute the values
            # This is a simplified approach - in practice you might want more sophisticated logic
            if hasattr(self, "group_alphas"):
                with torch.no_grad():
                    # Update group alphas with first few values
                    num_groups = len(self.group_alphas)
                    if len(value) >= num_groups:
                        self.group_alphas.data = value[:num_groups]
        else:
            self._alphas = value

    def get_entropy_loss(self) -> torch.Tensor:
        """Get entropy loss for exploration"""
        if self.use_hierarchical:
            probs = self._hierarchical_probability_vector(temperature=1.0)
            total_entropy = -(probs * torch.log(probs + 1e-8)).sum()
            # Mild sparsity pressure inside groups to reduce vote dispersion.
            l1_term = sum(alpha.abs().mean() for alpha in self.op_alphas.values())
            return -0.01 * total_entropy + 1e-4 * l1_term

        probs = F.softmax(self._alphas / self.op_temperature, dim=0)
        total_entropy = -(probs * torch.log(probs + 1e-8)).sum()
        return -0.01 * total_entropy  # Encourage exploration

    def get_efficiency_penalty(self) -> torch.Tensor:
        """Get efficiency penalty for regularization"""
        return getattr(self, "_last_efficiency_penalty", torch.tensor(0.0))

    def set_temperature(self, temp: float):
        """Set temperature for Gumbel softmax"""
        self.temperature = max(temp, 0.1)  # Prevent too low temperature
        self.op_temperature = max(temp, self.min_op_temperature)
        self.group_temperature = max(
            temp * self.group_temperature_mult, self.min_group_temperature
        )

    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about operation usage and performance"""
        stats = {}

        if self.adaptive_sampling:
            for i, op_name in enumerate(self.available_ops):
                usage = self.usage_counter[i].item()
                performance = self.performance_tracker[i].item()
                stats[op_name] = {
                    "usage_count": usage,
                    "avg_performance": performance,
                    "efficiency": self.op_efficiency.get(op_name, 0.5),
                }

        return stats

    def describe(self, top_k: int = 3) -> Dict[str, float]:
        """Return top-k operations and their weights for inspection"""
        alphas = self.get_alphas()
        topk_vals, topk_idx = torch.topk(alphas, min(top_k, len(alphas)))
        return {
            self.available_ops[i]: round(w.item(), 4)
            for i, w in zip(topk_idx.tolist(), topk_vals)
        }

    def get_raw_alphas(self):
        """Get raw alpha parameters for debugging/analysis"""
        if self.use_hierarchical:
            return {
                "group_alphas": self.group_alphas.detach(),
                "op_alphas": {
                    name: alpha.detach() for name, alpha in self.op_alphas.items()
                },
            }
        else:
            return {"alphas": self._alphas.detach()}


class DARTSCell(nn.Module):
    """Enhanced DARTS cell with progressive search and better aggregation"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        num_nodes: int = 4,
        initial_search: bool = False,
        selected_ops: Optional[List[str]] = None,
        aggregation: str = "weighted",
        temperature: float = 1.0,
        use_checkpoint: bool = False,
        progressive_stage: str = "basic",  # "basic", "intermediate", "advanced"
        group_temperature_mult: float = 1.5,
        min_group_temperature: float = 0.7,
        min_op_temperature: float = 0.3,
        group_min_prob: float = 0.03,
        competition_power: float = 1.25,
        adaptive_bias_scale: float = 0.15,
        performance_ema_decay: float = 0.95,
        pc_ratio: float = 0.25,
        use_drnas: bool = True,
        drnas_concentration: float = 8.0,
        use_fair_darts_hierarchical: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_nodes = num_nodes
        self.initial_search = initial_search
        self.selected_ops = list(selected_ops) if selected_ops else None
        self.aggregation = aggregation
        self.temperature = temperature
        self.use_checkpoint = use_checkpoint
        self.progressive_stage = progressive_stage
        self.group_temperature_mult = group_temperature_mult
        self.min_group_temperature = min_group_temperature
        self.min_op_temperature = min_op_temperature
        self.group_min_prob = group_min_prob
        self.competition_power = competition_power
        self.adaptive_bias_scale = adaptive_bias_scale
        self.performance_ema_decay = performance_ema_decay
        self.pc_ratio = float(min(max(pc_ratio, 0.0), 1.0))
        self.use_drnas = use_drnas
        self.drnas_concentration = drnas_concentration
        self.use_fair_darts_hierarchical = use_fair_darts_hierarchical

        # Progressive operation selection
        self.stage_operations = {
            "basic": ["Identity", "DLinear", "ResidualMLP", "TimeConv"],
            "intermediate": [
                "Identity",
                "DLinear",
                "TimeConv",
                "TCN",
                "ConvMixer",
                "GRN",
                "TimeMixer",
            ],
            "advanced": [
                "Identity",
                "DLinear",
                "TimeConv",
                "TCN",
                "ConvMixer",
                "GRN",
                "Fourier",
                "Wavelet",
                "MultiScaleConv",
                "PyramidConv",
                "Mamba",
                "PatchEmbed",
                "iTransformerBlock",
                "TimeMixer",
            ],
        }

        self.available_ops = self._select_operations(selected_ops)
        self.num_edges = sum(range(num_nodes))

        self._init_components()
        self._edge_indices = self._precompute_edge_indices()

    def _select_operations(self, selected_ops):
        """Select operations based on search stage"""
        if self.initial_search:
            return ["Identity", "TimeConv"]

        if selected_ops:
            return selected_ops

        return self.stage_operations.get(
            self.progressive_stage, self.stage_operations["basic"]
        )

    def _ops_for_stage(self, stage: str) -> List[str]:
        stage_ops = list(
            self.stage_operations.get(stage, self.stage_operations["basic"])
        )
        if self.selected_ops is None:
            return stage_ops

        selected = list(self.selected_ops)
        selected_set = set(selected)
        filtered = [op for op in stage_ops if op in selected_set]
        if "Identity" in selected_set and "Identity" not in filtered:
            filtered.insert(0, "Identity")

        if len(filtered) < 2:
            for op in selected:
                if op not in filtered:
                    filtered.append(op)
                if len(filtered) >= 2:
                    break

        if not filtered:
            filtered = selected[:2] if len(selected) >= 2 else selected

        deduped = []
        seen = set()
        for op in filtered:
            if op not in seen:
                deduped.append(op)
                seen.add(op)
        return deduped

    def set_progressive_stage(self, stage: str) -> bool:
        """Set progressive stage and rebuild edge search spaces if needed."""
        target_ops = self._ops_for_stage(stage)
        needs_update = (
            stage != self.progressive_stage or target_ops != list(self.available_ops)
        )
        if not needs_update:
            return False

        self.progressive_stage = stage
        self.available_ops = target_ops

        for edge in self.edges:
            edge_param = next(edge.parameters(), None)
            edge_device = edge_param.device if edge_param is not None else None
            edge_dtype = edge_param.dtype if edge_param is not None else None

            edge.available_ops = list(target_ops)
            if hasattr(edge, "_init_hierarchical_search"):
                edge._init_hierarchical_search()
            elif hasattr(edge, "_init_flat_search"):
                edge._init_flat_search()
            if hasattr(edge, "_profile_flops"):
                edge._profile_flops()

            if edge_device is not None:
                edge.to(device=edge_device, dtype=edge_dtype)

            edge.fallback_idx = (
                edge.available_ops.index("Identity")
                if "Identity" in edge.available_ops
                else 0
            )

        return True

    def _precompute_edge_indices(self):
        """Precompute edge indices for faster lookup"""
        return {
            (i, j): sum(range(i)) + j
            for i in range(1, self.num_nodes)
            for j in range(i)
        }

    def _init_components(self):
        """Initialize all components"""
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim, bias=False),
            RMSNorm(self.latent_dim),
            nn.GELU(),
        )

        # Enhanced mixed operations
        self.edges = nn.ModuleList(
            [
                MixedOp(
                    self.latent_dim,
                    self.latent_dim,
                    self.seq_length,
                    available_ops=self.available_ops,
                    temperature=self.temperature,
                    use_hierarchical=True,
                    adaptive_sampling=True,
                    group_temperature_mult=self.group_temperature_mult,
                    min_group_temperature=self.min_group_temperature,
                    min_op_temperature=self.min_op_temperature,
                    group_min_prob=self.group_min_prob,
                    competition_power=self.competition_power,
                    adaptive_bias_scale=self.adaptive_bias_scale,
                    performance_ema_decay=self.performance_ema_decay,
                    pc_ratio=self.pc_ratio,
                    use_drnas=self.use_drnas,
                    drnas_concentration=self.drnas_concentration,
                    use_fair_darts_hierarchical=self.use_fair_darts_hierarchical,
                )
                for _ in range(self.num_edges)
            ]
        )

        # Learnable residual weights per node
        self.residual_weights = nn.Parameter(torch.full((self.num_nodes,), 0.2))

        # Edge importance weights
        self.edge_importance = nn.Parameter(torch.ones(self.num_edges) * 0.5)

        # Aggregation weights if using weighted aggregation
        if self.aggregation == "weighted":
            self.agg_weights = nn.Parameter(torch.ones(self.num_edges) * 0.1)
        else:
            self.agg_weights = None

        # Output normalization
        self.out_norm = RMSNorm(self.latent_dim)

        # Progressive search parameters
        self.stage_gates = nn.Parameter(
            torch.ones(3)
        )  # [basic, intermediate, advanced]

    def _get_edge_index(self, node_idx, input_idx):
        """Get edge index efficiently"""
        return self._edge_indices[(node_idx, input_idx)]

    def _aggregate_inputs(self, inputs, edge_indices):
        """Aggregate inputs with different strategies"""
        if len(inputs) == 1:
            return inputs[0]

        stacked = torch.stack(inputs, dim=0)

        if self.aggregation == "weighted" and self.agg_weights is not None:
            # Use edge-specific weights
            weights = F.softmax(
                torch.stack([self.agg_weights[i] for i in edge_indices]), dim=0
            )
            # Match stacked shape [num_inputs, B, L, D] for safe broadcasting.
            weights = weights.view(-1, 1, 1, 1)
            return (weights * stacked).sum(dim=0)
        elif self.aggregation == "attention":
            # Simple attention mechanism
            attention_scores = torch.mean(stacked, dim=[2, 3])  # [num_inputs, batch]
            attention_weights = F.softmax(attention_scores, dim=0)
            attention_weights = attention_weights.view(-1, 1, 1, 1)
            return (attention_weights * stacked).sum(dim=0)
        elif self.aggregation == "max":
            return torch.max(stacked, dim=0)[0]
        else:  # mean
            return torch.mean(stacked, dim=0)

    def _apply_residual(self, node_output, residual_input, node_idx):
        """Apply learnable residual connection with proper dimension handling"""
        residual_weight = torch.sigmoid(self.residual_weights[node_idx])

        # Handle dimension mismatches
        if node_output.shape != residual_input.shape:
            # Temporal alignment
            if node_output.shape[1] != residual_input.shape[1]:
                residual_input = F.interpolate(
                    residual_input.transpose(1, 2),
                    size=node_output.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            # Feature dimension alignment
            if node_output.shape[2] != residual_input.shape[2]:
                if not hasattr(self, f"_residual_proj_{node_idx}"):
                    proj = nn.Linear(residual_input.shape[2], node_output.shape[2]).to(
                        residual_input.device
                    )
                    setattr(self, f"_residual_proj_{node_idx}", proj)
                residual_input = getattr(self, f"_residual_proj_{node_idx}")(
                    residual_input
                )

        return residual_weight * node_output + (1 - residual_weight) * residual_input

    def forward(self, x):
        """Enhanced forward pass with progressive search"""
        if not isinstance(x, torch.Tensor):
            x = torch.stack(x) if isinstance(x, (list, tuple)) else x

        x_proj = self.input_proj(x)
        nodes = [x_proj]

        total_efficiency_penalty = 0

        for node_idx in range(1, self.num_nodes):
            node_inputs, edge_indices = [], []

            for input_idx in range(node_idx):
                edge_idx = self._get_edge_index(node_idx, input_idx)
                edge = self.edges[edge_idx]

                # Apply edge importance gating
                edge_weight = torch.sigmoid(self.edge_importance[edge_idx])

                if edge_weight.item() > 0.1:  # Skip unimportant edges
                    # Use gradient checkpointing if enabled
                    if self.training and self.use_checkpoint:
                        out = torch.utils.checkpoint.checkpoint(
                            edge, nodes[input_idx], use_reentrant=False
                        )
                    else:
                        out = edge(nodes[input_idx])

                    # Apply edge weight
                    out = out * edge_weight
                    node_inputs.append(out)
                    edge_indices.append(edge_idx)

                    # Accumulate efficiency penalty
                    # total_efficiency_penalty += edge.get_efficiency_penalty()

            if node_inputs:
                # Aggregate inputs
                agg = self._aggregate_inputs(node_inputs, edge_indices)
                # Apply residual connection
                out = self._apply_residual(agg, nodes[node_idx - 1], node_idx)
            else:
                # Fallback to previous node
                out = nodes[node_idx - 1]

            nodes.append(out)

        # Apply final residual and normalization
        final = self._apply_residual(nodes[-1], x_proj, 0)
        result = self.out_norm(final)

        # Store efficiency penalty for regularization
        self._last_efficiency_penalty = total_efficiency_penalty

        return result

    def advance_progressive_stage(self):
        """Advance to next progressive search stage"""
        stages = ["basic", "intermediate", "advanced"]
        current_idx = stages.index(self.progressive_stage) if self.progressive_stage in stages else 0
        if current_idx < len(stages) - 1:
            self.set_progressive_stage(stages[current_idx + 1])

    def get_alphas(self):
        """Get all edge alphas"""
        return [edge.get_alphas() for edge in self.edges]

    def get_entropy_loss(self):
        """Get total entropy loss for exploration"""
        total = sum(edge.get_entropy_loss() for edge in self.edges)

        # Add aggregation entropy if using weighted aggregation
        if self.agg_weights is not None:
            probs = F.softmax(self.agg_weights, dim=0)
            agg_entropy = -(probs * torch.log(probs + 1e-8)).sum()
            total -= 0.005 * agg_entropy

        return total

    def get_efficiency_penalty(self) -> torch.Tensor:
        """Get total efficiency penalty"""
        return getattr(self, "_last_efficiency_penalty", torch.tensor(0.0))

    def set_temperature(self, temp: float):
        """Update temperature for all edges"""
        self.temperature = temp
        for edge in self.edges:
            edge.set_temperature(temp)

    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about edge usage"""
        stats = {}
        for i, edge in enumerate(self.edges):
            edge_stats = edge.get_operation_statistics()
            edge_weight = torch.sigmoid(self.edge_importance[i]).item()
            stats[f"edge_{i}"] = {
                "importance_weight": edge_weight,
                "operations": edge_stats,
                "top_ops": edge.describe(top_k=2),
            }
        return stats


class TimeSeriesDARTS(nn.Module):
    """Simplified TimeSeriesDARTS with essential features"""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        forecast_horizon: int = 24,
        seq_length: int = 48,
        num_cells: int = 2,
        num_nodes: int = 4,
        dropout: float = 0.1,
        initial_search: bool = False,
        selected_ops: Optional[List] = None,
        loss_type: str = "huber",
        use_gradient_checkpointing: bool = False,
        temperature: float = 1.0,
        temperature_schedule: str = "cosine",
        final_temperature: float = 0.1,
        temperature_warmup_epochs: int = 0,
        use_attention_bridge: bool = True,
        attention_layers: int = 2,
        single_path_search: bool = True,
        use_learned_memory_pooling: bool = True,
        memory_num_queries: int = 8,
        group_temperature_mult: float = 1.5,
        min_group_temperature: float = 0.7,
        min_op_temperature: float = 0.3,
        group_min_prob: float = 0.03,
        competition_power: float = 1.25,
        adaptive_bias_scale: float = 0.15,
        performance_ema_decay: float = 0.95,
        pc_ratio: float = 0.25,
        use_drnas: bool = True,
        drnas_concentration: float = 8.0,
        use_fair_darts_hierarchical: bool = True,
    ):
        super().__init__()

        self._config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "forecast_horizon": forecast_horizon,
            "seq_length": seq_length,
            "num_cells": num_cells,
            "num_nodes": num_nodes,
            "selected_ops": selected_ops,
            "single_path_search": single_path_search,
            "temperature_schedule": temperature_schedule,
            "final_temperature": final_temperature,
            "temperature_warmup_epochs": temperature_warmup_epochs,
            "use_learned_memory_pooling": use_learned_memory_pooling,
            "memory_num_queries": memory_num_queries,
            "group_temperature_mult": group_temperature_mult,
            "min_group_temperature": min_group_temperature,
            "min_op_temperature": min_op_temperature,
            "group_min_prob": group_min_prob,
            "competition_power": competition_power,
            "adaptive_bias_scale": adaptive_bias_scale,
            "performance_ema_decay": performance_ema_decay,
            "pc_ratio": pc_ratio,
            "use_drnas": use_drnas,
            "drnas_concentration": drnas_concentration,
            "use_fair_darts_hierarchical": use_fair_darts_hierarchical,
        }

        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.forecast_horizon = forecast_horizon
        self.seq_length = seq_length
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.initial_search = initial_search
        self.selected_ops = selected_ops
        self.loss_type = loss_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.temperature = temperature
        self.initial_temperature = float(temperature)
        self.temperature_schedule = temperature_schedule
        self.final_temperature = float(final_temperature)
        self.temperature_warmup_epochs = int(temperature_warmup_epochs)
        self.use_attention_bridge = use_attention_bridge
        self.attention_layers = attention_layers
        self.single_path_search = single_path_search
        self.use_learned_memory_pooling = use_learned_memory_pooling
        self.memory_num_queries = memory_num_queries
        self.group_temperature_mult = group_temperature_mult
        self.min_group_temperature = min_group_temperature
        self.min_op_temperature = min_op_temperature
        self.group_min_prob = group_min_prob
        self.competition_power = competition_power
        self.adaptive_bias_scale = adaptive_bias_scale
        self.performance_ema_decay = performance_ema_decay
        self.pc_ratio = float(min(max(pc_ratio, 0.0), 1.0))
        self.use_drnas = use_drnas
        self.drnas_concentration = drnas_concentration
        self.use_fair_darts_hierarchical = use_fair_darts_hierarchical

        # Searchable normalization
        self.norm_strategy = SearchableNorm(self.input_dim)
        self.norm_alpha = nn.Parameter(torch.zeros(3))

        # Pruning state tracking
        self.pruning_history = []
        self.operation_performance = {}
        self.pruned_operations = set()
        self._init_components()

    def get_config(self):
        return copy.deepcopy(self._config)

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    @classmethod
    def with_search_profile(
        cls, profile: str = "conservative", **overrides
    ) -> "TimeSeriesDARTS":
        """Build model using a named MixedOp search profile."""
        cfg = DARTSConfig.with_search_profile(profile=profile)
        params = {
            "input_dim": cfg.input_dim,
            "hidden_dim": cfg.hidden_dim,
            "latent_dim": cfg.latent_dim,
            "forecast_horizon": cfg.forecast_horizon,
            "seq_length": cfg.seq_length,
            "num_cells": cfg.num_cells,
            "num_nodes": cfg.num_nodes,
            "dropout": cfg.dropout,
            "initial_search": cfg.initial_search,
            "selected_ops": cfg.selected_ops,
            "loss_type": cfg.loss_type,
            "use_gradient_checkpointing": cfg.use_gradient_checkpointing,
            "temperature": cfg.temperature,
            "single_path_search": cfg.single_path_search,
            "use_learned_memory_pooling": cfg.use_learned_memory_pooling,
            "memory_num_queries": cfg.memory_num_queries,
            "group_temperature_mult": cfg.group_temperature_mult,
            "min_group_temperature": cfg.min_group_temperature,
            "min_op_temperature": cfg.min_op_temperature,
            "group_min_prob": cfg.group_min_prob,
            "competition_power": cfg.competition_power,
            "adaptive_bias_scale": cfg.adaptive_bias_scale,
            "performance_ema_decay": cfg.performance_ema_decay,
            "pc_ratio": cfg.pc_ratio,
            "use_drnas": cfg.use_drnas,
            "drnas_concentration": cfg.drnas_concentration,
            "use_fair_darts_hierarchical": cfg.use_fair_darts_hierarchical,
        }
        params.update(overrides)
        return cls(**params)

    def _init_components(self):
        """Initialize all model components"""
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # DARTS cells with projections and scaling
        self.cells = nn.ModuleList()
        self.cell_proj = nn.ModuleList()
        self.layer_scales = nn.ParameterList()

        for i in range(self.num_cells):
            # Temperature decay for deeper cells
            temp = self.temperature * (0.8**i)

            # Cell
            self.cells.append(
                DARTSCell(
                    input_dim=self.hidden_dim,
                    latent_dim=self.latent_dim,
                    seq_length=self.seq_length,
                    num_nodes=self.num_nodes,
                    initial_search=self.initial_search,
                    selected_ops=self.selected_ops,
                    aggregation="weighted",
                    temperature=temp,
                    use_checkpoint=self.use_gradient_checkpointing,
                    group_temperature_mult=self.group_temperature_mult,
                    min_group_temperature=self.min_group_temperature,
                    min_op_temperature=self.min_op_temperature,
                    group_min_prob=self.group_min_prob,
                    competition_power=self.competition_power,
                    adaptive_bias_scale=self.adaptive_bias_scale,
                    performance_ema_decay=self.performance_ema_decay,
                    pc_ratio=self.pc_ratio,
                    use_drnas=self.use_drnas,
                    drnas_concentration=self.drnas_concentration,
                    use_fair_darts_hierarchical=self.use_fair_darts_hierarchical,
                )
            )

            # Projection layer
            self.cell_proj.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim, bias=False),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout * 0.5),
                )
            )

            # Layer scaling
            self.layer_scales.append(nn.Parameter(torch.ones(1) * 0.1))

        # Cell combination weights
        self.cell_weights = nn.Parameter(torch.ones(self.num_cells) * 0.5)
        self.cell_importance = nn.Parameter(torch.ones(self.num_cells) * 0.8)
        self.global_skip = nn.Parameter(torch.tensor(0.1))

        self.feature_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Sigmoid()
        )

        self.feature_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        # Encoder and decoder
        self.forecast_encoder = MixedEncoder(
            self.hidden_dim,
            self.latent_dim,
            seq_len=self.seq_length,
            dropout=self.dropout,
            temperature=self.temperature,
            single_path_search=self.single_path_search,
        )

        self.forecast_decoder = MixedDecoder(
            self.input_dim,
            self.latent_dim,
            seq_len=self.seq_length,
            dropout=self.dropout,
            temperature=self.temperature,
            use_attention_bridge=self.use_attention_bridge,
            attention_layers=self.attention_layers,
            use_learned_memory_pooling=self.use_learned_memory_pooling,
            memory_num_queries=self.memory_num_queries,
            single_path_search=self.single_path_search,
        )

        # Feature fusion
        self.gate_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
        )

        self.output_layer = nn.Linear(self.latent_dim, self.input_dim, bias=False)

        self.residual_weights = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(self.num_cells)]
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is float32 for consistent computation"""
        return tensor.float() if tensor.dtype != torch.float32 else tensor

    def forward(
        self,
        x_seq: torch.Tensor,
        x_future: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass"""
        B, L, _ = x_seq.shape

        # Ensure consistent dtype
        x_seq = self._ensure_dtype(x_seq)

        # Searchable normalization (RevIN/InstanceNorm/Identity)
        x_seq = self.norm_strategy(x_seq, self.norm_alpha)

        # Input embedding
        x_emb = self.input_embedding(x_seq)
        original_input = x_emb

        # Process through enhanced DARTS cells
        current_input = x_emb
        cell_outputs = []

        for i, (cell, proj, scale, res_weight, importance) in enumerate(
            zip(
                self.cells,
                self.cell_proj,
                self.layer_scales,
                self.residual_weights,
                self.cell_importance,
            )
        ):
            # Apply cell with optional checkpointing
            if self.training and self.use_gradient_checkpointing:
                cell_out = torch.utils.checkpoint.checkpoint(
                    cell, current_input, use_reentrant=False
                )
            else:
                cell_out = cell(current_input)

            # Project and scale
            projected = proj(cell_out) * scale * torch.sigmoid(importance)
            cell_outputs.append(projected)

            # Enhanced residual connection between cells
            if i > 0:
                residual_contrib = current_input * res_weight
                current_input = cell_out + residual_contrib
            else:
                current_input = cell_out

        # Enhanced cell feature combination
        if len(cell_outputs) > 1:
            # Learnable weighted combination
            cell_weights_norm = F.softmax(self.cell_weights[: len(cell_outputs)], dim=0)
            cell_importance_norm = torch.sigmoid(
                self.cell_importance[: len(cell_outputs)]
            )

            # Combine with both weights and importance
            final_weights = cell_weights_norm * cell_importance_norm
            final_weights = final_weights / final_weights.sum()

            combined_features = sum(w * f for w, f in zip(final_weights, cell_outputs))
        else:
            combined_features = cell_outputs[0]

        # Enhanced feature fusion with gating
        concatenated = torch.cat([combined_features, original_input], dim=-1)
        gate = self.feature_gate(concatenated)
        gated_features = gate * combined_features + (1 - gate) * original_input

        # Apply feature transformation
        final_features = self.feature_transform(gated_features)

        # Global skip connection
        final_features = final_features + self.global_skip * original_input

        # Encoding
        h_enc, context, encoder_state = self.forecast_encoder(final_features)

        # Decoding
        forecasts = []
        decoder_input = x_seq[:, -1:, :]
        decoder_hidden = encoder_state

        # Ensure consistent dtypes
        decoder_input = self._ensure_dtype(decoder_input)
        context = self._ensure_dtype(context)
        h_enc = self._ensure_dtype(h_enc)

        if isinstance(decoder_hidden, tuple):
            decoder_hidden = tuple(self._ensure_dtype(h) for h in decoder_hidden)
        else:
            decoder_hidden = self._ensure_dtype(decoder_hidden)

        for t in range(self.forecast_horizon):
            # Decoder step
            out, decoder_hidden = self.forecast_decoder(
                decoder_input, context, decoder_hidden, h_enc
            )

            # Post-processing
            prediction = self.output_layer(out)
            forecasts.append(prediction.squeeze(1))

            # Teacher forcing
            if (
                self.training
                and x_future is not None
                and t < x_future.size(1)
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                decoder_input = x_future[:, t : t + 1]
                decoder_input = self._ensure_dtype(decoder_input)
            else:
                decoder_input = prediction

        forecasts_tensor = torch.stack(forecasts, dim=1)
        forecasts_tensor = self.norm_strategy.apply_output_denorm(
            forecasts_tensor,
            self.norm_alpha,
            selected_norm=getattr(self, "selected_norm", None),
        )
        return forecasts_tensor

    # Analysis methods
    def get_all_alphas(self) -> Dict[str, torch.Tensor]:
        """Extract all architecture parameters"""
        alphas = {}

        # Normalization alphas
        if hasattr(self, "norm_alpha"):
            alphas["norm"] = F.softmax(self.norm_alpha, dim=0)

        # Cell alphas
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas"):
                        alphas[f"cell_{i}_edge_{j}"] = edge.get_alphas()

        # Encoder/decoder alphas
        alphas["encoder"] = self.forecast_encoder.get_alphas()
        alphas["decoder"] = self.forecast_decoder.get_alphas()

        # Attention alphas
        if (
            self.use_attention_bridge
            and hasattr(self.forecast_decoder, "attention_alphas")
            and self.forecast_decoder.attention_alphas is not None
        ):
            alphas["attention_bridge"] = F.softmax(
                self.forecast_decoder.attention_alphas, dim=0
            )

        return alphas

    def derive_discrete_architecture(self, threshold: float = 0.3) -> Dict[str, Any]:
        """Derive discrete architecture from continuous weights"""
        discrete_arch = {}
        weights = self.get_operation_weights()

        for component_name, component_weights in weights.items():
            if not component_weights:  # Skip empty weight dictionaries
                continue

            above_threshold = {
                op: w for op, w in component_weights.items() if float(w) >= float(threshold)
            }
            candidate_pool = above_threshold if above_threshold else component_weights
            max_op = max(candidate_pool, key=candidate_pool.get)
            max_weight = candidate_pool[max_op]

            if component_name.startswith("cell_"):
                parts = component_name.split("_")
                if len(parts) >= 2:
                    cell_name = f"cell_{parts[1]}"
                    if cell_name not in discrete_arch:
                        discrete_arch[cell_name] = {}
                    edge_name = "_".join(parts[2:]) if len(parts) > 2 else "edge"
                    discrete_arch[cell_name][edge_name] = {
                        "operation": max_op,
                        "weight": max_weight,
                        "passed_threshold": max_op in above_threshold,
                    }
            else:
                discrete_arch[component_name] = {
                    "type": max_op,
                    "weight": max_weight,
                    "passed_threshold": max_op in above_threshold,
                }

        return discrete_arch

    def get_operation_weights(self) -> Dict[str, Dict[str, float]]:
        """Get normalized operation weights"""
        weights = {}

        # Normalization weights
        if hasattr(self, "norm_alpha"):
            norm_names = ["revin", "instance_norm", "identity"]
            soft_norm = F.softmax(self.norm_alpha, dim=0)
            weights["norm"] = {
                name: weight.item()
                for name, weight in zip(norm_names, soft_norm[: len(norm_names)])
            }

        # Cell weights
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas") and hasattr(edge, "available_ops"):
                        try:
                            alphas = edge.get_alphas()
                            if alphas.numel() > 0:
                                weights[f"cell_{i}_edge_{j}"] = {
                                    op: weight.item()
                                    for op, weight in zip(
                                        edge.available_ops, F.softmax(alphas, dim=0)
                                    )
                                }
                        except Exception:
                            continue

        # Encoder/Decoder weights
        for component_name, component in [
            ("encoder", self.forecast_encoder),
            ("decoder", self.forecast_decoder),
        ]:
            if hasattr(component, "get_alphas"):
                try:
                    alphas = component.get_alphas()
                    if alphas.numel() > 0:
                        names = getattr(
                            component,
                            f"{component_name}_names",
                            [f"op_{i}" for i in range(len(alphas))],
                        )
                        alpha_slice = alphas[: len(names)]
                        finite_ok = torch.isfinite(alpha_slice).all().item()
                        looks_like_probs = (
                            finite_ok
                            and alpha_slice.min().item() >= -1e-6
                            and alpha_slice.max().item() <= 1.0 + 1e-6
                        )
                        if looks_like_probs:
                            soft_alphas = alpha_slice.clamp_min(1e-8)
                            soft_alphas = soft_alphas / soft_alphas.sum().clamp_min(
                                1e-8
                            )
                        else:
                            soft_alphas = F.softmax(alpha_slice, dim=0)
                        weights[component_name] = {
                            name: weight.item()
                            for name, weight in zip(names, soft_alphas)
                        }
                except Exception:
                    continue

        return weights

    def set_temperature(self, temp: float):
        """Update temperature for all components"""
        temp = max(float(temp), 1e-4)
        self.temperature = temp
        for cell in self.cells:
            if hasattr(cell, "set_temperature"):
                cell.set_temperature(temp)
        self.forecast_encoder.set_temperature(temp)
        self.forecast_decoder.set_temperature(temp)

    def _schedule_progressive_stage(self, epoch: int, total_epochs: int) -> str:
        """Automatically schedule progressive search stages across all cells."""
        if not self.cells:
            return "basic"

        progress = float(epoch) / float(max(int(total_epochs), 1))
        if progress < 0.25:
            target_stage = "basic"
        elif progress < 0.55:
            target_stage = "intermediate"
        else:
            target_stage = "advanced"

        for cell in self.cells:
            if hasattr(cell, "set_progressive_stage"):
                cell.set_progressive_stage(target_stage)
            else:
                cell.progressive_stage = target_stage
                new_ops = cell.stage_operations.get(target_stage, cell.available_ops)
                cell.available_ops = new_ops
                for edge in getattr(cell, "edges", []):
                    edge.available_ops = list(new_ops)
                    if hasattr(edge, "_init_hierarchical_search"):
                        edge._init_hierarchical_search()
                    if hasattr(edge, "_profile_flops"):
                        edge._profile_flops()

        return target_stage

    def schedule_progressive_stage(self, epoch: int, total_epochs: int) -> str:
        """Public wrapper for epoch-wise progressive-stage scheduling."""
        return self._schedule_progressive_stage(epoch=epoch, total_epochs=total_epochs)

    def schedule_temperature(
        self,
        epoch: int,
        total_epochs: int,
        *,
        schedule_type: Optional[str] = None,
        final_temp: Optional[float] = None,
        warmup_epochs: Optional[int] = None,
    ) -> float:
        """
        Compute and apply temperature using the model's built-in schedule.
        Returns the applied temperature.
        """
        schedule = schedule_type or self.temperature_schedule
        final_t = (
            max(float(final_temp), 1e-4)
            if final_temp is not None
            else max(self.final_temperature, 1e-4)
        )
        warmup = (
            max(int(warmup_epochs), 0)
            if warmup_epochs is not None
            else max(self.temperature_warmup_epochs, 0)
        )

        if total_epochs <= 0:
            self.set_temperature(self.temperature)
            return self.temperature

        if epoch < warmup:
            temp = self.initial_temperature
        else:
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            progress = min(max(progress, 0.0), 1.0)

            if schedule == "cosine":
                temp = final_t + (self.initial_temperature - final_t) * (
                    1.0 + math.cos(math.pi * progress)
                ) / 2.0
            elif schedule == "exponential":
                ratio = final_t / max(self.initial_temperature, 1e-8)
                temp = self.initial_temperature * math.exp(math.log(ratio) * progress)
            elif schedule == "linear":
                temp = self.initial_temperature + (final_t - self.initial_temperature) * progress
            elif schedule == "step":
                if progress < 0.3:
                    temp = self.initial_temperature
                elif progress < 0.7:
                    temp = 0.5 * self.initial_temperature
                else:
                    temp = final_t
            else:
                temp = self.temperature

        temp = max(temp, final_t)
        self.set_temperature(temp)
        return temp

    # PRUNING METHODS
    def prune_weak_operations(
        self, threshold: float = 0.1, strategy: str = "probability"
    ) -> Dict[str, Any]:
        """
        Prune weak operations based on their weights/importance

        Args:
            threshold: Minimum weight to keep an operation
            strategy: "probability" | "gradient" | "entropy" | "performance"

        Returns:
            Dict with pruning statistics
        """
        pruning_stats = {
            "operations_pruned": 0,
            "operations_kept": 0,
            "pruned_details": {},
            "threshold_used": threshold,
            "strategy": strategy,
        }

        if strategy == "probability":
            pruning_stats.update(self._prune_by_probability(threshold))
        elif strategy == "gradient":
            pruning_stats.update(self._prune_by_gradient_magnitude(threshold))
        elif strategy == "entropy":
            pruning_stats.update(self._prune_by_entropy(threshold))
        elif strategy == "performance":
            pruning_stats.update(self._prune_by_performance(threshold))
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

        # Store pruning history
        self.pruning_history.append(pruning_stats)
        print(
            f"Pruning completed with strategy '{strategy}' and threshold {threshold}."
        )
        return pruning_stats

    def _prune_by_probability(self, threshold: float) -> Dict[str, Any]:
        """Prune operations based on their probability weights"""
        stats = {"operations_pruned": 0, "operations_kept": 0, "pruned_details": {}}

        # Prune cell operations
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas") and hasattr(edge, "available_ops"):
                        alphas = edge.get_alphas()
                        probs = F.softmax(alphas, dim=0)

                        # Find operations below threshold
                        weak_ops = []
                        for k, (op_name, prob) in enumerate(
                            zip(edge.available_ops, probs)
                        ):
                            if prob.item() < threshold and op_name != "Identity":
                                weak_ops.append((k, op_name, prob.item()))

                        # Prune weak operations by setting very low alpha values
                        if weak_ops:
                            for k, op_name, prob in weak_ops:
                                if self._set_edge_op_logit(edge, k, value=-10.0):
                                    # Track pruned operation
                                    op_id = f"cell_{i}_edge_{j}_{op_name}"
                                    self.pruned_operations.add(op_id)
                                    stats["pruned_details"][op_id] = prob
                                    stats["operations_pruned"] += 1

                        # Count kept operations
                        kept_ops = len(edge.available_ops) - len(weak_ops)
                        stats["operations_kept"] += kept_ops

        return stats

    def _set_edge_op_logit(self, edge: nn.Module, op_idx: int, value: float) -> bool:
        """Set a single operation logit for both flat and hierarchical MixedOp edges."""
        try:
            with torch.no_grad():
                if (
                    getattr(edge, "use_hierarchical", False)
                    and hasattr(edge, "op_to_group")
                    and hasattr(edge, "group_names")
                    and hasattr(edge, "op_alphas")
                    and op_idx in edge.op_to_group
                ):
                    group_idx, local_idx = edge.op_to_group[op_idx]
                    group_name = edge.group_names[group_idx]
                    edge.op_alphas[group_name][local_idx] = value
                    return True

                if hasattr(edge, "_alphas") and edge._alphas is not None:
                    edge._alphas[op_idx] = value
                    return True
        except Exception:
            return False

        return False

    def freeze_pruned_operations(
        self,
        pruning_stats: Optional[Dict[str, Any]] = None,
        logit_value: float = -20.0,
    ) -> int:
        """
        Hard-freeze previously pruned operations by forcing their logits to a low value.
        Returns number of logits successfully updated.
        """
        if pruning_stats and "pruned_details" in pruning_stats:
            pruned_ids = list(pruning_stats.get("pruned_details", {}).keys())
        else:
            pruned_ids = list(self.pruned_operations)

        frozen = 0
        for op_id in pruned_ids:
            match = re.match(r"^cell_(\d+)_edge_(\d+)_(.+)$", str(op_id))
            if match is None:
                continue

            cell_idx = int(match.group(1))
            edge_idx = int(match.group(2))
            op_name = match.group(3)

            if cell_idx < 0 or cell_idx >= len(self.cells):
                continue
            cell = self.cells[cell_idx]
            if not hasattr(cell, "edges") or edge_idx < 0 or edge_idx >= len(cell.edges):
                continue

            edge = cell.edges[edge_idx]
            if not hasattr(edge, "available_ops") or op_name not in edge.available_ops:
                continue

            op_idx = edge.available_ops.index(op_name)
            if self._set_edge_op_logit(edge, op_idx, value=float(logit_value)):
                frozen += 1

        return frozen

    def _prune_by_gradient_magnitude(self, threshold: float) -> Dict[str, Any]:
        """Prune operations with consistently low architecture-gradient signal."""
        stats = {
            "operations_pruned": 0,
            "operations_kept": 0,
            "pruned_details": {},
            "gradient_fallback_edges": 0,
        }
        fallback_edges = []

        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if (
                        not hasattr(edge, "available_ops")
                        or len(edge.available_ops) == 0
                    ):
                        continue

                    probs = edge.get_alphas().detach().float()
                    probs = probs / probs.sum().clamp_min(1e-8)

                    grad_scores = torch.zeros_like(probs)

                    if getattr(edge, "use_hierarchical", False) and hasattr(
                        edge, "op_to_group"
                    ):
                        for op_idx in range(len(edge.available_ops)):
                            group_idx, local_idx = edge.op_to_group[op_idx]
                            group_name = edge.group_names[group_idx]
                            op_alpha = edge.op_alphas[group_name]
                            if op_alpha.grad is not None:
                                grad_scores[op_idx] = op_alpha.grad[local_idx].abs()
                    elif hasattr(edge, "_alphas") and edge._alphas.grad is not None:
                        grad_scores = edge._alphas.grad.detach().abs().float()

                    max_grad = (
                        grad_scores.max().item() if grad_scores.numel() > 0 else 0.0
                    )
                    if max_grad > 1e-12:
                        grad_scores = grad_scores / max_grad
                    else:
                        # Explicit fallback when gradients are unavailable.
                        grad_scores = probs.clone()
                        stats["gradient_fallback_edges"] += 1
                        fallback_edges.append(f"cell_{i}_edge_{j}")

                    weak_ops = []
                    top_idx = (
                        int(torch.argmax(probs).item()) if probs.numel() > 0 else -1
                    )
                    for k, op_name in enumerate(edge.available_ops):
                        score = float(grad_scores[k].item())
                        if score < threshold and op_name != "Identity" and k != top_idx:
                            weak_ops.append((k, op_name, score))

                    for k, op_name, score in weak_ops:
                        if self._set_edge_op_logit(edge, k, value=-10.0):
                            op_id = f"cell_{i}_edge_{j}_{op_name}"
                            self.pruned_operations.add(op_id)
                            stats["pruned_details"][op_id] = score
                            stats["operations_pruned"] += 1

                    stats["operations_kept"] += len(edge.available_ops) - len(weak_ops)

        if fallback_edges:
            preview = ", ".join(fallback_edges[:5])
            if len(fallback_edges) > 5:
                preview += ", ..."
            warnings.warn(
                "Gradient-based pruning fell back to probability scores for "
                f"{len(fallback_edges)} edges (no gradients available). "
                f"Examples: {preview}",
                RuntimeWarning,
                stacklevel=2,
            )

        return stats

    def _prune_by_entropy(self, threshold: float) -> Dict[str, Any]:
        """Prune low-probability ops more aggressively when an edge distribution is confident."""
        stats = {"operations_pruned": 0, "operations_kept": 0, "pruned_details": {}}

        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if (
                        not hasattr(edge, "available_ops")
                        or len(edge.available_ops) <= 1
                    ):
                        continue

                    probs = edge.get_alphas().detach().float()
                    probs = probs / probs.sum().clamp_min(1e-8)

                    entropy = float((-(probs * torch.log(probs + 1e-8))).sum().item())
                    max_entropy = float(
                        torch.log(torch.tensor(float(len(probs)))).item()
                    )
                    confidence = 1.0 - (entropy / max(max_entropy, 1e-8))

                    # Increase pruning pressure only when the edge is already confident.
                    adaptive_threshold = threshold * (0.5 + confidence)
                    top_idx = int(torch.argmax(probs).item())

                    weak_ops = []
                    for k, (op_name, prob) in enumerate(zip(edge.available_ops, probs)):
                        p = float(prob.item())
                        if (
                            p < adaptive_threshold
                            and op_name != "Identity"
                            and k != top_idx
                        ):
                            weak_ops.append((k, op_name, p))

                    for k, op_name, p in weak_ops:
                        if self._set_edge_op_logit(edge, k, value=-10.0):
                            op_id = f"cell_{i}_edge_{j}_{op_name}"
                            self.pruned_operations.add(op_id)
                            stats["pruned_details"][op_id] = p
                            stats["operations_pruned"] += 1

                    stats["operations_kept"] += len(edge.available_ops) - len(weak_ops)

        return stats

    def _prune_by_performance(self, threshold: float) -> Dict[str, Any]:
        """Prune operations using combined probability and tracked runtime performance."""
        stats = {"operations_pruned": 0, "operations_kept": 0, "pruned_details": {}}

        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if (
                        not hasattr(edge, "available_ops")
                        or len(edge.available_ops) == 0
                    ):
                        continue

                    probs = edge.get_alphas().detach().float()
                    probs = probs / probs.sum().clamp_min(1e-8)

                    if hasattr(edge, "performance_tracker"):
                        perf = edge.performance_tracker.detach().float()
                        if perf.numel() == probs.numel():
                            p_min, p_max = perf.min(), perf.max()
                            if (p_max - p_min).abs().item() > 1e-8:
                                perf_norm = (perf - p_min) / (p_max - p_min)
                            else:
                                perf_norm = torch.sigmoid(perf)
                        else:
                            perf_norm = probs.clone()
                    else:
                        perf_norm = probs.clone()

                    usage = (
                        edge.usage_counter.detach().float()
                        if hasattr(edge, "usage_counter")
                        else torch.ones_like(probs)
                    )

                    combined = 0.7 * probs + 0.3 * perf_norm
                    top_idx = int(torch.argmax(combined).item())

                    weak_ops = []
                    for k, op_name in enumerate(edge.available_ops):
                        score = float(combined[k].item())
                        op_id = f"cell_{i}_edge_{j}_{op_name}"
                        self.operation_performance[op_id] = {
                            "probability": float(probs[k].item()),
                            "performance": float(perf_norm[k].item()),
                            "usage": float(usage[k].item()),
                            "combined": score,
                        }

                        # Avoid pruning very under-observed operations too early.
                        mature_enough = usage[k].item() >= 3
                        if (
                            score < threshold
                            and mature_enough
                            and op_name != "Identity"
                            and k != top_idx
                        ):
                            weak_ops.append((k, op_name, score))

                    for k, op_name, score in weak_ops:
                        if self._set_edge_op_logit(edge, k, value=-10.0):
                            op_id = f"cell_{i}_edge_{j}_{op_name}"
                            self.pruned_operations.add(op_id)
                            stats["pruned_details"][op_id] = score
                            stats["operations_pruned"] += 1

                    stats["operations_kept"] += len(edge.available_ops) - len(weak_ops)

        return stats


__all__ = [
    "DARTSConfig",
    "MixedOp",
    "DARTSCell",
    "TimeSeriesDARTS",
]
