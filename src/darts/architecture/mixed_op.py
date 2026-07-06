"""
MixedOp - Enhanced mixed operation for DARTS architecture search.

This module contains the MixedOp class which implements the searchable
operation with GDAS, DrNAS, and hierarchical search support.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.search.metrics import _default_enable_flops

from .advanced_ops import GatedGeLUFFNOp, GeGLUFFNOp, SwiGLUFFNOp
from .operation_blocks import (
    ConvMixerOp,
    DLinearOp,
    FourierOp,
    GRNOp,
    IdentityOp,
    InvertedAttentionOp,
    MLPMixerOp,
    MultiScaleConvOp,
    NBeatsOp,
    PatchEmbedOp,
    PyramidConvOp,
    ResidualMLPOp,
    RMSNorm,
    TCNOp,
    TimeConvOp,
    TimesNetOp,
    WaveletOp,
)


__all__ = ["MixedOp"]

class MixedOp(nn.Module):
    """Enhanced MixedOp using your existing operators with better search strategy"""

    _efficiency_cache: dict[Any, dict[str, float]] = {}
    _flops_warning_emitted: bool = False

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        available_ops: list[str] | None = None,
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
        op_gdas: bool = True,
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
        self.op_gdas = bool(op_gdas)
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
            "PatchEmbed": lambda: PatchEmbedOp(input_dim, latent_dim, patch_size=16),
            "InvertedAttention": lambda: InvertedAttentionOp(input_dim, latent_dim),
            "TimeMixer": lambda: MLPMixerOp(input_dim, latent_dim, seq_length),
            "DLinear": lambda: DLinearOp(input_dim, latent_dim),
            "NBeats": lambda: NBeatsOp(input_dim, latent_dim),
            "TimesNet": lambda: TimesNetOp(input_dim, latent_dim),
            "SwiGLU": lambda: SwiGLUFFNOp(input_dim, latent_dim),
            "GeGLU": lambda: GeGLUFFNOp(input_dim, latent_dim),
            "GatedGELU": lambda: GatedGeLUFFNOp(input_dim, latent_dim),
        }

        # Group operations by complexity/type for hierarchical search
        self.operation_groups = {
            "mlp": [
                "Identity",
                "ResidualMLP",
                "GRN",
                "TimeMixer",
                "NBeats",
            ],
            "conv": [
                "TimeConv",
                "TCN",
                "ConvMixer",
                "MultiScaleConv",
                "PyramidConv",
            ],
            "frequency": ["Fourier", "Wavelet", "DLinear", "TimesNet"],
            "attention": [
                "PatchEmbed",
                "InvertedAttention",
            ],
            "gated_ffn": [
                "SwiGLU",
                "GeGLU",
                "GatedGELU",
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
            "PatchEmbed": 0.7,
            "InvertedAttention": 0.55,
            "iTransformerBlock": 0.55,
            "TimeMixer": 0.68,
            "NBeats": 0.72,
            "TimesNet": 0.60,
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

        # Output projection for dimension mismatch — pre-allocated so it is
        # registered as a proper parameter group and trained by the optimizer.
        self.output_proj: nn.Module = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        )
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
            self.op_efficiency.update(
                copy.deepcopy(MixedOp._efficiency_cache[cache_key])
            )
            self._dynamic_efficiency_profiled = True
            self._flops_profiled = True
            return

        if not _default_enable_flops():
            self._apply_static_efficiency_priors()
            return

        try:
            from fvcore.nn import FlopCountAnalysis
        except Exception:
            self._apply_static_efficiency_priors()
            return

        ref_param = next(self.parameters(), None)
        device = ref_param.device if ref_param is not None else torch.device("cpu")
        dtype = ref_param.dtype if ref_param is not None else torch.float32
        dummy = torch.randn(
            2, self.seq_length, self.latent_dim, device=device, dtype=dtype
        )

        inverse_scores: dict[str, float] = {}
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

    def _apply_static_efficiency_priors(self):
        # Pre-computed from typical time-series shapes; higher means cheaper.
        static_priors = {
            "Identity": 1.0,
            "DLinear": 0.95,
            "ResidualMLP": 0.80,
            "NBeats": 0.78,
            "PatchEmbed": 0.68,
            "TimeMixer": 0.68,
            "GRN": 0.65,
            "TimesNet": 0.62,
            "TimeConv": 0.60,
            "ConvMixer": 0.58,
            "InvertedAttention": 0.55,
            "iTransformerBlock": 0.55,
            "TCN": 0.50,
            "Fourier": 0.45,
            "Wavelet": 0.45,
            "MultiScaleConv": 0.35,
            "PyramidConv": 0.25,
        }
        for op_name in self.available_ops:
            self.op_efficiency[op_name] = static_priors.get(op_name, 0.5)
        self._dynamic_efficiency_profiled = False
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

    def _get_weights(self, top_k: int | None = None):
        """Get operation weights with optional top-k selection"""
        if self.use_hierarchical:
            return self._get_hierarchical_weights(top_k)
        else:
            return self._get_flat_weights(top_k)

    def _get_hierarchical_weights(self, top_k: int | None = None):
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
                op_scores = self._sample_op_weights(
                    op_logits, independent_sigmoid=False
                )
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

    def _get_flat_weights(self, top_k: int | None = None):
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
        """Project to latent_dim if needed."""
        if x.shape[-1] != self.latent_dim:
            x = self.output_proj(x)
        return x

    def _gdas_forward(self, x: torch.Tensor) -> torch.Tensor:
        """GDAS single-path forward with straight-through gradient estimator.

        Samples exactly one operation per forward pass using Gumbel-Softmax.
        The selected op runs in full; gradients flow back through the soft
        probabilities (straight-through), so alpha logits are still updated.

        Gradient scaling: the straight-through factor is normalised by the
        number of candidates (``len(logits)``) so that the total gradient
        magnitude remains stable regardless of search-space size.
        """
        if self.use_hierarchical and hasattr(self, "group_names"):
            # Flatten to a joint logit vector: group_logit[g] + op_logit[g][k].
            # This approximates sampling from the joint product distribution.
            all_logits: list[torch.Tensor] = []
            all_global_indices: list[int] = []
            for g_idx, group_name in enumerate(self.group_names):
                g_logit = self.group_alphas[g_idx]
                op_logits = self.op_alphas[group_name]
                for k, global_idx in enumerate(self.group_op_indices[group_name]):
                    all_logits.append(g_logit + op_logits[k])
                    all_global_indices.append(global_idx)
            logits = torch.stack(all_logits)
        else:
            logits = self._alphas
            all_global_indices = list(range(len(self.available_ops)))

        tau = max(self.op_temperature, 1e-3)
        # Sample Gumbel noise and compute soft probabilities.
        gumbels = -torch.log(
            -torch.log(torch.rand_like(logits).clamp_(1e-10, 1 - 1e-10))
        )
        soft = F.softmax((logits + gumbels) / tau, dim=0)

        # Hard argmax index — no gradient through this selection.
        with torch.no_grad():
            sampled_pos = int(soft.argmax().item())

        op_global_idx = all_global_indices[sampled_pos]
        out = self.ops[op_global_idx](x)
        out = self._ensure_output_dim(out)

        # Update performance tracker for the sampled op only.
        # Uses a "quality score" based on output energy (mean-square norm),
        # which is a stable proxy for how much useful signal the op
        # contributes — not a gradient norm (gradients are unknown at this
        # point in the forward pass).
        if self.adaptive_sampling and self.training:
            with torch.no_grad():
                finite_ok = torch.isfinite(out).all()
                if finite_ok:
                    out_det = out.detach()
                    # Mean-square output energy (lower = more stable signal).
                    mean_sq = (out_det.square().mean().item())
                    score = 1.0 / (mean_sq + 1e-6)
                    self.performance_tracker[op_global_idx].mul_(
                        self.performance_ema_decay
                    ).add_((1 - self.performance_ema_decay) * score)
                else:
                    self.performance_tracker[op_global_idx].mul_(
                        self.performance_ema_decay
                    ).add_(-(1 - self.performance_ema_decay))
                self.usage_counter[op_global_idx] += 1

        # Straight-through scale: 1.0 in the forward pass, but the expression
        # carries a gradient w.r.t. soft[sampled_pos] (and thus the logits)
        # in the backward pass.  Divide by candidate count so total gradient
        # magnitude stays stable regardless of search-space size.
        n_candidates = max(len(logits), 1)
        scale = soft[sampled_pos] / (soft[sampled_pos].detach() * n_candidates)
        return out * scale

    def forward(self, x: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
        """Enhanced forward with better operation selection"""
        if self.op_gdas and self.training:
            return self._gdas_forward(x)

        op_weights = self._get_weights(top_k)

        if not op_weights:
            # Fallback to identity or first operation
            fallback_out = self.ops[self.fallback_idx](x)
            return self._ensure_output_dim(fallback_out)

        outputs = []
        total_weight: torch.Tensor | None = None

        # Precompute DropPath keep/drop decisions on the host in one shot so the
        # per-op loop below never issues a GPU→CPU ``.item()`` sync (each sync
        # stalls the CUDA pipeline). Drawing on CPU is equivalent in
        # distribution to the previous per-op ``torch.rand(...).item()`` call.
        if self.training and self.drop_prob > 0.0:
            drop_mask = (torch.rand(len(op_weights)) < self.drop_prob).tolist()
        else:
            drop_mask = [False] * len(op_weights)

        for path_idx, (op_idx, weight) in enumerate(op_weights):
            if drop_mask[path_idx]:
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
                    op_input = x * channel_mask.reshape(1, 1, channels)
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
            # Accumulate the normaliser as a tensor; dividing once at the end
            # avoids a per-op ``.item()`` sync (see DropPath note above).
            total_weight = weight if total_weight is None else total_weight + weight

            # Update performance tracker if adaptive sampling.
            # Uses mean-square output energy as a "quality score" — lower
            # energy means the op contributes stable, non-exploding signal.
            if self.adaptive_sampling and self.training:
                with torch.no_grad():
                    finite_ok = torch.isfinite(out).all()
                    if finite_ok:
                        out_det = out.detach()
                        mean_sq = float(out_det.square().mean().item())
                        score = 1.0 / (mean_sq + 1e-6)
                        self.performance_tracker[op_idx].mul_(
                            self.performance_ema_decay
                        ).add_((1 - self.performance_ema_decay) * score)
                    else:
                        self.performance_tracker[op_idx].mul_(
                            self.performance_ema_decay
                        ).add_(-(1 - self.performance_ema_decay))

                    self.usage_counter[op_idx] += 1

        if outputs:
            result = sum(outputs) / total_weight.clamp_min(1e-6)
            return result

        # Final fallback
        fallback_out = self.ops[self.fallback_idx](x)
        return self._ensure_output_dim(fallback_out)

    def _hierarchical_probability_vector(
        self, temperature: float = 1.0
    ) -> torch.Tensor:
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

    def set_temperature(self, temp: float):
        """Set temperature for Gumbel softmax"""
        self.temperature = max(temp, 0.1)  # Prevent too low temperature
        self.op_temperature = max(temp, self.min_op_temperature)
        self.group_temperature = max(
            temp * self.group_temperature_mult, self.min_group_temperature
        )

    def set_drnas_concentration(self, concentration: float) -> None:
        """Update the DrNAS Dirichlet concentration parameter."""
        self.drnas_concentration = max(float(concentration), 1e-3)

    def get_operation_statistics(self) -> dict[str, Any]:
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

    def describe(self, top_k: int = 3) -> dict[str, float]:
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
