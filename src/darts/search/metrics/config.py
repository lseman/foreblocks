import re
from dataclasses import dataclass, field

import torch


def _torch_version_tuple(version: str | None = None) -> tuple[int, int]:
    """Return the major/minor PyTorch version, ignoring local build suffixes."""
    raw = torch.__version__ if version is None else str(version)
    match = re.match(r"^\s*(\d+)\.(\d+)", raw)
    if match is None:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def _default_enable_flops() -> bool:
    """Disable FLOP proxy scoring on older PyTorch tracer stacks."""
    return _torch_version_tuple() >= (2, 12)


@dataclass
class Config:
    """Unified configuration"""

    max_samples: int = 32
    max_outputs: int = 10
    eps: float = 1e-8
    # NASWOT builds an [R, R] kernel from each layer's activation. For
    # transformer/MoE layers R = batch*seq (can be thousands); cap it so slogdet
    # stays cheap and bounded. Layers with R > 2*features are skipped as
    # rank-deficient (see metrics/naswot.py).
    naswot_max_rows: int = 256
    # Per-metric wall-clock timeout enforced via a daemon thread in
    # MetricsComputer._compute_safely. If a metric computation (e.g. GRASP
    # second-order backward) stalls in a CUDA kernel, the thread is abandoned
    # and the metric returns a failed Result so evaluation can proceed.
    timeout: float = 30.0
    enable_mixed_precision: bool = False
    # One probe is standard for proxy NAS and sufficient for candidate ranking;
    # more probes reduce Hutchinson estimator variance at the cost of extra backward passes.
    jacobian_probes: int = 1
    # SNIP is defined at initialization; keep this as the default behavior.
    snip_at_init: bool = True
    # Explicit mode: "current" is faster (no weight reset, no extra forward/backward
    # at init) and avoids hangs from reset_parameters on custom layers.
    # Use "init" only if paper-consistent SNIP scores are needed.
    snip_mode: str = "current"
    heavy_metrics_batches: int = 1
    gradient_max_samples: int = 4
    # False: uses the already-computed batch gradient squared (free, no extra
    # forward/backward passes). True: runs one forward+backward per sample
    # (more accurate diagonal Fisher, but O(N) passes extra).
    fisher_per_sample: bool = False
    enable_grasp: bool = True
    enable_jacobian: bool = True
    enable_synflow: bool = True
    enable_flops: bool = field(default_factory=_default_enable_flops)
    conditioning_every_n_layers: int = 3
    conditioning_min_out_features: int = 0
    conditioning_power_iters: int = 6
    conditioning_exact_max_dim: int = 64
    conditioning_inverse_shift: float = 1e-6
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "synflow": 0.25,
            "grasp": 0.20,
            "fisher": 0.20,
            "jacobian": 0.15,
            "naswot": 0.15,
            "snip": 0.15,
            "params": -0.05,
            "conditioning": -0.10,
            "flops": -0.05,
            "sensitivity": 0.10,
            "activation_diversity": 0.10,
        }
    )




# ─── Lazy presets ───────────────────────────────────────────────────────
# Built once on first access to avoid recursion during class definition.

def _get_presets() -> dict[str, "Config"]:
    """Return the named preset dict (built lazily on first call)."""
    return {
        "full": Config(),
        "smart_fast": Config(
            max_samples=32,
            max_outputs=10,
            jacobian_probes=1,
            gradient_max_samples=4,
            fisher_per_sample=False,
            enable_grasp=False,
            enable_jacobian=False,
            enable_synflow=True,
            snip_mode="current",
            conditioning_every_n_layers=3,
            heavy_metrics_batches=1,
            weights={
                "synflow": 0.20,
                "grasp": 0.0,
                "fisher": 0.18,
                "jacobian": 0.0,
                "naswot": 0.15,
                "snip": 0.18,
                "params": -0.05,
                "flops": -0.05,
                "conditioning": -0.05,
                "sensitivity": 0.12,
                "activation_diversity": 0.07,
            },
        ),
        "ultra_fast": Config(
            max_samples=16,
            max_outputs=5,
            jacobian_probes=1,
            gradient_max_samples=2,
            fisher_per_sample=False,
            enable_grasp=False,
            enable_jacobian=False,
            enable_synflow=True,
            snip_mode="current",
            conditioning_every_n_layers=10,
            heavy_metrics_batches=1,
            weights={
                "synflow": 0.30,
                "grasp": 0.0,
                "fisher": 0.0,
                "jacobian": 0.0,
                "naswot": 0.0,
                "snip": 0.25,
                "params": -0.10,
                "flops": 0.0,
                "conditioning": 0.0,
                "sensitivity": 0.0,
                "activation_diversity": 0.0,
            },
        ),
    }


@dataclass
class Result:
    """Metric computation result"""

    value: float
    success: bool = True
    error: str = ""
    time: float = 0.0

    def __repr__(self):
        status = "✓" if self.success else "✗"
        return f"Result({status} {self.value:.4f}, {self.time:.3f}s)"
