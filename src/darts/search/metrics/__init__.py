"""Zero-cost NAS metrics: re-exports for backward compatibility.

Implementation lives in :mod:`config`, :mod:`compatibility`,
:mod:`computer`, and :mod:`zero_cost_nas`; the individual proxy metrics
(SynFlow, Fisher, NASWOT, ...) live in their own sibling modules.
"""

from .activation_diversity import compute_activation_diversity
from .compatibility import CompatibilityHelper
from .computer import _ZC_GPU_LOCK as _ZC_GPU_LOCK, MetricsComputer
from .conditioning import compute_conditioning
from .config import (
    Config,
    Result,
    _default_enable_flops as _default_enable_flops,
    _get_presets as _get_presets,
    _torch_version_tuple as _torch_version_tuple,
)
from .fisher import compute_fisher
from .flops import compute_activation_flops
from .grasp import compute_grasp
from .jacobian import compute_jacobian
from .naswot import compute_naswot
from .params import compute_params
from .sensitivity import compute_sensitivity
from .snip import compute_snip
from .synflow import compute_synflow
from .zero_cost_nas import ZeroCostNAS

__all__ = [
    "CompatibilityHelper",
    "Config",
    "MetricsComputer",
    "Result",
    "ZeroCostNAS",
    "compute_activation_diversity",
    "compute_activation_flops",
    "compute_conditioning",
    "compute_fisher",
    "compute_grasp",
    "compute_jacobian",
    "compute_naswot",
    "compute_params",
    "compute_sensitivity",
    "compute_snip",
    "compute_synflow",
]
