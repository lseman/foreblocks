"""foreblocks.features.

Reusable feature transforms for time-series classification, anomaly
detection and regression. All follow the scikit-learn transformer API, so
they compose with `sklearn.pipeline.make_pipeline`.

Core API:
- Rocket, MiniRocket, MultiRocket: published Rocket-family transforms
  (univariate `[N, L]` or multivariate `[N, C, L]`)
- FusedRocket: ROCKET kernels + PPV/MPV pooling on fused Numba/Triton kernels
- Hydra (+ SparseScaler): competing convolutional kernels
- SelfRocket: MiniRocket with a learned representation / pooling operator
- RocketClassifier: Rocket-family transform(s) + scaling + RidgeClassifierCV,
  incl. "multirocket-hydra" and POCKET / S-ROCKET kernel pruning
- pocket_select, srocket_select: kernel pruning selectors
- SignalFeatures, extract_signal_features: engineered amplitude, spectral,
  band-power, wavelet and pulse features
- normalize_series: per-series z-score

Requires the `scientific` extra (numba, scipy, scikit-learn) and, for
`SignalFeatures`, PyWavelets.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.features._validation import normalize_series as normalize_series
    from foreblocks.features.rocket import (
        FusedRocket as FusedRocket,
        Hydra as Hydra,
        MiniRocket as MiniRocket,
        MultiRocket as MultiRocket,
        Rocket as Rocket,
        RocketClassifier as RocketClassifier,
        SelfRocket as SelfRocket,
        SparseScaler as SparseScaler,
        make_rocket_transform as make_rocket_transform,
        pocket_select as pocket_select,
        srocket_select as srocket_select,
    )
    from foreblocks.features.signal import (
        SignalFeatures as SignalFeatures,
        extract_signal_features as extract_signal_features,
    )

_EXPORTS = {
    "Rocket": "foreblocks.features.rocket",
    "MiniRocket": "foreblocks.features.rocket",
    "MultiRocket": "foreblocks.features.rocket",
    "FusedRocket": "foreblocks.features.rocket",
    "RocketClassifier": "foreblocks.features.rocket",
    "Hydra": "foreblocks.features.rocket",
    "SparseScaler": "foreblocks.features.rocket",
    "SelfRocket": "foreblocks.features.rocket",
    "pocket_select": "foreblocks.features.rocket",
    "srocket_select": "foreblocks.features.rocket",
    "make_rocket_transform": "foreblocks.features.rocket",
    "SignalFeatures": "foreblocks.features.signal",
    "extract_signal_features": "foreblocks.features.signal",
    "normalize_series": "foreblocks.features._validation",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
