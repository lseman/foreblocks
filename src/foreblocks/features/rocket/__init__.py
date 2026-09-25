"""Rocket-family random-convolution transforms for time-series classification.

- Rocket: random kernels, (PPV, max) pooling
- MiniRocket: 84 fixed kernels, data-fitted biases, PPV pooling
- MultiRocket: MiniRocket kernels on the series and its first difference,
  with PPV/MPV/MIPV/LSPV pooling
- FusedRocket: ROCKET kernels with PPV/MPV pooling on fused Numba/Triton
  kernels (GPU-capable)
- Hydra: competing-kernel dictionary transform (+ SparseScaler)
- SelfRocket: MiniRocket with a learned input representation / pooling operator
- RocketClassifier: transform(s) + per-branch scaling + RidgeClassifierCV,
  including "multirocket-hydra" and POCKET / S-ROCKET kernel pruning
- pocket_select, srocket_select: kernel pruning selectors

All transforms follow the scikit-learn transformer API and accept `[N, L]`
input; Rocket, MiniRocket and MultiRocket also accept multivariate
`[N, C, L]`.
"""

from foreblocks.features.rocket.classifier import (
    RocketClassifier,
    make_rocket_transform,
    softmax_scores,
)
from foreblocks.features.rocket.fused import FusedRocket
from foreblocks.features.rocket.hydra import Hydra, SparseScaler
from foreblocks.features.rocket.minirocket import MiniRocket
from foreblocks.features.rocket.multirocket import MultiRocket
from foreblocks.features.rocket.pruning import pocket_select, srocket_select
from foreblocks.features.rocket.rocket import Rocket
from foreblocks.features.rocket.selfrocket import SelfRocket

__all__ = [
    "Rocket",
    "MiniRocket",
    "MultiRocket",
    "FusedRocket",
    "Hydra",
    "SparseScaler",
    "SelfRocket",
    "pocket_select",
    "srocket_select",
    "RocketClassifier",
    "make_rocket_transform",
    "softmax_scores",
]
