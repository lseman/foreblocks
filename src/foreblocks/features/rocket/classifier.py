"""Rocket-family transform(s) + per-branch scaling + `RidgeClassifierCV`,
with optional kernel pruning (POCKET or S-ROCKET)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from foreblocks.features._validation import normalize_series

# Named combinations expand to several transforms whose (separately scaled)
# features are concatenated.
_COMBINATIONS = {"multirocket-hydra": ("multirocket", "hydra")}


def make_rocket_transform(name: str, **params):
    """Build a Rocket-family transform by name: 'rocket', 'minirocket',
    'multirocket', 'fused', 'hydra' or 'selfrocket'.
    """
    from foreblocks.features.rocket import (
        FusedRocket,
        Hydra,
        MiniRocket,
        MultiRocket,
        Rocket,
        SelfRocket,
    )

    transforms = {
        "rocket": Rocket,
        "minirocket": MiniRocket,
        "multirocket": MultiRocket,
        "fused": FusedRocket,
        "hydra": Hydra,
        "selfrocket": SelfRocket,
    }
    if name not in transforms:
        raise ValueError(f"Unknown transform {name!r}; expected one of {sorted(transforms)}.")
    return transforms[name](**params)


def _default_scaler(transform):
    from foreblocks.features.rocket.hydra import Hydra, SparseScaler

    return SparseScaler() if isinstance(transform, Hydra) else StandardScaler()


class RocketClassifier(ClassifierMixin, BaseEstimator):
    """The standard ROCKET-family pairing: random-convolution features,
    per-feature scaling, and a ridge classifier with cross-validated
    regularization.

    `transformer` is a name accepted by `make_rocket_transform`, a combination
    name (`"multirocket-hydra"`), an unfitted transformer instance (cloned), or
    a list of names/instances whose features are concatenated. Each branch is
    scaled separately: `SparseScaler` for Hydra, `StandardScaler` otherwise.
    `transformer_params` configures named transforms: either one dict applied
    to every named branch (e.g. `{"seed": 0}`) or a dict keyed by transform
    name (e.g. `{"multirocket": {"num_features": 20_000}, "hydra": {"g": 32}}`).
    Supervised transforms (`SelfRocket`) are fitted with the labels.

    `pruning` (`"pocket"` or `"s-rocket"`) keeps `prune_keep` (a fraction or a
    count) of the kernels of a single-branch Rocket, MiniRocket or MultiRocket
    transform, then refits the scaler and ridge on the surviving features;
    pruned kernels are not computed at predict time. `prune_params` are passed
    to `pocket_select` / `srocket_select`.

    `normalize` z-scores each series before the transform. `predict_proba`
    is a softmax over ridge decision scores: a monotone ranking, not a
    calibrated probability.
    """

    def __init__(
        self,
        transformer="minirocket",
        transformer_params: dict | None = None,
        normalize: bool = True,
        alphas=None,
        pruning: str | None = None,
        prune_keep: float | int = 0.4,
        prune_params: dict | None = None,
    ):
        self.transformer = transformer
        self.transformer_params = transformer_params
        self.normalize = normalize
        self.alphas = alphas
        self.pruning = pruning
        self.prune_keep = prune_keep
        self.prune_params = prune_params

    def _prepare(self, X) -> np.ndarray:
        return normalize_series(X) if self.normalize else np.asarray(X, dtype=np.float32)

    def _build_transforms(self) -> list:
        spec = self.transformer
        if isinstance(spec, str):
            spec = list(_COMBINATIONS.get(spec, (spec,)))
        elif not isinstance(spec, (list, tuple)):
            spec = [spec]
        params = dict(self.transformer_params or {})
        per_name = bool(params) and all(isinstance(v, dict) for v in params.values())
        names = [s for s in spec if isinstance(s, str)]
        transforms = []
        for s in spec:
            if isinstance(s, str):
                kwargs = params.get(s, {}) if per_name else params
                transforms.append(make_rocket_transform(s, **kwargs))
            else:
                transforms.append(clone(s))
        if per_name and set(params) - set(names):
            raise ValueError(f"transformer_params names {sorted(set(params) - set(names))} are not used.")
        return transforms

    def _branch_features(self, x: np.ndarray) -> list[np.ndarray]:
        return [t.transform(x) for t in self.transformers_]

    def fit(self, X, y) -> RocketClassifier:
        x = self._prepare(X)
        y = np.asarray(y)
        self.transformers_ = self._build_transforms()
        raw = [t.fit(x, y).transform(x) for t in self.transformers_]
        if self.pruning is not None:
            raw = self._prune(raw, y)
        self.scalers_ = [_default_scaler(t).fit(f) for t, f in zip(self.transformers_, raw)]
        features = np.concatenate([s.transform(f) for s, f in zip(self.scalers_, raw)], axis=1)
        alphas = np.logspace(-3, 3, 13) if self.alphas is None else self.alphas
        self.classifier_ = RidgeClassifierCV(alphas=alphas)
        self.classifier_.fit(features, y)
        self.classes_ = self.classifier_.classes_
        return self

    def _prune(self, raw: list[np.ndarray], y: np.ndarray) -> list[np.ndarray]:
        from foreblocks.features.rocket.pruning import pocket_select, srocket_select

        selectors = {"pocket": pocket_select, "s-rocket": srocket_select, "srocket": srocket_select}
        if self.pruning not in selectors:
            raise ValueError(f"Unknown pruning {self.pruning!r}; expected 'pocket' or 's-rocket'.")
        if len(self.transformers_) != 1 or not hasattr(self.transformers_[0], "select_kernels"):
            raise ValueError(
                "Pruning needs a single Rocket, MiniRocket or MultiRocket transform."
            )
        transform, features = self.transformers_[0], raw[0]
        groups = transform.kernel_groups_
        scaled = StandardScaler().fit_transform(features)
        kept = selectors[self.pruning](
            scaled, y, groups, keep=self.prune_keep, **(self.prune_params or {})
        )
        self.transformers_ = [transform.select_kernels(kept)]
        self.kept_kernels_ = kept
        return [features[:, np.isin(groups, kept)]]

    @property
    def transformer_(self):
        """The fitted transform (single-branch classifiers)."""
        return self.transformers_[0] if len(self.transformers_) == 1 else self.transformers_

    def features(self, X) -> np.ndarray:
        """Scaled transform features, as the ridge classifier sees them."""
        if not hasattr(self, "classifier_"):
            raise RuntimeError("Call fit before predict")
        x = self._prepare(X)
        return np.concatenate(
            [s.transform(f) for s, f in zip(self.scalers_, self._branch_features(x))], axis=1
        )

    @property
    def scaler_(self):
        return self.scalers_[0] if len(self.scalers_) == 1 else self.scalers_

    def decision_function(self, X) -> np.ndarray:
        return self.classifier_.decision_function(self.features(X))

    def predict(self, X) -> np.ndarray:
        return self.classifier_.predict(self.features(X))

    def predict_proba(self, X) -> np.ndarray:
        return softmax_scores(self.decision_function(X))


def softmax_scores(scores: np.ndarray) -> np.ndarray:
    """Softmax over ridge decision scores; a 1-D binary score becomes two columns."""
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)
