"""SelF-Rocket: MiniRocket with a learned input representation and pooling
operator (Lo et al., "Time series classification with random convolution
kernels: pooling operators and input representations matter", 2024).

https://arxiv.org/abs/2409.01115 — reference code: https://github.com/ANR-MYEL/SelF-Rocket

MiniRocket kernels are fitted on the series (BASE) and its first difference
(DIFF). Every (representation, pooling operator) candidate — representations
BASE, DIFF and MIX (both concatenated), operators PPV, GMP, MPV, MIPV, LSPV —
is scored by ridge "voters" trained on stratified splits of the training set.
The candidate with the highest median validation accuracy wins if enough voters
rank it in their top `top`; otherwise `default` is used. Only the winning
features are produced at transform time.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import StandardScaler

from foreblocks.features._validation import as_panel
from foreblocks.features.rocket._kernels import POOLING_OPERATORS
from foreblocks.features.rocket.minirocket import MiniRocketParameters

REPRESENTATIONS = ("base", "diff", "mix")
# Operators the kernel computes together: PPV alone, the MultiRocket four, or all five.
_N_POOL = {"ppv": 1, "mpv": 4, "mipv": 4, "lspv": 4, "gmp": 5}


class SelfRocket(TransformerMixin, BaseEstimator):
    """Supervised MiniRocket variant: `fit(X, y)` selects one input
    representation and pooling operator.

    `num_features` is per representation (rounded down to a multiple of 84), so
    the output is `[N, F]` for BASE/DIFF and `[N, 2F]` for MIX. After fitting,
    `selection_` is the chosen `(representation, operator)`, and `scores_`
    maps every candidate to its per-voter validation accuracies.

    Voters: `RepeatedStratifiedKFold(n_splits, n_repeats)`, or, above
    `max_samples` training series, a stratified shuffle split with the same
    number of voters whose train and validation parts together hold
    `max_samples` series. Requires `L >= 10`.
    """

    def __init__(
        self,
        num_features: int = 10_000,
        max_dilations_per_kernel: int = 32,
        representations: tuple[str, ...] = REPRESENTATIONS,
        pooling: tuple[str, ...] = ("ppv", "gmp", "mpv", "mipv", "lspv"),
        n_splits: int = 3,
        n_repeats: int = 2,
        max_samples: int = 2_000,
        top: int = 4,
        vote_threshold: float = 0.5,
        default: tuple[str, str] = ("mix", "ppv"),
        alphas=None,
        seed: int | None = None,
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.representations = representations
        self.pooling = pooling
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.max_samples = max_samples
        self.top = top
        self.vote_threshold = vote_threshold
        self.default = default
        self.alphas = alphas
        self.seed = seed

    # ------------------------------------------------------------------ helpers
    def _splits(self, y: np.ndarray, rng) -> list[tuple[np.ndarray, np.ndarray]]:
        n_voters = self.n_splits * self.n_repeats
        seed = int(rng.integers(2**31 - 1))
        _, counts = np.unique(y, return_counts=True)
        n = len(y)
        if n <= self.max_samples and counts.min() >= self.n_splits:
            cv = RepeatedStratifiedKFold(
                n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=seed
            )
            return list(cv.split(np.zeros(n), y))
        test = max(len(counts), int(min(n, self.max_samples) / self.n_splits))
        train = max(len(counts), min(n, self.max_samples) - test)
        if counts.min() >= 2 and train + test <= n:
            cv = StratifiedShuffleSplit(n_voters, test_size=test, train_size=train, random_state=seed)
            return list(cv.split(np.zeros(n), y))
        cv = ShuffleSplit(n_voters, test_size=0.3, random_state=seed)
        return list(cv.split(np.zeros(n)))

    @staticmethod
    def _block(features: np.ndarray, operator: str, n_features: int) -> np.ndarray:
        i = POOLING_OPERATORS.index(operator)
        return features[:, i * n_features : (i + 1) * n_features]

    def _candidate(self, base, diff, representation, operator) -> np.ndarray:
        blocks = []
        if representation in ("base", "mix"):
            blocks.append(self._block(base, operator, self.n_base_))
        if representation in ("diff", "mix"):
            blocks.append(self._block(diff, operator, self.n_diff_))
        return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]

    # --------------------------------------------------------------------- API
    def fit(self, X, y=None) -> SelfRocket:
        if y is None:
            raise ValueError("SelfRocket selects features with labels; call fit(X, y).")
        for r in self.representations:
            if r not in REPRESENTATIONS:
                raise ValueError(f"Unknown representation {r!r}; expected {REPRESENTATIONS}.")
        for p in self.pooling:
            if p not in POOLING_OPERATORS:
                raise ValueError(f"Unknown pooling operator {p!r}; expected {POOLING_OPERATORS}.")
        x = as_panel(X, min_length=10)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.base_parameters_ = MiniRocketParameters(
            x, self.num_features, self.max_dilations_per_kernel, rng
        )
        self.diff_parameters_ = MiniRocketParameters(
            np.ascontiguousarray(np.diff(x, axis=-1)),
            self.num_features,
            self.max_dilations_per_kernel,
            rng,
        )
        self.n_base_ = len(self.base_parameters_.biases)
        self.n_diff_ = len(self.diff_parameters_.biases)

        candidates = [(r, p) for r in self.representations for p in self.pooling]
        if len(candidates) == 1:
            self.selection_, self.scores_ = candidates[0], {}
            return self._finish()
        n_pool = max(_N_POOL[p] for p in self.pooling)
        uses_base = any(r in ("base", "mix") for r in self.representations)
        uses_diff = any(r in ("diff", "mix") for r in self.representations)
        base = self.base_parameters_.transform(x, n_pool) if uses_base else None
        diff = (
            self.diff_parameters_.transform(np.ascontiguousarray(np.diff(x, axis=-1)), n_pool)
            if uses_diff
            else None
        )
        alphas = np.logspace(-3, 3, 10) if self.alphas is None else self.alphas
        splits = self._splits(y, rng)
        accuracy = np.empty((len(splits), len(candidates)))
        for j, (r, p) in enumerate(candidates):
            features = self._candidate(base, diff, r, p)
            for v, (train, val) in enumerate(splits):
                scaler = StandardScaler().fit(features[train])
                ridge = RidgeClassifierCV(alphas=alphas).fit(scaler.transform(features[train]), y[train])
                accuracy[v, j] = ridge.score(scaler.transform(features[val]), y[val])

        best = int(np.argmax(np.median(accuracy, axis=0)))
        # Vote validation: rank of the winner for each voter (0 = best; ties
        # share the better rank).
        ranks = (accuracy > accuracy[:, [best]]).sum(axis=1)
        support = float(np.mean(ranks < self.top))
        default = tuple(self.default)
        if support < self.vote_threshold and default in candidates:
            self.selection_ = default
        else:
            self.selection_ = candidates[best]
        self.vote_support_ = support
        self.scores_ = {c: accuracy[:, j] for j, c in enumerate(candidates)}
        return self._finish()

    def _finish(self) -> SelfRocket:
        r, _ = self.selection_
        self.n_features_out_ = (self.n_base_ if r in ("base", "mix") else 0) + (
            self.n_diff_ if r in ("diff", "mix") else 0
        )
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "selection_"):
            raise RuntimeError("Call fit before transform")
        x = as_panel(X, min_length=10, allow_empty=True)
        r, p = self.selection_
        n_pool = _N_POOL[p]
        base = self.base_parameters_.transform(x, n_pool) if r in ("base", "mix") else None
        diff = (
            self.diff_parameters_.transform(np.ascontiguousarray(np.diff(x, axis=-1)), n_pool)
            if r in ("diff", "mix")
            else None
        )
        return np.ascontiguousarray(self._candidate(base, diff, r, p))
