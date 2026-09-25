"""Native random-network isolation with deviation-enhanced scoring.

Based on Xu et al., Deep Isolation Forest (2023), arXiv:2206.06602.
Uses independently sampled NumPy MLPs rather than the paper's CERE accelerator.
Representation normalization is fitted on training data and frozen at inference.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from foreblocks.models.anomaly.scorers.base import FittedScorer, positive_int


@dataclass
class _Node:
    size: int
    feature: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None


def _grow(x, depth, rng):
    node = _Node(len(x))
    if depth == 0 or len(x) < 2:
        return node
    low, high = x.min(axis=0), x.max(axis=0)
    candidates = np.flatnonzero(high > low)
    if not len(candidates):
        return node
    node.feature = int(rng.choice(candidates))
    node.threshold = float(rng.uniform(low[node.feature], high[node.feature]))
    # <= keeps the minimum on the left even when uniform returns its lower bound.
    mask = x[:, node.feature] <= node.threshold
    if mask.all():
        node.feature = -1
        return node
    node.left = _grow(x[mask], depth - 1, rng)
    node.right = _grow(x[~mask], depth - 1, rng)
    return node


def _traverse(root, x, corrections):
    lengths = np.empty(len(x))
    deviations = np.zeros(len(x))
    pending = [(root, np.arange(len(x)), 0, np.zeros(len(x)))]
    while pending:
        node, indices, depth, total = pending.pop()
        if node.feature < 0:
            lengths[indices] = depth + corrections[node.size]
            deviations[indices] = total / max(depth, 1)
            continue
        values = x[indices, node.feature]
        total = total + np.abs(values - node.threshold)
        mask = values <= node.threshold
        if mask.any():
            pending.append((node.left, indices[mask], depth + 1, total[mask]))
        if (~mask).any():
            pending.append((node.right, indices[~mask], depth + 1, total[~mask]))
    return lengths, deviations


class DeepIsolationForest(FittedScorer):
    """Optimization-free random MLP ensemble plus native isolation trees.

    The score is normalized path-length evidence times mean split deviation
    (DEAS). Training-fitted standardization followed by tanh bounds representation
    coordinates. No sklearn/PyOD tree implementation or neural training is used.
    """

    def __init__(
        self,
        n_ensemble=10,
        n_estimators=6,
        max_samples=256,
        hidden_sizes=(64, 32),
        representation_dim=16,
        batch_size=256,
        *,
        standardize=True,
        seed=42,
    ):
        super().__init__(standardize=standardize, seed=seed)
        self.n_ensemble = positive_int("n_ensemble", n_ensemble)
        self.n_estimators = positive_int("n_estimators", n_estimators)
        self.max_samples = positive_int("max_samples", max_samples, 2)
        self.hidden_sizes = tuple(
            positive_int("hidden_sizes entry", n) for n in hidden_sizes
        )
        self.representation_dim = positive_int("representation_dim", representation_dim)
        self.batch_size = positive_int("batch_size", batch_size)

    @staticmethod
    def _forward(x, layers):
        for i, (weight, bias) in enumerate(layers):
            x = x @ weight + bias
            if i < len(layers) - 1:
                x = np.tanh(x)
        return x

    def _fit(self, x):
        rng = np.random.default_rng(self.seed)
        sample_size = min(len(x), self.max_samples)
        self.sample_size_ = sample_size
        # c(n) = 2 H_(n-1) - 2(n-1)/n, including c(0)=c(1)=0.
        self.corrections_ = np.zeros(sample_size + 1)
        n = np.arange(2, sample_size + 1)
        self.corrections_[2:] = (
            2 * np.cumsum(1 / np.arange(1, sample_size)) - 2 * (n - 1) / n
        )
        self.members_ = []
        dimensions = (x.shape[1], *self.hidden_sizes, self.representation_dim)
        for _ in range(self.n_ensemble):
            layers = [
                (rng.normal(size=(a, b)) / np.sqrt(a), rng.normal(scale=0.1, size=b))
                for a, b in pairwise(dimensions)
            ]
            raw = np.concatenate(
                [
                    self._forward(x[i : i + self.batch_size], layers)
                    for i in range(0, len(x), self.batch_size)
                ]
            )
            mean, scale = raw.mean(axis=0), raw.std(axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            representation = np.tanh((raw - mean) / scale)
            trees = [
                _grow(
                    representation[rng.choice(len(x), sample_size, replace=False)],
                    int(np.ceil(np.log2(sample_size))),
                    rng,
                )
                for _ in range(self.n_estimators)
            ]
            self.members_.append((layers, mean, scale, trees))

    def _score(self, x):
        scores = np.empty(len(x))
        count = self.n_ensemble * self.n_estimators
        for start in range(0, len(x), self.batch_size):
            query = x[start : start + self.batch_size]
            lengths, deviations = np.zeros(len(query)), np.zeros(len(query))
            for layers, mean, scale, trees in self.members_:
                representation = np.tanh((self._forward(query, layers) - mean) / scale)
                for tree in trees:
                    path, deviation = _traverse(tree, representation, self.corrections_)
                    lengths += path
                    deviations += deviation
            scores[start : start + len(query)] = (
                np.exp2(-lengths / (count * self.corrections_[self.sample_size_]))
                * deviations
                / count
            )
        return scores
