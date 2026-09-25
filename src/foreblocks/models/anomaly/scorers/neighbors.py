"""Native nearest-neighbor anomaly detectors; no external estimator wrappers."""

import numpy as np

from foreblocks.models.anomaly.scorers.base import (
    FittedScorer,
    positive_int,
    squared_distances,
)


class KNNScorer(FittedScorer):
    """k-th/mean/median nearest training-neighbor distance, in bounded chunks.

    Training scores use the same reference-inclusive rule as query scores:
    an exact training point is its own zero-distance neighbor. This keeps
    calibration and prediction consistent; it is not leave-one-out scoring.
    """

    def __init__(
        self,
        n_neighbors=5,
        method="largest",
        chunk_size=256,
        *,
        standardize=True,
        seed=42,
    ):
        super().__init__(standardize=standardize, seed=seed)
        self.n_neighbors = positive_int("n_neighbors", n_neighbors)
        self.chunk_size = positive_int("chunk_size", chunk_size)
        if method not in {"largest", "mean", "median"}:
            raise ValueError("method must be 'largest', 'mean', or 'median'.")
        self.method = method

    def _fit(self, x):
        if self.n_neighbors > len(x):
            raise ValueError(
                "n_neighbors cannot exceed the number of training samples."
            )
        self.reference_ = x.copy()

    def _score(self, x):
        scores = np.empty(len(x))
        k = self.n_neighbors
        for start in range(0, len(x), self.chunk_size):
            query = x[start : start + self.chunk_size]
            best = np.full((len(query), k), np.inf)
            for offset in range(0, len(self.reference_), self.chunk_size):
                distances = squared_distances(
                    query, self.reference_[offset : offset + self.chunk_size]
                )
                candidates = np.concatenate([best, distances], axis=1)
                best = np.partition(candidates, k - 1, axis=1)[:, :k]
            best = np.sqrt(best)
            reduce = {"largest": np.max, "mean": np.mean, "median": np.median}[
                self.method
            ]
            scores[start : start + len(query)] = reduce(best, axis=1)
        return scores


class INNE(FittedScorer):
    """Isolation using nearest-neighbor hypersphere ensembles (Bandaragoda et al.).

    A query inherits the local isolation of its smallest enclosing sphere,
    or scores one if no sphere contains it. Duplicate centers are collapsed
    within each sampled ensemble; a single unique center has radius zero.
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=32,
        chunk_size=256,
        *,
        standardize=True,
        seed=42,
    ):
        super().__init__(standardize=standardize, seed=seed)
        self.n_estimators = positive_int("n_estimators", n_estimators)
        self.max_samples = positive_int("max_samples", max_samples, 2)
        self.chunk_size = positive_int("chunk_size", chunk_size)

    def _fit(self, x):
        rng = np.random.default_rng(self.seed)
        self.ensembles_ = []
        for _ in range(self.n_estimators):
            centers = np.unique(
                x[rng.choice(len(x), min(self.max_samples, len(x)), replace=False)],
                axis=0,
            )
            if len(centers) == 1:
                radii = np.zeros(1)
                isolation = np.zeros(1)
            else:
                distances = squared_distances(centers, centers)
                np.fill_diagonal(distances, np.inf)
                nearest = distances.argmin(axis=1)
                radii = np.sqrt(distances[np.arange(len(centers)), nearest])
                isolation = np.clip(1 - radii[nearest] / radii, 0, 1)
            self.ensembles_.append((centers, radii, isolation))

    def _score(self, x):
        scores = np.zeros(len(x))
        for start in range(0, len(x), self.chunk_size):
            query = x[start : start + self.chunk_size]
            total = np.zeros(len(query))
            for centers, radii, isolation in self.ensembles_:
                distance = squared_distances(query, centers)
                inside = (distance < radii[None, :] ** 2) | (
                    (radii[None, :] == 0) & (distance == 0)
                )
                smallest = np.where(inside, radii[None, :], np.inf).argmin(axis=1)
                total += np.where(inside.any(axis=1), isolation[smallest], 1.0)
            scores[start : start + len(query)] = total / self.n_estimators
        return scores
