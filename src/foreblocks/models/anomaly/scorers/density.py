"""Native projected histograms and diagonal Gaussian mixture EM."""

import numpy as np

from foreblocks.models.anomaly.scorers.base import (
    FittedScorer,
    positive_float,
    positive_int,
)


class LODA(FittedScorer):
    """Sparse random projections with smoothed equal-width histograms.

    Fixed projection/bin counts are used instead of adaptive stopping/bin
    selection. Out-of-range values get an empty-bin pseudocount probability.
    Scores average negative log *density*, not bin masses.
    """

    def __init__(
        self, n_projections=100, n_bins=20, alpha=0.1, *, standardize=True, seed=42
    ):
        super().__init__(standardize=standardize, seed=seed)
        self.n_projections = positive_int("n_projections", n_projections)
        self.n_bins = positive_int("n_bins", n_bins, 2)
        self.alpha = positive_float("alpha", alpha)

    def _fit(self, x):
        rng = np.random.default_rng(self.seed)
        d = x.shape[1]
        self.projections_ = np.zeros((d, self.n_projections))
        for i in range(self.n_projections):
            indices = rng.choice(d, max(1, int(np.sqrt(d))), replace=False)
            weights = rng.normal(size=len(indices))
            self.projections_[indices, i] = weights / max(
                np.linalg.norm(weights), 1e-12
            )
        self.histograms_ = []
        for col in (x @ self.projections_).T:
            if col.min() == col.max():
                self.histograms_.append((None, np.array([col[0]]), len(x)))
            else:
                counts, edges = np.histogram(col, bins=self.n_bins)
                self.histograms_.append((counts, edges, len(x)))

    def _score(self, x):
        scores = np.zeros(len(x))
        for col, (counts, edges, n) in zip((x @ self.projections_).T, self.histograms_):
            if counts is None:
                scores += np.where(
                    col == edges[0], 0, np.log((n + self.alpha) / self.alpha)
                )
            else:
                index = np.clip(
                    np.searchsorted(edges, col, side="right") - 1, 0, self.n_bins - 1
                )
                inside = (col >= edges[0]) & (col <= edges[-1])
                mass = (np.where(inside, counts[index], 0) + self.alpha) / (
                    n + self.alpha * self.n_bins
                )
                scores -= np.log(mass / np.diff(edges)[index])
        return scores / self.n_projections


def _logsumexp(x):
    maximum = x.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(x - maximum).sum(axis=1))


class GaussianMixtureScorer(FittedScorer):
    """Diagonal-covariance Gaussian mixture fitted by expectation maximization.

    Scores are negative log densities in normalized feature space. Diagonal
    covariance bounds memory on flattened windows; this is not a full-covariance
    mixture. ``lower_bounds_`` reports the EM average log likelihood history.
    """

    def __init__(
        self,
        n_components=3,
        max_iter=100,
        tol=1e-5,
        reg_covar=1e-5,
        *,
        standardize=True,
        seed=42,
    ):
        super().__init__(standardize=standardize, seed=seed)
        self.n_components = positive_int("n_components", n_components)
        self.max_iter = positive_int("max_iter", max_iter)
        self.tol = positive_float("tol", tol, allow_zero=True)
        self.reg_covar = positive_float("reg_covar", reg_covar)

    def _log_joint(self, x):
        # One [N, D] temporary per component, no [N, K, D] allocation.
        return np.column_stack(
            [
                np.log(weight)
                - 0.5
                * (
                    np.log(2 * np.pi * variance).sum()
                    + ((x - mean) ** 2 / variance).sum(axis=1)
                )
                for weight, mean, variance in zip(
                    self.weights_, self.means_, self.variances_
                )
            ]
        )

    def _fit(self, x):
        if self.n_components > len(x):
            raise ValueError(
                "n_components cannot exceed the number of training samples."
            )
        rng = np.random.default_rng(self.seed)
        # Distance-weighted seeds reduce duplicate initial components.
        centers = [x[rng.integers(len(x))]]
        distance = np.full(len(x), np.inf)
        for _ in range(1, self.n_components):
            distance = np.minimum(distance, ((x - centers[-1]) ** 2).sum(axis=1))
            index = (
                rng.choice(len(x), p=distance / distance.sum())
                if distance.sum() > 0
                else rng.integers(len(x))
            )
            centers.append(x[index])
        self.means_ = np.array(centers)
        self.variances_ = np.tile(
            np.maximum(x.var(axis=0), self.reg_covar), (self.n_components, 1)
        )
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.lower_bounds_ = []
        self.converged_ = False
        for _ in range(self.max_iter):
            joint = self._log_joint(x)
            norm = _logsumexp(joint)
            self.lower_bounds_.append(float(norm.mean()))
            if (
                len(self.lower_bounds_) > 1
                and abs(self.lower_bounds_[-1] - self.lower_bounds_[-2]) <= self.tol
            ):
                self.converged_ = True
                break
            responsibility = np.exp(joint - norm[:, None])
            mass = responsibility.sum(axis=0)
            for k in range(self.n_components):
                if mass[k] > 1e-12:
                    self.means_[k] = responsibility[:, k] @ x / mass[k]
                    self.variances_[k] = np.maximum(
                        responsibility[:, k] @ ((x - self.means_[k]) ** 2) / mass[k],
                        self.reg_covar,
                    )
            mass = np.maximum(mass, 1e-12)
            self.weights_ = mass / mass.sum()
        self.n_iter_ = len(self.lower_bounds_)

    def _score(self, x):
        return -_logsumexp(self._log_joint(x))
