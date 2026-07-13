"""Gaussian Process surrogate models with EI acquisition and ensemble support.

Replaces KDE-only TPE with proper GP uncertainty quantification.
Supports single GP, ensemble (bootstrap), and optional GP+TPE hybrid.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .utils import safe_log


class GPSurrogate:
    """Single GP surrogate for continuous-valued predictions."""

    def __init__(
        self,
        kernel: str = "rbf",
        alpha: float = 1e-6,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 5,
    ):
        self.kernel_name = kernel.lower()
        self.alpha = float(alpha)
        self.normalize_y = bool(normalize_y)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.model: GaussianProcessRegressor | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to training data.

        Args:
            X: Training inputs [n_samples, n_features]
            y: Training outputs [n_samples]
        """
        if len(X) == 0 or len(y) == 0:
            self.is_fitted = False
            return

        if len(X) != len(y):
            raise ValueError(f"X and y size mismatch: {len(X)} vs {len(y)}")

        # Standardize y for better GP numerics
        y_std = float(np.std(y)) if len(y) > 1 else 1.0
        y_std = max(y_std, 1e-12)
        y_normalized = (y - np.mean(y)) / y_std

        # Build kernel: RBF with constant multiplier + noise
        if self.kernel_name == "rbf":
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
                noise_level=self.alpha
            )
        else:
            kernel = RBF(length_scale=1.0) + WhiteKernel(
                noise_level=self.alpha
            )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=None,
        )
        self.model.fit(X, y_normalized)
        self.X_train = np.array(X, dtype=float)
        self.y_train = y_normalized
        self.is_fitted = True

    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict mean and optionally std at test points.

        Args:
            X: Test inputs [n_samples, n_features]
            return_std: Whether to return uncertainty

        Returns:
            (mu, sigma) if return_std=True, else mu
        """
        if not self.is_fitted or self.model is None:
            n = len(X)
            mu = np.zeros(n)
            sigma = np.ones(n) if return_std else None
            return (mu, sigma) if return_std else mu

        if return_std:
            mu, sigma = self.model.predict(X, return_std=True)
            sigma = np.maximum(sigma, 1e-12)
            return mu, sigma
        else:
            return self.model.predict(X, return_std=False)


class GPEnsemble:
    """Ensemble of GPs for robust uncertainty estimates.

    Trains K independent GPs on bootstrapped subsets of the data.
    """

    def __init__(
        self,
        n_models: int = 3,
        kernel: str = "rbf",
        alpha: float = 1e-6,
        subsample_frac: float = 0.8,
        seed: int | None = None,
    ):
        self.n_models = int(n_models)
        self.kernel = kernel
        self.alpha = float(alpha)
        self.subsample_frac = float(subsample_frac)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.models: list[GPSurrogate] = [
            GPSurrogate(kernel=kernel, alpha=alpha) for _ in range(n_models)
        ]
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ensemble via bootstrap subsampling.

        Args:
            X: Training inputs [n_samples, n_features]
            y: Training outputs [n_samples]
        """
        if len(X) == 0:
            self.is_fitted = False
            return

        n_subsample = max(2, int(self.subsample_frac * len(X)))
        for i, model in enumerate(self.models):
            seed_i = None if self.seed is None else self.seed + i
            rng_i = np.random.default_rng(seed_i)
            indices = rng_i.choice(len(X), size=n_subsample, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            model.fit(X_boot, y_boot)
        self.is_fitted = True

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict with ensemble averaging and uncertainty.

        Args:
            X: Test inputs [n_samples, n_features]
            return_std: If True, return mean ± ensemble std (aleatoric + epistemic)

        Returns:
            (mu, sigma) if return_std=True, else mu
        """
        if not self.is_fitted:
            n = len(X)
            mu = np.zeros(n)
            sigma = np.ones(n) if return_std else None
            return (mu, sigma) if return_std else mu

        preds = [m.predict(X, return_std=True) for m in self.models]
        mus = np.array([p[0] for p in preds])  # [n_models, n_samples]
        sigmas = np.array([p[1] for p in preds])  # [n_models, n_samples]

        mu = np.mean(mus, axis=0)
        if return_std:
            # Ensemble std: combination of aleatoric (avg within-model) and epistemic (across models)
            aleatoric = np.mean(sigmas, axis=0)
            epistemic = np.std(mus, axis=0)
            sigma = np.sqrt(aleatoric**2 + epistemic**2)
            return mu, sigma
        else:
            return mu


class ExpectedImprovement:
    """Expected Improvement acquisition function (Mockus et al.)."""

    def __init__(self, xi: float = 0.0):
        """
        Args:
            xi: Exploration-exploitation trade-off (0=exploit, >0=explore).
                Typically 0.0 (greedy) or 0.1-0.2 for exploration.
        """
        self.xi = float(xi)

    def score(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_best: float | None = None,
    ) -> np.ndarray:
        """Compute EI at test points.

        Args:
            mu: Predicted means [n_samples]
            sigma: Predicted stds [n_samples]
            y_best: Current best observed value. If None, uses min(mu).

        Returns:
            EI scores [n_samples], higher = more promising
        """
        if y_best is None:
            y_best = float(np.min(mu)) if len(mu) > 0 else 0.0

        sigma = np.maximum(sigma, 1e-12)
        Z = (y_best - mu - self.xi) / sigma
        ei = (y_best - mu - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.maximum(ei, 0.0)
        return ei


class UpperConfidenceBound:
    """Upper Confidence Bound (UCB) acquisition function."""

    def __init__(self, kappa: float = 2.576):
        """
        Args:
            kappa: Confidence level multiplier (default: 2.576 ≈ 99% CI).
        """
        self.kappa = float(kappa)

    def score(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Compute UCB at test points.

        Args:
            mu: Predicted means [n_samples]
            sigma: Predicted stds [n_samples]

        Returns:
            UCB scores [n_samples], lower = more promising (minimize)
        """
        return mu - self.kappa * sigma


def vectorized_ei_score(
    candidates: list[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    gp_model: GPSurrogate | GPEnsemble,
    encode_fn,
    ei_xi: float = 0.0,
) -> np.ndarray:
    """Vectorized EI scoring for batch acquisition.

    Args:
        candidates: List of candidate configs to score
        X_train: Training input data
        y_train: Training output data (losses, lower is better)
        gp_model: Fitted GP or ensemble
        encode_fn: Function to encode config dict -> feature vector
        ei_xi: Exploration parameter

    Returns:
        EI scores [n_candidates]
    """
    if not candidates or len(y_train) == 0:
        return np.zeros(len(candidates))

    # Encode all candidates at once
    X_cand = np.array([encode_fn(c) for c in candidates], dtype=float)

    # Predict with GP
    mu, sigma = gp_model.predict(X_cand, return_std=True)

    # Compute EI
    ei_fn = ExpectedImprovement(xi=ei_xi)
    y_best = float(np.min(y_train))
    ei_scores = ei_fn.score(mu, sigma, y_best=y_best)

    return ei_scores
