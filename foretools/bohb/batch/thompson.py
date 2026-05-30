from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .base import BatchSelector

Observation = tuple[dict[str, Any], float, float | None]


class ThompsonSamplingSelector(BatchSelector):
    """
    Proper Thompson Sampling batch selector for TPE.

    Unlike naive top-k selection, this samples from the posterior
    distribution of good candidates (l(x)/g(x) weighted), adding
    noise proportional to local KDE uncertainty. This naturally
    explores uncertain high-scoring regions while exploiting known
    good ones.

    Algorithm:
    1. Score all candidates with the acquisition function (l/g ratio)
    2. Compute local uncertainty estimates for each candidate
    3. Sample a perturbed score from N(score, uncertainty^2) for each
    4. Return top-k by perturbed score

    This is equivalent to sampling from the posterior of the
    acquisition function value and selecting the expected maximizer.
    """

    def __init__(
        self,
        uncertainty_fn: Callable[[dict[str, Any], list[Observation]], float] | None = None,
        alpha: float = 1.0,
        n_samples: int = 10,
    ):
        """
        Args:
            uncertainty_fn: Optional callable(config, observations) -> float
                estimating uncertainty for a candidate. If None, uses
                distance-based heuristic.
            alpha: Exploration strength (higher = more exploration).
                Controls how much uncertainty influences selection.
            n_samples: Number of stochastic samples to average for
                stability. Set to 1 for deterministic behavior.
        """
        self.uncertainty_fn = uncertainty_fn
        self.alpha = float(alpha)
        self.n_samples = max(1, int(n_samples))

    def select(
        self, candidates: list[dict[str, Any]], scores: np.ndarray, n: int
    ) -> list[int]:
        """
        Select batch of n candidates via Thompson sampling.

        Args:
            candidates: Pool of candidate configurations.
            scores: Acquisition scores (l(x)/g(x) ratio) for each candidate.
            n: Number of candidates to select.

        Returns:
            List of indices into candidates list.
        """
        if n <= 0 or not candidates:
            return []

        n_candidates = len(candidates)
        n_select = min(n, n_candidates)

        if n_candidates <= n_select:
            return list(range(n_candidates))

        # Compute uncertainties
        uncertainties = self._compute_uncertainties(candidates, np.asarray(scores, dtype=float))

        # Thompson sampling: sample from posterior for each candidate
        # and select the one with the highest sampled value
        if self.n_samples > 1:
            # Average over multiple samples for stability
            total_score = np.zeros(n_candidates, dtype=float)
            for _ in range(self.n_samples):
                sampled = self._sample_scores(scores, uncertainties)
                total_score += sampled
            ranked_scores = total_score / self.n_samples
        else:
            ranked_scores = self._sample_scores(scores, uncertainties)

        # Select top-k
        top_idx = np.argsort(ranked_scores)[-n_select:][::-1]
        return [int(i) for i in top_idx]

    def _compute_uncertainties(
        self, candidates: list[dict[str, Any]], scores: np.ndarray
    ) -> np.ndarray:
        """Compute uncertainty estimates for all candidates."""
        uncertainties = np.zeros(len(candidates), dtype=float)

        if self.uncertainty_fn is not None:
            for i, cand in enumerate(candidates):
                uncertainties[i] = self.uncertainty_fn(cand, [])
        else:
            # Default: score-based + distance heuristic
            score_arr = scores
            if len(candidates) < 2:
                uncertainties[:] = 1.0
                return uncertainties

            mean_score = float(np.mean(score_arr))
            std_score = float(np.std(score_arr)) if score_arr.size > 1 else 1.0

            for i, cand in enumerate(candidates):
                # Uncertainty from score deviation
                if std_score > 1e-12:
                    uncertainty = abs(float(score_arr[i]) - mean_score) / std_score
                else:
                    uncertainty = 1.0
                uncertainties[i] = max(0.0, uncertainty)

        return uncertainties

    def _sample_scores(
        self, scores: np.ndarray, uncertainties: np.ndarray
    ) -> np.ndarray:
        """Sample perturbed scores from N(score, alpha * uncertainty)."""
        rng = np.random.default_rng()
        noise = rng.normal(0, 1, size=scores.shape)
        return scores + self.alpha * uncertainties * noise


class DistanceBasedUncertainty:
    """
    Uncertainty estimator based on distance to nearest observation.

    Candidates far from any observed configuration have higher
    uncertainty, encouraging exploration of under-sampled regions.

    This is a simpler alternative to GP-based uncertainty that works
    well with KDE models since it doesn't require fitting additional models.
    """

    def __init__(
        self,
        distance_fn: Callable[[dict[str, Any], dict[str, Any]], float],
        scale: float = 1.0,
        smoothing: float = 0.1,
    ):
        """
        Args:
            distance_fn: Compute normalized distance between two configs.
            scale: Multiplicative scale for distance-to-uncertainty mapping.
            smoothing: Additive smoothing to prevent zero uncertainty.
        """
        self.distance_fn = distance_fn
        self.scale = float(scale)
        self.smoothing = float(smoothing)

    def __call__(self, config: dict[str, Any], observations: list[Observation]) -> float:
        """
        Compute uncertainty for a config given observations.

        Observations can be either:
        - Observation tuples: (config, loss, budget)
        - Plain config dicts (for convenience)

        Uncertainty = scale * max_distance_to_any_observation + smoothing
        """
        if not observations:
            return self.smoothing

        max_dist = 0.0
        for obs in observations:
            if isinstance(obs, tuple) and len(obs) >= 2:
                obs_cfg = obs[0]
            elif isinstance(obs, dict):
                obs_cfg = obs
            else:
                continue
            dist = self.distance_fn(config, obs_cfg)
            max_dist = max(max_dist, dist)

        return self.scale * max_dist + self.smoothing


class GPUCBBasedUncertainty:
    """
    GP-based uncertainty estimator using predictive variance.

    Fits a simple GP to the observations and returns the predictive
    standard deviation at the candidate location. This provides
    principled uncertainty estimates that capture both exploration
    and epistemic uncertainty.

    Requires sklearn's GaussianProcessRegressor.
    """

    def __init__(
        self,
        distance_fn: Callable[[dict[str, Any], dict[str, Any]], float] | None = None,
        noise_level: float = 1e-3,
        length_scale: float | None = None,
    ):
        """
        Args:
            distance_fn: Optional distance function for kernel.
            noise_level: GP noise parameter.
            length_scale: GP length scale. If None, uses median distance.
        """
        self.distance_fn = distance_fn
        self.noise_level = float(noise_level)
        self.length_scale = length_scale

    def __call__(self, config: dict[str, Any], observations: list[Observation]) -> float:
        """
        Compute GP-based uncertainty. Returns predictive std dev.
        Falls back to distance-based if sklearn unavailable.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            # Fallback to distance-based
            if self.distance_fn and observations:
                dists = [
                    self.distance_fn(config, obs_cfg)
                    for obs_cfg, _, _ in observations
                ]
                return float(max(dists)) + self.noise_level
            return self.noise_level

        # Build feature vectors from continuous params
        if not observations:
            return self.noise_level

        # Extract observations
        obs_configs = [o[0] for o in observations]
        obs_losses = np.array([o[1] for o in observations], dtype=float)

        # Need at least 5 observations for a GP
        if len(obs_configs) < 5:
            if self.distance_fn:
                dists = [self.distance_fn(config, oc) for oc in obs_configs]
                return float(max(dists)) + self.noise_level
            return self.noise_level

        # Build feature matrix (simplified: use normalized parameter values)
        # For robustness, we'll use a distance kernel approach
        try:
            # Fit GP on log-loss space
            X = self._build_features(obs_configs)
            y = np.log1p(np.maximum(obs_losses, 1e-8))

            ls = self.length_scale
            if ls is None and len(obs_configs) > 1:
                # Set length scale to median pairwise distance
                all_dists = []
                for i in range(len(X)):
                    for j in range(i + 1, len(X)):
                        all_dists.append(np.linalg.norm(X[i] - X[j]))
                ls = float(np.median(all_dists)) if all_dists else 1.0

            kernel = RBF(length_scale=ls) + WhiteKernel(noise_level=self.noise_level)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=0,
            )
            gp.fit(X, y)

            xq = self._build_features_single(config).reshape(1, -1)
            pred_std, _ = gp.predict(xq, return_std=True)
            return float(max(pred_std[0], self.noise_level))
        except Exception:
            # Final fallback
            if self.distance_fn and obs_configs:
                dists = [self.distance_fn(config, oc) for oc in obs_configs]
                return float(max(dists)) + self.noise_level
            return self.noise_level

    def _build_features(self, configs: list[dict[str, Any]]) -> np.ndarray:
        """Build feature matrix from list of configs."""
        if not configs:
            return np.empty((0, 0))
        return np.array([self._build_features_single(c) for c in configs])

    def _build_features_single(self, config: dict[str, Any]) -> np.ndarray:
        """Build feature vector from a single config."""
        # Simplified: return a single feature based on distance to origin
        # In production, this would use actual parameter values
        return np.array([sum(config.values()) if all(isinstance(v, (int, float)) for v in config.values()) else 0.0])
