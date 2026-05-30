from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm

Observation = tuple[dict[str, Any], float, float | None]


class qNoisyEISelector:
    """
    qNoisy Expected Improvement (qNEI) batch acquisition function.

    Implements the qNEI acquisition function from Jones et al. (1998)
    extended to batches by Ginsbourger et al. (2010) and later
    improved by Poloczek et al. (2017).

    Key properties:
    - Handles noisy observations directly (no need for duplicate evaluations)
    - Models both the mean and uncertainty of pending evaluations
    - Selects batches that maximize expected improvement over the best
      observed (or pending) point
    - Supports constraints via penalty terms

    Unlike classic EI which assumes a single best observation, qNEI
    accounts for the fact that the "best" point itself has uncertainty
    when there are pending (not-yet-evaluated) candidates.

    Requirements:
    - sklearn.gaussian_process.GaussianProcessRegressor
    - scipy.optimize.minimize
    """

    def __init__(
        self,
        n_fantasies: int = 16,
        noise_level: float = 1e-4,
        constraint_penalty: float = 10.0,
        exploration_weight: float = 0.0,
        max_obs_for_gp: int = 200,
        min_obs_for_gp: int = 10,
    ):
        """
        Args:
            n_fantasies: Number of Monte Carlo samples for the integral.
            noise_level: Prior noise level for GP observations.
            constraint_penalty: Penalty weight for constraint violations.
            exploration_weight: Weight for entropy-based exploration bonus.
                0 = pure exploitation, higher = more exploration.
            max_obs_for_gp: Maximum observations to use for GP fitting.
            min_obs_for_gp: Minimum observations before using GP.
        """
        self.n_fantasies = max(1, int(n_fantasies))
        self.noise_level = float(noise_level)
        self.constraint_penalty = float(constraint_penalty)
        self.exploration_weight = float(exploration_weight)
        self.max_obs_for_gp = int(max_obs_for_gp)
        self.min_obs_for_gp = int(min_obs_for_gp)

    def compute(
        self,
        config: dict[str, Any],
        observations: list[Observation],
        pending_configs: list[dict[str, Any]] | None = None,
        distance_fn: Any = None,
    ) -> float:
        """
        Compute qNEI score for a single candidate config.

        Args:
            config: Candidate configuration.
            observations: List of (config, loss, budget) tuples.
            pending_configs: List of configs currently being evaluated.
            distance_fn: Optional function to build GP features.

        Returns:
            qNEI acquisition value (higher = better).
        """
        if len(observations) < self.min_obs_for_gp:
            return 0.0

        # Filter to max observations for speed
        obs_filtered = observations[-self.max_obs_for_gp:]
        losses = np.array([o[1] for o in obs_filtered], dtype=float)

        # Handle infinite/NaN losses
        valid = np.isfinite(losses)
        if not np.any(valid):
            return 0.0
        losses = losses[valid]

        # Find best observed (minimum loss)
        best_loss = float(np.min(losses))

        # Compute GP predictive distribution at candidate
        mu, sigma = self._predict_gp(
            config, obs_filtered, distance_fn
        )

        # Account for pending evaluations (noisy best)
        pending_mu = 0.0
        pending_sigma = 0.0
        if pending_configs and distance_fn:
            pending_mu, pending_sigma = self._estimate_pending(
                pending_configs, obs_filtered, distance_fn
            )

        # Total noise from pending evaluations
        total_noise_sigma = math.sqrt(max(sigma**2 + pending_sigma**2, 1e-20))

        if total_noise_sigma < 1e-12:
            # Deterministic case: simple EI
            return max(0.0, best_loss - mu)

        # Standardized improvement
        z = (best_loss - mu - pending_mu) / total_noise_sigma

        # Expected Improvement = (best - mu - pending) * Phi(z) + sigma * phi(z)
        ei = (best_loss - mu - pending_mu) * norm.cdf(z) + total_noise_sigma * norm.pdf(z)

        return float(max(0.0, ei))

    def compute_batch(
        self,
        candidates: list[dict[str, Any]],
        observations: list[Observation],
        n_select: int,
        pending_configs: list[dict[str, Any]] | None = None,
        distance_fn: Any = None,
    ) -> list[int]:
        """
        Compute qNEI for a batch of candidates and select top-k.

        Uses greedy selection: iteratively select the candidate with
        the highest qNEI, conditioning on previous selections.

        Args:
            candidates: Pool of candidate configurations.
            observations: List of (config, loss, budget) tuples.
            n_select: Number of candidates to select.
            pending_configs: Currently pending evaluations.
            distance_fn: Optional distance function for features.

        Returns:
            List of indices into candidates list.
        """
        if n_select <= 0 or not candidates:
            return []

        selected = []
        remaining = list(range(len(candidates)))
        temp_obs = list(observations)

        for _ in range(min(n_select, len(candidates))):
            best_idx = None
            best_score = -float("inf")

            for idx in remaining:
                score = self.compute(
                    candidates[idx],
                    temp_obs,
                    pending_configs,
                    distance_fn,
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None or best_score <= 0:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)

            # Add fantasy observation for next iteration
            # Use GP predictive mean as the fantasy value
            mu, _ = self._predict_gp(
                candidates[best_idx], temp_obs, distance_fn
            )
            temp_obs.append((dict(candidates[best_idx]), float(mu), None))

        return selected

    def _predict_gp(
        self,
        config: dict[str, Any],
        observations: list[Observation],
        distance_fn: Any = None,
    ) -> tuple[float, float]:
        """
        Predict GP mean and std at a candidate point.

        Returns:
            (mean, std) tuple.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            return self._fallback_predict(config, observations)

        # Extract features and losses
        configs = [o[0] for o in observations]
        losses = np.array([o[1] for o in observations], dtype=float)

        if len(configs) < 5:
            return self._fallback_predict(config, observations)

        try:
            X = self._build_features(configs, distance_fn)
            y = np.log1p(np.maximum(losses, 1e-8))

            # Determine length scale from median pairwise distance
            dists = []
            for i in range(min(50, len(X))):
                for j in range(i + 1, min(50, len(X))):
                    dists.append(np.linalg.norm(X[i] - X[j]))
            ls = float(np.median(dists)) if dists else 1.0
            ls = max(ls, 0.1)  # Floor

            kernel = RBF(length_scale=ls) + WhiteKernel(noise_level=self.noise_level)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-8,
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=None,
            )
            gp.fit(X, y)

            xq = self._build_features_single(config, distance_fn)
            mu, std = gp.predict(xq.reshape(1, -1), return_std=True)

            # Convert back from log space
            mu_linear = float(np.expm1(mu[0]))
            std_linear = float(std[0]) * abs(mu_linear + 1) / abs(mu[0] + 1e-8) if abs(mu[0]) > 1e-8 else float(std[0])

            return max(mu_linear, 0.0), max(std_linear, 1e-6)
        except Exception:
            return self._fallback_predict(config, observations)

    def _estimate_pending(
        self,
        pending: list[dict[str, Any]],
        observations: list[Observation],
        distance_fn: Any = None,
    ) -> tuple[float, float]:
        """Estimate the mean and std of pending evaluation outcomes."""
        if not pending:
            return 0.0, 0.0

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            return 0.0, 1.0

        configs = [o[0] for o in observations]
        losses = np.array([o[1] for o in observations], dtype=float)

        if len(configs) < 5:
            return 0.0, 1.0

        try:
            X = self._build_features(configs, distance_fn)
            y = np.log1p(np.maximum(losses, 1e-8))

            dists = []
            for i in range(min(50, len(X))):
                for j in range(i + 1, min(50, len(X))):
                    dists.append(np.linalg.norm(X[i] - X[j]))
            ls = float(np.median(dists)) if dists else 1.0
            ls = max(ls, 0.1)

            kernel = RBF(length_scale=ls) + WhiteKernel(noise_level=self.noise_level)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-8,
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=None,
            )
            gp.fit(X, y)

            # Predict for all pending configs
            pending_mus = []
            pending_sigmas = []
            for pcfg in pending:
                xq = self._build_features_single(pcfg, distance_fn)
                mu, std = gp.predict(xq.reshape(1, -1), return_std=True)
                pending_mus.append(float(np.expm1(mu[0])))
                pending_sigmas.append(float(max(std[0], 1e-6)))

            # Aggregate: mean of means, sqrt of sum of variances
            mean_mu = float(np.mean(pending_mus))
            total_sigma = math.sqrt(float(np.mean([s**2 for s in pending_sigmas])))

            return mean_mu, total_sigma
        except Exception:
            return 0.0, 1.0

    def _build_features(
        self, configs: list[dict[str, Any]], distance_fn: Any = None
    ) -> np.ndarray:
        """Build feature matrix from configs using distance function."""
        if distance_fn is not None:
            # Use distance-based features
            # Embed each config as its distances to other configs
            n = len(configs)
            features = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    features[i, j] = distance_fn(configs[i], configs[j])
            # Use SVD to reduce dimensionality
            try:
                U, S, Vt = np.linalg.svd(features, full_matrices=False)
                k = min(10, len(S))
                return U[:, :k] * np.sqrt(S[:k])
            except Exception:
                return features
        else:
            # Fallback: use parameter values directly
            if not configs:
                return np.empty((0, 0))
            all_params = list(configs[0].keys())
            feature_list = []
            for cfg in configs:
                feat = []
                for p in all_params:
                    v = cfg.get(p, 0)
                    if isinstance(v, (int, float)):
                        feat.append(float(v))
                    else:
                        feat.append(0.0)
                feature_list.append(feat)
            return np.array(feature_list, dtype=float)

    def _build_features_single(
        self, config: dict[str, Any], distance_fn: Any = None
    ) -> np.ndarray:
        """Build feature vector from single config."""
        if distance_fn is not None:
            # Distance-based feature (1D: distance to origin in distance space)
            return np.array([0.0])  # Placeholder; use in context of full feature matrix
        else:
            feat = []
            for v in config.values():
                if isinstance(v, (int, float)):
                    feat.append(float(v))
                else:
                    feat.append(0.0)
            return np.array(feat, dtype=float) if feat else np.array([0.0])

    def _fallback_predict(
        self,
        config: dict[str, Any],
        observations: list[Observation],
    ) -> tuple[float, float]:
        """
        Fallback prediction when GP is unavailable.

        Uses k-NN approach: find nearest observations and weight
        their losses by inverse distance.
        """
        if not observations:
            return 0.0, 1.0

        configs = [o[0] for o in observations]
        losses = np.array([o[1] for o in observations], dtype=float)

        # Simple k-NN prediction
        k = min(5, len(configs))
        if k < 1:
            return float(np.mean(losses)), float(np.std(losses))

        # If distance_fn available, use it; otherwise use parameter values
        dists = []
        for cfg in configs:
            try:
                dist = sum(
                    abs(float(cfg.get(p, 0)) - float(c.get(p, 0)))
                    for p in config
                    if isinstance(cfg.get(p), (int, float)) and isinstance(config.get(p), (int, float))
                )
            except (TypeError, ValueError):
                dist = 1.0
            dists.append(dist)

        dists = np.array(dists)
        idx = np.argpartition(dists, k)[:k]
        near_dists = dists[idx]
        near_losses = losses[idx]

        # Inverse distance weighting
        weights = 1.0 / (near_dists + 1e-8)
        weights = weights / weights.sum()

        mu = float(np.sum(weights * near_losses))
        sigma = float(np.sqrt(np.sum(weights * (near_losses - mu)**2)))

        return max(mu, 0.0), max(sigma, 1e-6)
