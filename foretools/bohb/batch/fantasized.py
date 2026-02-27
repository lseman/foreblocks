from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

from .base import BatchSelector


class FantasizedBatchSelector(BatchSelector):
    def __init__(
        self,
        use_qlogei: bool,
        use_qnei: bool,
        q_fantasies: int,
        q_fantasy_noise: float,
        q_fantasy_weight: float,
    ):
        self.use_qlogei = bool(use_qlogei)
        self.use_qnei = bool(use_qnei)
        self.q_fantasies = max(1, int(q_fantasies))
        self.q_fantasy_noise = float(max(0.0, q_fantasy_noise))
        self.q_fantasy_weight = float(max(0.0, q_fantasy_weight))

    def _fantasy_utility(
        self,
        config: Dict[str, Any],
        obs: List[Tuple[Dict[str, Any], float, Optional[float]]],
        predict_fn: Callable[
            [Dict[str, Any], List[Tuple[Dict[str, Any], float, Optional[float]]], int],
            Tuple[float, float],
        ],
        rng: np.random.Generator,
        ei_k: int,
    ) -> float:
        if not obs:
            return 0.0
        mu, sigma = predict_fn(config, obs, ei_k)
        best = float(min(o[1] for o in obs))
        total_sigma = math.sqrt(max(sigma**2 + self.q_fantasy_noise**2, 1e-12))
        ys = rng.normal(loc=mu, scale=total_sigma, size=self.q_fantasies)
        improvements = np.maximum(0.0, best - ys)
        if self.use_qlogei and not self.use_qnei:
            return float(np.mean(np.log1p(improvements)))
        return float(np.mean(improvements))

    def _fantasize_loss(
        self,
        config: Dict[str, Any],
        obs: List[Tuple[Dict[str, Any], float, Optional[float]]],
        predict_fn: Callable[
            [Dict[str, Any], List[Tuple[Dict[str, Any], float, Optional[float]]], int],
            Tuple[float, float],
        ],
        rng: np.random.Generator,
        ei_k: int,
    ) -> float:
        mu, sigma = predict_fn(config, obs, ei_k)
        total_sigma = math.sqrt(max(sigma**2 + self.q_fantasy_noise**2, 1e-12))
        if self.use_qnei:
            return float(rng.normal(mu, total_sigma))
        return float(mu)

    def select(
        self,
        candidates: List[Dict[str, Any]],
        scores: np.ndarray,
        n: int,
        **kwargs: Any,
    ) -> List[int]:
        if n <= 0 or not candidates:
            return []
        observations = kwargs.get("observations")
        budget = kwargs.get("budget")
        acq_fn = kwargs.get("acq_fn")
        refit_fn = kwargs.get("refit_fn")
        predict_fn = kwargs.get("predict_fn")
        rng = kwargs.get("rng")
        ei_k = int(kwargs.get("ei_k", 10))
        if (
            observations is None
            or acq_fn is None
            or refit_fn is None
            or predict_fn is None
            or rng is None
        ):
            return list(np.argsort(scores)[-n:][::-1])

        selected: List[int] = []
        remaining = list(range(len(candidates)))
        temp_obs = list(observations)

        for _ in range(min(n, len(candidates))):
            good_models, bad_models = refit_fn(temp_obs)
            best_idx = None
            best_score = -float("inf")
            for idx in remaining:
                cand = candidates[idx]
                base_score = float(acq_fn(cand, good_models, bad_models))
                utility = self._fantasy_utility(cand, temp_obs, predict_fn, rng, ei_k)
                if self.use_qlogei and not self.use_qnei:
                    score = logsumexp(
                        [
                            math.log(max(base_score, 1e-300)),
                            self.q_fantasy_weight * utility,
                        ]
                    )
                else:
                    score = base_score * (1.0 + self.q_fantasy_weight * utility)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            loss_f = self._fantasize_loss(
                candidates[best_idx], temp_obs, predict_fn, rng, ei_k
            )
            b = None if budget is None else float(budget)
            temp_obs.append((dict(candidates[best_idx]), float(loss_f), b))

        return selected

