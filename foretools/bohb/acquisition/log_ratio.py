from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from scipy.special import logsumexp
from scipy.stats import norm

from ..utils import safe_log
from .base import AcquisitionStrategy


class LogRatioAcquisition(AcquisitionStrategy):
    def __init__(
        self,
        *,
        log_likelihood_fn: Callable[
            [Dict[str, Any], Dict[str, Dict[str, Any]]], float
        ],
        soft_constraint_violation_fn: Callable[[Dict[str, Any]], float],
        predict_mu_sigma_fn: Callable[[Dict[str, Any], int], Tuple[float, float]],
        gp_ucb_score_fn: Callable[[Dict[str, Any]], Optional[float]],
        observations_fn: Callable[
            [], List[Tuple[Dict[str, Any], float, Optional[float]]]
        ],
        soft_constraints_enabled: bool,
        ctpe_constraints: bool,
        constraint_violation_penalty: Optional[float],
        soft_penalty_weight: float,
        use_ei: bool,
        ei_k: int,
        use_ucb: bool,
        ucb_kappa: float,
        acq_log_space: bool,
        acq_softplus: bool,
    ):
        self.log_likelihood_fn = log_likelihood_fn
        self.soft_constraint_violation_fn = soft_constraint_violation_fn
        self.predict_mu_sigma_fn = predict_mu_sigma_fn
        self.gp_ucb_score_fn = gp_ucb_score_fn
        self.observations_fn = observations_fn
        self.soft_constraints_enabled = bool(soft_constraints_enabled)
        self.ctpe_constraints = bool(ctpe_constraints)
        self.constraint_violation_penalty = constraint_violation_penalty
        self.soft_penalty_weight = float(soft_penalty_weight)
        self.use_ei = bool(use_ei)
        self.ei_k = int(ei_k)
        self.use_ucb = bool(use_ucb)
        self.ucb_kappa = float(ucb_kappa)
        self.acq_log_space = bool(acq_log_space)
        self.acq_softplus = bool(acq_softplus)

    @staticmethod
    def _softplus(x: float) -> float:
        if x > 50.0:
            return float(x)
        if x < -50.0:
            return float(math.exp(x))
        return float(math.log1p(math.exp(x)))

    def score(
        self,
        config: Dict[str, Any],
        good_models: Dict[str, Dict[str, Any]],
        bad_models: Dict[str, Dict[str, Any]],
    ) -> float:
        log_l = self.log_likelihood_fn(config, good_models)
        if self.soft_constraints_enabled:
            penalty = self.soft_constraint_violation_fn(config)
            if penalty > 0:
                weight = (
                    float(self.constraint_violation_penalty)
                    if self.ctpe_constraints
                    and self.constraint_violation_penalty is not None
                    else float(self.soft_penalty_weight)
                )
                log_l -= weight * float(penalty)
        log_g = self.log_likelihood_fn(config, bad_models)
        v = float(log_l - log_g)

        obs = self.observations_fn()

        if self.acq_log_space:
            log_score = v
            if self.use_ei:
                log_ei = safe_log(1e-300)
                mu, sigma = self.predict_mu_sigma_fn(config, self.ei_k)
                if sigma > 1e-12:
                    best = min(o[1] for o in obs) if obs else mu
                    z = (best - mu) / sigma
                    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
                    log_ei = safe_log(max(float(ei), 1e-300))
                log_score += log_ei
            if self.use_ucb:
                ucb = self.gp_ucb_score_fn(config)
                if ucb is not None:
                    ucb_term = max(0.0, float(self.ucb_kappa) * float(ucb))
                    if ucb_term > 0.0:
                        log_score = float(logsumexp([log_score, safe_log(ucb_term)]))
            return self._softplus(log_score) if self.acq_softplus else float(log_score)

        if v > 50:
            score = float("inf")
        elif v < -50:
            score = 0.0
        else:
            score = float(math.exp(v))

        if self.use_ei and score > 0:
            mu, sigma = self.predict_mu_sigma_fn(config, self.ei_k)
            if sigma > 1e-9:
                best = min(o[1] for o in obs) if obs else mu
                z = (best - mu) / sigma
                ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
                score *= float(max(ei, 0.0))
        if self.use_ucb and score > 0:
            ucb = self.gp_ucb_score_fn(config)
            if ucb is not None:
                score += float(self.ucb_kappa) * float(ucb)
        return score

