from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .utils import (
    _clamp,
    _reflect_into_bounds,
    inv_yeojohnson,
    safe_log,
    safe_normalize,
    yeojohnson_forward,
    yeojohnson_log_jacobian,
)


class BaseParamModel:
    kind: str

    def sample(self) -> Any:
        raise NotImplementedError

    def log_pdf(self, x: Any) -> float:
        raise NotImplementedError


@dataclass
class FloatModel(BaseParamModel):
    kind: str
    param: str
    prior_w: float
    mu: Optional[float]
    vals: Optional[np.ndarray]
    bw: Optional[float]
    w: Optional[np.ndarray]
    min_bandwidth: float
    local_bandwidth: bool
    local_bandwidth_k: int
    transform: Callable[[float, str], float]
    inv_transform: Callable[[float, str], float]
    float_bounds: Callable[[str], tuple[float, float]]
    sample_prior_model_space: Callable[[str], float]
    log_prior_model_space: Callable[[str], float]
    sample_mixture_1d: Callable[[np.ndarray, float, Optional[np.ndarray]], float]
    local_bw_func: Callable[[float, np.ndarray, float, int], float]
    rng: np.random.Generator
    warp_lmbda: Optional[float] = None

    def sample(self) -> Any:
        lo, hi = self.float_bounds(self.param)
        if self.kind == "prior_only_float":
            x = self.sample_prior_model_space(self.param)
        else:
            if self.prior_w > 0 and self.rng.random() < self.prior_w:
                x = self.sample_prior_model_space(self.param)
            elif self.kind == "single_float":
                x = float(
                    self.rng.normal(loc=self.mu, scale=max(1.0, self.min_bandwidth))
                )
            else:
                x = self.sample_mixture_1d(self.vals, float(self.bw), self.w)

        if self.warp_lmbda is not None:
            x = inv_yeojohnson(float(x), self.warp_lmbda)

        x = _reflect_into_bounds(float(x), lo, hi)
        return self.inv_transform(x, self.param)

    def log_pdf(self, x: Any) -> float:
        val = self.transform(float(x), self.param)
        log_jac = 0.0
        if self.warp_lmbda is not None:
            log_jac = yeojohnson_log_jacobian(val, self.warp_lmbda)
            val = yeojohnson_forward(val, self.warp_lmbda)
        lo, hi = self.float_bounds(self.param)
        if val < lo - 1e-6 or val > hi + 1e-6:
            dist = min(abs(val - lo), abs(val - hi))
            return -10.0 * dist

        if self.kind == "prior_only_float":
            log_kde = self.log_prior_model_space(self.param)
        elif self.kind == "single_float":
            log_kde = norm.logpdf(val, loc=self.mu, scale=max(1.0, self.min_bandwidth))
        else:
            centers = self.vals
            bw = float(self.bw)
            if self.local_bandwidth and centers is not None and centers.size > 0:
                bw = self.local_bw_func(val, centers, bw, k=self.local_bandwidth_k)
            w = self.w
            z = (val - centers) / bw
            log_comp = norm.logpdf(z) - safe_log(bw)
            if w is None:
                log_kde = float(logsumexp(log_comp) - safe_log(len(centers)))
            else:
                log_kde = float(logsumexp(log_comp, b=w))

        if self.prior_w > 0 and self.kind != "prior_only_float":
            log_prior = self.log_prior_model_space(self.param)
            log_kde = float(
                logsumexp(
                    [
                        log_kde + safe_log(1.0 - self.prior_w),
                        log_prior + safe_log(self.prior_w),
                    ]
                )
            )
        return float(log_kde + log_jac)


@dataclass
class IntModel(BaseParamModel):
    kind: str
    param: str
    lo: int
    hi: int
    prior_w: float
    mu: Optional[float]
    vals: Optional[np.ndarray]
    bw: Optional[float]
    w: Optional[np.ndarray]
    local_bandwidth: bool
    local_bandwidth_k: int
    sample_prior_model_space: Callable[[str], float]
    log_prior_model_space: Callable[[str], float]
    sample_mixture_1d: Callable[[np.ndarray, float, Optional[np.ndarray]], float]
    local_bw_func: Callable[[float, np.ndarray, float, int], float]
    rng: np.random.Generator
    probs: Optional[dict[Any, float]] = None

    def sample(self) -> Any:
        if self.kind == "int_discrete":
            keys = list(self.probs.keys())
            p = safe_normalize(np.array([self.probs[k] for k in keys], dtype=float))
            idx = int(self.rng.choice(len(keys), p=p))
            return int(keys[idx])

        if self.kind == "prior_only_int":
            x = self.sample_prior_model_space(self.param)
        else:
            if self.prior_w > 0 and self.rng.random() < self.prior_w:
                x = self.sample_prior_model_space(self.param)
            elif self.kind == "single_int":
                x = float(self.rng.normal(loc=self.mu, scale=5.0))
            else:
                x = self.sample_mixture_1d(self.vals, float(self.bw), self.w)

        x = int(round(_reflect_into_bounds(float(x), float(self.lo), float(self.hi))))
        return int(_clamp(x, self.lo, self.hi))

    def log_pdf(self, x: Any) -> float:
        if self.kind == "int_discrete":
            p = float(self.probs.get(int(x), 1e-12))
            return float(safe_log(p))

        val = float(x)
        if self.kind == "single_int":
            log_kde = norm.logpdf(val, loc=self.mu, scale=5.0)
        else:
            centers = self.vals
            bw = float(self.bw)
            if self.local_bandwidth and centers is not None and centers.size > 0:
                bw = self.local_bw_func(val, centers, bw, k=self.local_bandwidth_k)
            w = self.w
            z = (val - centers) / bw
            log_comp = norm.logpdf(z) - safe_log(bw)
            if w is None:
                log_kde = float(logsumexp(log_comp) - safe_log(len(centers)))
            else:
                log_kde = float(logsumexp(log_comp, b=w))

        if self.prior_w > 0:
            log_prior = self.log_prior_model_space(self.param)
            log_kde = float(
                logsumexp(
                    [
                        log_kde + safe_log(1.0 - self.prior_w),
                        log_prior + safe_log(self.prior_w),
                    ]
                )
            )
        return float(log_kde)


@dataclass
class CatModel(BaseParamModel):
    kind: str
    probs: dict[Any, float]
    rng: np.random.Generator

    def sample(self) -> Any:
        keys = list(self.probs.keys())
        p = safe_normalize(np.array([self.probs[k] for k in keys], dtype=float))
        idx = int(self.rng.choice(len(keys), p=p))
        return keys[idx]

    def log_pdf(self, x: Any) -> float:
        p = float(self.probs.get(x, 1e-12))
        return float(safe_log(p))
