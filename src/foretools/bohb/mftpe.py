"""
Vectorized multi-fidelity multivariate TPE — the default BOHB sampler.

Design (following Falkner et al. 2018, "BOHB", and Watanabe 2023, "Tree-structured
Parzen Estimator: Understanding Its Algorithm Components"):

- Multi-fidelity model selection: the density model is fit on observations of the
  *largest* budget that has at least ``min_points_in_model`` results, so low-budget
  results guide search only until high-budget data is available.
- Multivariate Parzen estimators: each observation is one product-kernel component
  over all dimensions (captures parameter interactions, unlike independent 1-D TPE).
- Numeric dims live in the unit cube (log10 for log floats). Floats use truncated
  Gaussians; ints use the Gaussian mass over each integer's bin, so l(x)/g(x) is a
  proper ratio of probabilities on the integer lattice.
- Categoricals use an Aitchison-Aitken kernel.
- Uniform prior component in both l and g; Optuna-style age-decayed weights for g.
- Batch proposals use a constant liar: every pick is added to g as a pending "bad"
  point, which pushes subsequent picks of the same bracket apart.
- Startup samples are scrambled Sobol points for better space coverage.
- Conditional parameters (``{"condition": {"parent": ..., "values": [...]}}``) are
  supported: inactive dimensions are marginalised out of the kernels.

Everything is numpy-vectorized over (candidates × components × dims).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

from .utils import _canonical_config_key

try:
    from scipy.stats import qmc
except Exception:  # pragma: no cover - old scipy
    qmc = None

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
# Max (candidates × components × dims) elements materialised per chunk.
_CHUNK_ELEMS = 2_000_000


@dataclass
class _Dim:
    name: str
    kind: str  # "float" | "int" | "cat"
    lo: float = 0.0  # model-space lower bound (log10 for log floats)
    hi: float = 1.0
    log: bool = False
    count: int = 0  # number of integers / categories
    choices: list[Any] | None = None
    parent: str | None = None
    parent_values: list[Any] | None = None


def _parse_space(config_space: dict[str, tuple]) -> list[_Dim]:
    dims: list[_Dim] = []
    for name, spec in config_space.items():
        typ, rng = spec[0], spec[1]
        opts = spec[2] if len(spec) >= 3 and isinstance(spec[2], dict) else {}
        cond = opts.get("condition") if isinstance(opts.get("condition"), dict) else {}
        parent = cond.get("parent", opts.get("parent"))
        values = cond.get("values", opts.get("values"))
        if values is None:
            value = cond.get("value", opts.get("value"))
            values = None if value is None else [value]
        elif not isinstance(values, (list, tuple, set)):
            values = [values]

        if typ == "float":
            lo, hi = float(rng[0]), float(rng[1])
            is_log = len(rng) >= 3 and rng[2] == "log"
            if is_log:
                if lo <= 0:
                    raise ValueError(f"log-scaled param {name!r} needs lo > 0")
                lo, hi = math.log10(lo), math.log10(hi)
            d = _Dim(name, "float", lo=lo, hi=hi, log=is_log)
        elif typ == "int":
            lo_i, hi_i = int(rng[0]), int(rng[1])
            d = _Dim(name, "int", lo=float(lo_i), hi=float(hi_i), count=hi_i - lo_i + 1)
        elif typ == "choice":
            choices = list(rng)
            d = _Dim(name, "cat", count=len(choices), choices=choices)
        else:
            raise ValueError(f"Unknown param type {typ!r} for {name!r}")
        d.parent = None if parent is None else str(parent)
        d.parent_values = None if values is None else list(values)
        dims.append(d)

    # Parents before children so activity can be resolved in one pass.
    by_name = {d.name: d for d in dims}
    ordered: list[_Dim] = []
    placed: set[str] = set()
    remaining = list(dims)
    while remaining:
        progressed = False
        for d in list(remaining):
            if d.parent is None or d.parent in placed or d.parent not in by_name:
                ordered.append(d)
                placed.add(d.name)
                remaining.remove(d)
                progressed = True
        if not progressed:  # cycle: keep declaration order
            ordered.extend(remaining)
            break
    return ordered


class _Parzen:
    """Weighted mixture of product kernels (one per observation) + uniform prior."""

    def __init__(
        self,
        X: np.ndarray,  # (n, Dn) unit-cube numeric values, NaN = inactive
        C: np.ndarray,  # (n, Dc) category indices, -1 = inactive
        weights: np.ndarray,  # (n,)
        sigma: np.ndarray,  # (Dn,)
        cat_eps: float,
        num_is_int: np.ndarray,  # (Dn,) bool
        num_counts: np.ndarray,  # (Dn,) int counts (1 for floats)
        cat_counts: np.ndarray,  # (Dc,)
        prior_weight: float,
    ) -> None:
        n, dn = X.shape
        dc = C.shape[1]
        # Prior component: fully "inactive" => evaluates to the uniform density.
        self.mu = np.vstack([X, np.full((1, dn), np.nan)])
        self.cat = np.vstack([C, np.full((1, dc), -1, dtype=int)])
        w = np.append(np.asarray(weights, dtype=float), max(prior_weight, 0.0))
        self.weight_sum = float(w.sum())
        with np.errstate(divide="ignore"):
            self.log_w = np.log(w / self.weight_sum)
        self.sigma = sigma
        self.is_int = num_is_int
        self.half_bin = np.where(num_is_int, 0.5 / np.maximum(num_counts, 1), 0.0)
        self.num_prior_lp = np.where(
            num_is_int, -np.log(np.maximum(num_counts, 1)), 0.0
        )
        self.cat_counts = cat_counts
        k = np.maximum(cat_counts, 1).astype(float)
        eps = float(np.clip(cat_eps, 1e-6, 1.0))
        self.cat_lp_same = np.log(1.0 - eps * (k - 1.0) / k)
        self.cat_lp_other = np.log(np.maximum(eps / k, 1e-300))
        self.cat_prior_lp = -np.log(k)
        # Truncation normaliser of each numeric component on [0, 1].
        mu0 = np.nan_to_num(self.mu, nan=0.5)
        self.log_z = np.log(
            np.maximum(ndtr((1.0 - mu0) / sigma) - ndtr(-mu0 / sigma), 1e-300)
        )

    # ------------------------------------------------------------------ density
    def component_logpdf(
        self,
        Xq: np.ndarray,
        Cq: np.ndarray,
        mu: np.ndarray | None = None,
        cat: np.ndarray | None = None,
        log_z: np.ndarray | None = None,
    ) -> np.ndarray:
        """(m, K) log kernel density of each query under each component."""
        mu = self.mu if mu is None else mu
        cat = self.cat if cat is None else cat
        log_z = self.log_z if log_z is None else log_z
        m = Xq.shape[0]
        out = np.zeros((m, mu.shape[0]))
        if mu.shape[1]:
            sig = self.sigma
            diff = Xq[:, None, :] - mu[None, :, :]
            z = diff / sig
            lp_float = -0.5 * z * z - _LOG_SQRT_2PI - np.log(sig)
            if self.is_int.any():
                hb = self.half_bin / sig
                mass = ndtr(z + hb) - ndtr(z - hb)
                with np.errstate(divide="ignore"):
                    lp_int = np.where(
                        mass > 1e-12,
                        np.log(np.maximum(mass, 1e-300)),
                        # Deep-tail fallback: density × bin width.
                        -0.5 * z * z - _LOG_SQRT_2PI + np.log(2.0 * hb),
                    )
                lp = np.where(self.is_int, lp_int, lp_float)
            else:
                lp = lp_float
            lp = lp - log_z[None, :, :]
            lp = np.where(np.isnan(mu)[None, :, :], self.num_prior_lp, lp)
            lp = np.where(np.isnan(Xq)[:, None, :], 0.0, lp)
            out += lp.sum(axis=2)
        if cat.shape[1]:
            same = Cq[:, None, :] == cat[None, :, :]
            lp = np.where(same, self.cat_lp_same, self.cat_lp_other)
            lp = np.where((cat < 0)[None, :, :], self.cat_prior_lp, lp)
            lp = np.where((Cq < 0)[:, None, :], 0.0, lp)
            out += lp.sum(axis=2)
        return out

    def log_pdf_unnorm(self, Xq: np.ndarray, Cq: np.ndarray) -> np.ndarray:
        """log Σ_k w_k K_k(x) with *unnormalised* weights (w_k·weight_sum)."""
        m = Xq.shape[0]
        per_row = max(1, self.mu.shape[0] * max(1, Xq.shape[1] + Cq.shape[1]))
        step = max(1, _CHUNK_ELEMS // per_row)
        log_w_un = self.log_w + math.log(self.weight_sum)
        out = np.empty(m)
        for s in range(0, m, step):
            lp = self.component_logpdf(Xq[s : s + step], Cq[s : s + step])
            out[s : s + step] = logsumexp(lp + log_w_un[None, :], axis=1)
        return out

    def log_pdf(self, Xq: np.ndarray, Cq: np.ndarray) -> np.ndarray:
        return self.log_pdf_unnorm(Xq, Cq) - math.log(self.weight_sum)

    def point_logpdf(
        self, Xq: np.ndarray, Cq: np.ndarray, x: np.ndarray, c: np.ndarray
    ):
        """(m,) log kernel density of queries under a single component at (x, c)."""
        mu = x[None, :]
        mu0 = np.nan_to_num(mu, nan=0.5)
        log_z = np.log(
            np.maximum(ndtr((1.0 - mu0) / self.sigma) - ndtr(-mu0 / self.sigma), 1e-300)
        )
        return self.component_logpdf(Xq, Cq, mu=mu, cat=c[None, :], log_z=log_z)[:, 0]

    # ----------------------------------------------------------------- sampling
    def sample(self, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        p = np.exp(self.log_w)
        idx = rng.choice(len(p), size=m, p=p / p.sum())
        mu = self.mu[idx]
        dn = mu.shape[1]
        X = np.empty((m, dn))
        if dn:
            sig = self.sigma[None, :]
            inactive = np.isnan(mu)
            mu0 = np.where(inactive, 0.5, mu)
            pa = ndtr(-mu0 / sig)
            pb = ndtr((1.0 - mu0) / sig)
            r = rng.random((m, dn))
            q = np.clip(pa + r * (pb - pa), 1e-12, 1.0 - 1e-12)
            X = np.clip(mu0 + sig * ndtri(q), 0.0, 1.0)
            X = np.where(inactive, rng.random((m, dn)), X)
        cat = self.cat[idx]
        dc = cat.shape[1]
        Cs = np.empty((m, dc), dtype=int)
        for j in range(dc):
            k = int(self.cat_counts[j])
            base = cat[:, j]
            if k <= 1:
                Cs[:, j] = 0
                continue
            keep = rng.random(m) < np.exp(self.cat_lp_same[j])
            other = rng.integers(0, k - 1, size=m)
            other = np.where(other >= base, other + 1, other)
            col = np.where(keep, base, other)
            Cs[:, j] = np.where(base < 0, rng.integers(0, k, size=m), col)
        return X, Cs


class MultiFidelityTPE:
    """
    Multi-fidelity multivariate TPE with a BOHB-compatible interface
    (``observe`` / ``suggest`` / constraint helpers / ``diagnostics``).

    Args:
        config_space: ``{name: (type, spec[, options])}`` as in :class:`BOHB`.
        gamma: fraction of observations (at the model budget) forming l(x).
        n_ei_candidates: samples drawn from l(x) per proposed configuration.
        min_points_in_model: observations required at a budget before it is
            modelled. Defaults to ``d + 2``.
        random_fraction: probability that a proposal is a pure prior sample
            (keeps the search globally consistent).
        bandwidth_factor: multiplier on the Scott-rule kernel bandwidth.
        min_bandwidth: floor on the (unit-cube) numeric bandwidth.
        prior_weight: weight of the uniform prior component (observations have
            weight ~1 each).
        constant_liar: treat already-selected batch members as pending bad points.
        hard_constraints / soft_constraints: callables ``cfg -> float``; a value
            ``<= 0`` means satisfied.
    """

    def __init__(
        self,
        config_space: dict[str, tuple],
        *,
        gamma: float = 0.15,
        n_ei_candidates: int = 64,
        min_points_in_model: int | None = None,
        random_fraction: float = 0.1,
        bandwidth_factor: float = 1.5,
        min_bandwidth: float = 1e-3,
        prior_weight: float = 1.0,
        constant_liar: bool = True,
        max_good: int | None = 25,
        hard_constraints: list[Callable[[dict[str, Any]], float]] | None = None,
        soft_constraints: list[Callable[[dict[str, Any]], float]] | None = None,
        soft_penalty_weight: float = 1.0,
        constraint_max_attempts: int = 200,
        seed: int | None = None,
        **_ignored: Any,
    ) -> None:
        self.config_space = config_space
        self.dims = _parse_space(config_space)
        self.num_dims = [d for d in self.dims if d.kind != "cat"]
        self.cat_dims = [d for d in self.dims if d.kind == "cat"]
        self._num_col = {d.name: i for i, d in enumerate(self.num_dims)}
        self._cat_col = {d.name: i for i, d in enumerate(self.cat_dims)}
        self._num_is_int = np.array(
            [d.kind == "int" for d in self.num_dims], dtype=bool
        )
        self._num_counts = np.array(
            [d.count if d.kind == "int" else 1 for d in self.num_dims], dtype=int
        )
        self._cat_counts = np.array([d.count for d in self.cat_dims], dtype=int)

        n_dims = len(self.dims)
        self.gamma = float(min(max(gamma, 1e-3), 0.9))
        self.n_ei_candidates = max(1, int(n_ei_candidates))
        self.min_points_in_model = (
            n_dims + 2
            if min_points_in_model is None
            else max(2, int(min_points_in_model))
        )
        self.random_fraction = float(min(max(random_fraction, 0.0), 1.0))
        self.bandwidth_factor = float(bandwidth_factor)
        self.min_bandwidth = float(min_bandwidth)
        self.prior_weight = float(prior_weight)
        self.constant_liar = bool(constant_liar)
        self.max_good = None if max_good is None else max(1, int(max_good))
        self.hard_constraints = list(hard_constraints or [])
        self.soft_constraints = list(soft_constraints or [])
        self.soft_penalty_weight = float(soft_penalty_weight)
        self.constraint_max_attempts = max(1, int(constraint_max_attempts))
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._sobol = (
            qmc.Sobol(d=max(1, n_dims), scramble=True, seed=self._rng)
            if qmc is not None
            else None
        )

        # Observations, stored encoded for fast model fitting.
        self.observations: list[tuple[dict[str, Any], float, float | None]] = []
        self._X: list[np.ndarray] = []
        self._C: list[np.ndarray] = []
        self._keys: set[str] = set()
        self._last_model_budget: float | None = None
        self._n_model_proposals = 0
        self._n_random_proposals = 0

    # ---------------------------------------------------------------- encoding
    def _encode(self, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        x = np.full(len(self.num_dims), np.nan)
        c = np.full(len(self.cat_dims), -1, dtype=int)
        for i, d in enumerate(self.num_dims):
            if d.name not in config or config[d.name] is None:
                continue
            v = float(config[d.name])
            if d.kind == "int":
                x[i] = (round(v) - d.lo + 0.5) / d.count
            else:
                if d.log:
                    v = math.log10(max(v, 1e-300))
                x[i] = (v - d.lo) / (d.hi - d.lo) if d.hi > d.lo else 0.5
        for j, d in enumerate(self.cat_dims):
            if d.name not in config:
                continue
            try:
                c[j] = d.choices.index(config[d.name])  # type: ignore[union-attr]
            except ValueError:
                c[j] = -1
        return np.clip(x, 0.0, 1.0), c

    def _decode_num(self, d: _Dim, u: float) -> Any:
        if d.kind == "int":
            k = min(max(int(math.floor(u * d.count)), 0), d.count - 1)
            return int(d.lo) + k
        v = d.lo + u * (d.hi - d.lo)
        if d.log:
            lo, hi = 10.0**d.lo, 10.0**d.hi
            return float(min(max(10.0**v, lo), hi))
        return float(min(max(v, d.lo), d.hi))

    def _decode(self, x: np.ndarray, c: np.ndarray) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for d in self.dims:
            if d.kind == "cat":
                j = self._cat_col[d.name]
                if c[j] >= 0:
                    cfg[d.name] = d.choices[int(c[j])]  # type: ignore[index]
            else:
                i = self._num_col[d.name]
                if not np.isnan(x[i]):
                    cfg[d.name] = self._decode_num(d, float(x[i]))
        return cfg

    def _snap_ints(self, X: np.ndarray) -> np.ndarray:
        if not self._num_is_int.any():
            return X
        counts = self._num_counts[None, :]
        snapped = (np.clip(np.floor(X * counts), 0, counts - 1) + 0.5) / counts
        return np.where(self._num_is_int[None, :], snapped, X)

    def _apply_conditions(self, X: np.ndarray, C: np.ndarray) -> None:
        """Mark children of unsatisfied conditions inactive (in place)."""
        if not any(d.parent for d in self.dims):
            return
        m = X.shape[0]
        active: dict[str, np.ndarray] = {}
        for d in self.dims:
            ok = np.ones(m, dtype=bool)
            if d.parent is not None and d.parent in active:
                ok &= active[d.parent]
                if d.parent_values is not None:
                    ok &= self._parent_matches(d.parent, d.parent_values, X, C)
            active[d.name] = ok
            if d.kind == "cat":
                C[~ok, self._cat_col[d.name]] = -1
            else:
                X[~ok, self._num_col[d.name]] = np.nan

    def _parent_matches(
        self, parent: str, values: list[Any], X: np.ndarray, C: np.ndarray
    ) -> np.ndarray:
        pd = next(d for d in self.dims if d.name == parent)
        if pd.kind == "cat":
            idx = [i for i, ch in enumerate(pd.choices or []) if ch in values]
            return np.isin(C[:, self._cat_col[parent]], idx)
        col = X[:, self._num_col[parent]]
        decoded = np.array(
            [np.nan if np.isnan(u) else self._decode_num(pd, float(u)) for u in col],
            dtype=object,
        )
        return np.array([v in values for v in decoded], dtype=bool)

    # ------------------------------------------------------------------- data
    def observe(
        self, config: dict[str, Any], loss: float, budget: float | None = None
    ) -> None:
        loss_f = float(loss)
        if not math.isfinite(loss_f):
            return
        b = None if budget is None else float(f"{float(budget):.9g}")
        self.observations.append((dict(config), loss_f, b))
        x, c = self._encode(config)
        self._X.append(x)
        self._C.append(c)
        self._keys.add(_canonical_config_key(config))

    def _model_indices(self) -> tuple[np.ndarray, float | None]:
        """Indices of observations at the largest sufficiently-populated budget."""
        if not self.observations:
            return np.zeros(0, dtype=int), None
        budgets = np.array(
            [np.inf if b is None else b for _, _, b in self.observations], dtype=float
        )
        for b in np.unique(budgets)[::-1]:
            idx = np.flatnonzero(budgets == b)
            if idx.size >= self.min_points_in_model:
                return idx, (None if np.isinf(b) else float(b))
        return np.zeros(0, dtype=int), None

    def _bandwidth(self, X: np.ndarray, n_eff: int) -> np.ndarray:
        dn = X.shape[1]
        if dn == 0:
            return np.zeros(0)
        scott = max(n_eff, 1) ** (-1.0 / (dn + 4))
        with np.errstate(invalid="ignore"):
            std = np.nanstd(X, axis=0) if X.shape[0] > 1 else np.full(dn, np.nan)
        # Unit-cube std of a uniform is ~0.29; blend towards it when data is scarce.
        std = np.where(np.isfinite(std), std, 0.29)
        std = np.maximum(std, 0.29 / math.sqrt(max(n_eff, 1)))
        sigma = self.bandwidth_factor * 1.06 * std * scott
        floor = np.maximum(
            self.min_bandwidth, np.where(self._num_is_int, self.half_bin_floor, 0.0)
        )
        # Magic clip (Bergstra et al.): never narrower than range / min(100, n + 1).
        floor = np.maximum(floor, 1.0 / min(100.0, n_eff + 1.0))
        return np.clip(sigma, floor, 1.0)

    @property
    def half_bin_floor(self) -> np.ndarray:
        return 0.5 / np.maximum(self._num_counts, 1)

    @staticmethod
    def _bad_weights(n: int) -> np.ndarray:
        # Optuna default: keep the latest 25 at full weight, ramp older ones down.
        if n < 25:
            return np.ones(n)
        ramp = np.linspace(1.0 / n, 1.0, num=n - 25)
        return np.concatenate([ramp, np.ones(25)])

    def _fit(self, idx: np.ndarray) -> tuple[_Parzen, _Parzen]:
        losses = np.array([self.observations[i][1] for i in idx], dtype=float)
        order = np.argsort(losses, kind="stable")
        n = idx.size
        n_good = int(math.ceil(self.gamma * n))
        if self.max_good is not None:
            n_good = min(n_good, self.max_good)
        n_good = min(max(n_good, 1), n - 1)
        good = np.sort(idx[order[:n_good]])  # chronological order for weights
        bad = np.sort(idx[order[n_good:]])
        X = np.asarray(self._X)
        C = np.asarray(self._C, dtype=int).reshape(len(self._C), len(self.cat_dims))
        cat_eps = float(np.clip(0.4 * n ** (-1.0 / (len(self.dims) + 4)), 0.02, 0.5))

        def build(sel: np.ndarray, w: np.ndarray) -> _Parzen:
            return _Parzen(
                X[sel],
                C[sel],
                w,
                self._bandwidth(X[sel], sel.size),
                cat_eps,
                self._num_is_int,
                self._num_counts,
                self._cat_counts,
                self.prior_weight,
            )

        return build(good, np.ones(good.size)), build(bad, self._bad_weights(bad.size))

    # --------------------------------------------------------------- sampling
    def _random_encoded(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        d = len(self.dims)
        if self._sobol is not None and m > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # non-power-of-2 draws
                U = self._sobol.random(m)
        else:
            U = self._rng.random((m, max(d, 1)))
        X = np.empty((m, len(self.num_dims)))
        C = np.empty((m, len(self.cat_dims)), dtype=int)
        for k, dim in enumerate(self.dims):
            u = U[:, k]
            if dim.kind == "cat":
                C[:, self._cat_col[dim.name]] = np.minimum(
                    (u * dim.count).astype(int), dim.count - 1
                )
            else:
                X[:, self._num_col[dim.name]] = u
        X = self._snap_ints(X)
        self._apply_conditions(X, C)
        return X, C

    def _sample_prior(self) -> dict[str, Any]:
        X, C = self._random_encoded(1)
        return self._decode(X[0], C[0])

    def _hard_constraints_satisfied(self, config: dict[str, Any]) -> bool:
        return all(
            self._eval_constraint(fn, config) <= 0 for fn in self.hard_constraints
        )

    def _soft_constraint_violation(self, config: dict[str, Any]) -> float:
        return float(
            sum(
                max(0.0, self._eval_constraint(fn, config))
                for fn in self.soft_constraints
            )
        )

    @staticmethod
    def _eval_constraint(fn: Callable, config: dict[str, Any]) -> float:
        try:
            return float(fn(config))
        except Exception:
            return float("inf")

    def _random_feasible(self, taken: set[str]) -> dict[str, Any]:
        cfg = self._sample_prior()
        for _ in range(self.constraint_max_attempts):
            key = _canonical_config_key(cfg)
            if self._hard_constraints_satisfied(cfg) and key not in taken:
                break
            cfg = self._sample_prior()
        return cfg

    def suggest(
        self,
        n_candidates: int = 1,
        budget: float | None = None,
        return_scores: bool | str = False,
        **_ignored: Any,
    ):
        """
        Propose ``n_candidates`` configurations.

        ``return_scores`` truthy -> ``(configs, scores)`` with one acquisition
        score (log l(x) - log g(x); ``-inf`` for random proposals) per config.
        """
        n = max(1, int(n_candidates))
        idx, model_budget = self._model_indices()
        self._last_model_budget = model_budget
        taken = set(self._keys)
        configs: list[dict[str, Any]] = []
        scores: list[float] = []

        if idx.size < max(self.min_points_in_model, 2):
            X, C = self._random_encoded(n * 4)
            for i in range(X.shape[0]):
                if len(configs) >= n:
                    break
                cfg = self._decode(X[i], C[i])
                key = _canonical_config_key(cfg)
                if key in taken or not self._hard_constraints_satisfied(cfg):
                    continue
                taken.add(key)
                configs.append(cfg)
                scores.append(float("-inf"))
            while len(configs) < n:
                cfg = self._random_feasible(taken)
                taken.add(_canonical_config_key(cfg))
                configs.append(cfg)
                scores.append(float("-inf"))
            self._n_random_proposals += n
            return (configs, scores) if return_scores else configs

        l_model, g_model = self._fit(idx)
        k = self.n_ei_candidates
        Xc, Cc = l_model.sample(n * k, self._rng)
        Xc = self._snap_ints(Xc)
        self._apply_conditions(Xc, Cc)
        log_l = l_model.log_pdf(Xc, Cc)
        log_g_un = g_model.log_pdf_unnorm(Xc, Cc)
        g_weight = g_model.weight_sum
        liar_w = 1.0

        is_random = self._rng.random(n) < self.random_fraction
        for i in range(n):
            if is_random[i]:
                cfg = self._random_feasible(taken)
                score = float("-inf")
                x_sel, c_sel = self._encode(cfg)
                self._n_random_proposals += 1
            else:
                sl = slice(i * k, (i + 1) * k)
                acq = log_l[sl] - (log_g_un[sl] - math.log(g_weight))
                cfg = None
                for j in np.argsort(-acq):
                    cand = self._decode(Xc[sl][j], Cc[sl][j])
                    key = _canonical_config_key(cand)
                    if key in taken or not self._hard_constraints_satisfied(cand):
                        continue
                    cfg, score = cand, float(acq[j])
                    x_sel, c_sel = Xc[sl][j], Cc[sl][j]
                    break
                if cfg is None:
                    cfg = self._random_feasible(taken)
                    score = float("-inf")
                    x_sel, c_sel = self._encode(cfg)
                self._n_model_proposals += 1

            taken.add(_canonical_config_key(cfg))
            configs.append(cfg)
            scores.append(score)

            if self.constant_liar and i + 1 < n:
                rest = slice((i + 1) * k, None)
                kern = g_model.point_logpdf(Xc[rest], Cc[rest], x_sel, c_sel)
                log_g_un[rest] = np.logaddexp(log_g_un[rest], math.log(liar_w) + kern)
                g_weight += liar_w

        return (configs, scores) if return_scores else configs

    def diagnostics(self) -> dict[str, Any]:
        return {
            "sampler": "mftpe",
            "n_observations": len(self.observations),
            "model_budget": self._last_model_budget,
            "model_proposals": self._n_model_proposals,
            "random_proposals": self._n_random_proposals,
            "trust_region_enabled": False,
        }
