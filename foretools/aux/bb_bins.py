import warnings
from collections.abc import Iterable
from typing import Optional, Union

import numpy as np

warnings.filterwarnings("ignore")
ArrayLike = Union[np.ndarray, Iterable[float]]


# -----------------------------
# Base + Fitness subclasses
# -----------------------------
def _fitness_args_of(func):
    return func.__code__.co_varnames[: func.__code__.co_argcount]


class FitnessFunc:
    def __init__(
        self,
        p0: float = 0.05,
        gamma: Optional[float] = None,
        ncp_prior: Optional[float] = None,
    ) -> None:
        self.p0 = p0
        self.gamma = gamma
        self.ncp_prior = ncp_prior

    def validate_input(
        self,
        t: ArrayLike,
        x: Optional[ArrayLike] = None,
        sigma: Optional[Union[ArrayLike, float]] = None,
    ):
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or t.size == 0:
            raise ValueError("t must be non-empty 1D array")
        unq_t, unq_ind, unq_inv = np.unique(t, return_index=True, return_inverse=True)
        if x is None:
            if sigma is not None:
                raise ValueError(
                    "If sigma is provided, x must be provided for 'measures'."
                )
            sigma = 1.0
            if unq_t.size == t.size:
                x = np.ones_like(unq_t)
            else:
                x = np.bincount(unq_inv).astype(float)
            t = unq_t
        else:
            x = np.asarray(x, dtype=float)
            if x.shape not in [(), (1,), (t.size,)]:
                raise ValueError("x shape must be scalar or match t")
            if unq_t.size != t.size:
                raise ValueError(
                    "Repeated values in t not supported when x is provided"
                )
            t = unq_t
            x = x[unq_ind] if x.shape == (t.size,) else np.full_like(t, float(x))
        if sigma is None:
            sigma = 1.0
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape not in [(), (1,), (t.size,)]:
            raise ValueError("sigma shape must be scalar or match t/x")
        if sigma.shape != (t.size,):
            sigma = np.full_like(t, float(sigma))
        return t, x, sigma

    def p0_prior(self, N: int) -> float:
        return 4.0 - np.log(73.53 * self.p0 * (N**-0.478))

    def compute_ncp_prior(self, N: int) -> float:
        if self.ncp_prior is not None:
            return float(self.ncp_prior)
        if self.gamma is not None:
            return -np.log(float(self.gamma))
        if self.p0 is not None:
            return float(self.p0_prior(N))
        raise ValueError("Cannot compute ncp_prior: set p0 or gamma or ncp_prior.")

    @property
    def _fitness_args(self):
        return _fitness_args_of(self.fitness)

    def fitness(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def fit(
        self,
        t: ArrayLike,
        x: Optional[ArrayLike] = None,
        sigma: Optional[Union[ArrayLike, float]] = None,
    ) -> np.ndarray:
        t, x, sigma = self.validate_input(t, x, sigma)
        if "a_k" in self._fitness_args:
            a_raw = 1.0 / (sigma * sigma)
        if "b_k" in self._fitness_args:
            b_raw = x / (sigma * sigma)
        if "c_k" in self._fitness_args:
            c_raw = (x * x) / (sigma * sigma)
        edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
        block_len_from_right = t[-1] - edges
        N = t.size
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)
        ncp_prior = self.compute_ncp_prior(N)
        for R in range(N):
            kw = {}
            if "T_k" in self._fitness_args:
                kw["T_k"] = block_len_from_right[: R + 1] - block_len_from_right[R + 1]
            if "N_k" in self._fitness_args:
                kw["N_k"] = np.cumsum(x[: R + 1][::-1])[::-1]
            if "a_k" in self._fitness_args:
                kw["a_k"] = 0.5 * np.cumsum(a_raw[: R + 1][::-1])[::-1]
            if "b_k" in self._fitness_args:
                kw["b_k"] = -np.cumsum(b_raw[: R + 1][::-1])[::-1]
            if "c_k" in self._fitness_args:
                kw["c_k"] = 0.5 * np.cumsum(c_raw[: R + 1][::-1])[::-1]
            fit_vec = self.fitness(**kw)
            A_R = fit_vec - ncp_prior
            if R > 0:
                A_R[1:] += best[:R]
            i_max = int(np.argmax(A_R))
            last[R] = i_max
            best[R] = A_R[i_max]
        cps = []
        ind = N
        while ind > 0:
            cps.append(ind)
            ind = last[ind - 1]
            if ind == 0:
                cps.append(0)
                break
        cps = np.array(cps[::-1], dtype=int)
        return edges[cps]


class Events(FitnessFunc):
    def fitness(self, N_k: np.ndarray, T_k: np.ndarray) -> np.ndarray:
        out = np.zeros_like(N_k, dtype=float)
        mask = (N_k > 0) & (T_k > 0)
        rate = np.divide(N_k, T_k, out=np.zeros_like(N_k, dtype=float), where=mask)
        with np.errstate(invalid="ignore"):
            ln_rate = np.log(rate)
        np.multiply(N_k, ln_rate, out=out, where=mask)
        out[mask] -= N_k[mask]
        out[~mask] = -np.inf
        return out

    def validate_input(self, t, x, sigma):
        t, x, sigma = super().validate_input(t, x, sigma)
        if x is not None and (np.any(x < 0) or np.any(x % 1 != 0)):
            raise ValueError("For 'events', x must be non-negative integer counts.")
        return t, x, sigma


class RegularEvents(FitnessFunc):
    def __init__(self, dt: float, **kwargs) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive for 'regular_events'.")
        self.dt = float(dt)
        super().__init__(**kwargs)

    def validate_input(self, t, x=None, sigma=None):
        t, x, sigma = super().validate_input(t, x, sigma)
        if x is not None and not np.all((x == 0) | (x == 1)):
            raise ValueError("'regular_events' requires x in {0,1}.")
        return t, x, sigma

    def fitness(self, T_k: np.ndarray, N_k: np.ndarray) -> np.ndarray:
        M_k = T_k / self.dt
        eps = np.finfo(float).tiny
        q = np.clip(N_k / (M_k + eps), eps, 1 - eps)
        return N_k * np.log(q) + (M_k - N_k) * np.log(1 - q)


class PointMeasures(FitnessFunc):
    def fitness(self, a_k: np.ndarray, b_k: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            out = (b_k * b_k) / (4.0 * a_k)
        out[~np.isfinite(out)] = -np.inf
        return out

    def validate_input(self, t, x, sigma):
        if x is None:
            raise ValueError("'measures' requires x (and optionally sigma).")
        return super().validate_input(t, x, sigma)


# -----------------------------
# Encapsulation
# -----------------------------
class BayesianBlocks:
    """Encapsulated Bayesian Blocks API."""

    FITNESS = {
        "events": Events,
        "regular_events": RegularEvents,
        "measures": PointMeasures,
    }

    def __init__(
        self,
        fitness: Union[str, FitnessFunc] = "events",
        verbose: bool = False,
        **kwargs,
    ):
        self.fitness = fitness
        self.kwargs = kwargs
        self.verbose = verbose
        self.fitfunc: Optional[FitnessFunc] = None

    def _make_fitfunc(self):
        fitcls_or_obj = self.FITNESS.get(self.fitness, self.fitness)
        if isinstance(fitcls_or_obj, type) and issubclass(fitcls_or_obj, FitnessFunc):
            self.fitfunc = fitcls_or_obj(**self.kwargs)
        elif isinstance(fitcls_or_obj, FitnessFunc):
            self.fitfunc = fitcls_or_obj
        else:
            raise ValueError(
                "fitness must be 'events', 'regular_events', 'measures', a FitnessFunc subclass, or instance."
            )

    def fit_edges(
        self,
        t: ArrayLike,
        x: Optional[ArrayLike] = None,
        sigma: Optional[Union[ArrayLike, float]] = None,
    ) -> np.ndarray:
        if self.fitfunc is None:
            self._make_fitfunc()
        return self.fitfunc.fit(t, x, sigma)

    def fit_bins(
        self,
        t: ArrayLike,
        x: Optional[ArrayLike] = None,
        sigma: Optional[Union[ArrayLike, float]] = None,
        max_bins: int = 100,
        fallback: str = "sturges",
    ) -> int:
        """Return integer number of bins estimated by Bayesian Blocks."""
        try:
            edges = self.fit_edges(t, x, sigma)
            num_bins = len(edges) - 1
            return max(1, min(num_bins, max_bins))
        except Exception as e:
            if self.verbose:
                print(f"Warning: Bayesian Blocks failed with {self.fitness}: {e}")
            # Fallback: classic Sturges
            return int(np.ceil(np.log2(len(t)) + 1))
