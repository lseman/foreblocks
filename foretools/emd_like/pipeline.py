from __future__ import annotations

import gc
import inspect
import math
import time
import warnings
from typing import Any

import numpy as np
import optuna
from numba import njit
from scipy.signal import decimate

from .common import (
    BoundaryHandler,
    FFTWManager,
    ModeProcessor,
    SignalAnalyzer,
    _energy,
    box_counting_dimension,
)
from .config import HierarchicalParameters, VMDParameters
from .core import VMDCore, refine_modes_cross_nn, refine_modes_nn
from .emd import EMDVariants

# Notebook detection (Optuna progress bar can be annoying in some envs)
try:
    from IPython import get_ipython

    IS_NOTEBOOK = get_ipython() is not None
except Exception:
    IS_NOTEBOOK = False
from scipy.stats import kurtosis


@njit(cache=True)
def _coarse_grain_mean_numba(x: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return x
    n = (x.size // scale) * scale
    if n < scale:
        return x.copy()
    out_n = n // scale
    out = np.empty(out_n, dtype=np.float64)
    for i in range(out_n):
        s = 0.0
        start = i * scale
        for j in range(scale):
            s += x[start + j]
        out[i] = s / float(scale)
    return out


@njit(cache=True)
def _permutation_entropy_numba(
    s: np.ndarray, m: int, delay: int, normalize: int
) -> float:
    n = s.size
    windows = n - (m - 1) * delay
    if windows < 2 or m < 2 or delay < 1:
        return 0.0

    # Canonical permutation-pattern encoding (Lehmer rank), bins = m!
    fact = np.empty(m + 1, dtype=np.int64)
    fact[0] = 1
    for i in range(1, m + 1):
        fact[i] = fact[i - 1] * i
    bins = int(fact[m])
    counts = np.zeros(bins, dtype=np.int64)
    idx = np.empty(m, dtype=np.int64)

    for i in range(windows):
        for a in range(m):
            idx[a] = a
        for a in range(1, m):
            key_idx = idx[a]
            key_val = s[i + key_idx * delay]
            b = a - 1
            while b >= 0:
                cur_idx = idx[b]
                cur_val = s[i + cur_idx * delay]
                if (cur_val > key_val) or (cur_val == key_val and cur_idx > key_idx):
                    idx[b + 1] = idx[b]
                    b -= 1
                else:
                    break
            idx[b + 1] = key_idx

        code = 0
        for a in range(m):
            smaller = 0
            ia = idx[a]
            for b in range(a + 1, m):
                if idx[b] < ia:
                    smaller += 1
            code += smaller * int(fact[m - 1 - a])
        counts[code] += 1

    total = float(windows)
    if total <= 0.0:
        return 0.0
    h = 0.0
    eps = 1e-12
    for i in range(counts.size):
        c = counts[i]
        if c > 0:
            p = float(c) / total
            h -= p * math.log(p + eps)
    if normalize == 0:
        return h
    hmax = math.log(float(fact[m]) + eps)
    return h / (hmax + eps)


@njit(cache=True)
def _dispersion_entropy_numba(
    s: np.ndarray, m: int, c: int, delay: int, normalize: int
) -> float:
    n = s.size
    windows = n - (m - 1) * delay
    if windows < 2 or m < 2 or c < 2 or delay < 1:
        return 0.0

    mu = np.mean(s)
    std = np.std(s)
    if std < 1e-12:
        return 0.0

    bins = 1
    for _ in range(m):
        bins *= c
    counts = np.zeros(bins, dtype=np.int64)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for i in range(windows):
        code = 0
        mul = 1
        for j in range(m):
            v = s[i + j * delay]
            z = (v - mu) / (std + 1e-12)
            y = 0.5 * (1.0 + math.erf(z * inv_sqrt2))
            cls = int(y * c)
            if cls < 0:
                cls = 0
            elif cls >= c:
                cls = c - 1
            code += cls * mul
            mul *= c
        counts[code] += 1

    total = float(windows)
    if total <= 0.0:
        return 0.0
    h = 0.0
    eps = 1e-12
    for i in range(counts.size):
        cnt = counts[i]
        if cnt > 0:
            p = float(cnt) / total
            h -= p * math.log(p + eps)
    if normalize == 0:
        return h
    hmax = math.log(float(bins) + eps)
    return h / (hmax + eps)


class VMDOptimizer:
    """
    Single pipeline refactor:
      _evaluate_candidate() is the only place that runs:
        decompose -> postprocess -> cost
      It is reused for:
        - Optuna full objective (K, alpha)
        - Optuna alpha-only objective
        - Penalized K selection
        - Final solve
    """

    def __init__(self, fftw: FFTWManager):
        self.fftw = fftw
        self.core = VMDCore(fftw)
        self.proc = ModeProcessor()
        self.analyzer = SignalAnalyzer()
        self._core_decompose_params = set(
            inspect.signature(self.core.decompose).parameters
        )
        self._core_chirp_params = (
            set(inspect.signature(self.core.decompose_chirp).parameters)
            if hasattr(self.core, "decompose_chirp")
            else set()
        )
        self._core_mv_params = set(
            inspect.signature(self.core.decompose_multivariate).parameters
        )

        # cache: (K, alpha_bin) -> (modes_list, freqs_list, cost)
        self._cache: dict[
            tuple[int, int], tuple[list[np.ndarray], list[float], float]
        ] = {}
        self._cache_mv: dict[
            tuple[int, int], tuple[list[np.ndarray], list[float], float]
        ] = {}

    def _call_core_decompose(
        self,
        *,
        signal: np.ndarray,
        alpha: float,
        fs: float,
        K: int,
        init: int,
        p: VMDParameters,
        precomp: dict[str, Any] | None,
        trial: optuna.Trial | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = self._core_decompose_params
        use_if_tracking = bool(getattr(p, "if_tracking", False))
        kwargs: dict[str, Any] = {
            "signal": signal,
            "alpha": float(alpha),
            "tau": float(p.tau),
            "K": int(K),
            "DC": int(p.DC),
            "init": int(init),
            "random_seed": getattr(p, "random_seed", None),
            "tol": float(p.tol),
            "max_iter": int(p.max_iter),
            "fs": float(fs),
            "precomputed_fft": (None if use_if_tracking else precomp),
            "trial": trial,
            "enforce_uncorrelated": bool(p.enforce_uncorrelated),
            "corr_rho": float(p.corr_rho),
            "corr_update_every": int(p.corr_update_every),
            "corr_ema": float(p.corr_ema),
            "adaptive_alpha": bool(p.adaptive_alpha),
            "adaptive_alpha_start_iter": int(p.adaptive_alpha_start_iter),
            "adaptive_alpha_update_every": int(p.adaptive_alpha_update_every),
            "adaptive_alpha_lr": float(p.adaptive_alpha_lr),
            "adaptive_alpha_min_scale": float(p.adaptive_alpha_min_scale),
            "adaptive_alpha_max_scale": float(p.adaptive_alpha_max_scale),
            "adaptive_alpha_skip_dc": bool(p.adaptive_alpha_skip_dc),
            "omega_momentum": float(p.omega_momentum),
            "omega_shrinkage": float(p.omega_shrinkage),
            "omega_max_step": float(p.omega_max_step),
            "admm_over_relax": float(getattr(p, "admm_over_relax", 1.0)),
            # Backward/forward-compatible optional knobs.
            "if_window_size": int(getattr(p, "if_window_size", 256)),
            "if_hop_size": int(getattr(p, "if_hop_size", 128)),
            "if_center_smooth": float(getattr(p, "if_center_smooth", 0.85)),
            "omega_tol": (
                float(getattr(p, "tol_omega"))
                if getattr(p, "tol_omega", None) is not None
                else 1e-8
            ),
            "use_anderson": bool(getattr(p, "use_anderson", False)),
            "anderson_m": int(getattr(p, "anderson_m", 5)),
            "gram_schmidt_every": int(getattr(p, "gram_schmidt_every", 0)),
            "fft_backend": str(getattr(p, "fft_backend", "fftw")),
            "fft_device": str(getattr(p, "fft_device", "auto")),
        }

        # Route IF/chirp requests to decompose_chirp when available.
        if use_if_tracking and hasattr(self.core, "decompose_chirp"):
            cparams = self._core_chirp_params
            chirp_kwargs: dict[str, Any] = {
                "signal": signal,
                "alpha": float(alpha),
                "tau": float(p.tau),
                "K": int(K),
                "DC": int(p.DC),
                "init": int(init),
                "random_seed": getattr(p, "random_seed", None),
                "tol": float(p.tol),
                "max_iter": int(p.max_iter),
                "fs": float(fs),
                "admm_over_relax": float(getattr(p, "admm_over_relax", 1.0)),
                "if_window_size": int(getattr(p, "if_window_size", 256)),
                "if_hop_size": int(getattr(p, "if_hop_size", 128)),
                "if_center_smooth": float(getattr(p, "if_center_smooth", 0.85)),
                "if_update_step": float(getattr(p, "if_update_step", 1.0)),
                "vncmd_min_track_gap": getattr(p, "vncmd_min_track_gap", None),
                "vncmd_envelope_ridge_scale": float(
                    getattr(p, "vncmd_envelope_ridge_scale", 0.0)
                ),
                "boundary_method": str(p.boundary_method),
                "use_soft_junction": bool(p.use_soft_junction),
                "window_alpha": p.window_alpha,
                "use_anderson": bool(getattr(p, "use_anderson", False)),
                "omega_tol": (
                    float(getattr(p, "tol_omega"))
                    if getattr(p, "tol_omega", None) is not None
                    else 1e-8
                ),
                "trial": trial,
                "fft_backend": str(getattr(p, "fft_backend", "fftw")),
                "fft_device": str(getattr(p, "fft_device", "auto")),
            }
            filtered = (
                chirp_kwargs
                if "kwargs" in cparams
                else {k: v for k, v in chirp_kwargs.items() if k in cparams}
            )
            out = self.core.decompose_chirp(**filtered)
            if isinstance(out, tuple) and len(out) >= 3:
                return out[0], out[1], out[2]
            raise RuntimeError("decompose_chirp returned unexpected output format")

        if ("warm_start_state" in params) and (
            getattr(p, "warm_start_u_hat", None) is not None
            or getattr(p, "warm_start_omega", None) is not None
        ):
            ws: dict[str, Any] = {}
            if getattr(p, "warm_start_u_hat", None) is not None:
                ws["u_hat"] = np.asarray(p.warm_start_u_hat, dtype=np.complex128)
            if getattr(p, "warm_start_omega", None) is not None:
                ws["omega"] = np.asarray(p.warm_start_omega, dtype=np.float64)
            if "u_hat" in ws:
                T_ws = int(ws["u_hat"].shape[0])
            elif precomp is not None and "T" in precomp:
                T_ws = int(precomp["T"])
            else:
                T_ws = int(signal.size)
            ws["lam"] = np.zeros(T_ws, dtype=np.complex128)
            kwargs["warm_start_state"] = ws
            kwargs["init"] = 5

        filtered = {k: v for k, v in kwargs.items() if k in params}
        return self.core.decompose(**filtered)

    def _call_core_decompose_multivariate(
        self,
        *,
        signals: np.ndarray,
        fs: float,
        p: VMDParameters,
        K: int | None = None,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        params = self._core_mv_params
        kwargs: dict[str, Any] = {
            "signals": signals,
            "alpha": float(alpha if alpha is not None else p.alpha_min),
            "tau": float(p.tau),
            "K": int(K if K is not None else p.max_K),
            "DC": int(p.DC),
            "init": int(p.init),
            "random_seed": getattr(p, "random_seed", None),
            "tol": float(p.tol),
            "max_iter": int(p.max_iter),
            "boundary_method": str(p.boundary_method),
            "use_soft_junction": bool(p.use_soft_junction),
            "window_alpha": p.window_alpha,
            "fs": float(fs),
            "admm_over_relax": float(getattr(p, "admm_over_relax", 1.0)),
            "omega_momentum": float(p.omega_momentum),
            "omega_shrinkage": float(p.omega_shrinkage),
            "omega_max_step": float(p.omega_max_step),
            "omega_tol": (
                float(getattr(p, "tol_omega"))
                if getattr(p, "tol_omega", None) is not None
                else 1e-8
            ),
            "enforce_uncorrelated": bool(p.enforce_uncorrelated),
            "use_anderson": bool(getattr(p, "use_anderson", False)),
            "anderson_m": int(getattr(p, "anderson_m", 5)),
            "fft_backend": str(getattr(p, "fft_backend", "fftw")),
            "fft_device": str(getattr(p, "fft_device", "auto")),
        }
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return self.core.decompose_multivariate(**filtered)

    # -----------------------------
    # Shared helpers
    # -----------------------------
    @staticmethod
    def _alpha_bin(alpha: float, log_digits: int = 5) -> int:
        """
        Alpha cache key in log-space.
        Preserves relative precision across scales and avoids coarse linear bins.
        """
        a = float(alpha)
        if not np.isfinite(a) or a <= 0.0:
            return -1
        scale = float(10 ** int(log_digits))
        return int(np.round(np.log(a) * scale))

    @staticmethod
    def _kurtosis_reward(
        modes_list: list[np.ndarray], high_freq_weight: float = 0.6
    ) -> float:
        if not modes_list:
            return 0.0
        kurts = [float(kurtosis(m, fisher=True, nan_policy="omit")) for m in modes_list]
        if len(kurts) < 2:
            return float(np.mean(kurts))
        weights = np.linspace(0.4, 1.0, len(kurts))
        weighted_kurt = np.average(kurts, weights=weights)
        return float(weighted_kurt * high_freq_weight)

    def _postprocess_modes(
        self,
        modes: np.ndarray,
        signal: np.ndarray,
        fs: float,
        p: VMDParameters,
    ) -> tuple[list[np.ndarray], list[float]]:
        x = np.asarray(signal, dtype=np.float64)
        total_E = _energy(x) + 1e-12

        keep: list[np.ndarray] = []
        for k in range(modes.shape[0]):
            if _energy(modes[k]) / total_E > float(p.mode_energy_floor):
                keep.append(np.asarray(modes[k], dtype=np.float64))

        if not keep:
            return [], []

        merged = self.proc.merge_similar_modes(
            keep, fs, freq_tol=float(p.merge_freq_tol)
        )
        sorted_modes, sorted_freqs = self.proc.sort_modes_by_frequency(
            merged, fs, low_to_high=True
        )
        return sorted_modes, sorted_freqs

    def _penalized_K_score(
        self,
        signal: np.ndarray,
        fs: float,
        raw_modes: np.ndarray,
        p: VMDParameters,
        K: int,
    ) -> float:
        x = np.asarray(signal, dtype=np.float64)
        recon = np.sum(raw_modes, axis=0) if raw_modes.size else 0.0
        err = float(np.sum((x - recon) ** 2)) / (float(np.sum(x**2)) + 1e-12)

        dom = np.sort(
            np.array(
                [
                    self.proc.dominant_frequency(raw_modes[k], fs)
                    for k in range(raw_modes.shape[0])
                ],
                dtype=np.float64,
            )
        )
        if dom.size > 1:
            gaps = np.diff(dom)
            overlap = float(np.mean(np.exp(-gaps / (np.median(gaps) + 1e-12))))
        else:
            overlap = 0.0

        return float(
            err + float(p.k_penalty_lambda) * float(K) + float(p.k_overlap_mu) * overlap
        )

    @staticmethod
    def _permutation_entropy(
        x: np.ndarray, m: int = 3, delay: int = 1, normalize: bool = True
    ) -> float:
        s = np.asarray(x, dtype=np.float64)
        return float(
            _permutation_entropy_numba(
                s, int(m), int(delay), 1 if bool(normalize) else 0
            )
        )

    @staticmethod
    def _dispersion_entropy(
        x: np.ndarray,
        m: int = 3,
        c: int = 6,
        delay: int = 1,
        normalize: bool = True,
    ) -> float:
        s = np.asarray(x, dtype=np.float64)
        return float(
            _dispersion_entropy_numba(
                s, int(m), int(c), int(delay), 1 if bool(normalize) else 0
            )
        )

    def _estimate_alpha_entropy(self, signal: np.ndarray, p: VMDParameters) -> float:
        m = int(max(2, p.entropy_embed_dim))
        d = int(max(1, p.entropy_delay))
        c = int(max(2, p.entropy_classes))
        pe = self._permutation_entropy(signal, m=m, delay=d, normalize=True)
        de = self._dispersion_entropy(signal, m=m, c=c, delay=d, normalize=True)
        h = float(p.entropy_weight_pe * pe + p.entropy_weight_de * de)
        h = float(np.clip(h, 0.0, 1.0))
        # Higher complexity -> higher alpha for sharper band separation.
        alpha = float(p.alpha_min + h * (p.alpha_max - p.alpha_min))
        return float(np.clip(alpha, p.alpha_min, p.alpha_max))

    @staticmethod
    def _freq_overlap_penalty(freqs: list[float]) -> float:
        if len(freqs) <= 1:
            return 0.0
        dom = np.sort(np.asarray(freqs, dtype=np.float64))
        gaps = np.diff(dom)
        return float(np.mean(np.exp(-gaps / (np.median(gaps) + 1e-12))))

    @staticmethod
    def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
        s = np.asarray(x, dtype=np.float64)
        return _coarse_grain_mean_numba(s, int(scale))

    def _multi_scale_dispersion_entropy(self, x: np.ndarray, p: VMDParameters) -> float:
        scales = int(max(1, p.entropy_scales))
        m = int(max(2, p.entropy_embed_dim))
        c = int(max(2, p.entropy_classes))
        d = int(max(1, p.entropy_delay))

        vals: list[float] = [
            self._dispersion_entropy(x, m=m, c=c, delay=d, normalize=True)
        ]
        if scales == 1:
            return vals[0]

        for s in range(2, scales + 1):
            if bool(p.entropy_refined_composite):
                for off in range(s):
                    xs = np.asarray(x, dtype=np.float64)[off:]
                    cg = self._coarse_grain(xs, s)
                    if cg.size >= (m - 1) * d + 2:
                        vals.append(
                            self._dispersion_entropy(
                                cg, m=m, c=c, delay=d, normalize=True
                            )
                        )
            else:
                cg = self._coarse_grain(x, s)
                if cg.size >= (m - 1) * d + 2:
                    vals.append(
                        self._dispersion_entropy(cg, m=m, c=c, delay=d, normalize=True)
                    )

        return float(np.mean(vals))

    def _entropy_K_score(
        self,
        signal: np.ndarray,
        fs: float,
        modes_list: list[np.ndarray],
        freqs_list: list[float],
        legacy_cost: float,
        p: VMDParameters,
        K: int,
    ) -> float:
        """
        More modern / literature-aligned version (2023–2025 style)
        Focus: low average mode RCMDE + acceptable recon error + kurtosis reward

        The default weights are chosen to sum to roughly 1.0 before the
        kurtosis reward / penalty terms, but user overrides are not
        renormalized. Changing them therefore changes the absolute score scale,
        so entropy scores are only directly comparable across runs that use the
        same weight configuration.
        """
        if not modes_list:
            return float("inf")

        x = np.asarray(signal, dtype=np.float64)
        recon = np.sum(np.stack(modes_list, axis=0), axis=0)
        err_norm = float(np.sum((x - recon) ** 2)) / (float(np.sum(x**2)) + 1e-12)

        # Primary term: average refined composite multi-scale dispersion entropy of modes
        mode_ent = [self._multi_scale_dispersion_entropy(m, p) for m in modes_list]
        avg_mode_ent = float(np.mean(mode_ent)) if mode_ent else 1.0

        # Secondary: residual entropy (should be high/random if good separation)
        residual_ent = float(self._multi_scale_dispersion_entropy(x - recon, p))

        # Overlap penalty (keep modes separated in frequency)
        overlap_pen = self._freq_overlap_penalty(freqs_list)

        # Kurtosis reward (subtract — encourages impulsive high-freq modes)
        kurt_reward = self._kurtosis_reward(modes_list, high_freq_weight=0.5)

        # K penalty (soft — avoid unnecessary modes)
        kn = float(K) / float(max(2, p.max_K))

        # These weights are read directly from VMDParameters and are not
        # normalized here. Override them only if you also accept that the
        # absolute score scale will shift across runs.
        w_mode = float(getattr(p, "entropy_w_mode_de", 0.55))
        w_recon = float(getattr(p, "entropy_w_recon", 0.18))
        w_resid = float(getattr(p, "entropy_w_residual_de", 0.12))
        w_overlap = float(
            getattr(
                p,
                "entropy_w_overlap",
                getattr(p, "entropy_overlap_weight", 0.08),
            )
        )
        w_k = float(getattr(p, "entropy_w_k", getattr(p, "entropy_k_penalty", 0.07)))

        # Composite score (lower = better)
        score = (
            w_mode * avg_mode_ent  # dominant: modes should be regular
            + w_recon * err_norm  # reconstruction fidelity
            + w_resid * (1.0 - residual_ent)  # residual should be complex/random
            + w_overlap * overlap_pen
            + w_k * kn
            - 0.20 * kurt_reward  # reward (negative term)
        )

        # Hard penalty if reconstruction is too bad
        if err_norm > float(p.entropy_recon_threshold or 0.15):
            score += float(getattr(p, "entropy_recon_penalty", 5.0)) * (
                err_norm - float(p.entropy_recon_threshold)
            )

        # Small legacy term (optional bridge)
        score += 0.05 * float(legacy_cost)

        return float(score)

    def estimate_K_alpha_entropy(
        self,
        signal: np.ndarray,
        fs: float,
        p: VMDParameters,
        precomp: dict[str, Any],
    ) -> tuple[int, float, float]:
        alpha0 = self._estimate_alpha_entropy(signal, p)
        span = float(max(1.05, p.entropy_alpha_span))
        n_grid = int(max(3, p.entropy_alpha_grid_points))
        if bool(p.entropy_alpha_full_range):
            alpha_grid = np.geomspace(
                float(p.alpha_min), float(p.alpha_max), num=n_grid
            )
        else:
            lo = max(float(p.alpha_min), alpha0 / span)
            hi = min(float(p.alpha_max), alpha0 * span)
            alpha_grid = np.geomspace(lo, hi, num=n_grid)
        alpha_grid = np.unique(np.clip(alpha_grid, p.alpha_min, p.alpha_max))

        bestK, bestScore = None, float("inf")
        bestAlpha = float(alpha0)
        for K in range(2, int(p.max_K) + 1):
            for alpha in alpha_grid:
                try:
                    modes_list, freqs_list, cost, _ = self._evaluate_candidate(
                        K=int(K),
                        alpha=float(alpha),
                        signal=signal,
                        fs=float(fs),
                        precomp=precomp,
                        p=p,
                        trial=None,
                        allow_cache=True,
                    )
                    score = self._entropy_K_score(
                        signal, fs, modes_list, freqs_list, cost, p, int(K)
                    )
                    if score < bestScore:
                        bestK, bestScore, bestAlpha = int(K), float(score), float(alpha)
                except Exception as e:
                    warnings.warn(
                        f"Entropy candidate failed for K={K}, alpha={float(alpha):.3g}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
        if bestK is None:
            bestK = int(max(2, p.max_K))
        return int(bestK), float(bestAlpha), float(bestScore)

    def _evaluate_candidate(
        self,
        *,
        K: int,
        alpha: float,
        signal: np.ndarray,
        fs: float,
        precomp: dict[str, Any],
        p: VMDParameters,
        trial: optuna.Trial | None = None,
        allow_cache: bool = True,
        return_raw_u: bool = False,
    ) -> tuple[list[np.ndarray], list[float], float, np.ndarray | None]:
        """
        The canonical evaluation path:
          - optional cache
          - decompose (with pruning)
          - postprocess (filter/merge/sort)
          - compute cost on postprocessed modes
        """
        if p.warm_start_u_hat is not None or p.warm_start_omega is not None:
            allow_cache = False

        alpha_b = self._alpha_bin(alpha)
        cacheable_alpha = alpha_b != -1
        key = (int(K), int(alpha_b))
        if allow_cache and cacheable_alpha and key in self._cache:
            cm, cf, cc = self._cache[key]
            raw = None
            return [m.copy() for m in cm], list(cf), float(cc), raw

        u, _, _ = self._call_core_decompose(
            signal=signal,
            alpha=float(alpha),
            fs=float(fs),
            K=int(K),
            init=int(p.init),
            p=p,
            precomp=precomp,
            trial=trial,
        )
        modes_list, freqs_list = self._postprocess_modes(u, signal, fs, p)
        cost = float(self.proc.cost_signal(modes_list, signal, fs))
        raw_out = np.asarray(u, dtype=np.float64).copy() if return_raw_u else None

        if allow_cache and cacheable_alpha and len(self._cache) < 80:
            self._cache[key] = (
                [m.copy() for m in modes_list],
                list(freqs_list),
                float(cost),
            )

        return [m.copy() for m in modes_list], list(freqs_list), float(cost), raw_out

    def _postprocess_modes_mv(
        self,
        modes: np.ndarray,  # (C, K, N)
        signals: np.ndarray,  # (C, N)
        fs: float,
        p: VMDParameters,
    ) -> tuple[list[np.ndarray], list[float], float]:
        X = np.asarray(signals, dtype=np.float64)
        U = np.asarray(modes, dtype=np.float64)
        if X.ndim != 2 or U.ndim != 3:
            return [], [], float("inf")

        C = int(X.shape[0])
        all_modes: list[np.ndarray] = []
        all_freqs: list[float] = []
        channel_costs: list[float] = []

        for ch in range(C):
            modes_ch, freqs_ch = self._postprocess_modes(U[ch], X[ch], fs, p)
            all_modes.extend([np.asarray(m, dtype=np.float64).copy() for m in modes_ch])
            all_freqs.extend([float(f) for f in freqs_ch])
            channel_costs.append(float(self.proc.cost_signal(modes_ch, X[ch], fs)))

        if not all_modes:
            return [], [], float(np.mean(channel_costs) if channel_costs else 10.0)

        all_modes, all_freqs = self.proc.sort_modes_by_frequency(
            all_modes, fs, low_to_high=True
        )
        return all_modes, all_freqs, float(np.mean(channel_costs))

    def _evaluate_candidate_mv(
        self,
        *,
        K: int,
        alpha: float,
        signals: np.ndarray,  # (C, N)
        fs: float,
        p: VMDParameters,
        allow_cache: bool = True,
        return_raw_u: bool = False,
    ) -> tuple[list[np.ndarray], list[float], float, np.ndarray | None]:
        if p.warm_start_u_hat is not None or p.warm_start_omega is not None:
            allow_cache = False

        alpha_b = self._alpha_bin(alpha)
        cacheable_alpha = alpha_b != -1
        key = (int(K), int(alpha_b))
        if allow_cache and cacheable_alpha and key in self._cache_mv:
            cm, cf, cc = self._cache_mv[key]
            return [m.copy() for m in cm], list(cf), float(cc), None

        u, _, _ = self._call_core_decompose_multivariate(
            signals=signals,
            fs=float(fs),
            p=p,
            K=int(K),
            alpha=float(alpha),
        )
        modes_list, freqs_list, cost = self._postprocess_modes_mv(u, signals, fs, p)
        raw_out = np.asarray(u, dtype=np.float64).copy() if return_raw_u else None

        if allow_cache and cacheable_alpha and len(self._cache_mv) < 80:
            self._cache_mv[key] = (
                [m.copy() for m in modes_list],
                list(freqs_list),
                float(cost),
            )

        return [m.copy() for m in modes_list], list(freqs_list), float(cost), raw_out

    # -----------------------------
    # K selection
    # -----------------------------
    def select_K(
        self,
        signal: np.ndarray,
        fs: float,
        p: VMDParameters,
        precomp: dict[str, Any],
        alpha_default: float = 2000.0,
    ) -> int:
        K_candidates = range(2, int(p.max_K) + 1)

        if p.k_selection.lower() == "fbd":
            bestK, bestScore = None, float("inf")
            for K in K_candidates:
                try:
                    u, _, _ = self._call_core_decompose(
                        signal=signal,
                        alpha=float(alpha_default),
                        fs=float(fs),
                        K=int(K),
                        init=1,
                        p=p,
                        precomp=precomp,
                        trial=None,
                    )
                    res = np.asarray(signal, dtype=np.float64) - np.sum(u, axis=0)
                    score = float(box_counting_dimension(res))
                    if score < bestScore:
                        bestK, bestScore = int(K), score
                except Exception as e:
                    warnings.warn(
                        f"FBD K selection failed at K={K}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
            return int(bestK) if bestK is not None else int(p.max_K)

        # default: penalized
        bestK, bestScore = None, float("inf")
        for K in K_candidates:
            try:
                u, _, _ = self._call_core_decompose(
                    signal=signal,
                    alpha=float(alpha_default),
                    fs=float(fs),
                    K=int(K),
                    init=int(p.init),
                    p=p,
                    precomp=precomp,
                    trial=None,
                )
                score = self._penalized_K_score(signal, fs, u, p, int(K))
                if score < bestScore:
                    bestK, bestScore = int(K), float(score)
            except Exception as e:
                warnings.warn(
                    f"Penalized K selection failed at K={K}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

        return int(bestK) if bestK is not None else int(p.max_K)

    def select_K_mv(
        self,
        signals: np.ndarray,
        fs: float,
        p: VMDParameters,
        alpha_default: float = 2000.0,
    ) -> int:
        X = np.asarray(signals, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("MVMD requires signals shaped (channels, samples)")

        K_candidates = range(2, int(p.max_K) + 1)
        bestK, bestScore = None, float("inf")

        if p.k_selection.lower() == "fbd":
            for K in K_candidates:
                try:
                    u, _, _ = self._call_core_decompose_multivariate(
                        signals=X,
                        fs=float(fs),
                        p=p,
                        K=int(K),
                        alpha=float(alpha_default),
                    )
                    score = 0.0
                    for ch in range(X.shape[0]):
                        recon = np.sum(u[ch], axis=0)
                        res = X[ch] - recon
                        score += float(box_counting_dimension(res))
                    score /= float(max(1, X.shape[0]))
                    if score < bestScore:
                        bestK, bestScore = int(K), float(score)
                except Exception as e:
                    warnings.warn(
                        f"MVMD FBD K={K} failed: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return int(bestK) if bestK is not None else int(p.max_K)

        for K in K_candidates:
            try:
                modes_list, freqs_list, cost, _ = self._evaluate_candidate_mv(
                    K=int(K),
                    alpha=float(alpha_default),
                    signals=X,
                    fs=float(fs),
                    p=p,
                    allow_cache=True,
                    return_raw_u=False,
                )
                overlap = self._freq_overlap_penalty(freqs_list)
                score = (
                    float(cost)
                    + float(p.k_penalty_lambda) * float(K)
                    + float(p.k_overlap_mu) * float(overlap)
                )
                if score < bestScore:
                    bestK, bestScore = int(K), float(score)
            except Exception as e:
                warnings.warn(
                    f"MVMD penalized K={K} failed: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return int(bestK) if bestK is not None else int(p.max_K)

    # -----------------------------
    # Optuna objectives (thin wrappers)
    # -----------------------------
    def _objective_full(
        self,
        trial: optuna.Trial,
        signal: np.ndarray,
        fs: float,
        precomp: dict[str, Any],
        p: VMDParameters,
    ) -> float:
        K = int(trial.suggest_int("K", 2, int(p.max_K)))
        alpha = float(
            trial.suggest_float(
                "alpha", float(p.alpha_min), float(p.alpha_max), log=True
            )
        )
        _, _, cost, _ = self._evaluate_candidate(
            K=K,
            alpha=alpha,
            signal=signal,
            fs=fs,
            precomp=precomp,
            p=p,
            trial=trial,
            allow_cache=True,
        )
        return float(cost)

    def _objective_alpha_only(
        self,
        trial: optuna.Trial,
        signal: np.ndarray,
        fs: float,
        precomp: dict[str, Any],
        p: VMDParameters,
        K_fixed: int,
    ) -> float:
        alpha = float(
            trial.suggest_float(
                "alpha", float(p.alpha_min), float(p.alpha_max), log=True
            )
        )
        _, _, cost, _ = self._evaluate_candidate(
            K=int(K_fixed),
            alpha=alpha,
            signal=signal,
            fs=fs,
            precomp=precomp,
            p=p,
            trial=trial,
            allow_cache=True,
        )
        return float(cost)

    def _objective_full_mv(
        self,
        trial: optuna.Trial,
        signals: np.ndarray,
        fs: float,
        p: VMDParameters,
    ) -> float:
        K = int(trial.suggest_int("K", 2, int(p.max_K)))
        alpha = float(
            trial.suggest_float(
                "alpha", float(p.alpha_min), float(p.alpha_max), log=True
            )
        )
        _, _, cost, _ = self._evaluate_candidate_mv(
            K=K,
            alpha=alpha,
            signals=np.asarray(signals, dtype=np.float64),
            fs=float(fs),
            p=p,
            allow_cache=True,
            return_raw_u=False,
        )
        return float(cost)

    def _objective_alpha_only_mv(
        self,
        trial: optuna.Trial,
        signals: np.ndarray,
        fs: float,
        p: VMDParameters,
        K_fixed: int,
    ) -> float:
        alpha = float(
            trial.suggest_float(
                "alpha", float(p.alpha_min), float(p.alpha_max), log=True
            )
        )
        _, _, cost, _ = self._evaluate_candidate_mv(
            K=int(K_fixed),
            alpha=alpha,
            signals=np.asarray(signals, dtype=np.float64),
            fs=float(fs),
            p=p,
            allow_cache=True,
            return_raw_u=False,
        )
        return float(cost)

    # -----------------------------
    # Public optimize()
    # -----------------------------
    def optimize(
        self,
        signal: np.ndarray | list[float],
        fs: float,
        auto_params: bool = True,
        refine_modes: bool = False,
        refine_epochs: int = 50,
        refine_method: str = "informer",
        use_mvmd: bool = False,
        return_raw_modes: bool = True,
        **overrides,
    ):
        x = np.asarray(signal, dtype=np.float64)

        if use_mvmd:
            self._cache_mv.clear()
            if x.ndim != 2:
                raise ValueError("MVMD requires signal shaped (channels, samples)")
            p = (
                self.analyzer.assess_complexity(x[0], fs)
                if auto_params
                else VMDParameters()
            )
            # `overrides` target VMDParameters fields rather than optimize()
            # signature parameters, so caller-side convenience kwargs must stay
            # aligned with the dataclass attribute names.
            for k, v in overrides.items():
                if hasattr(p, k):
                    setattr(p, k, v)

            if (
                str(p.search_method).lower() == "entropy"
                or p.k_selection.lower() == "entropy"
            ):
                warnings.warn(
                    "Entropy K/alpha search is not implemented for MVMD in this "
                    "pipeline; falling back to Optuna search.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            best_K_fixed: int | None = None
            if p.k_selection.lower() in ("penalized", "fbd"):
                best_K_fixed = self.select_K_mv(
                    x, float(fs), p, alpha_default=float(p.entropy_alpha_default)
                )

            study = optuna.create_study(direction="minimize")
            show_bar = bool(IS_NOTEBOOK)

            if best_K_fixed is not None:
                n_trials = max(10, int(p.n_trials) // 2)
                study.optimize(
                    lambda t: self._objective_alpha_only_mv(
                        t, x, float(fs), p, best_K_fixed
                    ),
                    n_trials=int(n_trials),
                    show_progress_bar=show_bar,
                )
                best_alpha = float(study.best_params["alpha"])
                best_K = int(best_K_fixed)
            else:
                study.optimize(
                    lambda t: self._objective_full_mv(t, x, float(fs), p),
                    n_trials=int(p.n_trials),
                    show_progress_bar=show_bar,
                )
                best_alpha = float(study.best_params["alpha"])
                best_K = int(study.best_params["K"])

            modes_list, freqs_list, best_cost, raw_from_eval = (
                self._evaluate_candidate_mv(
                    K=int(best_K),
                    alpha=float(best_alpha),
                    signals=x,
                    fs=float(fs),
                    p=p,
                    allow_cache=False,
                    return_raw_u=bool(return_raw_modes),
                )
            )

            if refine_modes and modes_list:
                if str(refine_method).lower() == "cross_mode":
                    refined = refine_modes_cross_nn(
                        np.stack(modes_list, axis=0), epochs=int(refine_epochs)
                    )
                else:
                    refined = refine_modes_nn(
                        np.stack(modes_list, axis=0), epochs=int(refine_epochs)
                    )
                modes_list = [
                    refined[i].astype(np.float64, copy=False)
                    for i in range(refined.shape[0])
                ]
                modes_list, freqs_list = self.proc.sort_modes_by_frequency(
                    modes_list, fs
                )

            if bool(p.apply_tapering) and modes_list:
                taper_len = int(min(100, x.shape[1] // 10))
                modes_list = BoundaryHandler.taper_boundaries(modes_list, taper_len)

            self.fftw.save_wisdom()

            post_modes_arr = (
                np.stack(modes_list, axis=0)
                if modes_list
                else np.zeros((0, x.shape[1]), dtype=np.float64)
            )

            if bool(return_raw_modes):
                raw_mv = (
                    np.asarray(raw_from_eval, dtype=np.float64)
                    if raw_from_eval is not None
                    else np.zeros((0, 0, x.shape[1]), dtype=np.float64)
                )
                raw_flat = (
                    raw_mv.reshape(-1, raw_mv.shape[-1])
                    if raw_mv.ndim == 3
                    else np.zeros((0, x.shape[1]), dtype=np.float64)
                )
                raw_freqs = [
                    float(self.proc.dominant_frequency(raw_flat[i], fs))
                    for i in range(raw_flat.shape[0])
                ]
                optinfo: dict[str, Any] = {
                    "best": (int(best_K), float(best_alpha), float(best_cost)),
                    "raw_modes": raw_flat,
                    "raw_freqs": raw_freqs,
                    "post_modes": post_modes_arr,
                    "post_freqs": list(freqs_list),
                    "mv_shape": tuple(raw_mv.shape),
                }
                return raw_flat, raw_freqs, optinfo

            return (
                post_modes_arr,
                freqs_list,
                (
                    int(best_K),
                    float(best_alpha),
                    float(best_cost),
                ),
            )

        self._cache.clear()
        p = self.analyzer.assess_complexity(x, fs) if auto_params else VMDParameters()
        # `overrides` target VMDParameters fields rather than optimize()
        # signature parameters, so caller-side convenience kwargs must stay
        # aligned with the dataclass attribute names.
        for k, v in overrides.items():
            if hasattr(p, k):
                setattr(p, k, v)

        precomp = self.core.precompute_fft(
            x,
            boundary_method=str(p.boundary_method),
            use_soft_junction=bool(p.use_soft_junction),
            window_alpha=p.window_alpha,
            fft_backend=str(getattr(p, "fft_backend", "fftw")),
            fft_device=str(getattr(p, "fft_device", "auto")),
        )

        use_entropy_search = (
            str(p.search_method).lower() == "entropy"
            or p.k_selection.lower() == "entropy"
        )
        best_K_fixed: int | None = None
        if (not use_entropy_search) and p.k_selection.lower() in ("penalized", "fbd"):
            # The entropy search path performs its own K sweep, so skip the
            # separate penalized/FBD pre-pass to avoid redundant decomposition.
            best_K_fixed = self.select_K(x, fs, p, precomp, alpha_default=2000.0)

        if use_entropy_search:
            best_K, best_alpha, _ = self.estimate_K_alpha_entropy(x, fs, p, precomp)
        else:
            study = optuna.create_study(direction="minimize")
            show_bar = bool(IS_NOTEBOOK)

            if best_K_fixed is not None:
                n_trials = max(10, int(p.n_trials) // 2)
                study.optimize(
                    lambda t: self._objective_alpha_only(
                        t, x, fs, precomp, p, best_K_fixed
                    ),
                    n_trials=int(n_trials),
                    show_progress_bar=show_bar,
                )
                best_alpha = float(study.best_params["alpha"])
                best_K = int(best_K_fixed)
            else:
                study.optimize(
                    lambda t: self._objective_full(t, x, fs, precomp, p),
                    n_trials=int(p.n_trials),
                    show_progress_bar=show_bar,
                )
                best_alpha = float(study.best_params["alpha"])
                best_K = int(study.best_params["K"])

        modes_list, freqs_list, best_cost, raw_from_eval = self._evaluate_candidate(
            K=int(best_K),
            alpha=float(best_alpha),
            signal=x,
            fs=float(fs),
            precomp=precomp,
            p=p,
            trial=None,
            allow_cache=False,
            return_raw_u=bool(return_raw_modes),
        )

        raw_modes_np: np.ndarray | None = None
        raw_freqs_list: list[float] = []
        if bool(return_raw_modes):
            raw_modes_np = (
                np.asarray(raw_from_eval, dtype=np.float64)
                if raw_from_eval is not None
                else np.zeros((0, x.size), dtype=np.float64)
            )
            raw_freqs_list = [
                float(self.proc.dominant_frequency(raw_modes_np[k], fs))
                for k in range(raw_modes_np.shape[0])
            ]

        if refine_modes and modes_list:
            if str(refine_method).lower() == "cross_mode":
                refined = refine_modes_cross_nn(
                    np.stack(modes_list, axis=0), epochs=int(refine_epochs)
                )
            else:
                refined = refine_modes_nn(
                    np.stack(modes_list, axis=0), epochs=int(refine_epochs)
                )
            modes_list = [
                refined[i].astype(np.float64, copy=False)
                for i in range(refined.shape[0])
            ]
            modes_list, freqs_list = self.proc.sort_modes_by_frequency(modes_list, fs)

        if bool(p.apply_tapering) and modes_list:
            taper_len = int(min(100, x.size // 10))
            modes_list = BoundaryHandler.taper_boundaries(modes_list, taper_len)

        self.fftw.save_wisdom()

        modes_arr = (
            np.stack(modes_list, axis=0)
            if modes_list
            else np.zeros((0, x.size), dtype=np.float64)
        )

        if bool(return_raw_modes):
            out_raw_modes = (
                raw_modes_np
                if raw_modes_np is not None
                else np.zeros((0, x.size), dtype=np.float64)
            )
            out_raw_freqs = (
                raw_freqs_list if raw_modes_np is not None else list(freqs_list)
            )
            optinfo: dict[str, Any] = {
                "best": (int(best_K), float(best_alpha), float(best_cost)),
                "raw_modes": out_raw_modes,
                "raw_freqs": out_raw_freqs,
                "post_modes": modes_arr,
                "post_freqs": list(freqs_list),
            }
            return out_raw_modes, out_raw_freqs, optinfo

        return modes_arr, freqs_list, (int(best_K), float(best_alpha), float(best_cost))


# -----------------------------------------------------------------------------
# Hierarchical VMD (minor de-dup: upsample helper)
# -----------------------------------------------------------------------------
class HierarchicalVMD:
    def __init__(self, optimizer: VMDOptimizer):
        self.opt = optimizer

    @staticmethod
    def _upsample_modes_to(modes: np.ndarray, target_len: int) -> np.ndarray:
        if modes.size == 0:
            return modes
        if modes.shape[1] == target_len:
            return modes
        up = []
        for i in range(modes.shape[0]):
            m = modes[i]
            xo = np.linspace(0.0, 1.0, m.size)
            xn = np.linspace(0.0, 1.0, target_len)
            up.append(np.interp(xn, xo, m))
        return np.stack(up, axis=0).astype(np.float64)

    def decompose(
        self,
        signal: np.ndarray,
        fs: float,
        params: HierarchicalParameters,
        refine_modes: bool = True,
        refine_epochs: int = 50,
        refine_method: str = "informer",
    ) -> tuple[np.ndarray, list[float], list[dict[str, Any]]]:
        x = np.asarray(signal, dtype=np.float64)
        original_energy = _energy(x) + 1e-12

        all_modes: list[np.ndarray] = []
        level_info: list[dict[str, Any]] = []
        residual = x.copy()

        print(f"Starting Hierarchical VMD (max_levels={params.max_levels})")

        for level in range(int(params.max_levels)):
            downsample_factor = 2**level

            if level == 0:
                signal_level = residual.copy()
                fs_level = float(fs)
            else:
                expected_len = residual.size // downsample_factor
                if expected_len < int(params.min_samples_per_level):
                    print(
                        f"Stopping: projected len {expected_len} < min {params.min_samples_per_level}"
                    )
                    break

                use_aa = params.use_anti_aliasing_higher_levels
                do_fir = (
                    bool(use_aa)
                    and (residual.size >= int(params.min_samples_for_fir_decimation))
                    and (expected_len >= 100)
                )

                if do_fir:
                    try:
                        signal_level = decimate(
                            residual, q=downsample_factor, ftype="fir", zero_phase=True
                        )
                    except Exception:
                        signal_level = residual[::downsample_factor]
                else:
                    signal_level = residual[::downsample_factor]

                fs_level = float(fs) / float(downsample_factor)

            print(
                f"Level {level + 1}/{params.max_levels}: len={signal_level.size} @ fs={fs_level:.2f} Hz"
            )

            try:
                t0 = time.time()
                # These are VMDParameters overrides, not explicit optimize()
                # keyword parameters. They rely on matching dataclass fields.
                modes_level, freqs_level, opt_params = self.opt.optimize(
                    signal_level,
                    fs_level,
                    auto_params=True,
                    refine_modes=refine_modes,
                    refine_method=refine_method,
                    apply_tapering=False,
                    k_selection="penalized",
                )
                dt = time.time() - t0

                if params.use_emd_hybrid and modes_level.shape[0] > 0:
                    ent = [
                        self.opt.proc.entropy(modes_level[i])
                        for i in range(modes_level.shape[0])
                    ]
                    idx = int(np.argmax(ent))
                    imfs = EMDVariants.ceemdan(modes_level[idx])
                    if len(imfs) > 1:
                        modes_level[idx] = np.sum(np.stack(imfs[:-1], axis=0), axis=0)

                level_info.append(
                    {
                        "level": level + 1,
                        "downsample_factor": downsample_factor,
                        "fs": fs_level,
                        "n_modes": int(modes_level.shape[0]),
                        "frequencies": freqs_level,
                        "energy_ratio": _energy(signal_level) / original_energy,
                        "computation_time": dt,
                        "parameters": opt_params,
                    }
                )
            except Exception as e:
                print(f"FAILED at level {level + 1}: {e}")
                break

            # Upsample to original length if needed
            if level > 0 and modes_level.shape[0] > 0:
                modes_level = self._upsample_modes_to(modes_level, x.size)

            for i in range(modes_level.shape[0]):
                all_modes.append(modes_level[i].copy())

            recon = np.sum(modes_level, axis=0) if modes_level.shape[0] else 0.0
            residual = residual - recon
            res_ratio = _energy(residual) / original_energy

            if res_ratio < float(params.energy_threshold):
                print("Stopping: residual below threshold")
                break

        merged = self.opt.proc.merge_similar_modes(all_modes, fs, freq_tol=0.2)
        sorted_modes, sorted_freqs = self.opt.proc.sort_modes_by_frequency(merged, fs)

        if refine_modes and sorted_modes:
            if str(refine_method).lower() == "cross_mode":
                refined = refine_modes_cross_nn(
                    np.stack(sorted_modes, axis=0), epochs=int(refine_epochs)
                )
            else:
                refined = refine_modes_nn(
                    np.stack(sorted_modes, axis=0), epochs=int(refine_epochs)
                )
            sorted_modes = [refined[i] for i in range(refined.shape[0])]
            sorted_modes, sorted_freqs = self.opt.proc.sort_modes_by_frequency(
                sorted_modes, fs
            )

        out = (
            np.stack(sorted_modes, axis=0)
            if sorted_modes
            else np.zeros((0, x.size), dtype=np.float64)
        )
        return out, sorted_freqs, level_info

    @staticmethod
    def print_summary(level_info: list[dict[str, Any]]) -> None:
        print("\n" + "=" * 60)
        print("HIERARCHICAL VMD SUMMARY")
        print("=" * 60)
        total_modes = sum(int(info["n_modes"]) for info in level_info)
        total_time = sum(float(info["computation_time"]) for info in level_info)
        print(f"Levels processed: {len(level_info)}")
        print(f"Total modes found: {total_modes}")
        print(f"Total computation time: {total_time:.2f}s")
        if level_info:
            print(f"Average time per level: {total_time / len(level_info):.2f}s")
        print("\nLevel details:")
        for info in level_info:
            freqs = info.get("frequencies", [])
            fpreview = (
                [f"{float(f):.2f}" for f in freqs[:3]]
                if isinstance(freqs, list)
                else []
            )
            print(
                f" - Level {info['level']}: fs={info['fs']:.2f}Hz, down={info['downsample_factor']}x, "
                f"modes={info['n_modes']}, energy={100 * info['energy_ratio']:.1f}%, time={info['computation_time']:.2f}s, "
                f"dom={fpreview}"
            )


# -----------------------------------------------------------------------------
# User-facing API
# -----------------------------------------------------------------------------
class FastVMD:
    def __init__(self, wisdom_file: str = "vmd_fftw_wisdom.dat"):
        self.fftw = FFTWManager(wisdom_file)
        self.opt = VMDOptimizer(self.fftw)
        self.hvmd = HierarchicalVMD(self.opt)

    def decompose(
        self, signal: np.ndarray, fs: float, method: str = "standard", **kwargs
    ):
        refine_modes = bool(kwargs.pop("refine_modes", True))
        refine_epochs = int(kwargs.pop("refine_epochs", 50))
        refine_method = str(kwargs.pop("refine_method", "informer"))
        return_raw_modes = bool(kwargs.pop("return_raw_modes", True))

        if method == "hierarchical":
            hp = HierarchicalParameters(
                max_levels=int(kwargs.pop("max_levels", 3)),
                energy_threshold=float(kwargs.pop("energy_threshold", 0.01)),
                min_samples_per_level=int(kwargs.pop("min_samples_per_level", 100)),
                use_anti_aliasing_level_0=bool(
                    kwargs.pop("use_anti_aliasing_level_0", True)
                ),
                use_anti_aliasing_higher_levels=bool(
                    kwargs.pop("use_anti_aliasing_higher_levels", False)
                ),
                min_samples_for_fir_decimation=int(
                    kwargs.pop("min_samples_for_fir_decimation", 600)
                ),
                use_emd_hybrid=bool(kwargs.pop("use_emd_hybrid", False)),
            )
            return self.hvmd.decompose(
                signal,
                fs,
                hp,
                refine_modes=refine_modes,
                refine_epochs=refine_epochs,
                refine_method=refine_method,
            )

        return self.opt.optimize(
            signal,
            fs,
            refine_modes=refine_modes,
            refine_epochs=refine_epochs,
            refine_method=refine_method,
            return_raw_modes=return_raw_modes,
            **kwargs,
        )

    def clear_cache(self):
        self.opt._cache.clear()
        gc.collect()

    def __del__(self):
        try:
            self.clear_cache()
        except Exception:
            pass
