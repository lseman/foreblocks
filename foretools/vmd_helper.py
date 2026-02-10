# -*- coding: utf-8 -*-
"""
Fast Optuna-Optimized VMD with FFT caching (CLEAN + FIXED) — FULL REFACTORED (NO DUPLICATED PIPELINES)
====================================================================================================
This version keeps all functionality you had, but removes repeated code by introducing ONE canonical
"evaluate candidate" pipeline inside VMDOptimizer.

Kept:
- Boundary fixes:
  - init==3 + precomputed_fft FIX (store fMirr)
  - odd-length trim FIX (orig_len decremented)
  - soft-junction taper_len clamp FIX (min with ext_len)
- Optuna pruning/reporting safety (no forced matplotlib)
- Optional Torch refinement guarded
- Minimal MVMD (shared omega estimate)
- Hierarchical VMD + optional EMD hybrid

Main refactor:
- VMDOptimizer has a single _evaluate_candidate(K, alpha) that does:
    decompose -> postprocess -> compute cost
  and it is reused for:
    - Optuna objective (K, alpha)
    - Optuna objective (alpha-only)
    - K selection (penalized, fbd)
    - Final extraction

Notes:
- No emojis in this file (per preference).
"""

from __future__ import annotations

import gc
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pyfftw
import pyfftw.interfaces.numpy_fft as fftw_np
from numba import njit
from scipy.interpolate import interp1d, Akima1DInterpolator, PchipInterpolator
from scipy.signal import find_peaks, hilbert, decimate


# -----------------------------------------------------------------------------
# Optional torch refinement
# -----------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Adam = None
    
# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


# -----------------------------------------------------------------------------
# Fractal Dimension (kept)
# -----------------------------------------------------------------------------
class FractalDimension:
    """
    Multiple methods for estimating fractal/complexity dimension of 1D signals.
    """

    @staticmethod
    def box_counting(
        signal: np.ndarray,
        n_scales: int = 12,
        min_box: int = 4,
        max_box_ratio: float = 0.25,
    ) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 64 or np.allclose(x, x[0], atol=1e-12):
            return 1.0

        x_centered = x - np.median(x)
        mad = np.median(np.abs(x_centered)) + 1e-12
        x_norm = x_centered / (6.0 * mad)
        x_norm = np.clip(x_norm, -1.0, 1.0)
        y = 0.5 * (x_norm + 1.0)

        max_box = max(min_box * 2, int(N * max_box_ratio))
        sizes = np.unique(
            np.logspace(np.log10(min_box), np.log10(max_box), n_scales).astype(int)
        )
        sizes = sizes[(sizes > 1) & (sizes < N // 2)]
        if len(sizes) < 4:
            return 1.0

        counts = []
        inv_sizes = []
        for box_size in sizes:
            n_boxes_x = int(np.ceil(N / box_size))
            n_boxes_y = max(2, int(np.ceil(N / box_size)))
            box_x = (np.arange(N) // box_size).astype(int)
            box_y = np.floor(y * n_boxes_y).astype(int)
            box_y = np.clip(box_y, 0, n_boxes_y - 1)

            occupied = set(zip(box_x.tolist(), box_y.tolist()))
            counts.append(len(occupied))
            inv_sizes.append(1.0 / box_size)

        if len(counts) < 4:
            return 1.0

        log_inv = np.log(np.array(inv_sizes))
        log_cnt = np.log(np.array(counts) + 1e-12)
        coeffs = np.polyfit(log_inv, log_cnt, deg=1)
        dim = coeffs[0]
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 0.5, 2.5))

    @staticmethod
    def differential_box_counting(
        signal: np.ndarray,
        n_scales: int = 10,
        min_scale: int = 2,
        max_scale_ratio: float = 0.25,
    ) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 64:
            return 1.0

        G = 256
        x_min, x_max = np.min(x), np.max(x)
        if np.isclose(x_max, x_min):
            return 1.0
        x_norm = ((x - x_min) / (x_max - x_min) * (G - 1)).astype(int)

        max_scale = max(min_scale + 1, int(N * max_scale_ratio))
        scales = np.unique(
            np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
        )
        scales = scales[(scales >= min_scale) & (scales <= N // 4)]
        if len(scales) < 4:
            return FractalDimension.box_counting(signal)

        counts = []
        log_scales = []
        for r in scales:
            s = max(1, G // r)
            n_columns = N // r
            if n_columns < 1:
                continue

            total_boxes = 0
            for col in range(n_columns):
                start_idx = col * r
                end_idx = min(start_idx + r, N)
                column_vals = x_norm[start_idx:end_idx]
                if len(column_vals) == 0:
                    continue
                col_min = np.min(column_vals)
                col_max = np.max(column_vals)
                k_min = col_min // s
                k_max = col_max // s
                total_boxes += (k_max - k_min + 1)

            if total_boxes > 0:
                counts.append(total_boxes)
                log_scales.append(np.log(1.0 / r))

        if len(counts) < 4:
            return FractalDimension.box_counting(signal)

        log_counts = np.log(np.array(counts) + 1e-12)
        log_scales = np.array(log_scales)
        coeffs = np.polyfit(log_scales, log_counts, deg=1)
        dim = coeffs[0]
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 1.0, 2.0))

    @staticmethod
    def higuchi(signal: np.ndarray, k_max: int = 10) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 20:
            return 1.0

        k_max = min(k_max, N // 4)
        if k_max < 2:
            return 1.0

        L = []
        k_values = []
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(1, k + 1):
                idx = np.arange(m - 1, N, k)
                if len(idx) < 2:
                    continue
                n_segments = len(idx) - 1
                if n_segments < 1:
                    continue
                sub_series = x[idx]
                length = np.sum(np.abs(np.diff(sub_series)))
                norm_factor = (N - 1) / (k * n_segments * k)
                Lk.append(length * norm_factor)
            if len(Lk) > 0:
                L.append(np.mean(Lk))
                k_values.append(k)

        if len(L) < 3:
            return 1.0

        log_k = np.log(np.array(k_values))
        log_L = np.log(np.array(L) + 1e-12)
        coeffs = np.polyfit(log_k, log_L, deg=1)
        dim = -coeffs[0]
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 1.0, 2.0))

    @staticmethod
    def katz(signal: np.ndarray) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 10:
            return 1.0

        L = np.sum(np.sqrt(1 + np.diff(x) ** 2))
        distances = np.sqrt(np.arange(N) ** 2 + (x - x[0]) ** 2)
        d = np.max(distances)
        if d < 1e-12 or L < 1e-12:
            return 1.0

        n = N - 1
        dim = np.log10(n) / (np.log10(n) + np.log10(d / L))
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 1.0, 2.0))

    @staticmethod
    def petrosian(signal: np.ndarray) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 10:
            return 1.0

        dx = np.diff(x)
        sign_changes = np.sum(dx[:-1] * dx[1:] < 0)
        if sign_changes == 0:
            return 1.0

        dim = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * sign_changes)))
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 1.0, 2.0))

    @staticmethod
    def estimate_complexity(signal: np.ndarray, method: str = "auto") -> float:
        if method == "higuchi":
            return FractalDimension.higuchi(signal)
        if method == "dbc":
            return FractalDimension.differential_box_counting(signal)
        if method == "box":
            return FractalDimension.box_counting(signal)
        if method == "katz":
            return FractalDimension.katz(signal)
        if method == "petrosian":
            return FractalDimension.petrosian(signal)
        if method == "auto":
            estimates = [
                FractalDimension.higuchi(signal),
                FractalDimension.differential_box_counting(signal),
                FractalDimension.katz(signal),
            ]
            return float(np.median(estimates))
        raise ValueError(f"Unknown FD method: {method}")


def box_counting_dimension(signal: np.ndarray, **kwargs) -> float:
    return FractalDimension.estimate_complexity(signal, method="auto")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class VMDParameters:
    # Optuna
    n_trials: int = 30
    max_K: int = 6

    # VMD
    tol: float = 1e-6
    alpha_min: float = 500
    alpha_max: float = 5000
    tau: float = 0.0
    DC: int = 0
    init: int = 1
    max_iter: int = 300

    # boundary
    boundary_method: str = "reflect"
    use_soft_junction: bool = False
    window_alpha: Optional[float] = None  # if None -> auto

    # post
    apply_tapering: bool = True
    mode_energy_floor: float = 0.01
    merge_freq_tol: float = 0.15

    # extras
    use_fs_vmd: bool = False
    use_mvmd: bool = False

    # K selection
    k_selection: str = "penalized"  # "penalized" | "fbd" | "optuna"
    k_penalty_lambda: float = 0.02
    k_overlap_mu: float = 0.10


@dataclass
class HierarchicalParameters:
    max_levels: int = 3
    energy_threshold: float = 0.01
    min_samples_per_level: int = 100

    # downsampling behavior
    use_anti_aliasing_level_0: bool = True
    use_anti_aliasing_higher_levels: bool = False
    min_samples_for_fir_decimation: int = 600

    # hybrid
    use_emd_hybrid: bool = False


# -----------------------------------------------------------------------------
# FFTW
# -----------------------------------------------------------------------------
class FFTWManager:
    def __init__(self, wisdom_file: str = "vmd_fftw_wisdom.dat"):
        self.wisdom_file = wisdom_file
        self._setup_fftw()
        self.load_wisdom()

    def _setup_fftw(self):
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(7200)
        pyfftw.config.NUM_THREADS = -1

    def load_wisdom(self):
        if os.path.exists(self.wisdom_file):
            try:
                with open(self.wisdom_file, "rb") as f:
                    pyfftw.import_wisdom(pickle.load(f))
            except Exception:
                pass

    def save_wisdom(self):
        try:
            with open(self.wisdom_file, "wb") as f:
                pickle.dump(pyfftw.export_wisdom(), f)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Boundary handling (single source of truth)
# -----------------------------------------------------------------------------
class BoundaryHandler:
    @staticmethod
    def adaptive_extension_ratio(signal: np.ndarray) -> float:
        N = len(signal)
        if N < 500:
            return 0.30
        if N < 2000:
            return 0.20
        return 0.15

    @staticmethod
    def auto_window_alpha(
        signal: np.ndarray, min_alpha: float = 0.01, max_alpha: float = 0.1
    ) -> float:
        s = np.asarray(signal, dtype=np.float64)
        if s.size < 8:
            return min_alpha
        deriv = np.diff(s)
        deriv_var = np.var(deriv)
        norm_var = deriv_var / (np.mean(np.abs(s)) ** 2 + 1e-12)
        smoothness = 1.0 / (1.0 + norm_var)
        return float(min_alpha + (max_alpha - min_alpha) * smoothness)

    @staticmethod
    def extend_signal(
        signal: np.ndarray, method: str, extension_ratio: float
    ) -> Tuple[np.ndarray, int, int]:
        s = np.asarray(signal, dtype=np.float64)
        N = s.size
        ext_len = int(N * float(extension_ratio))
        if ext_len <= 0 or method == "none":
            return s, 0, 0

        if ext_len >= N - 2:
            ext_len = max(1, N // 4)

        if method == "mirror":
            left_ext = s[1 : ext_len + 1][::-1]
            right_ext = s[-(ext_len + 1) : -1][::-1]
            out = np.concatenate([left_ext, s, right_ext])

        elif method == "reflect":
            left_val, right_val = s[0], s[-1]
            left_ext = 2 * left_val - s[1 : ext_len + 1][::-1]
            right_ext = 2 * right_val - s[-(ext_len + 1) : -1][::-1]
            out = np.concatenate([left_ext, s, right_ext])

        elif method == "linear":
            left_slope = s[1] - s[0]
            right_slope = s[-1] - s[-2]
            left_ext = s[0] + left_slope * np.arange(-ext_len, 0)
            right_ext = s[-1] + right_slope * np.arange(1, ext_len + 1)
            out = np.concatenate([left_ext, s, right_ext])

        elif method == "constant":
            out = np.concatenate(
                [np.full(ext_len, s[0]), s, np.full(ext_len, s[-1])]
            )

        else:
            return s, 0, 0

        return out, ext_len, ext_len

    @staticmethod
    def smooth_edge_junction(
        extended: np.ndarray,
        original_len: int,
        ext_len: int,
        smooth_ratio: float = 0.02,
    ) -> np.ndarray:
        if ext_len <= 0:
            return extended
        taper_len = max(2, int(original_len * smooth_ratio))

        # FIX: clamp with ext_len
        taper_len = min(taper_len, ext_len)
        if taper_len < 1:
            return extended

        ramp_up = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_len)))
        ramp_down = ramp_up[::-1]

        out = extended.copy()
        out[ext_len - taper_len : ext_len] *= ramp_up
        end_idx = ext_len + original_len
        out[end_idx : end_idx + taper_len] *= ramp_down
        return out

    @staticmethod
    def taper_boundaries(modes: List[np.ndarray], taper_length: int) -> List[np.ndarray]:
        out = []
        for m in modes:
            x = np.asarray(m, dtype=np.float64).copy()
            N = x.size
            tlen = int(min(max(2, taper_length), N // 4))
            if tlen <= 1:
                out.append(x)
                continue

            w = np.ones(N, dtype=np.float64)
            w[:tlen] = np.sin(np.linspace(0, np.pi / 2, tlen)) ** 2
            w[-tlen:] = np.cos(np.linspace(0, np.pi / 2, tlen)) ** 2
            out.append(x * w)
        return out


# -----------------------------------------------------------------------------
# Signal analysis
# -----------------------------------------------------------------------------
class SignalAnalyzer:
    @staticmethod
    def estimate_snr(signal: np.ndarray, fs: float) -> float:
        s = np.asarray(signal, dtype=np.float64)
        spec = np.abs(fftw_np.rfft(s))
        power_total = np.sum(spec**2)
        noise_floor = np.median(spec)
        power_noise = (noise_floor**2) * len(spec)
        return float(10 * np.log10((power_total + 1e-12) / (power_noise + 1e-12)))

    @staticmethod
    def dominant_freq(signal: np.ndarray, fs: float) -> float:
        s = np.asarray(signal, dtype=np.float64)
        if s.size < 8:
            return 0.0
        spec = np.abs(fftw_np.rfft(s))
        freqs = fftw_np.rfftfreq(s.size, d=1 / fs)
        if spec.size == 0:
            return 0.0
        spec = spec.copy()
        spec[0] = 0.0
        return float(freqs[int(np.argmax(spec))])

    @staticmethod
    def assess_complexity(signal: np.ndarray, fs: float) -> VMDParameters:
        s = np.asarray(signal, dtype=np.float64)
        N = s.size

        spec = np.abs(fftw_np.rfft(s))
        freqs = fftw_np.rfftfreq(N, d=1 / fs)
        spec_norm = spec / (np.sum(spec) + 1e-12)
        spectral_entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-12))

        spec2 = spec.copy()
        if spec2.size:
            spec2[0] = 0
        peak_idx = int(np.argmax(spec2)) if spec2.size else 0
        dom = float(freqs[peak_idx]) if freqs.size else 0.0

        thr = 0.05 * (np.max(spec2) if spec2.size else 0.0)
        spread = float(len(freqs[spec2 > thr])) / max(1, len(freqs))

        variability = float(np.std(np.diff(s))) / (float(np.std(s)) + 1e-12)

        complexity_score = (
            0.4 * (spectral_entropy / 10.0)
            + 0.3 * spread
            + 0.3 * min(variability, 2.0) / 2.0
        )
        snr_db = SignalAnalyzer.estimate_snr(s, fs)

        if complexity_score < 0.3:
            p = VMDParameters(
                n_trials=max(10, min(15, N // 200)),
                max_K=4,
                tol=1e-5,
                alpha_min=1000,
                alpha_max=3000,
            )
        elif complexity_score < 0.6:
            p = VMDParameters(
                n_trials=max(15, min(25, N // 150)),
                max_K=6,
                tol=1e-6,
                alpha_min=500,
                alpha_max=5000,
            )
        else:
            p = VMDParameters(
                n_trials=max(20, min(35, N // 100)),
                max_K=8,
                tol=1e-7,
                alpha_min=200,
                alpha_max=8000,
            )

        if snr_db < 10:
            p.alpha_min *= 2.0
            p.alpha_max *= 2.5
            p.max_K = max(2, p.max_K // 2)
        elif snr_db < 20:
            p.alpha_min *= 1.5
            p.alpha_max *= 1.5
        else:
            p.max_K = min(p.max_K + 2, 10)
            p.alpha_min *= 0.8
            p.alpha_max *= 0.8

        if N > 3000:
            p.n_trials = max(10, p.n_trials // 2)
            p.tol *= 2.0

        print(
            f"complexity={complexity_score:.3f} | SNR={snr_db:.1f} dB | dom≈{dom:.2f} Hz"
        )
        print(
            f" => trials={p.n_trials}, max_K={p.max_K}, alpha∈[{p.alpha_min:.0f},{p.alpha_max:.0f}]"
        )
        return p


# -----------------------------------------------------------------------------
# EMD Variants (kept as-is with your stability improvements)
# -----------------------------------------------------------------------------
class EMDVariants:
    @staticmethod
    def emd(
        signal: np.ndarray,
        max_imfs: int = 10,
        max_sifts: int = 100,
        sift_threshold: float = 0.05,
        energy_threshold: float = 1e-8,
        envelope_method: str = "akima",
    ) -> List[np.ndarray]:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size

        if N < 10 or np.allclose(x, x[0], atol=1e-12):
            return [x.copy()]

        imfs = []
        residual = x.copy()
        original_energy = np.sum(x**2) + 1e-12

        for _ in range(max_imfs):
            if np.sum(residual**2) / original_energy < energy_threshold:
                break
            if EMDVariants._is_monotonic(residual):
                break

            imf = EMDVariants._sift(
                residual,
                max_sifts=max_sifts,
                threshold=sift_threshold,
                envelope_method=envelope_method,
            )
            if imf is None:
                break

            imfs.append(imf)
            residual = residual - imf

            if np.std(imf) < 1e-10 * np.std(x):
                break

        if len(imfs) == 0 or np.sum(residual**2) > energy_threshold * original_energy:
            imfs.append(residual)

        return imfs

    @staticmethod
    def _sift(
        signal: np.ndarray,
        max_sifts: int = 100,
        threshold: float = 0.05,
        envelope_method: str = "akima",
    ) -> Optional[np.ndarray]:
        h = signal.copy()
        N = len(h)

        for iteration in range(max_sifts):
            max_idx, max_val = EMDVariants._find_extrema(h, kind="max")
            min_idx, min_val = EMDVariants._find_extrema(h, kind="min")

            if len(max_idx) < 3 or len(min_idx) < 3:
                return h if iteration > 0 else None

            upper = EMDVariants._interpolate_envelope(max_idx, max_val, N, method=envelope_method)
            lower = EMDVariants._interpolate_envelope(min_idx, min_val, N, method=envelope_method)
            if upper is None or lower is None:
                return h if iteration > 0 else None

            mean_env = 0.5 * (upper + lower)
            h_new = h - mean_env

            diff = np.sum((h_new - h) ** 2) / (np.sum(h**2) + 1e-12)
            h = h_new

            if diff < threshold:
                break
            if np.std(mean_env) < threshold * np.std(h):
                break

        return h

    @staticmethod
    def _find_extrema(signal: np.ndarray, kind: str = "max") -> Tuple[np.ndarray, np.ndarray]:
        x = signal
        N = len(x)

        if kind == "max":
            peaks, _ = find_peaks(x)
            if N > 2:
                if x[0] > x[1]:
                    peaks = np.concatenate([[0], peaks])
                if x[-1] > x[-2]:
                    peaks = np.concatenate([peaks, [N - 1]])
        else:
            peaks, _ = find_peaks(-x)
            if N > 2:
                if x[0] < x[1]:
                    peaks = np.concatenate([[0], peaks])
                if x[-1] < x[-2]:
                    peaks = np.concatenate([peaks, [N - 1]])

        peaks = np.unique(peaks)
        return peaks, x[peaks]

    @staticmethod
    def _interpolate_envelope(
        idx: np.ndarray,
        values: np.ndarray,
        N: int,
        method: str = "akima",
    ) -> Optional[np.ndarray]:
        if len(idx) < 2:
            return None

        order = np.argsort(idx)
        idx = idx[order]
        values = values[order]

        unique_mask = np.concatenate([[True], np.diff(idx) > 0])
        idx = idx[unique_mask]
        values = values[unique_mask]
        if len(idx) < 2:
            return None

        x_interp = np.arange(N)

        try:
            if method == "akima" and len(idx) >= 5:
                interp_func = Akima1DInterpolator(idx, values)
                envelope = interp_func(x_interp)
            elif method == "pchip" and len(idx) >= 2:
                interp_func = PchipInterpolator(idx, values, extrapolate=True)
                envelope = interp_func(x_interp)
            elif method == "cubic" and len(idx) >= 4:
                interp_func = interp1d(
                    idx, values, kind="cubic", fill_value="extrapolate", bounds_error=False
                )
                envelope = interp_func(x_interp)
            else:
                envelope = np.interp(x_interp, idx, values)

            if np.any(~np.isfinite(envelope)):
                envelope = np.interp(x_interp, idx, values)

            return envelope
        except Exception:
            try:
                return np.interp(x_interp, idx, values)
            except Exception:
                return None

    @staticmethod
    def _is_monotonic(signal: np.ndarray) -> bool:
        diff = np.diff(signal)
        return np.all(diff >= -1e-10) or np.all(diff <= 1e-10)

    @staticmethod
    def _precompute_noise_imf_bank(
        N: int,
        n_ensembles: int,
        max_imfs: int,
        max_sifts_noise: int = 50,
        seed: Optional[int] = None,
        envelope_method: str = "akima",
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        rng = np.random.default_rng(seed)
        noises = rng.standard_normal((n_ensembles, N)).astype(np.float64, copy=False)

        bank: List[List[np.ndarray]] = []
        for i in range(n_ensembles):
            imfs_i = EMDVariants.emd(
                noises[i],
                max_imfs=max_imfs,
                max_sifts=max_sifts_noise,
                sift_threshold=0.1,
                energy_threshold=1e-10,
                envelope_method=envelope_method,
            )
            bank.append([np.asarray(imf, dtype=np.float64) for imf in imfs_i])

        return noises, bank

    @staticmethod
    def _get_noise_component(bank_i: List[np.ndarray], noise_fallback: np.ndarray, k: int) -> np.ndarray:
        if k < len(bank_i):
            return bank_i[k]
        return noise_fallback

    @staticmethod
    def ceemdan(
        signal: np.ndarray,
        noise_std: float = 0.2,
        n_ensembles: int = 100,
        max_imfs: int = 10,
        epsilon: float = 1e-6,
        seed: Optional[int] = 0,
        envelope_method: str = "akima",
        max_sifts_signal: int = 100,
        max_sifts_noise: int = 40,
        **emd_kwargs,
    ) -> List[np.ndarray]:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 20:
            return [x.copy()]

        x_std0 = float(np.std(x))
        if x_std0 < 1e-12 or np.allclose(x, x[0], atol=1e-12):
            return [x.copy()]

        n_ensembles = int(max(1, n_ensembles))
        max_imfs = int(max(1, max_imfs))

        noises, noise_bank = EMDVariants._precompute_noise_imf_bank(
            N=N,
            n_ensembles=n_ensembles,
            max_imfs=max_imfs,
            max_sifts_noise=int(max_sifts_noise),
            seed=seed,
            envelope_method=envelope_method,
        )

        imfs: List[np.ndarray] = []
        residual = x.copy()

        for k in range(max_imfs):
            res_std = float(np.std(residual))
            if res_std < float(epsilon) * x_std0:
                break
            if EMDVariants._is_monotonic(residual):
                break

            beta_k = float(noise_std) / float(k + 1)
            amp = beta_k * res_std

            imf_sum = np.zeros(N, dtype=np.float64)
            valid = 0

            for i in range(n_ensembles):
                e_k = EMDVariants._get_noise_component(noise_bank[i], noises[i], k)
                noisy = residual + amp * e_k

                trial_imfs = EMDVariants.emd(
                    noisy,
                    max_imfs=1,
                    max_sifts=int(max_sifts_signal),
                    envelope_method=envelope_method,
                    **emd_kwargs,
                )
                if len(trial_imfs) > 0:
                    imf_sum += np.asarray(trial_imfs[0], dtype=np.float64)
                    valid += 1

            if valid == 0:
                break

            imf_k = imf_sum / float(valid)
            imfs.append(imf_k)
            residual = residual - imf_k

        if np.sum(residual**2) > (float(epsilon) ** 2) * (np.sum(x**2) + 1e-12):
            imfs.append(residual)

        return imfs

    @staticmethod
    def iceemdan(
        signal: np.ndarray,
        noise_std: float = 0.2,
        n_ensembles: int = 100,
        max_imfs: int = 10,
        epsilon: float = 1e-6,
        seed: Optional[int] = 0,
        envelope_method: str = "akima",
        max_sifts_noise: int = 40,
        **emd_kwargs,
    ) -> List[np.ndarray]:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 20:
            return [x.copy()]

        x_std0 = float(np.std(x))
        if x_std0 < 1e-12 or np.allclose(x, x[0], atol=1e-12):
            return [x.copy()]

        n_ensembles = int(max(1, n_ensembles))
        max_imfs = int(max(1, max_imfs))

        noises, noise_bank = EMDVariants._precompute_noise_imf_bank(
            N=N,
            n_ensembles=n_ensembles,
            max_imfs=max_imfs,
            max_sifts_noise=int(max_sifts_noise),
            seed=seed,
            envelope_method=envelope_method,
        )

        imfs: List[np.ndarray] = []
        residual = x.copy()

        for k in range(max_imfs):
            res_std = float(np.std(residual))
            if res_std < float(epsilon) * x_std0:
                break
            if EMDVariants._is_monotonic(residual):
                break

            beta_k = float(noise_std) / float(k + 1)
            amp = beta_k * res_std

            local_mean_sum = np.zeros(N, dtype=np.float64)
            valid = 0

            for i in range(n_ensembles):
                e_k = EMDVariants._get_noise_component(noise_bank[i], noises[i], k)
                noisy = residual + amp * e_k

                trial_imfs = EMDVariants.emd(
                    noisy,
                    max_imfs=1,
                    max_sifts=1,
                    envelope_method=envelope_method,
                    **emd_kwargs,
                )
                if len(trial_imfs) > 0:
                    first_imf = np.asarray(trial_imfs[0], dtype=np.float64)
                    local_mean_sum += (noisy - first_imf)
                    valid += 1

            if valid == 0:
                break

            avg_local_mean = local_mean_sum / float(valid)
            imf_k = residual - avg_local_mean

            imfs.append(imf_k)
            residual = avg_local_mean

        if np.sum(residual**2) > (float(epsilon) ** 2) * (np.sum(x**2) + 1e-12):
            imfs.append(residual)

        return imfs

    @staticmethod
    def compute_orthogonality_index(imfs: List[np.ndarray]) -> float:
        if len(imfs) < 2:
            return 0.0
        n_imfs = len(imfs)
        total_energy = sum(np.sum(imf**2) for imf in imfs)
        if total_energy < 1e-12:
            return 0.0
        cross_terms = 0.0
        for i in range(n_imfs):
            for j in range(i + 1, n_imfs):
                cross_terms += np.abs(np.sum(imfs[i] * imfs[j]))
        oi = 2 * cross_terms / total_energy
        return float(oi)

    @staticmethod
    def compute_instantaneous_frequency(imf: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        analytic = hilbert(imf)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.gradient(phase) * fs / (2 * np.pi)
        inst_freq = np.clip(inst_freq, 0, fs / 2)
        return inst_freq, amplitude


# -----------------------------------------------------------------------------
# Mode processing + cost
# -----------------------------------------------------------------------------
class ModeProcessor:
    @staticmethod
    def dominant_frequency(sig: np.ndarray, fs: float) -> float:
        x = np.asarray(sig, dtype=np.float64)
        N = x.size
        if N < 8 or np.allclose(x, 0, atol=1e-12):
            return 0.0
        freqs = fftw_np.rfftfreq(N, d=1 / fs)
        spec = np.abs(fftw_np.rfft(x))
        spec = spec.copy()
        if spec.size:
            spec[0] = 0.0
        return float(freqs[int(np.argmax(spec))]) if spec.size else 0.0

    @staticmethod
    def merge_similar_modes(modes: List[np.ndarray], fs: float, freq_tol: float) -> List[np.ndarray]:
        if len(modes) <= 1:
            return modes
        dom = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        used = np.zeros(len(modes), dtype=bool)
        merged: List[np.ndarray] = []
        for i in range(len(modes)):
            if used[i]:
                continue
            group = [modes[i]]
            used[i] = True
            fi = dom[i]
            for j in range(i + 1, len(modes)):
                if used[j]:
                    continue
                fj = dom[j]
                if abs(fi - fj) / max(fi, 1e-6) < freq_tol:
                    group.append(modes[j])
                    used[j] = True
            merged.append(np.sum(group, axis=0))
        return merged

    @staticmethod
    def sort_modes_by_frequency(
        modes: List[np.ndarray], fs: float, low_to_high: bool = True
    ) -> Tuple[List[np.ndarray], List[float]]:
        dom = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        order = np.argsort(dom)
        if not low_to_high:
            order = order[::-1]
        return [modes[i] for i in order], [float(dom[i]) for i in order]

    @staticmethod
    def cost_signal(modes: List[np.ndarray], signal: np.ndarray, fs: float) -> float:
        if len(modes) == 0:
            return 10.0
        x = np.asarray(signal, dtype=np.float64)
        total_energy = np.sum(x**2) + 1e-12
        recon = np.sum(modes, axis=0)
        residual_energy = np.sum((x - recon) ** 2) / total_energy

        dom_freqs = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        overlap_penalty = (
            np.mean(np.exp(-np.diff(np.sort(dom_freqs)))) if len(dom_freqs) > 1 else 0.0
        )

        oi = EMDVariants.compute_orthogonality_index(modes)

        entropy_vals = []
        for m in modes:
            spec = np.abs(fftw_np.rfft(m)) + 1e-12
            p = spec / np.sum(spec)
            H = -np.sum(p * np.log(p + 1e-12))
            Hn = H / np.log(p.size + 1e-12)
            entropy_vals.append(Hn)

        avg_entropy = float(np.mean(entropy_vals))
        return float(0.5 * residual_energy + 0.2 * overlap_penalty + 0.1 * avg_entropy + 0.2 * oi)

    @staticmethod
    def entropy(mode: np.ndarray) -> float:
        x = np.asarray(mode, dtype=np.float64)
        spec = np.abs(fftw_np.rfft(x)) + 1e-12
        p = spec / (np.sum(spec) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))


# -----------------------------------------------------------------------------
# VMD core
# -----------------------------------------------------------------------------
class VMDCore:
    def __init__(self, fftw: FFTWManager):
        self.fftw = fftw
        self.boundary = BoundaryHandler()

    def _prepare_signal(
        self,
        signal: np.ndarray,
        boundary_method: str,
        use_soft_junction: bool,
        window_alpha: Optional[float],
    ) -> Dict[str, Any]:
        """
        Canonical prepare stage:
        - trim odd length (FIX: update orig_len)
        - extend (mirror/reflect/...)
        - optional smooth junction (FIX: taper clamp)
        - optional tukey window
        - compute f_hat_plus, freqs
        Returns everything needed for decompose.
        """
        f = np.asarray(signal, dtype=np.float64)
        orig_len = int(f.size)

        # FIX: keep orig_len consistent if we trim odd length
        if orig_len % 2:
            f = f[:-1]
            orig_len -= 1

        ratio = self.boundary.adaptive_extension_ratio(f)
        fMirr, left_ext, right_ext = self.boundary.extend_signal(
            f, method=boundary_method, extension_ratio=ratio
        )

        if use_soft_junction and boundary_method != "none" and left_ext > 0:
            fMirr = self.boundary.smooth_edge_junction(
                fMirr, original_len=orig_len, ext_len=left_ext
            )

        if window_alpha is None:
            window_alpha = self.boundary.auto_window_alpha(f)
        if window_alpha and window_alpha > 0:
            from scipy.signal import windows
            win = windows.tukey(len(fMirr), alpha=float(window_alpha))
            fMirr = fMirr * win

        T = int(len(fMirr))
        freqs = np.fft.fftshift(np.fft.fftfreq(T))
        f_hat = fftw_np.fftshift(fftw_np.fft(fMirr))
        f_hat_plus = f_hat.copy()
        f_hat_plus[: T // 2] = 0

        # FIX: store fMirr for init==3 even when fft is cached
        return {
            "orig_len": orig_len,
            "boundary_method": boundary_method,
            "fMirr": fMirr,
            "left_ext": int(left_ext),
            "right_ext": int(right_ext),
            "T": T,
            "half_T": T // 2,
            "freqs": freqs,
            "f_hat_plus": f_hat_plus,
        }

    def precompute_fft(
        self,
        signal: np.ndarray,
        boundary_method: str = "reflect",
        use_soft_junction: bool = False,
        window_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._prepare_signal(signal, boundary_method, use_soft_junction, window_alpha)

    @staticmethod
    def _init_from_spectrum(real_signal: np.ndarray, K: int) -> np.ndarray:
        s = np.asarray(real_signal, dtype=np.float64)
        spec = np.abs(fftw_np.rfft(s))
        freqs = fftw_np.rfftfreq(s.size)
        if spec.size == 0:
            return np.zeros(K, dtype=np.float64)
        peak_idx = np.argsort(spec)[-K:]
        return np.sort(freqs[peak_idx]).astype(np.float64)

    def decompose(
        self,
        signal: np.ndarray,
        alpha: float,
        tau: float,
        K: int,
        DC: int,
        init: int,
        tol: float,
        max_iter: int,
        fs: float = 1.0,
        use_fs_vmd: bool = False,
        precomputed_fft: Optional[Dict[str, Any]] = None,
        boundary_method: str = "reflect",
        use_soft_junction: bool = False,
        window_alpha: Optional[float] = None,
        trial: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core VMD decomposition (real signal).
        Returns:
            u: (K, N)
            u_hat_full: (T, K)
            omega: (K,)
        """
        x = np.asarray(signal, dtype=np.float64)

        # FS-VMD: simple demod/remod around dominant freq (real-only safe)
        if use_fs_vmd:
            f0 = SignalAnalyzer.dominant_freq(x, fs)
            t = np.arange(x.size, dtype=np.float64) / float(fs)
            x_work = x * np.cos(2 * np.pi * f0 * t)
        else:
            f0 = 0.0
            x_work = x

        if precomputed_fft is None:
            precomputed_fft = self._prepare_signal(
                x_work,
                boundary_method=boundary_method,
                use_soft_junction=use_soft_junction,
                window_alpha=window_alpha,
            )

        f_hat_plus = precomputed_fft["f_hat_plus"]
        freqs = precomputed_fft["freqs"]
        T = int(precomputed_fft["T"])
        half_T = int(precomputed_fft["half_T"])
        orig_len = int(precomputed_fft["orig_len"])
        left_ext = int(precomputed_fft["left_ext"])
        boundary_method = precomputed_fft.get("boundary_method", boundary_method)
        fMirr = precomputed_fft.get("fMirr", None)

        Alpha = (alpha * np.ones(K)).astype(np.float64)
        omega = np.zeros(K, dtype=np.float64)

        if init == 1:
            omega = (np.arange(K) * (0.5 / K)).astype(np.float64)
        elif init == 2:
            fs0 = 1.0 / T
            omega = np.sort(
                np.exp(np.log(fs0) + (np.log(0.5) - np.log(fs0)) * np.random.rand(K))
            ).astype(np.float64)
        elif init == 3:
            base = np.asarray(fMirr if (fMirr is not None) else x_work, dtype=np.float64)
            omega = self._init_from_spectrum(base, K)

        if DC:
            omega[0] = 0.0

        lam = np.zeros(len(freqs), dtype=np.complex128)
        u_hat_prev = np.zeros((len(freqs), K), dtype=np.complex128)

        uDiff = float(tol) + 1.0
        prev_diff = float("inf")
        stagnation = 0
        adaptive_tol = float(max(tol, 1e-7))

        for n in range(int(max_iter)):
            sum_uk = np.sum(u_hat_prev, axis=1).astype(np.complex128)

            u_hat_next, omega_next, lam_next, diff_norm = update_modes_numba(
                freqs, half_T, f_hat_plus,
                sum_uk, lam, Alpha, omega, u_hat_prev, K, tau
            )

            if n > 5:
                uDiff = float(diff_norm)

            if n > 10:
                if abs(diff_norm - prev_diff) < adaptive_tol * 0.1:
                    stagnation += 1
                else:
                    stagnation = 0
                if stagnation >= 5:
                    break

            if trial is not None and n % 15 == 0:
                trial.report(uDiff, step=n)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            u_hat_prev = u_hat_next
            lam = lam_next
            omega = omega_next
            prev_diff = float(diff_norm)

            if uDiff <= tol:
                break

        # rebuild full spectrum
        u_hat_full = np.zeros((T, K), dtype=np.complex128)
        u_hat_full[half_T:T, :] = u_hat_prev[half_T:T, :]
        idxs = np.arange(1, half_T)
        u_hat_full[idxs, :] = np.conj(u_hat_full[T - idxs, :])
        u_hat_full[0, :] = np.conj(u_hat_full[-1, :])

        u = np.real(fftw_np.ifft(fftw_np.ifftshift(u_hat_full, axes=0), axis=0)).T  # (K, T)

        # crop back to original length
        if boundary_method != "none":
            start = left_ext
            end = start + orig_len
            u = u[:, start:end]

        # safety: ensure exact length
        if u.shape[1] != orig_len:
            x_old = np.linspace(0, 1, u.shape[1])
            x_new = np.linspace(0, 1, orig_len)
            u = np.vstack([np.interp(x_new, x_old, uk) for uk in u])

        # remodulate if FS-VMD
        if use_fs_vmd and f0 != 0.0:
            t = np.arange(orig_len, dtype=np.float64) / float(fs)
            carrier = np.cos(2 * np.pi * f0 * t)
            u = u * carrier[None, :]

        return u, u_hat_full, omega

    def decompose_multivariate(
        self,
        signals: np.ndarray,  # (channels, samples)
        alpha: float,
        tau: float,
        K: int,
        DC: int,
        init: int,
        tol: float,
        max_iter: int,
        boundary_method: str,
        use_soft_junction: bool,
        window_alpha: Optional[float],
        fs: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Minimal MVMD: decompose each channel and average omegas."""
        X = np.asarray(signals, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("MVMD expects signals shaped (channels, samples)")
        C, N = X.shape

        modes = np.zeros((C, K, N), dtype=np.float64)
        omega_shared = np.zeros(K, dtype=np.float64)

        precomps = [
            self.precompute_fft(
                X[ch],
                boundary_method=boundary_method,
                use_soft_junction=use_soft_junction,
                window_alpha=window_alpha,
            )
            for ch in range(C)
        ]

        for ch in range(C):
            u, _, om = self.decompose(
                X[ch],
                alpha=alpha,
                tau=tau,
                K=K,
                DC=DC,
                init=init,
                tol=tol,
                max_iter=max_iter,
                fs=fs,
                use_fs_vmd=False,
                precomputed_fft=precomps[ch],
                boundary_method=boundary_method,
                use_soft_junction=use_soft_junction,
                window_alpha=window_alpha,
            )
            modes[ch] = u
            omega_shared = (omega_shared * ch + om) / float(ch + 1)

        return modes, None, omega_shared

