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
- Joint MVMD (shared omega enforced during iteration)
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

import os
import pickle
import warnings

import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fftw_np
import scipy.stats

from .config import VMDParameters
from .emd import EMDVariants

# -----------------------------------------------------------------------------
# Optional torch refinement
# -----------------------------------------------------------------------------
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None


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
                total_boxes += k_max - k_min + 1

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
    def estimate(signal: np.ndarray, method: str = "auto") -> float:
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

    estimate_complexity = estimate


def fractal_dimension(signal: np.ndarray, **kwargs) -> float:
    """
    Composite fractal-dimension estimate.

    Despite the legacy name ``box_counting_dimension``, this function does not
    return a pure box-counting dimension. It returns the median of the Higuchi,
    differential box-counting, and Katz estimates for a more stable aggregate
    complexity score.
    """
    return FractalDimension.estimate(signal, method="auto")


box_counting_dimension = fractal_dimension


# -----------------------------------------------------------------------------
# FFTW
# -----------------------------------------------------------------------------
class FFTWManager:
    def __init__(self, wisdom_file: str = "vmd_fftw_wisdom.dat"):
        self.wisdom_file = wisdom_file
        self._warned_backend_keys = set()
        self._setup_fftw()
        self.load_wisdom()

    def _setup_fftw(self):
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(7200)
        pyfftw.config.NUM_THREADS = max(1, os.cpu_count() or 4)

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

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_backend_keys:
            return
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        self._warned_backend_keys.add(key)

    def resolve_backend(
        self, backend: str = "fftw", device: str = "auto"
    ) -> tuple[str, str]:
        name = str(backend).lower()
        dev = str(device).lower()

        if name not in ("fftw", "torch"):
            self._warn_once(
                f"fft_backend_invalid:{name}",
                f"Unknown fft_backend={backend!r}; falling back to 'fftw'.",
            )
            return "fftw", "cpu"

        if name == "torch":
            if not TORCH_AVAILABLE or torch is None:
                self._warn_once(
                    "fft_backend_torch_missing",
                    "fft_backend='torch' requested but PyTorch is unavailable; "
                    "falling back to 'fftw'.",
                )
                return "fftw", "cpu"

            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            elif dev not in ("cpu", "cuda"):
                self._warn_once(
                    f"fft_device_invalid:{dev}",
                    f"Unknown fft_device={device!r}; using 'auto' resolution.",
                )
                dev = "cuda" if torch.cuda.is_available() else "cpu"

            if dev == "cuda" and not torch.cuda.is_available():
                self._warn_once(
                    "fft_backend_cuda_unavailable",
                    "fft_device='cuda' requested but CUDA is unavailable; "
                    "falling back to CPU torch FFT.",
                )
                dev = "cpu"

            return "torch", dev

        return "fftw", "cpu"

    @staticmethod
    def _numpy_axes(axes, ndim: int):
        if axes is None:
            return tuple(range(ndim))
        if isinstance(axes, tuple):
            return axes
        if isinstance(axes, list):
            return tuple(axes)
        return int(axes)

    @staticmethod
    def _to_numpy(x):
        if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _to_torch(x, device: str):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, device=device)
        if isinstance(x, torch.Tensor):
            return x.to(device=device)
        return torch.as_tensor(np.asarray(x), device=device)

    def fft(self, x, axis: int = -1, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            return self._to_numpy(torch.fft.fft(xt, dim=axis))
        return fftw_np.fft(np.asarray(x), axis=axis)

    def ifft(self, x, axis: int = -1, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            return self._to_numpy(torch.fft.ifft(xt, dim=axis))
        return fftw_np.ifft(np.asarray(x), axis=axis)

    def rfft(self, x, axis: int = -1, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            return self._to_numpy(torch.fft.rfft(xt, dim=axis))
        return fftw_np.rfft(np.asarray(x), axis=axis)

    def fftshift(self, x, axes=None, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            dims = self._numpy_axes(axes, xt.ndim)
            return self._to_numpy(torch.fft.fftshift(xt, dim=dims))
        return fftw_np.fftshift(np.asarray(x), axes=axes)

    def ifftshift(self, x, axes=None, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            dims = self._numpy_axes(axes, xt.ndim)
            return self._to_numpy(torch.fft.ifftshift(xt, dim=dims))
        return fftw_np.ifftshift(np.asarray(x), axes=axes)

    def fftfreq(
        self, n: int, d: float = 1.0, backend: str = "fftw", device: str = "auto"
    ):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            return self._to_numpy(
                torch.fft.fftfreq(int(n), d=float(d), device=dev, dtype=torch.float64)
            )
        return np.fft.fftfreq(int(n), d=float(d))

    def rfftfreq(
        self, n: int, d: float = 1.0, backend: str = "fftw", device: str = "auto"
    ):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            return self._to_numpy(
                torch.fft.rfftfreq(int(n), d=float(d), device=dev, dtype=torch.float64)
            )
        return fftw_np.rfftfreq(int(n), d=float(d))


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
    ) -> tuple[np.ndarray, int, int]:
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
            out = np.concatenate([np.full(ext_len, s[0]), s, np.full(ext_len, s[-1])])

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
    def taper_boundaries(
        modes: list[np.ndarray], taper_length: int
    ) -> list[np.ndarray]:
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
# Signal analysis (improved 2026 version)
# -----------------------------------------------------------------------------
class SignalAnalyzer:
    @staticmethod
    def estimate_snr(signal: np.ndarray, fs: float) -> float:
        """Improved SNR: uses lower percentile of log-spectrum for noise floor."""
        s = np.asarray(signal, dtype=np.float64)
        spec = np.abs(fftw_np.rfft(s)) + 1e-12
        spec_db = 20 * np.log10(spec)

        # Take ~bottom 40% as noise-dominated region (conservative)
        sorted_db = np.sort(spec_db)
        noise_floor_db = np.median(sorted_db[: int(0.4 * len(sorted_db))])

        # Signal power approximated by peak + some margin
        signal_power_db = np.max(spec_db)

        snr_db = float(signal_power_db - noise_floor_db)
        return np.clip(snr_db, -10.0, 60.0)  # realistic bounds

    @staticmethod
    def dominant_freq(signal: np.ndarray, fs: float) -> float:
        # unchanged — it's already good
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
    def assess_complexity(
        signal: np.ndarray,
        fs: float,
        entropy_estimator=None,  # optional: pass your _multi_scale_dispersion_entropy if available
    ) -> VMDParameters:
        s = np.asarray(signal, dtype=np.float64)
        N = s.size
        if N < 32:
            return VMDParameters(  # very short → conservative
                n_trials=8,
                max_K=3,
                tol=1e-5,
                alpha_min=1500,
                alpha_max=4000,
            )

        # Spectral features
        spec = np.abs(fftw_np.rfft(s)) + 1e-12
        freqs = fftw_np.rfftfreq(N, d=1 / fs)
        spec_norm = spec / (np.sum(spec) + 1e-12)
        spectral_entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-12))

        # Bandwidth-like spread (more robust: 10 dB down from peak)
        spec2 = spec.copy()
        spec2[0] = 0
        peak_val = np.max(spec2)
        thr = peak_val / np.sqrt(10)  # ≈ 10 dB down
        significant_bins = np.sum(spec2 > thr)
        spread = float(significant_bins) / max(1, len(freqs))

        # Time-domain variability
        variability = float(np.std(np.diff(s))) / (float(np.std(s)) + 1e-12)

        # Kurtosis (captures impulsiveness / transients)
        kurt = float(scipy.stats.kurtosis(s, fisher=True))

        # Optional: dispersion entropy if you pass the function
        de = 0.5
        if entropy_estimator is not None:
            try:
                de = entropy_estimator(s)  # assume normalized [0,1]
            except:
                pass

        # Normalized features (roughly [0,1] range)
        se_norm = np.clip(spectral_entropy / 10.0, 0.0, 1.0)
        spread_norm = np.clip(spread, 0.0, 1.0)
        var_norm = np.clip(variability / 3.0, 0.0, 1.0)  # cap outliers
        kurt_norm = np.clip(kurt / 15.0, 0.0, 1.0)  # reasonable cap

        # Composite complexity score (weighted)
        complexity_score = (
            0.35 * se_norm
            + 0.25 * spread_norm
            + 0.20 * var_norm
            + 0.15 * kurt_norm
            + 0.05 * de
        )
        complexity_score = np.clip(complexity_score, 0.0, 1.2)

        # SNR-aware adjustment
        snr_db = SignalAnalyzer.estimate_snr(s, fs)

        # Smooth mapping to parameters (continuous instead of thresholds)
        base_K = 3 + int(7 * complexity_score)  # 3 → 10
        base_alpha_min = 300 + 1700 * (1 - complexity_score)
        base_alpha_max = 2500 + 7500 * complexity_score

        p = VMDParameters(
            n_trials=int(max(10, min(35, N // 120))),
            max_K=min(max(3, base_K), 10),
            tol=1e-6 * (1 + 0.5 * complexity_score),
            alpha_min=base_alpha_min,
            alpha_max=base_alpha_max,
        )

        # SNR corrections (multiplicative)
        if snr_db < 8:
            p.alpha_min *= 2.2
            p.alpha_max *= 2.8
            p.max_K = max(2, p.max_K // 2)
            p.n_trials = max(8, p.n_trials // 2)
        elif snr_db < 18:
            p.alpha_min *= 1.6
            p.alpha_max *= 1.7
        elif snr_db > 30:
            p.max_K = min(p.max_K + 2, 12)
            p.alpha_min *= 0.75
            p.alpha_max *= 0.8

        # Keep the alpha search interval ordered and non-degenerate after all
        # multiplicative corrections.
        p.alpha_max = max(p.alpha_max, p.alpha_min * 1.5)

        # Long signal corrections
        if N > 4000:
            p.n_trials = max(10, p.n_trials // 2)
            p.tol *= 1.8

        # Diagnostics
        print(
            f"complexity={complexity_score:.3f} | SNR={snr_db:.1f} dB | "
            f"dom≈{SignalAnalyzer.dominant_freq(s, fs):.2f} Hz | "
            f"kurt={kurt:.1f}"
        )
        print(
            f" => trials={p.n_trials}, max_K={p.max_K}, "
            f"alpha∈[{p.alpha_min:.0f}, {p.alpha_max:.0f}]"
        )

        return p


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
    def merge_similar_modes(
        modes: list[np.ndarray], fs: float, freq_tol: float
    ) -> list[np.ndarray]:
        if len(modes) <= 1:
            return modes
        dom = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        used = np.zeros(len(modes), dtype=bool)
        merged: list[np.ndarray] = []
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
        modes: list[np.ndarray], fs: float, low_to_high: bool = True
    ) -> tuple[list[np.ndarray], list[float]]:
        dom = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        order = np.argsort(dom)
        if not low_to_high:
            order = order[::-1]
        return [modes[i] for i in order], [float(dom[i]) for i in order]

    @staticmethod
    def cost_signal(modes: list[np.ndarray], signal: np.ndarray, fs: float) -> float:
        if len(modes) == 0:
            return 10.0
        x = np.asarray(signal, dtype=np.float64)
        total_energy = np.sum(x**2) + 1e-12
        recon = np.sum(modes, axis=0)
        residual_energy = np.sum((x - recon) ** 2) / total_energy

        dom_freqs = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        if len(dom_freqs) > 1:
            gaps = np.diff(np.sort(dom_freqs))
            overlap_penalty = float(np.mean(np.exp(-gaps / (np.median(gaps) + 1e-12))))
        else:
            overlap_penalty = 0.0

        oi = EMDVariants.compute_orthogonality_index(modes)

        entropy_vals = []
        for m in modes:
            spec = np.abs(fftw_np.rfft(m)) + 1e-12
            p = spec / np.sum(spec)
            H = -np.sum(p * np.log(p + 1e-12))
            Hn = H / np.log(p.size + 1e-12)
            entropy_vals.append(Hn)

        avg_entropy = float(np.mean(entropy_vals))
        return float(
            0.5 * residual_energy + 0.2 * overlap_penalty + 0.1 * avg_entropy + 0.2 * oi
        )

    @staticmethod
    def entropy(mode: np.ndarray) -> float:
        x = np.asarray(mode, dtype=np.float64)
        spec = np.abs(fftw_np.rfft(x)) + 1e-12
        p = spec / (np.sum(spec) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))
