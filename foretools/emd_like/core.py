"""
VMDCore — Variational Mode Decomposition (SOTA-optimized)

Improvements over baseline
──────────────────────────
  Auto-K        : fast Welch-PSD peak counter + accurate VMD-residual-entropy sweep
  VNCMD / NCMD  : sparse quadrature-envelope solve with iterative IF-track refinement
  True MVMD     : joint Gauss-Seidel sweep with pooled cross-channel ω update
                  (Rehman & Aftab 2019) — not post-hoc channel averaging
  ADMM          : over-relaxed dual update (ρ ∈ [1, 2), default 1.6)
  Anderson mix  : Type-1 acceleration (m=5) on the u_hat fixed-point
  Dual criterion: converge when BOTH relative mode-sum Δu < tol AND max|Δω| < ω_tol
  Init-4        : Hilbert IF histogram — robust for AM-FM signals
  Warm-start    : init=5, reuse previous solution for rolling/streaming use
  Gram-Schmidt  : optional frequency-domain orthogonalisation every N iters
  Adaptive α    : bandwidth-driven per-mode alpha (unchanged, kept)
  Neural refine : Transformer-based per-mode and cross-mode refiners
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import optuna
from numba import njit
from scipy.signal import find_peaks, hilbert, spectrogram, welch
from scipy.sparse import bmat, csc_matrix, diags, eye
from scipy.sparse.linalg import factorized, spsolve


try:
    from .common import BoundaryHandler, FFTWManager
except Exception:
    from vmd_common import BoundaryHandler, FFTWManager

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

try:
    from mamba_ssm import Mamba as _Mamba

    MAMBA_AVAILABLE = True
except Exception:
    MAMBA_AVAILABLE = False
    _Mamba = None


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level signal-analysis helpers
# ─────────────────────────────────────────────────────────────────────────────


def _hilbert_if(u: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Instantaneous frequency (normalised 0..0.5) of real signal u
    via Hilbert analytic signal + phase unwrapping.
    """
    analytic = hilbert(u)
    phase = np.unwrap(np.angle(analytic))
    if_est = np.diff(phase) / (2.0 * np.pi)  # cycles / sample
    return np.concatenate([[if_est[0]], if_est])  # same length as u


def _restore_hermitian_from_positive(u_hat_pos: np.ndarray, half_T: int) -> np.ndarray:
    """
    Rebuild full fftshifted Hermitian spectrum from its non-negative half.

    Parameters
    ----------
    u_hat_pos : (T, K) complex
        Spectrum with valid bins in [half_T:] (DC + positive frequencies).
        For even T, the Nyquist bin at index 0 must also be preserved.
    half_T : int
        Index of the DC bin in fftshifted ordering.
    """
    T, _ = u_hat_pos.shape
    out = np.zeros_like(u_hat_pos, dtype=np.complex128)
    out[half_T:, :] = u_hat_pos[half_T:, :]

    if T % 2 == 0:
        # Even T: bins 1..half_T-1 mirror from T-1..half_T+1.
        if half_T > 1:
            out[1:half_T, :] = np.conj(out[T - 1 : half_T : -1, :])
        # Nyquist bin (index 0 in fftshifted layout) is self-conjugate.
        out[0, :] = np.real(u_hat_pos[0, :]) + 0.0j
    else:
        # Odd T: bins 0..half_T-1 mirror from T-1..half_T+1.
        if half_T > 0:
            out[0:half_T, :] = np.conj(out[T - 1 : half_T : -1, :])

    # DC bin must also be self-conjugate for real time-domain reconstruction.
    out[half_T, :] = np.real(out[half_T, :]) + 0.0j
    return out


def _restore_hermitian_from_positive_mv(
    u_hat_pos: np.ndarray, half_T: int
) -> np.ndarray:
    """
    Channel-wise variant of ``_restore_hermitian_from_positive`` for (C, T, K).
    For even T, the Nyquist bin at index 0 must also be preserved.
    """
    C, T, K = u_hat_pos.shape
    out = np.zeros((C, T, K), dtype=np.complex128)
    out[:, half_T:, :] = u_hat_pos[:, half_T:, :]

    if T % 2 == 0:
        if half_T > 1:
            out[:, 1:half_T, :] = np.conj(out[:, T - 1 : half_T : -1, :])
        out[:, 0, :] = np.real(u_hat_pos[:, 0, :]) + 0.0j
    else:
        if half_T > 0:
            out[:, 0:half_T, :] = np.conj(out[:, T - 1 : half_T : -1, :])

    out[:, half_T, :] = np.real(out[:, half_T, :]) + 0.0j
    return out


def _gram_schmidt_freq(u_hat: np.ndarray, half_T: int) -> np.ndarray:
    """
    Sequential frequency-domain Gram-Schmidt orthogonalisation.
    Operates on the positive-frequency half [half_T:] of u_hat (T, K),
    then restores full Hermitian symmetry.
    """
    T, K = u_hat.shape
    U = u_hat[half_T:, :].copy()

    for k in range(1, K):
        for j in range(k):
            denom = float(np.sum(U[:, j].real ** 2 + U[:, j].imag ** 2)) + 1e-14
            proj = np.dot(U[:, j].conj(), U[:, k]) / denom
            U[:, k] -= proj * U[:, j]

    out = u_hat.copy()
    out[half_T:, :] = U
    return _restore_hermitian_from_positive(out, half_T)


# ─────────────────────────────────────────────────────────────────────────────
#  Anderson mixing (Type-1)
# ─────────────────────────────────────────────────────────────────────────────


class AndersonMixer:
    """
    Type-1 Anderson mixing for fixed-point iteration acceleration.

    Call ``apply(x, F(x))`` each iteration to get the accelerated next iterate.
    Works transparently on real *and* complex arrays: complex128 arrays are
    viewed as twice-as-long float64 vectors for the internal least-squares.

    Reference: Toth & Walker (2015), "Convergence analysis for Anderson mixing".
    """

    def __init__(self, m: int = 5, beta: float = 1.0):
        self.m = max(1, int(m))
        self.beta = float(beta)
        self._hist: list[tuple[np.ndarray, np.ndarray]] = []  # (x_r, g_r) pairs

    def reset(self) -> None:
        self._hist.clear()

    def apply(self, x: np.ndarray, Fx: np.ndarray) -> np.ndarray:
        """Return accelerated iterate given current x and map output F(x)."""
        # View arrays as real vectors regardless of dtype
        xr = np.ascontiguousarray(x).view(np.float64).ravel()
        Fxr = np.ascontiguousarray(Fx).view(np.float64).ravel()
        gr = Fxr - xr

        self._hist.append((xr.copy(), gr.copy()))
        if len(self._hist) > self.m + 1:
            self._hist.pop(0)

        m_cur = len(self._hist)
        if m_cur < 2:
            return Fx.copy()

        dG = np.column_stack(
            [self._hist[i + 1][1] - self._hist[i][1] for i in range(m_cur - 1)]
        )
        dX = np.column_stack(
            [self._hist[i + 1][0] - self._hist[i][0] for i in range(m_cur - 1)]
        )

        g_n = self._hist[-1][1]
        # Use the provided F(x_n) directly to avoid tiny layout/view-dependent
        # reconstruction differences from x_n + g_n.
        Fxr_n = Fxr

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                theta, _, _, _ = np.linalg.lstsq(dG, g_n, rcond=None)
        except np.linalg.LinAlgError:
            return Fx.copy()

        # x_{n+1} = F(x_n) − (dX + β·dG) θ*
        accel_r = Fxr_n - (dX + self.beta * dG) @ theta
        return accel_r.view(x.dtype).reshape(x.shape)


# ─────────────────────────────────────────────────────────────────────────────
#  VMDCore
# ─────────────────────────────────────────────────────────────────────────────


class VMDCore:
    def __init__(self, fftw: FFTWManager):
        self.fftw = fftw
        self.boundary = BoundaryHandler()

    def _spectral_entropy_backend(
        self,
        signal: np.ndarray,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> float:
        spec = (
            np.abs(self.fftw.rfft(signal, backend=fft_backend, device=fft_device)) ** 2
        )
        total = spec.sum()
        if total < 1e-18:
            return 1.0
        p = spec / total
        return float(np.clip(-np.sum(p * np.log(p + 1e-16)) / np.log(len(p)), 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────────
    #  Signal preparation (unchanged API)
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_signal(
        self,
        signal: np.ndarray,
        boundary_method: str,
        use_soft_junction: bool,
        window_alpha: float | None,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> dict[str, Any]:
        """
        Canonical prepare stage: trim, extend, optional soft-junction & Tukey
        window, compute fft-shifted analytic spectrum and frequency grid.
        [WARNING: Tukey window conflicts with boundary extension; use one or the other.]
        """
        fft_backend, fft_device = self.fftw.resolve_backend(fft_backend, fft_device)
        f = np.asarray(signal, dtype=np.float64)
        orig_len = int(f.size)
        if orig_len % 2:
            f = f[:-1]
            orig_len -= 1

        if window_alpha is not None and window_alpha > 0 and boundary_method != "none":
            raise ValueError(
                f"window_alpha with boundary_method={boundary_method!r} is unsupported: "
                "use either boundary extension or Tukey windowing, not both."
            )

        ratio = (
            0.5
            if boundary_method == "mirror"
            else self.boundary.adaptive_extension_ratio(f)
        )
        fMirr, left_ext, right_ext = self.boundary.extend_signal(
            f, method=boundary_method, extension_ratio=ratio
        )

        if use_soft_junction and boundary_method != "none" and left_ext > 0:
            fMirr = self.boundary.smooth_edge_junction(
                fMirr, original_len=orig_len, ext_len=left_ext
            )

        if window_alpha is not None and window_alpha > 0:
            from scipy.signal import windows

            fMirr = fMirr * windows.tukey(len(fMirr), alpha=float(window_alpha))

        T = int(len(fMirr))
        if left_ext + orig_len + right_ext != T:
            raise ValueError(
                "Boundary extension bookkeeping mismatch: "
                f"left_ext({left_ext}) + orig_len({orig_len}) + "
                f"right_ext({right_ext}) != T({T})"
            )
        freqs = self.fftw.fftshift(
            self.fftw.fftfreq(T, backend=fft_backend, device=fft_device),
            backend=fft_backend,
            device=fft_device,
        )
        f_hat = self.fftw.fftshift(
            self.fftw.fft(fMirr, backend=fft_backend, device=fft_device),
            backend=fft_backend,
            device=fft_device,
        )
        f_hat_plus = f_hat.copy()
        if T % 2 == 0:
            # In fftshifted ordering, index 0 is the Nyquist bin for even T.
            # Keep it as a real self-conjugate bin and zero only the strictly
            # negative frequencies 1..half_T-1.
            # The iterative mode updates operate on the full T-bin spectra, so
            # preserving this bin allows Nyquist energy to be apportioned across
            # modes instead of being discarded up front.
            f_hat_plus[0] = np.real(f_hat_plus[0]) + 0.0j
            f_hat_plus[1 : T // 2] = 0.0
        else:
            f_hat_plus[: T // 2] = 0.0

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
            "fft_backend": str(fft_backend),
            "fft_device": str(fft_device),
        }

    def precompute_fft(
        self,
        signal: np.ndarray,
        boundary_method: str = "mirror",
        use_soft_junction: bool = False,
        window_alpha: float | None = None,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> dict[str, Any]:
        return self._prepare_signal(
            signal,
            boundary_method,
            use_soft_junction,
            window_alpha,
            fft_backend,
            fft_device,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Initialisations
    # ──────────────────────────────────────────────────────────────────────────

    def _init_from_spectrum(
        self,
        real_signal: np.ndarray,
        K: int,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> np.ndarray:
        s = np.asarray(real_signal, dtype=np.float64)
        spec = np.abs(self.fftw.rfft(s, backend=fft_backend, device=fft_device))
        fq = self.fftw.rfftfreq(s.size, backend=fft_backend, device=fft_device)
        if K <= 0:
            return np.zeros(0, dtype=np.float64)
        if spec.size == 0 or fq.size == 0:
            return np.zeros(K, dtype=np.float64)

        work = spec.astype(np.float64, copy=True)
        if work.size > 0:
            work[0] = 0.0  # ignore DC for centre-frequency initialisation

        max_h = float(np.max(work)) if work.size else 0.0
        if max_h > 0.0:
            dist = max(1, work.size // max(2, 2 * K))
            peaks, _ = find_peaks(
                work,
                height=0.01 * max_h,
                prominence=0.005 * max_h,
                distance=dist,
            )
        else:
            peaks = np.empty(0, dtype=np.int64)

        selected: list[int] = []
        if peaks.size > 0:
            top = peaks[np.argsort(work[peaks])[-min(K, peaks.size) :]]
            selected.extend(int(i) for i in top.tolist())

        if len(selected) < K:
            order = np.argsort(work)[::-1]
            used = set(selected)
            for idx in order.tolist():
                if idx in used:
                    continue
                selected.append(int(idx))
                used.add(int(idx))
                if len(selected) >= K:
                    break

        if len(selected) < K:
            # Pad deterministically if spectrum is too short.
            selected.extend([0] * (K - len(selected)))

        return np.sort(fq[np.asarray(selected[:K], dtype=np.int64)]).astype(np.float64)

    @staticmethod
    def _init_hilbert(signal: np.ndarray, K: int) -> np.ndarray:
        """
        Estimate K centre frequencies from an amplitude-weighted instantaneous-
        frequency (IF) histogram obtained via the Hilbert transform.

        More robust than raw FFT peak picking for AM-FM and chirp signals
        because it weights spectral regions by instantaneous signal energy,
        not aggregate power, so narrow-band bursts are not buried by broad ones.
        """
        s = np.asarray(signal, dtype=np.float64)
        analyt = hilbert(s)
        amp = np.abs(analyt[:-1])
        if_est = np.clip(
            np.abs(np.diff(np.unwrap(np.angle(analyt))) / (2.0 * np.pi)), 0.0, 0.5
        )

        hist, edges = np.histogram(if_est, bins=256, weights=amp)
        centres = 0.5 * (edges[:-1] + edges[1:])

        pks, _ = find_peaks(
            hist, height=0.01 * hist.max(), distance=max(1, 256 // (2 * K))
        )
        if len(pks) >= K:
            top = pks[np.argsort(hist[pks])[-K:]]
            return np.sort(centres[top]).astype(np.float64)

        # Fallback: uniform
        return (np.arange(K) * (0.5 / K)).astype(np.float64)

    # ──────────────────────────────────────────────────────────────────────────
    #  Auto-K selection
    # ──────────────────────────────────────────────────────────────────────────

    def estimate_K_fast(
        self,
        signal: np.ndarray,
        K_min: int = 2,
        K_max: int = 10,
        min_peak_energy_ratio: float = 0.02,
        nperseg_frac: float = 0.25,
    ) -> int:
        """
        O(N log N) K estimator based on Welch PSD peak counting.
        Use as a cheap pre-filter before ``estimate_K``.
        """
        x = np.asarray(signal, dtype=np.float64)
        nperseg = max(16, int(len(x) * nperseg_frac))
        _, psd = welch(x, nperseg=nperseg)
        psd_n = psd / (psd.max() + 1e-16)
        dist = max(1, len(psd_n) // (K_max + 2))
        pks, _ = find_peaks(
            psd_n,
            height=min_peak_energy_ratio,
            prominence=min_peak_energy_ratio * 0.5,
            distance=dist,
        )
        return int(np.clip(len(pks), K_min, K_max))

    def estimate_K(
        self,
        signal: np.ndarray,
        K_min: int = 2,
        K_max: int = 8,
        alpha: float = 2000.0,
        tol: float = 1e-6,
        max_iter: int = 150,
        energy_threshold: float = 0.01,
        entropy_gain_threshold: float = 0.015,
        boundary_method: str = "mirror",
        use_fast_primary: bool = True,
        refine_margin: int = 2,
        fast_min_peak_energy_ratio: float = 0.02,
        fast_nperseg_frac: float = 0.25,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> int:
        """
        Accurate incremental-VMD K estimator with two complementary stopping rules.
        By default, ``estimate_K_fast`` is used as a cheap front-end and the
        expensive VMD sweep is used only as a local refinement.

        1. **Mode energy ratio** < ``energy_threshold`` — the newest mode carries
           negligible energy; additional modes are mostly noise.
        2. **Residual spectral-entropy gain** < ``entropy_gain_threshold`` — the
           residual is no longer becoming more structured; stop.

        Both conditions must be met to stop early (conservative by default).
        A candidate K is accepted as useful if at least one condition is still
        informative (i.e. not both below threshold).
        Returns K ∈ [K_min, K_max]. Set ``use_fast_primary=False`` to recover a
        full incremental sweep up to ``K_max``.
        """
        x = np.asarray(signal, dtype=np.float64)
        precomputed_fft = self._prepare_signal(
            x,
            boundary_method=boundary_method,
            use_soft_junction=False,
            window_alpha=None,
            fft_backend=fft_backend,
            fft_device=fft_device,
        )
        x_entropy = np.asarray(precomputed_fft["fMirr"], dtype=np.float64)
        total_energy = float(np.sum(x**2)) + 1e-16
        prev_entropy = self._spectral_entropy_backend(
            x_entropy, fft_backend=fft_backend, fft_device=fft_device
        )
        best_K = K_min
        search_max = int(K_max)

        if use_fast_primary:
            k_fast = self.estimate_K_fast(
                x,
                K_min=K_min,
                K_max=K_max,
                min_peak_energy_ratio=float(fast_min_peak_energy_ratio),
                nperseg_frac=float(fast_nperseg_frac),
            )
            if int(refine_margin) <= 0:
                return int(k_fast)
            search_max = min(K_max, int(k_fast) + max(0, int(refine_margin)))

        for k in range(K_min, search_max + 1):
            u, u_hat_full, _ = self.decompose(
                x,
                alpha=alpha,
                tau=0.0,
                K=k,
                DC=0,
                init=1,
                tol=tol,
                max_iter=max_iter,
                precomputed_fft=precomputed_fft,
                boundary_method=boundary_method,
                fft_backend=fft_backend,
                fft_device=fft_device,
            )
            energy_ratio = float(np.sum(u[-1] ** 2)) / total_energy
            u_ext = np.real(
                self.fftw.ifft(
                    self.fftw.ifftshift(
                        u_hat_full,
                        axes=0,
                        backend=fft_backend,
                        device=fft_device,
                    ),
                    axis=0,
                    backend=fft_backend,
                    device=fft_device,
                )
            ).T
            res_entropy = self._spectral_entropy_backend(
                x_entropy - np.sum(u_ext, axis=0),
                fft_backend=fft_backend,
                fft_device=fft_device,
            )
            entropy_gain = prev_entropy - res_entropy

            mode_is_weak = (
                energy_ratio < energy_threshold
                and entropy_gain < entropy_gain_threshold
            )
            if mode_is_weak:
                if k == K_min:
                    warnings.warn(
                        f"estimate_K stopped at the first candidate K={k}; "
                        f"returning K_min={K_min} because smaller decompositions "
                        "were not searched. Lower K_min if you want the estimator "
                        "to consider fewer modes.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                break

            mode_is_useful = not mode_is_weak
            if mode_is_useful:
                best_K = k
            prev_entropy = res_entropy

        return best_K

    # ──────────────────────────────────────────────────────────────────────────
    #  Core decomposition
    # ──────────────────────────────────────────────────────────────────────────

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
        precomputed_fft: dict[str, Any] | None = None,
        boundary_method: str = "reflect",
        use_soft_junction: bool = False,
        window_alpha: float | None = None,
        trial: Any | None = None,
        enforce_uncorrelated: bool = False,
        corr_rho: float = 0.1,
        corr_update_every: int = 20,
        corr_ema: float = 0.9,
        adaptive_alpha: bool = False,
        adaptive_alpha_start_iter: int = 10,
        adaptive_alpha_update_every: int = 5,
        adaptive_alpha_lr: float = 0.15,
        adaptive_alpha_min_scale: float = 0.3,
        adaptive_alpha_max_scale: float = 6.0,
        adaptive_alpha_skip_dc: bool = True,
        omega_momentum: float = 0.0,
        omega_shrinkage: float = 0.0,
        omega_max_step: float = 0.0,
        admm_over_relax: float = 1.6,
        # ── new parameters ─────────────────────────────────────────────────
        use_anderson: bool = False,
        anderson_m: int = 5,
        gram_schmidt_every: int = 0,  # 0 = disabled; N → apply every N iters
        omega_tol: float = 1e-8,  # secondary convergence gate on max|Δω|
        warm_start_state: dict[str, Any] | None = None,  # used when init=5
        random_seed: int | None = None,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        VMD decomposition with ADMM over-relaxation, optional Anderson mixing,
        optional Gram-Schmidt orthogonalisation, dual convergence criterion,
        and warm-start support.

        init codes
        ----------
        1  uniform spacing  (default safe choice)
        2  log-uniform random  (seed with ``random_seed``)
        3  spectral peak detection (rfft)
        4  Hilbert IF histogram        ← best for AM-FM / chirp signals
        5  warm start (pass warm_start_state from get_warm_start_state)

        Returns
        -------
        u          : (K, N)   real time-domain IMF modes
        u_hat_full : (T, K)   full fftshifted analytic spectrum
        omega      : (K,)     converged centre frequencies (normalised 0..0.5)

        Notes
        -----
        The convergence score compared against ``tol`` is a sum of per-mode
        relative changes, so its magnitude scales roughly with ``K``. In
        practice, larger decompositions may require a looser ``tol`` than
        single-mode runs.
        """
        x = np.asarray(signal, dtype=np.float64)

        if precomputed_fft is None:
            precomputed_fft = self._prepare_signal(
                x,
                boundary_method=boundary_method,
                use_soft_junction=use_soft_junction,
                window_alpha=window_alpha,
                fft_backend=fft_backend,
                fft_device=fft_device,
            )

        f_hat_plus = precomputed_fft["f_hat_plus"]
        freqs = precomputed_fft["freqs"]
        T = int(precomputed_fft["T"])
        half_T = int(precomputed_fft["half_T"])
        orig_len = int(precomputed_fft["orig_len"])
        left_ext = int(precomputed_fft["left_ext"])
        right_ext = int(precomputed_fft.get("right_ext", 0))
        boundary_method = precomputed_fft.get("boundary_method", boundary_method)
        fMirr = precomputed_fft.get("fMirr", None)
        fft_backend = str(precomputed_fft.get("fft_backend", fft_backend))
        fft_device = str(precomputed_fft.get("fft_device", fft_device))

        Alpha = alpha * np.ones(K, dtype=np.float64)

        # ── Initialise state ────────────────────────────────────────────────
        if init == 5 and warm_start_state is not None:
            omega = warm_start_state["omega"].copy().astype(np.float64)
            u_hat_prev = warm_start_state["u_hat"].copy().astype(np.complex128)
            lam = warm_start_state.get("lam", np.zeros(T, dtype=np.complex128)).copy()
        else:
            u_hat_prev = np.zeros((T, K), dtype=np.complex128)
            lam = np.zeros(T, dtype=np.complex128)
            base = np.asarray(fMirr if fMirr is not None else x, dtype=np.float64)

            if init == 2:
                rng = np.random.default_rng(random_seed)
                fs0 = 1.0 / T
                omega = np.sort(
                    np.exp(np.log(fs0) + (np.log(0.5) - np.log(fs0)) * rng.random(K))
                ).astype(np.float64)
            elif init == 3:
                omega = self._init_from_spectrum(
                    base, K, fft_backend=fft_backend, fft_device=fft_device
                )
            elif init == 4:
                omega = self._init_hilbert(base, K)
            else:  # init == 1 or fallback
                omega = (np.arange(K) * (0.5 / K)).astype(np.float64)

        if DC and K > 0:
            omega[0] = 0.0

        Gamma = np.zeros((K, K), dtype=np.float64)
        C_ema = None
        over_relax = float(np.clip(float(admm_over_relax), 0.0, 2.0))
        mixer = AndersonMixer(m=anderson_m) if use_anderson else None

        # ── Main iteration ──────────────────────────────────────────────────
        for n in range(int(max_iter)):
            sum_uk = np.sum(u_hat_prev, axis=1).astype(np.complex128)

            if enforce_uncorrelated and (n % corr_update_every == 0) and n > 0:
                C_est = self._corr_matrix_time_equiv_from_uhat(
                    u_hat_prev, half_T, remove_dc=True
                )
                Gamma, C_ema = self._update_gamma(
                    Gamma, C_est, rho=corr_rho, ema=corr_ema, C_ema_prev=C_ema
                )

            u_hat_next, omega_next, lam_next, diff_norm = update_modes_numba(
                freqs,
                half_T,
                f_hat_plus,
                sum_uk,
                lam,
                Alpha,
                omega,
                u_hat_prev,
                K,
                tau,
                Gamma,
                float(omega_momentum),
                float(omega_shrinkage),
                float(omega_max_step),
                over_relax,
            )

            # Anderson mixing on the u_hat fixed-point
            transformed = False
            if mixer is not None:
                u_hat_next = mixer.apply(u_hat_prev, u_hat_next)
                transformed = True

            # Optional Gram-Schmidt orthogonalisation
            if gram_schmidt_every > 0 and n > 0 and (n % gram_schmidt_every == 0):
                u_hat_next = _gram_schmidt_freq(u_hat_next, half_T)
                if mixer is not None:
                    mixer.reset()
                transformed = True

            if transformed:
                omega_next = self._reestimate_omega_from_uhat(
                    u_hat=u_hat_next,
                    freqs=freqs,
                    half_T=half_T,
                    omega_prev=omega,
                    omega_momentum=float(omega_momentum),
                    omega_shrinkage=float(omega_shrinkage),
                    omega_max_step=float(omega_max_step),
                    DC=int(DC),
                )
                diff_norm = self._relative_diff_norm(u_hat_next, u_hat_prev)

                sum_uk_new = np.sum(u_hat_next, axis=1).astype(np.complex128)
                r_new = sum_uk_new - f_hat_plus
                r_old = sum_uk - f_hat_plus
                r_relax = over_relax * r_new + (1.0 - over_relax) * r_old
                lam_next = lam + tau * r_relax

            if DC and K > 0:
                omega_next[0] = 0.0

            if (
                adaptive_alpha
                and n >= adaptive_alpha_start_iter
                and adaptive_alpha_update_every > 0
                and (n % adaptive_alpha_update_every == 0)
            ):
                Alpha = self._adapt_alpha_vector(
                    freqs=freqs,
                    half_T=half_T,
                    u_hat=u_hat_next,
                    omega=omega_next,
                    alpha_vec=Alpha,
                    base_alpha=float(alpha),
                    lr=float(adaptive_alpha_lr),
                    min_scale=float(adaptive_alpha_min_scale),
                    max_scale=float(adaptive_alpha_max_scale),
                    skip_dc=bool(adaptive_alpha_skip_dc and DC),
                )

            # ── Dual convergence criterion ──────────────────────────────────
            uDiff = float(diff_norm)
            omega_diff = float(np.max(np.abs(omega_next - omega)))

            if trial is not None and n % 15 == 0:
                trial.report(uDiff, step=n)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            u_hat_prev = u_hat_next
            lam = lam_next
            omega = omega_next

            if uDiff <= tol and omega_diff <= omega_tol:
                break

        # ── Rebuild full spectrum (Hermitian symmetry) ──────────────────────
        u_hat_full = _restore_hermitian_from_positive(u_hat_prev, half_T)

        u = np.real(
            self.fftw.ifft(
                self.fftw.ifftshift(
                    u_hat_full, axes=0, backend=fft_backend, device=fft_device
                ),
                axis=0,
                backend=fft_backend,
                device=fft_device,
            )
        ).T

        if boundary_method != "none":
            crop_stop = u.shape[1] - right_ext
            if crop_stop - left_ext != orig_len:
                raise ValueError(
                    "Signal crop bookkeeping mismatch in decompose: "
                    f"crop_stop({crop_stop}) - left_ext({left_ext}) != "
                    f"orig_len({orig_len})"
                )
            u = u[:, left_ext:crop_stop]

        if u.shape[1] != orig_len:
            xo = np.linspace(0, 1, u.shape[1])
            xn = np.linspace(0, 1, orig_len)
            u = np.vstack([np.interp(xn, xo, uk) for uk in u])

        return u, u_hat_full, omega

    # ──────────────────────────────────────────────────────────────────────────
    #  Warm-start helper
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_warm_start_state(
        u_hat_full: np.ndarray,
        omega: np.ndarray,
        T: int,
    ) -> dict[str, Any]:
        """
        Package the output of a previous ``decompose`` call for use as a warm
        start in the next call (pass as ``warm_start_state``, set ``init=5``).

        The dual variable λ is intentionally reset to zero so that the new
        signal window is not biased by the previous window's residual.
        Useful for rolling-window / streaming forecasting pipelines.
        """
        return {
            "omega": omega.copy(),
            "u_hat": u_hat_full.copy(),
            "lam": np.zeros(T, dtype=np.complex128),
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  VNCMD helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _vncmd_smooth_track(track: np.ndarray, smooth: float) -> np.ndarray:
        """Zero-phase style EMA smoothing used for IF-track stabilisation."""
        x = np.asarray(track, dtype=np.float64)
        if x.size <= 2:
            return x.copy()
        a = float(np.clip(smooth, 0.0, 0.999))
        if a <= 0.0:
            return x.copy()

        fwd = x.copy()
        for i in range(1, x.size):
            fwd[i] = a * fwd[i - 1] + (1.0 - a) * x[i]

        out = fwd.copy()
        for i in range(x.size - 2, -1, -1):
            out[i] = a * out[i + 1] + (1.0 - a) * fwd[i]
        return out

    @staticmethod
    def _vncmd_enforce_track_separation(
        tracks: np.ndarray,
        min_gap: float,
        DC: int,
    ) -> np.ndarray:
        """Keep IF tracks ordered and separated by a small positive gap."""
        y = np.asarray(tracks, dtype=np.float64).copy()
        K, N = y.shape
        if K <= 1 or float(min_gap) <= 0.0:
            if int(DC) and K > 0:
                y[0, :] = 0.0
            return np.clip(y, 0.0, 0.5)

        order = np.argsort(np.median(y, axis=1))
        y = y[order]
        gap = float(min_gap)

        for t in range(N):
            if int(DC):
                y[0, t] = 0.0
                start = 1
            else:
                y[0, t] = max(0.0, y[0, t])
                start = 1

            for k in range(start, K):
                y[k, t] = max(y[k, t], y[k - 1, t] + gap)

            excess = y[K - 1, t] - 0.5
            if excess > 0.0:
                y[start:, t] -= excess
                if not int(DC):
                    y[0, t] = max(0.0, y[0, t] - excess)
                for k in range(start, K):
                    y[k, t] = max(y[k, t], y[k - 1, t] + gap)

        if int(DC) and K > 0:
            y[0, :] = 0.0
        return np.clip(y, 0.0, 0.5)

    @staticmethod
    def _vncmd_second_diff_penalty(N: int) -> csc_matrix:
        """Second-difference smoothness penalty H^T H."""
        if N <= 2:
            return csc_matrix((N, N), dtype=np.float64)
        H = diags(
            diagonals=(np.ones(N - 2), -2.0 * np.ones(N - 2), np.ones(N - 2)),
            offsets=(0, 1, 2),
            shape=(N - 2, N),
            format="csc",
            dtype=np.float64,
        )
        return (H.T @ H).tocsc()

    def _vncmd_initial_centers(
        self,
        signal: np.ndarray,
        K: int,
        init: int,
        random_seed: int | None,
    ) -> np.ndarray:
        """Initial centre frequencies used to seed VNCMD IF tracks."""
        s = np.asarray(signal, dtype=np.float64)
        N = max(1, int(s.size))
        if K <= 0:
            return np.zeros(0, dtype=np.float64)
        if init == 2:
            rng = np.random.default_rng(random_seed)
            fs0 = 1.0 / float(N)
            return np.sort(
                np.exp(np.log(fs0) + (np.log(0.5) - np.log(fs0)) * rng.random(K))
            ).astype(np.float64)
        if init == 3:
            return self._init_from_spectrum(s, K)
        if init == 4:
            return self._init_hilbert(s, K)
        return (np.arange(K, dtype=np.float64) * (0.5 / max(K, 1))).astype(np.float64)

    def _vncmd_init_if_tracks(
        self,
        signal: np.ndarray,
        K: int,
        init: int,
        if_window_size: int,
        if_hop_size: int,
        if_center_smooth: float,
        min_track_gap: float,
        DC: int,
        random_seed: int | None,
        init_if_tracks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Initialise IF tracks from STFT ridges with smooth fallback centres."""
        s = np.asarray(signal, dtype=np.float64)
        N = int(s.size)
        if K <= 0:
            return np.zeros((0, N), dtype=np.float64)

        if init_if_tracks is not None:
            tracks = np.asarray(init_if_tracks, dtype=np.float64)
            if tracks.shape != (K, N):
                raise ValueError(
                    "init_if_tracks must have shape (K, N) matching the signal."
                )
            tracks = np.clip(tracks, 0.0, 0.5)
            return self._vncmd_enforce_track_separation(tracks, min_track_gap, DC)

        centres = self._vncmd_initial_centers(s, K, init, random_seed)
        tracks = np.repeat(centres[:, None], N, axis=1)
        if N < 8:
            return self._vncmd_enforce_track_separation(tracks, min_track_gap, DC)

        nperseg = max(16, min(int(if_window_size), N))
        hop = max(1, min(int(if_hop_size), nperseg))
        noverlap = max(0, nperseg - hop)
        freqs, times, Sxx = spectrogram(
            s,
            fs=1.0,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            mode="magnitude",
            scaling="spectrum",
        )
        if Sxx.size == 0:
            return self._vncmd_enforce_track_separation(tracks, min_track_gap, DC)

        valid = freqs <= 0.5 + 1e-12
        freqs = freqs[valid]
        Sxx = Sxx[valid]
        if freqs.size == 0 or Sxx.shape[1] == 0:
            return self._vncmd_enforce_track_separation(tracks, min_track_gap, DC)

        frame_count = int(Sxx.shape[1])
        cand_count = max(K + 2, 2 * K)
        ridge_frames = np.repeat(centres[:, None], frame_count, axis=1)
        max_mag = float(np.max(Sxx)) + 1e-12

        for j in range(frame_count):
            power = Sxx[:, j]
            if power.size == 0 or float(np.max(power)) <= 0.0:
                if j > 0:
                    ridge_frames[:, j] = ridge_frames[:, j - 1]
                continue

            order = np.argsort(power)[-min(cand_count, power.size) :][::-1]
            cand_freqs = freqs[order]
            cand_scores = power[order] / max_mag
            used = np.zeros(cand_freqs.size, dtype=bool)
            prev = centres if j == 0 else ridge_frames[:, j - 1]

            for k in np.argsort(prev):
                costs = np.abs(cand_freqs - prev[k]) - 0.05 * cand_scores
                costs[used] = np.inf
                idx = int(np.argmin(costs))
                if not np.isfinite(costs[idx]):
                    ridge_frames[k, j] = prev[k]
                    continue
                ridge_frames[k, j] = cand_freqs[idx]
                used[idx] = True

        if frame_count == 1:
            tracks = np.repeat(ridge_frames, N, axis=1)
        else:
            sample_times = np.clip(times.astype(np.float64), 0.0, float(max(0, N - 1)))
            for k in range(K):
                tracks[k] = np.interp(
                    np.arange(N, dtype=np.float64),
                    sample_times,
                    ridge_frames[k],
                    left=ridge_frames[k, 0],
                    right=ridge_frames[k, -1],
                )

        for k in range(K):
            tracks[k] = self._vncmd_smooth_track(tracks[k], if_center_smooth)

        return self._vncmd_enforce_track_separation(tracks, min_track_gap, DC)

    @staticmethod
    def _vncmd_build_phases(if_tracks: np.ndarray) -> np.ndarray:
        """Integrate IF tracks into samplewise phases."""
        phases = (
            2.0 * np.pi * np.cumsum(np.asarray(if_tracks, dtype=np.float64), axis=1)
        )
        if phases.shape[1] > 0:
            phases -= phases[:, :1]
        return phases

    @staticmethod
    def _vncmd_modes_from_quadratures(
        u_quad: np.ndarray, v_quad: np.ndarray, phases: np.ndarray
    ) -> np.ndarray:
        """Recover original-frame modes from quadrature envelopes and phases."""
        return u_quad * np.cos(phases) + v_quad * np.sin(phases)

    @staticmethod
    def _vncmd_energy_weighted_centres(
        if_tracks: np.ndarray, amplitudes: np.ndarray, DC: int
    ) -> np.ndarray:
        """Summarise time-varying IF tracks as energy-weighted centre frequencies."""
        K = int(if_tracks.shape[0])
        omega = np.zeros(K, dtype=np.float64)
        eps = 1e-12
        for k in range(K):
            w = amplitudes[k] * amplitudes[k]
            den = float(np.sum(w))
            if den > eps:
                omega[k] = float(np.sum(if_tracks[k] * w) / den)
            else:
                omega[k] = float(np.mean(if_tracks[k])) if if_tracks.shape[1] else 0.0
        if int(DC) and K > 0:
            omega[0] = 0.0
        return np.clip(omega, 0.0, 0.5)

    @staticmethod
    def _vncmd_relative_mode_diff(
        modes_next: np.ndarray, modes_prev: np.ndarray, eps: float = 1e-12
    ) -> float:
        """Mode-summed relative convergence score for VNCMD."""
        K = int(modes_next.shape[0])
        diff = 0.0
        for k in range(K):
            d = modes_next[k] - modes_prev[k]
            num = float(np.sum(d * d))
            den = float(np.sum(modes_prev[k] * modes_prev[k])) + eps
            diff += num / den
        return float(diff)

    @staticmethod
    def _vncmd_solve_quadratures(
        signal: np.ndarray,
        phases: np.ndarray,
        env_penalty: float,
        HtH: csc_matrix,
        amp_ridge_scale: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the NCMD/VNCMD quadrature-envelope subproblem:
            min ||x - A(if) z||^2 + lambda ||D z||^2
        where z stacks cosine/sine envelopes for all modes.
        """
        x = np.asarray(signal, dtype=np.float64)
        K, N = phases.shape
        if K <= 0:
            return np.zeros((0, N), dtype=np.float64), np.zeros(
                (0, N), dtype=np.float64
            )

        cos_terms = np.cos(phases)
        sin_terms = np.sin(phases)
        amp_ridge = max(0.0, float(amp_ridge_scale)) * float(max(env_penalty, 0.0))
        smooth_block = float(max(env_penalty, 0.0)) * HtH + (
            max(1e-9, amp_ridge) * eye(N, format="csc", dtype=np.float64)
        )
        block_rows: list[list[csc_matrix]] = []
        rhs_parts: list[np.ndarray] = []

        for k in range(K):
            ck = cos_terms[k]
            sk = sin_terms[k]
            rhs_parts.append(ck * x)
            rhs_parts.append(sk * x)

        for k in range(K):
            ck = cos_terms[k]
            sk = sin_terms[k]
            row_cos: list[csc_matrix] = []
            row_sin: list[csc_matrix] = []
            for j in range(K):
                cj = cos_terms[j]
                sj = sin_terms[j]
                row_cos.append(diags(ck * cj, format="csc", dtype=np.float64))
                row_cos.append(diags(ck * sj, format="csc", dtype=np.float64))
                row_sin.append(diags(sk * cj, format="csc", dtype=np.float64))
                row_sin.append(diags(sk * sj, format="csc", dtype=np.float64))
            row_cos[2 * k] = (row_cos[2 * k] + smooth_block).tocsc()
            row_sin[2 * k + 1] = (row_sin[2 * k + 1] + smooth_block).tocsc()
            block_rows.append(row_cos)
            block_rows.append(row_sin)

        system = bmat(block_rows, format="csc")
        rhs = np.concatenate(rhs_parts, axis=0)
        sol = np.asarray(spsolve(system, rhs), dtype=np.float64)

        u_quad = np.zeros((K, N), dtype=np.float64)
        v_quad = np.zeros((K, N), dtype=np.float64)
        offset = 0
        for k in range(K):
            u_quad[k] = sol[offset : offset + N]
            offset += N
            v_quad[k] = sol[offset : offset + N]
            offset += N
        return u_quad, v_quad

    def _vncmd_update_if_tracks(
        self,
        if_prev: np.ndarray,
        u_quad: np.ndarray,
        v_quad: np.ndarray,
        if_smoother: Any,
        if_center_smooth: float,
        if_step: float,
        min_track_gap: float,
        DC: int,
    ) -> np.ndarray:
        """Update IF tracks from the demodulated quadratures."""
        K, N = if_prev.shape
        if_next = np.asarray(if_prev, dtype=np.float64).copy()
        eps = 1e-12

        for k in range(K):
            if int(DC) and k == 0:
                if_next[k, :] = 0.0
                continue

            uk = u_quad[k]
            vk = v_quad[k]
            duk = np.gradient(uk)
            dvk = np.gradient(vk)
            delta_if = (uk * dvk - vk * duk) / (2.0 * np.pi * (uk * uk + vk * vk + eps))
            delta_if = np.asarray(if_smoother(delta_if), dtype=np.float64)
            # Modes are reconstructed as Re[(u - j v) exp(j phi)] = u cos(phi) + v sin(phi),
            # so delta_if is the negative envelope-phase derivative correction.
            track = np.clip(if_prev[k] - float(if_step) * delta_if, 0.0, 0.5)
            if_next[k] = self._vncmd_smooth_track(track, if_center_smooth)

        return self._vncmd_enforce_track_separation(if_next, min_track_gap, DC)

    # ──────────────────────────────────────────────────────────────────────────
    #  VNCMD / NCMD
    # ──────────────────────────────────────────────────────────────────────────

    def decompose_vncmd(
        self,
        signal: np.ndarray,
        alpha: float,
        tau: float,
        K: int,
        DC: int,
        init: int,
        tol: float,
        max_iter: int,
        max_chirp_outer: int | None = None,
        chirp_tol: float = 1e-4,
        fs: float = 1.0,
        if_window_size: int = 256,
        if_hop_size: int = 128,
        if_center_smooth: float = 0.85,
        if_update_step: float = 1.0,
        admm_over_relax: float = 1.6,
        boundary_method: str = "mirror",
        use_soft_junction: bool = False,
        window_alpha: float | None = None,
        use_anderson: bool = True,
        omega_tol: float = 1e-8,
        random_seed: int | None = None,
        trial: Any | None = None,
        init_if_tracks: np.ndarray | None = None,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full single-channel VNCMD solver for AM-FM / nonlinear-IF signals.

        Mode model
        ----------
        x[n] ≈ Σ_k { u_k[n] cos(phi_k[n]) + v_k[n] sin(phi_k[n]) }

        Algorithm
        ---------
        1. Initialise IF tracks from STFT ridges (or supplied ``init_if_tracks``).
        2. Given IF tracks, solve the sparse quadrature-envelope subproblem with
           second-difference smoothness regularisation.
        3. Update each IF track from the quadrature phase derivative and smooth it.
        4. Iterate until both the mode reconstruction change and the IF-track
           change fall below tolerance.

        This is the actual NCMD/VNCMD structure: alternating between an envelope
        least-squares solve and IF-track refinement, not an inner VMD heuristic.

        Returns
        -------
        u         : (K, N)  reconstructed original-frame modes
        u_hat     : (T, K)  full fftshifted spectra of the extended modes
        omega     : (K,)    energy-weighted mean IFs (normalised 0..0.5)
        if_tracks : (K, N)  converged instantaneous-frequency tracks

        Notes
        -----
        ``alpha`` is mapped to the envelope smoothness penalty of the VNCMD
        subproblem rather than the VMD bandwidth penalty. ``tau``,
        ``admm_over_relax``, ``use_anderson``, and ``omega_tol`` are accepted
        for API compatibility and ignored here.
        """
        envelope_ridge_scale = float(
            kwargs.pop(
                "envelope_ridge_scale",
                kwargs.pop("vncmd_envelope_ridge_scale", 0.0),
            )
        )
        min_track_gap = kwargs.pop("vncmd_min_track_gap", None)
        del tau, fs, admm_over_relax, use_anderson, omega_tol, kwargs

        x = np.asarray(signal, dtype=np.float64)
        precomp = self._prepare_signal(
            x,
            boundary_method=boundary_method,
            use_soft_junction=use_soft_junction,
            window_alpha=window_alpha,
            fft_backend=fft_backend,
            fft_device=fft_device,
        )
        x_work = np.asarray(precomp["fMirr"], dtype=np.float64)
        T = int(precomp["T"])
        orig_len = int(precomp["orig_len"])
        left_ext = int(precomp["left_ext"])
        right_ext = int(precomp.get("right_ext", 0))
        bm_used = str(precomp.get("boundary_method", boundary_method))
        fft_backend = str(precomp.get("fft_backend", fft_backend))
        fft_device = str(precomp.get("fft_device", fft_device))

        n_iter = int(max_iter if max_chirp_outer is None else max_chirp_outer)
        n_iter = max(1, n_iter)
        if min_track_gap is None:
            min_track_gap = min(0.02, 2.0 / float(max(int(if_window_size), 16)))
        min_track_gap = max(0.0, float(min_track_gap))
        if_tracks = self._vncmd_init_if_tracks(
            x_work,
            K=K,
            init=init,
            if_window_size=int(if_window_size),
            if_hop_size=int(if_hop_size),
            if_center_smooth=float(if_center_smooth),
            min_track_gap=min_track_gap,
            DC=int(DC),
            random_seed=random_seed,
            init_if_tracks=init_if_tracks,
        )
        Nw = x_work.size
        HtH = self._vncmd_second_diff_penalty(Nw)
        env_penalty = float(max(alpha, 1e-6)) / float(max(Nw, 1))
        if_reg = float(
            2.0
            * max(float(if_center_smooth), 1e-3)
            / max(1e-3, 1.0 - float(if_center_smooth))
        )
        if_smooth_matrix = (
            eye(Nw, format="csc", dtype=np.float64) + if_reg * HtH
        ).tocsc()
        if_smoother = factorized(if_smooth_matrix)

        modes_prev = np.zeros((K, Nw), dtype=np.float64)
        omega = np.zeros(K, dtype=np.float64)
        u_quad = np.zeros((K, Nw), dtype=np.float64)
        v_quad = np.zeros((K, Nw), dtype=np.float64)

        for n in range(n_iter):
            phases = self._vncmd_build_phases(if_tracks)
            u_quad, v_quad = self._vncmd_solve_quadratures(
                x_work,
                phases,
                env_penalty=env_penalty,
                HtH=HtH,
                amp_ridge_scale=envelope_ridge_scale,
            )
            modes_next = self._vncmd_modes_from_quadratures(u_quad, v_quad, phases)
            amplitudes = np.sqrt(u_quad * u_quad + v_quad * v_quad)
            if_next = self._vncmd_update_if_tracks(
                if_prev=if_tracks,
                u_quad=u_quad,
                v_quad=v_quad,
                if_smoother=if_smoother,
                if_center_smooth=float(if_center_smooth),
                if_step=float(if_update_step),
                min_track_gap=min_track_gap,
                DC=int(DC),
            )

            mode_diff = self._vncmd_relative_mode_diff(modes_next, modes_prev)
            if_diff = (
                float(np.max(np.abs(if_next - if_tracks))) if if_tracks.size else 0.0
            )

            if trial is not None and n % 10 == 0:
                trial.report(mode_diff, step=n)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            modes_prev = modes_next
            if_tracks = if_next
            omega = self._vncmd_energy_weighted_centres(
                if_tracks=if_tracks,
                amplitudes=amplitudes,
                DC=int(DC),
            )

            if mode_diff <= tol and if_diff <= chirp_tol:
                break

        u_ext = modes_prev
        u_hat_full = self.fftw.fftshift(
            self.fftw.fft(u_ext, axis=1, backend=fft_backend, device=fft_device),
            axes=1,
            backend=fft_backend,
            device=fft_device,
        ).T
        if_tracks_out = if_tracks.copy()

        if bm_used != "none":
            crop_stop = u_ext.shape[1] - right_ext
            if crop_stop - left_ext != orig_len:
                raise ValueError(
                    "Signal crop bookkeeping mismatch in decompose_vncmd: "
                    f"crop_stop({crop_stop}) - left_ext({left_ext}) != "
                    f"orig_len({orig_len})"
                )
            u = u_ext[:, left_ext:crop_stop]
            if_tracks_out = if_tracks_out[:, left_ext:crop_stop]
        else:
            u = u_ext

        if u.shape[1] != orig_len:
            xo = np.linspace(0, 1, u.shape[1])
            xn = np.linspace(0, 1, orig_len)
            u = np.vstack([np.interp(xn, xo, uk) for uk in u])
            if_tracks_out = np.vstack([np.interp(xn, xo, fk) for fk in if_tracks_out])

        return u.astype(np.float64), u_hat_full, omega, if_tracks_out.astype(np.float64)

    def decompose_chirp(
        self,
        signal: np.ndarray,
        alpha: float,
        tau: float,
        K: int,
        DC: int,
        init: int,
        tol: float,
        max_iter: int,
        max_chirp_outer: int | None = None,
        chirp_tol: float = 1e-4,
        fs: float = 1.0,
        if_window_size: int = 256,
        if_hop_size: int = 128,
        if_center_smooth: float = 0.85,
        if_update_step: float = 1.0,
        admm_over_relax: float = 1.6,
        boundary_method: str = "mirror",
        use_soft_junction: bool = False,
        window_alpha: float | None = None,
        use_anderson: bool = True,
        omega_tol: float = 1e-8,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
        random_seed: int | None = None,
        trial: Any | None = None,
        init_if_tracks: np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Backward-compatible alias for ``decompose_vncmd``."""
        return self.decompose_vncmd(
            signal=signal,
            alpha=alpha,
            tau=tau,
            K=K,
            DC=DC,
            init=init,
            tol=tol,
            max_iter=max_iter,
            max_chirp_outer=max_chirp_outer,
            chirp_tol=chirp_tol,
            fs=fs,
            if_window_size=if_window_size,
            if_hop_size=if_hop_size,
            if_center_smooth=if_center_smooth,
            if_update_step=if_update_step,
            admm_over_relax=admm_over_relax,
            boundary_method=boundary_method,
            use_soft_junction=use_soft_junction,
            window_alpha=window_alpha,
            use_anderson=use_anderson,
            omega_tol=omega_tol,
            fft_backend=fft_backend,
            fft_device=fft_device,
            random_seed=random_seed,
            trial=trial,
            init_if_tracks=init_if_tracks,
            **kwargs,
        )

    def decompose_ncmd(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Alias for ``decompose_vncmd`` for callers using NCMD terminology."""
        return self.decompose_vncmd(*args, **kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    #  True joint MVMD  (Rehman & Aftab 2019)
    # ──────────────────────────────────────────────────────────────────────────

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
        window_alpha: float | None,
        fs: float,
        admm_over_relax: float = 1.6,
        omega_momentum: float = 0.0,
        omega_shrinkage: float = 0.0,
        omega_max_step: float = 0.0,
        omega_tol: float = 1e-8,
        enforce_uncorrelated: bool = False,
        use_anderson: bool = False,
        anderson_m: int = 5,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Joint MVMD (Rehman & Aftab 2019 style).

        All channels participate in a single Gauss-Seidel sweep.
        The shared centre frequency ω_k is updated from the pooled
        cross-channel spectral centroid at each iteration, not post-hoc.
        Per-channel dual variables λ_c maintain channel-specific fidelity.

        Correctness note
        ----------------
        The baseline implementation averaged independent per-channel omegas,
        which is *not* MVMD — it is just K independent VMDs followed by an
        average.  The joint sweep in ``update_modes_mvmd_numba`` is the
        actual Rehman & Aftab formulation.
        """
        X = np.asarray(signals, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("MVMD expects (channels, samples)")
        C, _ = X.shape

        precomps = [
            self.precompute_fft(
                X[ch],
                boundary_method=boundary_method,
                use_soft_junction=use_soft_junction,
                window_alpha=window_alpha,
                fft_backend=fft_backend,
                fft_device=fft_device,
            )
            for ch in range(C)
        ]

        ref = precomps[0]
        T = int(ref["T"])
        half_T = int(ref["half_T"])
        orig_len = int(ref["orig_len"])
        left_ext = int(ref["left_ext"])
        right_ext = int(ref.get("right_ext", 0))
        bm_used = str(ref.get("boundary_method", boundary_method))
        freqs = np.asarray(ref["freqs"], dtype=np.float64)
        fft_backend = str(ref.get("fft_backend", fft_backend))
        fft_device = str(ref.get("fft_device", fft_device))

        f_hat_plus = np.zeros((C, T), dtype=np.complex128)
        for ch, pc in enumerate(precomps):
            if int(pc["T"]) != T or int(pc["orig_len"]) != orig_len:
                raise ValueError(
                    f"Channel {ch}: FFT grid mismatch.  All channels must have "
                    "identical lengths and identical boundary extension settings."
                )
            f_hat_plus[ch] = np.asarray(pc["f_hat_plus"], dtype=np.complex128)

        Alpha = alpha * np.ones(K, dtype=np.float64)

        # Shared centre-frequency initialisation
        mean_signal = np.mean(X, axis=0)
        if init == 2:
            rng = np.random.default_rng(random_seed)
            fs0 = 1.0 / T
            omega = np.sort(
                np.exp(np.log(fs0) + (np.log(0.5) - np.log(fs0)) * rng.random(K))
            ).astype(np.float64)
        elif init == 3:
            omega = self._init_from_spectrum(
                mean_signal, K, fft_backend=fft_backend, fft_device=fft_device
            )
        elif init == 4:
            omega = self._init_hilbert(mean_signal, K)
        else:
            omega = (np.arange(K) * (0.5 / K)).astype(np.float64)

        if DC and K > 0:
            omega[0] = 0.0

        lam = np.zeros((C, T), dtype=np.complex128)
        u_hat_prev = np.zeros((C, T, K), dtype=np.complex128)
        over_relax = float(np.clip(float(admm_over_relax), 0.0, 2.0))
        mixer = AndersonMixer(m=anderson_m) if use_anderson else None
        if enforce_uncorrelated:
            warnings.warn(
                "MVMD enforce_uncorrelated uses per-channel frequency-domain "
                "Gram-Schmidt projection each sweep (Gamma dual penalty is "
                "not implemented for MVMD), with shared omega re-estimated "
                "after projection.",
                RuntimeWarning,
                stacklevel=2,
            )

        for _ in range(int(max_iter)):
            sum_uk = np.sum(u_hat_prev, axis=2).astype(np.complex128)

            u_hat_next, omega_next, lam_next, diff_norm = update_modes_mvmd_numba(
                freqs,
                half_T,
                f_hat_plus,
                sum_uk,
                lam,
                Alpha,
                omega,
                u_hat_prev,
                C,
                K,
                tau,
                float(omega_momentum),
                float(omega_shrinkage),
                float(omega_max_step),
                over_relax,
            )

            transformed = False
            if mixer is not None:
                u_hat_next = mixer.apply(u_hat_prev, u_hat_next)
                transformed = True

            if enforce_uncorrelated:
                for c in range(C):
                    u_hat_next[c, :, :] = _gram_schmidt_freq(
                        u_hat_next[c, :, :], half_T
                    )
                if mixer is not None:
                    mixer.reset()
                transformed = True

            if transformed:
                omega_next = self._reestimate_mvmd_omega_from_uhat(
                    u_hat=u_hat_next,
                    freqs=freqs,
                    half_T=half_T,
                    omega_prev=omega,
                    omega_momentum=float(omega_momentum),
                    omega_shrinkage=float(omega_shrinkage),
                    omega_max_step=float(omega_max_step),
                    DC=int(DC),
                )
                diff_norm = self._mvmd_relative_diff_norm(u_hat_next, u_hat_prev)

                sum_uk_new = np.sum(u_hat_next, axis=2).astype(np.complex128)
                r_new = sum_uk_new - f_hat_plus
                r_old = sum_uk - f_hat_plus
                r_relax = over_relax * r_new + (1.0 - over_relax) * r_old
                lam_next = lam + tau * r_relax

            if DC and K > 0:
                omega_next[0] = 0.0

            omega_diff = float(np.max(np.abs(omega_next - omega)))

            u_hat_prev = u_hat_next
            lam = lam_next
            omega = omega_next

            if float(diff_norm) <= tol and omega_diff <= omega_tol:
                break

        # ── Reconstruct ─────────────────────────────────────────────────────
        u_hat_full = _restore_hermitian_from_positive_mv(u_hat_prev, half_T)

        u = np.real(
            self.fftw.ifft(
                self.fftw.ifftshift(
                    u_hat_full, axes=1, backend=fft_backend, device=fft_device
                ),
                axis=1,
                backend=fft_backend,
                device=fft_device,
            )
        )
        u = np.transpose(u, (0, 2, 1))  # (C, K, T)

        if bm_used != "none":
            crop_stop = u.shape[2] - right_ext
            if crop_stop - left_ext != orig_len:
                raise ValueError(
                    "Signal crop bookkeeping mismatch in decompose_multivariate: "
                    f"crop_stop({crop_stop}) - left_ext({left_ext}) != "
                    f"orig_len({orig_len})"
                )
            u = u[:, :, left_ext:crop_stop]

        if u.shape[2] != orig_len:
            xo = np.linspace(0, 1, u.shape[2])
            xn = np.linspace(0, 1, orig_len)
            out = np.empty((C, K, orig_len), dtype=np.float64)
            for ch in range(C):
                for k in range(K):
                    out[ch, k] = np.interp(xn, xo, u[ch, k])
            u = out

        return u.astype(np.float64), u_hat_full, omega

    # ──────────────────────────────────────────────────────────────────────────
    #  Iteration helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _relative_diff_norm(
        u_hat_next: np.ndarray, u_hat_prev: np.ndarray, eps: float = 1e-14
    ) -> float:
        """
        Relative VMD convergence norm: sum_k ||du||^2/(||u||^2+eps).

        This is intentionally a sum over modes, so the same numerical ``tol``
        becomes stricter as ``K`` increases.
        """
        K = int(u_hat_next.shape[1])
        diff_norm = 0.0
        for k in range(K):
            d = u_hat_next[:, k] - u_hat_prev[:, k]
            num = float(np.sum(d.real * d.real + d.imag * d.imag))
            den = float(
                np.sum(
                    u_hat_prev[:, k].real * u_hat_prev[:, k].real
                    + u_hat_prev[:, k].imag * u_hat_prev[:, k].imag
                )
            )
            diff_norm += num / (den + eps)
        return float(diff_norm)

    @staticmethod
    def _reestimate_omega_from_uhat(
        u_hat: np.ndarray,  # (T, K)
        freqs: np.ndarray,
        half_T: int,
        omega_prev: np.ndarray,
        omega_momentum: float,
        omega_shrinkage: float,
        omega_max_step: float,
        DC: int,
    ) -> np.ndarray:
        """Re-estimate VMD centre frequencies from accepted spectra."""
        eps = 1e-14
        _, K = u_hat.shape
        pos_freqs = freqs[half_T:]
        omega_next = np.asarray(omega_prev, dtype=np.float64).copy()

        for k in range(K):
            u_pos = u_hat[half_T:, k]
            w = u_pos.real * u_pos.real + u_pos.imag * u_pos.imag
            den = float(np.sum(w))
            om_prev = float(omega_prev[k])

            if den > eps:
                om = (1.0 - float(omega_momentum)) * (
                    float(np.sum(pos_freqs * w)) / den
                ) + float(omega_momentum) * om_prev

                if float(omega_shrinkage) > 0.0 or float(omega_max_step) > 0.0:
                    delta = (om - om_prev) * (1.0 - float(omega_shrinkage))
                    if float(omega_max_step) > 0.0:
                        delta = float(
                            np.clip(
                                delta, -float(omega_max_step), float(omega_max_step)
                            )
                        )
                    om = om_prev + delta

                omega_next[k] = float(np.clip(om, 0.0, 0.5))
            else:
                omega_next[k] = om_prev

        if int(DC) and K > 0:
            omega_next[0] = 0.0
        return omega_next

    @staticmethod
    def _mvmd_relative_diff_norm(
        u_hat_next: np.ndarray, u_hat_prev: np.ndarray, eps: float = 1e-14
    ) -> float:
        """
        Relative MVMD convergence norm: (1/C) sum_{c,k} ||du||^2/(||u||^2+eps).

        This averages over channels but still sums over modes, so the same
        numerical ``tol`` becomes stricter as ``K`` increases.
        """
        C = int(u_hat_next.shape[0])
        K = int(u_hat_next.shape[2])
        diff_norm = 0.0
        for c in range(C):
            for k in range(K):
                d = u_hat_next[c, :, k] - u_hat_prev[c, :, k]
                num = float(np.sum(d.real * d.real + d.imag * d.imag))
                den = float(
                    np.sum(
                        u_hat_prev[c, :, k].real * u_hat_prev[c, :, k].real
                        + u_hat_prev[c, :, k].imag * u_hat_prev[c, :, k].imag
                    )
                )
                diff_norm += num / (den + eps)
        if C > 0:
            diff_norm /= float(C)
        return float(diff_norm)

    @staticmethod
    def _reestimate_mvmd_omega_from_uhat(
        u_hat: np.ndarray,  # (C, T, K)
        freqs: np.ndarray,
        half_T: int,
        omega_prev: np.ndarray,
        omega_momentum: float,
        omega_shrinkage: float,
        omega_max_step: float,
        DC: int,
    ) -> np.ndarray:
        """Re-estimate shared MVMD centre frequencies from pooled projected spectra."""
        eps = 1e-14
        C, _, K = u_hat.shape
        pos_freqs = freqs[half_T:]
        omega_next = np.asarray(omega_prev, dtype=np.float64).copy()

        for k in range(K):
            pooled_num = 0.0
            pooled_den = 0.0
            for c in range(C):
                u_pos = u_hat[c, half_T:, k]
                w = u_pos.real * u_pos.real + u_pos.imag * u_pos.imag
                den_c = float(np.sum(w))
                if den_c > eps:
                    pooled_den += den_c
                    pooled_num += float(np.sum(pos_freqs * w))

            om_prev = float(omega_prev[k])
            if pooled_den > eps:
                om = (1.0 - float(omega_momentum)) * (pooled_num / pooled_den) + float(
                    omega_momentum
                ) * om_prev

                if float(omega_shrinkage) > 0.0 or float(omega_max_step) > 0.0:
                    delta = (om - om_prev) * (1.0 - float(omega_shrinkage))
                    if float(omega_max_step) > 0.0:
                        delta = float(
                            np.clip(
                                delta, -float(omega_max_step), float(omega_max_step)
                            )
                        )
                    om = om_prev + delta

                omega_next[k] = float(np.clip(om, 0.0, 0.5))
            else:
                omega_next[k] = om_prev

        if int(DC) and K > 0:
            omega_next[0] = 0.0
        return omega_next

    # ──────────────────────────────────────────────────────────────────────────
    #  Adaptive alpha
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _adapt_alpha_vector(
        freqs: np.ndarray,
        half_T: int,
        u_hat: np.ndarray,
        omega: np.ndarray,
        alpha_vec: np.ndarray,
        base_alpha: float,
        lr: float,
        min_scale: float,
        max_scale: float,
        skip_dc: bool,
    ) -> np.ndarray:
        """
        Per-mode alpha driven by spectral bandwidth.
        Broad modes → higher alpha (narrower Wiener filter).
        Narrow modes → lower alpha (wider filter, less energy loss).
        """
        eps = 1e-12
        pos_freqs = freqs[half_T:]
        K = u_hat.shape[1]
        bws = np.zeros(K, dtype=np.float64)

        for k in range(K):
            up = u_hat[half_T:, k]
            w = up.real**2 + up.imag**2
            den = float(np.sum(w))
            if den <= eps:
                bws[k] = eps
                continue
            bws[k] = float(
                np.sqrt(np.sum(w * (pos_freqs - float(omega[k])) ** 2) / (den + eps))
            )

        valid = bws[np.isfinite(bws) & (bws > eps)]
        if valid.size == 0:
            return alpha_vec

        target_bw = float(np.median(valid))
        lo = float(base_alpha) * float(min_scale)
        hi = float(base_alpha) * float(max_scale)
        if hi <= lo:
            return alpha_vec

        max_pf = float(np.max(pos_freqs)) if pos_freqs.size else 0.0
        low_freq_thr = 0.01 * max_pf
        out = np.asarray(alpha_vec, dtype=np.float64).copy()

        for k in range(K):
            if skip_dc and k == 0:
                out[k] = float(np.clip(base_alpha * 1.5, lo, hi))
                continue
            ratio = bws[k] / (target_bw + eps) - 1.0
            out[k] = float(out[k] * np.exp(float(lr) * ratio))
            if omega[k] < low_freq_thr:
                out[k] = float(np.clip(out[k] * 1.8, lo, hi))
            out[k] = float(np.clip(out[k], lo, hi))
        return out

    # ──────────────────────────────────────────────────────────────────────────
    #  Correlation helpers (enforce_uncorrelated penalty)
    # ──────────────────────────────────────────────────────────────────────────

    def _corr_matrix_time_equiv_from_uhat(
        self,
        u_hat: np.ndarray,
        half_T: int,
        eps: float = 1e-12,
        remove_dc: bool = True,
    ) -> np.ndarray:
        """Parseval-based time-domain normalised inner products (no IFFT required)."""
        U = u_hat[half_T:, :]
        if remove_dc and U.shape[0] > 1:
            U = U[1:, :]
        p = 2.0 * np.sum(U.real**2 + U.imag**2, axis=0)
        scale = np.sqrt(np.maximum(p, eps))
        U_norm = U / scale[None, :]
        C = 2.0 * (U_norm.conj().T @ U_norm).real
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 0.0)
        return np.clip(C, -1.0, 1.0)

    def _update_gamma(
        self,
        Gamma: np.ndarray,
        C_est: np.ndarray,
        rho: float,
        ema: float,
        C_ema_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """EMA-smoothed correlation estimate → dual-ascent Gamma update."""
        C_ema = C_est if C_ema_prev is None else ema * C_ema_prev + (1.0 - ema) * C_est
        Gamma = Gamma + rho * C_ema
        Gamma = 0.5 * (Gamma + Gamma.T)
        np.fill_diagonal(Gamma, 0.0)
        return Gamma, C_ema


# ─────────────────────────────────────────────────────────────────────────────
#  Numba kernels
# ─────────────────────────────────────────────────────────────────────────────


@njit(fastmath=True)
def update_modes_numba(
    freqs,
    half_T,
    f_hat_plus,
    sum_uk_init,
    lambda_hat_n,
    Alpha,
    omega_n,
    u_hat_prev,
    K,
    tau,
    Gamma,
    omega_momentum,
    omega_shrinkage,
    omega_max_step,
    admm_over_relax,
):
    """
    One Gauss-Seidel sweep over K modes with ADMM over-relaxation.

    Mode update  (Dragomiretskiy & Zosso 2014, eq. 14, extended):
        ũ_k = (f̂₊ − Σⱼ≠ₖ û_j − λ/2 − Γ-term) / (1 + αₖ(f−ωₖ)²)
        û_k ← ρ·ũ_k + (1−ρ)·û_k_prev          [over-relaxation]

    λ update (ADMM with relaxed residual):
        r_relax = ρ·(Σ û_k_new − f̂₊) + (1−ρ)·(Σ û_k_old − f̂₊)
        λ ← λ + τ·r_relax

    Convergence metric (relative, amplitude-invariant):
        diff = Σₖ  ‖û_k_new − û_k_old‖² / (‖û_k_old‖² + ε)
    The threshold on this quantity scales with K because it is a mode-sum.
    """
    T = len(freqs)
    eps = 1e-14
    u_hat_next = np.zeros((T, K), dtype=np.complex128)
    omega_next = np.zeros(K, dtype=np.float64)
    sum_uk_prev = sum_uk_init.copy()
    sum_uk = sum_uk_prev.copy()
    positive_freqs = freqs[half_T:]
    corr_term = np.zeros(T, dtype=np.complex128)

    for k in range(K):
        sum_others = sum_uk - u_hat_prev[:, k]
        omega_k = omega_n[k]
        alpha_k = Alpha[k]

        fd = freqs - omega_k
        denom = 1.0 + alpha_k * fd * fd + eps

        # Correlation penalty (Gauss-Seidel: fresh modes for j < k)
        corr_term.real[:] = 0.0
        corr_term.imag[:] = 0.0
        for j in range(K):
            if j == k:
                continue
            g = Gamma[k, j]
            if g == 0.0:
                continue
            if j < k:
                corr_term += g * u_hat_next[:, j]
            else:
                corr_term += g * u_hat_prev[:, j]

        # Wiener filter + ADMM over-relaxation
        u_raw = (f_hat_plus - sum_others - 0.5 * lambda_hat_n - corr_term) / denom
        u_new = admm_over_relax * u_raw + (1.0 - admm_over_relax) * u_hat_prev[:, k]
        u_hat_next[:, k] = u_new
        sum_uk = sum_others + u_new

        # Centre-frequency update: spectral centroid of positive half
        u_pos = u_new[half_T:]
        weights = u_pos.real * u_pos.real + u_pos.imag * u_pos.imag
        den = np.sum(weights)
        if den > eps:
            om = (1.0 - omega_momentum) * (
                np.sum(positive_freqs * weights) / den
            ) + omega_momentum * omega_k

            if omega_shrinkage > 0.0 or omega_max_step > 0.0:
                delta = (om - omega_k) * (1.0 - omega_shrinkage)
                if omega_max_step > 0.0:
                    if delta > omega_max_step:
                        delta = omega_max_step
                    elif delta < -omega_max_step:
                        delta = -omega_max_step
                om = omega_k + delta

            omega_next[k] = max(0.0, min(0.5, om))
        else:
            omega_next[k] = omega_k

    # Relative convergence criterion
    diff_norm = 0.0
    for k in range(K):
        d = u_hat_next[:, k] - u_hat_prev[:, k]
        num = np.sum(d.real * d.real + d.imag * d.imag)
        den_n = np.sum(u_hat_prev[:, k].real ** 2 + u_hat_prev[:, k].imag ** 2) + eps
        diff_norm += num / den_n

    # ADMM dual update with over-relaxed primal residual, using explicit
    # full-iterate sums for both the old and new residuals.
    sum_uk_next = np.zeros(T, dtype=np.complex128)
    for k in range(K):
        sum_uk_next += u_hat_next[:, k]

    r_new = sum_uk_next - f_hat_plus
    r_old = sum_uk_prev - f_hat_plus
    r_relax = admm_over_relax * r_new + (1.0 - admm_over_relax) * r_old
    lam_next = lambda_hat_n + tau * r_relax

    return u_hat_next, omega_next, lam_next, diff_norm


@njit(fastmath=True)
def update_modes_mvmd_numba(
    freqs,
    half_T,
    f_hat_plus,
    sum_uk_init,
    lambda_hat_n,
    Alpha,
    omega_n,
    u_hat_prev,
    C,
    K,
    tau,
    omega_momentum,
    omega_shrinkage,
    omega_max_step,
    admm_over_relax,
):
    """
    Joint MVMD Gauss-Seidel sweep (Rehman & Aftab 2019 style).

    For each mode k:
      - All C channels are updated using the same shared ω_k.
      - ω_k is updated from the pooled cross-channel spectral centroid.
      - Each channel has its own dual variable λ_c (per-channel fidelity).
      - The convergence score averages over channels but still sums over modes.
    """
    T = len(freqs)
    eps = 1e-14
    u_hat_next = np.zeros((C, T, K), dtype=np.complex128)
    omega_next = np.zeros(K, dtype=np.float64)
    lam_next = np.zeros((C, T), dtype=np.complex128)
    sum_uk_prev = sum_uk_init.copy()
    sum_uk = sum_uk_prev.copy()
    pos_freqs = freqs[half_T:]

    for k in range(K):
        omega_k = omega_n[k]
        alpha_k = Alpha[k]
        fd = freqs - omega_k
        denom = 1.0 + alpha_k * fd * fd + eps

        pooled_num = 0.0
        pooled_den = 0.0

        for c in range(C):
            sum_others = sum_uk[c, :] - u_hat_prev[c, :, k]
            u_raw = (f_hat_plus[c, :] - sum_others - 0.5 * lambda_hat_n[c, :]) / denom
            u_new = (
                admm_over_relax * u_raw + (1.0 - admm_over_relax) * u_hat_prev[c, :, k]
            )
            u_hat_next[c, :, k] = u_new
            sum_uk[c, :] = sum_others + u_new

            u_pos = u_new[half_T:]
            w = u_pos.real * u_pos.real + u_pos.imag * u_pos.imag
            d_c = np.sum(w)
            if d_c > eps:
                pooled_den += d_c
                pooled_num += np.sum(pos_freqs * w)

        # Shared centre-frequency update from pooled energy
        if pooled_den > eps:
            om = (1.0 - omega_momentum) * (
                pooled_num / pooled_den
            ) + omega_momentum * omega_k

            if omega_shrinkage > 0.0 or omega_max_step > 0.0:
                delta = (om - omega_k) * (1.0 - omega_shrinkage)
                if omega_max_step > 0.0:
                    if delta > omega_max_step:
                        delta = omega_max_step
                    elif delta < -omega_max_step:
                        delta = -omega_max_step
                om = omega_k + delta

            omega_next[k] = max(0.0, min(0.5, om))
        else:
            omega_next[k] = omega_k

    # Channel-averaged relative convergence criterion
    diff_norm = 0.0
    for c in range(C):
        for k in range(K):
            d = u_hat_next[c, :, k] - u_hat_prev[c, :, k]
            num = np.sum(d.real * d.real + d.imag * d.imag)
            den_n = (
                np.sum(u_hat_prev[c, :, k].real ** 2 + u_hat_prev[c, :, k].imag ** 2)
                + eps
            )
            diff_norm += num / den_n
    if C > 0:
        diff_norm /= C

    # Per-channel dual update with over-relaxed primal residual, using explicit
    # full-iterate sums for both the old and new residuals.
    sum_uk_next = np.zeros((C, T), dtype=np.complex128)
    for c in range(C):
        for k in range(K):
            sum_uk_next[c, :] += u_hat_next[c, :, k]

    for c in range(C):
        r_new = sum_uk_next[c, :] - f_hat_plus[c, :]
        r_old = sum_uk_prev[c, :] - f_hat_plus[c, :]
        r_relax = admm_over_relax * r_new + (1.0 - admm_over_relax) * r_old
        lam_next[c] = lambda_hat_n[c, :] + tau * r_relax

    return u_hat_next, omega_next, lam_next, diff_norm


# ─────────────────────────────────────────────────────────────────────────────
#  Neural refiners  (post-processing, independent of core VMD)
# ─────────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------------
# Optional: mode refinement (post-processing, independent of core VMD)
# -----------------------------------------------------------------------------


if TORCH_AVAILABLE:

    class InformerRefiner(nn.Module):
        """Small Transformer encoder used as a per-mode denoiser/refiner."""

        def __init__(self, seq_len: int, d_model: int = 64, n_heads: int = 4):
            super().__init__()
            self.embed = nn.Linear(1, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.proj = nn.Linear(d_model, 1)
            self.seq_len = seq_len

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)  # (B, L, 1)
            h = self.embed(x)
            h = self.encoder(h)
            y = self.proj(h)
            return y.transpose(1, 2)  # (B, 1, L)

    class CrossModeRefiner(nn.Module):
        """
        Refiner with cross-mode attention then temporal attention.
        Input/Output shape: (B, K, L).
        """

        def __init__(
            self,
            num_modes: int,
            seq_len: int,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers_cross: int = 2,
            n_layers_time: int = 1,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.num_modes = int(num_modes)
            self.seq_len = int(seq_len)
            self.embed = nn.Linear(1, d_model)

            cross_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.cross_encoder = nn.TransformerEncoder(
                cross_layer, num_layers=n_layers_cross
            )

            time_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.time_encoder = nn.TransformerEncoder(
                time_layer, num_layers=n_layers_time
            )
            self.proj = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(-1)  # (1, K, L, 1)
            elif x.dim() == 3:
                x = x.unsqueeze(-1)  # (B, K, L, 1)

            B, K, L, _ = x.shape
            h = self.embed(x)  # (B, K, L, d_model)

            # Cross-mode attention at each time step
            h = h.permute(0, 2, 1, 3).reshape(B * L, K, -1)
            h = self.cross_encoder(h)

            # Temporal attention per mode
            h = h.reshape(B, L, K, -1).permute(0, 2, 1, 3)
            h = h.reshape(B * K, L, -1)
            h = self.time_encoder(h)
            h = h.reshape(B, K, L, -1)

            y = self.proj(h)  # (B, K, L, 1)
            return y.squeeze(-1)  # (B, K, L)

else:

    class InformerRefiner:  # pragma: no cover - import-safe fallback
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PyTorch is required for InformerRefiner")

    class CrossModeRefiner:  # pragma: no cover - import-safe fallback
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PyTorch is required for CrossModeRefiner")


def _refiner_corrupt_input(x: Any, noise_std: float) -> Any:
    """Inject light Gaussian corruption for self-supervised denoising."""
    if not TORCH_AVAILABLE or float(noise_std) <= 0.0:
        return x
    return x + float(noise_std) * torch.randn_like(x)


def _refiner_smoothness_penalty(y: Any) -> Any:
    """Second-difference penalty to discourage high-frequency copy-through."""
    if not TORCH_AVAILABLE:
        return 0.0
    if y.shape[-1] < 3:
        return torch.zeros((), dtype=y.dtype, device=y.device)
    d2 = y[..., 2:] - 2.0 * y[..., 1:-1] + y[..., :-2]
    return torch.mean(d2 * d2)


def refine_modes_nn(
    modes: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    use_gpu: bool = True,
    target_modes: np.ndarray | None = None,
    noise_std: float = 0.05,
    smoothness_weight: float = 1e-3,
) -> np.ndarray:
    """Per-mode denoising refinement (InformerRefiner)."""
    if not TORCH_AVAILABLE:
        return modes

    m = np.asarray(modes, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    K, L = m.shape

    target = m
    if target_modes is not None:
        target = np.asarray(target_modes, dtype=np.float64)
        if target.ndim == 1:
            target = target[None, :]
        if target.shape != m.shape:
            raise ValueError("target_modes must match modes shape in refine_modes_nn")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    model = InformerRefiner(seq_len=L).to(device)
    opt = Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    mean = m.mean(axis=1, keepdims=True)
    std = m.std(axis=1, keepdims=True) + 1e-8
    mn = (m - mean) / std
    target_n = (target - mean) / std

    x = torch.from_numpy(mn[:, None, :]).float().to(device)  # (K, 1, L)
    target_x = torch.from_numpy(target_n[:, None, :]).float().to(device)
    model.train()
    for _ in range(int(epochs)):
        y = model(_refiner_corrupt_input(x, noise_std))
        recon_loss = crit(y, target_x)
        smooth_loss = _refiner_smoothness_penalty(y)
        loss = recon_loss + float(smoothness_weight) * smooth_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        y = model(x).cpu().numpy().squeeze(1)

    return (y * std + mean).astype(np.float64)


def refine_modes_cross_nn(
    modes: np.ndarray,
    epochs: int = 100,
    lr: float = 5e-4,
    use_gpu: bool = True,
    target_modes: np.ndarray | None = None,
    noise_std: float = 0.05,
    smoothness_weight: float = 1e-3,
) -> np.ndarray:
    """Cross-mode denoising refinement (CrossModeRefiner)."""
    if not TORCH_AVAILABLE:
        return modes

    m = np.asarray(modes, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    K, L = m.shape

    target = m
    if target_modes is not None:
        target = np.asarray(target_modes, dtype=np.float64)
        if target.ndim == 1:
            target = target[None, :]
        if target.shape != m.shape:
            raise ValueError(
                "target_modes must match modes shape in refine_modes_cross_nn"
            )

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    model = CrossModeRefiner(num_modes=K, seq_len=L).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()

    mean = m.mean(axis=1, keepdims=True)
    std = m.std(axis=1, keepdims=True) + 1e-8
    mn = (m - mean) / std
    target_n = (target - mean) / std

    x = torch.from_numpy(mn).float().to(device).unsqueeze(0)  # (1, K, L)
    target_x = torch.from_numpy(target_n).float().to(device).unsqueeze(0)
    model.train()
    for _ in range(int(epochs)):
        y = model(_refiner_corrupt_input(x, noise_std))
        recon_loss = crit(y, target_x)
        smooth_loss = _refiner_smoothness_penalty(y)
        loss = recon_loss + float(smoothness_weight) * smooth_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        y = model(x).squeeze(0).cpu().numpy()

    return (y * std + mean).astype(np.float64)
