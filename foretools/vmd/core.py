# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pyfftw.interfaces.numpy_fft as fftw_np
from numba import njit

try:
    from .common import BoundaryHandler, FFTWManager
except Exception:
    from vmd_common import BoundaryHandler, FFTWManager

# Optional torch refinement
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
        - trim odd length
        - extend (mirror/reflect/...)
        - optional smooth junction
        - optional tukey window  [WARNING: conflicts with mirroring; use one or the other]
        - compute f_hat_plus, freqs
        Returns everything needed for decompose.
        """
        f = np.asarray(signal, dtype=np.float64)
        orig_len = int(f.size)

        if orig_len % 2:
            f = f[:-1]
            orig_len -= 1

        if boundary_method == "mirror":
            ratio = 0.5
        else:
            ratio = self.boundary.adaptive_extension_ratio(f)
        fMirr, left_ext, right_ext = self.boundary.extend_signal(
            f, method=boundary_method, extension_ratio=ratio
        )

        if use_soft_junction and boundary_method != "none" and left_ext > 0:
            fMirr = self.boundary.smooth_edge_junction(
                fMirr, original_len=orig_len, ext_len=left_ext
            )

        if window_alpha is not None and window_alpha > 0:
            from scipy.signal import windows

            win = windows.tukey(len(fMirr), alpha=float(window_alpha))
            fMirr = fMirr * win

        T = int(len(fMirr))
        freqs = np.fft.fftshift(np.fft.fftfreq(T))
        f_hat = fftw_np.fftshift(fftw_np.fft(fMirr))
        f_hat_plus = f_hat.copy()
        f_hat_plus[: T // 2] = 0

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
        boundary_method: str = "mirror",
        use_soft_junction: bool = False,
        window_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._prepare_signal(
            signal, boundary_method, use_soft_junction, window_alpha
        )

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
        precomputed_fft: Optional[Dict[str, Any]] = None,
        boundary_method: str = "reflect",
        use_soft_junction: bool = False,
        window_alpha: Optional[float] = None,
        trial: Optional[Any] = None,
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core VMD decomposition (real signal), following Dragomiretskiy & Zosso (2014).

        Returns raw time-domain modes — no frequency-shift demodulation applied.

        Returns:
            u        : (K, N)   — raw time-domain IMF modes
            u_hat_full: (T, K)  — full (fftshifted) analytic spectrum
            omega    : (K,)     — converged centre frequencies (normalised, 0..0.5)
        """
        x = np.asarray(signal, dtype=np.float64)

        if precomputed_fft is None:
            precomputed_fft = self._prepare_signal(
                x,
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

        # Per-mode alpha vector (starts uniform, may be adapted)
        Alpha = (alpha * np.ones(K)).astype(np.float64)

        # Initialise centre frequencies
        omega = np.zeros(K, dtype=np.float64)
        if init == 1:
            omega = (np.arange(K) * (0.5 / K)).astype(np.float64)
        elif init == 2:
            fs0 = 1.0 / T
            omega = np.sort(
                np.exp(np.log(fs0) + (np.log(0.5) - np.log(fs0)) * np.random.rand(K))
            ).astype(np.float64)
        elif init == 3:
            base = np.asarray(fMirr if (fMirr is not None) else x, dtype=np.float64)
            omega = self._init_from_spectrum(base, K)

        if DC:
            omega[0] = 0.0

        # Dual variable and mode spectrum
        lam = np.zeros(len(freqs), dtype=np.complex128)
        u_hat_prev = np.zeros((len(freqs), K), dtype=np.complex128)

        # Correlation-penalty matrix (used only when enforce_uncorrelated=True)
        Gamma = np.zeros((K, K), dtype=np.float64)
        C_ema = None

        uDiff = float(tol) + 1.0

        for n in range(int(max_iter)):
            sum_uk = np.sum(u_hat_prev, axis=1).astype(np.complex128)

            # --- Optional: enforce decorrelated modes via dual-ascent penalty ---
            if enforce_uncorrelated and (n % corr_update_every == 0) and n > 0:
                C_est = self._corr_matrix_time_equiv_from_uhat(
                    u_hat_prev, half_T, remove_dc=True
                )
                Gamma, C_ema = self._update_gamma(
                    Gamma, C_est, rho=corr_rho, ema=corr_ema, C_ema_prev=C_ema
                )

            # --- Main Gauss-Seidel update (Numba) ---
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
            )

            if DC and K > 0:
                omega_next[0] = 0.0

            # --- Optional: adaptive per-mode alpha ---
            if (
                adaptive_alpha
                and n >= int(adaptive_alpha_start_iter)
                and int(adaptive_alpha_update_every) > 0
                and (n % int(adaptive_alpha_update_every) == 0)
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

            # --- Convergence: relative normalised criterion (canonical VMD) ---
            # diff_norm from the Numba kernel is already: sum_k ||Δu_k||² / (||u_k_prev||² * T)
            uDiff = float(diff_norm)

            # Optuna pruning hook
            if trial is not None and n % 15 == 0:
                trial.report(uDiff, step=n)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            u_hat_prev = u_hat_next
            lam = lam_next
            omega = omega_next

            if uDiff <= tol:
                break

        # --- Rebuild full spectrum with Hermitian symmetry (vmdpy-compatible) ---
        u_hat_full = np.zeros((T, K), dtype=np.complex128)
        u_hat_full[half_T:, :] = u_hat_prev[half_T:, :]
        if half_T > 1:
            u_hat_full[half_T - 1 : 0 : -1, :] = np.conj(u_hat_prev[half_T + 1 : T, :])
        u_hat_full[0, :] = np.conj(u_hat_full[-1, :])

        # IFFT → real time-domain modes, shape (K, T)
        u = np.real(
            fftw_np.ifft(fftw_np.ifftshift(u_hat_full, axes=0), axis=0)
        ).T  # (K, T)

        # --- Crop back to original signal length ---
        if boundary_method != "none":
            start = left_ext
            end = start + orig_len
            u = u[:, start:end]

        # Safety: interpolate to exact length if any rounding mismatch
        if u.shape[1] != orig_len:
            x_old = np.linspace(0, 1, u.shape[1])
            x_new = np.linspace(0, 1, orig_len)
            u = np.vstack([np.interp(x_new, x_old, uk) for uk in u])

        # NOTE: No FS-VMD remodulation. Modes are returned as-is (raw IMFs),
        # identical in spirit to the original Dragomiretskiy & Zosso (2014) output.

        return u, u_hat_full, omega

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
        Adaptive per-mode alpha update based on mode spectral bandwidth.
        Broad modes get higher alpha (narrower band), narrow modes get lower alpha.
        """
        eps = 1e-12
        pos_freqs = freqs[half_T:]
        K = u_hat.shape[1]
        bws = np.zeros(K, dtype=np.float64)

        for k in range(K):
            up = u_hat[half_T:, k]
            w = up.real * up.real + up.imag * up.imag
            den = float(np.sum(w))
            if den <= eps:
                bws[k] = eps
                continue
            diff = pos_freqs - float(omega[k])
            bws[k] = float(np.sqrt(np.sum(w * diff * diff) / (den + eps)))

        valid = bws[np.isfinite(bws) & (bws > eps)]
        if valid.size == 0:
            return alpha_vec
        target_bw = float(np.median(valid))
        if target_bw <= eps:
            return alpha_vec

        lo = float(base_alpha) * float(min_scale)
        hi = float(base_alpha) * float(max_scale)
        if hi <= lo:
            return alpha_vec

        max_pos_freq = float(np.max(pos_freqs)) if pos_freqs.size else 0.0
        low_freq_thr = 0.01 * max_pos_freq

        out = np.asarray(alpha_vec, dtype=np.float64).copy()
        for k in range(K):
            if skip_dc and k == 0:
                out[k] = float(np.clip(base_alpha * 1.5, lo, hi))
                continue
            ratio = float(bws[k] / (target_bw + eps)) - 1.0
            out[k] = float(out[k] * np.exp(float(lr) * ratio))
            if omega[k] < low_freq_thr:
                out[k] = float(np.clip(out[k] * 1.8, lo, hi))
            out[k] = float(np.clip(out[k], lo, hi))
        return out

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
        """Minimal MVMD: decompose each channel independently and average omegas."""
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
                precomputed_fft=precomps[ch],
                boundary_method=boundary_method,
                use_soft_junction=use_soft_junction,
                window_alpha=window_alpha,
            )
            modes[ch] = u
            omega_shared = (omega_shared * ch + om) / float(ch + 1)

        return modes, None, omega_shared

    # ------------------------------------------------------------------
    # Correlation helpers (enforce_uncorrelated penalty)
    # ------------------------------------------------------------------

    def _corr_matrix_ensemble(
        self, u_hat: np.ndarray, half_T: int, n_blocks: int = 4, eps: float = 1e-12
    ) -> np.ndarray:
        U = u_hat[half_T:, :]
        Tpos = U.shape[0]
        if Tpos < 64 or n_blocks <= 1:
            return self._corr_matrix_from_uhat(u_hat, half_T, eps)

        blocks = np.array_split(np.arange(Tpos), n_blocks)
        Cs = []
        for idx in blocks:
            Ub = U[idx, :]
            G = (Ub.conj().T @ Ub).real
            p = np.diag(G).copy()
            denom = np.sqrt(np.outer(p, p)) + eps
            C = G / denom
            np.fill_diagonal(C, 0.0)
            Cs.append(C)
        return np.clip(np.mean(Cs, axis=0), -1.0, 1.0)

    def _corr_matrix_time_equiv_from_uhat(
        self,
        u_hat: np.ndarray,
        half_T: int,
        eps: float = 1e-12,
        remove_dc: bool = True,
    ) -> np.ndarray:
        """
        Returns time-domain normalised inner products via Parseval (no IFFT).
        remove_dc=True approximates mean removal by dropping the DC bin.
        """
        U = u_hat[half_T:, :]
        if remove_dc and U.shape[0] > 1:
            U = U[1:, :]

        p = 2.0 * np.sum(U.real * U.real + U.imag * U.imag, axis=0)
        scale = np.sqrt(np.maximum(p, eps))
        U_norm = U / scale[None, :]

        C = 2.0 * (U_norm.conj().T @ U_norm).real
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 0.0)
        return np.clip(C, -1.0, 1.0)

    def _corr_matrix_from_uhat(
        self, u_hat: np.ndarray, half_T: int, eps: float = 1e-12
    ) -> np.ndarray:
        U = u_hat[half_T:, :]
        G = (U.conj().T @ U).real
        p = np.diag(G).copy()
        denom = np.sqrt(np.outer(p, p)) + eps
        C = G / denom
        np.fill_diagonal(C, 0.0)
        return np.clip(C, -1.0, 1.0)

    def _update_gamma(
        self,
        Gamma: np.ndarray,
        C_est: np.ndarray,
        rho: float,
        ema: float,
        C_ema_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """EMA-smoothed correlation estimate → dual-ascent Gamma update."""
        C_ema = (
            C_est if C_ema_prev is None else (ema * C_ema_prev + (1.0 - ema) * C_est)
        )
        Gamma = Gamma + rho * C_ema
        Gamma = 0.5 * (Gamma + Gamma.T)
        np.fill_diagonal(Gamma, 0.0)
        return Gamma, C_ema


# -----------------------------------------------------------------------------
# Numba update kernel — canonical VMD Gauss-Seidel step
# -----------------------------------------------------------------------------


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
):
    """
    One full Gauss-Seidel sweep over all K modes.

    Convergence metric:
        diff_norm = sum_k  ||u_k_new - u_k_old||² / (||u_k_old||² + eps)  / T

    This is the *relative* normalised criterion from Dragomiretskiy & Zosso (2014),
    eq. (26).  It is amplitude-invariant, so `tol=1e-7` works for any signal scale.
    """
    T = len(freqs)
    eps = 1e-14

    u_hat_plus_next = np.zeros((T, K), dtype=np.complex128)
    omega_next = np.zeros(K, dtype=np.float64)

    sum_uk = sum_uk_init.copy()
    positive_freqs = freqs[half_T:]

    # Per-mode correlation penalty accumulator (reused each k)
    corr_term = np.zeros(T, dtype=np.complex128)

    for k in range(K):
        # Remove current mode from running sum → sum of all OTHER modes
        sum_others = sum_uk - u_hat_prev[:, k]

        omega_k = omega_n[k]
        alpha_k = Alpha[k]

        # Wiener-filter denominator: 1 + alpha * (f - omega_k)²
        freq_diff = freqs - omega_k
        denom = 1.0 + alpha_k * freq_diff * freq_diff + eps

        # Correlation penalty: Gauss-Seidel mixing
        #   j < k → use freshly updated u_hat_plus_next[:, j]
        #   j > k → use u_hat_prev[:, j]
        corr_term.real[:] = 0.0
        corr_term.imag[:] = 0.0
        for j in range(K):
            if j == k:
                continue
            g = Gamma[k, j]
            if g == 0.0:
                continue
            if j < k:
                corr_term += g * u_hat_plus_next[:, j]
            else:
                corr_term += g * u_hat_prev[:, j]

        # Mode update (eq. 14 of Dragomiretskiy & Zosso 2014, extended)
        u_new = (f_hat_plus - sum_others - 0.5 * lambda_hat_n - corr_term) / denom
        u_hat_plus_next[:, k] = u_new

        # Update running sum with new mode
        sum_uk = sum_others + u_new

        # --- Centre-frequency update (spectral centroid of positive half) ---
        u_pos = u_new[half_T:]
        weights = u_pos.real * u_pos.real + u_pos.imag * u_pos.imag
        den = np.sum(weights)
        if den > eps:
            num = np.sum(positive_freqs * weights)
            omega_new = num / den

            # Damped update: blend new estimate with previous centre freq
            om = (1.0 - omega_momentum) * omega_new + omega_momentum * omega_k

            # Optional shrinkage / max-step clamping
            if omega_shrinkage > 0.0 or omega_max_step > 0.0:
                delta = (om - omega_k) * (1.0 - omega_shrinkage)
                if omega_max_step > 0.0:
                    if delta > omega_max_step:
                        delta = omega_max_step
                    elif delta < -omega_max_step:
                        delta = -omega_max_step
                om = omega_k + delta

            # Clamp to [0, 0.5] (normalised frequency range for real signals)
            if om < 0.0:
                om = 0.0
            elif om > 0.5:
                om = 0.5
            omega_next[k] = om
        else:
            omega_next[k] = omega_k

    # --- Relative normalised convergence criterion (amplitude-invariant) ---
    # diff_norm = (1/T) * sum_k  ||u_new_k - u_old_k||² / (||u_old_k||² + eps)
    diff_norm = 0.0
    for k in range(K):
        diff = u_hat_plus_next[:, k] - u_hat_prev[:, k]
        num = np.sum(diff.real * diff.real + diff.imag * diff.imag)
        den_norm = (
            np.sum(
                u_hat_prev[:, k].real * u_hat_prev[:, k].real
                + u_hat_prev[:, k].imag * u_hat_prev[:, k].imag
            )
            + eps
        )
        diff_norm += num / (den_norm * T)

    # Dual-variable (Lagrange multiplier) update
    lambda_next = lambda_hat_n + tau * (sum_uk - f_hat_plus)

    return u_hat_plus_next, omega_next, lambda_next, diff_norm


# -----------------------------------------------------------------------------
# Optional: mode refinement (post-processing, independent of core VMD)
# -----------------------------------------------------------------------------


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
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=n_layers_time)
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


def refine_modes_nn(
    modes: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    use_gpu: bool = True,
) -> np.ndarray:
    """Per-mode autoencoder refinement (InformerRefiner)."""
    if not TORCH_AVAILABLE:
        return modes

    m = np.asarray(modes, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    K, L = m.shape

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    model = InformerRefiner(seq_len=L).to(device)
    opt = Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    mean = m.mean(axis=1, keepdims=True)
    std = m.std(axis=1, keepdims=True) + 1e-8
    mn = (m - mean) / std

    x = torch.from_numpy(mn[:, None, :]).float().to(device)  # (K, 1, L)
    model.train()
    for epoch in range(int(epochs)):
        y = model(x)
        loss = crit(y, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Epoch", epoch, "Loss", loss.item())

    model.eval()
    with torch.no_grad():
        y = model(x).cpu().numpy().squeeze(1)

    return (y * std + mean).astype(np.float64)


def refine_modes_cross_nn(
    modes: np.ndarray,
    epochs: int = 100,
    lr: float = 5e-4,
    use_gpu: bool = True,
) -> np.ndarray:
    """Cross-mode attention refinement (CrossModeRefiner)."""
    if not TORCH_AVAILABLE:
        return modes

    m = np.asarray(modes, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    K, L = m.shape

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    model = CrossModeRefiner(num_modes=K, seq_len=L).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()

    mean = m.mean(axis=1, keepdims=True)
    std = m.std(axis=1, keepdims=True) + 1e-8
    mn = (m - mean) / std

    x = torch.from_numpy(mn).float().to(device).unsqueeze(0)  # (1, K, L)
    model.train()
    for _ in range(int(epochs)):
        y = model(x)
        loss = crit(y, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        y = model(x).squeeze(0).cpu().numpy()

    return (y * std + mean).astype(np.float64)
