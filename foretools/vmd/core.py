# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
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
        use_fs_vmd: bool = False,
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
        omega_momentum: float = 0.3,
        omega_shrinkage: float = 0.0,
        omega_max_step: float = 0.0,
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

        Gamma = np.zeros((K, K), dtype=np.float64)
        C_ema = None

        if init == 1:
            omega = (np.arange(K) * (0.5 / K)).astype(np.float64)
        elif init == 2:
            fs0 = 1.0 / T
            omega = np.sort(
                np.exp(np.log(fs0) + (np.log(0.5) - np.log(fs0)) * np.random.rand(K))
            ).astype(np.float64)
        elif init == 3:
            base = np.asarray(
                fMirr if (fMirr is not None) else x_work, dtype=np.float64
            )
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

            if enforce_uncorrelated and (n % corr_update_every == 0) and n > 0:
                # Use ensemble estimate if you want:
                # C_est = self._corr_matrix_ensemble(u_hat_prev, half_T, n_blocks=4)
                # Or single-shot:
                # C_est = self._corr_matrix_from_uhat(u_hat_prev, half_T)
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
            )

            if DC and K > 0:
                omega_next[0] = 0.0

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

        u = np.real(
            fftw_np.ifft(fftw_np.ifftshift(u_hat_full, axes=0), axis=0)
        ).T  # (K, T)

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
            # multiplicative step for positivity/stability
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
        C_mean = np.mean(Cs, axis=0)
        return np.clip(C_mean, -1.0, 1.0)

    def _corr_matrix_time_equiv_from_uhat(
        self,
        u_hat: np.ndarray,  # (T, K), fftshifted, "plus" spectrum (DC+pos half meaningful)
        half_T: int,
        eps: float = 1e-12,
        remove_dc: bool = True,
    ) -> np.ndarray:
        """
        Returns a correlation-like matrix that corresponds to *time-domain* normalized
        inner products of real modes u_k(t), using Parseval (no IFFT needed).

        remove_dc=True approximates mean removal by dropping the DC bin.
        """
        U = u_hat[half_T:, :]  # includes DC at row 0 of this slice

        if remove_dc and U.shape[0] > 1:
            U = U[1:, :]  # drop DC component (mean)

        # Inner products for real signals using one-sided spectrum:
        # <u_k, u_j> ∝ Re( U[:,k]^H U[:,j] ) with a factor 2 for positive freqs.
        G = 2.0 * (U.conj().T @ U).real  # (K,K)

        p = np.diag(G).copy()
        denom = np.sqrt(np.outer(p, p)) + eps
        C = G / denom

        np.fill_diagonal(C, 0.0)
        return np.clip(C, -1.0, 1.0)

    def _corr_matrix_from_uhat(
        self, u_hat: np.ndarray, half_T: int, eps: float = 1e-12
    ) -> np.ndarray:
        """
        u_hat: (T, K) complex (your u_hat_prev)
        Uses positive freqs only (half_T:).
        Returns: (K, K) real correlation-like coefficients in [-1,1].
        """
        U = u_hat[half_T:, :]  # (Tpos, K)
        # Gram matrix: G[k,j] = <u_k, u_j>
        G = (U.conj().T @ U).real  # (K,K) real part
        p = np.diag(G).copy()
        denom = np.sqrt(np.outer(p, p)) + eps
        C = G / denom
        np.fill_diagonal(C, 0.0)
        # Clip for stability
        return np.clip(C, -1.0, 1.0)

    def _update_gamma(
        self,
        Gamma: np.ndarray,
        C_est: np.ndarray,
        rho: float,
        ema: float,
        C_ema_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        EMA smooth C, then dual ascent on Gamma.
        Keeps Gamma symmetric and zero diagonal.
        """
        if C_ema_prev is None:
            C_ema = C_est
        else:
            C_ema = ema * C_ema_prev + (1.0 - ema) * C_est

        Gamma = Gamma + rho * C_ema
        # enforce symmetry & zero diag (important)
        Gamma = 0.5 * (Gamma + Gamma.T)
        np.fill_diagonal(Gamma, 0.0)
        return Gamma, C_ema


# -----------------------------------------------------------------------------
# Numba update kernel
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
    T = len(freqs)

    u_hat_plus_next = np.zeros((T, K), dtype=np.complex128)
    omega_next = np.zeros(K, dtype=np.float64)

    sum_uk = sum_uk_init.copy()

    eps = 1e-14
    freq_slice_start = half_T
    positive_freqs = freqs[freq_slice_start:]

    # PRE-ALLOCATED reusable buffers
    corr_term = np.zeros(T, dtype=np.complex128)

    for k in range(K):
        sum_others = sum_uk - u_hat_prev[:, k]

        omega_k = omega_n[k]
        alpha_k = Alpha[k]
        freq_diff = freqs - omega_k
        denom = 1.0 + alpha_k * freq_diff * freq_diff + eps

        # corr_term = sum_{j!=k} Gamma[k,j] * u_j
        # Use Gauss–Seidel mixing:
        #   - for j < k, use freshly updated u_hat_plus_next[:, j]
        #   - for j > k, use u_hat_prev[:, j]
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

        # IMPORTANT: keep the correlation term (do NOT overwrite this line)
        u_new = (f_hat_plus - sum_others - 0.5 * lambda_hat_n - corr_term) / denom
        u_hat_plus_next[:, k] = u_new

        sum_uk = sum_others + u_new

        u_pos = u_new[freq_slice_start:]
        weights = u_pos.real * u_pos.real + u_pos.imag * u_pos.imag
        den = np.sum(weights)
        if den > eps:
            num = np.sum(positive_freqs * weights)
            omega_new = num / den
            # Simple damped update:
            # omega_next = (1-momentum)*omega_new + momentum*omega_prev
            om = (1.0 - omega_momentum) * omega_new + omega_momentum * omega_k
            # Optional additional shrinkage/clamp (disabled by default).
            if omega_shrinkage > 0.0 or omega_max_step > 0.0:
                delta = (om - omega_k) * (1.0 - omega_shrinkage)
                if omega_max_step > 0.0:
                    if delta > omega_max_step:
                        delta = omega_max_step
                    elif delta < -omega_max_step:
                        delta = -omega_max_step
                om = omega_k + delta
            if om < 0.0:
                om = 0.0
            elif om > 0.5:
                om = 0.5
            omega_next[k] = om
        else:
            omega_next[k] = omega_k

    diff_norm = 0.0
    for k in range(K):
        diff = u_hat_plus_next[:, k] - u_hat_prev[:, k]
        diff_norm += np.sum((diff.real * diff.real + diff.imag * diff.imag)) / T

    lambda_next = lambda_hat_n + tau * (sum_uk - f_hat_plus)

    return u_hat_plus_next, omega_next, lambda_next, diff_norm


# -----------------------------------------------------------------------------
# Optional: mode refinement
# -----------------------------------------------------------------------------
class InformerRefiner(nn.Module):
    """Small Transformer encoder used as a denoiser/refiner."""

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
    Refiner with cross-mode attention, then temporal attention.
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
        self.cross_encoder = nn.TransformerEncoder(cross_layer, num_layers=n_layers_cross)

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

        # Cross-mode attention at each time step.
        h = h.permute(0, 2, 1, 3).reshape(B * L, K, -1)  # (B*L, K, d_model)
        h = self.cross_encoder(h)

        # Temporal attention per mode.
        h = h.reshape(B, L, K, -1).permute(0, 2, 1, 3)  # (B, K, L, d_model)
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

    x = torch.from_numpy(mn[:, None, :]).float().to(device)  # (K,1,L)
    model.train()
    for epoch in range(int(epochs)):
        y = model(x)
        loss = crit(y, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Epoch", epoch, "Loss", loss)

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


# -----------------------------------------------------------------------------
# Optimizer (REFactored)
# -----------------------------------------------------------------------------
