"""
MultiWavelet Transform (MWT) based neural operator layer
Inspired by: "Multiwavelet-based Operator Learning for Differential Equations" (NeurIPS 2021)
Cleaned-up, fixed and modernized version (2025/2026 style)
"""

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import eval_legendre
from sympy import Poly, Symbol, legendre


def legendre_der(k: int, x: np.ndarray) -> np.ndarray:
    """Derivative of Legendre polynomial (for quadrature weights)."""
    return (2 * k + 1) * eval_legendre(k, x)


def phi_fn(
    coeff: np.ndarray, x: np.ndarray, lb: float = 0.0, ub: float = 1.0
) -> np.ndarray:
    """Evaluate polynomial with domain mask."""
    poly = np.polynomial.polynomial.Polynomial(coeff)
    mask = ((x < lb) | (x > ub)).astype(np.float64)
    return poly(x) * (1.0 - mask)


def get_phi_psi(k: int, base: str = "legendre") -> Tuple[List, List, List]:
    """
    Compute scaling functions phi and wavelets psi1, psi2.
    Returns list of callable polynomial functions (np.poly1d or partial).
    """
    if base not in ["legendre", "chebyshev"]:
        raise ValueError(f"Unsupported base: {base}")

    x_sym = Symbol("x")
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))

    if base == "legendre":
        for ki in range(k):
            # phi(x) = sqrt(2k+1) * P_k(2x-1)
            c = Poly(legendre(ki, 2 * x_sym - 1), x_sym).all_coeffs()
            phi_coeff[ki, : ki + 1] = np.flip(
                np.sqrt(2 * ki + 1) * np.array(c, dtype=np.float64)
            )

            # phi(2x)
            c2 = Poly(legendre(ki, 4 * x_sym - 1), x_sym).all_coeffs()
            phi_2x_coeff[ki, : ki + 1] = np.flip(
                np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(c2, dtype=np.float64)
            )

        psi1_coeff = phi_2x_coeff.copy()
        psi2_coeff = phi_2x_coeff.copy()

        for ki in range(k):
            # Orthogonalize against all previous phi
            for i in range(k):
                a = phi_2x_coeff[ki, : ki + 1]
                b = phi_coeff[i, : i + 1]
                prod = np.convolve(a, b)
                prod[np.abs(prod) < 1e-10] = 0
                proj = np.sum(
                    prod
                    / (np.arange(len(prod)) + 1)
                    * (0.5) ** (1 + np.arange(len(prod)))
                )
                psi1_coeff[ki] -= proj * phi_coeff[i]
                psi2_coeff[ki] -= proj * phi_coeff[i]

            # Orthogonalize against previous psi1
            for j in range(ki):
                a = phi_2x_coeff[ki, : ki + 1]
                b = psi1_coeff[j, :]
                prod = np.convolve(a, b)
                prod[np.abs(prod) < 1e-10] = 0
                proj = np.sum(
                    prod
                    / (np.arange(len(prod)) + 1)
                    * (0.5) ** (1 + np.arange(len(prod)))
                )
                psi1_coeff[ki] -= proj * psi1_coeff[j]
                psi2_coeff[ki] -= proj * psi2_coeff[j]

            # Normalize
            prod1 = np.convolve(psi1_coeff[ki], psi1_coeff[ki])
            norm1 = np.sum(
                prod1
                / (np.arange(len(prod1)) + 1)
                * (0.5) ** (1 + np.arange(len(prod1)))
            )
            prod2 = np.convolve(psi2_coeff[ki], psi2_coeff[ki])
            norm2 = np.sum(
                prod2
                / (np.arange(len(prod2)) + 1)
                * (1 - (0.5) ** (1 + np.arange(len(prod2))))
            )
            norm = math.sqrt(max(norm1 + norm2, 1e-16))
            psi1_coeff[ki] /= norm
            psi2_coeff[ki] /= norm

        phi = [np.poly1d(np.flip(phi_coeff[i])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i])) for i in range(k)]

    else:  # chebyshev (simplified – full version omitted for brevity; extend if needed)
        raise NotImplementedError(
            "Chebyshev support – implement similarly to legendre path"
        )

    return phi, psi1, psi2


def get_filter(base: str, k: int) -> Tuple[np.ndarray, ...]:
    """Compute filter matrices H0,H1,G0,G1 and reconstruction PHI0,PHI1."""
    phi, psi1, psi2 = get_phi_psi(k, base)

    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.eye(k)
    PHI1 = np.eye(k)

    x_sym = Symbol("x")

    if base == "legendre":
        roots = Poly(legendre(k, 2 * x_sym - 1)).all_roots()
        x_m = np.array([float(rt.evalf(18)) for rt in roots], dtype=np.float64)
        wm = 1.0 / (
            k * legendre_der(k, 2 * x_m - 1) * eval_legendre(k - 1, 2 * x_m - 1)
        )

        def psi_eval(i, t):
            return psi1[i](t) if t <= 0.5 else psi2[i](t)

        for ki in range(k):
            for kp in range(k):
                H0[ki, kp] = (1 / math.sqrt(2)) * np.sum(
                    wm * phi[ki](x_m / 2) * phi[kp](x_m)
                )
                G0[ki, kp] = (1 / math.sqrt(2)) * np.sum(
                    wm * psi_eval(ki, x_m / 2) * phi[kp](x_m)
                )
                H1[ki, kp] = (1 / math.sqrt(2)) * np.sum(
                    wm * phi[ki]((x_m + 1) / 2) * phi[kp](x_m)
                )
                G1[ki, kp] = (1 / math.sqrt(2)) * np.sum(
                    wm * psi_eval(ki, (x_m + 1) / 2) * phi[kp](x_m)
                )

    # Zero small values
    for mat in [H0, H1, G0, G1, PHI0, PHI1]:
        mat[np.abs(mat) < 1e-10] = 0.0

    return H0, H1, G0, G1, PHI0, PHI1


class SparseKernelFT1d(nn.Module):
    """Low-frequency Fourier kernel in coefficient space."""

    def __init__(self, k: int, modes: int, c: int = 16):
        super().__init__()
        self.modes = modes
        self.c_k = c * k
        scale = 1.0 / (self.c_k**0.5 + 1e-6)

        self.weight_real = nn.Parameter(scale * torch.randn(self.c_k, self.c_k, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(self.c_k, self.c_k, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, c*k]
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)  # [B, c*k, N]
        x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")

        l = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            B, self.c_k, x_ft.shape[-1], dtype=torch.cfloat, device=x.device
        )

        w = torch.complex(self.weight_real[:, :, :l], self.weight_imag[:, :, :l])
        out_ft[:, :, :l] = torch.einsum("bix,iox->box", x_ft[:, :, :l], w)

        x_out = torch.fft.irfft(out_ft, n=N, dim=-1, norm="ortho")
        return x_out.permute(0, 2, 1)  # [B, N, c*k]


class MWT_CZ1d(nn.Module):
    """MultiWavelet Transform block (core operator layer)."""

    def __init__(
        self,
        k: int = 5,
        modes: int = 64,
        levels: int = 0,  # L in paper (coarsest levels to skip)
        c: int = 16,
        base: str = "legendre",
    ):
        super().__init__()
        self.k = k
        self.c = c
        self.levels = levels
        self.base = base

        # Precompute filters
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)

        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        # To torch tensors (float32)
        def to_t(m):
            return torch.from_numpy(m).to(torch.float32)

        self.register_buffer("ec_s", torch.cat([to_t(H0.T), to_t(H1.T)], dim=0))
        self.register_buffer("ec_d", torch.cat([to_t(G0.T), to_t(G1.T)], dim=0))
        self.register_buffer("rc_e", torch.cat([to_t(H0r), to_t(G0r)], dim=0))
        self.register_buffer("rc_o", torch.cat([to_t(H1r), to_t(G1r)], dim=0))

        self.A = SparseKernelFT1d(k, modes, c)
        self.B = SparseKernelFT1d(k, modes, c)
        self.C = SparseKernelFT1d(k, modes, c)

        self.norm = nn.LayerNorm(c * k)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, c, k]
        Returns: [B, N, c, k]
        """
        B, N_orig, _, _ = x.shape
        nl = 2 ** math.ceil(math.log2(max(N_orig, 1)))
        pad_len = nl - N_orig

        # Zero-pad on the right (symmetric alternatives: 'reflect', 'replicate')
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len), mode="constant", value=0)

        ns = int(math.log2(nl))

        Ud: List[torch.Tensor] = []
        Us: List[torch.Tensor] = []

        # Decomposition
        for _ in range(ns - self.levels):
            d, x = self.wavelet_transform(x)
            Ud.append(self.A(d) + self.B(x))
            Us.append(self.C(d))

        # Coarsest scale
        x = self.act(self.norm(x))

        # Reconstruction
        for i in range(ns - 1 - self.levels, -1, -1):
            x = x + Us[i]
            x = torch.cat([x, Ud[i]], dim=-1)
            x = self.even_odd(x)

        return x[..., :N_orig, :, :]

    def wavelet_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One level decomposition."""
        xa = torch.cat([x[:, ::2], x[:, 1::2]], dim=-1)
        d = xa @ self.ec_d
        s = xa @ self.ec_s
        return d, s

    def even_odd(self, x: torch.Tensor) -> torch.Tensor:
        """Even-odd interleaving reconstruction."""
        x_e = x @ self.rc_e
        x_o = x @ self.rc_o
        out = torch.zeros(
            B=x.shape[0],
            N=x.shape[1] * 2,
            C=x.shape[2],
            k=self.k,
            device=x.device,
            dtype=x.dtype,
        )
        out[..., ::2, :, :] = x_e
        out[..., 1::2, :, :] = x_o
        return out


class MultiWaveletFeatureExtractor(nn.Module):
    """
    Multi-wavelet feature extractor / operator layer stack.
    Input / output: [B, L, input_channels] ↔ [B, L, out_channels]
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 128,  # c * k latent dim
        modes: int = 32,
        levels: int = 0,
        n_layers: int = 4,
        base: str = "legendre",
        out_channels: int | None = None,
    ):
        super().__init__()
        self.k = 5  # wavelet order – usually 3..7
        self.c = hidden_channels // self.k
        self.out_channels = out_channels or input_channels

        self.project_in = nn.Linear(input_channels, self.c * self.k)

        self.layers = nn.ModuleList(
            [
                MWT_CZ1d(k=self.k, modes=modes, levels=levels, c=self.c, base=base)
                for _ in range(n_layers)
            ]
        )

        self.project_out = nn.Linear(self.c * self.k, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, input_channels]
        Returns: [B, L, out_channels]
        """
        B, L, Cin = x.shape

        x = self.project_in(x)  # → [B, L, c*k]
        x = x.view(B, L, self.c, self.k)

        residual = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
            # Optional residual connection every few layers
            if (i + 1) % 2 == 0:
                x = x + residual

        x = x.view(B, L, -1)
        out = self.project_out(x)

        return out
