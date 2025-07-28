from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import eval_legendre
from sympy import Poly, Symbol, chebyshevt, legendre
from torch import Tensor, nn


def legendreDer(k, x):
    def _legendre(k, x):
        return (2 * k + 1) * eval_legendre(k, x)

    out = 0
    for i in np.arange(k - 1, -1, -2):
        out += _legendre(i, x)
    return out


def phi_(phi_c, x, lb=0, ub=1):
    mask = np.logical_or(x < lb, x > ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)


def get_phi_psi(k, base):
    x = Symbol("x")
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    if base == "legendre":
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
            phi_coeff[ki, : ki + 1] = np.flip(
                np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64)
            )
            coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
            phi_2x_coeff[ki, : ki + 1] = np.flip(
                np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64)
            )

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                a = phi_2x_coeff[ki, : ki + 1]
                b = phi_coeff[i, : i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (
                    prod_
                    * 1
                    / (np.arange(len(prod_)) + 1)
                    * np.power(0.5, 1 + np.arange(len(prod_)))
                ).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                a = phi_2x_coeff[ki, : ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (
                    prod_
                    * 1
                    / (np.arange(len(prod_)) + 1)
                    * np.power(0.5, 1 + np.arange(len(prod_)))
                ).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            a = psi1_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm1 = (
                prod_
                * 1
                / (np.arange(len(prod_)) + 1)
                * np.power(0.5, 1 + np.arange(len(prod_)))
            ).sum()

            a = psi2_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm2 = (
                prod_
                * 1
                / (np.arange(len(prod_)) + 1)
                * (1 - np.power(0.5, 1 + np.arange(len(prod_))))
            ).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]

    elif base == "chebyshev":
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, : ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, : ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x - 1), x).all_coeffs()
                phi_coeff[ki, : ki + 1] = np.flip(
                    2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64)
                )
                coeff_ = Poly(chebyshevt(ki, 4 * x - 1), x).all_coeffs()
                phi_2x_coeff[ki, : ki + 1] = np.flip(
                    np.sqrt(2)
                    * 2
                    / np.sqrt(np.pi)
                    * np.array(coeff_).astype(np.float64)
                )

        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]

        x = Symbol("x")
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1)

    return phi, psi1, psi2


def get_filter(base, k):
    def psi(psi1, psi2, i, inp):
        mask = (inp <= 0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)

    if base not in ["legendre", "chebyshev"]:
        raise Exception("Base not supported")

    x = Symbol("x")
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.zeros((k, k))
    PHI1 = np.zeros((k, k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == "legendre":
        roots = Poly(legendre(k, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1 / k / legendreDer(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = (
                    1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                )
                G0[ki, kpi] = (
                    1
                    / np.sqrt(2)
                    * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                )
                H1[ki, kpi] = (
                    1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                )
                G1[ki, kpi] = (
                    1
                    / np.sqrt(2)
                    * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()
                )

        PHI0 = np.eye(k)
        PHI1 = np.eye(k)

    elif base == "chebyshev":
        x = Symbol("x")
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = (
                    1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                )
                G0[ki, kpi] = (
                    1
                    / np.sqrt(2)
                    * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                )
                H1[ki, kpi] = (
                    1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                )
                G1[ki, kpi] = (
                    1
                    / np.sqrt(2)
                    * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()
                )

                PHI0[ki, kpi] = (wm * phi[ki](2 * x_m) * phi[kpi](2 * x_m)).sum() * 2
                PHI1[ki, kpi] = (
                    wm * phi[ki](2 * x_m - 1) * phi[kpi](2 * x_m - 1)
                ).sum() * 2

        PHI0[np.abs(PHI0) < 1e-8] = 0
        PHI1[np.abs(PHI1) < 1e-8] = 0

    H0[np.abs(H0) < 1e-8] = 0
    H1[np.abs(H1) < 1e-8] = 0
    G0[np.abs(G0) < 1e-8] = 0
    G1[np.abs(G1) < 1e-8] = 0

    return H0, H1, G0, G1, PHI0, PHI1


class sparseKernelFT1d(nn.Module):
    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = 1 / (c * k * c * k)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.float)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.float)
        )
        self.weights1.requires_grad = True
        self.weights2.requires_grad = True
        self.k = k

    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(
                weights, torch.zeros_like(weights).to(weights.device)
            )
        if x_flag or w_flag:
            return torch.complex(
                torch.einsum(order, x.real, weights.real)
                - torch.einsum(order, x.imag, weights.imag),
                torch.einsum(order, x.real, weights.imag)
                + torch.einsum(order, x.imag, weights.real),
            )
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(
            "bix,iox->box",
            x_fft[:, :, :l],
            torch.complex(self.weights1, self.weights2)[:, :, :l],
        )
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MWT_CZ1d(nn.Module):
    def __init__(
        self, k=3, alpha=64, L=0, c=1, base="legendre", initializer=None, **kwargs
    ):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer("ec_s", torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer("ec_d", torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer("rc_e", torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer("rc_o", torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, k)
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0 : nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]

        return x

    def wavelet_transform(self, x):
        xa = torch.cat(
            [
                x[:, ::2, :, :],
                x[:, 1::2, :, :],
            ],
            -1,
        )
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class MultiWaveletFeatureExtractor(nn.Module):
    """
    Simplified wavelet-based feature extractor.
    Applies Multiwavelet Transform (MWT_CZ) to extract time-frequency features.
    """

    def __init__(
        self,
        input_channels=1,
        wavelet_dim=128,
        wavelet_modes=8,
        alpha=16,
        L=0,
        n_layers=1,
        base="legendre",
        out_channels=1,
    ):
        super().__init__()
        self.k = wavelet_modes
        self.c = wavelet_dim
        self.L = L
        self.n_layers = n_layers

        # Project input to wavelet space
        self.project_in = nn.Linear(input_channels, self.c * self.k)
        self.wavelet_blocks = nn.ModuleList(
            [
                MWT_CZ1d(k=self.k, alpha=alpha, L=L, c=self.c, base=base)
                for _ in range(n_layers)
            ]
        )
        # Optional projection back to original dimension or another latent space
        self.project_out = nn.Linear(self.c * self.k, out_channels)

    def forward(self, x):
        """
        x: [B, L, C]
        Returns: [B, L, C] — extracted wavelet-based features
        """
        B, L, C = x.shape
        x_proj = self.project_in(x)  # [B, L, c * k]
        x_wavelet = x_proj.view(B, L, self.c, self.k)

        for i, layer in enumerate(self.wavelet_blocks):
            x_wavelet = layer(x_wavelet)
            if i < self.n_layers - 1:
                x_wavelet = F.relu(x_wavelet)

        x_flat = x_wavelet.view(B, L, -1)  # [B, L, c * k]
        out = self.project_out(x_flat)  # [B, L, C]
        return out
