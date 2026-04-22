from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor, nn

PolyFamily = Literal[
    "hahn",
    "chebyshev",
    "jacobi",
    "legendre",
    "fourier",
    "wavelet",
    "hermite",
]


@dataclass(frozen=True)
class PolyLayerConfig:
    degree: int = 3
    hahn_alpha: float = 1.0
    hahn_beta: float = 1.0
    hahn_N: int = 7
    jacobi_alpha: float = 0.0
    jacobi_beta: float = 0.0
    wavelet_num: int = 8
    wavelet_base_freq: float = 1.0
    wavelet_learn_freq: bool = True
    wavelet_learn_scale: bool = True
    wavelet_learn_shift: bool = True
    fourier_base_freq: float = 1.0
    fourier_learn_freq: bool = False

    def with_updates(self, **kwargs) -> PolyLayerConfig:
        return replace(self, **kwargs)


def _reshape_in_out(x: Tensor, input_dim: int) -> tuple[Tensor, tuple[int, ...]]:
    if x.shape[-1] != input_dim:
        raise ValueError(f"Expected last dim {input_dim}, got {x.shape[-1]}")
    orig = x.shape
    return x.reshape(-1, input_dim), orig


def _restore_shape(y: Tensor, orig_shape: tuple[int, ...], output_dim: int) -> Tensor:
    return y.reshape(*orig_shape[:-1], output_dim)


def _tanh_to_unit(x: Tensor) -> Tensor:
    return torch.tanh(x)


def _init_coeffs(input_dim: int, output_dim: int, basis_size: int) -> nn.Parameter:
    coeffs = nn.Parameter(torch.empty(input_dim, output_dim, basis_size))
    nn.init.normal_(coeffs, mean=0.0, std=1.0 / (input_dim * basis_size))
    return coeffs


def _resolve_family_name(family: str) -> str:
    family_name = str(family).strip().lower()
    if family_name not in POLY_FAMILIES:
        available = ", ".join(POLY_FAMILIES)
        raise ValueError(f"Unknown family '{family}'. Available: {available}")
    return family_name


POLY_FAMILIES: tuple[PolyFamily, ...] = (
    "hahn",
    "chebyshev",
    "jacobi",
    "legendre",
    "fourier",
    "wavelet",
    "hermite",
)

DEFAULT_POLY_FAMILIES: tuple[PolyFamily, ...] = (
    "hahn",
    "chebyshev",
    "jacobi",
    "legendre",
)


class HahnPolynomials(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int,
        alpha: float,
        beta: float,
        N: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.a = float(alpha)
        self.b = float(beta)
        self.N = int(N)
        self.coeffs = _init_coeffs(input_dim, output_dim, degree + 1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        terms: list[Tensor] = []
        T0 = torch.ones(x2.shape[0], self.input_dim, device=x2.device, dtype=x2.dtype)
        terms.append(T0)

        if self.degree >= 1:
            T1 = 1.0 - (
                ((self.a + self.b + 2.0) * x2) / ((self.a + 1.0) * float(self.N))
            )
            terms.append(T1)

        for n in range(2, self.degree + 1):
            m = float(n - 1)
            A = (m + self.a + self.b + 1.0) * (m + self.a + 1.0) * (float(self.N) - m)
            A /= 2.0 * m + self.a + self.b + 1.0
            A /= 2.0 * m + self.a + self.b + 2.0

            C = m * (m + self.a + self.b + float(self.N) + 1.0) * (m + self.b)
            C /= 2.0 * m + self.a + self.b
            C /= 2.0 * m + self.a + self.b + 1.0

            Tn = ((A + C - x2) * terms[-1] - (C * terms[-2])) / A
            terms.append(Tn)

        basis = torch.stack(terms, dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y


class ChebyshevPolynomials(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, degree: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.coeffs = _init_coeffs(input_dim, output_dim, degree + 1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        terms: list[Tensor] = [torch.ones_like(x2)]
        if self.degree >= 1:
            terms.append(x2)
        for _ in range(2, self.degree + 1):
            terms.append(2.0 * x2 * terms[-1] - terms[-2])

        basis = torch.stack(terms, dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y


class JacobiPolynomials(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int,
        alpha: float = 0.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.coeffs = _init_coeffs(input_dim, output_dim, degree + 1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        a = self.alpha
        b = self.beta

        terms: list[Tensor] = [torch.ones_like(x2)]
        if self.degree >= 1:
            terms.append(0.5 * ((a - b) + (a + b + 2.0) * x2))

        for n in range(2, self.degree + 1):
            nf = float(n)
            A = 2.0 * nf * (nf + a + b) * (2.0 * nf + a + b - 2.0)
            Bc = (
                (2.0 * nf + a + b - 1.0)
                * (2.0 * nf + a + b)
                * (2.0 * nf + a + b - 2.0)
            )
            C = (2.0 * nf + a + b - 1.0) * (a * a - b * b)
            D = 2.0 * (nf + a - 1.0) * (nf + b - 1.0) * (2.0 * nf + a + b)
            terms.append(((Bc * x2 + C) * terms[-1] - D * terms[-2]) / A)

        basis = torch.stack(terms, dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y


class WaveletKAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_wavelets: int = 8,
        base_freq: float = 1.0,
        learn_freq: bool = True,
        learn_scale: bool = True,
        learn_shift: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_wavelets = num_wavelets

        self.scales = nn.Parameter(
            torch.ones(input_dim, num_wavelets), requires_grad=learn_scale
        )
        self.shifts = nn.Parameter(
            torch.linspace(-1.5, 1.5, num_wavelets).repeat(input_dim, 1),
            requires_grad=learn_shift,
        )
        self.freqs = nn.Parameter(
            torch.ones(input_dim, num_wavelets) * base_freq, requires_grad=learn_freq
        )
        self.coeffs = _init_coeffs(input_dim, output_dim, num_wavelets + 1)

    def _morlet(self, x: Tensor) -> Tensor:
        diff = x.unsqueeze(-1) - self.shifts
        gauss = torch.exp(-0.5 * (diff / (self.scales + 1e-6)) ** 2)
        oscil = torch.cos(self.freqs * diff)
        return gauss * oscil

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        basis = self._morlet(x2)
        basis = torch.cat([basis, torch.ones_like(basis[..., :1])], dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y


class FourierKAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int = 5,
        base_freq: float = 1.0,
        learn_freq: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.num_terms = 2 * degree + 1
        self.freq = nn.Parameter(
            torch.full((input_dim,), base_freq), requires_grad=learn_freq
        )
        self.coeffs = _init_coeffs(input_dim, output_dim, self.num_terms)

    def _fourier_basis(self, x: Tensor) -> Tensor:
        freq = self.freq.unsqueeze(0).unsqueeze(-1)
        omega_x = 2.0 * math.pi * freq * x.unsqueeze(-1)
        k = torch.arange(0, self.degree + 1, device=x.device, dtype=x.dtype)
        angle = k.unsqueeze(0).unsqueeze(0) * omega_x
        cos_terms = torch.cos(angle)
        sin_terms = torch.sin(angle[..., 1:])
        return torch.cat([cos_terms, sin_terms], dim=-1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)
        basis = self._fourier_basis(x2)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y


class ProbHermitePolynomials(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, degree: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.coeffs = _init_coeffs(input_dim, output_dim, degree + 1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        terms: list[Tensor] = [torch.ones_like(x2)]
        if self.degree >= 1:
            terms.append(x2)
        for n in range(1, self.degree):
            terms.append(x2 * terms[-1] - float(n) * terms[-2])

        basis = torch.stack(terms, dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y


def build_poly_layer(
    family: PolyFamily,
    input_dim: int,
    output_dim: int,
    config: PolyLayerConfig | None = None,
) -> nn.Module:
    cfg = config or PolyLayerConfig()
    family_name = _resolve_family_name(family)

    if family_name == "hahn":
        return HahnPolynomials(
            input_dim=input_dim,
            output_dim=output_dim,
            degree=cfg.degree,
            alpha=cfg.hahn_alpha,
            beta=cfg.hahn_beta,
            N=cfg.hahn_N,
        )
    if family_name == "chebyshev":
        return ChebyshevPolynomials(input_dim, output_dim, cfg.degree)
    if family_name == "jacobi":
        return JacobiPolynomials(
            input_dim=input_dim,
            output_dim=output_dim,
            degree=cfg.degree,
            alpha=cfg.jacobi_alpha,
            beta=cfg.jacobi_beta,
        )
    if family_name == "legendre":
        return JacobiPolynomials(input_dim, output_dim, cfg.degree, 0.0, 0.0)
    if family_name == "wavelet":
        num_wavelets = cfg.degree if cfg.degree > 4 else cfg.wavelet_num
        return WaveletKAN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_wavelets=num_wavelets,
            base_freq=cfg.wavelet_base_freq,
            learn_freq=cfg.wavelet_learn_freq,
            learn_scale=cfg.wavelet_learn_scale,
            learn_shift=cfg.wavelet_learn_shift,
        )
    if family_name == "fourier":
        return FourierKAN(
            input_dim=input_dim,
            output_dim=output_dim,
            degree=cfg.degree,
            base_freq=cfg.fourier_base_freq,
            learn_freq=cfg.fourier_learn_freq,
        )
    if family_name == "hermite":
        return ProbHermitePolynomials(input_dim, output_dim, cfg.degree)
    raise AssertionError(f"Unhandled family '{family_name}'")


def list_poly_families() -> tuple[PolyFamily, ...]:
    return POLY_FAMILIES


__all__ = [
    "ChebyshevPolynomials",
    "DEFAULT_POLY_FAMILIES",
    "FourierKAN",
    "HahnPolynomials",
    "JacobiPolynomials",
    "POLY_FAMILIES",
    "PolyFamily",
    "PolyLayerConfig",
    "ProbHermitePolynomials",
    "WaveletKAN",
    "build_poly_layer",
    "list_poly_families",
]
