from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .poly import DEFAULT_POLY_FAMILIES
from .poly import PolyFamily
from .poly import PolyLayerConfig
from .poly import build_poly_layer
from .router import RouterConfig
from .router import TokenRouter


def resolve_poly_config(
    base: PolyLayerConfig | None = None,
    *,
    degree: int | None = None,
    hahn_alpha: float | None = None,
    hahn_beta: float | None = None,
    hahn_N: int | None = None,
    jacobi_alpha: float | None = None,
    jacobi_beta: float | None = None,
    wavelet_num: int | None = None,
    wavelet_base_freq: float | None = None,
    wavelet_learn_freq: bool | None = None,
    wavelet_learn_scale: bool | None = None,
    wavelet_learn_shift: bool | None = None,
    fourier_base_freq: float | None = None,
    fourier_learn_freq: bool | None = None,
) -> PolyLayerConfig:
    cfg = base or PolyLayerConfig()
    updates = {
        "degree": cfg.degree if degree is None else degree,
        "hahn_alpha": cfg.hahn_alpha if hahn_alpha is None else hahn_alpha,
        "hahn_beta": cfg.hahn_beta if hahn_beta is None else hahn_beta,
        "hahn_N": cfg.hahn_N if hahn_N is None else hahn_N,
        "jacobi_alpha": cfg.jacobi_alpha if jacobi_alpha is None else jacobi_alpha,
        "jacobi_beta": cfg.jacobi_beta if jacobi_beta is None else jacobi_beta,
        "wavelet_num": cfg.wavelet_num if wavelet_num is None else wavelet_num,
        "wavelet_base_freq": (
            cfg.wavelet_base_freq if wavelet_base_freq is None else wavelet_base_freq
        ),
        "wavelet_learn_freq": (
            cfg.wavelet_learn_freq if wavelet_learn_freq is None else wavelet_learn_freq
        ),
        "wavelet_learn_scale": (
            cfg.wavelet_learn_scale
            if wavelet_learn_scale is None
            else wavelet_learn_scale
        ),
        "wavelet_learn_shift": (
            cfg.wavelet_learn_shift
            if wavelet_learn_shift is None
            else wavelet_learn_shift
        ),
        "fourier_base_freq": (
            cfg.fourier_base_freq if fourier_base_freq is None else fourier_base_freq
        ),
        "fourier_learn_freq": (
            cfg.fourier_learn_freq if fourier_learn_freq is None else fourier_learn_freq
        ),
    }
    return cfg.with_updates(**updates)


class PolyKAN(nn.Module):
    def __init__(
        self,
        family: PolyFamily,
        input_dim: int,
        output_dim: int,
        *,
        poly_config: PolyLayerConfig | None = None,
        degree: int | None = None,
        hahn_alpha: float | None = None,
        hahn_beta: float | None = None,
        hahn_N: int | None = None,
        jacobi_alpha: float | None = None,
        jacobi_beta: float | None = None,
        wavelet_num: int | None = None,
        wavelet_base_freq: float | None = None,
        wavelet_learn_freq: bool | None = None,
        wavelet_learn_scale: bool | None = None,
        wavelet_learn_shift: bool | None = None,
        fourier_base_freq: float | None = None,
        fourier_learn_freq: bool | None = None,
    ):
        super().__init__()
        self.family = family
        self.config = resolve_poly_config(
            poly_config,
            degree=degree,
            hahn_alpha=hahn_alpha,
            hahn_beta=hahn_beta,
            hahn_N=hahn_N,
            jacobi_alpha=jacobi_alpha,
            jacobi_beta=jacobi_beta,
            wavelet_num=wavelet_num,
            wavelet_base_freq=wavelet_base_freq,
            wavelet_learn_freq=wavelet_learn_freq,
            wavelet_learn_scale=wavelet_learn_scale,
            wavelet_learn_shift=wavelet_learn_shift,
            fourier_base_freq=fourier_base_freq,
            fourier_learn_freq=fourier_learn_freq,
        )
        self.layer = build_poly_layer(
            family=family,
            input_dim=input_dim,
            output_dim=output_dim,
            config=self.config,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class PolyKANBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_num: int,
        family: PolyFamily,
        *,
        poly_config: PolyLayerConfig | None = None,
        degree_intra: int | None = None,
        degree_inter: int | None = None,
        hahn_alpha: float | None = None,
        hahn_beta: float | None = None,
        hahn_N: int | None = None,
        jacobi_alpha: float | None = None,
        jacobi_beta: float | None = None,
        wavelet_num: int | None = None,
        wavelet_base_freq: float | None = None,
        wavelet_learn_freq: bool | None = None,
        wavelet_learn_scale: bool | None = None,
        wavelet_learn_shift: bool | None = None,
        fourier_base_freq: float | None = None,
        fourier_learn_freq: bool | None = None,
    ):
        super().__init__()
        base_config = resolve_poly_config(
            poly_config,
            hahn_alpha=hahn_alpha,
            hahn_beta=hahn_beta,
            hahn_N=hahn_N,
            jacobi_alpha=jacobi_alpha,
            jacobi_beta=jacobi_beta,
            wavelet_num=wavelet_num,
            wavelet_base_freq=wavelet_base_freq,
            wavelet_learn_freq=wavelet_learn_freq,
            wavelet_learn_scale=wavelet_learn_scale,
            wavelet_learn_shift=wavelet_learn_shift,
            fourier_base_freq=fourier_base_freq,
            fourier_learn_freq=fourier_learn_freq,
        )
        intra_config = base_config.with_updates(
            degree=base_config.degree if degree_intra is None else degree_intra
        )
        inter_config = base_config.with_updates(
            degree=base_config.degree if degree_inter is None else degree_inter
        )

        self.intrapatch_kan = PolyKAN(
            family=family,
            input_dim=dim,
            output_dim=dim,
            poly_config=intra_config,
        )
        self.interpatch_kan = PolyKAN(
            family=family,
            input_dim=patch_num,
            output_dim=patch_num,
            poly_config=inter_config,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.intrapatch_kan(x)
        x = x.permute(0, 2, 1)
        x = self.interpatch_kan(x)
        x = x.permute(0, 2, 1)
        return x


class HeteroMoKANLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_num: int,
        families: Sequence[PolyFamily],
        *,
        poly_config: PolyLayerConfig | None = None,
        router_config: RouterConfig | None = None,
        degree_intra: int | None = None,
        degree_inter: int | None = None,
        top_k: int | None = 2,
        router_temperature: float | None = None,
        router_hidden: int | None = None,
        load_balance_coef: float = 0.0,
        hahn_alpha: float | None = None,
        hahn_beta: float | None = None,
        hahn_N: int | None = None,
        jacobi_alpha: float | None = None,
        jacobi_beta: float | None = None,
        wavelet_num: int | None = None,
        wavelet_base_freq: float | None = None,
        wavelet_learn_freq: bool | None = None,
        wavelet_learn_scale: bool | None = None,
        wavelet_learn_shift: bool | None = None,
        fourier_base_freq: float | None = None,
        fourier_learn_freq: bool | None = None,
    ):
        super().__init__()
        self.families = tuple(families)
        if not self.families:
            raise ValueError("HeteroMoKANLayer requires at least one expert family")
        self.num_experts = len(self.families)
        self.load_balance_coef = float(load_balance_coef)

        self.router = TokenRouter(
            d_model=d_model,
            num_experts=self.num_experts,
            router_config=router_config,
            hidden=router_hidden,
            temperature=router_temperature,
            top_k=top_k,
        )
        self.experts = nn.ModuleList(
            [
                PolyKANBlock(
                    dim=d_model,
                    patch_num=patch_num,
                    family=family,
                    poly_config=poly_config,
                    degree_intra=degree_intra,
                    degree_inter=degree_inter,
                    hahn_alpha=hahn_alpha,
                    hahn_beta=hahn_beta,
                    hahn_N=hahn_N,
                    jacobi_alpha=jacobi_alpha,
                    jacobi_beta=jacobi_beta,
                    wavelet_num=wavelet_num,
                    wavelet_base_freq=wavelet_base_freq,
                    wavelet_learn_freq=wavelet_learn_freq,
                    wavelet_learn_scale=wavelet_learn_scale,
                    wavelet_learn_shift=wavelet_learn_shift,
                    fourier_base_freq=fourier_base_freq,
                    fourier_learn_freq=fourier_learn_freq,
                )
                for family in self.families
            ]
        )
        self.last_router_probs: Tensor | None = None
        self.last_router_logits: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        probs, logits = self.router(x)
        self.last_router_probs = probs.detach()
        self.last_router_logits = logits.detach()

        outs = torch.stack([expert(x) for expert in self.experts], dim=0)
        weights = probs.permute(2, 0, 1).unsqueeze(-1)
        y = (outs * weights).sum(dim=0)

        mean_probs = probs.mean(dim=(0, 1))
        aux: dict[str, Tensor] = {
            "router_probs": probs,
            "router_logits": logits,
            "mean_expert_prob": mean_probs,
        }

        if self.load_balance_coef > 0.0:
            uniform = torch.full_like(mean_probs, 1.0 / self.num_experts)
            aux["load_balance_loss"] = self.load_balance_coef * F.mse_loss(
                mean_probs, uniform
            )
        else:
            aux["load_balance_loss"] = x.new_zeros(())
        return y, aux


class FlattenHead(nn.Module):
    def __init__(self, nf: int, target_window: int):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, 336)
        self.linear2 = nn.Linear(336, target_window)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Backbone(nn.Module):
    def __init__(
        self,
        patch_num: int,
        patch_len: int,
        d_model: int = 128,
        depth: int = 5,
        families: Sequence[PolyFamily] | None = None,
        *,
        poly_config: PolyLayerConfig | None = None,
        router_config: RouterConfig | None = None,
        degree_intra: int | None = None,
        degree_inter: int | None = None,
        top_k: int | None = 2,
        router_temperature: float | None = None,
        router_hidden: int | None = None,
        load_balance_coef: float = 0.0,
        hahn_alpha: float | None = None,
        hahn_beta: float | None = None,
        hahn_N: int | None = None,
        jacobi_alpha: float | None = None,
        jacobi_beta: float | None = None,
        wavelet_num: int | None = None,
        wavelet_base_freq: float | None = None,
        wavelet_learn_freq: bool | None = None,
        wavelet_learn_scale: bool | None = None,
        wavelet_learn_shift: bool | None = None,
        fourier_base_freq: float | None = None,
        fourier_learn_freq: bool | None = None,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.seq_len = patch_num

        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = nn.Parameter(torch.randn(1, patch_num, d_model))

        expert_families = tuple(families or DEFAULT_POLY_FAMILIES)
        self.encoder = nn.ModuleList(
            [
                HeteroMoKANLayer(
                    d_model=d_model,
                    patch_num=patch_num,
                    families=expert_families,
                    poly_config=poly_config,
                    router_config=router_config,
                    degree_intra=degree_intra,
                    degree_inter=degree_inter,
                    top_k=top_k,
                    router_temperature=router_temperature,
                    router_hidden=router_hidden,
                    load_balance_coef=load_balance_coef,
                    hahn_alpha=hahn_alpha,
                    hahn_beta=hahn_beta,
                    hahn_N=hahn_N,
                    jacobi_alpha=jacobi_alpha,
                    jacobi_beta=jacobi_beta,
                    wavelet_num=wavelet_num,
                    wavelet_base_freq=wavelet_base_freq,
                    wavelet_learn_freq=wavelet_learn_freq,
                    wavelet_learn_scale=wavelet_learn_scale,
                    wavelet_learn_shift=wavelet_learn_shift,
                    fourier_base_freq=fourier_base_freq,
                    fourier_learn_freq=fourier_learn_freq,
                )
                for _ in range(depth)
            ]
        )
        self.last_aux: dict[str, Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)

        z = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        z = z + self.W_pos[:, : self.seq_len, :]

        lb_losses = []
        for layer in self.encoder:
            y, aux = layer(z)
            z = z + y
            lb_losses.append(aux["load_balance_loss"].reshape(()))

        self.last_aux = {"moe_load_balance_loss": torch.stack(lb_losses).sum()}
        z = z.reshape(-1, n_vars, z.shape[-2], z.shape[-1])
        z = z.permute(0, 1, 3, 2)
        return z


BackBone = Backbone
Flatten_Head = FlattenHead


__all__ = [
    "BackBone",
    "Backbone",
    "Flatten_Head",
    "FlattenHead",
    "HeteroMoKANLayer",
    "PolyFamily",
    "PolyKAN",
    "PolyKANBlock",
    "PolyLayerConfig",
    "resolve_poly_config",
]
