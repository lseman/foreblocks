from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch import nn

from .backbone import Backbone
from .backbone import FlattenHead
from .poly import PolyFamily
from .poly import PolyLayerConfig
from .router import RouterConfig


def compute_patch_num(
    context_window: int,
    patch_len: int,
    stride: int,
    padding_patch: str | None = None,
) -> int:
    patch_num = int((context_window - patch_len) / stride + 1)
    if padding_patch == "end":
        patch_num += 1
    return patch_num


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unsupported RevIN mode '{mode}'")

    def _get_statistics(self, x: Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


class Model(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        padding_patch: str | None = None,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
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
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = compute_patch_num(
            context_window=context_window,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
        )
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        self.backbone = Backbone(
            patch_num=patch_num,
            patch_len=patch_len,
            d_model=d_model,
            depth=depth,
            families=families,
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
        self.head_nf = d_model * patch_num
        self.head = FlattenHead(self.head_nf, target_window)
        self.revert = nn.Linear(d_model, c_in)

    def forward(self, z: Tensor) -> Tensor:
        if self.revin:
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)
        z = self.head(z)

        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
        return z


__all__ = ["Model", "RevIN", "compute_patch_num"]
