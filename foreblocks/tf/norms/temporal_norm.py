from typing import Optional, Tuple

import torch
import torch.nn as nn


class TemporalNorm(nn.Module):
    """
    Temporal normalization for [B, T, D] tensors.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        affine: bool = True,
        mode: str = "standard",
        causal: bool = False,
        window_size: Optional[int] = None,
        center: bool = True,
        scale: bool = True,
    ):
        super().__init__()
        assert mode in {"standard", "robust"}
        self.d_model = d_model
        self.eps = eps
        self.affine = affine
        self.mode = mode
        self.causal = causal
        self.window_size = window_size
        self.center = center
        self.scale = scale

        if affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @staticmethod
    def _apply_mask(
        x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mask is None:
            return x, None
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.expand(-1, -1, x.size(-1))
        return x * mask, mask

    def _reduce_full(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "standard":
            if mask is None:
                loc = (
                    x.mean(dim=1, keepdim=True)
                    if self.center
                    else torch.zeros_like(x[:, :1, :])
                )
                var = (
                    x.var(dim=1, unbiased=False, keepdim=True)
                    if self.scale
                    else torch.ones_like(x[:, :1, :])
                )
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                loc = (
                    x.sum(dim=1, keepdim=True) / denom
                    if self.center
                    else torch.zeros_like(x[:, :1, :])
                )
                diff = x - loc
                var = (
                    (diff * diff * mask).sum(dim=1, keepdim=True) / denom
                    if self.scale
                    else torch.ones_like(loc)
                )
            scale = (var + self.eps).sqrt()
        else:
            if mask is not None:
                x_ = torch.where(mask > 0, x, torch.nan)
                loc = (
                    torch.nanmedian(x_, dim=1, keepdim=True).values
                    if self.center
                    else torch.zeros_like(x[:, :1, :])
                )
                diff = torch.abs(x - loc)
                mad = (
                    torch.nanmedian(
                        torch.where(mask > 0, diff, torch.nan), dim=1, keepdim=True
                    ).values
                    if self.scale
                    else torch.ones_like(loc)
                )
            else:
                loc = (
                    torch.median(x, dim=1, keepdim=True).values
                    if self.center
                    else torch.zeros_like(x[:, :1, :])
                )
                mad = (
                    torch.median(torch.abs(x - loc), dim=1, keepdim=True).values
                    if self.scale
                    else torch.ones_like(loc)
                )
            scale = mad * (1.0 / 0.741) + self.eps
        return loc, scale

    def _reduce_rolling(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        W = int(self.window_size)
        idx = torch.arange(T, device=x.device)

        if self.causal:
            l = (idx - W + 1).clamp_min(0)
            r = idx
        else:
            half = W // 2
            l = (idx - half).clamp_min(0)
            r = (idx + (W - half - 1)).clamp_max(T - 1)

        if self.mode == "standard":
            if mask is None:
                m = x.new_ones(B, T, D)
            else:
                m = mask

            mcum = m.cumsum(dim=1)
            sx = (x * m).cumsum(dim=1)
            sx2 = (x * x * m).cumsum(dim=1)

            def window_sum(cum):
                pad = cum.new_zeros(B, 1, D)
                cum_pad = torch.cat([pad, cum], dim=1)
                r1 = cum_pad.gather(1, (r + 1)[None, :, None].expand(B, -1, D))
                l0 = cum_pad.gather(1, l[None, :, None].expand(B, -1, D))
                return r1 - l0

            win_m = window_sum(mcum)
            win_sx = window_sum(sx)
            win_sx2 = window_sum(sx2)

            denom = win_m.clamp_min(1.0)
            loc = win_sx / denom if self.center else x.new_zeros(B, T, D)
            var = win_sx2 / denom - loc * loc if self.scale else x.new_ones(B, T, D)
            scale = (var + self.eps).sqrt()
            return loc, scale

        loc_list = []
        scale_list = []
        half = W // 2

        for t in range(T):
            if self.causal:
                a, b = max(0, t - W + 1), t + 1
            else:
                a, b = max(0, t - half), min(T, t + (W - half))
            xw = x[:, a:b, :]
            if mask is not None:
                mw = mask[:, a:b, :].bool()
                xw = torch.where(mw, xw, torch.nan)
                med = (
                    torch.nanmedian(xw, dim=1, keepdim=True).values
                    if self.center
                    else xw.new_zeros(B, 1, D)
                )
                mad = (
                    torch.nanmedian(torch.abs(xw - med), dim=1, keepdim=True).values
                    if self.scale
                    else xw.new_ones(B, 1, D)
                )
            else:
                med = (
                    torch.median(xw, dim=1, keepdim=True).values
                    if self.center
                    else xw.new_zeros(B, 1, D)
                )
                mad = (
                    torch.median(torch.abs(xw - med), dim=1, keepdim=True).values
                    if self.scale
                    else xw.new_ones(B, 1, D)
                )
            loc_list.append(med.squeeze(1))
            scale_list.append((mad * (1.0 / 0.741) + self.eps).squeeze(1))
        loc = torch.stack(loc_list, dim=1)
        scale = torch.stack(scale_list, dim=1)
        return loc, scale

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                f"TemporalNorm expects [B, T, D={self.d_model}], got {tuple(x.shape)}"
            )
        B, T, D = x.shape

        x_masked, m = self._apply_mask(x, mask)

        if self.window_size is None:
            loc, scale = self._reduce_full(x_masked, m)
            loc = loc.expand(B, T, D)
            scale = scale.expand(B, T, D)
        else:
            loc, scale = self._reduce_rolling(x_masked, m)

        y = x
        if self.center:
            y = y - loc
        if self.scale:
            y = y / scale

        if self.affine:
            y = y * self.weight.view(1, 1, D)
            if self.bias is not None:
                y = y + self.bias.view(1, 1, D)

        if return_stats:
            return y, {"loc": loc.detach(), "scale": scale.detach()}
        return y

    @torch.no_grad()
    def inverse(self, y: torch.Tensor, stats: dict) -> torch.Tensor:
        loc, scale = stats["loc"], stats["scale"]
        x = y
        D = self.d_model
        if self.affine:
            x = x / self.weight.view(1, 1, D)
            if self.bias is not None:
                x = x - self.bias.view(1, 1, D)
        if self.scale:
            x = x * scale
        if self.center:
            x = x + loc
        return x


__all__ = ["TemporalNorm"]
