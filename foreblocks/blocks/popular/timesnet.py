from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ui.node_spec import node

from .transformer_aux import create_norm_layer  # your norm factory


# ------------------------------------------------------------
# Period estimation (FFT-based autocorrelation top-k periods)
# ------------------------------------------------------------
def _topk_periods(
    x: torch.Tensor,
    k: int,
    min_period: int = 2,
    max_period: Optional[int] = None,
    max_period_frac: float = 0.5,
) -> torch.Tensor:
    """
    x: [B, L, C]
    Returns: periods [B, k] (int64)
    """
    B, L, C = x.shape
    x0 = x - x.mean(dim=1, keepdim=True)
    Xf = torch.fft.rfft(x0.float(), dim=1)  # [B, L//2+1, C]
    Sxx = (Xf * torch.conj(Xf)).real.sum(dim=-1)  # [B, L//2+1]
    ac = torch.fft.irfft(Sxx, n=L, dim=1).real  # [B, L]

    lo = max(min_period, 1)
    hi = min(L - 1, max_period if max_period is not None else int(L * max_period_frac))
    if lo > hi:
        lo, hi = 1, L - 1

    band = ac[:, lo : hi + 1]
    k = min(k, band.size(1))
    idx = torch.topk(band, k=k, dim=1, largest=True).indices + lo  # [B, k]
    return idx


# ------------------------------------------------------------
# Inception-style 2D conv block over (patches × period)
# ------------------------------------------------------------
class Inception2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ks: Tuple[int, ...] = (3, 5, 7),
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = max(in_ch, out_ch) * expand
        self.branches = nn.ModuleList()
        for k in ks:
            pad = (k - 1) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch, in_ch, (k, k), padding=pad, groups=in_ch, bias=False
                    ),
                    nn.GELU(),
                    nn.Conv2d(in_ch, hidden, 1, bias=False),
                )
            )
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden * len(ks), out_ch, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        return self.proj(y)


# ------------------------------------------------------------
# TimesBlock (fixed version)
# ------------------------------------------------------------
@node(
    type_id="times_block",
    name="Times Block",
    category="Popular",
)
class TimesBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        k_periods: int = 3,
        hidden: Optional[int] = None,
        ks: Tuple[int, ...] = (3, 5, 7),
        expand: int = 2,
        dropout: float = 0.1,
        norm_type: str = "rms",
        layer_norm_eps: float = 1e-5,
        use_glu_gate: bool = True,
        max_period_frac: float = 0.5,
    ):
        super().__init__()
        self.k = k_periods
        self.max_period_frac = max_period_frac

        self.norm = create_norm_layer(norm_type, d_model, eps=layer_norm_eps)
        self.proj_in = nn.Linear(d_model, d_model)

        if use_glu_gate:
            self.glu_proj = nn.Linear(d_model, d_model * 2)
            self.glu = nn.GLU(dim=-1)
        else:
            self.glu_proj = self.glu = None

        out_inner = d_model if hidden is None else hidden
        self.inception = Inception2D(
            d_model, out_inner, ks=ks, expand=expand, dropout=dropout
        )
        self.proj_out = nn.Linear(out_inner, d_model)
        self.drop = nn.Dropout(dropout)

    def _fold(self, x: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
        B, L, C = x.shape
        Np = (L + p - 1) // p
        pad = Np * p - L
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))  # pad right
        x = x.view(B, Np, p, C).permute(0, 3, 1, 2).contiguous()  # [B, C, Np, p]
        return x, pad

    def _unfold(self, x: torch.Tensor, pad: int, orig_L: int) -> torch.Tensor:
        B, C, Np, p = x.shape
        y = x.permute(0, 2, 3, 1).contiguous().view(B, Np * p, C)
        if pad > 0:
            y = y[:, :-pad, :]
        return y[:, :orig_L, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        res = x

        h = self.norm(x)

        if self.glu is not None:
            h = self.glu(self.glu_proj(h))  # → [B,L,d_model]
        h = self.proj_in(h)

        periods = _topk_periods(
            h, k=self.k, min_period=2, max_period_frac=self.max_period_frac
        )  # [B, k]

        agg = torch.zeros_like(h)

        for i in range(self.k):
            # Use per-sample period (better than global max)
            p_batch = periods[:, i].clamp(min=2)  # [B]
            # For simplicity we still process one p at a time → could vectorize with masking
            for b in range(B):
                p = p_batch[b].item()
                Xp, pad = self._fold(h[b : b + 1], p)  # [1, C, Np, p]
                Yp = self.inception(Xp)
                y = self._unfold(Yp, pad, L)
                agg[b : b + 1] += y

        agg = agg / self.k

        out = self.proj_out(self.drop(agg))
        return res + out


# ------------------------------------------------------------
# TimesNet Head (improved readout + fixes)
# ------------------------------------------------------------
@node(
    type_id="timesnet",
    name="TimesNet",
    category="Backbone",
    color="bg-gradient-to-br from-rose-600 to-rose-800",
)
class TimesNetHeadCustom(nn.Module):
    def __init__(
        self,
        pred_len: int,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 512,
        n_blocks: int = 2,
        k_periods: int = 3,
        inception_kernels: Tuple[int, ...] = (3, 5, 7),
        expand: int = 2,
        dropout: float = 0.1,
        norm_type: str = "rms",
        layer_norm_eps: float = 1e-5,
        use_glu_gate: bool = True,
        use_channel_mixer: bool = False,
        quantiles: Optional[Tuple[float, ...]] = None,
        max_period_frac: float = 0.5,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.quantiles = quantiles

        self.enc_in = nn.Linear(in_channels, d_model)

        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=d_model,
                    k_periods=k_periods,
                    ks=inception_kernels,
                    expand=expand,
                    dropout=dropout,
                    norm_type=norm_type,
                    layer_norm_eps=layer_norm_eps,
                    use_glu_gate=use_glu_gate,
                    max_period_frac=max_period_frac,
                )
                for _ in range(n_blocks)
            ]
        )

        d_out = out_channels if quantiles is None else out_channels * len(quantiles)

        # Readout: mean + max + last → slightly richer summary
        self.pool_mean = nn.AdaptiveAvgPool1d(1)
        self.pool_max = lambda z: z.max(dim=1, keepdim=True)[0]
        self.pool_last = lambda z: z[:, -1:, :]

        self.horizon_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * d_model),
        )
        self.step_proj = nn.Linear(d_model, d_out)

        self.post_mixer = (
            nn.Linear(out_channels, out_channels)
            if use_channel_mixer and quantiles is None
            else nn.Identity()
        )

        self.final_act = nn.GELU()  # small help for regression stability

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Cin = x.shape
        assert Cin == self.in_channels, (
            f"Expected {self.in_channels} channels, got {Cin}"
        )

        z = self.enc_in(x)

        for blk in self.blocks:
            z = blk(z)

        # richer pooling summary
        z_mean = self.pool_mean(z.permute(0, 2, 1)).squeeze(-1)  # [B, D]
        z_max = self.pool_max(z).squeeze(1)  # [B, D]
        z_last = self.pool_last(z).squeeze(1)  # [B, D]
        z_sum = torch.cat([z_mean, z_max, z_last], dim=-1)  # [B, 3D]

        htem = self.horizon_mlp(z_sum).view(B, self.pred_len, -1)  # [B, H, D]
        y = self.step_proj(htem)  # [B, H, C_out×Q]
        y = self.final_act(y)

        if isinstance(self.post_mixer, nn.Linear):
            y = self.post_mixer(y)

        return y

    def split_quantiles(self, y: torch.Tensor) -> Dict[float, torch.Tensor]:
        if self.quantiles is None:
            raise ValueError("No quantiles configured")
        Q = len(self.quantiles)
        B, H, _ = y.shape
        yq = y.view(B, H, self.out_channels, Q)
        return {q: yq[..., i] for i, q in enumerate(self.quantiles)}
