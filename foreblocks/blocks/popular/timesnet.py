import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.tf.norms import create_norm_layer  # your norm factory
from foreblocks.ui.node_spec import node


# ------------------------------------------------------------
# Period estimation alternatives
# ------------------------------------------------------------
def _period_bounds(
    L: int,
    min_period: int = 2,
    max_period: Optional[int] = None,
    max_period_frac: float = 0.5,
) -> Tuple[int, int]:
    lo = max(min_period, 1)
    hi = min(L - 1, max_period if max_period is not None else int(L * max_period_frac))
    if lo > hi:
        lo, hi = 1, L - 1
    return lo, hi


def _topk_periods_amplitude(
    x: torch.Tensor,
    k: int,
    min_period: int = 2,
    max_period: Optional[int] = None,
    max_period_frac: float = 0.5,
) -> torch.Tensor:
    """
    x: [B, L, C]
    Returns: periods [B, k] (int64), selected from dominant FFT amplitudes.
    """
    B, L, C = x.shape
    if k <= 0:
        return torch.empty(B, 0, dtype=torch.long, device=x.device)

    x0 = x - x.mean(dim=1, keepdim=True)
    Xf = torch.fft.rfft(x0.float(), dim=1)  # [B, L//2+1, C]
    amp = Xf.abs().mean(dim=-1)  # [B, L//2+1], avg over channels
    spec = amp.mean(dim=0)  # [L//2+1], avg over batch

    lo, hi = _period_bounds(
        L, min_period=min_period, max_period=max_period, max_period_frac=max_period_frac
    )

    # Ignore DC and map period bounds [lo, hi] to frequency bounds [f_lo, f_hi].
    if spec.numel() > 0:
        spec = spec.clone()
        spec[0] = 0
    f_lo = max(1, math.ceil(L / hi))
    f_hi = min(L // 2, max(1, int(round(L / lo))))
    if f_lo > f_hi:
        f_lo, f_hi = 1, max(1, L // 2)

    band = spec[f_lo : f_hi + 1]
    if band.numel() == 0:
        periods = torch.full((k,), lo, dtype=torch.long, device=x.device)
    else:
        k_eff = min(k, band.numel())
        top_freq = torch.topk(band, k=k_eff, dim=0, largest=True).indices + f_lo
        periods = torch.div(L + top_freq // 2, top_freq, rounding_mode="floor")
        periods = periods.to(torch.long).clamp(min=lo, max=hi)
        if k_eff < k:
            periods = torch.cat([periods, periods[-1:].expand(k - k_eff)], dim=0)

    return periods.unsqueeze(0).expand(B, -1).contiguous()


def _topk_periods_autocorr(
    x: torch.Tensor,
    k: int,
    min_period: int = 2,
    max_period: Optional[int] = None,
    max_period_frac: float = 0.5,
) -> torch.Tensor:
    """
    x: [B, L, C]
    Returns: periods [B, k] (int64), selected from FFT-based autocorrelation peaks.
    """
    B, L, C = x.shape
    if k <= 0:
        return torch.empty(B, 0, dtype=torch.long, device=x.device)

    lo, hi = _period_bounds(
        L, min_period=min_period, max_period=max_period, max_period_frac=max_period_frac
    )

    x0 = x - x.mean(dim=1, keepdim=True)
    Xf = torch.fft.rfft(x0.float(), dim=1)  # [B, L//2+1, C]
    Sxx = (Xf * torch.conj(Xf)).real.sum(dim=-1)  # [B, L//2+1]
    ac = torch.fft.irfft(Sxx, n=L, dim=1).real  # [B, L]
    band = ac[:, lo : hi + 1]

    if band.numel() == 0:
        return torch.full((B, k), lo, dtype=torch.long, device=x.device)

    k_eff = min(k, band.size(1))
    idx = torch.topk(band, k=k_eff, dim=1, largest=True).indices + lo  # [B, k_eff]
    idx = idx.to(torch.long).clamp(min=lo, max=hi)
    if k_eff < k:
        idx = torch.cat([idx, idx[:, -1:].expand(B, k - k_eff)], dim=1)
    return idx


def _topk_periods(
    x: torch.Tensor,
    k: int,
    min_period: int = 2,
    max_period: Optional[int] = None,
    max_period_frac: float = 0.5,
    method: str = "amplitude",
) -> torch.Tensor:
    if method == "amplitude":
        return _topk_periods_amplitude(
            x,
            k,
            min_period=min_period,
            max_period=max_period,
            max_period_frac=max_period_frac,
        )
    if method in ("autocorr", "autocorrelation"):
        return _topk_periods_autocorr(
            x,
            k,
            min_period=min_period,
            max_period=max_period,
            max_period_frac=max_period_frac,
        )
    if method in ("hybrid", "ensemble", "amp_autocorr"):
        # Keep total selection budget at k while mixing both selectors.
        # Duplicate periods naturally receive larger weights later via multiplicity counts.
        if k <= 0:
            return torch.empty(x.size(0), 0, dtype=torch.long, device=x.device)

        k_amp = (k + 1) // 2
        k_ac = k - k_amp

        p_amp = _topk_periods_amplitude(
            x,
            k_amp,
            min_period=min_period,
            max_period=max_period,
            max_period_frac=max_period_frac,
        )
        if k_ac <= 0:
            return p_amp

        p_ac = _topk_periods_autocorr(
            x,
            k_ac,
            min_period=min_period,
            max_period=max_period,
            max_period_frac=max_period_frac,
        )
        return torch.cat([p_amp, p_ac], dim=1).contiguous()
    raise ValueError(
        "Unsupported period method "
        f"'{method}'. Use 'amplitude', 'autocorr', or 'hybrid'."
    )


# ------------------------------------------------------------
# Inception-style 2D conv block over (patches × period)
# ------------------------------------------------------------
class Inception2D(nn.Module):
    class _ChannelFirstNorm2D(nn.Module):
        """Apply channel-last norm layers to [B, C, H, W] tensors."""

        def __init__(self, norm_type: str, channels: int, eps: float):
            super().__init__()
            self.norm = create_norm_layer(norm_type, channels, eps=eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            return x.permute(0, 3, 1, 2).contiguous()

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ks: Tuple[int, ...] = (3, 5, 7),
        expand: int = 2,
        dropout: float = 0.0,
        norm_type: str = "rms",
        layer_norm_eps: float = 1e-5,
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
            nn.Conv2d(hidden * len(ks), out_ch, 1, bias=True),
            Inception2D._ChannelFirstNorm2D(
                norm_type=norm_type,
                channels=out_ch,
                eps=layer_norm_eps,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
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
        pad_mode: str = "right",
        period_method: str = "amplitude",
    ):
        super().__init__()
        self.k = k_periods
        self.max_period_frac = max_period_frac
        self.pad_mode = pad_mode
        self.period_method = period_method

        self.norm = create_norm_layer(norm_type, d_model, eps=layer_norm_eps)
        self.proj_in = nn.Linear(d_model, d_model)

        if use_glu_gate:
            self.glu_proj = nn.Linear(d_model, d_model * 2)
            self.glu = nn.GLU(dim=-1)
        else:
            self.glu_proj = self.glu = None

        out_inner = d_model if hidden is None else hidden
        self.inception = Inception2D(
            d_model,
            out_inner,
            ks=ks,
            expand=expand,
            dropout=dropout,
            norm_type=norm_type,
            layer_norm_eps=layer_norm_eps,
        )
        self.proj_out = nn.Linear(out_inner, d_model)
        self.drop = nn.Dropout(dropout)

    def _pad_to_period(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return x

        if self.pad_mode == "right":
            return F.pad(x, (0, 0, 0, pad))

        x_t = x.transpose(1, 2)  # [B, C, L]
        if self.pad_mode == "symmetric":
            left = pad // 2
            right = pad - left
            mode = "reflect" if x_t.size(-1) > 1 else "replicate"
            x_t = F.pad(x_t, (left, right), mode=mode)
        elif self.pad_mode == "circular":
            x_t = F.pad(x_t, (0, pad), mode="circular")
        else:
            raise ValueError(
                f"Unsupported pad_mode='{self.pad_mode}'. Use 'right', 'symmetric', or 'circular'."
            )
        return x_t.transpose(1, 2).contiguous()

    def _fold(self, x: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
        B, L, C = x.shape
        Np = (L + p - 1) // p
        pad = Np * p - L
        if pad > 0:
            x = self._pad_to_period(x, pad)
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
        if L < 2:
            return res

        h = self.norm(x)

        if self.glu is not None:
            h = self.glu(self.glu_proj(h))  # → [B,L,d_model]
        h = self.proj_in(h)

        periods = _topk_periods(
            h,
            k=self.k,
            min_period=2,
            max_period_frac=self.max_period_frac,
            method=self.period_method,
        )  # [B, k]

        n_periods = periods.size(1)
        if n_periods == 0:
            agg = torch.zeros_like(h)
        else:
            # Build unique (sample, period) pairs with multiplicities in one pass.
            # This avoids O(U * B * k) repeated equality scans when many periods are active.
            p_all = periods.clamp(min=2, max=max(2, L))  # [B, k]
            denom = float(n_periods)
            agg = torch.zeros_like(h, dtype=torch.float32)

            sample_ids = (
                torch.arange(B, device=h.device)
                .unsqueeze(1)
                .expand(B, n_periods)
                .reshape(-1)
            )  # [B*k]
            period_flat = p_all.reshape(-1)  # [B*k]

            key_stride = int(L + 1)  # periods are in [2, L]
            pair_key = sample_ids * key_stride + period_flat

            uniq_key, mult = torch.unique(pair_key, sorted=False, return_counts=True)
            idx_unique = torch.div(uniq_key, key_stride, rounding_mode="floor").to(
                torch.long
            )
            period_unique = (uniq_key % key_stride).to(torch.long)

            # Sort once by period so each contiguous bucket can be processed with one fold+conv.
            order = torch.argsort(period_unique)
            idx_unique = idx_unique.index_select(0, order)
            period_unique = period_unique.index_select(0, order)
            mult = mult.index_select(0, order)

            unique_p, group_sizes = torch.unique_consecutive(
                period_unique, return_counts=True
            )

            start = 0
            for p_t, gsize in zip(unique_p, group_sizes):
                end = start + int(gsize.item())
                idx = idx_unique[start:end]
                mult_g = mult[start:end]
                p = int(p_t.item())

                Xp, pad = self._fold(h.index_select(0, idx), p)  # [Bg, C, Np, p]
                Yp = self.inception(Xp)
                y = self._unfold(Yp, pad, L).to(agg.dtype)  # [Bg, L, C]

                w = mult_g.to(agg.dtype).view(-1, 1, 1) / denom
                agg.index_add_(0, idx, y * w)
                start = end

            agg = agg.to(h.dtype)

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
        pad_mode: str = "right",
        period_method: str = "amplitude",
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
                    pad_mode=pad_mode,
                    period_method=period_method,
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

        self.final_act = nn.Identity()

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
