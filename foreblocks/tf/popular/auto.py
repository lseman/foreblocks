import math

# autoformer_head_custom.py
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_aux import (
    create_norm_layer,  # uses your fast Layer/RMS/Temporal/RevIN, etc.
)


# =========================================================
# 1) Series decomposition with moving average (same spirit
#    as FED/Autoformer; shared here for isolation).
# =========================================================
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        pad = (kernel_size - 1) // 2
        self.avg = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=pad, bias=False)
        nn.init.constant_(self.avg.weight, 1.0 / kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, C]  ->  seasonal, trend: [B, T, C]
        """
        B, T, C = x.shape
        u = x.permute(0, 2, 1).contiguous().view(B * C, 1, T)  # [B*C,1,T]
        t = self.avg(u).view(B, C, T).permute(0, 2, 1).contiguous()
        s = x - t
        return s, t

# =========================================================
# 2) Auto-Correlation (multi-head) per Autoformer
#    Efficient FFT-based correlation + top-k delays.
# =========================================================
def _topk_delays(corr: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    corr: [B, Hh, Lq, Lk] correlation map (time-delay domain).
    Returns:
      idx: [B, Hh, K]   (best delays)
      w:   [B, Hh, K]   (normalized weights)
    """
    B, Hh, Lq, Lk = corr.shape
    # mean over query length to get global importance per delay
    score = corr.mean(dim=2)  # [B,Hh,Lk]
    K = min(k, Lk)
    vals, idx = torch.topk(score, k=K, dim=-1, largest=True)
    w = (vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)).detach()
    return idx, w

class AutoCorrelation(nn.Module):
    """
    Auto-Correlation attention:
      - compute correlation via FFT: Corr(q,k)(τ) = IFFT(FFT(q) * conj(FFT(k)))
      - aggregate V shifted by top-k delays weighted by correlation
    Shapes use standard MH projections with head dim dh = d_model // n_heads.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, k_delays: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.k_delays = k_delays
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _fft_corr(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q,k: [B, Hh, L, dh] -> corr over delays: [B, Hh, L, L]
        """
        B, Hh, L, Dh = q.shape
        # zero-mean along time to stabilize
        qz = q - q.mean(dim=2, keepdim=True)
        kz = k - k.mean(dim=2, keepdim=True)
        Qf = torch.fft.rfft(qz, dim=2)             # [B,Hh,F,dh]
        Kf = torch.fft.rfft(kz, dim=2)
        # IFFT of elementwise product with conjugate
        S = Qf * torch.conj(Kf)                    # [B,Hh,F,dh]
        corr = torch.fft.irfft(S, n=L, dim=2).real # [B,Hh,L,dh] correlation over lags 0..L-1 per feature
        # aggregate over feature dim to get scalar correlation map per head
        corr = corr.mean(dim=-1)                   # [B,Hh,L]
        # Build [B,Hh,Lq,Lk] by broadcasting over queries (use same L)
        return corr[:, :, None, :].expand(B, Hh, L, L)

    @staticmethod
    def _shift(v: torch.Tensor, delay: int) -> torch.Tensor:
        # circular shift to align with correlation delay (as in Autoformer)
        return torch.roll(v, shifts=-delay, dims=2)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        x_q, x_kv: [B, L, D]
        returns:   [B, L, D]
        """
        B, L, D = x_q.shape
        Hh, Dh = self.n_heads, self.dh

        q = self.q_proj(x_q).view(B, L, Hh, Dh).permute(0, 2, 1, 3)  # [B,Hh,L,Dh]
        k = self.k_proj(x_kv).view(B, L, Hh, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).view(B, L, Hh, Dh).permute(0, 2, 1, 3)

        # FFT-based correlation over delays
        corr = self._fft_corr(q, k)                  # [B,Hh,L,L]
        idx, w = _topk_delays(corr, self.k_delays)   # [B,Hh,K], [B,Hh,K]

        # time-delay aggregation: sum_k w_k * shift(v, delay_k)
        out = 0.0
        for j in range(idx.size(-1)):
            delay = idx[:, :, j]                     # [B,Hh]
            # shift per (B,Hh): vectorized by looping heads; light loop (K up to ~8)
            v_shift = []
            for h in range(Hh):
                v_shift.append(self._shift(v[:, h:h+1, :, :], delay[:, h].view(B, 1, 1, 1).item()))
            v_shift = torch.cat(v_shift, dim=1)      # [B,Hh,L,Dh]
            out = out + v_shift * w[:, :, j].view(B, Hh, 1, 1)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.o_proj(self.drop(out))

# =========================================================
# 3) Autoformer encoder/decoder layers (with decomposition)
# =========================================================
class AutoformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int = 512,
        dropout: float = 0.1,
        k_delays: int = 8,
        norm_type: str = "rms",
        eps: float = 1e-5,
        activation: str = "gelu",
        kernel_size: int = 25,
    ):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.attn = AutoCorrelation(d_model, n_heads, dropout, k_delays)
        self.drop = nn.Dropout(dropout)
        self.norm1 = create_norm_layer(norm_type, d_model, eps=eps)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = create_norm_layer(norm_type, d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D]
        returns: seasonal_out, trend_residual
        """
        s, t = self.decomp(x)                 # [B,L,D] each
        s = s + self.drop(self.attn(self.norm1(s), self.norm1(s)))
        s = s + self.drop(self.ff(self.norm2(s)))
        return s, t

class AutoformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int = 512,
        dropout: float = 0.1,
        k_delays: int = 8,
        norm_type: str = "rms",
        eps: float = 1e-5,
        activation: str = "gelu",
        kernel_size: int = 25,
    ):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.self_attn = AutoCorrelation(d_model, n_heads, dropout, k_delays)
        self.cross_attn = AutoCorrelation(d_model, n_heads, dropout, k_delays)
        self.drop = nn.Dropout(dropout)
        self.norm1 = create_norm_layer(norm_type, d_model, eps=eps)
        self.norm2 = create_norm_layer(norm_type, d_model, eps=eps)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm3 = create_norm_layer(norm_type, d_model, eps=eps)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:   seasonal input [B, Ld, D]
        mem: encoder seasonal memory [B, Le, D]
        returns: seasonal_out, trend_residual
        """
        s, t = self.decomp(x)
        s = s + self.drop(self.self_attn(self.norm1(s), self.norm1(s)))
        s = s + self.drop(self.cross_attn(self.norm2(s), self.norm2(mem)))
        s = s + self.drop(self.ff(self.norm3(s)))
        return s, t

# =========================================================
# 4) Full Autoformer Head
# =========================================================
class AutoformerHeadCustom(nn.Module):
    """
    Autoformer forecasting head.

    Inputs:
      x: [B, L_in, C_in]

    Hyperparams of interest:
      label_len: length of decoder warm-up (uses last part of encoder input)
      pred_len:  forecast horizon

    Output:
      y: [B, pred_len, C_out]  (or C_out * Q if quantiles)
    """
    def __init__(
        self,
        pred_len: int,
        label_len: int = 48,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers_enc: int = 2,
        n_layers_dec: int = 1,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        k_delays: int = 8,
        kernel_size: int = 25,
        norm_type: str = "rms",     # try "temporal" or "revin" too
        eps: float = 1e-5,
        activation: str = "gelu",
        quantiles: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.quantiles = quantiles

        # projections
        self.enc_in = nn.Linear(in_channels, d_model)
        self.dec_in = nn.Linear(in_channels, d_model)

        # encoder/decoder stacks (seasonal path)
        self.encoder = nn.ModuleList([
            AutoformerEncoderLayer(
                d_model=d_model, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout,
                k_delays=k_delays, norm_type=norm_type, eps=eps, activation=activation, kernel_size=kernel_size
            ) for _ in range(n_layers_enc)
        ])
        self.decoder = nn.ModuleList([
            AutoformerDecoderLayer(
                d_model=d_model, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout,
                k_delays=k_delays, norm_type=norm_type, eps=eps, activation=activation, kernel_size=kernel_size
            ) for _ in range(n_layers_dec)
        ])

        # trend linear head (applied to trend decoder input)
        d_out = out_channels if quantiles is None else out_channels * len(quantiles)
        self.trend_proj = nn.Linear(d_model, d_out)
        self.seasonal_proj = nn.Linear(d_model, d_out)

        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    # ---------- helper: build decoder inputs per Autoformer ----------
    def _build_decoder_inputs(self, s_enc: torch.Tensor, t_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        s_enc, t_enc: [B, L_in, D]
        Returns:
          s_dec: seasonal input  = last label_len seasonal + zeros(pred_len)
          t_dec: trend input     = last label_len trend + repeat(last trend) over pred_len
        """
        B, Lin, D = s_enc.shape
        Lc = min(self.label_len, Lin)
        s_label = s_enc[:, -Lc:, :]
        t_label = t_enc[:, -Lc:, :]

        s_pred0 = s_enc.new_zeros(B, self.pred_len, D)
        s_dec = torch.cat([s_label, s_pred0], dim=1)   # [B, Lc+H, D]

        last_tr = t_enc[:, -1:, :].expand(B, self.pred_len, D)
        t_dec = torch.cat([t_label, last_tr], dim=1)   # [B, Lc+H, D]
        return s_dec, t_dec

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L_in, C_in] -> y: [B, H, C_out]
        """
        if x.dim() != 3 or x.size(-1) != self.in_channels:
            raise ValueError(f"Expected x [B, L, C_in={self.in_channels}], got {tuple(x.shape)}")
        B, L, Cin = x.shape

        z = self.enc_in(x)                 # [B,L,D]

        # initial decomp
        s, t = SeriesDecomp(kernel_size=3)(z)  # light pre-smoothing before layers

        # encoder on seasonal path; pass trend along residually
        for layer in self.encoder:
            s, t_res = layer(s)
            t = t + t_res

        mem = s                              # encoder seasonal memory

        # build decoder inputs per Autoformer
        s_dec, t_dec = self._build_decoder_inputs(s, t)  # [B, Lc+H, D] each

        # run decoder layers
        s_cur, t_cur = s_dec, t_dec
        for layer in self.decoder:
            s_cur, t_res = layer(s_cur, mem)
            t_cur = t_cur + t_res

        # take only the prediction tail
        s_out = s_cur[:, -self.pred_len:, :]
        t_out = t_cur[:, -self.pred_len:, :]

        y = self.seasonal_proj(s_out) + self.trend_proj(t_out)  # [B,H,C_out*(Q?)]

        return y

    # Convenience splitter for quantile outputs
    def split_quantiles(self, y: torch.Tensor):
        assert self.quantiles is not None
        Q = len(self.quantiles)
        B, H, _ = y.shape
        y = y.view(B, H, self.out_channels, Q)
        return {q: y[..., i] for i, q in enumerate(self.quantiles)}
