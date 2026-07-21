"""foreblocks.modules.attention.implementations.frequency_att.

Frequency-domain attention operations via Fourier transforms.

Implements FourierCrossAttention (FEDformer) — transforms queries, keys, and
values via rFFT, applies attention over a small set of Fourier modes, then
inverts back to the time domain. Also provides FourierBlock and
FourierModeSelector for standalone spectral mixing. Use when frequency-domain
mixing is needed as a building block.

Core API:
- FrequencyAttention: FourierCrossAttention from FEDformer
- FourierBlock: FEDformer-style frequency-domain mixing block
- FourierModeSelector: frequency mode selection (topk or fixed)

"""

from typing import Literal

import torch
import torch.nn as nn

from foreblocks.layers.norms import create_norm_layer


class FourierModeSelector(nn.Module):
    def __init__(
        self,
        modes: int = 32,
        mode_select: Literal["topk", "fixed"] = "topk",
        skip_dc: bool = True,
    ):
        super().__init__()
        if mode_select not in ("topk", "fixed"):
            raise ValueError("mode_select must be 'topk' or 'fixed'")
        self.modes = int(modes)
        self.mode_select = mode_select
        self.skip_dc = skip_dc

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.dim() != 3:
            raise ValueError("spectrum must have shape [B, F, C]")

        B, F, _ = spectrum.shape
        start = 1 if self.skip_dc and F > 1 else 0
        available = F - start
        if available <= 0:
            return torch.zeros(B, 0, dtype=torch.long, device=spectrum.device)

        K = min(self.modes, available)
        if self.mode_select == "fixed":
            return torch.arange(start, start + K, device=spectrum.device).expand(B, K)

        magnitude = spectrum[:, start:, :].abs().pow(2).sum(dim=-1)
        return magnitude.topk(K, dim=-1, largest=True).indices + start


class FourierBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        modes: int = 32,
        mode_select: Literal["topk", "fixed"] = "topk",
        dropout: float = 0.0,
        custom_norm: str = "rms",
        eps: float = 1e-5,
        skip_dc: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.modes = int(modes)
        self.selector = FourierModeSelector(
            modes=modes,
            mode_select=mode_select,
            skip_dc=skip_dc,
        )
        self.weight = nn.Parameter(torch.randn(max(1, self.modes), d_model, 2) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = create_norm_layer(custom_norm, d_model, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape [B, T, C]")
        B, T, C = x.shape
        if self.d_model != C:
            raise ValueError(f"expected last dim {self.d_model}, got {C}")

        spectrum = torch.fft.rfft(x, dim=1)  # [B, F, C]
        idx = self.selector(spectrum)  # [B, K]
        K = idx.size(1)
        if K == 0:
            return self.norm(self.dropout(torch.zeros_like(x)))

        gather_idx = idx[:, :, None].expand(B, K, C)
        selected = spectrum.gather(dim=1, index=gather_idx)

        weight = torch.view_as_complex(self.weight[:K].contiguous())
        mixed = selected * weight.unsqueeze(0)

        out_spectrum = torch.zeros_like(spectrum)
        out_spectrum.scatter_(dim=1, index=gather_idx, src=mixed)

        y = torch.fft.irfft(out_spectrum, n=T, dim=1)
        return self.norm(self.dropout(y))


class FrequencyAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        modes: int = 32,
        activation: str = "tanh",
        mode_select: str = "low",
    ):
        super().__init__()
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError("n_heads must be positive and divide d_model")
        if activation not in ("tanh", "softmax"):
            raise ValueError("activation must be 'tanh' or 'softmax'")
        if mode_select not in ("low", "fixed", "random"):
            raise ValueError("mode_select must be 'low', 'fixed', or 'random'")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = int(modes)
        self.activation = activation
        self.mode_select = mode_select

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable complex spectral weights: [H, E_in, E_out, modes].
        # Scaled as in FEDformer: 1 / (E_in * E_out).
        scale = 1.0 / (self.head_dim * self.head_dim)
        self.weights1 = nn.Parameter(
            scale
            * torch.rand(
                n_heads,
                self.head_dim,
                self.head_dim,
                max(1, self.modes),
                2,
            )
        )  # last dim packs (real, imag); viewed as complex at use time

    def _select_modes(self, n_freq: int, device: torch.device) -> torch.Tensor:
        modes = min(self.modes, n_freq)
        if self.mode_select == "random" and modes < n_freq:
            idx = torch.randperm(n_freq, device=device)[:modes]
            return torch.sort(idx).values
        return torch.arange(modes, device=device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if key is None:
            key = query
        if value is None:
            value = key

        B, L_q, _ = query.shape
        L_k = key.size(1)
        H, E = self.n_heads, self.head_dim

        # Project → [B, H, E, L] so the FFT runs over the time axis (dim=-1),
        # matching FEDformer's layout.
        q = self.q_proj(query).view(B, L_q, H, E).permute(0, 2, 3, 1)
        k = self.k_proj(key).view(B, L_k, H, E).permute(0, 2, 3, 1)
        v = self.v_proj(value).view(B, L_k, H, E).permute(0, 2, 3, 1)

        qf = torch.fft.rfft(q, dim=-1)  # [B, H, E, Fq]
        kf = torch.fft.rfft(k, dim=-1)  # [B, H, E, Fk]
        vf = torch.fft.rfft(v, dim=-1)  # [B, H, E, Fk]

        Fq = qf.size(-1)
        q_idx = self._select_modes(Fq, qf.device)
        kv_idx = self._select_modes(kf.size(-1), kf.device)
        m = min(q_idx.numel(), kv_idx.numel())
        q_idx, kv_idx = q_idx[:m], kv_idx[:m]

        xq = qf[:, :, :, q_idx]  # [B, H, E, m]
        xk = kf[:, :, :, kv_idx]  # [B, H, E, m]
        xv = vf[:, :, :, kv_idx]  # [B, H, E, m]

        # Frequency-domain attention between modes (contract over feature E).
        xqk = torch.einsum("bhex,bhey->bhxy", xq, xk)  # [B, H, m_q, m_k]
        if self.activation == "tanh":
            xqk = torch.complex(xqk.real.tanh(), xqk.imag.tanh())
        else:  # softmax over the key-mode axis, using magnitudes
            xqk = torch.softmax(xqk.abs(), dim=-1).to(xqk.dtype)

        # Mix with value spectrum, then apply learnable complex weights.
        xqkv = torch.einsum("bhxy,bhey->bhex", xqk, xv)  # [B, H, E, m_q]
        w = torch.view_as_complex(self.weights1.contiguous())[:, :, :, :m]
        out_modes = torch.einsum("bhex,heox->bhox", xqkv, w)  # [B, H, E, m_q]

        # Scatter back into the full query spectrum and invert.
        out_ft = torch.zeros(B, H, E, Fq, dtype=qf.dtype, device=qf.device)
        out_ft[:, :, :, q_idx] = out_modes
        out_ft = out_ft / (E * E)  # FEDformer normalisation (in/out channels)

        out = torch.fft.irfft(out_ft, n=L_q, dim=-1)  # [B, H, E, L_q]
        out = out.permute(0, 3, 1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.dropout(self.out_proj(out))
        return out, None
