"""foreblocks.modules.attention.modules.dwt_att.

Spectral attention operating in the wavelet-coefficient domain.

Applies a single-level orthogonal Haar discrete wavelet transform to split
the sequence into approximation (low-pass) and detail (high-pass) bands, then
computes attention over the leading wavelet coefficients before inverting back
to the time domain. Use when you want frequency-domain attention without FFT.

Core API:
- DWTAttention: Haar wavelet-domain attention with per-coefficient spectral weights

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        modes: int = 32,
    ):
        super().__init__()
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError("n_heads must be positive and divide d_model")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = int(modes)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Per-head, per-coefficient learnable spectral weight.
        self.wavelet_weight = nn.Parameter(
            torch.randn(n_heads, max(1, self.modes), self.head_dim) * 0.02
        )

    def _haar_dwt(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(2)
        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode="reflect")
        approx = (x[:, :, ::2, :] + x[:, :, 1::2, :]) / 2
        detail = (x[:, :, ::2, :] - x[:, :, 1::2, :]) / 2
        return torch.cat([approx, detail], dim=2)

    def _haar_idwt(self, coeffs: torch.Tensor, target_len: int) -> torch.Tensor:
        half_L = coeffs.size(2) // 2
        approx = coeffs[:, :, :half_L, :]
        detail = coeffs[:, :, half_L:, :]
        even = approx + detail
        odd = approx - detail
        result = torch.zeros(
            coeffs.size(0),
            coeffs.size(1),
            half_L * 2,
            coeffs.size(3),
            device=coeffs.device,
            dtype=coeffs.dtype,
        )
        result[:, :, ::2, :] = even
        result[:, :, 1::2, :] = odd
        return result[:, :, :target_len, :]

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
        H, E = self.n_heads, self.head_dim

        q = self.q_proj(query).view(B, L_q, H, E).transpose(1, 2)
        k = self.k_proj(key).view(B, key.size(1), H, E).transpose(1, 2)
        v = self.v_proj(value).view(B, value.size(1), H, E).transpose(1, 2)

        q_dwt = self._haar_dwt(q)  # [B, H, Lc, E]
        k_dwt = self._haar_dwt(k)
        v_dwt = self._haar_dwt(v)

        m = min(self.modes, q_dwt.size(2), k_dwt.size(2))
        q_m = q_dwt[:, :, :m, :]  # [B, H, m, E]
        k_m = k_dwt[:, :, :m, :]
        v_m = v_dwt[:, :, :m, :]

        # Attention between wavelet coefficients (contract feature dim E).
        scores = torch.einsum("bhie,bhje->bhij", q_m, k_m) * self.scale
        attn = torch.softmax(scores, dim=-1)  # over key coefficients
        mixed = torch.einsum("bhij,bhje->bhie", attn, v_m)  # [B, H, m, E]
        mixed = mixed * self.wavelet_weight[:, :m, :].unsqueeze(0)

        out_dwt = torch.zeros_like(q_dwt)
        out_dwt[:, :, :m, :] = mixed

        out_time = self._haar_idwt(out_dwt, L_q)
        out = out_time.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.dropout(self.out_proj(out))

        attn_out = attn if need_weights else None
        return out, attn_out
