import torch
import torch.nn as nn


class FrequencyAttention(nn.Module):
    """Frequency Enhanced Attention (FEA) in the frequency domain.

    Faithful re-implementation of the ``FourierCrossAttention`` block from
    FEDformer:

        Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for
        Long-term Series Forecasting", ICML 2022.
        Paper:  https://arxiv.org/abs/2201.12740
        Code:   https://github.com/MAZiqing/FEDformer
                (``layers/FourierCorrelation.py`` → ``FourierCrossAttention``)

    Algorithm (per head, feature dim E = head_dim, sequence length L)
    ----------------------------------------------------------------
    1. rFFT q, k, v along the time axis.
    2. Select ``modes`` Fourier components (lowest frequencies, or a random
       subset) from each of q and k.
    3. Frequency-domain attention between modes (contracting the feature dim):
           xqk[b,h,x,y] = Σ_e xq[b,h,e,x] · xk[b,h,e,y]
       followed by ``tanh`` (default) or ``softmax(|xqk|)`` activation.
    4. Mix with the value spectrum:
           xqkv[b,h,e,x] = Σ_y xqk[b,h,x,y] · xv[b,h,e,y]
    5. Per-head learnable complex spectral weights:
           out[b,h,o,x] = Σ_e xqkv[b,h,e,x] · W[h,e,o,x]
    6. Scatter back into the full spectrum and irFFT to time domain.

    Notes
    -----
    * The official code multiplies the attention by ``xk`` again (not ``xv``)
      in step 4 — that only matches the paper when q, k, v share a length.
      Here we use ``xv`` so cross-attention (L_q ≠ L_k) is well defined; for
      self-attention the two are identical.
    * ``modes`` is clamped to the available frequency count of each tensor.
    """

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
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        if activation not in ("tanh", "softmax"):
            raise ValueError("activation must be 'tanh' or 'softmax'")
        if mode_select not in ("low", "random"):
            raise ValueError("mode_select must be 'low' or 'random'")

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
        """Return sorted indices of the Fourier modes to keep (≤ n_freq)."""
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
