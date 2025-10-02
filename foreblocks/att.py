"""
Multi-Method Attention Layer with Optimized Backends (revised)

Adds: masks, safer hidden extraction, training-aware dropout, residual+LayerNorm (optional),
consistent scaling, robust xformers/flash calls, and a workable autocorr path for single-query use.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Optional deps ----------
try:
    import xformers
    import xformers.ops as xops
    HAS_XFORMERS = True
except Exception:
    HAS_XFORMERS = False

try:
    # flash-attn v2 API; we wrap to avoid hard dependency
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
    HAS_FLASH_ATTN = True
except Exception:
    _flash_attn_func = None
    HAS_FLASH_ATTN = False


class AttentionLayer(nn.Module):
    """
    Flexible attention supporting:
      methods: 'dot', 'mha', 'prob', 'temporal', 'multiscale', 'autocorr'
      backends: 'torch', 'xformers', 'flash'
    """

    _VALID_METHODS = frozenset({"dot", "mha", "prob", "temporal", "multiscale", "autocorr"})
    _VALID_BACKENDS = frozenset({"torch", "xformers", "flash"})
    _MHA_METHODS = frozenset({"mha", "multiscale", "autocorr"})

    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: Optional[int] = None,
        method: str = "dot",
        attention_backend: str = "torch",
        nhead: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 8,
        num_scales: int = 3,
        use_residual_ln: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        backend = attention_backend.lower()
        self._validate_inputs(method, backend, decoder_hidden_size, nhead)

        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size or decoder_hidden_size
        self.method = method
        self.backend = backend
        self.nhead = nhead
        self.head_dim = decoder_hidden_size // nhead
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)
        self.use_residual_ln = use_residual_ln

        self.dropout = nn.Dropout(dropout)
        self._build_projections()
        self._build_method_specific_layers(time_embed_dim, num_scales)

        if use_residual_ln:
            self.post_ln = nn.LayerNorm(self.decoder_hidden_size)

        if verbose:
            self._print_initialization_info()

    # ----- setup -----

    def _validate_inputs(self, method: str, backend: str, hidden_size: int, nhead: int):
        if method not in self._VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from {self._VALID_METHODS}")
        if backend not in self._VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Choose from {self._VALID_BACKENDS}")
        if hidden_size % nhead != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by nhead {nhead}")
        if backend == "flash" and not HAS_FLASH_ATTN:
            raise ImportError("FlashAttention not available but requested")
        if backend == "xformers" and not HAS_XFORMERS:
            raise ImportError("xFormers not available but requested")

    def _build_projections(self):
        # combine decoder hidden + context
        self.combined_layer = nn.Linear(self.decoder_hidden_size * 2, self.decoder_hidden_size)

        # align encoder dim to decoder dim if needed
        self.encoder_projection = (
            nn.Identity()
            if self.decoder_hidden_size == self.encoder_hidden_size
            else nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=False)
        )

        # qkv only when we truly need multihead projections or optimized backend for dot
        if self.method in self._MHA_METHODS or (self.method == "dot" and self.backend != "torch"):
            self.q_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.k_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.v_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)

        # flash needs a final concat projection when returning [B, 1, H*D] to [B, 1, D_total]
        if self.backend == "flash":
            self.context_proj = nn.Linear(self.nhead * self.head_dim, self.decoder_hidden_size, bias=False)

    def _build_method_specific_layers(self, time_embed_dim: int, num_scales: int):
        if self.method == "temporal":
            self.time_bias = nn.Linear(time_embed_dim, self.decoder_hidden_size, bias=False)

        elif self.method == "multiscale":
            self.scale_projections = nn.ModuleList([
                nn.ModuleDict({
                    "k": nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False),
                    "v": nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False),
                })
                for _ in range(num_scales)
            ])
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.dilations = tuple(2 ** i for i in range(num_scales))
            self.scale_out_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)

    def _print_initialization_info(self):
        print(f"[Attention] Method: {self.method}, Backend: {self.backend}")
        if HAS_XFORMERS:
            print(f"[Attention] xFormers: {getattr(xformers, '__version__', 'unknown')}")
        if HAS_FLASH_ATTN:
            print(f"[Attention] FlashAttention available")

    # ----- helpers -----

    @staticmethod
    def _last_hidden(dec_h: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Accepts GRU h_n [L,B,H] or LSTM (h_n, c_n). Returns [B,H].
        Takes last layer; if bidirectional was used in the decoder, the caller should ensure H matches.
        """
        if isinstance(dec_h, tuple):
            dec_h = dec_h[0]
        # dec_h: [L, B, H] -> last layer
        return dec_h[-1]

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, nH, T, Hd]
        B, T, D = x.shape
        return x.view(B, T, self.nhead, self.head_dim).transpose(1, 2).contiguous()

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, nH, T, Hd] -> [B, T, D]
        B, Hh, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, Hh * Hd)

    def _sdpa(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """
        Unified scaled-dot product attention call (torch backend).
        q,k,v: [B, nH, T, Hd]
        attn_mask: [Tq, Tk] or broadcastable
        key_padding_mask: [B, Tk] (True for masked positions)
        """
        drop = self.dropout.p if self.training else 0.0
        # torch SDPA expects [*, L, E]; merge batch&heads
        B, Hh, Tq, Hd = q.shape
        _, _, Tk, _ = k.shape

        q_ = q.reshape(B * Hh, Tq, Hd)
        k_ = k.reshape(B * Hh, Tk, Hd)
        v_ = v.reshape(B * Hh, Tk, Hd)

        # build a merged mask if key_padding_mask provided
        attn_mask_ = None
        if attn_mask is not None:
            attn_mask_ = attn_mask  # [Tq, Tk] or [B*Hh, Tq, Tk]

        if key_padding_mask is not None:
            # convert to additive mask of shape [B*Hh, 1, Tk]
            pad = key_padding_mask.unsqueeze(1).expand(B, Hh, Tk).reshape(B * Hh, 1, Tk)
        else:
            pad = None

        out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask_, dropout_p=drop, is_causal=False, key_padding_mask=pad)
        return out.reshape(B, Hh, Tq, Hd)

    def _flash(self, q, k, v):
        """
        FlashAttention wrapper. Expects [B, nH, T, Hd] -> returns [B, T, nH*Hd]
        """
        assert HAS_FLASH_ATTN and _flash_attn_func is not None
        # flash-attn expects float16/bfloat16 and contiguous
        dtype = torch.float16 if q.dtype == torch.float16 else torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float16
        q, k, v = (x.contiguous().to(dtype) for x in (q, k, v))
        drop = self.dropout.p if self.training else 0.0
        # flash_attn works with [B, T, nH, Hd]
        qf = q.transpose(1, 2)  # [B,T,nH,Hd]
        kf = k.transpose(1, 2)
        vf = v.transpose(1, 2)
        out = _flash_attn_func(qf, kf, vf, dropout_p=drop, softmax_scale=None, causal=False)  # [B,T,nH,Hd]
        out = out.to(torch.float32).contiguous().view(q.size(0), -1, self.nhead * self.head_dim)  # [B,T,D]
        return out

    def _xformers(self, q, k, v):
        """
        xFormers memory-efficient attention. Expects [B, nH, T, Hd].
        """
        qx = q.transpose(1, 2)  # [B,T,nH,Hd]
        kx = k.transpose(1, 2)
        vx = v.transpose(1, 2)
        out = xops.memory_efficient_attention(qx, kx, vx, p=self.dropout.p if self.training else 0.0)  # [B,T,nH,Hd]
        return out.transpose(1, 2).contiguous()  # [B,nH,T,Hd]

    # ----- attention kernels -----

    def _compute_attn_weights(self, query_1t: torch.Tensor, enc: torch.Tensor, key_padding_mask=None):
        """
        query_1t: [B,1,D]; enc: [B,T,D]
        returns softmax over T: [B,T]
        """
        scores = torch.bmm(enc, query_1t.transpose(1, 2)).squeeze(2) * (1.0 / math.sqrt(enc.size(-1)))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        return F.softmax(scores, dim=1)

    def _dot_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        dec_h: [B,D]; enc: [B,T,D]
        """
        q1 = dec_h.unsqueeze(1)  # [B,1,D]

        # backend path if we have qkv projections
        if hasattr(self, "q_proj"):
            q = self.q_proj(q1)
            k = self.k_proj(enc)
            v = self.v_proj(enc)
            q, k, v = (self._split_heads(x) for x in (q, k, v))  # [B,nH,T,Dh] (T=1 for q)

            if self.backend == "flash":
                out = self._flash(q, k, v)  # [B,Tq=1, D]
                ctx = self.context_proj(out)[:, 0]  # [B,D]
            elif self.backend == "xformers":
                out = self._xformers(q, k, v)  # [B,nH,1,Hd]
                ctx = self._combine_heads(out)[:, 0]  # [B,D]
            else:
                out = self._sdpa(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)  # [B,nH,1,Hd]
                ctx = self._combine_heads(out)[:, 0]  # [B,D]

            # weights (approximate, for logging/compat)
            with torch.no_grad():
                w = self._compute_attn_weights(q1, enc, key_padding_mask)
            return ctx, w

        # pure vanilla dot
        w = self._compute_attn_weights(q1, enc, key_padding_mask)
        ctx = torch.bmm(w.unsqueeze(1), enc).squeeze(1)
        return ctx, w

    def _mha_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q = self._split_heads(self.q_proj(dec_h.unsqueeze(1)))   # [B,nH,1,Hd]
        k = self._split_heads(self.k_proj(enc))                  # [B,nH,T,Hd]
        v = self._split_heads(self.v_proj(enc))                  # [B,nH,T,Hd]

        if self.backend == "flash":
            out = self._flash(q, k, v)                           # [B,1,D]
            ctx = self.context_proj(out)[:, 0]
        elif self.backend == "xformers":
            out = self._xformers(q, k, v)                        # [B,nH,1,Hd]
            ctx = self._combine_heads(out)[:, 0]
        else:
            out = self._sdpa(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            ctx = self._combine_heads(out)[:, 0]
        return ctx, None

    def _prob_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, key_padding_mask=None):
        """
        Lightweight 'prob' attention: scaled scores with optional top-k sparsification.
        """
        q1 = dec_h.unsqueeze(1)  # [B,1,D]
        scores = torch.bmm(q1, enc.transpose(1, 2)) * (1.0 / math.sqrt(enc.size(-1)))  # [B,1,T]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)  # [B,1,T]
        ctx = torch.bmm(attn, enc).squeeze(1)
        return ctx, attn.squeeze(1)

    def _temporal_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, timestamps: Optional[torch.Tensor], key_padding_mask=None):
        if timestamps is not None:
            enc = enc + self.time_bias(timestamps)  # [B,T,D]
        return self._dot_attention(dec_h, enc, key_padding_mask=key_padding_mask)

    def _multiscale_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q = self._split_heads(self.q_proj(dec_h.unsqueeze(1)))  # [B,nH,1,Hd]
        outs = []
        for i, d in enumerate(self.dilations):
            e = enc[:, ::d, :] if d > 1 else enc
            k_i = self._split_heads(self.scale_projections[i]["k"](e))
            v_i = self._split_heads(self.scale_projections[i]["v"](e))
            if self.backend == "xformers":
                o = self._xformers(q, k_i, v_i)
            else:
                o = self._sdpa(q, k_i, v_i, attn_mask=attn_mask, key_padding_mask=(key_padding_mask[:, ::d] if key_padding_mask is not None and d > 1 else key_padding_mask))
            outs.append(self._combine_heads(o))  # [B,1,D]

        S = torch.stack(outs, dim=0)  # [S,B,1,D]
        w = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        combined = (S * w).sum(dim=0)  # [B,1,D]
        ctx = self.scale_out_proj(combined)[:, 0]
        return ctx, None

    def _autocorr_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, topk: int = 3):
        """
        Simple auto-correlation aggregation:
          - compute autocorr of encoder along time (per feature),
          - pick top-k lags globally per batch,
          - average encoder outputs shifted by those lags.
        This works even with a single-step query.
        """
        B, T, D = enc.shape
        # FFT over time for encoder only
        enc_fft = torch.fft.rfft(enc.transpose(1, 2).contiguous(), dim=-1)  # [B,D,T//2+1]
        power = enc_fft * torch.conj(enc_fft)  # [B,D,F]
        ac = torch.fft.irfft(power, n=T, dim=-1).real  # [B,D,T]
        # mean over features -> prominence per lag
        lag_score = ac.mean(dim=1)  # [B,T]
        # ignore lag 0
        lag_score[:, 0] = float("-inf")
        k = min(topk, T)
        _, idx = torch.topk(lag_score, k=k, dim=1)

        # aggregate shifted encodings
        rolled = []
        for b in range(B):
            shifts = []
            for j in range(k):
                s = idx[b, j].item()
                shifts.append(torch.roll(enc[b], -s, dims=0))
            rolled.append(torch.stack(shifts, dim=0).mean(dim=0))  # [T,D]
        agg = torch.stack(rolled, dim=0)  # [B,T,D]
        ctx = agg.mean(dim=1)  # [B,D]
        return ctx, None

    # ----- forward -----

    def forward(
        self,
        decoder_hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        encoder_outputs: torch.Tensor,
        encoder_timestamps: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,           # [Tq,Tk] (Tq=1 here)
        key_padding_mask: Optional[torch.Tensor] = None,    # [B,Tk], True = mask
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          attended: [B,D]
          attn_weights: [B,T] or None
        """
        dec_h = self._last_hidden(decoder_hidden)                # [B,H]
        enc = self.encoder_projection(encoder_outputs)           # [B,T,D]

        if self.method == "dot":
            ctx, w = self._dot_attention(dec_h, enc, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        elif self.method == "mha":
            ctx, w = self._mha_attention(dec_h, enc, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        elif self.method == "prob":
            ctx, w = self._prob_attention(dec_h, enc, key_padding_mask=key_padding_mask)
        elif self.method == "temporal":
            ctx, w = self._temporal_attention(dec_h, enc, encoder_timestamps, key_padding_mask=key_padding_mask)
        elif self.method == "multiscale":
            ctx, w = self._multiscale_attention(dec_h, enc, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        elif self.method == "autocorr":
            ctx, w = self._autocorr_attention(dec_h, enc)
        else:
            raise ValueError(f"Unknown attention method: {self.method}")

        out = self.combined_layer(torch.cat([ctx, dec_h], dim=1))  # [B,D]
        if self.use_residual_ln:
            out = self.post_ln(out + dec_h)
        return torch.tanh(out), w
