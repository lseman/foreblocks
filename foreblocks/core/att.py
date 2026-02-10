"""
Enhanced SOTA Multi-Method Attention Layer (SAFE SDPA + Large-Batch Guard)
- Updated to use torch.nn.attention.sdpa_kernel() (deprecates torch.backends.cuda.sdp_kernel)

New Methods:
- ProbSparse: Efficient sparse attention from Informer
- Linear: Linear attention with kernel approximation
- Sliding Window: Local attention with configurable window
- LSH: Locality-sensitive hashing attention

Existing Improvements:
- GQA/MQA support for efficient KV caching
- Flash Attention 2 compatibility (explicit backend "flash")
- Proper mask handling for PyTorch 2.0+ SDPA
- xFormers integration
- SAFE SDPA path that avoids invalid CUDA launch configs
- Large-batch (>65k) guard: auto-disable mem-efficient + chunked fallback
"""

import math
from contextlib import nullcontext
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import (
    SDPBackend,  # <- NEW API
    sdpa_kernel,
)

# ---------- Optional deps ----------
try:
    import xformers
    import xformers.ops as xops
    HAS_XFORMERS = True
except Exception:
    HAS_XFORMERS = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    HAS_FLASH_ATTN = False


class AttentionLayer(nn.Module):
    _VALID_METHODS = frozenset({
        "dot", "mha", "multiscale", "probsparse",
        "linear", "sliding_window", "lsh"
    })
    _VALID_BACKENDS = frozenset({"torch", "xformers", "flash"})
    _MHA_METHODS = frozenset({"mha", "multiscale", "probsparse", "sliding_window", "lsh"})

    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: Optional[int] = None,
        method: str = "mha",
        attention_backend: str = "torch",
        nhead: int = 4,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        num_scales: int = 3,
        use_residual_ln: bool = False,
        use_swiglu: bool = True,
        # ProbSparse params
        sampling_factor: int = 5,
        top_k: Optional[int] = None,
        # Sliding window params
        window_size: int = 256,
        # LSH params
        n_hashes: int = 4,
        bucket_size: int = 64,
        verbose: bool = False,
        # Safety / Debug
        force_math_kernel_if_hd_exotic: bool = True,
        sanity_asserts: bool = True,
        large_batch_limit: int = 65535,   # mem-efficient SDPA internal limit ~65k
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

        # Safety / debug flags
        self.force_math_kernel_if_hd_exotic = force_math_kernel_if_hd_exotic
        self.sanity_asserts = sanity_asserts
        self.large_batch_limit = int(large_batch_limit)

        # GQA setup
        self.n_kv_heads = n_kv_heads or nhead
        assert nhead % self.n_kv_heads == 0, "nhead must be divisible by n_kv_heads"
        self.n_rep = nhead // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Method-specific params
        self.sampling_factor = sampling_factor
        self.top_k = top_k
        self.window_size = window_size
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self.dropout = nn.Dropout(dropout)
        self._build_projections(use_swiglu)
        self._build_method_specific_layers(num_scales)

        if use_residual_ln:
            self.post_ln = nn.LayerNorm(self.decoder_hidden_size)

        if verbose:
            self._print_initialization_info()

    # ----- Setup -----

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

    def _build_projections(self, use_swiglu: bool):
        if use_swiglu:
            self.combined_layer = nn.Sequential(
                nn.Linear(self.decoder_hidden_size * 2, self.decoder_hidden_size * 4, bias=False),
                nn.SiLU(),
                nn.Linear(self.decoder_hidden_size * 4, self.decoder_hidden_size, bias=False)
            )
        else:
            self.combined_layer = nn.Linear(self.decoder_hidden_size * 2, self.decoder_hidden_size)

        self.encoder_projection = (
            nn.Identity()
            if self.decoder_hidden_size == self.encoder_hidden_size
            else nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=False)
        )

        if self.method in self._MHA_METHODS or self.method == "dot":
            self.q_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.k_proj = nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False)
            if self.backend in ["flash", "xformers"] or self.method in ["linear", "lsh"]:
                self.out_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)

    def _build_method_specific_layers(self, num_scales: int):
        if self.method == "multiscale":
            self.scale_projections = nn.ModuleList([
                nn.ModuleDict({
                    "k": nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False),
                    "v": nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False),
                })
                for _ in range(num_scales)
            ])
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.dilations = tuple(2 ** i for i in range(num_scales))

        elif self.method == "linear":
            self.feature_dim = self.head_dim
            self.feature_map = nn.Sequential(
                nn.Linear(self.head_dim, self.feature_dim),
                nn.ReLU()
            )

        elif self.method == "lsh":
            self.register_buffer(
                "random_rotations",
                torch.randn(self.n_hashes, self.head_dim, max(1, self.head_dim // 2))
            )

    def _print_initialization_info(self):
        gqa_info = f"GQA({self.nhead}q/{self.n_kv_heads}kv)" if self.n_rep > 1 else "MHA"
        print(f"[Attention] {gqa_info}, Method: {self.method}, Backend: {self.backend}")
        if self.method == "probsparse":
            print(f"[Attention] ProbSparse: sampling_factor={self.sampling_factor}, top_k={self.top_k}")
        elif self.method == "sliding_window":
            print(f"[Attention] Sliding window size: {self.window_size}")
        elif self.method == "lsh":
            print(f"[Attention] LSH: n_hashes={self.n_hashes}, bucket_size={self.bucket_size}")
        if HAS_XFORMERS:
            print(f"[Attention] xFormers: {getattr(xformers, '__version__', 'unknown')}")
        if HAS_FLASH_ATTN:
            print(f"[Attention] FlashAttention 2 available")

    # ----- Helpers -----

    @staticmethod
    def _last_hidden(dec_h: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        if isinstance(dec_h, tuple):
            dec_h = dec_h[0]
        return dec_h[-1]  # [L,B,H] -> [B,H]

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, n_kv_heads, T, Hd = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_heads, self.n_rep, T, Hd)
        return x.reshape(B, n_kv_heads * self.n_rep, T, Hd)

    def _project_qkv(self, query: torch.Tensor, enc: torch.Tensor):
        B, T, _ = enc.shape
        q = self.q_proj(query).view(B, 1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, Hh, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, Hh * Hd)

    # -------- SAFE SDPA selection (NEW API) --------

    def _safe_sdpa_ctx(self, q: torch.Tensor):
        """
        Choose safe SDPA backends using the new torch.nn.attention.sdpa_kernel API.
        - Disable FLASH entirely in this path.
        - If exotic head_dim or very large effective batch -> MATH only.
        - Else allow [MATH, EFFICIENT].
        """
        if not q.is_cuda:
            return nullcontext()

        B, H = q.shape[0], q.shape[1]
        eff_batch = B * H

        # math-only if exotic head_dim or very large effective batch
        if (self.force_math_kernel_if_hd_exotic and self.head_dim not in (8, 16, 32, 64, 128)) or (eff_batch > self.large_batch_limit):
            return sdpa_kernel(backends=[SDPBackend.MATH])

        # General safe default: allow EFFICIENT but not FLASH
        return sdpa_kernel(backends=[SDPBackend.MATH, SDPBackend.EFFICIENT])

    def _sdpa_math_only(self, q, k, v, attn_mask, drop):
        """Single SDPA call under MATH-only context (used by retry/chunk)."""
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=False
            )

    def _torch_sdpa(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """PyTorch SDPA with safety guards + large-batch fallback (NEW API)."""
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

        if self.sanity_asserts:
            assert q.dim() == k.dim() == v.dim() == 4, "q/k/v must be [B,H,T,Hd]"
            assert q.shape[0] == k.shape[0] == v.shape[0], "Batch mismatch"
            assert q.shape[1] == self.nhead, f"q must have {self.nhead} heads"
            assert k.shape[1] == v.shape[1], "KV heads mismatch"
            assert k.shape[-1] == v.shape[-1] == self.head_dim, "Head dim mismatch"

        B, H, Tq, _ = q.shape
        Tk = k.shape[2]
        eff_batch = B * H

        # Build broadcastable boolean mask [B,H,Tq,Tk] if needed
        combined_mask = None
        if attn_mask is not None or key_padding_mask is not None:
            combined_mask = torch.zeros(B, H, Tq, Tk, dtype=torch.bool, device=q.device)
            if attn_mask is not None:
                if attn_mask.dim() == 2 and attn_mask.shape == (Tq, Tk):
                    combined_mask |= attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 1 and attn_mask.shape[0] == Tk:
                    combined_mask |= attn_mask.view(1, 1, 1, Tk)
                else:
                    combined_mask |= attn_mask  # assume already broadcastable
            if key_padding_mask is not None:
                combined_mask |= key_padding_mask.view(B, 1, 1, Tk)

        drop = self.dropout.p if self.training else 0.0

        # If effective batch is too large, try MATH-only directly or chunk
        if q.is_cuda and eff_batch > self.large_batch_limit:
            try:
                return self._sdpa_math_only(q, k, v, combined_mask, drop)
            except RuntimeError as e:
                if "batch size exceeds" not in str(e) and "valid seed and offset" not in str(e):
                    raise
                # chunk size per B so that (chunkB * H) <= limit
                chunk_b = max(1, self.large_batch_limit // H)
                outs = []
                for s in range(0, B, chunk_b):
                    e_ = min(s + chunk_b, B)
                    out_chunk = self._sdpa_math_only(
                        q[s:e_], k[s:e_], v[s:e_],
                        combined_mask[s:e_] if combined_mask is not None else None,
                        drop
                    )
                    outs.append(out_chunk)
                return torch.cat(outs, dim=0)

        # Normal path: allow EFFICIENT unless _safe_sdpa_ctx disables it
        try:
            with self._safe_sdpa_ctx(q):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=combined_mask,
                    dropout_p=drop,
                    is_causal=False
                )
            return out
        except RuntimeError as e:
            # Retry with MATH-only if EFFICIENT tripped the 65k limit (or similar)
            msg = str(e)
            if ("batch size exceeds" in msg) or ("valid seed and offset" in msg):
                return self._sdpa_math_only(q, k, v, combined_mask, drop)
            raise

    def _flash_attn(self, q, k, v):
        assert HAS_FLASH_ATTN and flash_attn_func is not None
        k = self._repeat_kv(k); v = self._repeat_kv(v)
        dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q = q.transpose(1, 2).contiguous().to(dtype)
        k = k.transpose(1, 2).contiguous().to(dtype)
        v = v.transpose(1, 2).contiguous().to(dtype)
        drop = self.dropout.p if self.training else 0.0
        out = flash_attn_func(q, k, v, dropout_p=drop, causal=False)
        return out.transpose(1, 2).to(torch.float32)

    def _xformers_attn(self, q, k, v):
        k = self._repeat_kv(k); v = self._repeat_kv(v)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        drop = self.dropout.p if self.training else 0.0
        out = xops.memory_efficient_attention(q, k, v, p=drop)
        return out.transpose(1, 2)

    # ----- Attention Methods -----

    def _compute_attn_weights(self, query_1t: torch.Tensor, enc: torch.Tensor, key_padding_mask=None):
        scores = torch.bmm(enc, query_1t.transpose(1, 2)).squeeze(2) * self.scale_factor
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        return F.softmax(scores, dim=1)

    def _dot_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        if hasattr(self, "q_proj"):
            q, k, v = self._project_qkv(q1, enc)
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            if self.backend == "flash":
                out = self._flash_attn(q, k, v)
                ctx = self._combine_heads(out)[:, 0]; ctx = self.out_proj(ctx)
            elif self.backend == "xformers":
                out = self._xformers_attn(q, k, v)
                ctx = self._combine_heads(out)[:, 0]; ctx = self.out_proj(ctx)
            else:
                out = self._torch_sdpa(q, k, v, attn_mask, key_padding_mask)
                ctx = self._combine_heads(out)[:, 0]
            with torch.no_grad():
                w = self._compute_attn_weights(q1, enc, key_padding_mask)
            return ctx, w
        w = self._compute_attn_weights(q1, enc, key_padding_mask)
        ctx = torch.bmm(w.unsqueeze(1), enc).squeeze(1)
        return ctx, w

    def _mha_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        if self.backend == "flash":
            out = self._flash_attn(q, k, v)
            ctx = self._combine_heads(out)[:, 0]; ctx = self.out_proj(ctx)
        elif self.backend == "xformers":
            out = self._xformers_attn(q, k, v)
            ctx = self._combine_heads(out)[:, 0]; ctx = self.out_proj(ctx)
        else:
            out = self._torch_sdpa(q, k, v, attn_mask, key_padding_mask)
            ctx = self._combine_heads(out)[:, 0]
        return ctx, None

    def _multiscale_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q = self.q_proj(q1).view(-1, 1, self.nhead, self.head_dim).transpose(1, 2).contiguous()
        outs = []
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                e = enc[:, ::dilation, :]
                kpm = key_padding_mask[:, ::dilation] if key_padding_mask is not None else None
            else:
                e = enc
                kpm = key_padding_mask
            B, T, _ = e.shape
            k_i = self.scale_projections[i]["k"](e).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            v_i = self.scale_projections[i]["v"](e).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            if self.backend == "xformers":
                o = self._xformers_attn(q, k_i, v_i)
            else:
                o = self._torch_sdpa(q, k_i, v_i, attn_mask, kpm)
            outs.append(self._combine_heads(o))
        S = torch.stack(outs, dim=0)
        w = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        combined = (S * w).sum(dim=0)
        ctx = combined[:, 0]
        return ctx, None

    def _probsparse_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        k_full = self._repeat_kv(k).contiguous()
        v_full = self._repeat_kv(v).contiguous()
        U_part = min(self.sampling_factor * int(math.log(max(2, Tk))), Tk)
        u = torch.randint(0, Tk, (U_part,), device=q.device)
        k_sample = k_full[:, :, u, :]
        q_k_sample = torch.einsum('bhqd,bhkd->bhqk', q, k_sample)
        M = q_k_sample.max(dim=-1)[0] - q_k_sample.mean(dim=-1)
        top_k = self.top_k or min(25, Tq)
        _, top_idx = M.topk(min(top_k, Tq), dim=-1, sorted=False)
        q_reduced = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D))
        scores = torch.einsum('bhqd,bhkd->bhqk', q_reduced, k_full) * self.scale_factor
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, Tk), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v_full)
        ctx = self._combine_heads(out)[:, 0]
        return ctx, None

    def _linear_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        k = self._repeat_kv(k).contiguous()
        v = self._repeat_kv(v).contiguous()
        q = (F.elu(q) + 1).contiguous()
        k = (F.elu(k) + 1).contiguous()
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = k.sum(dim=2)
        num = torch.einsum('bhqd,bhde->bhqe', q, kv)
        denom = torch.einsum('bhqd,bhd->bhq', q, z).unsqueeze(-1)
        out = num / (denom + 1e-6)
        ctx = self._combine_heads(out)[:, 0]
        ctx = self.out_proj(ctx)
        return ctx, None

    def _sliding_window_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        Tk = k.shape[2]
        k = self._repeat_kv(k).contiguous()
        v = self._repeat_kv(v).contiguous()
        win = min(self.window_size, Tk)
        k_win = k[:, :, -win:, :]
        v_win = v[:, :, -win:, :]
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k_win) * self.scale_factor
        if key_padding_mask is not None:
            mask_win = key_padding_mask[:, -win:]
            scores = scores.masked_fill(mask_win.view(q.shape[0], 1, 1, -1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v_win)
        ctx = self._combine_heads(out)[:, 0]
        return ctx, None

    def _lsh_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        Tk = k.shape[2]
        k = self._repeat_kv(k).contiguous()
        v = self._repeat_kv(v).contiguous()
        D2 = self.random_rotations.shape[-1]
        q_rot = torch.einsum('bhqd,hdk->bhqhk', q, self.random_rotations)
        k_rot = torch.einsum('bhkd,hdk->bhkhj', k, self.random_rotations)
        q_hash = (q_rot > 0).long()
        k_hash = (k_rot > 0).long()
        bits = min(8, D2)
        pow2 = torch.tensor([2**i for i in range(bits)], device=q.device, dtype=torch.long)
        q_buckets = (q_hash[..., :bits] * pow2).sum(-1)
        k_buckets = (k_hash[..., :bits] * pow2).sum(-1)
        matching_mask = torch.zeros(q.shape[0], q.shape[1], 1, Tk, dtype=torch.bool, device=q.device)
        for h in range(self.n_hashes):
            q_b = q_buckets[:, :, :, h]
            k_b = k_buckets[:, :, :, h]
            matching_mask |= (q_b.unsqueeze(-1) == k_b.unsqueeze(2))
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale_factor
        scores = scores.masked_fill(~matching_mask, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(q.shape[0], 1, 1, Tk), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        ctx = self._combine_heads(out)[:, 0]
        ctx = self.out_proj(ctx)
        return ctx, None

    # ----- Forward -----

    def forward(
        self,
        decoder_hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        encoder_outputs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dec_h = self._last_hidden(decoder_hidden)
        enc = self.encoder_projection(encoder_outputs)

        if self.method == "dot":
            ctx, w = self._dot_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "mha":
            ctx, w = self._mha_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "multiscale":
            ctx, w = self._multiscale_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "probsparse":
            ctx, w = self._probsparse_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "linear":
            ctx, w = self._linear_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "sliding_window":
            ctx, w = self._sliding_window_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "lsh":
            ctx, w = self._lsh_attention(dec_h, enc, attn_mask, key_padding_mask)
        else:
            raise ValueError(f"Unknown attention method: {self.method}")

        out = self.combined_layer(torch.cat([ctx, dec_h], dim=1))
        if self.use_residual_ln:
            out = self.post_ln(out + dec_h)
        return torch.tanh(out), w
