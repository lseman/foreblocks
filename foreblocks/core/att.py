import math
from contextlib import nullcontext
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Optional dependencies
try:
    import xformers.ops as xops

    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False


class AttentionLayer(nn.Module):
    """
    Enhanced multi-method attention layer with safe SDPA + modern backend support.

    Supports:
    - Full MHA / GQA / MQA
    - Multiscale (dilated)
    - ProbSparse (Informer-style)
    - Linear attention (kernel approximation)
    - Sliding window (local)
    - LSH (basic Reformer-style)

    Backends: torch (SDPA), xformers, flash-attn
    """

    _VALID_METHODS = frozenset(
        {"dot", "mha", "multiscale", "probsparse", "linear", "sliding_window", "lsh"}
    )
    _VALID_BACKENDS = frozenset({"torch", "xformers", "flash"})

    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: Optional[int] = None,
        method: str = "mha",
        attention_backend: str = "torch",
        nhead: int = 8,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        num_scales: int = 3,
        use_residual_ln: bool = True,
        use_swiglu: bool = True,
        # Method-specific
        sampling_factor: int = 5,
        top_k: Optional[int] = None,
        window_size: int = 512,
        n_hashes: int = 4,
        bucket_size: int = 64,
        # Safety / performance
        force_math_if_exotic_head: bool = True,
        large_batch_limit: int = 65536,
        compile_if_possible: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        self._validate(method, attention_backend, decoder_hidden_size, nhead)

        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size or decoder_hidden_size
        self.method = method.lower()
        self.backend = attention_backend.lower()
        self.nhead = nhead
        self.head_dim = decoder_hidden_size // nhead
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

        # GQA / MQA
        self.n_kv_heads = n_kv_heads or nhead
        assert self.nhead % self.n_kv_heads == 0
        self.n_rep = self.nhead // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Method params
        self.sampling_factor = sampling_factor
        self.top_k = top_k
        self.window_size = window_size
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.large_batch_limit = large_batch_limit
        self.force_math_if_exotic_head = force_math_if_exotic_head

        # Build layers
        self._build_projections(use_swiglu)
        self._build_method_specific(num_scales)

        if use_residual_ln:
            self.post_ln = nn.LayerNorm(decoder_hidden_size)

        if verbose:
            self._print_config()

        # Optional torch.compile
        if compile_if_possible and torch.compile is not None:
            try:
                self.forward = torch.compile(
                    self.forward, mode="reduce-overhead", fullgraph=True, dynamic=True
                )
                if verbose:
                    print("[AttentionLayer] torch.compile applied")
            except Exception as e:
                print(f"[AttentionLayer] torch.compile failed: {e}")

    def _validate(self, method: str, backend: str, hidden: int, nhead: int):
        if method not in self._VALID_METHODS:
            raise ValueError(f"Invalid method: {method}")
        if backend not in self._VALID_BACKENDS:
            raise ValueError(f"Invalid backend: {backend}")
        if hidden % nhead != 0:
            raise ValueError(
                f"decoder_hidden_size {hidden} must be divisible by nhead {nhead}"
            )
        if backend == "flash" and not HAS_FLASH_ATTN:
            raise ImportError("flash-attn not installed")
        if backend == "xformers" and not HAS_XFORMERS:
            raise ImportError("xformers not installed")

    def _build_projections(self, use_swiglu: bool):
        if use_swiglu:
            self.combined_layer = nn.Sequential(
                nn.Linear(
                    self.decoder_hidden_size * 2,
                    self.decoder_hidden_size * 4,
                    bias=False,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.decoder_hidden_size * 4, self.decoder_hidden_size, bias=False
                ),
            )
        else:
            self.combined_layer = nn.Linear(
                self.decoder_hidden_size * 2, self.decoder_hidden_size
            )

        self.encoder_projection = (
            nn.Identity()
            if self.decoder_hidden_size == self.encoder_hidden_size
            else nn.Linear(
                self.encoder_hidden_size, self.decoder_hidden_size, bias=False
            )
        )

        if self.method in {
            "mha",
            "multiscale",
            "probsparse",
            "sliding_window",
            "lsh",
            "dot",
        }:
            self.q_proj = nn.Linear(
                self.decoder_hidden_size, self.decoder_hidden_size, bias=False
            )
            self.k_proj = nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False)

        if self.backend in {"flash", "xformers"} or self.method in {"linear", "lsh"}:
            self.out_proj = nn.Linear(
                self.decoder_hidden_size, self.decoder_hidden_size, bias=False
            )

    def _build_method_specific(self, num_scales: int):
        if self.method == "multiscale":
            self.scale_projections = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "k": nn.Linear(
                                self.decoder_hidden_size, self.kv_dim, bias=False
                            ),
                            "v": nn.Linear(
                                self.decoder_hidden_size, self.kv_dim, bias=False
                            ),
                        }
                    )
                    for _ in range(num_scales)
                ]
            )
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.dilations = tuple(2**i for i in range(num_scales))
        elif self.method == "linear":
            self.feature_map = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim), nn.ReLU()
            )
        elif self.method == "lsh":
            self.register_buffer(
                "random_rotations",
                torch.randn(self.n_hashes, self.head_dim, self.head_dim // 2),
            )

    def _print_config(self):
        gqa = f"GQA ({self.nhead}q / {self.n_kv_heads}kv)" if self.n_rep > 1 else "MHA"
        print(f"[AttentionLayer] {gqa} | method={self.method} | backend={self.backend}")
        print(f"  head_dim={self.head_dim} | dropout={self.dropout_p:.2f}")

    # ────────────────────────────────────────────────────────────────
    #   Backend Helpers
    # ────────────────────────────────────────────────────────────────

    def _safe_sdpa_context(self, q: torch.Tensor):
        if not q.is_cuda:
            return nullcontext()

        B, nh = q.shape[:2]
        eff_batch = B * nh

        preferred = []
        if HAS_FLASH_ATTN and self.head_dim in (64, 128, 256):
            preferred.append(SDPBackend.FLASH_ATTENTION)
        preferred += [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

        if (
            self.force_math_if_exotic_head
            and self.head_dim not in (8, 16, 32, 64, 128, 256)
        ) or eff_batch > self.large_batch_limit:
            preferred = [SDPBackend.MATH]

        return sdpa_kernel(backends=preferred)

    def _torch_sdpa(self, q, k, v, attn_mask=None, key_padding_mask=None):
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        B, H, Tq, D = q.shape
        Tk = k.shape[2]

        combined_mask = None
        if attn_mask is not None or key_padding_mask is not None:
            combined_mask = torch.zeros(B, H, Tq, Tk, dtype=torch.bool, device=q.device)
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    combined_mask |= attn_mask[None, None, :, :]
                elif attn_mask.dim() == 4:
                    combined_mask |= attn_mask
            if key_padding_mask is not None:
                combined_mask |= key_padding_mask[:, None, None, :]

        drop_p = self.dropout_p if self.training else 0.0

        eff_batch = B * H
        if eff_batch > self.large_batch_limit:
            chunk_b = max(1, self.large_batch_limit // H)
            outs = []
            for s in range(0, B, chunk_b):
                e = min(s + chunk_b, B)
                chunk_mask = combined_mask[s:e] if combined_mask is not None else None
                out_chunk = F.scaled_dot_product_attention(
                    q[s:e],
                    k[s:e],
                    v[s:e],
                    attn_mask=chunk_mask,
                    dropout_p=drop_p,
                    is_causal=False,
                )
                outs.append(out_chunk)
            return torch.cat(outs, dim=0)

        try:
            with self._safe_sdpa_context(q):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=combined_mask, dropout_p=drop_p, is_causal=False
                )
        except RuntimeError as e:
            if "batch size" in str(e).lower() or "seed" in str(e).lower():
                with sdpa_kernel([SDPBackend.MATH]):
                    return F.scaled_dot_product_attention(
                        q, k, v, combined_mask, drop_p, is_causal=False
                    )
            raise

    def _flash(self, q, k, v):
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q = q.transpose(1, 2).contiguous().to(dtype)
        k = k.transpose(1, 2).contiguous().to(dtype)
        v = v.transpose(1, 2).contiguous().to(dtype)
        drop = self.dropout_p if self.training else 0.0
        out = flash_attn_func(q, k, v, dropout_p=drop, causal=False)
        return out.transpose(1, 2).to(q.dtype)

    def _xformers(self, q, k, v):
        q = q.transpose(1, 2).contiguous()
        k = self._repeat_kv(k).transpose(1, 2).contiguous()
        v = self._repeat_kv(v).transpose(1, 2).contiguous()
        drop = self.dropout_p if self.training else 0.0
        out = xops.memory_efficient_attention(q, k, v, p=drop)
        return out.transpose(1, 2)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, nh_kv, T, D = x.shape
        x = x[:, :, None].expand(-1, -1, self.n_rep, -1, -1)
        return x.reshape(B, nh_kv * self.n_rep, T, D)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    # ────────────────────────────────────────────────────────────────
    #   Forward
    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        decoder_hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        encoder_outputs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(decoder_hidden, tuple):
            dec_h = decoder_hidden[0][-1]
        else:
            dec_h = (
                decoder_hidden[:, -1] if decoder_hidden.dim() == 3 else decoder_hidden
            )

        enc = self.encoder_projection(encoder_outputs)

        method_fn = {
            "dot": self._dot_attention,
            "mha": self._mha_attention,
            "multiscale": self._multiscale_attention,
            "probsparse": self._probsparse_attention,
            "linear": self._linear_attention,
            "sliding_window": self._sliding_window_attention,
            "lsh": self._lsh_attention,
        }[self.method]

        ctx, weights = method_fn(dec_h, enc, attn_mask, key_padding_mask)

        combined = torch.cat([ctx, dec_h], dim=-1)
        out = self.combined_layer(combined)

        if hasattr(self, "post_ln"):
            out = self.post_ln(out + dec_h)

        return torch.tanh(out), weights

    # ────────────────────────────────────────────────────────────────
    #   Method implementations
    # ────────────────────────────────────────────────────────────────

    def _project_qkv(self, query: torch.Tensor, enc: torch.Tensor):
        B, T, _ = enc.shape
        q = self.q_proj(query).view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _dot_attention(self, dec_h, enc, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)

        if self.backend == "flash":
            out = self._flash(q, k, v)
        elif self.backend == "xformers":
            out = self._xformers(q, k, v)
        else:
            out = self._torch_sdpa(q, k, v, attn_mask, key_padding_mask)

        ctx = self._combine_heads(out)[:, 0]
        if hasattr(self, "out_proj"):
            ctx = self.out_proj(ctx)

        weights = None
        if not self.training:
            scores = torch.bmm(enc, q1.transpose(1, 2)).squeeze(2) * self.scale_factor
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)

        return ctx, weights

    def _mha_attention(self, dec_h, enc, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)

        if self.backend == "flash":
            out = self._flash(q, k, v)
        elif self.backend == "xformers":
            out = self._xformers(q, k, v)
        else:
            out = self._torch_sdpa(q, k, v, attn_mask, key_padding_mask)

        ctx = self._combine_heads(out)[:, 0]
        if hasattr(self, "out_proj"):
            ctx = self.out_proj(ctx)

        return ctx, None

    def _multiscale_attention(self, dec_h, enc, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q = (
            self.q_proj(q1)
            .view(-1, 1, self.nhead, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        outs = []
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                e = enc[:, ::dilation, :]
                kpm = (
                    key_padding_mask[:, ::dilation]
                    if key_padding_mask is not None
                    else None
                )
            else:
                e = enc
                kpm = key_padding_mask

            B, T, _ = e.shape
            k_i = (
                self.scale_projections[i]["k"](e)
                .view(B, T, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v_i = (
                self.scale_projections[i]["v"](e)
                .view(B, T, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )

            if self.backend == "xformers":
                o = self._xformers(q, k_i, v_i)
            else:
                o = self._torch_sdpa(q, k_i, v_i, attn_mask, kpm)

            outs.append(self._combine_heads(o))

        S = torch.stack(outs, dim=0)
        w = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        combined = (S * w).sum(dim=0)
        ctx = combined[:, 0]

        return ctx, None

    def _probsparse_attention(self, dec_h, enc, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        k_full = self._repeat_kv(k).contiguous()
        v_full = self._repeat_kv(v).contiguous()
        U_part = min(self.sampling_factor * int(math.log(max(2, Tk))), Tk)
        u = torch.randint(0, Tk, (U_part,), device=q.device)
        k_sample = k_full[:, :, u, :]
        q_k_sample = torch.einsum("bhqd,bhkd->bhqk", q, k_sample)
        M = q_k_sample.max(dim=-1)[0] - q_k_sample.mean(dim=-1)
        top_k = self.top_k or min(25, Tq)
        _, top_idx = M.topk(min(top_k, Tq), dim=-1, sorted=False)
        q_reduced = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D))
        scores = torch.einsum("bhqd,bhkd->bhqk", q_reduced, k_full) * self.scale_factor
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(B, 1, 1, Tk), float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v_full)
        ctx = self._combine_heads(out)[:, 0]
        return ctx, None

    def _linear_attention(self, dec_h, enc, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        k = self._repeat_kv(k).contiguous()
        v = self._repeat_kv(v).contiguous()
        q = (F.elu(q) + 1).contiguous()
        k = (F.elu(k) + 1).contiguous()
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = k.sum(dim=2)
        num = torch.einsum("bhqd,bhde->bhqe", q, kv)
        denom = torch.einsum("bhqd,bhd->bhq", q, z).unsqueeze(-1)
        out = num / (denom + 1e-6)
        ctx = self._combine_heads(out)[:, 0]
        ctx = self.out_proj(ctx)
        return ctx, None

    def _sliding_window_attention(
        self, dec_h, enc, attn_mask=None, key_padding_mask=None
    ):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        Tk = k.shape[2]
        k = self._repeat_kv(k).contiguous()
        v = self._repeat_kv(v).contiguous()
        win = min(self.window_size, Tk)
        k_win = k[:, :, -win:, :]
        v_win = v[:, :, -win:, :]
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k_win) * self.scale_factor
        if key_padding_mask is not None:
            mask_win = key_padding_mask[:, -win:]
            scores = scores.masked_fill(
                mask_win.view(q.shape[0], 1, 1, -1), float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v_win)
        ctx = self._combine_heads(out)[:, 0]
        return ctx, None

    def _lsh_attention(self, dec_h, enc, attn_mask=None, key_padding_mask=None):
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        Tk = k.shape[2]
        k = self._repeat_kv(k).contiguous()
        v = self._repeat_kv(v).contiguous()
        D2 = self.random_rotations.shape[-1]
        q_rot = torch.einsum("bhqd,hdk->bhqhk", q, self.random_rotations)
        k_rot = torch.einsum("bhkd,hdk->bhkhj", k, self.random_rotations)
        q_hash = (q_rot > 0).long()
        k_hash = (k_rot > 0).long()
        bits = min(8, D2)
        pow2 = torch.tensor(
            [2**i for i in range(bits)], device=q.device, dtype=torch.long
        )
        q_buckets = (q_hash[..., :bits] * pow2).sum(-1)
        k_buckets = (k_hash[..., :bits] * pow2).sum(-1)
        matching_mask = torch.zeros(
            q.shape[0], q.shape[1], 1, Tk, dtype=torch.bool, device=q.device
        )
        for h in range(self.n_hashes):
            q_b = q_buckets[:, :, :, h]
            k_b = k_buckets[:, :, :, h]
            matching_mask |= q_b.unsqueeze(-1) == k_b.unsqueeze(2)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale_factor
        scores = scores.masked_fill(~matching_mask, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(q.shape[0], 1, 1, Tk), float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        ctx = self._combine_heads(out)[:, 0]
        ctx = self.out_proj(ctx)
        return ctx, None
