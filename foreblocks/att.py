"""
AttentionLayer: Flexible and optimized attention module for sequence-to-sequence models.

Supported Attention Mechanisms
------------------------------

Set via `method='mha'` and `attention_backend=...`

- "dot": 
    • Classic dot-product attention (no heads, single projection).
    • Use: method='dot'

- "prob":
    • Probabilistic attention using top-k scores (fast for long sequences).
    • Use: method='mha', attention_backend='prob'

- "xformers":
    • xFormers memory-efficient attention.
    • Use: method='mha', attention_backend='xformers'
    • Requires: `xformers` package

- "flash":
    • FlashAttention v2 or PyTorch-native FlashAttention (scalable, efficient).
    • Use: method='mha', attention_backend='flash'
    • Requires: `flash_attn` or PyTorch >= 2.0 with hardware support (A100/H100)

- "autocorr":
    • AutoCorrelation attention (as in Autoformer/FEDformer).
    • Use: method='mha', attention_backend='autocorr'

- "multiscale":
    • Multiscale (dilated) attention with multiple resolutions.
    • Use: method='mha', attention_backend='multiscale'

- "temporal":
    • Temporal-bias attention using timestamp embeddings.
    • Use: method='mha', attention_backend='temporal'
    • Requires: encoder_timestamps input

Usage
-----
layer = AttentionLayer(
    decoder_hidden_size=128,
    method="mha",
    attention_backend="flash"  # change to any of the backends above
)

output, attn_weights = layer(dec_hidden, enc_outputs, encoder_timestamps=optional)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.functional import scaled_dot_product_attention as torch_sdp_attention
    HAS_TORCH_SDP = True
except ImportError:
    HAS_TORCH_SDP = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import xformers.ops as xops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False


class AttentionLayer(nn.Module):
    def __init__(
        self,
        decoder_hidden_size,
        encoder_hidden_size=None,
        attention_size=None,
        method='dot',
        nhead=4,
        dropout=0.1,
        use_xformers=True,
        attention_backend='xformers',
        time_embed_dim=8,
        num_scales=3,
        verbose=True
    ):
        super().__init__()
        if verbose:
            print(f"[Attention] Method: {method}, Backend: {attention_backend}")
            # print if xformers is installed and version
            if HAS_XFORMERS:
                import xformers
                print(f"[Attention] xformers version: {xformers.__version__}")
            if HAS_FLASH_ATTN:
                import flash_attn
                print(f"[Attention] FlashAttention version: {flash_attn.__version__}")

        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size or decoder_hidden_size
        self.attention_size = attention_size or decoder_hidden_size
        self.method = method
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.use_xformers = use_xformers and HAS_XFORMERS
        self.attention_backend = attention_backend
        self.time_embed_dim = time_embed_dim
        self.num_scales = num_scales

        self.head_dim = decoder_hidden_size // nhead
        assert self.head_dim * nhead == decoder_hidden_size, "Hidden size must be divisible by nhead"

        self.qkv_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size * 3)
        self.out_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)

        self.encoder_projection = (
            nn.Identity() if decoder_hidden_size == self.encoder_hidden_size
            else nn.Linear(self.encoder_hidden_size, decoder_hidden_size)
        )

        if attention_backend == 'temporal':
            self.time_bias = nn.Linear(time_embed_dim, decoder_hidden_size)

        if attention_backend == 'multiscale':
            self.k_projs = nn.ModuleList([nn.Linear(decoder_hidden_size, decoder_hidden_size) for _ in range(num_scales)])
            self.v_projs = nn.ModuleList([nn.Linear(decoder_hidden_size, decoder_hidden_size) for _ in range(num_scales)])
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.dilations = [2 ** i for i in range(num_scales)]
            self.scale_out_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)

        self.combined_layer = nn.Linear(decoder_hidden_size + decoder_hidden_size, decoder_hidden_size)

        self.context_proj = (
            nn.Identity()  # default: no-op
            if self.attention_backend != "flash"
            else nn.Linear(self.nhead * self.head_dim, self.decoder_hidden_size)
        )

    def _prepare_decoder_hidden(self, h):
        if isinstance(h, tuple):
            h = h[0]
        return h[-1] if h.dim() == 3 else h  # [B, H]

    def _split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def _dilate_sequence(self, x, dilation):
        return x[:, ::dilation, :] if dilation > 1 else x

    def _auto_correlation_attention(self, q, k, v, topk=3):
        # FFT-based autocorrelation attention (Autoformer)
        q_fft = torch.fft.rfft(q, dim=1)
        k_fft = torch.fft.rfft(k, dim=1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=1)
        topk_idx = torch.topk(corr, k=topk, dim=1).indices
        agg = torch.stack([v.roll(-d.item(), dims=1) for d in topk_idx[0]], dim=0).mean(dim=0)
        return agg.unsqueeze(1)

    def _scaled_dot_attention(self, q, k, v):
        B = q.size(0)
        q, k, v = map(self._split_heads, (q, k, v))  # [B, nhead, T, head_dim]

        if self.attention_backend == 'flash':
            if HAS_FLASH_ATTN:
                q = q.contiguous().to(torch.float16)
                k = k.contiguous().to(torch.float16)
                v = v.contiguous().to(torch.float16)


                out = flash_attn_func(q, k, v, dropout_p=self.dropout.p, causal=False)  # [B, nhead, T, head_dim]
                out = out.transpose(1, 2)  # [B, T, nhead, head_dim]
                out = out.contiguous().view(B, -1, self.nhead * self.head_dim)  # [B, T, nhead * head_dim]
                out = out.to(torch.float32)  # convert back to float32
                return self.context_proj(out)  # project [B, T, D] to decoder_hidden_size
            elif HAS_TORCH_SDP:
                out = torch_sdp_attention(q, k, v, dropout_p=self.dropout.p, is_causal=False)
                return self._combine_heads(out)
            else:
                raise RuntimeError("Flash attention requested but not available.")
        elif self.use_xformers and self.attention_backend == 'xformers':
            out = xops.memory_efficient_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            return self._combine_heads(out.transpose(1, 2))
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
            return self._combine_heads(out)


    def _multiscale_attention(self, query, key, value):
        B = query.size(0)
        q = self._split_heads(query)
        outputs = []

        for i, dilation in enumerate(self.dilations):
            k_i = self.k_projs[i](self._dilate_sequence(key, dilation))
            v_i = self.v_projs[i](self._dilate_sequence(value, dilation))
            k_i = self._split_heads(k_i)
            v_i = self._split_heads(v_i)

            if self.use_xformers:
                out = xops.memory_efficient_attention(q.transpose(1, 2), k_i.transpose(1, 2), v_i.transpose(1, 2))
                out = self._combine_heads(out.transpose(1, 2))
            else:
                out = F.scaled_dot_product_attention(q, k_i, v_i, dropout_p=self.dropout.p)
                out = self._combine_heads(out)

            outputs.append(out)

        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * o for w, o in zip(weights, outputs))
        return self.scale_out_proj(combined)

    def forward(self, decoder_hidden, encoder_outputs, encoder_timestamps=None):
        B, T, _ = encoder_outputs.shape
        decoder_hidden = self._prepare_decoder_hidden(decoder_hidden)  # [B, D]

        encoder_outputs = self.encoder_projection(encoder_outputs)

        if self.attention_backend == 'temporal' and encoder_timestamps is not None:
            encoder_outputs = encoder_outputs + self.time_bias(encoder_timestamps)

        query = decoder_hidden.unsqueeze(1)  # [B, 1, D]

        if self.method == 'mha':
            qkv = self.qkv_proj(torch.cat([query.expand(-1, T, -1)], dim=1))  # [B, T, 3D]
            q, k, v = qkv.chunk(3, dim=-1)

            if self.attention_backend == 'multiscale':
                context = self._multiscale_attention(q, encoder_outputs, encoder_outputs)
            elif self.attention_backend == 'autocorr':
                context = self._auto_correlation_attention(q, encoder_outputs, encoder_outputs)
            else:
                context = self._scaled_dot_attention(q, encoder_outputs, encoder_outputs)

            context = context[:, 0]  # remove sequence dimension
        elif self.attention_backend == 'prob':
            scores = torch.bmm(query, encoder_outputs.transpose(1, 2)) / math.sqrt(self.decoder_hidden_size)
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
        else:  # dot
            attn_weights = torch.bmm(encoder_outputs, query.transpose(1, 2)).squeeze(2)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        combined = self.combined_layer(torch.cat([context, decoder_hidden], dim=1))
        return torch.tanh(combined), attn_weights if 'attn_weights' in locals() else None
