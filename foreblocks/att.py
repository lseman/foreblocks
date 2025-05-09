import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        num_scales=10,
        verbose=False
    ):
        super().__init__()
        if verbose:
            print(f"[Attention] Method: {method}, Backend: {attention_backend}")

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

        self.needs_projection = self.decoder_hidden_size != self.encoder_hidden_size
        if self.needs_projection:
            self.encoder_projection = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)

        if method == 'mha':
            assert decoder_hidden_size % nhead == 0, "Hidden size must be divisible by number of heads"
            self.head_dim = decoder_hidden_size // nhead
            self.q_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)
            self.k_proj = nn.Linear(self.encoder_hidden_size, decoder_hidden_size)
            self.v_proj = nn.Linear(self.encoder_hidden_size, decoder_hidden_size)
            self.out_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)
            self.scale = math.sqrt(self.head_dim)

            if attention_backend == 'multiscale':
                self.k_projs = nn.ModuleList([nn.Linear(self.encoder_hidden_size, decoder_hidden_size) for _ in range(num_scales)])
                self.v_projs = nn.ModuleList([nn.Linear(self.encoder_hidden_size, decoder_hidden_size) for _ in range(num_scales)])
                self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
                self.dilations = [1, 2, 4][:num_scales]
                self.scale_out_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)

        if attention_backend == 'temporal':
            self.time_bias = nn.Linear(time_embed_dim, self.encoder_hidden_size)

        self.combined_layer = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.decoder_hidden_size)

    def _prepare_decoder_hidden(self, decoder_hidden):
        if isinstance(decoder_hidden, tuple):
            decoder_hidden = decoder_hidden[0]
        return decoder_hidden[-1] if decoder_hidden.dim() == 3 else decoder_hidden

    def _split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.head_dim)

    def _dilate_sequence(self, x, dilation):
        if dilation == 1:
            return x
        indices = torch.arange(0, x.size(1), dilation, device=x.device)
        return x[:, indices, :]

    def _mha_attention(self, query, key, value):
        B = query.size(0)
        q = self._split_heads(self.q_proj(query), B)
        k = self._split_heads(self.k_proj(key), B)
        v = self._split_heads(self.v_proj(value), B)

        if self.use_xformers and self.attention_backend == 'xformers':
            out = xops.memory_efficient_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            out = self._combine_heads(out.transpose(1, 2), B)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
            out = self._combine_heads(out, B)

        return self.out_proj(out), None

    def _multiscale_attention(self, query, key, value):
        B = query.size(0)
        q = self._split_heads(self.q_proj(query), B)
        outputs = []

        for i, dilation in enumerate(self.dilations):
            k_i = self.k_projs[i](self._dilate_sequence(key, dilation))
            v_i = self.v_projs[i](self._dilate_sequence(value, dilation))
            k_i = self._split_heads(k_i, B)
            v_i = self._split_heads(v_i, B)

            if self.use_xformers and self.attention_backend == 'xformers':
                out = xops.memory_efficient_attention(q.transpose(1, 2), k_i.transpose(1, 2), v_i.transpose(1, 2))
                out = self._combine_heads(out.transpose(1, 2), B)
            else:
                out = F.scaled_dot_product_attention(q, k_i, v_i, dropout_p=self.dropout.p)
                out = self._combine_heads(out, B)

            outputs.append(out)

        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * o for w, o in zip(weights, outputs))
        return self.scale_out_proj(combined), None

    def forward(self, decoder_hidden, encoder_outputs, encoder_timestamps=None):
        B, T, _ = encoder_outputs.size()
        decoder_hidden = self._prepare_decoder_hidden(decoder_hidden)

        if self.needs_projection:
            encoder_outputs = self.encoder_projection(encoder_outputs)

        if self.attention_backend == 'temporal' and encoder_timestamps is not None:
            encoder_outputs = encoder_outputs + self.time_bias(encoder_timestamps)

        query = decoder_hidden.unsqueeze(1)

        if self.method == 'mha':
            if self.attention_backend == 'multiscale':
                context, _ = self._multiscale_attention(query, encoder_outputs, encoder_outputs)
            else:
                context, _ = self._mha_attention(query, encoder_outputs, encoder_outputs)
            context = context.squeeze(1)
        elif self.attention_backend == 'prob':
            scores = torch.bmm(query, encoder_outputs.transpose(1, 2))
            topk = max(1, int(T * 0.3))
            topk_vals, topk_idx = torch.topk(scores, topk, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
        else:  # Dot-product attention
            attn_weights = torch.bmm(encoder_outputs, query.transpose(1, 2)).squeeze(2)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        combined = self.combined_layer(torch.cat((context, decoder_hidden), dim=1))
        return torch.tanh(combined), attn_weights if 'attn_weights' in locals() else None
