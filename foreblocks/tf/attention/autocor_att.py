import math

import torch
import torch.nn as nn


class AutoCorrelation(nn.Module):
    """
    Fixed AutoCorrelation Mechanism - correcting critical bugs in the current implementation.
    """

    def __init__(
        self,
        mask_flag=True,
        factor=1,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

    def time_delay_agg(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Vectorized time delay aggregation using batched gather.
        values: [B, H, D, L]
        corr:   [B, L, H, D]
        """
        B, H, D, L = values.shape
        device = values.device
        top_k = max(1, min(int(self.factor * math.log(L)), L))

        mean_corr = corr.mean(dim=2).mean(dim=2)
        global_mean = mean_corr.mean(dim=0)
        topk = torch.topk(global_mean, top_k, dim=0).indices

        weights = mean_corr[:, topk]
        soft_weights = torch.softmax(weights, dim=-1)

        base = torch.arange(L, device=device)
        shifts = topk.view(-1, 1)
        indices = (base[None, :] - shifts) % L

        values_exp = values.unsqueeze(3).expand(B, H, D, top_k, L)
        gather_idx = indices.view(1, 1, 1, top_k, L).expand(B, H, D, top_k, L)

        rolled = torch.gather(values_exp, dim=-1, index=gather_idx)

        soft_weights = soft_weights.transpose(0, 1).view(top_k, B, 1, 1, 1)
        weighted = rolled.permute(3, 0, 1, 2, 4) * soft_weights

        return weighted.sum(dim=0)

    def _safe_fft_operations(self, queries, keys):
        """
        Fixed FFT operations with proper length specification and tensor handling.
        """
        B, L, H, E = queries.shape
        original_dtype = queries.dtype
        device = queries.device

        needs_conversion = (
            original_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
        )

        if needs_conversion:
            queries = queries.float()
            keys = keys.float()

        q_perm = queries.permute(0, 2, 3, 1).contiguous()
        k_perm = keys.permute(0, 2, 3, 1).contiguous()

        q_fft = torch.fft.rfft(q_perm, dim=-1)
        k_fft = torch.fft.rfft(k_perm, dim=-1)

        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        if needs_conversion:
            corr = corr.to(original_dtype)

        corr = corr.permute(0, 3, 1, 2).contiguous()

        return corr

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :, :])
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        corr = self._safe_fft_operations(queries, keys)

        values_perm = values.permute(0, 2, 3, 1).contiguous()
        V = self.time_delay_agg(values_perm, corr)
        V = V.permute(0, 3, 1, 2).contiguous()

        if self.output_attention:
            return V, corr
        else:
            return V, None


class AutoCorrelationLayer(nn.Module):
    """AutoCorrelation layer with projections."""

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, self.d_keys)
        keys = self.key_projection(keys).view(B, S, H, self.d_keys)
        values = self.value_projection(values).view(B, S, H, self.d_values)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)

        out = out.view(B, L, H * self.d_values)
        return self.out_projection(out), attn
