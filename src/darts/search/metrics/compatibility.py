import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompatibilityHelper:
    """Handles model compatibility issues and fallback routines"""

    @staticmethod
    @contextlib.contextmanager
    def safe_mode(model):
        """Temporarily disables CuDNN and replaces SDPA with manual attention"""
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        original_sdpa = getattr(F, "scaled_dot_product_attention", None)
        if original_sdpa:
            F.scaled_dot_product_attention = CompatibilityHelper._manual_attention

        try:
            yield
        finally:
            torch.backends.cudnn.enabled = prev_cudnn
            if original_sdpa:
                F.scaled_dot_product_attention = original_sdpa

    @staticmethod
    def _manual_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        """Simplified SDPA fallback using matmul and masking"""
        if enable_gqa:
            if query.size(-3) % key.size(-3) != 0:
                raise ValueError("Query heads must be divisible by key/value heads")
            key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)
        scale = query.size(-1) ** -0.5 if scale is None else scale
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale

        if is_causal:
            causal = torch.ones(
                (query.size(-2), key.size(-2)),
                dtype=torch.bool,
                device=query.device,
            ).tril()
            attn = attn.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, value)

    @staticmethod
    def prepare_data(outputs, targets):
        """Align output and target shapes for metric computation"""
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Handle sequence-to-sequence alignment
        if outputs.ndim == 3 or targets.ndim == 3:
            outputs = outputs[:, -1] if outputs.ndim == 3 else outputs
            targets = targets[:, -1] if targets.ndim == 3 else targets

        # Classification: long targets
        if targets.dtype == torch.long:
            if outputs.ndim > 2:
                outputs = outputs.reshape(outputs.size(0), -1)
            if targets.ndim == 2 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            if outputs.size(-1) == 1 and targets.max() <= 1:
                outputs = outputs.squeeze(-1)
        else:
            # Regression: force shape match
            if outputs.shape != targets.shape:
                try:
                    outputs = outputs.reshape_as(targets)
                except RuntimeError:
                    if outputs.ndim == 1:
                        outputs = outputs.unsqueeze(1)
                    if targets.ndim == 1:
                        targets = targets.unsqueeze(1)

        return outputs, targets

    @staticmethod
    def get_loss_fn(targets):
        """Select appropriate loss function based on task type"""
        return nn.CrossEntropyLoss() if targets.dtype == torch.long else nn.MSELoss()
