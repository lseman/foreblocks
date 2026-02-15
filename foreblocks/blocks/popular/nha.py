import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------
def _choose_heads_that_divide(hidden_dim: int, desired_heads: int) -> int:
    """
    Pick a number of heads that divides hidden_dim and is close to desired_heads.
    Falls back to gcd or 1 if needed.
    """
    desired_heads = max(1, int(desired_heads))
    if hidden_dim % desired_heads == 0:
        return desired_heads
    for h in range(desired_heads, 0, -1):
        if hidden_dim % h == 0:
            return h
    g = math.gcd(hidden_dim, desired_heads)
    return g if g > 0 else 1


# ---------------------------
# Attention (per level)
# ---------------------------
class HierarchicalAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with fused QKV projection.
    Supports optional causal masking.
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.causal = causal

        self.qkv = nn.Linear(input_dim, head_dim * num_heads * 3, bias=False)
        self.proj_out = nn.Linear(head_dim * num_heads, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, T, H, D]

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) * self.scale  # [B, H, T, T]

        if self.causal:
            mask = torch.triu(
                torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        ctx = attn @ v  # [B, H, T, D]
        ctx = (
            ctx.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        )
        out = self.proj_out(ctx)
        return out


# ---------------------------
# Temporal Conv (per level)
# ---------------------------
class TemporalConvLayer(nn.Module):
    """
    Causal 1D convolution with grouped conv safety, LayerNorm + GELU + Dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = True,
        dropout: float = 0.1,
        groups: int = 1,
    ):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        gcap = math.gcd(in_channels, out_channels)
        self.groups = max(1, min(groups, gcap))

        if causal:
            self.padding_left = (kernel_size - 1) * dilation
            padding = 0
        else:
            self.padding_left = 0
            padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=self.groups,
            bias=False,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x.transpose(1, 2)  # [B, C, T]
        if self.causal and self.padding_left > 0:
            x1 = F.pad(x1, (self.padding_left, 0))
        y = self.conv(x1).transpose(1, 2)  # [B, T, C_out]
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        return y


# ---------------------------
# Hierarchical Block
# ---------------------------
class HierarchicalBlock(nn.Module):
    """
    Multi-level temporal block with:
    - Multi-rate input smoothing
    - Per-level dilated conv + attention
    - Gated residual fusion
    - Optional cross-level attention
    - Backcast + feature output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_levels: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        pooling_kernel: int = 2,
        groups_base: int = 8,
        use_cross_level_attn: bool = True,
        attn_causal: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.pooling_kernel = max(1, pooling_kernel)
        self.use_cross_level_attn = use_cross_level_attn

        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        dilations = [2**i for i in range(num_levels)]
        conv_groups = max(1, hidden_dim // max(1, groups_base))

        self.temporal_convs = nn.ModuleList(
            [
                TemporalConvLayer(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dil,
                    causal=True,
                    dropout=dropout,
                    groups=conv_groups,
                )
                for dil in dilations
            ]
        )

        self.attn_layers = nn.ModuleList(
            [
                HierarchicalAttention(
                    input_dim=hidden_dim,
                    head_dim=max(8, hidden_dim // max(1, attention_heads * 2)),
                    num_heads=max(1, attention_heads // (i + 1)),
                    dropout=dropout,
                    causal=attn_causal,
                )
                for i in range(num_levels)
            ]
        )

        self.fuse_linear = nn.ModuleList(
            [
                nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
                for _ in range(num_levels)
            ]
        )

        self.level_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.backcast_proj = nn.Linear(hidden_dim, input_dim, bias=False)
        self.forecast_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if use_cross_level_attn and num_levels > 1:
            desired_heads = max(1, attention_heads // 2)
            heads = _choose_heads_that_divide(hidden_dim, desired_heads)
            self.cross_level_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
                bias=False,
            )
        else:
            self.cross_level_attn = None

    @staticmethod
    def _down_up(x: torch.Tensor, stride: int) -> torch.Tensor:
        if stride <= 1:
            return x
        B, T, C = x.shape
        xc = x.transpose(1, 2)
        xd = F.avg_pool1d(xc, kernel_size=stride, stride=stride)
        xu = F.interpolate(xd, size=T, mode="linear", align_corners=False)
        return xu.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mr = self._down_up(x, self.pooling_kernel)
        h = self.in_proj(x_mr)

        level_feats: List[torch.Tensor] = []
        cur = h

        for i in range(self.num_levels):
            conv_out = self.temporal_convs[i](cur)
            attn_out = self.attn_layers[i](conv_out)
            gate = torch.sigmoid(
                self.fuse_linear[i](torch.cat([cur, attn_out], dim=-1))
            )
            cur = self.level_norm(cur + gate * attn_out)
            cur = self.dropout(cur)
            level_feats.append(cur)

        if self.cross_level_attn is not None and len(level_feats) > 1:
            L = len(level_feats)
            stacked = torch.stack(level_feats, dim=2)  # [B, T, L, H]
            B, T, _, H = stacked.shape
            seq = stacked.view(B * T, L, H)
            attn_out, _ = self.cross_level_attn(seq, seq, seq)
            attn_out = attn_out.view(B, T, L, H)
            fused = attn_out.mean(dim=2)  # can be replaced with learned weighting later
        else:
            fused = torch.stack(level_feats, dim=0).mean(dim=0)

        backcast = self.backcast_proj(fused)
        features = self.forecast_proj(fused)

        return backcast, features


# ---------------------------
# NHA (full model)
# ---------------------------
class NHA(nn.Module):
    """
    Neural Hierarchical Architecture:
    - Stacks HierarchicalBlocks with backcast residual removal
    - Accumulates block features → final per-timestep embedding
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        num_levels_per_block: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        share_blocks: bool = False,
        pooling_kernels: Optional[List[int]] = None,
        groups_base: int = 8,
        use_cross_level_attn: bool = True,
        attn_causal: bool = False,
        accumulation_scale: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        if pooling_kernels is None:
            pooling_kernels = [2] * num_blocks
        pooling_kernels = [max(1, int(k)) for k in pooling_kernels[:num_blocks]]
        self.pooling_kernels = pooling_kernels

        if share_blocks:
            shared = HierarchicalBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_levels=num_levels_per_block,
                kernel_size=kernel_size,
                attention_heads=attention_heads,
                dropout=dropout,
                pooling_kernel=pooling_kernels[0],
                groups_base=groups_base,
                use_cross_level_attn=use_cross_level_attn,
                attn_causal=attn_causal,
            )
            self.blocks = nn.ModuleList([shared] * num_blocks)
        else:
            self.blocks = nn.ModuleList(
                [
                    HierarchicalBlock(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_levels=num_levels_per_block,
                        kernel_size=kernel_size,
                        attention_heads=max(1, attention_heads // (i + 1)),
                        dropout=dropout,
                        pooling_kernel=pooling_kernels[i],
                        groups_base=groups_base,
                        use_cross_level_attn=use_cross_level_attn,
                        attn_causal=attn_causal,
                    )
                    for i in range(num_blocks)
                ]
            )

        self.embed_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.embed_norm = nn.LayerNorm(embedding_dim)
        self.embed_drop = nn.Dropout(dropout)

        # cache for rare backcast dimension mismatches
        self._proj_cache = nn.ModuleDict()

        # scale features once at the end to prevent growth with num_blocks
        self.accumulation_scale = (
            1.0 / math.sqrt(max(1, num_blocks))
            if accumulation_scale is None
            else float(accumulation_scale)
        )

    def _get_or_create_proj(
        self, key: str, in_features: int, out_features: int, device: torch.device
    ) -> nn.Linear:
        if key not in self._proj_cache:
            self._proj_cache[key] = nn.Linear(in_features, out_features, bias=False).to(
                device
            )
        return self._proj_cache[key]

    @staticmethod
    def _match_time(backcast: torch.Tensor, target_T: int) -> torch.Tensor:
        if backcast.size(1) == target_T:
            return backcast
        bc = F.interpolate(
            backcast.transpose(1, 2), size=target_T, mode="linear", align_corners=False
        ).transpose(1, 2)
        return bc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_blocks == 0:
            # fallback: just project input
            emb = self.embed_proj(x)
            emb = self.embed_norm(emb)
            return self.embed_drop(emb)

        residual = x
        features_acc = torch.zeros_like(
            x
        )  # [B, T, input_dim] → will be projected later

        for block in self.blocks:
            backcast, feat = block(residual)

            backcast = self._match_time(backcast, residual.size(1))

            if backcast.size(-1) != residual.size(-1):
                key = f"backcast_proj_{backcast.size(-1)}_{residual.size(-1)}"
                proj = self._get_or_create_proj(
                    key, backcast.size(-1), residual.size(-1), backcast.device
                )
                backcast = proj(backcast)

            residual = residual - backcast
            features_acc = features_acc + feat

        # Scale accumulated features once (stability with many blocks)
        features_acc = features_acc * self.accumulation_scale

        emb = self.embed_proj(features_acc)
        emb = self.embed_norm(emb)
        emb = self.embed_drop(emb)
        return emb

    def get_pooled_embedding(
        self, x: torch.Tensor, pooling: str = "mean"
    ) -> torch.Tensor:
        seq_emb = self.forward(x)
        if pooling == "mean":
            return seq_emb.mean(dim=1)
        elif pooling == "max":
            return seq_emb.max(dim=1)[0]
        elif pooling == "last":
            return seq_emb[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    @torch.no_grad()
    def extract_hierarchical_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        residual = x.clone()
        features_acc = torch.zeros_like(x)

        for block in self.blocks:
            backcast, feat = block(residual)
            backcast = self._match_time(backcast, residual.size(1))
            if backcast.size(-1) != residual.size(-1):
                key = f"backcast_proj_{backcast.size(-1)}_{residual.size(-1)}"
                proj = self._get_or_create_proj(
                    key, backcast.size(-1), residual.size(-1), backcast.device
                )
                backcast = proj(backcast)
            residual = residual - backcast
            features_acc = features_acc + feat

        features_acc = features_acc * self.accumulation_scale
        seq_emb = self.embed_drop(self.embed_norm(self.embed_proj(features_acc)))

        return {
            "residuals": residual,
            "sequence_embedding": seq_emb,
        }
