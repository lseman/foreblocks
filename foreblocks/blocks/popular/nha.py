# nha.py
# Cleaned-up, safe-by-default Neural Hierarchical Architecture (NHA)
# - Fixes grouped-conv divisibility via gcd
# - Proper multi-rate (downsample → upsample) sampling
# - Uses backcast (residual removal) explicitly, N-BEATS-style
# - Lighter, consistent gating
# - Safer cross-level attention and projections
# - Shape-stable, torchscript-friendly

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Attention (per level)
# ---------------------------
class HierarchicalAttention(nn.Module):
    """
    Standard multi-head scaled dot-product attention with fused QKV projection.
    hidden_dim here is the per-head dimension. Output is [B, T, H*num_heads].
    """

    def __init__(
        self,
        input_dim: int,
        per_head_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.per_head_dim = per_head_dim
        self.scale = per_head_dim**-0.5

        self.qkv = nn.Linear(input_dim, per_head_dim * num_heads * 3, bias=False)
        self.proj_out = nn.Linear(per_head_dim * num_heads, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, H]
        returns: [B, T, H] (same last dim as input_dim)
        """
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.per_head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, T, h, d]

        q = q.transpose(1, 2)  # [B, h, T, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) * self.scale  # [B, h, T, T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        ctx = attn @ v  # [B, h, T, d]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.per_head_dim)
        out = self.proj_out(ctx)  # project back to input_dim
        return out


# ---------------------------
# Temporal Conv (per level)
# ---------------------------
class TemporalConvLayer(nn.Module):
    """
    1D convolution over time with proper causal padding and grouped-conv safety.
    LayerNorm + GELU + Dropout post-activation in [B, T, C] layout.
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

        # Ensure divisibility for grouped convs
        gcap = math.gcd(in_channels, out_channels)
        self.groups = max(1, min(groups, gcap))

        if causal:
            # we will manually left-pad to keep T length
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
        """
        x: [B, T, C_in] → y: [B, T, C_out]
        """
        x1 = x.transpose(1, 2)  # [B, C, T]
        if self.causal and self.padding_left > 0:
            x1 = F.pad(x1, (self.padding_left, 0))  # left-pad in time

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
    Multi-level temporal processing with:
      - Multi-rate sampling (downsample then upsample back to T)
      - Temporal conv + attention per level
      - Gated residual fusion
      - Optional cross-level attention
      - Backcast/forecast style: block returns (backcast, features)
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.pooling_kernel = max(1, pooling_kernel)
        self.use_cross_level_attn = use_cross_level_attn

        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # levels
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

        # attention per level; later levels can use fewer heads
        self.attn_layers = nn.ModuleList(
            [
                HierarchicalAttention(
                    hidden_dim,
                    per_head_dim=max(8, hidden_dim // max(1, attention_heads * 2)),
                    num_heads=max(1, attention_heads // (i + 1)),
                    dropout=dropout,
                )
                for i in range(num_levels)
            ]
        )

        # simple gated fusion per level: gate = sigmoid(Wg[concat(h, feat)])
        self.fuse_linear = nn.ModuleList(
            [
                nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
                for _ in range(num_levels)
            ]
        )

        self.level_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # backcast/forecast projections
        self.backcast_proj = nn.Linear(hidden_dim, input_dim, bias=False)
        self.forecast_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if use_cross_level_attn and num_levels > 1:
            self.cross_level_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=max(1, attention_heads // 2),
                dropout=dropout,
                batch_first=True,
                bias=False,
            )
        else:
            self.cross_level_attn = None

    @staticmethod
    def _down_up(x: torch.Tensor, stride: int) -> torch.Tensor:
        """
        Downsample by stride with avg_pool1d, then upsample back to original T with linear interp.
        x: [B, T, C] → same shape
        """
        if stride <= 1:
            return x
        B, T, C = x.shape
        xc = x.transpose(1, 2)  # [B, C, T]
        xd = F.avg_pool1d(xc, kernel_size=stride, stride=stride)  # [B, C, T/stride]
        xu = F.interpolate(xd, size=T, mode="linear", align_corners=False)  # [B, C, T]
        return xu.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, C_in]
        returns:
            backcast: [B, T, C_in]         (what to subtract from residual)
            features/forecast: [B, T, H]   (block features to pass forward)
        """
        # multi-rate sampling (same T on output, smoothed)
        x_mr = self._down_up(x, self.pooling_kernel)

        h = self.in_proj(x_mr)  # [B, T, H]

        level_feats: List[torch.Tensor] = []
        cur = h
        for i in range(self.num_levels):
            conv_out = self.temporal_convs[i](cur)            # [B, T, H]
            attn_out = self.attn_layers[i](conv_out)          # [B, T, H]
            # gated fusion with current state
            gate = torch.sigmoid(self.fuse_linear[i](torch.cat([cur, attn_out], dim=-1)))
            cur = self.level_norm(cur + gate * attn_out)      # residual + gate
            cur = self.dropout(cur)
            level_feats.append(cur)

        # optional cross-level attention over the level axis
        if self.cross_level_attn is not None and len(level_feats) > 1:
            # stack levels along an axis and attend across levels (treat levels as sequence)
            # shape: [B, T, L, H] → [B*T, L, H]
            L = len(level_feats)
            stacked = torch.stack(level_feats, dim=2)  # [B, T, L, H]
            B, T, _, H = stacked.shape
            seq = stacked.view(B * T, L, H)
            attn_out, _ = self.cross_level_attn(seq, seq, seq)  # [B*T, L, H]
            attn_out = attn_out.view(B, T, L, H)
            # simple mean over levels after cross-level attn
            fused = attn_out.mean(dim=2)  # [B, T, H]
        else:
            fused = torch.stack(level_feats, dim=0).mean(dim=0)  # [B, T, H]

        backcast = self.backcast_proj(fused)   # [B, T, C_in]
        forecast = self.forecast_proj(fused)   # [B, T, H] (features forward)
        return backcast, forecast


# ---------------------------
# NHA (stack of blocks)
# ---------------------------
class NHA(nn.Module):
    """
    Neural Hierarchical Architecture
      - Stacks HierarchicalBlocks
      - Uses explicit backcast subtraction (residual refinement) per block
      - Accumulates features; final per-timestep embedding via projection
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.share_blocks = share_blocks

        if pooling_kernels is None:
            # smaller stride for deeper blocks to keep more detail
            pooling_kernels = [2] + [2] * max(0, num_blocks - 1)
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
            )
            self.blocks = nn.ModuleList([shared] * num_blocks)
        else:
            self.blocks = nn.ModuleList(
                [
                    HierarchicalBlock(
                        input_dim=input_dim if i == 0 else input_dim,  # backcast always in input space
                        hidden_dim=hidden_dim,
                        num_levels=num_levels_per_block,
                        kernel_size=kernel_size,
                        attention_heads=max(1, attention_heads // (i + 1)),
                        dropout=dropout,
                        pooling_kernel=pooling_kernels[i],
                        groups_base=groups_base,
                        use_cross_level_attn=use_cross_level_attn,
                    )
                    for i in range(num_blocks)
                ]
            )

        self.embed_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.embed_norm = nn.LayerNorm(embedding_dim)
        self.embed_drop = nn.Dropout(dropout)

        # cache of linear projections for backcast size alignment (if ever needed)
        self._proj_cache = nn.ModuleDict()

    def _get_or_create_proj(
        self,
        key: str,
        in_features: int,
        out_features: int,
        device: torch.device,
    ) -> nn.Linear:
        if key not in self._proj_cache:
            self._proj_cache[key] = nn.Linear(in_features, out_features, bias=False).to(device)
        return self._proj_cache[key]

    @staticmethod
    def _match_time(backcast: torch.Tensor, target_T: int) -> torch.Tensor:
        # backcast: [B, T, C]
        if backcast.size(1) == target_T:
            return backcast
        bc = F.interpolate(
            backcast.transpose(1, 2), size=target_T, mode="linear", align_corners=False
        ).transpose(1, 2)
        return bc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C_in]
        returns: per-timestep embeddings [B, T, E]
        """
        residual = x
        features_acc = None  # [B, T, H]

        for i, block in enumerate(self.blocks):
            backcast, feat = block(residual)  # backcast: [B,T,C_in], feat: [B,T,H]

            # time align (should already match)
            backcast = self._match_time(backcast, residual.size(1))

            # channel align (should already match C_in; keep safety)
            if backcast.size(-1) != residual.size(-1):
                key = f"backcast_proj_{i}_{backcast.size(-1)}_{residual.size(-1)}"
                proj = self._get_or_create_proj(key, backcast.size(-1), residual.size(-1), backcast.device)
                backcast = proj(backcast)

            # explicit residual refinement
            residual = residual - backcast  # remove what block explained

            # accumulate features
            features_acc = feat if features_acc is None else (features_acc + feat)

        # final embedding
        emb = self.embed_proj(features_acc)  # [B, T, E]
        emb = self.embed_norm(emb)
        emb = self.embed_drop(emb)
        return emb

    def get_pooled_embedding(self, x: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
        """
        x: [B, T, C] → [B, E]
        """
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
        """
        Lightweight interpretability hooks:
          - 'residuals': final residual after all backcasts
          - 'sequence_embedding': final per-timestep embedding [B,T,E]
        """
        residual = x.clone()
        feats_acc = None
        for i, block in enumerate(self.blocks):
            backcast, feat = block(residual)
            backcast = self._match_time(backcast, residual.size(1))
            if backcast.size(-1) != residual.size(-1):
                key = f"backcast_proj_{i}_{backcast.size(-1)}_{residual.size(-1)}"
                proj = self._get_or_create_proj(key, backcast.size(-1), residual.size(-1), backcast.device)
                backcast = proj(backcast)
            residual = residual - backcast
            feats_acc = feat if feats_acc is None else feats_acc + feat

        seq_emb = self.embed_drop(self.embed_norm(self.embed_proj(feats_acc)))
        return {
            "residuals": residual,                # what remains unexplained in input space
            "sequence_embedding": seq_emb,        # final [B, T, E]
        }
# nha.py
# Cleaned-up, safe-by-default Neural Hierarchical Architecture (NHA)
# - Fixes grouped-conv divisibility via gcd
# - Proper multi-rate (downsample → upsample) sampling
# - Uses backcast (residual removal) explicitly, N-BEATS-style
# - Lighter, consistent gating
# - Safer cross-level attention and projections
# - Shape-stable, torchscript-friendly
# - Optional causal mask for attention
# - Stable feature accumulation scaling

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
    Falls back to gcd(hidden_dim, desired_heads) or 1 if needed.
    """
    desired_heads = max(1, int(desired_heads))
    if hidden_dim % desired_heads == 0:
        return desired_heads

    # Try descending from desired_heads to 1 for a divisor
    for h in range(desired_heads, 0, -1):
        if hidden_dim % h == 0:
            return h

    # Fallback (very unlikely to hit since h=1 always divides)
    g = math.gcd(hidden_dim, desired_heads)
    return g if g > 0 else 1


# ---------------------------
# Attention (per level)
# ---------------------------
class HierarchicalAttention(nn.Module):
    """
    Standard multi-head scaled dot-product attention with fused QKV projection.
    - input:  x ∈ R^{B×T×C}  (C = input_dim)
    - params: per-head dimension = head_dim, num_heads = H
    - internal: Q,K,V ∈ R^{B×T×(H·head_dim)}
    - output: same channel dimension as input_dim (projects back to C)
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        causal: bool = False,  # optional causal masking
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.causal = causal

        self.qkv = nn.Linear(input_dim, head_dim * num_heads * 3, bias=False)
        self.proj_out = nn.Linear(head_dim * num_heads, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns: [B, T, C] (same last dim as input_dim)
        """
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [B, T, H, D]

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, H, T, D]
        v = v.transpose(1, 2)  # [B, H, T, D]

        attn = (q @ k.transpose(-1, -2)) * self.scale  # [B, H, T, T]

        if self.causal:
            # Causal mask: prevent attending to future positions
            # mask shape [T, T], True above diagonal
            mask = torch.triu(
                torch.ones((T, T), device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        ctx = attn @ v  # [B, H, T, D]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.proj_out(ctx)  # [B, T, C]
        return out


# ---------------------------
# Temporal Conv (per level)
# ---------------------------
class TemporalConvLayer(nn.Module):
    """
    1D convolution over time with proper causal padding and grouped-conv safety.
    LayerNorm + GELU + Dropout post-activation in [B, T, C] layout.
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

        # Ensure divisibility for grouped convs
        gcap = math.gcd(in_channels, out_channels)
        self.groups = max(1, min(groups, gcap))

        if causal:
            # we will manually left-pad to keep T length
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
        """
        x: [B, T, C_in] → y: [B, T, C_out]
        """
        x1 = x.transpose(1, 2)  # [B, C, T]
        if self.causal and self.padding_left > 0:
            x1 = F.pad(x1, (self.padding_left, 0))  # left-pad in time

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
    Multi-level temporal processing with:
      - Multi-rate sampling (downsample then upsample back to T)
      - Temporal conv + attention per level
      - Gated residual fusion
      - Optional cross-level attention
      - Backcast/forecast style: block returns (backcast, features)
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
        attn_causal: bool = False,  # pass-through to per-level attention
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.pooling_kernel = max(1, pooling_kernel)
        self.use_cross_level_attn = use_cross_level_attn

        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # levels
        dilations = [2 ** i for i in range(num_levels)]
        # groups is #groups, not channels-per-group
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

        # attention per level; later levels can use fewer heads
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

        # simple gated fusion per level: gate = sigmoid(Wg[concat(h, feat)])
        self.fuse_linear = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim, bias=True) for _ in range(num_levels)]
        )

        self.level_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # backcast/forecast projections
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
        """
        Downsample by stride with avg_pool1d, then upsample back to original T with linear interp.
        x: [B, T, C] → same shape
        """
        if stride <= 1:
            return x
        B, T, C = x.shape
        xc = x.transpose(1, 2)  # [B, C, T]
        xd = F.avg_pool1d(xc, kernel_size=stride, stride=stride)  # [B, C, T/stride]
        xu = F.interpolate(xd, size=T, mode="linear", align_corners=False)  # [B, C, T]
        return xu.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, C_in]
        returns:
            backcast: [B, T, C_in]         (what to subtract from residual)
            features/forecast: [B, T, H]   (block features to pass forward)
        """
        # multi-rate sampling (same T on output, smoothed)
        x_mr = self._down_up(x, self.pooling_kernel)

        h = self.in_proj(x_mr)  # [B, T, H]

        level_feats: List[torch.Tensor] = []
        cur = h
        for i in range(self.num_levels):
            conv_out = self.temporal_convs[i](cur)            # [B, T, H]
            attn_out = self.attn_layers[i](conv_out)          # [B, T, H]
            # gated fusion with current state
            gate = torch.sigmoid(self.fuse_linear[i](torch.cat([cur, attn_out], dim=-1)))
            cur = self.level_norm(cur + gate * attn_out)      # residual + gate
            cur = self.dropout(cur)
            level_feats.append(cur)

        # optional cross-level attention over the level axis
        if self.cross_level_attn is not None and len(level_feats) > 1:
            # stack levels along an axis and attend across levels (treat levels as sequence)
            # shape: [B, T, L, H] → [B*T, L, H]
            L = len(level_feats)
            stacked = torch.stack(level_feats, dim=2)  # [B, T, L, H]
            B, T, _, H = stacked.shape
            seq = stacked.view(B * T, L, H)
            attn_out, _ = self.cross_level_attn(seq, seq, seq)  # [B*T, L, H]
            attn_out = attn_out.view(B, T, L, H)
            # simple mean over levels after cross-level attn
            fused = attn_out.mean(dim=2)  # [B, T, H]
        else:
            fused = torch.stack(level_feats, dim=0).mean(dim=0)  # [B, T, H]

        backcast = self.backcast_proj(fused)   # [B, T, C_in]
        forecast = self.forecast_proj(fused)   # [B, T, H] (features forward)
        return backcast, forecast


# ---------------------------
# NHA (stack of blocks)
# ---------------------------
class NHA(nn.Module):
    """
    Neural Hierarchical Architecture
      - Stacks HierarchicalBlocks
      - Uses explicit backcast subtraction (residual refinement) per block
      - Accumulates features; final per-timestep embedding via projection
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
        share_blocks: bool = False,               # True => strict weight sharing
        pooling_kernels: Optional[List[int]] = None,
        groups_base: int = 8,
        use_cross_level_attn: bool = True,
        attn_causal: bool = False,                # pass-through to per-level attention
        accumulation_scale: Optional[float] = None,  # if None, uses 1/sqrt(num_blocks)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.share_blocks = share_blocks

        if pooling_kernels is None:
            pooling_kernels = [2] + [2] * max(0, num_blocks - 1)
        pooling_kernels = [max(1, int(k)) for k in pooling_kernels[:num_blocks]]
        self.pooling_kernels = pooling_kernels

        if share_blocks:
            # Strict sharing: same module instance reused (shared weights & dropout state)
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
                        input_dim=input_dim,  # backcast always in input space
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

        # cache of linear projections for backcast size alignment (if ever needed)
        self._proj_cache = nn.ModuleDict()

        # accumulation scaling to avoid growth with #blocks
        self.accumulation_scale = (
            (1.0 / math.sqrt(max(1, num_blocks))) if accumulation_scale is None else float(accumulation_scale)
        )

    def _get_or_create_proj(
        self,
        key: str,
        in_features: int,
        out_features: int,
        device: torch.device,
    ) -> nn.Linear:
        if key not in self._proj_cache:
            self._proj_cache[key] = nn.Linear(in_features, out_features, bias=False).to(device)
        return self._proj_cache[key]

    @staticmethod
    def _match_time(backcast: torch.Tensor, target_T: int) -> torch.Tensor:
        # backcast: [B, T, C]
        if backcast.size(1) == target_T:
            return backcast
        bc = F.interpolate(
            backcast.transpose(1, 2), size=target_T, mode="linear", align_corners=False
        ).transpose(1, 2)
        return bc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C_in]
        returns: per-timestep embeddings [B, T, E]
        """
        residual = x
        features_acc = None  # [B, T, H]

        for i, block in enumerate(self.blocks):
            backcast, feat = block(residual)  # backcast: [B,T,C_in], feat: [B,T,H]

            # time align (should already match)
            backcast = self._match_time(backcast, residual.size(1))

            # channel align (should already match C_in; keep safety)
            if backcast.size(-1) != residual.size(-1):
                key = f"backcast_proj_{i}_{backcast.size(-1)}_{residual.size(-1)}"
                proj = self._get_or_create_proj(key, backcast.size(-1), residual.size(-1), backcast.device)
                backcast = proj(backcast)

            # explicit residual refinement
            residual = residual - backcast  # remove what block explained

            # accumulate features
            features_acc = feat if features_acc is None else (features_acc + feat)

        # scale accumulation for stability
        if features_acc is None:
            raise RuntimeError("NHA forward: features_acc is None (no blocks?)")
        features_acc = features_acc * self.accumulation_scale

        # final embedding
        emb = self.embed_proj(features_acc)  # [B, T, E]
        emb = self.embed_norm(emb)
        emb = self.embed_drop(emb)
        return emb

    def get_pooled_embedding(self, x: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
        """
        x: [B, T, C] → [B, E]
        """
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
        """
        Lightweight interpretability hooks:
          - 'residuals': final residual after all backcasts
          - 'sequence_embedding': final per-timestep embedding [B,T,E]
        """
        residual = x.clone()
        feats_acc = None
        for i, block in enumerate(self.blocks):
            backcast, feat = block(residual)
            backcast = self._match_time(backcast, residual.size(1))
            if backcast.size(-1) != residual.size(-1):
                key = f"backcast_proj_{i}_{backcast.size(-1)}_{residual.size(-1)}"
                proj = self._get_or_create_proj(key, backcast.size(-1), residual.size(-1), backcast.device)
                backcast = proj(backcast)
            residual = residual - backcast
            feats_acc = feat if feats_acc is None else feats_acc + feat

        feats_acc = feats_acc * self.accumulation_scale
        seq_emb = self.embed_drop(self.embed_norm(self.embed_proj(feats_acc)))
        return {
            "residuals": residual,                # what remains unexplained in input space
            "sequence_embedding": seq_emb,        # final [B, T, E]
        }
