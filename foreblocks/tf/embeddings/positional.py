import math
from typing import Dict, Optional, Tuple

# patches.py
# --- patches.py (new helper) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# ------------------------- Triton availability -------------------------
try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    triton = None
    tl = None


# =============================== PositionalEncoding (Sinusoidal) ===============================


# =============================== Sinusoidal PositionalEncoding (patched) ===============================
class PositionalEncoding(nn.Module):
    """
    Sinusoidal PE with caching + scale. Patched: when 'pos' is provided,
    we ALWAYS ensure the table covers the maximum index.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000, scale: float = 1.0, cache_limit: int = 8):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scale = scale
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.cache_limit = cache_limit

        # prebuild base table for (d_model, max_len)
        self.register_buffer("pe", self._build_table(d_model, max_len), persistent=False)  # [1,T,D]
        self._pe_cache: Dict[int, torch.Tensor] = {}  # d_model -> [1,T,D]

    @staticmethod
    def _build_table(D: int, T: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        pos = torch.arange(T, dtype=dtype, device=device).unsqueeze(1)       # [T,1]
        div = torch.exp(torch.arange(0, D, 2, dtype=dtype, device=device) * (-math.log(10000.0) / D))  # [D/2]
        pe = torch.empty(T, D, dtype=dtype, device=device)
        ang = pos * div  # [T, D/2]
        pe[:, 0::2] = torch.sin(ang)
        if D % 2 == 1:
            pe[:, 1::2] = torch.cos(ang[:, :-1]); pe[:, -1] = 0
        else:
            pe[:, 1::2] = torch.cos(ang)
        return pe.unsqueeze(0)  # [1,T,D]

    def _ensure_table(self, D: int, T: int, device: torch.device) -> torch.Tensor:
        if D == self.d_model and self.pe.size(1) >= T:
            return self.pe[:, :T].to(device=device)
        # cache per D
        cached = self._pe_cache.get(D, None)
        if cached is None or cached.size(1) < T:
            tbl = self._build_table(D, max(T, 8), device=device)  # at least 8
            if D <= 2048:
                if len(self._pe_cache) >= self.cache_limit:
                    self._pe_cache.pop(next(iter(self._pe_cache)))
                self._pe_cache[D] = tbl
            return tbl[:, :T]
        return cached[:, :T].to(device=device)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if pos is not None:
            # normalize pos to [B,T]
            if pos.dim() == 1:
                pos = pos.unsqueeze(0).expand(B, -1)
            max_idx = int(pos.max().item())
            base = self._ensure_table(D, max_idx + 1, device).squeeze(0)      # [Tmax,D]
            pe = F.embedding(pos.to(device=device, dtype=torch.long), base)   # [B,T,D]
        else:
            pe = self._ensure_table(D, T, device).expand(B, -1, -1)           # [B,T,D]

        pe = pe.to(dtype)
        out = x + pe * self.scale
        return self.dropout(out) if self.dropout else out


# =============================== InformerTimeEmbedding ===============================
class InformerTimeEmbedding(nn.Module):
    """Compact time embedding with projection, sane init, and simple normalization."""

    def __init__(self, d_model: int):
        super().__init__()
        embed_dim = min(d_model // 4, 64)
        self.hour_embed = nn.Embedding(24, embed_dim)
        self.weekday_embed = nn.Embedding(7, embed_dim)
        self.day_embed = nn.Embedding(32, embed_dim)
        self.month_embed = nn.Embedding(13, embed_dim)
        self.projection = nn.Linear(embed_dim * 4, d_model) if embed_dim * 4 != d_model else None
        self.norm_factor = 1.0 / math.sqrt(4.0)
        # init
        for emb in [self.hour_embed, self.weekday_embed, self.day_embed, self.month_embed]:
            nn.init.normal_((emb.weight), mean=0.0, std=0.02)

    def forward(self, time_feats: torch.Tensor) -> torch.Tensor:
        """
        time_feats: [B, T, 4] -> [month, weekday, hour, day]
        """
        month = torch.clamp(time_feats[..., 0].long(), 0, 12)
        weekday = torch.clamp(time_feats[..., 1].long(), 0, 6)
        hour = torch.clamp(time_feats[..., 2].long(), 0, 23)
        day = torch.clamp(time_feats[..., 3].long(), 0, 31)
        embs = torch.cat(
            [self.month_embed(month), self.weekday_embed(weekday),
             self.hour_embed(hour), self.day_embed(day)],
            dim=-1
        )
        if self.projection is not None:
            embs = self.projection(embs)
        return embs * self.norm_factor


# =============================== LearnablePositionalEncoding (patched) ===============================
class LearnablePositionalEncoding(nn.Module):
    """
    Learnable PE with optional low-rank factors. Patched with optional per-head scaling.
    Use 'per_head_scale=True' to give each hidden dim its own learnable multiplier.
    """
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        initialization: str = "normal",
        scale_strategy: str = "fixed",  # ["fixed", "learnable", "none"]
        scale_value: Optional[float] = None,
        use_layer_norm: bool = True,
        norm_strategy: str = "pre_add",  # or "post_add"
        low_rank_dim: Optional[int] = None,
        per_head_scale: bool = False,    # NEW
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.low_rank_dim = low_rank_dim
        self.norm_strategy = norm_strategy

        # parameters (standard or low-rank)
        if low_rank_dim is None:
            self.pe = nn.Parameter(self._init_pe(initialization, (max_len, d_model)))  # [T,D]
        else:
            self.U = nn.Parameter(self._init_pe(initialization, (max_len, low_rank_dim)))  # [T,r]
            self.V = nn.Parameter(self._init_pe(initialization, (low_rank_dim, d_model)))  # [r,D]

        # scaling
        if scale_strategy == "learnable":
            init_scale = scale_value or math.sqrt(d_model)
            self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        elif scale_strategy == "fixed":
            self.register_buffer("scale", torch.tensor(scale_value or math.sqrt(d_model), dtype=torch.float32))
        else:
            self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))

        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.per_head_scale = nn.Parameter(torch.ones(d_model)) if per_head_scale else None
        self._cache: Dict[int, torch.Tensor] = {}  # T -> [1,T,D]

    def _init_pe(self, mode: str, shape: tuple) -> torch.Tensor:
        if mode == "normal":
            return torch.randn(shape) * math.sqrt(2.0 / shape[-1])
        if mode == "uniform":
            bound = math.sqrt(6.0 / shape[-1]); return torch.empty(shape).uniform_(-bound, bound)
        if mode == "zero":
            return torch.zeros(shape)
        return torch.randn(shape) * 0.02

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if self.low_rank_dim is None:
            if positions is None:
                if T not in self._cache:
                    self._cache[T] = self.pe[:T].unsqueeze(0)  # [1,T,D]
                pe = self._cache[T].to(device=device)
            else:
                pe = F.embedding(positions.to(device=device, dtype=torch.long), self.pe)  # [B,T,D]
        else:
            if positions is None:
                pe = (self.U[:T] @ self.V).unsqueeze(0)  # [1,T,D]
                pe = pe.expand(B, -1, -1)
            else:
                Usel = F.embedding(positions.to(device=device, dtype=torch.long), self.U)  # [B,T,r]
                pe = torch.bmm(Usel, self.V.unsqueeze(0).expand(B, -1, -1))                # [B,T,D]

        pe = pe.to(dtype)
        if self.per_head_scale is not None:
            pe = pe * self.per_head_scale.to(dtype=dtype, device=device)

        if self.layer_norm and self.norm_strategy == "pre_add":
            x = self.layer_norm(x)

        x = x + pe * (self.scale.to(dtype=dtype, device=device) if isinstance(self.scale, torch.Tensor) else float(self.scale))

        if self.layer_norm and self.norm_strategy == "post_add":
            x = self.layer_norm(x)
        return self.dropout(x) if self.dropout else x


# =============================== Patch Embeddings (single definitions) ===============================
class PatchEmbedding(nn.Module):
    """
    PatchTST-style temporal patching:
      Input:  x [B, T, C] -> Output: [B, T_p, D]
      T_p = floor((T - patch_len) / stride) + 1
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_len: int = 16, patch_stride: int = 8, bias: bool = True):
        super().__init__()
        assert patch_len >= 1 and patch_stride >= 1
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_len, stride=patch_stride, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # [B,C,T]
        y = self.proj(x).transpose(1, 2)  # [B,Tp,D]
        return y

    def output_len(self, T: int) -> int:
        return 0 if T < self.patch_len else (T - self.patch_len) // self.patch_stride + 1


class CIPatchEmbedding(nn.Module):
    """
    Channel-Independent patching with grouped convs.
    Input:  [B,T,C] -> [B,T_p,C,D]
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_len: int = 16, patch_stride: int = 8, bias: bool = True):
        super().__init__()
        assert patch_len >= 1 and patch_stride >= 1
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.proj = nn.Conv1d(in_channels, in_channels * embed_dim, kernel_size=patch_len, stride=patch_stride,
                              padding=0, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)              # [B,C,T]
        y = self.proj(x)                   # [B,C*D,Tp]
        B, CD, Tp = y.shape
        C, D = self.in_channels, self.embed_dim
        return y.view(B, C, D, Tp).permute(0, 3, 1, 2).contiguous()  # [B,Tp,C,D]

    def output_len(self, T: int) -> int:
        return 0 if T < self.patch_len else (T - self.patch_len) // self.patch_stride + 1
