# dlinear_head_custom.py
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import foreblocks.tf.experts.moe as txmoe
import foreblocks.tf.norms as txaux

# Use your project modules
from foreblocks.tf.embeddings import PositionalEncoding
from foreblocks.tf.attention.multi_att import MultiAttention
from foreblocks.tf.norms import create_norm_layer


def _patchify_1d(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    x: [B, L] -> patches: [B, N, P]
    If L < P, left-pad to make at least one patch.
    Uses as_strided (view) for speed.
    """
    B, L = x.shape
    if L < patch_len:
        x = F.pad(x, (patch_len - L, 0))  # left pad
        L = x.size(1)
    N = 1 + (L - patch_len) // stride
    s0, s1 = x.stride()
    return x.as_strided(size=(B, N, patch_len), stride=(s0, s1 * stride, s1))


class PatchTokenEncoder(nn.Module):
    """
    A lightweight encoder stack for patch tokens that reuses your custom
    TransformerEncoderLayer (MultiAttention + NormWrapper + FF/MoE).

    Expects inputs: [B, N, D] (tokens).
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "swiglu",
        att_type: str = "standard",
        freq_modes: int = 32,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        layer_norm_eps: float = 1e-5,
        use_final_norm: bool = True,
    ):
        super().__init__()
        from foreblocks.tf.transformer import TransformerEncoderLayer as _EncLayer

        layer_kwargs = dict(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type=att_type,
            freq_modes=freq_modes,
            use_swiglu=use_swiglu,
            layer_norm_eps=layer_norm_eps,
            norm_strategy=norm_strategy,
            custom_norm=custom_norm,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_capacity_factor=moe_capacity_factor,
        )
        self.layers = nn.ModuleList([_EncLayer(**layer_kwargs) for _ in range(n_layers)])
        self.final_norm = (
            create_norm_layer(custom_norm, d_model, layer_norm_eps)
            if use_final_norm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        h = x
        for layer in self.layers:
            # We pass None masks for token encoder (no padding by default)
            h = layer(h, src_mask=None, src_key_padding_mask=None)
        return self.final_norm(h)


class PatchTSTHeadCustom(nn.Module):
    """
    PatchTST-style head built on your custom attention stack.

    Input:
        x : [B, L, C_in]
    Output:
        y : [B, pred_len, C_out]  (C_out=C_in if channel_mixer is None)

    Steps per channel:
      1) Patchify [B, L] -> [B, N, P]
      2) Linear patch embedding [P] -> [D]
      3) (Optional) prepend [CLS] then positional encoding over tokens
      4) Encode with PatchTokenEncoder (uses your TransformerEncoderLayer + MultiAttention)
      5) Output mixing (choose via `output_mode`):
         - "pooled": pool tokens ("mean" | "last" | "cls"), then MLP -> [T]
         - "nonpool_linear": learned A∈R^{T×N_eff}, H=A@tokens -> per-horizon linear -> [T]
         - "nonpool_attn": learned horizon queries Q∈R^{T×D} with your MultiAttention (cross-attn)
      6) Reshape back to [B, T, C_in]; optional channel_mixer maps C_in->C_out
    """

    def __init__(
        self,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        freq_modes: int = 32,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        layer_norm_eps: float = 1e-5,
        use_final_norm: bool = True,
        pooling: Literal["mean", "last", "cls"] = "mean",
        use_cls_token: bool = False,
        head_hidden: int = 0,
        output_mode: Literal["pooled", "nonpool_linear", "nonpool_attn"] = "pooled",
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        max_patches: int = 10_000,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.pooling = "cls" if use_cls_token else pooling
        self.use_cls = use_cls_token
        self.output_mode = output_mode

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Optional [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_cls else None

        # Positional encoding for tokens
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_patches + (1 if self.use_cls else 0))

        # Token encoder stack using your custom layers
        self.encoder = PatchTokenEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type=att_type,
            freq_modes=freq_modes,
            norm_strategy=norm_strategy,
            custom_norm=custom_norm,
            use_swiglu=use_swiglu,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_capacity_factor=moe_capacity_factor,
            layer_norm_eps=layer_norm_eps,
            use_final_norm=use_final_norm,
        )

        # ---- Output mixing heads ----
        if output_mode == "pooled":
            self.horizon_proj = self._make_horizon_mlp(d_model, pred_len, head_hidden, dropout)
            self.token_to_horizon = None
            self.horizon_queries = None
            self.horizon_attn = None
            self.horizon_scalar = None

        elif output_mode == "nonpool_linear":
            # Will be lazily initialized on first forward pass
            self.token_to_horizon = None
            self._n_eff_cache = None
            self.horizon_scalar = self._make_out_proj(d_model, head_hidden, dropout)
            self.horizon_proj = None
            self.horizon_queries = None
            self.horizon_attn = None

        elif output_mode == "nonpool_attn":
            # Will be lazily initialized on first forward pass
            self.horizon_queries = None
            self.horizon_attn = MultiAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                attention_type=att_type,
                freq_modes=freq_modes,
                cross_attention=True,
            )
            self.horizon_scalar = self._make_out_proj(d_model, head_hidden, dropout)
            self.token_to_horizon = None
            self.horizon_proj = None
        else:
            raise ValueError(f"Unknown output_mode={output_mode}")

        # Optional channel mixer (C_in -> C_out)
        self.channel_mixer = None
        if in_channels is not None and out_channels is not None and out_channels != in_channels:
            self.channel_mixer = nn.Linear(in_channels, out_channels, bias=True)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    # ---------- builders & init ----------

    @staticmethod
    def _make_horizon_mlp(d_in: int, pred_len: int, hidden: int, dropout: float) -> nn.Sequential:
        if hidden and hidden > 0:
            return nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, pred_len),
            )
        return nn.Sequential(nn.Linear(d_in, pred_len))

    @staticmethod
    def _make_out_proj(d_in: int, hidden: int, dropout: float) -> nn.Sequential:
        # D -> 1 with optional hidden
        if hidden and hidden > 0:
            return nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        return nn.Sequential(nn.Linear(d_in, 1))

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _init_token_to_horizon(self, n_eff: int, device: torch.device, dtype: torch.dtype):
        """Initialize the learnable token-to-horizon projection matrix."""
        A = torch.empty(self.pred_len, n_eff, device=device, dtype=dtype)
        nn.init.xavier_uniform_(A)
        self.token_to_horizon = nn.Parameter(A)
        self._n_eff_cache = n_eff

    def _init_horizon_queries(self, device: torch.device, dtype: torch.dtype):
        """Initialize the learnable horizon query embeddings."""
        Q = torch.randn(self.pred_len, self.d_model, device=device, dtype=dtype)
        Q = Q * (1.0 / self.d_model**0.5)
        self.horizon_queries = nn.Parameter(Q)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B, L, C], got {tuple(x.shape)}")
        B, L, C = x.shape

        # Work in input dtype/device throughout
        device = x.device
        dtype = x.dtype

        # Channel-independent: (B*C, L)
        x_bc = x.permute(0, 2, 1).contiguous().view(B * C, L)

        # Patchify -> [B*C, N, P]
        patches = _patchify_1d(x_bc, self.patch_len, self.stride)
        N = patches.size(1)

        # Patch embedding -> [B*C, N, D]
        tok = self.patch_embed(patches)

        # Optional CLS
        if self.use_cls:
            cls = self.cls_token.expand(tok.size(0), 1, -1)
            tok = torch.cat([cls, tok], dim=1)
            N_eff = N + 1
        else:
            N_eff = N

        # Positional encoding + custom encoder stack
        tok = self.pos_enc(tok)
        tok = self.encoder(tok)  # [B*C, N_eff, D]

        if self.output_mode == "pooled":
            if self.use_cls or self.pooling == "cls":
                pooled = tok[:, 0, :]
            elif self.pooling == "last":
                pooled = tok[:, -1, :]
            else:
                pooled = tok.mean(dim=1)
            pooled = self.dropout(pooled)
            y_bc = self.horizon_proj(pooled)  # [B*C, T]
            y = y_bc.view(B, C, self.pred_len).permute(0, 2, 1).contiguous()

        elif self.output_mode == "nonpool_linear":
            # Lazy initialization of token_to_horizon if needed
            if self.token_to_horizon is None or self._n_eff_cache != N_eff:
                self._init_token_to_horizon(N_eff, device, dtype)
            
            H = torch.einsum("tn,bnd->btd", self.token_to_horizon, tok)  # [B*C, T, D]
            H = self.dropout(H)
            y_bc = self.horizon_scalar(H).squeeze(-1)  # [B*C, T]
            y = y_bc.view(B, C, self.pred_len).permute(0, 2, 1).contiguous()

        else:  # "nonpool_attn"
            # Lazy initialization of horizon_queries if needed
            if self.horizon_queries is None:
                self._init_horizon_queries(device, dtype)

            Q = self.horizon_queries.unsqueeze(0).expand(tok.size(0), -1, -1)  # [B*C, T, D]
            H, _, _ = self.horizon_attn(Q, tok, tok)  # [B*C, T, D]
            H = self.dropout(H)
            y_bc = self.horizon_scalar(H).squeeze(-1)  # [B*C, T]
            y = y_bc.view(B, C, self.pred_len).permute(0, 2, 1).contiguous()

        if self.channel_mixer is not None:
            y = self.channel_mixer(y)  # [B, T, C_out]

        return y
