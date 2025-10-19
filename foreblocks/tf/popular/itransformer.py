# itransformer_head_custom.py
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.tf.embeddings import PositionalEncoding
from foreblocks.tf.transformer_att import MultiAttention
from foreblocks.tf.norms import create_norm_layer


class _TemporalCompressor(nn.Module):
    """
    Compress each variable's full history [T] into a D-dim token.
    Two options:
      - 'linear': a single Linear(T -> D)
      - 'conv': 1D Conv over time with global pooling (lightweight, shift-aware)
    Input:  x [B, T, C]
    Output: tok [B, C, D]
    """
    def __init__(
        self,
        T: int,
        d_model: int,
        mode: Literal["linear", "conv"] = "linear",
        conv_channels: int = 0,  # if 0, will default to d_model
        conv_kernel: int = 5,
        conv_stride: int = 1,
        conv_groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if mode == "linear":
            self.proj = nn.Linear(T, d_model)
        elif mode == "conv":
            C_mid = conv_channels or d_model
            # We operate with (B*C, 1, T) to keep it simple and channel-wise
            self.conv = nn.Conv1d(1, C_mid, kernel_size=conv_kernel, stride=conv_stride, padding=conv_kernel//2, groups=1, bias=True)
            self.proj = nn.Linear(C_mid, d_model)
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown compressor mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> tokens: [B, C, D]
        B, T, C = x.shape
        if self.mode == "linear":
            # Bring time to last, project T->D per (B*C)
            xc = x.permute(0, 2, 1).contiguous().view(B * C, T)  # [B*C, T]
            tok = self.proj(xc)                                  # [B*C, D]
            tok = tok.view(B, C, -1)                             # [B, C, D]
        else:
            # conv mode: (B*C, 1, T) -> (B*C, C_mid, T) -> pool over T -> [B*C, C_mid] -> proj
            xc = x.permute(0, 2, 1).contiguous().view(B * C, 1, T)
            h = self.conv(xc)         # [B*C, C_mid, T']
            h = self.act(h)
            h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # [B*C, C_mid]
            tok = self.proj(h)        # [B*C, D]
            tok = tok.view(B, C, -1)
        return self.dropout(tok)


class VariableTokenEncoder(nn.Module):
    """
    Encoder stack over variable tokens, reusing your custom TransformerEncoderLayer.
    Expects [B, Nv, D] (Nv = #variables = channels).
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
            if use_final_norm else nn.Identity()
        )

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        # tok: [B, Nv, D]
        h = tok
        for layer in self.layers:
            # No padding mask by default (Nv is fixed per batch)
            h = layer(h, src_mask=None, src_key_padding_mask=None)  # shape preserved
        return self.final_norm(h)


class ITransformerHeadCustom(nn.Module):
    """
    iTransformer-style head using your own Transformer blocks.

    Steps:
      1) Temporal compress per variable: x[B,T,C] -> var tokens h[B,C,D]
      2) Optional [CLS] and variable positional encoding on tokens
      3) Encode with VariableTokenEncoder (custom TransformerEncoderLayer / MultiAttention)
      4) Output mixing:
         - "pooled": pool tokens ("mean" | "last" | "cls") -> MLP -> [T_pred]
         - "nonpool_linear": learn A∈R^{T_pred × N_eff}, H = A @ tokens -> per-horizon linear
         - "nonpool_attn": learned horizon queries Q∈R^{T_pred×D}, cross-attend over tokens
      5) Per-variable outputs -> [B, T_pred, C]; optional channel_mixer maps C->C_out
    """
    def __init__(
        self,
        input_len: int,
        pred_len: int,
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
        max_vars: int = 2048,
        compressor_mode: Literal["linear", "conv"] = "linear",
        compressor_conv_channels: int = 0,
        compressor_conv_kernel: int = 5,
        compressor_conv_stride: int = 1,
        instance_norm_time: bool = False,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.pooling = "cls" if use_cls_token else pooling
        self.use_cls = use_cls_token
        self.output_mode = output_mode
        self.instance_norm_time = instance_norm_time  # (optional) per-variable norm over time

        # Optional per-variable instance normalization over time
        if instance_norm_time:
            # We normalize across time per variable. Implement as LayerNorm over T via reshape.
            # (Kept simple to avoid external deps.)
            self._eps = 1e-5
        else:
            self._eps = None

        # 1) Temporal compressor: [B,T,C] -> [B,C,D]
        self.temporal = _TemporalCompressor(
            T=input_len,
            d_model=d_model,
            mode=compressor_mode,
            conv_channels=compressor_conv_channels,
            conv_kernel=compressor_conv_kernel,
            conv_stride=compressor_conv_stride,
            dropout=dropout,
        )

        # 2) Optional [CLS] + positional encoding (over variables)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_cls else None
        self.pos_enc_vars = PositionalEncoding(d_model=d_model, max_len=max_vars + (1 if self.use_cls else 0))

        # 3) Variable-token encoder stack
        self.encoder = VariableTokenEncoder(
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

        # 4) Output mixers
        if output_mode == "pooled":
            self.horizon_proj = self._make_horizon_mlp(d_model, pred_len, head_hidden, dropout)
            self.token_to_horizon = None
            self.horizon_queries = None
            self.horizon_attn = None
            self.horizon_scalar = None

        elif output_mode == "nonpool_linear":
            self.token_to_horizon = None  # lazy init (depends on N_eff)
            self._n_eff_cache = None
            self.horizon_scalar = self._make_out_proj(d_model, head_hidden, dropout)
            self.horizon_queries = None
            self.horizon_attn = None
            self.horizon_proj = None

        elif output_mode == "nonpool_attn":
            self.horizon_queries = None  # lazy init
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

        # 5) Optional channel mixer (C_in -> C_out)
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

    # ---------- utils ----------
    def _maybe_instance_norm_time(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Optional instance normalization over time per variable.
        x: [B, T, C]
        Returns x_norm, mu, sigma (to optionally denorm outside if needed).
        """
        if not self.instance_norm_time:
            return x, None, None
        mu = x.mean(dim=1, keepdim=True)                       # [B,1,C]
        var = x.var(dim=1, unbiased=False, keepdim=True)       # [B,1,C]
        sigma = torch.sqrt(var + self._eps)                    # [B,1,C]
        x_norm = (x - mu) / sigma
        return x_norm, mu, sigma

    def _init_token_to_horizon(self, n_eff: int, device: torch.device, dtype: torch.dtype):
        A = torch.empty(self.pred_len, n_eff, device=device, dtype=dtype)
        nn.init.xavier_uniform_(A)
        self.token_to_horizon = nn.Parameter(A)
        self._n_eff_cache = n_eff

    def _init_horizon_queries(self, device: torch.device, dtype: torch.dtype):
        Q = torch.randn(self.pred_len, self.d_model, device=device, dtype=dtype) * (1.0 / self.d_model ** 0.5)
        self.horizon_queries = nn.Parameter(Q)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_in, C_in]
        Returns:
            y: [B, pred_len, C_out]  (C_out=C_in if channel_mixer is None)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x [B, T, C], got {tuple(x.shape)}")
        B, T, C = x.shape
        if T != self.input_len:
            # We keep it strict; resize outside if needed
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")

        device, dtype = x.device, x.dtype

        # (Optional) per-variable instance norm over time
        x, _, _ = self._maybe_instance_norm_time(x)  # we can expose mu/sigma if you want reversible norm

        # 1) Compress each variable's history -> tokens per variable
        # x: [B,T,C] -> tok: [B,C,D]
        tok = self.temporal(x)  # [B, C, D]

        # 2) Optional CLS + positional encoding over variables (token index)
        if self.use_cls:
            cls = self.cls_token.expand(tok.size(0), 1, -1)  # [B, 1, D]
            tok = torch.cat([cls, tok], dim=1)               # [B, C+1, D]
            N_eff = C + 1
        else:
            N_eff = C

        tok = self.pos_enc_vars(tok)             # [B, N_eff, D]

        # 3) Encode over variables
        h = self.encoder(tok)                    # [B, N_eff, D]

        # 4) Output mixing -> [B, pred_len, C]
        if self.output_mode == "pooled":
            if self.use_cls or self.pooling == "cls":
                pooled = h[:, 0, :]             # [B, D]
                # Global pooled token -> same horizon for all variables, then broadcast and optionally mix channels
                y_t = self.horizon_proj(self.dropout(pooled))  # [B, T_pred]
                y = y_t.unsqueeze(-1).expand(B, self.pred_len, C)  # [B, T_pred, C]
            elif self.pooling == "last":
                pooled = h[:, -1, :]            # [B, D]
                y_t = self.horizon_proj(self.dropout(pooled))
                y = y_t.unsqueeze(-1).expand(B, self.pred_len, C)
            else:  # mean
                pooled = h.mean(dim=1)          # [B, D]
                y_t = self.horizon_proj(self.dropout(pooled))
                y = y_t.unsqueeze(-1).expand(B, self.pred_len, C)

        elif self.output_mode == "nonpool_linear":
            # Lazy init for token-to-horizon A (T_pred × N_eff)
            if self.token_to_horizon is None or self._n_eff_cache != N_eff:
                self._init_token_to_horizon(N_eff, device, dtype)

            # H: [B, T_pred, D] by mixing tokens -> then per-variable scalar per horizon
            # But we want per-variable outputs. Strategy:
            #   Compute per-variable hidden for each horizon with A @ tokens first, preserving token dimension,
            #   then apply a per-token scalar head. A cleaner: project tokens -> scalar per horizon directly.
            # We'll do: mix tokens into horizon embeddings, then per-variable scalar via linear on corresponding token.
            # Simpler and efficient approach:
            #   For each horizon t, mix tokens to a horizon-context vector, then predict each variable via a shared MLP applied on (horizon-context + token).
            # We’ll produce horizon-wise token embeddings first:
            H = torch.einsum("tn,bnd->btd", self.token_to_horizon, h)  # [B, T_pred, D]
            H = self.dropout(H)

            # Now predict each variable with the same scalar head applied to per-variable token features gated by H.
            # A simple and effective compositional rule is elementwise fusion (FiLM-like): y_bc = W([H ⊙ tok_i]) or W([H + tok_i]).
            # Use additive fusion for stability:
            tok_no_cls = h[:, -C:, :] if self.use_cls else h  # [B, C, D]
            H_exp = H.unsqueeze(2).expand(B, self.pred_len, C, self.d_model)      # [B, T_pred, C, D]
            Tok_exp = tok_no_cls.unsqueeze(1).expand(B, self.pred_len, C, self.d_model)  # [B, T_pred, C, D]
            fused = H_exp + Tok_exp
            y = self.horizon_scalar(fused).squeeze(-1)  # [B, T_pred, C]

        else:  # "nonpool_attn"
            # Lazy init horizon queries
            if self.horizon_queries is None:
                self._init_horizon_queries(device, dtype)

            Q = self.horizon_queries.unsqueeze(0).expand(B, -1, -1)  # [B, T_pred, D]
            # Cross-attend horizons over variable tokens
            H, _, _ = self.horizon_attn(Q, h, h)  # [B, T_pred, D]
            H = self.dropout(H)

            # Per-variable scalar via shared head over (H + token_i)
            tok_no_cls = h[:, -C:, :] if self.use_cls else h  # [B, C, D]
            H_exp = H.unsqueeze(2).expand(B, self.pred_len, C, self.d_model)      # [B, T_pred, C, D]
            Tok_exp = tok_no_cls.unsqueeze(1).expand(B, self.pred_len, C, self.d_model)
            fused = H_exp + Tok_exp
            y = self.horizon_scalar(fused).squeeze(-1)  # [B, T_pred, C]

        # Optional channel mixer (C_in -> C_out)
        if self.channel_mixer is not None:
            y = self.channel_mixer(y)  # [B, T_pred, C_out]

        return y
