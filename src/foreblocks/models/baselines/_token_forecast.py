"""Transformer encoder and forecast projections shared by the token-based baselines."""

import torch
import torch.nn as nn

from foreblocks.nn.normalization import create_norm_layer


class TokenEncoder(nn.Module):
    """Encode a batch of fixed-count tokens shaped [batch, tokens, width]."""

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
        layer_norm_eps: float = 1e-5,
        use_final_norm: bool = True,
    ):
        super().__init__()
        from foreblocks.nn.transformer import TransformerEncoderLayer as _EncLayer

        layer_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=dim_feedforward,
            dropout=dropout,
            activation=activation,
            attention=att_type,
            frequency_modes=freq_modes,
            swiglu=use_swiglu,
            norm_eps=layer_norm_eps,
            norm_placement=norm_strategy.removesuffix("_norm"),
            norm=custom_norm,
            moe_experts=num_experts if use_moe else 0,
            moe_top_k=top_k,
        )
        self.layers = nn.ModuleList(
            [_EncLayer(**layer_kwargs) for _ in range(n_layers)]
        )
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
            h, _ = layer(h, src_mask=None, src_key_padding_mask=None)
        return self.final_norm(h)


class TokenForecastHead(nn.Module):
    """Build and initialize the projections used by patch and variable token heads."""

    @staticmethod
    def _make_horizon_mlp(
        d_in: int, pred_len: int, hidden: int, dropout: float
    ) -> nn.Sequential:
        if hidden and hidden > 0:
            return nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, pred_len),
            )
        return nn.Sequential(nn.Linear(d_in, pred_len))

    @staticmethod
    def _make_out_proj(d_in: int, hidden: int, dropout: float) -> nn.Sequential:
        return TokenForecastHead._make_horizon_mlp(d_in, 1, hidden, dropout)

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
