# fedformer_encoder_decoder_custom.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Your blocks
from foreblocks.tf.transformer import TransformerDecoder, TransformerEncoder
from foreblocks.tf.transformer_aux import create_norm_layer


# ----------------------------
# Series decomposition (centered moving average)
# ----------------------------
class SeriesDecompMovingAvg(nn.Module):
    """
    Per-channel moving average via depthwise Conv1d.
    Input:  x [B, T, C]
    Output: seasonal [B, T, C], trend [B, T, C]
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        assert kernel_size > 0 and kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        xc = x.permute(0, 2, 1).contiguous()  # [B, C, T]
        weight = x.new_ones(C, 1, self.kernel_size) / float(self.kernel_size)
        trend = F.conv1d(xc, weight=weight, bias=None, stride=1, padding=self.pad, groups=C)
        trend = trend.permute(0, 2, 1).contiguous()   # [B, T, C]
        seasonal = x - trend
        return seasonal, trend


# ----------------------------
# FEDFormer Encoder (separate)
# ----------------------------
class FEDFormerEncoderCustom(nn.Module):
    """
    Encoder side of FEDFormer:
      - Decompose x into seasonal + trend
      - Encode seasonal with frequency attention
    Returns:
      memory [B, T_enc, d_model], seasonal_enc [B, T_enc, C], trend_enc [B, T_enc, C]
    """
    def __init__(
        self,
        enc_in: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers_enc: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        freq_modes: int = 32,
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        use_final_norm: bool = True,
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        max_seq_len: int = 5000,
        pos_encoding_scale: float = 1.0,
        use_time_encoding_enc: bool = True,
        decomp_kernel: int = 25,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.use_time_encoding_enc = use_time_encoding_enc
        self.decomp = SeriesDecompMovingAvg(decomp_kernel)

        self.encoder = TransformerEncoder(
            input_size=enc_in,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_enc,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type="freq",
            freq_modes=freq_modes,
            layer_norm_eps=layer_norm_eps,
            norm_strategy=norm_strategy,
            custom_norm=custom_norm,
            max_seq_len=max_seq_len,
            pos_encoding_scale=pos_encoding_scale,
            use_final_norm=use_final_norm,
            use_swiglu=use_swiglu,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_capacity_factor=moe_capacity_factor,
        )

    def forward(
        self,
        x_enc: torch.Tensor,                       # [B, T_enc, C]
        x_mark_enc: Optional[torch.Tensor] = None  # optional time features
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_enc.dim() != 3:
            raise ValueError(f"Expected x_enc [B, T, C], got {tuple(x_enc.shape)}")
        if x_enc.size(-1) != self.enc_in:
            raise ValueError(f"enc_in={self.enc_in} but got C={x_enc.size(-1)}")

        # Decompose
        seasonal_enc, trend_enc = self.decomp(x_enc)  # [B, T, C]

        # Encode seasonal with freq attention
        enc_time = x_mark_enc if self.use_time_encoding_enc else None
        memory = self.encoder(
            seasonal_enc, src_mask=None, src_key_padding_mask=None, time_features=enc_time
        )  # [B, T_enc, d_model]
        return memory, seasonal_enc, trend_enc


# ----------------------------
# FEDFormer Decoder (separate)
# ----------------------------
class FEDFormerDecoderCustom(nn.Module):
    """
    Decoder side of FEDFormer:
      - Build seasonal_dec = seasonal_label + zeros(pred_len)
      - Build trend_dec    = trend_label + repeat(last_trend, pred_len)
      - Decode seasonal with freq attention + cross-attn to memory
      - Project seasonal -> C_out and add trend horizon
    """
    def __init__(
        self,
        dec_in: int,
        c_out: int,
        label_len: int,
        pred_len: int,

        d_model: int = 512,
        nhead: int = 8,
        num_layers_dec: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        freq_modes: int = 32,
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        use_final_norm: bool = True,
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        max_seq_len: int = 5000,
        pos_encoding_scale: float = 1.0,
        use_time_encoding_dec: bool = True,
        informer_like: bool = True,   # Informer-style masking (non-causal)
    ):
        super().__init__()
        self.dec_in = dec_in
        self.c_out = c_out
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_time_encoding_dec = use_time_encoding_dec

        self.decoder = TransformerDecoder(
            input_size=dec_in,
            output_size=d_model,        # keep in model space; project after
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_dec,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type="freq",
            freq_modes=freq_modes,
            layer_norm_eps=layer_norm_eps,
            norm_strategy=norm_strategy,
            custom_norm=custom_norm,
            max_seq_len=max_seq_len,
            pos_encoding_scale=pos_encoding_scale,
            use_final_norm=use_final_norm,
            use_swiglu=use_swiglu,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_capacity_factor=moe_capacity_factor,
            informer_like=informer_like,
            use_time_encoding=use_time_encoding_dec,
        )

        # Seasonal path projection and channel alignment for trend if needed
        self.proj_seasonal = nn.Linear(d_model, c_out)
        self.trend_mixer = (
            nn.Identity() if c_out == dec_in else nn.Linear(dec_in, c_out, bias=True)
        )

    @staticmethod
    def build_decoder_inputs(
        seasonal_enc: torch.Tensor, trend_enc: torch.Tensor, label_len: int, pred_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct decoder inputs:
          seasonal_dec = seasonal_last_label_len + zeros(pred_len)
          trend_dec    = trend_last_label_len + repeat(last_trend, pred_len)
        Returns (seasonal_dec, trend_dec) both [B, label_len+pred_len, C]
        """
        B, T, C = seasonal_enc.shape
        ll, pl = label_len, pred_len
        seasonal_label = seasonal_enc[:, -ll:, :]
        seasonal_pred_init = seasonal_enc.new_zeros(B, pl, C)
        seasonal_dec = torch.cat([seasonal_label, seasonal_pred_init], dim=1)

        trend_label = trend_enc[:, -ll:, :]
        last_trend = trend_enc[:, -1:, :].expand(B, pl, C)
        trend_dec = torch.cat([trend_label, last_trend], dim=1)
        return seasonal_dec, trend_dec

    def forward(
        self,
        memory: torch.Tensor,                     # [B, T_enc, d_model]
        seasonal_enc: torch.Tensor,               # [B, T_enc, C]
        trend_enc: torch.Tensor,                  # [B, T_enc, C]
        x_mark_dec: Optional[torch.Tensor] = None # optional time features for decoder horizon
    ) -> torch.Tensor:
        if seasonal_enc.size(-1) != self.dec_in or trend_enc.size(-1) != self.dec_in:
            raise ValueError(
                f"dec_in={self.dec_in} but got seasonal C={seasonal_enc.size(-1)} "
                f"and trend C={trend_enc.size(-1)}"
            )

        # Build decoder inputs
        seasonal_dec, trend_dec = self.build_decoder_inputs(
            seasonal_enc, trend_enc, self.label_len, self.pred_len
        )  # [B, ll+pl, C]

        # Decode seasonal with cross-attention to encoder memory
        dec_time = x_mark_dec if self.use_time_encoding_dec else None
        seasonal_dec_out = self.decoder(
            seasonal_dec, memory,
            tgt_mask=None, memory_mask=None,
            tgt_key_padding_mask=None, memory_key_padding_mask=None,
            incremental_state=None, return_incremental_state=False,
            time_features=dec_time,
        )  # [B, ll+pl, d_model]

        # Project seasonal to channels and slice pred horizon
        seasonal_hat = self.proj_seasonal(seasonal_dec_out[:, -self.pred_len:, :])  # [B, pred_len, C_out]

        # Trend horizon (align channels if needed)
        trend_h = trend_dec[:, -self.pred_len:, :]                                   # [B, pred_len, C]
        trend_hat = self.trend_mixer(trend_h)                                        # [B, pred_len, C_out]

        # Final: seasonal + trend
        y_hat = seasonal_hat + trend_hat
        return y_hat
