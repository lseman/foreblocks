# tft_head_custom.py
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import InformerTimeEmbedding, PositionalEncoding
from .transformer import (
    TransformerDecoder,  # your custom classes
    TransformerEncoder,
)
from .transformer_att import MultiAttention
from .transformer_aux import create_norm_layer

# ---------------------------
# Core TFT building blocks
# ---------------------------

class GLU(nn.Module):
    def __init__(self, d: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d, 2 * d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, g = self.proj(x).chunk(2, dim=-1)
        return self.dropout(torch.sigmoid(g) * h)


class GRN(nn.Module):
    """
    Gated Residual Network (TFT). Supports optional context concat.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0, custom_norm: str = "rms", eps: float = 1e-5):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.elu = nn.ELU()
        self.glu = GLU(d_out, dropout=dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = create_norm_layer(custom_norm, d_out, eps)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        x: [B, *, d_in]
        context (optional): [B, *, d_in] or broadcastable to x
        """
        y = x if context is None else torch.cat([x, context], dim=-1)
        h = self.fc2(self.elu(self.fc1(y)))
        z = self.glu(h)
        return self.norm(self.skip(x) + z)


class VariableSelectionNetwork(nn.Module):
    """
    Per-time-step variable selection with learned embeddings per variable.
    Follows TFT: computes importance weights and gated transformations for each variable,
    then weighted sum across variables.
    """
    def __init__(
        self,
        num_vars: int,
        d_in_per_var: int,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.0,
        custom_norm: str = "rms",
        eps: float = 1e-5,
        name: str = "VSN",
    ):
        super().__init__()
        self.num_vars = num_vars
        self.name = name

        # Per-variable encoders map each var feature to d_model
        self.var_encoders = nn.ModuleList([nn.Linear(d_in_per_var, d_model) for _ in range(num_vars)])

        # Gating: compute weights over variables
        # Aggregator input is concatenated encodings [B, T, num_vars * d_model]
        self.grn_gates = GRN(d_in=num_vars * d_model, d_hidden=d_hidden, d_out=num_vars, dropout=dropout, custom_norm=custom_norm, eps=eps)

        # Post-selection transformation of each variable to d_model (shared GRN)
        self.grn_feat = GRN(d_in=d_model, d_hidden=d_hidden, d_out=d_model, dropout=dropout, custom_norm=custom_norm, eps=eps)

    def forward(self, x_vars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_vars: [B, T, num_vars, d_in_per_var]
        Returns:
          z: [B, T, d_model] fused features
          attn_w: [B, T, num_vars] variable weights (softmax)
        """
        B, T, V, Dv = x_vars.shape
        assert V == self.num_vars

        # Encode per variable to d_model
        encs = []
        for i in range(V):
            e = self.var_encoders[i](x_vars[:, :, i, :])  # [B, T, d_model]
            encs.append(e)
        E = torch.stack(encs, dim=2)  # [B, T, V, d_model]

        # Compute selection weights
        gates_in = E.reshape(B, T, V * E.size(-1))       # [B, T, V*d_model]
        logits = self.grn_gates(gates_in)                # [B, T, V]
        attn_w = torch.softmax(logits, dim=-1)           # [B, T, V]

        # Weighted sum of transformed features
        E_tr = self.grn_feat(E)                          # [B, T, V, d_model]
        z = torch.einsum("btv,btvd->btd", attn_w, E_tr)  # [B, T, d_model]
        return z, attn_w


class GateAddNorm(nn.Module):
    """
    Gated residual connection + normalization around a sublayer.
    """
    def __init__(self, d_model: int, dropout: float = 0.0, custom_norm: str = "rms", eps: float = 1e-5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_model, dropout=0.0)
        self.norm = create_norm_layer(custom_norm, d_model, eps)

    def forward(self, x, sub_out):
        return self.norm(x + self.glu(self.dropout(sub_out)))


# ---------------------------
# Temporal Fusion Transformer Head
# ---------------------------

class TemporalFusionTransformerHead(nn.Module):
    """
    TFT-style forecasting head using your custom Transformer encoder/decoder backbone.

    Inputs:
      - x_hist_vars: dict with keys:
          * "target":        [B, L_in, 1]              (required)
          * "observed":      [B, L_in, V_obs, D_obs]   (optional, per-var features)
          * "known":         [B, L_in, V_known_in, D_k] (optional, per-var features for encoder)
      - x_fut_vars: dict with keys:
          * "known":         [B, L_out, V_known_out, D_k] (future known covariates for decoder)
      - x_static:            [B, V_static, D_s]        (optional, static feats per var)
      - time_hist, time_fut: any (optional), passed to InformerTimeEmbedding if enabled.

    Output:
      - y_hat: [B, L_out, C_out]

    Notes:
      * Variable selection for (observed + known + target) on encoder side.
      * Variable selection for (known + previous target) on decoder side.
      * Static enrichment via GRN-conditioned context added to both sides.
      * Fusion via Transformer cross-attention.
      * Final projection to output channels.
    """

    def __init__(
        self,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_enc: int = 2,
        n_layers_dec: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        freq_modes: int = 32,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        # Variable groups
        num_obs_vars: int = 0,
        d_obs_per_var: int = 1,
        num_known_in_vars: int = 0,
        num_known_out_vars: int = 0,
        d_known_per_var: int = 1,
        num_static_vars: int = 0,
        d_static_per_var: int = 1,
        # Heads
        out_channels: int = 1,
        # Options
        use_time_encoding: bool = True,
        use_positional_encoding: bool = True,
        quantiles: Optional[Tuple[float, ...]] = None,  # e.g., (0.1, 0.5, 0.9)
    ):
        super().__init__()

        self.pred_len = pred_len
        self.out_channels = out_channels
        self.quantiles = quantiles

        # -------- Variable Selection Networks --------
        # Encoder variables: target (1 var, D=1) + observed + known_in
        self.has_target = True
        enc_vars = int(self.has_target) + num_obs_vars + num_known_in_vars
        self.vsn_enc = VariableSelectionNetwork(
            num_vars=enc_vars,
            d_in_per_var=1 if enc_vars == 1 else max(1, max(d_obs_per_var, d_known_per_var, 1)),
            d_model=d_model,
            d_hidden=dim_ff,
            dropout=dropout,
            custom_norm=custom_norm,
            eps=layer_norm_eps,
            name="VSN_Enc",
        )

        # Decoder variables: previous target (teacher forcing at train) + known_out
        dec_vars = int(self.has_target) + num_known_out_vars
        self.vsn_dec = VariableSelectionNetwork(
            num_vars=dec_vars,
            d_in_per_var=1 if dec_vars == 1 else max(1, d_known_per_var),
            d_model=d_model,
            d_hidden=dim_ff,
            dropout=dropout,
            custom_norm=custom_norm,
            eps=layer_norm_eps,
            name="VSN_Dec",
        )

        # -------- Static encoders and enrichment --------
        self.num_static_vars = num_static_vars
        if num_static_vars > 0:
            self.static_proj = nn.Linear(d_static_per_var, d_model)
            self.static_grn_c = GRN(d_in=d_model, d_hidden=dim_ff, d_out=d_model, dropout=dropout, custom_norm=custom_norm, eps=layer_norm_eps)
        else:
            self.static_proj = None
            self.static_grn_c = None

        # -------- Encoder/Decoder backbones (your custom ones) --------
        # We pass input_size=d_model because VSNs output fused d_model features per step
        self.encoder = TransformerEncoder(
            input_size=d_model,
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers_enc,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            att_type=att_type,
            layer_norm_eps=layer_norm_eps,
            norm_strategy=norm_strategy,
            custom_norm=custom_norm,
            max_seq_len=10_000,
            pos_encoder=(PositionalEncoding(d_model, max_len=10_000) if use_positional_encoding else nn.Identity()),
            use_final_norm=True,
            use_swiglu=use_swiglu,
            freq_modes=freq_modes,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
        )

        self.decoder = TransformerDecoder(
            input_size=d_model,
            output_size=d_model,
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers_dec,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            att_type=att_type,
            layer_norm_eps=layer_norm_eps,
            norm_strategy=norm_strategy,
            custom_norm=custom_norm,
            max_seq_len=10_000,
            pos_encoder=(PositionalEncoding(d_model, max_len=10_000) if use_positional_encoding else nn.Identity()),
            use_final_norm=True,
            use_swiglu=use_swiglu,
            freq_modes=freq_modes,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            informer_like=False,
            use_time_encoding=use_time_encoding,
        )

        # -------- Fusion & output --------
        self.enrich_enc = GRN(d_in=d_model, d_hidden=dim_ff, d_out=d_model, dropout=dropout, custom_norm=custom_norm, eps=layer_norm_eps)
        self.enrich_dec = GRN(d_in=d_model, d_hidden=dim_ff, d_out=d_model, dropout=dropout, custom_norm=custom_norm, eps=layer_norm_eps)

        d_out = out_channels if quantiles is None else out_channels * len(quantiles)
        self.head = nn.Linear(d_model, d_out)

    # ---------- helpers ----------

    def _build_encoder_input(self, x_hist_vars: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare encoder VSN input:
          target:   [B, L_in, 1]
          observed: [B, L_in, V_obs, D_obs] or None
          known:    [B, L_in, V_known_in, D_k] or None
        Returns fused enc_input [B, L_in, d_model] and selection weights [B, L_in, V_enc]
        """
        x_target = x_hist_vars["target"]                           # [B, L_in, 1]
        B, Lin, _ = x_target.shape

        xs = []
        # target as one "variable" of (D=1)
        xs.append(x_target.unsqueeze(2))                           # -> [B, L_in, 1, 1]

        if "observed" in x_hist_vars and x_hist_vars["observed"] is not None:
            xs.append(x_hist_vars["observed"])                     # [B, L_in, V_obs, D_obs]
        if "known" in x_hist_vars and x_hist_vars["known"] is not None:
            xs.append(x_hist_vars["known"])                        # [B, L_in, V_known_in, D_k]

        X = torch.cat(xs, dim=2)                                   # [B, L_in, V_enc, D_enc]
        z_enc, w_enc = self.vsn_enc(X)                             # [B, L_in, d_model], [B, L_in, V_enc]
        return z_enc, w_enc

    def _build_decoder_input(self, x_fut_vars: Dict[str, torch.Tensor], prev_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare decoder VSN input:
          prev_target: [B, L_out, 1]  (teacher forcing during train; shift(1))
          known_out:   [B, L_out, V_known_out, D_k] or None
        Returns fused dec_input [B, L_out, d_model] and selection weights [B, L_out, V_dec]
        """
        xs = [prev_target.unsqueeze(2)]                             # [B, L_out, 1, 1]
        if "known" in x_fut_vars and x_fut_vars["known"] is not None:
            xs.append(x_fut_vars["known"])                          # [B, L_out, V_known_out, D_k]
        X = torch.cat(xs, dim=2)                                    # [B, L_out, V_dec, D_dec]
        z_dec, w_dec = self.vsn_dec(X)                              # [B, L_out, d_model], [B, L_out, V_dec]
        return z_dec, w_dec

    def _static_context(self, x_static: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        x_static: [B, V_static, D_s] -> context c [B, 1, d_model]
        """
        if x_static is None or self.num_static_vars == 0:
            return None
        # Mean over static vars -> project -> GRN
        c = self.static_proj(x_static).mean(dim=1, keepdim=True)    # [B, 1, d_model]
        return self.static_grn_c(c)                                 # [B, 1, d_model]

    # ---------- forward ----------

    def forward(
        self,
        x_hist_vars: Dict[str, torch.Tensor],
        x_fut_vars: Dict[str, torch.Tensor],
        x_static: Optional[torch.Tensor] = None,
        time_hist: Optional[torch.Tensor] = None,
        time_fut: Optional[torch.Tensor] = None,
        teacher_forcing_target: Optional[torch.Tensor] = None,
        return_selection: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          dict with keys:
            'y': [B, L_out, C_out] or [B, L_out, C_out*Q] if quantiles set
            optional: 'w_enc', 'w_dec' (selection weights), 'context'
        """
        # 1) VSN encoder (past)
        z_enc, w_enc = self._build_encoder_input(x_hist_vars)       # [B, L_in, d_model], [B, L_in, V_enc]

        # 2) Static enrichment (context)
        c = self._static_context(x_static)                           # [B, 1, d_model] or None
        if c is not None:
            z_enc = self.enrich_enc(z_enc, context=c.expand(-1, z_enc.size(1), -1))  # GRN(x || c)

        # 3) Encode with custom TransformerEncoder (pos/time enc handled internally)
        mem = self.encoder(z_enc, time_features=time_hist)          # [B, L_in, d_model]

        # 4) Prepare decoder inputs
        B = z_enc.size(0)
        Lout = self.pred_len
        if teacher_forcing_target is not None:
            # Use provided teacher forcing targets -> shift right (prepend last history value)
            prev_tgt = teacher_forcing_target.clone()               # [B, L_out, 1]
        else:
            # Zero previous targets at inference start (simple choice)
            prev_tgt = z_enc.new_zeros(B, Lout, 1)

        z_dec_in, w_dec = self._build_decoder_input(x_fut_vars, prev_tgt)  # [B, L_out, d_model], [B, L_out, V_dec]
        if c is not None:
            z_dec_in = self.enrich_dec(z_dec_in, context=c.expand(-1, z_dec_in.size(1), -1))

        # 5) Decode with custom TransformerDecoder
        dec = self.decoder(
            z_dec_in,
            mem,
            time_features=time_fut,
        )  # [B, L_out, d_model]

        # 6) Project to outputs
        y = self.head(dec)  # [B, L_out, C_out] or [B, L_out, C_out*Q]

        out = {"y": y}
        if return_selection:
            out.update({"w_enc": w_enc, "w_dec": w_dec})
        if c is not None:
            out["context"] = c
        return out

    # Optional convenience for quantile split at caller side
    def split_quantiles(self, y: torch.Tensor) -> Dict[float, torch.Tensor]:
        """
        If quantiles were configured, split head output into {q: tensor}.
        y: [B, L_out, C_out * Q]
        """
        assert self.quantiles is not None, "No quantiles configured."
        Q = len(self.quantiles)
        B, T, _ = y.shape
        y = y.view(B, T, self.out_channels, Q)
        return {q: y[..., i] for i, q in enumerate(self.quantiles)}
