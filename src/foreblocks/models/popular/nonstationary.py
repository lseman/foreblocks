"""foreblocks.models.popular.nonstationary.

Non-stationary Transformer framework with stationarization and de-stationary attention.

Based on: Liu et al., "Non-stationary Transformers: Exploring the Stationarity
in Time Series Forecasting", NeurIPS 2022.
Paper: https://arxiv.org/abs/2205.14415

Normalizes input to zero-mean/unit-variance for stable training, processes through
a de-stationary transformer (tau/delta-modulated attention), then denormalizes
output. Provides both a full NonStationaryTransformer and a wrapper that enhances
any existing attention-based transformer head.

Core API:
- NonStationaryTransformer: full NSformer with stationarization, de-stationary encoder, and denormalization
- NonStationaryWrapper: wrap any transformer head with NS stationarization benefits
- DSAttention: de-stationary attention with learned tau (scaling) and delta (shift) factors
- DeStationaryProjector: MLP learner for tau/delta factors from raw input statistics

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.norms import create_norm_layer


# =========================================================
# 1) De-stationary Attention
# =========================================================
class DSAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.output_attention = output_attention

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, _ = q.shape
        S = k.size(1)
        H = self.n_heads
        Dh = self.dh

        Q = self.q_proj(q).view(B, L, H, Dh)  # [B,L,H,Dh]
        K = self.k_proj(k).view(B, S, H, Dh)  # [B,S,H,Dh]
        V = self.v_proj(v).view(B, S, H, Dh)  # [B,S,H,Dh]

        if tau is None:
            tau = 1.0
        else:
            tau = tau.reshape(B, -1)[:, :1].unsqueeze(1).unsqueeze(1)  # [B,1,1,1]

        if delta is None:
            delta = 0.0
        else:
            if delta.size(-1) != S:
                delta = F.interpolate(
                    delta.unsqueeze(1),
                    size=S,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            delta = delta.unsqueeze(1).unsqueeze(1)  # [B,1,1,S]

        # Reference formulation: Softmax(scale * (QK^T * tau + delta)).
        logits = torch.einsum("blhd,bshd->bhls", Q, K)
        scores = (logits * tau + delta) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), float("-inf"))

        A = self.drop(torch.softmax(scores, dim=-1))  # [B,H,L,S]
        out = torch.einsum("bhls,bshd->blhd", A, V)  # [B,H,L,Dh]
        out = out.contiguous().view(B, L, -1)  # [B,L,D]
        out = self.o_proj(self.drop(out))

        attn = None if not self.output_attention else A
        return out, attn


# =========================================================
# 2) Attention Layer wrapper
# =========================================================
class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = attention
        self.d_model = d_model

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out, attn = self.attention(q, k, v, attn_mask, tau=tau, delta=delta)
        return out, attn


# =========================================================
# 3) Tau/Delta Learner (Projector)
# =========================================================
class DeStationaryProjector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dims: list[int],
        hidden_layers: int,
        output_dim: int,
        seq_len: int | None = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        concat_dim = 2 * in_channels
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self._series_convs = nn.ModuleDict()
        if seq_len is not None:
            self._series_convs[str(seq_len)] = self._make_series_conv(seq_len)

        layers: list[nn.Module] = [nn.Linear(concat_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
            ]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def _make_series_conv(self, seq_len: int) -> nn.Conv1d:
        padding = (self.kernel_size - 1) // 2
        return nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

    def _series_conv(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        key = str(seq_len)
        if key not in self._series_convs:
            conv = self._make_series_conv(seq_len).to(device=x.device, dtype=x.dtype)
            self._series_convs[key] = conv
        return self._series_convs[key](x)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        B, _, _ = x.shape
        x_conv = self._series_conv(x)  # [B, 1, C]
        concat = torch.cat([x_conv, stats], dim=1)  # [B, 2, C] -> [B, 2C]
        return self.backbone(concat.view(B, -1))  # [B, out_dim]


# =========================================================
# 4) Transformer Encoder Layer with De-stationary Attention
# =========================================================
class NSEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.attention = AttentionLayer(
            DSAttention(d_model=d_model, n_heads=n_heads, dropout=dropout),
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.conv1 = nn.Conv1d(d_model, dim_ff, 1)
        self.conv2 = nn.Conv1d(dim_ff, d_model, 1)
        self.norm1 = create_norm_layer(norm_type, d_model, eps=eps)
        self.norm2 = create_norm_layer(norm_type, d_model, eps=eps)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.drop(new_x)

        y = self.norm1(x)
        y = self.drop(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.drop(self.conv2(y).transpose(-1, 1))
        out = self.norm2(x + y)

        return out, attn


class NSFormerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NSEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = create_norm_layer(norm_type, d_model, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return self.norm(x), attns


class NSDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attention = AttentionLayer(
            DSAttention(d_model=d_model, n_heads=n_heads, dropout=dropout),
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.cross_attention = AttentionLayer(
            DSAttention(d_model=d_model, n_heads=n_heads, dropout=dropout),
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.conv1 = nn.Conv1d(d_model, dim_ff, 1)
        self.conv2 = nn.Conv1d(dim_ff, d_model, 1)
        self.norm1 = create_norm_layer(norm_type, d_model, eps=eps)
        self.norm2 = create_norm_layer(norm_type, d_model, eps=eps)
        self.norm3 = create_norm_layer(norm_type, d_model, eps=eps)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Delta is not used for decoder self-attention in the reference model.
        x = x + self.drop(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)
        x = x + self.drop(
            self.cross_attention(
                x,
                cross,
                cross,
                attn_mask=cross_mask,
                tau=tau,
                delta=delta,
            )[0]
        )
        y = x = self.norm2(x)
        y = self.drop(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.drop(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class NSFormerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NSDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = create_norm_layer(norm_type, d_model, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                tau=tau,
                delta=delta,
            )
        return self.norm(x)


# =========================================================
# 5) Full NonStationary Transformer
# =========================================================
class NonStationaryTransformer(nn.Module):
    def __init__(
        self,
        pred_len: int,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        d_layers: int = 1,
        label_len: int | None = None,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
        eps: float = 1e-5,
        quantiles: tuple[float, ...] | None = None,
        p_hidden_dims: list[int] | None = None,
        p_hidden_layers: int = 2,
        tau_threshold: float = 80.0,
        max_seq_len: int = 512,
        input_len: int | None = None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.quantiles = quantiles
        self.tau_threshold = tau_threshold
        self.label_len = label_len
        self.max_seq_len = max_seq_len

        d_out = out_channels if quantiles is None else out_channels * len(quantiles)

        # Input projection
        self.enc_in = nn.Linear(in_channels, d_model)

        # De-stationary encoder
        self.encoder = NSFormerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
        )

        self.decoder = NSFormerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=d_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
        )

        # De-stationary factor learners
        p_hidden_dims = p_hidden_dims or [256, 128]
        self.tau_learner = DeStationaryProjector(
            in_channels=in_channels,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=1,
            seq_len=input_len,
        )
        self.delta_learner = DeStationaryProjector(
            in_channels=in_channels,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=max_seq_len,
            seq_len=input_len,
        )

        self.output_projection = nn.Linear(d_model, d_out)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _match_stats(stats: torch.Tensor, channels: int) -> torch.Tensor:
        if stats.size(-1) == channels:
            return stats
        if channels == 1:
            return stats.mean(dim=-1, keepdim=True)
        if stats.size(-1) == 1:
            return stats.expand(-1, -1, channels)
        repeats = math.ceil(channels / stats.size(-1))
        return stats.repeat(1, 1, repeats)[..., :channels]

    def _denormalize(
        self,
        y: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        if self.quantiles is None:
            out_mean = self._match_stats(mean, y.size(-1))
            out_std = self._match_stats(std, y.size(-1))
            return y * out_std + out_mean

        Q = len(self.quantiles)
        yq = y.view(y.size(0), y.size(1), self.out_channels, Q)
        out_mean = self._match_stats(mean, self.out_channels).unsqueeze(-1)
        out_std = self._match_stats(std, self.out_channels).unsqueeze(-1)
        return (yq * out_std + out_mean).view(y.size(0), y.size(1), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.in_channels:
            raise ValueError(
                f"Expected x [B, L, C_in={self.in_channels}], got {tuple(x.shape)}"
            )
        B, L_in, C_in = x.shape
        raw = x

        # ---- Series Stationarization ----
        mean = x.mean(dim=1, keepdim=True).detach()  # [B, 1, C_in]
        x_centered = x - mean
        std = torch.sqrt(
            torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # [B, 1, C_in]
        x_norm = x_centered / std  # [B, L, C_in]

        # ---- Learn de-stationary factors ----
        tau = self.tau_learner(raw, std)  # [B, 1]
        tau = torch.clamp(tau, min=-self.tau_threshold, max=self.tau_threshold).exp()

        delta = self.delta_learner(raw, mean)  # [B, max_seq_len]
        if delta.size(1) != L_in:
            delta = F.interpolate(
                delta.unsqueeze(1),
                size=L_in,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        # ---- Encode ----
        z = self.enc_in(x_norm)  # [B, L, D]
        enc_out, _ = self.encoder(z, attn_mask=None, tau=tau, delta=delta)  # [B, L, D]

        # ---- Decoder: normalized tail + zero future tokens ----
        label_len = min(
            L_in,
            self.label_len if self.label_len is not None else max(1, L_in // 2),
        )
        future = torch.zeros(B, self.pred_len, C_in, device=x.device, dtype=x.dtype)
        dec_in = torch.cat([x_norm[:, -label_len:, :], future], dim=1)
        dec_out = self.decoder(
            self.enc_in(dec_in),
            enc_out,
            tau=tau,
            delta=delta,
        )
        y = self.output_projection(dec_out[:, -self.pred_len :, :])
        return self._denormalize(y, mean, std)

    def split_quantiles(self, y: torch.Tensor) -> dict[float, torch.Tensor]:
        if self.quantiles is None:
            raise ValueError("No quantiles configured")
        Q = len(self.quantiles)
        B, H, _ = y.shape
        yq = y.view(B, H, self.out_channels, Q)
        return {q: yq[..., i] for i, q in enumerate(self.quantiles)}


# =========================================================
# 6) NonStationary Wrapper — enhance any forecasting module
# =========================================================
class NonStationaryWrapper(nn.Module):
    def __init__(
        self,
        wrapped_head: nn.Module,
        tau_threshold: float = 80.0,
    ):
        super().__init__()
        self.wrapped_head = wrapped_head
        self.tau_threshold = tau_threshold

        # Extract channel count from wrapped head
        in_ch = getattr(wrapped_head, "in_channels", 1)
        pred_len = getattr(wrapped_head, "pred_len", 96)

        p_hidden_dims = [256, 128]
        self.tau_learner = DeStationaryProjector(
            in_channels=in_ch,
            hidden_dims=p_hidden_dims,
            hidden_layers=2,
            output_dim=1,
        )
        self.delta_learner = DeStationaryProjector(
            in_channels=in_ch,
            hidden_dims=p_hidden_dims,
            hidden_layers=2,
            output_dim=512,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L_in, C_in = x.shape
        raw = x

        # ---- Stationarize ----
        mean = x.mean(dim=1, keepdim=True).detach()  # [B, 1, C]
        x_centered = x - mean
        std = torch.sqrt(
            torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # [B, 1, C]
        x_norm = x_centered / std  # [B, L, C]

        # ---- Learn factors ----
        tau = self.tau_learner(raw, std)  # [B, 1]
        tau = torch.clamp(tau, min=-self.tau_threshold, max=self.tau_threshold).exp()
        delta = self.delta_learner(raw, mean)
        if delta.size(1) != L_in:
            delta = F.interpolate(
                delta.unsqueeze(1),
                size=L_in,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        # ---- Call wrapped head ----
        try:
            y = self.wrapped_head(x_norm, tau=tau, delta=delta)
        except TypeError:
            y = self.wrapped_head(x_norm)

        # ---- Denormalize ----
        out_std = NonStationaryTransformer._match_stats(std, y.size(-1))
        out_mean = NonStationaryTransformer._match_stats(mean, y.size(-1))
        return y * out_std + out_mean
