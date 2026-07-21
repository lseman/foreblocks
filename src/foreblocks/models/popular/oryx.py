"""Oryx multi-mixer block and transformer stack.

Multi-mixer architecture combining softmax attention and a linear recurrent mixer
(Gated Delta Net) with shared key/value projections across both mixers. Uses
causal depthwise convolutions on K/V, GatedRMSNorm, and mixer-specific Q
projections for efficient multi-mode sequence modeling.

Based on: Li et al., "Multi-Mixer Models: Flexible Sequence Modeling with Shared
Representations", arXiv 2605.28769v1.
Paper: https://arxiv.org/abs/2605.28769

Core API:
- OryxMixerBlock: shared K/V multi-mixer with attention + GDN modes
- OryxLayer: residual transformer layer with shared Oryx mixer
- OryxTransformer: stacked Oryx transformer encoder

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.norms import create_norm_layer
from foreblocks.modules.attention.modules.linear_att.gated_delta import (
    GatedDeltaNet,
)
from foreblocks.modules.attention.multi_att import MultiAttention


class _CausalDepthwiseConv(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=self.kernel_size,
            groups=d_model,
            padding=self.pad,
            bias=False,
        )

    def forward(self, x: torch.Tensor, activation: str | None = "silu") -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self.pad > 0:
            x = x[:, :, : -self.pad]
        x = x.transpose(1, 2)
        if activation == "silu":
            x = F.silu(x)
        return x


class _NoOpConv(nn.Module):
    def forward(self, x: torch.Tensor, activation: str | None = None) -> torch.Tensor:
        return x


class _GatedRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], gate: [B, T, D]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * gate * self.weight  # [B, T, D]


class OryxMixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        attention_type: str = "standard",
        linear_mode: str = "gdn",
        use_short_conv: bool = True,
        conv_kernel: int = 4,
        gate: bool = True,
        norm_type: str = "rms",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_short_conv = bool(use_short_conv)
        self.gate = bool(gate)
        self.linear_mode = str(linear_mode).lower()

        self.attn = MultiAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_type=attention_type,
            use_mla=False,
            use_paged_cache=False,
            use_swiglu=False,
        )
        self._linear_mixer = None
        if self.linear_mode not in {"gdn", "gated_delta", "linear"}:
            raise ValueError(
                f"Unknown linear_mode={linear_mode}. Use 'gdn' or 'linear'."
            )
        self.gdn = GatedDeltaNet(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_type="standard",
            freq_modes=0,
            cross_attention=False,
            use_short_conv=False,
            conv_kernel=conv_kernel,
        )

        # Shared key/value projections across both mixers.
        self.shared_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.shared_v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn.k_proj = self.shared_k_proj
        self.attn.v_proj = self.shared_v_proj
        self.gdn.k_proj = self.shared_k_proj
        self.gdn.v_proj = self.shared_v_proj

        # Disable built-in output projections; use a shared final projection.
        self.attn.out_proj = nn.Identity()
        self.gdn.o_proj = nn.Identity()

        self.k_conv = (
            _CausalDepthwiseConv(d_model, conv_kernel)
            if self.use_short_conv
            else _NoOpConv()
        )
        self.v_conv = (
            _CausalDepthwiseConv(d_model, conv_kernel)
            if self.use_short_conv
            else _NoOpConv()
        )

        self.gate_proj = nn.Linear(d_model, d_model, bias=False) if self.gate else None
        # GatedRMSNorm per Oryx paper: Y = GatedRMSNorm(O, G) W_O
        # (gating + RMSNorm fused, not applied separately)
        self.gate_norm = _GatedRMSNorm(d_model, eps=layer_norm_eps)
        # Determine conv activation per linear mixer type (paper Appendix B):
        #   GDN needs SiLU on K/V conv for modeling/retrieval performance
        #   Mamba-2 removes activations after conv for Transformer consistency
        self._conv_activation = "silu" if self.linear_mode == "gdn" else None

        # Disable GDN internal gate + norm — Oryx applies GatedRMSNorm externally.
        # This ensures both mixers share the same final processing path.
        self.gdn.h_rms = nn.Identity()
        self.gdn.g_up = nn.Identity()
        self.gdn.g_down = nn.Identity()
        self.gdn.o_proj = nn.Identity()  # GDN already outputs [B,T,D]
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "attention",
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        layer_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B,T,C], got {tuple(x.shape)}")

        mode = str(mode).lower()
        # Apply conv with activation per linear mixer type (paper Appendix B)
        k = self.k_conv(x, activation=self._conv_activation)
        v = self.v_conv(x, activation=self._conv_activation)

        if mode == "attention":
            out, _, state = self.attn(
                x,
                k,
                v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=True,
                need_weights=False,
                layer_state=layer_state,
            )
        elif mode in {"gdn", "linear", "gated_delta"}:
            # Skip GDN's internal cross-attention detection and gate+norm.
            # GDN uses pre-projected+convolved K/V directly in self-attention
            # recurrent mode (Oryx design: shared K/V, mixer-specific Q).
            out, _, state = self.gdn(
                x,
                k,
                v,
                attn_mask=None,
                key_padding_mask=key_padding_mask,
                is_causal=True,
                layer_state=layer_state,
                skip_cross_attention=True,
                skip_gate_norm=True,
            )
        else:
            raise ValueError(f"Unsupported mode={mode}. Use 'attention' or 'gdn'.")

        gate = None
        if self.gate and self.gate_proj is not None:
            # SiLU activation per Oryx paper (σ is usually SiLU)
            gate = F.silu(self.gate_proj(x))

        # GatedRMSNorm: Y = GatedRMSNorm(O, G) W_O (paper Section 3, Eq 2)
        if gate is not None:
            out = self.gate_norm(out, gate)
        else:
            out = self.gate_norm(out, torch.ones_like(out))
        out = self.out_proj(out)
        return out, state


class OryxLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attention_type: str = "standard",
        linear_mode: str = "gdn",
        use_short_conv: bool = True,
        conv_kernel: int = 4,
        gate: bool = True,
        norm_type: str = "rms",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.mixer = OryxMixerBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_type=attention_type,
            linear_mode=linear_mode,
            use_short_conv=use_short_conv,
            conv_kernel=conv_kernel,
            gate=gate,
            norm_type=norm_type,
            layer_norm_eps=layer_norm_eps,
        )
        self.norm1 = create_norm_layer(norm_type, d_model, eps=layer_norm_eps)
        self.norm2 = create_norm_layer(norm_type, d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "attention",
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        layer_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        residual = x
        out, state = self.mixer(
            x,
            mode=mode,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            layer_state=layer_state,
        )
        x = residual + out
        x = self.norm1(x)

        residual = x
        x = self.ffn(x)
        x = residual + x
        x = self.norm2(x)
        return x, state


class OryxTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attention_type: str = "standard",
        linear_mode: str = "gdn",
        use_short_conv: bool = True,
        conv_kernel: int = 4,
        gate: bool = True,
        norm_type: str = "rms",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.layers = nn.ModuleList(
            [
                OryxLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attention_type=attention_type,
                    linear_mode=linear_mode,
                    use_short_conv=use_short_conv,
                    conv_kernel=conv_kernel,
                    gate=gate,
                    norm_type=norm_type,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = create_norm_layer(norm_type, d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "attention",
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        layer_states: list[dict[str, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor] | None]]:
        if layer_states is None:
            layer_states = [None] * len(self.layers)
        if len(layer_states) != len(self.layers):
            raise ValueError(
                f"layer_states length {len(layer_states)} does not match num_layers {len(self.layers)}"
            )

        states: list[dict[str, torch.Tensor] | None] = []
        for layer, state in zip(self.layers, layer_states):
            x, new_state = layer(
                x,
                mode=mode,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                layer_state=state,
            )
            states.append(new_state)
        x = self.final_norm(x)
        return x, states
