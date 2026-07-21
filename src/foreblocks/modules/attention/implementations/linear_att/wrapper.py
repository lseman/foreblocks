"""foreblocks.modules.attention.implementations.linear_att.wrapper.

Modular linear attention with swappable backends.

Dispatches to one of six backend implementations (RDA, GLA, DeltaNet,
GatedDeltaNet, GatedDeltaNet2, KimiAttention) based on a string key, all
implementing the same drop-in forward API. Use when you need to switch
backends without changing model code.

Core API:
- ModernLinearAttention: backend-dispatching wrapper with unified forward API

"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from foreblocks.modules.attention.implementations.linear_att.deltanet import (
    DeltaNetBackend,
)
from foreblocks.modules.attention.implementations.linear_att.gated_delta import (
    GatedDeltaNet,
)
from foreblocks.modules.attention.implementations.linear_att.gated_deltanet2 import (
    GatedDeltaNet2,
)
from foreblocks.modules.attention.implementations.linear_att.gla import GLABackend
from foreblocks.modules.attention.implementations.linear_att.kimi import KimiAttention
from foreblocks.modules.attention.implementations.linear_att.rda import RDABackend

GatedDeltaBackend = GatedDeltaNet
KimiBackend = KimiAttention


_BACKEND_MAP = {
    "rda": RDABackend,
    "gla": GLABackend,
    "deltanet": DeltaNetBackend,
    "gated_delta": GatedDeltaBackend,
    "gated_deltanet": GatedDeltaBackend,
    "gated_deltanet2": GatedDeltaNet2,
    "gdn2": GatedDeltaNet2,
    "kimi": KimiBackend,
}


class ModernLinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        backend: Literal[
            "rda",
            "gla",
            "deltanet",
            "gated_delta",
            "gated_deltanet",
            "gated_deltanet2",
            "gdn2",
            "kimi",
        ] = "rda",
        state: str = "elu",
        mode: Literal["chunk", "recurrent"] = "chunk",
        chunk_size: int = 64,
        pos_encoding_type: str = "sinusoidal",
        **backend_kwargs,
    ):
        super().__init__()
        self.backend_name = backend
        self.d_model = d_model
        self.n_heads = n_heads
        self.mode = mode
        self.pos_encoding_type = pos_encoding_type

        if backend not in _BACKEND_MAP:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {sorted(_BACKEND_MAP.keys())}"
            )

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        # Pass relevant kwargs to backend
        kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            pos_encoding_type=pos_encoding_type,
        )
        if backend == "rda":
            kwargs["feature_map"] = state
            kwargs["num_features"] = backend_kwargs.get("num_features")
        elif backend == "gla":
            kwargs["mode"] = mode
            kwargs["chunk_size"] = chunk_size
            kwargs["gate_logit_normalizer"] = backend_kwargs.get(
                "gate_logit_normalizer", 16.0
            )
            kwargs["gate_low_rank_dim"] = backend_kwargs.get("gate_low_rank_dim", 16)
            kwargs["clamp_min"] = backend_kwargs.get("clamp_min")
            kwargs["use_output_gate"] = backend_kwargs.get("use_output_gate", True)
        elif backend == "deltanet":
            kwargs["mode"] = mode
            kwargs["chunk_size"] = chunk_size
            kwargs["conv_size"] = backend_kwargs.get("conv_size", 4)
        elif backend in {"gated_delta", "gated_deltanet"}:
            kwargs["chunk_size"] = chunk_size if mode == "chunk" else 0
            kwargs["conv_kernel"] = backend_kwargs.get(
                "conv_kernel", backend_kwargs.get("conv_size", 4)
            )
            state_gate = state.lower()
            if state_gate in {"mamba", "mamba2"}:
                kwargs["use_mamba_gate"] = True
            elif state_gate in {"legacy", "sigmoid"}:
                kwargs["use_mamba_gate"] = False
            for name in (
                "d_key",
                "d_val",
                "use_short_conv",
                "beta_max",
                "eps",
                "use_mamba_gate",
                "attention_type",
                "freq_modes",
                "cross_attention",
            ):
                if name in backend_kwargs:
                    kwargs[name] = backend_kwargs[name]
        elif backend == "kimi":
            kwargs["chunk_size"] = chunk_size if mode == "chunk" else 0
            kwargs["conv_kernel"] = backend_kwargs.get(
                "conv_kernel", backend_kwargs.get("conv_size", 4)
            )
            if "use_short_conv" in backend_kwargs:
                kwargs["shortconv_mode"] = (
                    "depthwise" if backend_kwargs["use_short_conv"] else "off"
                )
            for name in (
                "shortconv_mode",
                "safe_updates",
                "alpha_min",
                "use_triton",
                "cross_attention",
            ):
                if name in backend_kwargs:
                    kwargs[name] = backend_kwargs[name]
            if "d_key" in backend_kwargs:
                kwargs["d_key"] = backend_kwargs["d_key"]
            if "d_val" in backend_kwargs:
                kwargs["d_val"] = backend_kwargs["d_val"]
            if "expand_v" in backend_kwargs:
                kwargs["expand_v"] = backend_kwargs["expand_v"]
        elif backend in {"gated_deltanet2", "gdn2"}:
            kwargs["chunk_size"] = chunk_size if mode == "chunk" else 0
            kwargs["conv_kernel"] = backend_kwargs.get(
                "conv_kernel", backend_kwargs.get("conv_size", 4)
            )
            for name in (
                "d_key",
                "d_val",
                "use_short_conv",
                "eps",
                "allow_neg_eigval",
                "attention_type",
                "freq_modes",
                "cross_attention",
            ):
                if name in backend_kwargs:
                    kwargs[name] = backend_kwargs[name]

        self.impl = _BACKEND_MAP[backend](**kwargs)

    @property
    def state_key(self) -> str:
        if self.backend_name in {"gated_delta", "gated_deltanet"}:
            return "gdn_state"
        if self.backend_name in {"gated_deltanet2", "gdn2"}:
            return "gdn2_state"
        if self.backend_name == "kimi":
            return "S"
        return f"{self.backend_name}_S"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        layer_state: dict | None = None,
    ) -> tuple[torch.Tensor, None, dict | None]:
        # Positional encoding (RoPE/ALiBi) is handled inside each backend, after
        # its own Q/K projection — applying it here would double-project.
        return self.impl(
            query, key, value, attn_mask, key_padding_mask, is_causal, layer_state
        )

    def reset_state(self, layer_state: dict) -> None:
        layer_state.pop(self.state_key, None)
