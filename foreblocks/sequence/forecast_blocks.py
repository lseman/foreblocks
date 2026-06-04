from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from foreblocks.ui.node_spec import node


class _SequenceModelBlock(nn.Module):
    """Base wrapper for sequence mixers used as ForecastingModel backbones."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        layer_factory: Callable[[int], nn.Module],
        device: str | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.preferred_device = device

        self.input_proj = (
            nn.Identity()
            if input_size == hidden_size
            else nn.Linear(input_size, hidden_size)
        )
        self.layers = nn.ModuleList(layer_factory(i) for i in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def _run_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = layer(x)
        return out[0] if isinstance(out, tuple) else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"{self.__class__.__name__} expects [B, T, F], got {tuple(x.shape)}"
            )
        if self.preferred_device is not None:
            target = torch.device(self.preferred_device)
            if target.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    f"{self.__class__.__name__} requires CUDA, but CUDA is not available."
                )
            if x.device != target:
                self.to(target)
                x = x.to(target)

        x = self.input_proj(x)
        for layer in self.layers:
            layer_out = self._run_layer(layer, x)
            x = x + self.dropout(layer_out)

        return self.norm(x)


class _FLAMamba2Layer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        d_state: int,
        d_conv: int,
        num_heads: int,
        n_groups: int,
        chunk_size: int,
        backend: str,
        layer_idx: int,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        intermediate_size = expand * hidden_size
        if intermediate_size % num_heads != 0:
            raise ValueError(
                "expand * hidden_size must be divisible by num_heads for FLA Mamba2."
            )

        from foreblocks.ops.raven.backend import import_fla_module

        Mamba2 = import_fla_module("fla.layers.mamba2").Mamba2
        self.layer = Mamba2(
            hidden_size=hidden_size,
            state_size=d_state,
            conv_kernel=d_conv,
            num_heads=num_heads,
            head_dim=intermediate_size // num_heads,
            expand=expand,
            n_groups=n_groups,
            chunk_size=chunk_size,
            backend=backend,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out


@node(
    type_id="mamba2_model_block",
    name="Mamba2 Model Block",
    category="Model Blocks",
    color="bg-gradient-to-br from-cyan-700 to-blue-800",
    outputs=["encoder"],
    inputs=[],
    infer=True,
)
class Mamba2ModelBlock(_SequenceModelBlock):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 4,
        n_groups: int = 1,
        chunk_size: int = 256,
        use_fused_path: bool = True,
        use_triton_ssd: bool = True,
        implementation: str = "fla",
        backend: str = "cuda",
        device: str = "cuda",
    ) -> None:
        impl = implementation.lower()
        if impl not in {"fla", "custom"}:
            raise ValueError("implementation must be either 'fla' or 'custom'")

        if impl == "fla":
            layer_factory = lambda layer_idx: _FLAMamba2Layer(
                hidden_size=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                num_heads=num_heads,
                n_groups=n_groups,
                chunk_size=chunk_size,
                backend=backend,
                layer_idx=layer_idx,
            )
            preferred_device = device
        else:
            from foreblocks.sequence.mamba_hybrid.mamba2 import Mamba2Block

            layer_factory = lambda _: Mamba2Block(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                num_heads=num_heads,
                n_groups=n_groups,
                chunk_size=chunk_size,
                use_fused_path=use_fused_path,
                use_triton_ssd=use_triton_ssd,
            )
            preferred_device = None

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=preferred_device,
            layer_factory=layer_factory,
        )


@node(
    type_id="mamba3_model_block",
    name="Mamba3 Model Block",
    category="Model Blocks",
    color="bg-gradient-to-br from-sky-700 to-indigo-800",
    outputs=["encoder"],
    inputs=[],
    infer=True,
)
class Mamba3ModelBlock(_SequenceModelBlock):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        num_heads: int = 4,
        n_groups: int = 1,
        chunk_size: int = 256,
        rope_fraction: float = 0.5,
        use_triton_ssd: bool = True,
    ) -> None:
        from foreblocks.sequence.mamba_hybrid.mamba3 import Mamba3Block

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=None,
            layer_factory=lambda _: Mamba3Block(
                d_model=hidden_size,
                d_inner=2 * hidden_size,
                d_state=d_state,
                head_dim=(2 * hidden_size) // num_heads,
                num_heads=num_heads,
                n_groups=n_groups,
                chunk_size=chunk_size,
                rope_fraction=rope_fraction,
                use_triton_ssd=use_triton_ssd,
            ),
        )


@node(
    type_id="hybrid_mamba_model_block",
    name="Hybrid Mamba Model Block",
    category="Model Blocks",
    color="bg-gradient-to-br from-teal-700 to-cyan-800",
    outputs=["encoder"],
    inputs=[],
    infer=True,
)
class HybridMambaModelBlock(_SequenceModelBlock):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 4,
        n_groups: int = 1,
        window_size: int = 128,
        mamba_mode: str = "mamba2",
    ) -> None:
        from foreblocks.sequence.mamba_hybrid.hybrid import HybridMamba2Block

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=None,
            layer_factory=lambda _: HybridMamba2Block(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                num_heads=num_heads,
                n_groups=n_groups,
                window_size=window_size,
                mamba_mode=mamba_mode,
            ),
        )


@node(
    type_id="raven_model_block",
    name="Raven Model Block",
    category="Model Blocks",
    color="bg-gradient-to-br from-fuchsia-700 to-rose-800",
    outputs=["encoder"],
    inputs=[],
    infer=True,
)
class RavenModelBlock(_SequenceModelBlock):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        num_slots: int = 64,
        topk: int = 32,
        feature_map: str = "relu",
        decay_type: str = "Mamba2",
        router_score: str = "sigmoid",
        router_type: str = "lin",
        gate_fn: str = "relu",
        use_output_gate: bool = False,
        use_rope: bool = False,
        fuse_norm: bool = False,
        device: str = "cuda",
    ) -> None:
        from foreblocks.sequence.raven.blocks.raven import Raven

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            layer_factory=lambda layer_idx: Raven(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                num_slots=num_slots,
                topk=topk,
                feature_map=feature_map,
                decay_type=decay_type,
                router_score=router_score,
                router_type=router_type,
                gate_fn=gate_fn,
                use_output_gate=use_output_gate,
                use_rope=use_rope,
                fuse_norm=fuse_norm,
                layer_idx=layer_idx,
            ),
        )


__all__ = [
    "HybridMambaModelBlock",
    "Mamba2ModelBlock",
    "Mamba3ModelBlock",
    "RavenModelBlock",
]
