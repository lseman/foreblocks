"""Immutable architecture descriptions. Each occurrence builds a separate layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoPE:
    base: float = 10000.0


@dataclass(frozen=True)
class GQA:
    heads: int
    kv_heads: int
    position: RoPE = RoPE()
    qk_norm: bool = False

    def __post_init__(self):
        if self.heads < 1 or self.kv_heads < 1 or self.heads % self.kv_heads:
            raise ValueError("kv_heads must be positive and divide heads")
        if not isinstance(self.position, RoPE) or self.position.base <= 0:
            raise ValueError("GQA requires RoPE with a positive base")


@dataclass(frozen=True)
class GatedDeltaNet:
    heads: int


@dataclass(frozen=True)
class SwiGLU:
    hidden_dim: int


@dataclass(frozen=True)
class MoE:
    expert: SwiGLU
    num_experts: int
    top_k: int = 2


@dataclass(frozen=True)
class RMSNorm:
    eps: float = 1e-5


@dataclass(frozen=True)
class LayerNorm:
    eps: float = 1e-5


@dataclass(frozen=True)
class Residual:
    """Add each sublayer's output to its input."""


@dataclass(frozen=True)
class MHC:
    """Carry parallel residual streams across the stack."""

    streams: int = 4


@dataclass(frozen=True)
class AttnRes:
    """Accumulate attention-weighted residuals across layers."""


@dataclass(frozen=True)
class EncoderBlock:
    """Describe one transformer layer; repeated descriptions have independent weights."""

    mixer: GQA | GatedDeltaNet
    ffn: SwiGLU | MoE
    norm: RMSNorm | LayerNorm = RMSNorm()
    residual: Residual | MHC | AttnRes = Residual()

    def __post_init__(self):
        if not isinstance(self.mixer, (GQA, GatedDeltaNet)) or self.mixer.heads < 1:
            raise ValueError("mixer must specify a positive head count")
        expert = self.ffn
        if isinstance(expert, MoE):
            if not 1 <= expert.top_k <= expert.num_experts:
                raise ValueError("top_k must be between 1 and num_experts")
            expert = expert.expert
        if not isinstance(expert, SwiGLU) or expert.hidden_dim < 1:
            raise ValueError("SwiGLU requires a positive hidden_dim")
        if not isinstance(self.norm, (LayerNorm, RMSNorm)) or not isinstance(
            self.residual, (Residual, MHC, AttnRes)
        ):
            raise ValueError("unsupported block normalization or residual")

    @property
    def attention_name(self) -> str:
        return "standard" if isinstance(self.mixer, GQA) else "gated_deltanet"

    def layer_options(self) -> dict[str, Any]:
        moe = isinstance(self.ffn, MoE)
        width = self.ffn.expert.hidden_dim if moe else self.ffn.hidden_dim
        gqa = isinstance(self.mixer, GQA)
        residual_name = {Residual: "standard", MHC: "mhc", AttnRes: "attention"}[
            type(self.residual)
        ]
        return dict(
            attention=self.attention_name,
            n_heads=self.mixer.heads,
            n_kv_heads=self.mixer.kv_heads if gqa else None,
            position="rope" if gqa else "none",
            rope_base=self.mixer.position.base if gqa else 10000.0,
            attention_options={"use_mla": False, "qk_norm": self.mixer.qk_norm}
            if gqa
            else {},
            norm="rms" if isinstance(self.norm, RMSNorm) else "layer",
            norm_eps=self.norm.eps,
            residual=residual_name,
            mhc_streams=self.residual.streams if isinstance(self.residual, MHC) else 4,
            ff_dim=width,
            swiglu=True,
            moe_experts=self.ffn.num_experts if moe else 0,
            moe_top_k=self.ffn.top_k if moe else 1,
            moe_options={} if moe else {"swiglu_hidden_dim": width},
        )

    def to_dict(self):
        data = asdict(self)
        for key in ("mixer", "ffn", "norm", "residual"):
            data[key]["kind"] = type(getattr(self, key)).__name__
        if isinstance(self, DecoderBlock):
            data["cross_attention"]["kind"] = "GQA"
        return data

    @classmethod
    def from_dict(cls, data):
        def component(values, types):
            values = dict(values)
            kind = values.pop("kind")
            if kind not in types:
                raise ValueError(f"Unknown block component: {kind}")
            if kind == "GQA":
                values["position"] = RoPE(**values["position"])
            if kind == "MoE":
                values["expert"] = SwiGLU(**values["expert"])
            return types[kind](**values)

        block_type = DecoderBlock if "cross_attention" in data else EncoderBlock
        cross = (
            {"cross_attention": component(data["cross_attention"], {"GQA": GQA})}
            if block_type is DecoderBlock
            else {}
        )
        return block_type(
            component(data["mixer"], {"GQA": GQA, "GatedDeltaNet": GatedDeltaNet}),
            component(data["ffn"], {"SwiGLU": SwiGLU, "MoE": MoE}),
            component(data["norm"], {"RMSNorm": RMSNorm, "LayerNorm": LayerNorm}),
            component(
                data["residual"], {"Residual": Residual, "MHC": MHC, "AttnRes": AttnRes}
            ),
            **cross,
        )


@dataclass(frozen=True)
class DecoderBlock(EncoderBlock):
    """Causal target mixer followed by cross-attention over encoder memory."""

    cross_attention: GQA = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.cross_attention, GQA):
            raise TypeError("cross_attention must be a GQA description")


# Compatibility with the original encoder-only spelling and pickles.
Block = EncoderBlock
