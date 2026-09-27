"""
Serializable record of a discretized DARTS architecture.

``derive_final_architecture`` (see ``finalization.py``) already computes
every decision needed to describe a discovered architecture — which op won
each cell edge, which transformer sub-choices were selected, which
normalization mode was frozen — but historically discarded all of it after
mutating the model in place. There was no way to export, version, diff, or
rebuild a found architecture without re-running search.

``Genotype`` captures those decisions in one serializable object.
``build_model_from_genotype`` applies a recorded ``Genotype`` to a freshly
constructed *mixed* (searchable) model of matching shape — same
``input_dim``/``hidden_dim``/``num_cells``/``num_nodes``/``arch_mode`` — to
reproduce the same discrete architecture without re-running the bilevel
search. The cell topology is rebuilt exactly (each edge's op is looked up by
name in the central :mod:`op_registry` and wrapped in ``FixedOp``, so it
does not depend on the fresh model's own randomly-initialized alphas at
all). The transformer/normalization sub-choices are forced via the explicit
override kwargs that :class:`ArchitectureConverter` accepts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ..common.norms import NORM_NAMES
from ..ops.fixed import FixedOp
from ..ops.registry import OP_REGISTRY, build_op
from .converter import ArchitectureConverter

__all__ = [
    "EdgeGenotype",
    "CellGenotype",
    "TransformerGenotype",
    "Genotype",
    "apply_cell_choices",
    "apply_norm_choice",
    "build_model_from_genotype",
    "resolve_op_name",
]

SCHEMA_VERSION = 1


@dataclass
class EdgeGenotype:
    edge_idx: int
    source: int
    target: int
    operation: str
    op_weight: float | None = None
    importance: float = 1.0


@dataclass
class CellGenotype:
    cell_idx: int
    num_nodes: int
    edges: list[EdgeGenotype] = field(default_factory=list)
    op_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class TransformerGenotype:
    self_attention_type: str = "unknown"
    self_attention_position: str = "unknown"
    ffn_mode: str = "unknown"
    # Encoder-only.
    patch_mode: str | None = None
    # Decoder-only.
    cross_attention_type: str | None = None
    cross_attention_position: str | None = None
    decode_style: str | None = None


@dataclass
class Genotype:
    """A fully-discretized DARTS architecture, independent of its trained weights."""

    arch_mode: str = "unknown"
    input_dim: int | None = None
    hidden_dim: int | None = None
    seq_length: int | None = None
    forecast_horizon: int | None = None
    norm: str | None = None
    cells: list[CellGenotype] = field(default_factory=list)
    encoder: TransformerGenotype | None = None
    decoder: TransformerGenotype | None = None
    decoder_query_mode: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Genotype:
        data = dict(data)
        cells = [
            CellGenotype(
                cell_idx=c["cell_idx"],
                num_nodes=c["num_nodes"],
                edges=[EdgeGenotype(**e) for e in c.get("edges", [])],
                op_distribution=dict(c.get("op_distribution", {})),
            )
            for c in data.get("cells", [])
        ]
        encoder = data.get("encoder")
        decoder = data.get("decoder")
        return cls(
            arch_mode=data.get("arch_mode", "unknown"),
            input_dim=data.get("input_dim"),
            hidden_dim=data.get("hidden_dim"),
            seq_length=data.get("seq_length"),
            forecast_horizon=data.get("forecast_horizon"),
            norm=data.get("norm"),
            cells=cells,
            encoder=TransformerGenotype(**encoder) if encoder else None,
            decoder=TransformerGenotype(**decoder) if decoder else None,
            decoder_query_mode=data.get("decoder_query_mode"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Genotype:
        return cls.from_dict(json.loads(text))


# Earlier ``derive_final_architecture`` recorded ops by class name minus the
# ``Op`` suffix instead of by registry name. These are the only four where the
# two differ; mapping them keeps genotypes saved under the old scheme loadable.
_LEGACY_OP_NAMES = {
    "MLPMixer": "TimeMixer",
    "SwiGLUFFN": "SwiGLU",
    "GeGLUFFN": "GeGLU",
    "GatedGeLUFFN": "GatedGELU",
}


def resolve_op_name(name: str) -> str:
    """Map a recorded edge operation to its :data:`OP_REGISTRY` key."""
    if name in OP_REGISTRY:
        return name
    legacy = _LEGACY_OP_NAMES.get(name)
    if legacy is not None:
        return legacy
    raise ValueError(
        f"unknown operation {name!r}; expected one of {sorted(OP_REGISTRY)}"
    )


def apply_cell_choices(
    model: nn.Module, cells: list[CellGenotype], *, reuse_trained_ops: bool
) -> None:
    """Replace every mixed cell edge of *model* with its recorded op, in place.

    With ``reuse_trained_ops`` the op module is taken from the mixed edge
    itself (keeping its searched weights); otherwise a fresh op is built from
    the registry.
    """
    if len(model.cells) != len(cells):
        raise ValueError(
            f"genotype has {len(cells)} cells but model has "
            f"{len(model.cells)} — build the mixed model with the "
            "same num_cells as the search that produced this genotype."
        )
    for cell, cell_geno in zip(model.cells, cells):
        if len(cell.edges) != len(cell_geno.edges):
            raise ValueError(
                f"cell {cell_geno.cell_idx} has {len(cell_geno.edges)} "
                f"recorded edges but the mixed model's cell has "
                f"{len(cell.edges)} — num_nodes must match."
            )
        new_edges = nn.ModuleList()
        for edge, edge_geno in zip(cell.edges, cell_geno.edges):
            name = resolve_op_name(edge_geno.operation)
            if reuse_trained_ops:
                if name not in edge.available_ops:
                    raise ValueError(
                        f"cell {cell_geno.cell_idx} edge {edge_geno.edge_idx}: "
                        f"{name!r} is not in this edge's search space "
                        f"{list(edge.available_ops)}"
                    )
                op = edge.ops[edge.available_ops.index(name)]
            else:
                op = build_op(name, edge.input_dim, edge.latent_dim, edge.seq_length)
            new_edges.append(FixedOp(op))
        cell.edges = new_edges
        cell.selected_edge_specs = [asdict(e) for e in cell_geno.edges]
        cell.selected_op_distribution = dict(cell_geno.op_distribution)


def apply_norm_choice(model: nn.Module, norm: str) -> None:
    """Pin *model*'s searchable input normalization to *norm*, in place."""
    if not hasattr(model, "norm_alpha"):
        return
    if norm not in NORM_NAMES:
        raise ValueError(f"unknown normalization {norm!r}; expected one of {NORM_NAMES}")
    if model.norm_alpha.numel() != len(NORM_NAMES):
        raise ValueError(
            f"model has {model.norm_alpha.numel()} normalization modes, "
            f"expected {len(NORM_NAMES)} ({NORM_NAMES})"
        )
    with torch.no_grad():
        hard_logits = torch.full_like(model.norm_alpha, -12.0)
        hard_logits[NORM_NAMES.index(norm)] = 12.0
        model.norm_alpha.copy_(hard_logits)
    model.norm_alpha.requires_grad_(False)
    model.selected_norm = norm


def build_model_from_genotype(mixed_model: nn.Module, genotype: Genotype) -> nn.Module:
    """Apply a recorded :class:`Genotype` to a freshly built mixed model.

    *mixed_model* must be a searchable ``TimeSeriesDARTS`` built with the
    same search-space shape (``input_dim``, ``hidden_dim``, ``num_cells``,
    per-cell ``num_nodes``, ``arch_mode``) as the one *genotype* was derived
    from — this function does not re-derive that shape, it only forces the
    discrete choices onto it. Returns a new, discretized model; does not
    train it further.
    """
    new_model = mixed_model

    if hasattr(new_model, "cells"):
        apply_cell_choices(new_model, genotype.cells, reuse_trained_ops=False)
    if genotype.norm is not None:
        apply_norm_choice(new_model, genotype.norm)

    device = next(new_model.parameters()).device

    def _or_none(value: str | None) -> str | None:
        # "unknown" means extraction couldn't determine the original choice
        # (see finalization.py's ``_extract_*`` helpers) — forcing that
        # literal string would fail to match a real mode, so let the
        # converter fall back to its own auto-resolution instead.
        return None if value in (None, "unknown") else value

    encoder = getattr(new_model, "forecast_encoder", None)
    if encoder is not None and genotype.encoder is not None:
        new_model.forecast_encoder = ArchitectureConverter.create_fixed_encoder(
            encoder,
            self_attention_type=_or_none(genotype.encoder.self_attention_type),
            self_attention_position_mode=_or_none(
                genotype.encoder.self_attention_position
            ),
            ffn_mode=_or_none(genotype.encoder.ffn_mode),
            patching_mode=_or_none(genotype.encoder.patch_mode),
        ).to(device)

    decoder = getattr(new_model, "forecast_decoder", None)
    if decoder is not None and genotype.decoder is not None:
        new_model.forecast_decoder = ArchitectureConverter.create_fixed_decoder(
            decoder,
            self_attention_type=_or_none(genotype.decoder.self_attention_type),
            self_attention_position_mode=_or_none(
                genotype.decoder.self_attention_position
            ),
            cross_attention_type=_or_none(genotype.decoder.cross_attention_type),
            cross_attention_position_mode=_or_none(
                genotype.decoder.cross_attention_position
            ),
            ffn_mode=_or_none(genotype.decoder.ffn_mode),
            decode_style=_or_none(genotype.decoder.decode_style),
        ).to(device)

    if genotype.decoder_query_mode is not None and hasattr(
        new_model, "freeze_decoder_query_mode"
    ):
        try:
            new_model.freeze_decoder_query_mode(genotype.decoder_query_mode)
        except Exception:
            pass

    return new_model
