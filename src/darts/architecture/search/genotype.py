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

from .converter import ArchitectureConverter
from ..ops.fixed import FixedOp
from ..ops.registry import build_op

__all__ = ["EdgeGenotype", "CellGenotype", "TransformerGenotype", "Genotype", "build_model_from_genotype"]

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
    def from_dict(cls, data: dict[str, Any]) -> "Genotype":
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
    def from_json(cls, text: str) -> "Genotype":
        return cls.from_dict(json.loads(text))


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
        if len(new_model.cells) != len(genotype.cells):
            raise ValueError(
                f"genotype has {len(genotype.cells)} cells but model has "
                f"{len(new_model.cells)} — build the mixed model with the "
                "same num_cells as the search that produced this genotype."
            )
        for cell, cell_geno in zip(new_model.cells, genotype.cells):
            if len(cell.edges) != len(cell_geno.edges):
                raise ValueError(
                    f"cell {cell_geno.cell_idx} has {len(cell_geno.edges)} "
                    f"recorded edges but the mixed model's cell has "
                    f"{len(cell.edges)} — num_nodes must match."
                )
            new_edges = nn.ModuleList()
            for edge, edge_geno in zip(cell.edges, cell_geno.edges):
                op = build_op(
                    edge_geno.operation,
                    edge.input_dim,
                    edge.latent_dim,
                    edge.seq_length,
                )
                new_edges.append(FixedOp(op))
            cell.edges = new_edges
            cell.selected_edge_specs = [asdict(e) for e in cell_geno.edges]
            cell.selected_op_distribution = dict(cell_geno.op_distribution)

    if genotype.norm is not None and hasattr(new_model, "norm_alpha"):
        norm_names = ["revin", "instance_norm", "identity"]
        if genotype.norm in norm_names:
            top_idx = norm_names.index(genotype.norm)
            with torch.no_grad():
                hard_logits = torch.full_like(new_model.norm_alpha, -12.0)
                hard_logits[top_idx] = 12.0
                new_model.norm_alpha.copy_(hard_logits)
            new_model.norm_alpha.requires_grad_(False)
            new_model.selected_norm = genotype.norm

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
