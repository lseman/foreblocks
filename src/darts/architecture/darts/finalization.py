"""Discretize a searched DARTS model into a fixed architecture.

Two separate steps:

* :func:`select_architecture` reads the searched model and decides the
  discrete architecture — per-edge cell ops (diversity-aware), normalization
  mode, transformer sub-choices — returning a :class:`Genotype`. It does not
  modify the model.
* :func:`derive_final_architecture` selects, then converts a copy of the
  model to that architecture, keeping the searched weights of every chosen
  op. :func:`genotype.build_model_from_genotype` is the other conversion,
  for applying a saved genotype to a freshly built model.
"""

import copy
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.tensors import as_probability_vector as _default_as_probability_vector
from ..blocks.transformers import (
    LightweightTransformerDecoder,
    LightweightTransformerEncoder,
)
from ..common.inspector import _mean_softmax_top, _softmax_top
from ..common.norms import NORM_NAMES
from .converter import ArchitectureConverter
from .genotype import (
    CellGenotype,
    EdgeGenotype,
    Genotype,
    TransformerGenotype,
    apply_cell_choices,
    apply_norm_choice,
)

__all__ = ["select_architecture", "derive_final_architecture"]


def _layer_component(layer, key: str):
    if isinstance(layer, dict):
        return layer.get(key)
    if hasattr(layer, "get"):
        try:
            return layer.get(key)
        except Exception:
            pass
    if hasattr(layer, "__contains__") and key in layer:
        return layer[key]
    return None


def _collect_layer_components(module_obj, key: str):
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return []
    out = []
    for layer in layers:
        item = _layer_component(layer, key)
        if item is not None:
            out.append(item)
    return out


def _edge_selection_weights(edge):
    if hasattr(edge, "_get_weights") and hasattr(edge, "ops"):
        try:
            routed = edge._get_weights(top_k=None)
            if routed:
                weights = torch.zeros(
                    len(edge.ops),
                    device=routed[0][1].device,
                    dtype=routed[0][1].dtype,
                )
                for op_idx, weight in routed:
                    weights[op_idx] = weight
                if weights.sum().item() > 0:
                    weights = weights / weights.sum().clamp_min(1e-8)
                    return weights
        except Exception:
            pass

    if hasattr(edge, "get_alphas"):
        return edge.get_alphas(detach=True)

    return None


def _extract_self_attention_type(module_obj) -> str:
    """Best-effort extraction of configured self-attention type."""
    if module_obj is None:
        return "unknown"

    direct = getattr(module_obj, "self_attention_type", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    components = _collect_layer_components(module_obj, "self_attn")
    if components:
        modes = getattr(
            components[0],
            "MODES",
            ("sdp", "linear", "probsparse", "cosine", "local"),
        )
        top = _mean_softmax_top(
            components,
            direct_attr="attention_type",
            logits_attr="attn_alphas",
            mode_names=modes,
        )
        if top is not None:
            return str(top)

    return "unknown"


def _extract_encoder_patch_mode(module_obj) -> str:
    if module_obj is None:
        return "unknown"

    direct = getattr(module_obj, "patching_mode", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    logits = getattr(module_obj, "patch_alpha_logits", None)
    mode_names = getattr(module_obj, "patch_mode_names", ("direct", "patch"))
    if isinstance(logits, torch.Tensor) and logits.numel() == len(mode_names):
        probs = F.softmax(logits.detach(), dim=0)
        top_idx = int(torch.argmax(probs).item())
        if 0 <= top_idx < len(mode_names):
            return str(mode_names[top_idx])

    resolver = getattr(module_obj, "resolve_patch_mode", None)
    if callable(resolver):
        try:
            return str(resolver())
        except Exception:
            pass

    return "unknown"


def _extract_self_attention_position(module_obj) -> str:
    if module_obj is None:
        return "unknown"
    components = _collect_layer_components(module_obj, "self_attn")
    if not components:
        return "unknown"
    modes = tuple(getattr(components[0], "POSITION_MODES",
        ("rope", "alibi", "none", "seasonal", "sinusoidal", "learned", "relative")))
    top = _mean_softmax_top(
        components,
        direct_attr="position_mode",
        logits_attr="position_alphas",
        mode_names=modes,
    )
    return str(top) if top is not None else "unknown"


def _extract_ffn_mode(module_obj) -> str:
    if module_obj is None:
        return "unknown"
    components = _collect_layer_components(module_obj, "ffn")
    if not components:
        return "unknown"
    modes = tuple(getattr(components[0], "MODE_NAMES", ("swiglu", "moe")))
    top = _mean_softmax_top(
        components,
        direct_attr="ffn_mode",
        logits_attr="ffn_alphas",
        mode_names=modes,
    )
    return str(top) if top is not None else "unknown"


def _extract_decoder_style(module_obj) -> str:
    if module_obj is None:
        return "unknown"

    direct = getattr(module_obj, "decode_style", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    logits = getattr(module_obj, "decode_style_alphas", None)
    style_names = tuple(
        getattr(module_obj, "decode_style_names",
            ("autoregressive", "informer", "autoformer"))
    )
    top = _softmax_top(logits, style_names)
    if top is not None:
        return top

    resolver = getattr(module_obj, "resolve_decode_style", None)
    if callable(resolver):
        try:
            return str(resolver())
        except Exception:
            pass

    return "unknown"


def _extract_cross_attention_type(module_obj) -> str:
    if module_obj is None:
        return "unknown"

    submodule = getattr(module_obj, "transformer", None)
    if submodule is None:
        submodule = getattr(module_obj, "rnn", None)
    if submodule is None:
        return "unknown"

    direct = getattr(submodule, "cross_attention_type", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    components = _collect_layer_components(submodule, "cross_attn")
    if not components:
        return "unknown"
    modes = tuple(getattr(components[0], "MODES", ("none", "sdp", "linear", "probsparse", "cosine", "local")))
    top = _mean_softmax_top(
        components,
        direct_attr="attention_type",
        logits_attr="attn_alphas",
        mode_names=modes,
    )
    return str(top) if top is not None else "unknown"


def _extract_cross_attention_position(module_obj) -> str:
    if module_obj is None:
        return "unknown"
    submodule = getattr(module_obj, "transformer", None)
    if submodule is None:
        submodule = getattr(module_obj, "rnn", None)
    if submodule is None:
        return "unknown"
    components = _collect_layer_components(submodule, "cross_attn")
    if not components:
        return "unknown"
    modes = tuple(getattr(components[0], "POSITION_MODES",
        ("rope", "alibi", "none", "seasonal", "sinusoidal", "learned", "relative")))
    top = _mean_softmax_top(
        components,
        direct_attr="position_mode",
        logits_attr="position_alphas",
        mode_names=modes,
    )
    return str(top) if top is not None else "unknown"


def _extract_decoder_query_mode(model_obj) -> str:
    if model_obj is None:
        return "unknown"
    direct = getattr(model_obj, "decoder_query_mode", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct
    logits = getattr(model_obj, "decoder_query_alphas", None)
    names = getattr(
        model_obj,
        "decoder_query_mode_names",
        (
            "repeat_last",
            "zeros",
            "learned_horizon_queries",
            "shifted_target",
            "future_covariate_queries",
        ),
    )
    if isinstance(logits, torch.Tensor) and logits.numel() == len(names):
        probs = F.softmax(logits.detach(), dim=0)
        return str(names[int(torch.argmax(probs).item())])
    resolver = getattr(model_obj, "resolve_decoder_query_mode", None)
    if callable(resolver):
        try:
            return str(resolver())
        except Exception:
            pass
    return "unknown"


def _assign_cell_edges_with_diversity(
    cell: nn.Module,
    edge_weight_list: list[torch.Tensor | None],
    edge_importance: list[float] | None = None,
) -> tuple[list[int], dict[str, int]]:
    """
    Diversity-aware edge assignment:
    1) first cover a target number of unique ops inside the cell
    2) then fill remaining edges with repetition penalties
    """
    n_edges = len(getattr(cell, "edges", []))
    if n_edges == 0:
        return [], {}
    if edge_importance is None or len(edge_importance) != n_edges:
        edge_importance = [1.0] * n_edges

    assignments: list[int | None] = [None] * n_edges
    op_counts_by_name: dict[str, int] = {}
    used_unique_ops = set()

    # Estimate available operation names for target diversity sizing.
    available_names = set()
    for edge in cell.edges:
        edge_names = getattr(edge, "available_ops", None)
        if edge_names is not None:
            available_names.update(edge_names)

    unique_pool = [name for name in available_names if name != "Identity"]
    pool_size = len(unique_pool) if len(unique_pool) > 0 else len(available_names)

    # Target unique ops: use a sub-linear scaling that avoids over-diversifying
    # small cells (where n_edges=2 forces 100% unique) and under-diversifying
    # large cells (where n_edges=12 only gets 6 unique).
    # Formula: sqrt(pool_size) clamped to [2, n_edges], giving ~3 for
    # small cells and ~5 for medium cells.
    target_unique = min(
        max(2, int(math.sqrt(pool_size))),
        max(pool_size, 1),
        n_edges,
    )
    # But never more than 70% of edges (reserve room for repetition).
    target_unique = min(target_unique, max(2, int(n_edges * 0.7)))

    # Pass 1: cover unique operations with highest-confidence (edge, op) pairs.
    candidates = []
    for edge_idx, weights in enumerate(edge_weight_list):
        if weights is None or weights.numel() == 0:
            continue
        for op_idx, w in enumerate(weights):
            score = float(w.item()) * (
                0.65 + 0.35 * float(edge_importance[edge_idx])
            )
            candidates.append((score, edge_idx, int(op_idx)))
    candidates.sort(key=lambda x: x[0], reverse=True)

    for raw_score, edge_idx, op_idx in candidates:
        if len(used_unique_ops) >= target_unique:
            break
        if assignments[edge_idx] is not None:
            continue
        edge = cell.edges[edge_idx]
        op_name = edge.available_ops[op_idx]
        if op_name in used_unique_ops:
            continue
        if op_name == "Identity" and len(used_unique_ops) < max(
            target_unique - 1, 1
        ):
            continue

        assignments[edge_idx] = op_idx
        used_unique_ops.add(op_name)
        op_counts_by_name[op_name] = op_counts_by_name.get(op_name, 0) + 1

    # Pass 2: fill remaining edges with repetition-aware penalties.
    remaining_edges = [idx for idx in range(n_edges) if assignments[idx] is None]
    # Prefer to place higher-confidence undecided edges first.
    remaining_edges.sort(
        key=lambda e_idx: (
            float(edge_weight_list[e_idx].max().item())
            * float(edge_importance[e_idx])
            if edge_weight_list[e_idx] is not None
            and edge_weight_list[e_idx].numel() > 0
            else -1e9
        ),
        reverse=True,
    )

    # Max copies per op: divide edges evenly across unique ops, with a
    # floor of 1 so every op gets at least one chance.
    max_per_op = max(1, int(math.ceil(n_edges / max(target_unique, 1))))
    repeat_penalty = 0.30
    identity_extra_penalty = 0.40

    for edge_idx in remaining_edges:
        edge = cell.edges[edge_idx]
        weights = edge_weight_list[edge_idx]
        if weights is None or weights.numel() == 0:
            chosen_idx = 0
        else:
            adjusted = weights.clone()
            for op_idx, op_name in enumerate(edge.available_ops):
                count = op_counts_by_name.get(op_name, 0)
                penalty = repeat_penalty * float(count)
                if op_name == "Identity":
                    penalty += identity_extra_penalty * float(count)
                    if float(edge_importance[edge_idx]) < 0.35:
                        penalty -= 0.15
                adjusted[op_idx] = adjusted[op_idx] - penalty

            ranked = torch.argsort(adjusted, descending=True).tolist()
            chosen_idx = int(ranked[0])
            for candidate_idx in ranked:
                candidate_name = edge.available_ops[int(candidate_idx)]
                c = op_counts_by_name.get(candidate_name, 0)
                if c < max_per_op or int(candidate_idx) == ranked[0]:
                    chosen_idx = int(candidate_idx)
                    break

        assignments[edge_idx] = chosen_idx
        chosen_name = edge.available_ops[chosen_idx]
        op_counts_by_name[chosen_name] = op_counts_by_name.get(chosen_name, 0) + 1

    # Pass 3: hard cap repeated operations per cell by reassigning the weakest
    # edges first to their best alternatives.
    # Max repeat should not exceed max_per_op and should scale with target_unique.
    max_repeat = min(max_per_op, max(1, int(math.ceil(n_edges * 0.5))))
    for _ in range(n_edges * 2):
        changed = False
        by_freq = sorted(
            op_counts_by_name.items(), key=lambda kv: kv[1], reverse=True
        )
        for op_name, count in by_freq:
            if count <= max_repeat:
                continue

            holders = []
            for edge_idx, op_idx in enumerate(assignments):
                if op_idx is None:
                    continue
                edge = cell.edges[edge_idx]
                chosen_name = edge.available_ops[int(op_idx)]
                if chosen_name != op_name:
                    continue

                weights = edge_weight_list[edge_idx]
                confidence = (
                    float(weights[int(op_idx)].item())
                    if weights is not None and weights.numel() > int(op_idx)
                    else -1e9
                )
                holders.append(
                    (confidence * float(edge_importance[edge_idx]), edge_idx)
                )

            holders.sort(key=lambda x: x[0])
            replaced = False
            for _, edge_idx in holders:
                edge = cell.edges[edge_idx]
                weights = edge_weight_list[edge_idx]
                if weights is None or weights.numel() == 0:
                    ranked = list(range(len(edge.available_ops)))
                else:
                    ranked = torch.argsort(weights, descending=True).tolist()

                for cand_idx in ranked:
                    cand_idx = int(cand_idx)
                    cand_name = edge.available_ops[cand_idx]
                    if cand_name == op_name:
                        continue
                    cand_count = op_counts_by_name.get(cand_name, 0)
                    if cand_count >= max_repeat:
                        continue
                    if (
                        cand_name == "Identity"
                        and float(edge_importance[edge_idx]) > 0.65
                    ):
                        continue

                    old_idx = int(assignments[edge_idx])
                    old_name = edge.available_ops[old_idx]
                    assignments[edge_idx] = cand_idx

                    op_counts_by_name[old_name] = (
                        op_counts_by_name.get(old_name, 1) - 1
                    )
                    if op_counts_by_name[old_name] <= 0:
                        op_counts_by_name.pop(old_name, None)
                    op_counts_by_name[cand_name] = cand_count + 1
                    replaced = True
                    changed = True
                    break

                if replaced:
                    break

        if not changed:
            break

    final_assignments = [int(a) if a is not None else 0 for a in assignments]
    return final_assignments, op_counts_by_name


def _print_decomposition_choice(module: nn.Module, label: str):
    """Print decomposition status and dominant mode for a mixed block."""
    decomp = getattr(module, "searchable_decomp", None)
    if decomp is None:
        print(f"   → {label} Decomposition: disabled")
        return

    logits = getattr(decomp, "alpha_logits", None)
    if logits is None or logits.numel() == 0:
        print(f"   → {label} Decomposition: enabled (weights unavailable)")
        return

    with torch.no_grad():
        weights = F.softmax(logits.detach(), dim=0)
        top_idx = int(weights.argmax().item())
        top_weight = float(weights[top_idx].item())
        mode_names = [
            "none",
            "moving_avg_trend",
            "seasonal_residual",
            "learnable_filter",
        ]
        mode = (
            mode_names[top_idx] if top_idx < len(mode_names) else f"mode_{top_idx}"
        )
        enabled = mode != "none"
        status = "enabled" if enabled else "disabled"
        print(
            f"   → {label} Decomposition: {status} ({mode}, weight: {top_weight:.3f})"
        )


def _edge_endpoints(edge_idx: int) -> tuple[int, int]:
    """(source, target) node of the *edge_idx*-th edge in a DARTS cell."""
    target = 1
    while edge_idx >= target:
        edge_idx -= target
        target += 1
    return edge_idx, target


def _select_norm(model: nn.Module, prob_fn) -> str | None:
    alpha = getattr(model, "norm_alpha", None)
    if alpha is None:
        return None
    if alpha.numel() != len(NORM_NAMES):
        raise ValueError(
            f"model has {alpha.numel()} normalization modes, "
            f"expected {len(NORM_NAMES)} ({NORM_NAMES})"
        )
    norm_weights = prob_fn(alpha, 1.0).detach()
    return NORM_NAMES[int(norm_weights.argmax().item())]


def _select_cell(cell_idx: int, cell: nn.Module) -> CellGenotype:
    edge_weights = [_edge_selection_weights(edge) for edge in cell.edges]
    edge_importance = None
    if (
        hasattr(cell, "edge_importance")
        and cell.edge_importance is not None
        and len(cell.edge_importance) == len(cell.edges)
    ):
        with torch.no_grad():
            edge_importance = torch.sigmoid(cell.edge_importance.detach()).cpu().tolist()
    selected_indices, op_counts = _assign_cell_edges_with_diversity(
        cell, edge_weights, edge_importance=edge_importance
    )

    edges = []
    for edge_idx, edge in enumerate(cell.edges):
        top_op_idx = selected_indices[edge_idx]
        weights = edge_weights[edge_idx]
        confidence = (
            float(weights[top_op_idx].item())
            if weights is not None and weights.numel() > top_op_idx
            else float("nan")
        )
        source, target = _edge_endpoints(edge_idx)
        edges.append(
            EdgeGenotype(
                edge_idx=edge_idx,
                source=source,
                target=target,
                # Registry name, so build_op can reconstruct the op.
                operation=edge.available_ops[top_op_idx],
                op_weight=confidence if math.isfinite(confidence) else None,
                importance=(
                    float(edge_importance[edge_idx])
                    if edge_importance is not None
                    else 1.0
                ),
            )
        )
    return CellGenotype(
        cell_idx=cell_idx,
        num_nodes=int(getattr(cell, "num_nodes", 0)),
        edges=edges,
        op_distribution=dict(op_counts),
    )


def _forecast_submodule(block: nn.Module | None) -> nn.Module | None:
    if block is None:
        return None
    sub = getattr(block, "transformer", None)
    return sub if sub is not None else getattr(block, "rnn", None)


def _select_encoder(model: nn.Module) -> TransformerGenotype | None:
    top_encoder = _forecast_submodule(getattr(model, "forecast_encoder", None))
    if not isinstance(top_encoder, LightweightTransformerEncoder):
        return None
    return TransformerGenotype(
        self_attention_type=_extract_self_attention_type(top_encoder),
        self_attention_position=_extract_self_attention_position(top_encoder),
        patch_mode=_extract_encoder_patch_mode(top_encoder),
        ffn_mode=_extract_ffn_mode(top_encoder),
    )


def _select_decoder(model: nn.Module) -> TransformerGenotype | None:
    decoder = getattr(model, "forecast_decoder", None)
    top_decoder = _forecast_submodule(decoder)
    if top_decoder is None:
        return None
    geno = TransformerGenotype(decode_style=_extract_decoder_style(decoder))
    if isinstance(top_decoder, LightweightTransformerDecoder):
        geno.self_attention_type = _extract_self_attention_type(top_decoder)
        geno.self_attention_position = _extract_self_attention_position(top_decoder)
        geno.cross_attention_type = _extract_cross_attention_type(decoder)
        geno.cross_attention_position = _extract_cross_attention_position(decoder)
        geno.ffn_mode = _extract_ffn_mode(top_decoder)
    return geno


def select_architecture(
    model: nn.Module,
    as_probability_vector_fn: Callable[[torch.Tensor, float], torch.Tensor]
    | None = None,
) -> Genotype:
    """Decide the discrete architecture a searched model converged to.

    Reads architecture weights only; *model* is not modified. The returned
    :class:`Genotype` names cell ops by their :data:`OP_REGISTRY` key and the
    normalization mode by its :data:`NORM_NAMES` entry, so it can be applied
    with :func:`genotype.build_model_from_genotype`.
    """
    # Edge routing weights differ between train and eval mode; select from the
    # eval-mode weights, then restore each module's own training flag.
    training_flags = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        return _select_architecture(model, as_probability_vector_fn)
    finally:
        for module, was_training in training_flags:
            module.training = was_training


def _select_architecture(model: nn.Module, as_probability_vector_fn) -> Genotype:
    prob_fn = as_probability_vector_fn or _default_as_probability_vector
    decoder_query_mode = _extract_decoder_query_mode(model)
    return Genotype(
        arch_mode=str(getattr(model, "arch_mode", "unknown")),
        input_dim=getattr(model, "input_dim", None),
        hidden_dim=getattr(model, "hidden_dim", None),
        seq_length=getattr(model, "seq_length", None),
        forecast_horizon=getattr(model, "forecast_horizon", None),
        norm=_select_norm(model, prob_fn),
        cells=[
            _select_cell(cell_idx, cell)
            for cell_idx, cell in enumerate(getattr(model, "cells", []))
        ],
        encoder=_select_encoder(model),
        decoder=_select_decoder(model),
        decoder_query_mode=(
            decoder_query_mode if decoder_query_mode != "unknown" else None
        ),
    )


def _print_genotype(genotype: Genotype, model: nn.Module) -> None:
    if genotype.norm is not None:
        print(f"   → Fixing Input Normalization: {genotype.norm}")
        if hasattr(model, "forecast_encoder"):
            _print_decomposition_choice(model.forecast_encoder, "Encoder")
        if hasattr(model, "forecast_decoder"):
            _print_decomposition_choice(model.forecast_decoder, "Decoder")

    for cell in genotype.cells:
        for edge in cell.edges:
            weight = edge.op_weight if edge.op_weight is not None else float("nan")
            print(
                f"   Cell {cell.cell_idx}, Edge {edge.edge_idx}: {edge.operation} "
                f"(weight: {weight:.3f})"
            )
        print(
            f"   Cell {cell.cell_idx}: unique ops selected={len(cell.op_distribution)}, "
            f"distribution={cell.op_distribution}"
        )

    enc = genotype.encoder
    if enc is not None:
        print(f"   → Encoder Self-Attention: {enc.self_attention_type}")
        print(f"   → Encoder Attention Position: {enc.self_attention_position}")
        print(f"   → Encoder Tokenizer: {enc.patch_mode}")
        print(f"   → Encoder FFN: {enc.ffn_mode}")

    dec = genotype.decoder
    if dec is not None:
        if dec.self_attention_type != "unknown" or dec.cross_attention_type is not None:
            print(f"   → Decoder Self-Attention: {dec.self_attention_type}")
            print(f"   → Decoder Attention Position: {dec.self_attention_position}")
            print(f"   → Decoder Cross-Attention: {dec.cross_attention_type}")
            print(f"   → Decoder Cross Position: {dec.cross_attention_position}")
            print(f"   → Decoder FFN: {dec.ffn_mode}")
        print(f"   → Decoder Style: {dec.decode_style}")
        print(f"   → Decoder Query Generator: {genotype.decoder_query_mode or 'unknown'}")


def derive_final_architecture(
    model: nn.Module,
    as_probability_vector_fn: Callable[[torch.Tensor, float], torch.Tensor]
    | None = None,
    *,
    return_genotype: bool = False,
) -> nn.Module | tuple[nn.Module, Genotype]:
    """Create optimized model with fixed operations based on search results.

    Runs :func:`select_architecture`, then converts a deep copy of *model* to
    the selected architecture, reusing the searched weights of every chosen
    op. When ``return_genotype`` is True, also returns that :class:`Genotype`
    so the architecture can be exported, inspected, or rebuilt later via
    :func:`genotype.build_model_from_genotype` without re-running search.
    """
    new_model = copy.deepcopy(model)
    new_model.eval()

    print("🔧 Deriving final architecture...")
    genotype = select_architecture(new_model, as_probability_vector_fn)
    _print_genotype(genotype, new_model)

    if genotype.norm is not None:
        apply_norm_choice(new_model, genotype.norm)
    if hasattr(new_model, "cells"):
        apply_cell_choices(new_model, genotype.cells, reuse_trained_ops=True)

    device = next(new_model.parameters()).device
    arch_mode = genotype.arch_mode

    # Transformer blocks: the converter resolves sub-choices from the trained
    # alphas itself (and keeps soft choices for non-GDAS models), so only the
    # cross-attention choice is passed through, as before.
    encoder = getattr(new_model, "forecast_encoder", None)
    if encoder is not None:
        try:
            top_encoder = _forecast_submodule(encoder)
            if top_encoder is None:
                raise ValueError("Could not locate forecast encoder transformer")
            print(f"   → Fixing Forecast Encoder: {type(top_encoder).__name__}")
            new_model.forecast_encoder = ArchitectureConverter.create_fixed_encoder(
                encoder,
            ).to(device)
        except Exception as e:
            print(f"Warning: Could not fix encoder architecture: {e}")
            print("Falling back to weight fixing...")
            ArchitectureConverter.fix_mixed_weights(encoder)
    else:
        print(f"   → Fixing Forecast Encoder: not used in arch_mode={arch_mode}")

    decoder = getattr(new_model, "forecast_decoder", None)
    if decoder is not None:
        try:
            top_decoder = _forecast_submodule(decoder)
            if top_decoder is None:
                raise ValueError("Could not locate forecast decoder transformer")
            print(f"   → Fixing Forecast Decoder: {type(top_decoder).__name__}")
            cross_attention_type = (
                genotype.decoder.cross_attention_type
                if genotype.decoder is not None
                and genotype.decoder.cross_attention_type is not None
                else _extract_cross_attention_type(decoder)
            )
            print(f"   → Fixing Decoder Cross-Attention: {cross_attention_type}")
            new_model.forecast_decoder = ArchitectureConverter.create_fixed_decoder(
                decoder,
                cross_attention_type=cross_attention_type,
            ).to(device)
        except Exception as e:
            print(f"Warning: Could not fix decoder architecture: {e}")
            print("Falling back to weight fixing...")
            ArchitectureConverter.fix_mixed_weights(decoder)
    else:
        print(f"   → Fixing Forecast Decoder: not used in arch_mode={arch_mode}")
        print(f"   → Fixing Decoder Cross-Attention: not used in arch_mode={arch_mode}")

    if genotype.decoder_query_mode is not None and hasattr(
        new_model, "freeze_decoder_query_mode"
    ):
        try:
            new_model.freeze_decoder_query_mode(genotype.decoder_query_mode)
        except Exception:
            pass

    print("✓ Architecture derivation completed")

    if not return_genotype:
        return new_model
    return new_model, genotype
