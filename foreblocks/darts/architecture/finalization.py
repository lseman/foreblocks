import copy
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_blocks import ArchitectureConverter
from .bb_transformers import (
    LightweightTransformerDecoder,
    LightweightTransformerEncoder,
)
from .operation_blocks import FixedOp


def _default_as_probability_vector(
    alpha_like: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Convert logits/probabilities to a normalized probability vector safely."""
    if alpha_like.numel() == 0:
        return alpha_like

    with torch.no_grad():
        flat = alpha_like.detach().reshape(-1)
        finite_ok = torch.isfinite(flat).all().item()
        in_range = flat.min().item() >= -1e-6 and flat.max().item() <= 1.0 + 1e-6
        sum_close = abs(flat.sum().item() - 1.0) <= 1e-4
        looks_like_probs = finite_ok and in_range and sum_close

    t = max(float(temperature), 1e-6)
    if looks_like_probs:
        probs = alpha_like.clamp_min(1e-8)
        if abs(t - 1.0) > 1e-8:
            probs = probs.pow(1.0 / t)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return F.softmax(alpha_like / t, dim=-1)


def derive_final_architecture(
    model: nn.Module,
    as_probability_vector_fn: None
    | (Callable[[torch.Tensor, float], torch.Tensor]) = None,
) -> nn.Module:
    """Create optimized model with fixed operations based on search results."""
    prob_fn = as_probability_vector_fn or _default_as_probability_vector

    new_model = copy.deepcopy(model)
    new_model.eval()

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

    def _mean_mode_probs(components, *, direct_attr: str, logits_attr: str, mode_names):
        probs = []
        mode_names = tuple(mode_names)
        for component in components:
            direct = getattr(component, direct_attr, None)
            if isinstance(direct, str) and direct in mode_names and direct != "auto":
                ref = next(component.parameters(), None)
                p = (
                    ref.new_zeros(len(mode_names))
                    if ref is not None
                    else torch.zeros(len(mode_names))
                )
                p[mode_names.index(direct)] = 1.0
                probs.append(p)
                continue
            logits = getattr(component, logits_attr, None)
            if isinstance(logits, torch.Tensor) and logits.numel() == len(mode_names):
                probs.append(F.softmax(logits.detach(), dim=0))
        if not probs:
            return None
        return torch.stack(probs, dim=0).mean(dim=0)

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

        if hasattr(edge, "alphas"):
            return prob_fn(edge.alphas)

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
            probs = _mean_mode_probs(
                components,
                direct_attr="attention_type",
                logits_attr="attn_alphas",
                mode_names=modes,
            )
            if probs is not None:
                top_idx = int(torch.argmax(probs).item())
                if 0 <= top_idx < len(modes):
                    return str(modes[top_idx])

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
        modes = getattr(
            components[0], "POSITION_MODES", ("rope", "alibi", "none", "seasonal")
        )
        probs = _mean_mode_probs(
            components,
            direct_attr="position_mode",
            logits_attr="position_alphas",
            mode_names=modes,
        )
        if probs is not None:
            return str(modes[int(torch.argmax(probs).item())])
        return "unknown"

    def _extract_ffn_mode(module_obj) -> str:
        if module_obj is None:
            return "unknown"
        components = _collect_layer_components(module_obj, "ffn")
        if not components:
            return "unknown"
        modes = getattr(components[0], "MODE_NAMES", ("swiglu", "moe"))
        probs = _mean_mode_probs(
            components,
            direct_attr="ffn_mode",
            logits_attr="ffn_alphas",
            mode_names=modes,
        )
        if probs is not None:
            return str(modes[int(torch.argmax(probs).item())])
        return "unknown"

    def _extract_decoder_style(module_obj) -> str:
        if module_obj is None:
            return "unknown"

        direct = getattr(module_obj, "decode_style", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct

        logits = getattr(module_obj, "decode_style_alphas", None)
        style_names = getattr(
            module_obj, "decode_style_names", ("autoregressive", "informer")
        )
        if isinstance(logits, torch.Tensor) and logits.numel() == len(style_names):
            probs = F.softmax(logits.detach(), dim=0)
            top_idx = int(torch.argmax(probs).item())
            if 0 <= top_idx < len(style_names):
                return str(style_names[top_idx])

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
        modes = getattr(
            components[0],
            "MODES",
            ("none", "sdp", "linear", "probsparse", "cosine", "local"),
        )
        probs = _mean_mode_probs(
            components,
            direct_attr="attention_type",
            logits_attr="attn_alphas",
            mode_names=modes,
        )
        if probs is not None:
            top_idx = int(torch.argmax(probs).item())
            if 0 <= top_idx < len(modes):
                return str(modes[top_idx])

        return "unknown"

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
        modes = getattr(
            components[0], "POSITION_MODES", ("rope", "alibi", "none", "seasonal")
        )
        probs = _mean_mode_probs(
            components,
            direct_attr="position_mode",
            logits_attr="position_alphas",
            mode_names=modes,
        )
        if probs is not None:
            return str(modes[int(torch.argmax(probs).item())])
        return "unknown"

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
        target_unique = min(max(2, int(math.ceil(n_edges * 0.5))), max(pool_size, 1))

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
        max_repeat = max(1, int(math.ceil(n_edges * 0.40)))
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

    print("🔧 Deriving final architecture...")

    if hasattr(new_model, "norm_alpha"):
        try:
            norm_weights = prob_fn(new_model.norm_alpha, 1.0).detach()
            if norm_weights.numel() > 0:
                norm_names = ["revin", "instance_norm", "identity"]
                top_idx = int(norm_weights.argmax().item())
                top_weight = float(norm_weights[top_idx].item())
                norm_name = (
                    norm_names[top_idx]
                    if top_idx < len(norm_names)
                    else f"norm_{top_idx}"
                )
                print(
                    f"   → Fixing Input Normalization: {norm_name} (weight: {top_weight:.3f})"
                )
                with torch.no_grad():
                    hard_logits = torch.full_like(new_model.norm_alpha, -12.0)
                    hard_logits[top_idx] = 12.0
                    new_model.norm_alpha.copy_(hard_logits)
                new_model.norm_alpha.requires_grad_(False)
                setattr(new_model, "selected_norm", norm_name)

            # Print decomposition status near normalization for clearer architecture trace.
            if hasattr(new_model, "forecast_encoder"):
                _print_decomposition_choice(new_model.forecast_encoder, "Encoder")
            if hasattr(new_model, "forecast_decoder"):
                _print_decomposition_choice(new_model.forecast_decoder, "Decoder")
        except Exception as e:
            print(f"Warning: Could not fix input normalization: {e}")

    if hasattr(new_model, "cells"):
        for cell_idx, cell in enumerate(new_model.cells):
            edge_weights = [_edge_selection_weights(edge) for edge in cell.edges]
            edge_importance = None
            if (
                hasattr(cell, "edge_importance")
                and getattr(cell, "edge_importance") is not None
                and len(getattr(cell, "edge_importance")) == len(cell.edges)
            ):
                with torch.no_grad():
                    edge_importance = (
                        torch.sigmoid(cell.edge_importance.detach()).cpu().tolist()
                    )
            selected_indices, op_counts = _assign_cell_edges_with_diversity(
                cell, edge_weights, edge_importance=edge_importance
            )

            new_edges = nn.ModuleList()
            for edge_idx, edge in enumerate(cell.edges):
                top_op_idx = selected_indices[edge_idx]
                weights = edge_weights[edge_idx]
                confidence = (
                    float(weights[top_op_idx].item())
                    if weights is not None and weights.numel() > top_op_idx
                    else float("nan")
                )
                top_op = edge.ops[top_op_idx]
                print(
                    f"   Cell {cell_idx}, Edge {edge_idx}: {type(top_op).__name__} "
                    f"(weight: {confidence:.3f})"
                )

                fixed_edge = FixedOp(top_op)
                new_edges.append(fixed_edge)

            unique_ops = len(op_counts)
            print(
                f"   Cell {cell_idx}: unique ops selected={unique_ops}, "
                f"distribution={op_counts}"
            )
            cell.edges = new_edges

    device = next(new_model.parameters()).device
    arch_mode = str(getattr(new_model, "arch_mode", "unknown"))
    printed_encoder_fix = False
    printed_decoder_fix = False
    printed_attention_fix = False

    if (
        hasattr(new_model, "forecast_encoder")
        and new_model.forecast_encoder is not None
    ):
        try:
            top_encoder = getattr(new_model.forecast_encoder, "transformer", None)
            if top_encoder is None:
                top_encoder = getattr(new_model.forecast_encoder, "rnn", None)
            if top_encoder is None:
                raise ValueError("Could not locate forecast encoder transformer")

            print(f"   → Fixing Forecast Encoder: {type(top_encoder).__name__}")
            if isinstance(top_encoder, LightweightTransformerEncoder):
                print(
                    "   → Encoder Self-Attention: "
                    f"{_extract_self_attention_type(top_encoder)}"
                )
                print(
                    "   → Encoder Attention Position: "
                    f"{_extract_self_attention_position(top_encoder)}"
                )
                print(
                    "   → Encoder Tokenizer: "
                    f"{_extract_encoder_patch_mode(top_encoder)}"
                )
                print(f"   → Encoder FFN: {_extract_ffn_mode(top_encoder)}")
            printed_encoder_fix = True

            new_model.forecast_encoder = ArchitectureConverter.create_fixed_encoder(
                new_model.forecast_encoder,
            ).to(device)

        except Exception as e:
            print(f"Warning: Could not fix encoder architecture: {e}")
            print("Falling back to weight fixing...")
            ArchitectureConverter.fix_mixed_weights(new_model.forecast_encoder)
            printed_encoder_fix = True

    if (
        hasattr(new_model, "forecast_decoder")
        and new_model.forecast_decoder is not None
    ):
        try:
            top_decoder = getattr(new_model.forecast_decoder, "transformer", None)
            if top_decoder is None:
                top_decoder = getattr(new_model.forecast_decoder, "rnn", None)
            if top_decoder is None:
                raise ValueError("Could not locate forecast decoder transformer")

            print(f"   → Fixing Forecast Decoder: {type(top_decoder).__name__}")
            if isinstance(top_decoder, LightweightTransformerDecoder):
                print(
                    "   → Decoder Self-Attention: "
                    f"{_extract_self_attention_type(top_decoder)}"
                )
                print(
                    "   → Decoder Attention Position: "
                    f"{_extract_self_attention_position(top_decoder)}"
                )
                print(
                    "   → Decoder Cross-Attention: "
                    f"{_extract_cross_attention_type(new_model.forecast_decoder)}"
                )
                print(
                    "   → Decoder Cross Position: "
                    f"{_extract_cross_attention_position(new_model.forecast_decoder)}"
                )
                print(f"   → Decoder FFN: {_extract_ffn_mode(top_decoder)}")
            print(
                "   → Decoder Style: "
                f"{_extract_decoder_style(new_model.forecast_decoder)}"
            )
            print(
                "   → Decoder Query Generator: "
                f"{_extract_decoder_query_mode(new_model)}"
            )
            printed_decoder_fix = True

            cross_attention_type = _extract_cross_attention_type(
                new_model.forecast_decoder
            )
            print(f"   → Fixing Decoder Cross-Attention: {cross_attention_type}")
            printed_attention_fix = True

            new_model.forecast_decoder = ArchitectureConverter.create_fixed_decoder(
                new_model.forecast_decoder,
                cross_attention_type=cross_attention_type,
            ).to(device)

        except Exception as e:
            print(f"Warning: Could not fix decoder architecture: {e}")
            print("Falling back to weight fixing...")
            ArchitectureConverter.fix_mixed_weights(new_model.forecast_decoder)
            printed_decoder_fix = True

    # Always report block-fix status, even when topology omits modules.
    if not printed_encoder_fix:
        enc_obj = getattr(new_model, "forecast_encoder", None)
        if enc_obj is None:
            print(f"   → Fixing Forecast Encoder: not used in arch_mode={arch_mode}")
        else:
            print(
                "   → Fixing Forecast Encoder: module present but no searchable logits"
            )

    if not printed_decoder_fix:
        dec_obj = getattr(new_model, "forecast_decoder", None)
        if dec_obj is None:
            print(f"   → Fixing Forecast Decoder: not used in arch_mode={arch_mode}")
        else:
            print(
                "   → Fixing Forecast Decoder: module present but no searchable logits"
            )

    if not printed_attention_fix:
        dec_obj = getattr(new_model, "forecast_decoder", None)
        if dec_obj is None:
            print(
                f"   → Fixing Decoder Cross-Attention: not used in arch_mode={arch_mode}"
            )
        else:
            print("   → Fixing Decoder Cross-Attention: selection unavailable")

    if hasattr(new_model, "freeze_decoder_query_mode"):
        try:
            new_model.freeze_decoder_query_mode(_extract_decoder_query_mode(new_model))
        except Exception:
            pass

    print("✓ Architecture derivation completed")
    return new_model
