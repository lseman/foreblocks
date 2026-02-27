import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_blocks import ArchitectureConverter
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
    as_probability_vector_fn: Optional[
        Callable[[torch.Tensor, float], torch.Tensor]
    ] = None,
) -> nn.Module:
    """Create optimized model with fixed operations based on search results."""
    prob_fn = as_probability_vector_fn or _default_as_probability_vector

    new_model = copy.deepcopy(model)
    new_model.eval()
    variant_attrs = {
        "encoder": {0: "lstm", 1: "gru", 2: "transformer"},
        "decoder": {0: "lstm", 1: "gru", 2: "transformer"},
    }

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

    def _pick_variant(module, top_idx: int, collection_attr: str, role: str):
        if hasattr(module, collection_attr):
            variants = getattr(module, collection_attr)
            if top_idx < len(variants):
                return variants[top_idx]
        attr_name = variant_attrs[role].get(top_idx)
        if attr_name and hasattr(module, attr_name):
            return getattr(module, attr_name)
        raise ValueError(f"Could not find {role} for index {top_idx}")

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
        except Exception as e:
            print(f"Warning: Could not fix input normalization: {e}")

    if hasattr(new_model, "cells"):
        for cell_idx, cell in enumerate(new_model.cells):
            new_edges = nn.ModuleList()
            cell_selected_counts = {}
            for edge_idx, edge in enumerate(cell.edges):
                weights = _edge_selection_weights(edge)
                if weights is None:
                    top_op_idx = 0
                    confidence = float("nan")
                else:
                    penalty_strength = 0.10
                    adjusted = weights.clone()
                    if adjusted.numel() > 1:
                        counts = torch.zeros_like(adjusted)
                        for op_idx, cnt in cell_selected_counts.items():
                            if 0 <= op_idx < counts.numel():
                                counts[op_idx] = float(cnt)
                        adjusted = adjusted - penalty_strength * counts

                    top_op_idx = adjusted.argmax().item()
                    confidence = weights[top_op_idx].item()
                top_op = edge.ops[top_op_idx]
                cell_selected_counts[top_op_idx] = (
                    cell_selected_counts.get(top_op_idx, 0) + 1
                )

                print(
                    f"   Cell {cell_idx}, Edge {edge_idx}: {type(top_op).__name__} "
                    f"(weight: {confidence:.3f})"
                )

                fixed_edge = FixedOp(top_op)
                new_edges.append(fixed_edge)
            cell.edges = new_edges

    device = next(new_model.parameters()).device

    if hasattr(new_model, "forecast_encoder") and hasattr(
        new_model.forecast_encoder, "alphas"
    ):
        try:
            if hasattr(new_model.forecast_encoder, "get_alphas"):
                encoder_weights = new_model.forecast_encoder.get_alphas()[:3]
            else:
                encoder_weights = F.softmax(new_model.forecast_encoder.alphas, dim=-1)
            top_idx = encoder_weights.argmax().item()
            top_encoder = _pick_variant(
                new_model.forecast_encoder, top_idx, "encoders", "encoder"
            )

            print(
                f"   → Fixing Forecast Encoder: {type(top_encoder).__name__} "
                f"(weight: {encoder_weights[top_idx]:.3f})"
            )

            new_model.forecast_encoder = ArchitectureConverter.create_fixed_encoder(
                new_model.forecast_encoder
            ).to(device)

        except Exception as e:
            print(f"Warning: Could not fix encoder architecture: {e}")
            print("Falling back to weight fixing...")
            ArchitectureConverter.fix_mixed_weights(new_model.forecast_encoder)

    if hasattr(new_model, "forecast_decoder") and hasattr(
        new_model.forecast_decoder, "alphas"
    ):
        try:
            if hasattr(new_model.forecast_decoder, "get_alphas"):
                decoder_weights = new_model.forecast_decoder.get_alphas()[:3]
            else:
                decoder_weights = F.softmax(new_model.forecast_decoder.alphas, dim=-1)
            top_idx = decoder_weights.argmax().item()
            top_decoder = _pick_variant(
                new_model.forecast_decoder, top_idx, "decoders", "decoder"
            )

            print(
                f"   → Fixing Forecast Decoder: {type(top_decoder).__name__} "
                f"(weight: {decoder_weights[top_idx]:.3f})"
            )

            attention_choice = "no_attention"
            max_att_idx = 0

            use_attention = getattr(new_model, "use_attention_bridge", False)

            if use_attention and hasattr(
                new_model.forecast_decoder, "attention_alphas"
            ):
                try:
                    attention_weights = F.softmax(
                        new_model.forecast_decoder.attention_alphas, dim=0
                    )
                    max_idx = attention_weights.argmax().item()
                    attention_choice = (
                        "no_attention"
                        if max_idx == len(attention_weights) - 1
                        else f"attention_layer_{max_idx}"
                    )
                    max_att_idx = max(0, max_idx)

                    print("   → Using Attention Bridge:", attention_choice)
                except Exception as e:
                    print(f"Warning: Could not determine attention choice: {e}")
                    attention_choice = "attention" if use_attention else "no_attention"

            attention_bridge_sources = {
                "attention_bridges": lambda m: m.attention_bridges,
                "attention_bridge": lambda m: [m.attention_bridge],
            }
            attention_bridges = None
            for attr_name, getter in attention_bridge_sources.items():
                if hasattr(new_model.forecast_decoder, attr_name):
                    attention_bridges = getter(new_model.forecast_decoder)
                    break

            use_attention_final = use_attention and attention_choice != "no_attention"

            new_model.forecast_decoder = ArchitectureConverter.create_fixed_decoder(
                new_model.forecast_decoder, use_attention_bridge=use_attention_final
            ).to(device)

            if use_attention_final and attention_bridges is not None:
                try:
                    if (
                        isinstance(attention_bridges, (list, nn.ModuleList))
                        and len(attention_bridges) > max_att_idx
                    ):
                        if hasattr(new_model.forecast_decoder, "attention_bridges"):
                            new_model.forecast_decoder.attention_bridges = (
                                nn.ModuleList([attention_bridges[max_att_idx]])
                            )
                        elif hasattr(new_model.forecast_decoder, "attention_bridge"):
                            new_model.forecast_decoder.attention_bridge.load_state_dict(
                                attention_bridges[max_att_idx].state_dict()
                            )
                    else:
                        print(
                            "Warning: Could not assign attention bridges - index out of range or invalid format"
                        )
                except Exception as e:
                    print(f"Warning: Could not assign attention bridges: {e}")
            else:
                if hasattr(new_model.forecast_decoder, "attention_bridges"):
                    new_model.forecast_decoder.attention_bridges = None

        except Exception as e:
            print(f"Warning: Could not fix decoder architecture: {e}")
            print("Falling back to weight fixing...")
            ArchitectureConverter.fix_mixed_weights(new_model.forecast_decoder)

    print("✓ Architecture derivation completed")
    return new_model
