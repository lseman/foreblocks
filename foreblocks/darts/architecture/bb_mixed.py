"""Searchable mixed blocks and fixed deployment wrappers."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_attention import LearnedPoolingBridge
from .bb_sequence import ArchitectureNormalizer
from .bb_sequence import BaseFixedSequenceBlock
from .bb_sequence import SearchableDecomposition
from .bb_sequence import SequenceStateAdapter
from .bb_transformers import LightweightTransformerDecoder
from .bb_transformers import LightweightTransformerEncoder


__all__ = [
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
]


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


def _collect_layer_components(module_obj, key: str) -> list[nn.Module]:
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return []
    out: list[nn.Module] = []
    for layer in layers:
        component = _layer_component(layer, key)
        if component is not None:
            out.append(component)
    return out


def _mean_component_mode_probs(
    components: list[nn.Module],
    *,
    direct_attr: str,
    logits_attr: str,
    mode_names,
) -> torch.Tensor | None:
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


def _resolve_searchable_self_attention_type(module_obj, fallback: str = "sdp") -> str:
    if module_obj is None:
        return fallback

    direct = getattr(module_obj, "self_attention_type", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    components = _collect_layer_components(module_obj, "self_attn")
    if components:
        modes = getattr(
            components[0], "MODES", ("sdp", "linear", "probsparse", "cosine", "local")
        )
        probs = _mean_component_mode_probs(
            components,
            direct_attr="attention_type",
            logits_attr="attn_alphas",
            mode_names=modes,
        )
        if probs is not None:
            idx = int(torch.argmax(probs).item())
            if 0 <= idx < len(modes):
                return str(modes[idx])

    return direct if isinstance(direct, str) and direct else fallback


def _freeze_transformer_self_attention(module_obj, attention_type: str) -> None:
    if module_obj is None:
        return

    resolved = str(attention_type).lower()
    if hasattr(module_obj, "self_attention_type"):
        module_obj.self_attention_type = resolved

    layers = getattr(module_obj, "layers", None)
    if not layers:
        return

    for layer in layers:
        self_attn = None
        if isinstance(layer, dict):
            self_attn = layer.get("self_attn")
        elif hasattr(layer, "get"):
            self_attn = layer.get("self_attn")
        elif hasattr(layer, "__contains__") and "self_attn" in layer:
            self_attn = layer["self_attn"]
        if self_attn is None:
            continue
        self_attn.attention_type = resolved
        self_attn.searchable = False
        if hasattr(self_attn, "attn_alphas"):
            self_attn._parameters.pop("attn_alphas", None)
            try:
                delattr(self_attn, "attn_alphas")
            except AttributeError:
                pass


def _resolve_searchable_self_attention_position(
    module_obj, fallback: str = "rope"
) -> str:
    if module_obj is None:
        return fallback
    components = _collect_layer_components(module_obj, "self_attn")
    if not components:
        return fallback
    modes = getattr(
        components[0], "POSITION_MODES", ("rope", "alibi", "none", "seasonal")
    )
    probs = _mean_component_mode_probs(
        components,
        direct_attr="position_mode",
        logits_attr="position_alphas",
        mode_names=modes,
    )
    if probs is not None:
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(modes):
            return str(modes[idx])
    return fallback


def _freeze_transformer_self_attention_position(module_obj, position_mode: str) -> None:
    if module_obj is None:
        return
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return
    for layer in layers:
        self_attn = None
        if isinstance(layer, dict):
            self_attn = layer.get("self_attn")
        elif hasattr(layer, "get"):
            self_attn = layer.get("self_attn")
        elif hasattr(layer, "__contains__") and "self_attn" in layer:
            self_attn = layer["self_attn"]
        if self_attn is None:
            continue
        freezer = getattr(self_attn, "freeze_position_mode", None)
        if callable(freezer):
            freezer(position_mode)
        else:
            self_attn.position_mode = str(position_mode).lower()


def _resolve_searchable_cross_attention_type(module_obj, fallback: str = "sdp") -> str:
    if module_obj is None:
        return fallback

    direct = getattr(module_obj, "cross_attention_type", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    components = _collect_layer_components(module_obj, "cross_attn")
    if components:
        modes = getattr(
            components[0],
            "MODES",
            ("none", "sdp", "linear", "probsparse", "cosine", "local"),
        )
        probs = _mean_component_mode_probs(
            components,
            direct_attr="attention_type",
            logits_attr="attn_alphas",
            mode_names=modes,
        )
        if probs is not None:
            idx = int(torch.argmax(probs).item())
            if 0 <= idx < len(modes):
                return str(modes[idx])

    return direct if isinstance(direct, str) and direct else fallback


def _freeze_transformer_cross_attention(module_obj, attention_type: str) -> None:
    if module_obj is None:
        return

    resolved = str(attention_type).lower()
    if hasattr(module_obj, "cross_attention_type"):
        module_obj.cross_attention_type = resolved

    layers = getattr(module_obj, "layers", None)
    if not layers:
        return

    for layer in layers:
        cross_attn = None
        if isinstance(layer, dict):
            cross_attn = layer.get("cross_attn")
        elif hasattr(layer, "get"):
            cross_attn = layer.get("cross_attn")
        elif hasattr(layer, "__contains__") and "cross_attn" in layer:
            cross_attn = layer["cross_attn"]
        if cross_attn is None:
            continue
        cross_attn.attention_type = resolved
        cross_attn.searchable = False
        if hasattr(cross_attn, "attn_alphas"):
            cross_attn._parameters.pop("attn_alphas", None)
            try:
                delattr(cross_attn, "attn_alphas")
            except AttributeError:
                pass


def _resolve_searchable_cross_attention_position(
    module_obj, fallback: str = "rope"
) -> str:
    if module_obj is None:
        return fallback
    components = _collect_layer_components(module_obj, "cross_attn")
    if not components:
        return fallback
    modes = getattr(
        components[0], "POSITION_MODES", ("rope", "alibi", "none", "seasonal")
    )
    probs = _mean_component_mode_probs(
        components,
        direct_attr="position_mode",
        logits_attr="position_alphas",
        mode_names=modes,
    )
    if probs is not None:
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(modes):
            return str(modes[idx])
    return fallback


def _freeze_transformer_cross_attention_position(
    module_obj, position_mode: str
) -> None:
    if module_obj is None:
        return
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return
    for layer in layers:
        cross_attn = None
        if isinstance(layer, dict):
            cross_attn = layer.get("cross_attn")
        elif hasattr(layer, "get"):
            cross_attn = layer.get("cross_attn")
        elif hasattr(layer, "__contains__") and "cross_attn" in layer:
            cross_attn = layer["cross_attn"]
        if cross_attn is None:
            continue
        freezer = getattr(cross_attn, "freeze_position_mode", None)
        if callable(freezer):
            freezer(position_mode)
        else:
            cross_attn.position_mode = str(position_mode).lower()


def _resolve_searchable_ffn_mode(module_obj, fallback: str = "swiglu") -> str:
    if module_obj is None:
        return fallback
    components = _collect_layer_components(module_obj, "ffn")
    if not components:
        return fallback
    modes = getattr(components[0], "MODE_NAMES", ("swiglu", "moe"))
    probs = _mean_component_mode_probs(
        components,
        direct_attr="ffn_mode",
        logits_attr="ffn_alphas",
        mode_names=modes,
    )
    if probs is not None:
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(modes):
            return str(modes[idx])
    return fallback


def _freeze_transformer_ffn_mode(module_obj, ffn_mode: str) -> None:
    if module_obj is None:
        return
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return
    resolved = "moe" if str(ffn_mode).lower() == "moe" else "swiglu"
    if hasattr(module_obj, "ffn_variant"):
        module_obj.ffn_variant = resolved
    if hasattr(module_obj, "use_moe"):
        module_obj.use_moe = resolved == "moe"
    for layer in layers:
        ffn = None
        if isinstance(layer, dict):
            ffn = layer.get("ffn")
        elif hasattr(layer, "get"):
            ffn = layer.get("ffn")
        elif hasattr(layer, "__contains__") and "ffn" in layer:
            ffn = layer["ffn"]
        if ffn is None:
            continue
        freezer = getattr(ffn, "freeze_ffn_mode", None)
        if callable(freezer):
            freezer(resolved)
        else:
            ffn.ffn_mode = resolved


def _resolve_searchable_patch_mode(module_obj, fallback: str = "direct") -> str:
    if module_obj is None:
        return fallback

    direct = getattr(module_obj, "patching_mode", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    logits = getattr(module_obj, "patch_alpha_logits", None)
    mode_names = getattr(module_obj, "patch_mode_names", ("direct", "patch"))
    if isinstance(logits, torch.Tensor) and logits.numel() == len(mode_names):
        probs = F.softmax(logits.detach(), dim=0)
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(mode_names):
            return str(mode_names[idx])

    resolver = getattr(module_obj, "resolve_patch_mode", None)
    if callable(resolver):
        try:
            return str(resolver())
        except Exception:
            pass

    return direct if isinstance(direct, str) and direct else fallback


def _freeze_transformer_patch_mode(module_obj, patch_mode: str) -> None:
    if module_obj is None:
        return

    resolved = str(patch_mode).lower()
    if resolved == "patch":
        resolved = "patch_16"
    valid_modes = tuple(
        getattr(
            module_obj,
            "patch_mode_names",
            (
                "direct",
                "patch_8",
                "patch_16",
                "patch_32",
                "multi_scale_patch",
                "variate_tokens",
            ),
        )
    )
    if resolved not in valid_modes:
        resolved = "direct"
    if hasattr(module_obj, "freeze_patch_mode"):
        try:
            module_obj.freeze_patch_mode(resolved)
            return
        except Exception:
            pass

    if hasattr(module_obj, "patching_mode"):
        module_obj.patching_mode = resolved
    if hasattr(module_obj, "patch_alpha_logits"):
        module_obj._parameters.pop("patch_alpha_logits", None)
        try:
            delattr(module_obj, "patch_alpha_logits")
        except AttributeError:
            pass


def _resolve_searchable_decoder_style(
    module_obj, fallback: str = "autoregressive"
) -> str:
    if module_obj is None:
        return fallback

    direct = getattr(module_obj, "decode_style", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    logits = getattr(module_obj, "decode_style_alphas", None)
    style_names = getattr(
        module_obj, "decode_style_names", ("autoregressive", "informer")
    )
    if isinstance(logits, torch.Tensor) and logits.numel() == len(style_names):
        probs = F.softmax(logits.detach(), dim=0)
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(style_names):
            return str(style_names[idx])

    resolver = getattr(module_obj, "resolve_decode_style", None)
    if callable(resolver):
        try:
            return str(resolver())
        except Exception:
            pass

    return direct if isinstance(direct, str) and direct else fallback


def _freeze_transformer_decoder_style(module_obj, decode_style: str) -> None:
    if module_obj is None:
        return

    resolved = (
        "informer" if str(decode_style).lower() == "informer" else "autoregressive"
    )
    if hasattr(module_obj, "freeze_decode_style"):
        try:
            module_obj.freeze_decode_style(resolved)
            return
        except Exception:
            pass

    if hasattr(module_obj, "decode_style"):
        module_obj.decode_style = resolved
    if hasattr(module_obj, "decode_style_alphas"):
        module_obj._parameters.pop("decode_style_alphas", None)
        try:
            delattr(module_obj, "decode_style_alphas")
        except AttributeError:
            pass


class MixedEncoder(nn.Module):
    """Transformer-family encoder search block.

    Sequence architecture search is intentionally limited to one transformer
    branch. Patching is searched inside the encoder itself via patch-mode
    alphas rather than as a separate top-level architecture option.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        single_path_search: bool = True,
        arch_path_keep_prob: float = 0.85,
        include_patch: bool = True,
        transformer_self_attention_type: str = "auto",
        transformer_use_moe: bool = False,
        transformer_ffn_variant: str = "auto",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.temperature = max(float(temperature), 1e-3)
        self.num_layers = 2
        self.single_path_search = bool(single_path_search)
        self.arch_path_keep_prob = float(min(max(arch_path_keep_prob, 0.0), 1.0))
        self.include_patch = bool(include_patch)
        self.rnn_type = "transformer"
        self.searchable_decomp = SearchableDecomposition(c_in=input_dim)

        resolved_self_attention_type = str(transformer_self_attention_type).lower()
        self.transformer_use_moe = bool(transformer_use_moe)
        self.transformer_ffn_variant = str(transformer_ffn_variant).lower()

        self.transformer = LightweightTransformerEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            num_layers=2,
            dropout=dropout,
            self_attention_type=resolved_self_attention_type,
            self_attention_position_mode="auto",
            use_moe=self.transformer_use_moe,
            ffn_variant=self.transformer_ffn_variant,
            use_checkpoint=use_checkpoint,
            temperature=self.temperature,
            single_path_search=self.single_path_search,
            enable_patch_search=self.include_patch,
            patching_mode="auto" if self.include_patch else "direct",
        )
        self.encoders = nn.ModuleList([self.transformer])
        self.encoder_names = ["transformer"]
        self.encoder_name_to_idx = {
            name: idx for idx, name in enumerate(self.encoder_names)
        }

        self.last_selected_output_idx: int | None = None
        self.last_selected_layer_idx: torch.Tensor | None = None
        self.normalizer = ArchitectureNormalizer(latent_dim)
        self.context_proj = nn.Linear(latent_dim, latent_dim)
        self.rnn = self.transformer

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.searchable_decomp(x, temperature=self.temperature)
        single_path_active = self.training and self.single_path_search
        self.last_selected_output_idx = 0
        self.last_selected_layer_idx = torch.zeros(
            self.num_layers, device=x.device, dtype=torch.long
        )
        trans_out, trans_ctx, trans_state = self.transformer(
            x,
            temperature=self.temperature,
            single_path=single_path_active,
        )
        output = self.normalizer.normalize_output(trans_out, "transformer")
        context = trans_ctx
        state = self.normalizer.normalize_state(trans_state, "transformer")

        context = self.context_proj(context)

        h_blended = state[0]
        c_blended = state[1]

        return output, context, (h_blended, c_blended)

    def set_temperature(self, temp: float):
        self.temperature = max(float(temp), 1e-3)
        if hasattr(self.transformer, "set_temperature"):
            self.transformer.set_temperature(self.temperature)

    def orthogonal_regularization(self) -> torch.Tensor:
        ref = next(self.parameters(), None)
        if ref is None:
            return torch.tensor(0.0)
        return ref.new_zeros(())


class MixedDecoder(nn.Module):
    """Transformer-only decoder search block."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_attention_bridge: bool = True,
        attention_layers: int = 1,
        use_learned_memory_pooling: bool = True,
        memory_num_queries: int = 8,
        single_path_search: bool = True,
        arch_path_keep_prob: float = 0.85,
        attention_temperature_mult: float = 0.7,
        min_attention_temperature: float = 0.25,
        memory_query_options: list[int] | None = None,
        transformer_self_attention_type: str = "auto",
        transformer_cross_attention_type: str = "auto",
        transformer_use_moe: bool = False,
        transformer_ffn_variant: str = "auto",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.temperature = max(float(temperature), 1e-3)
        self.num_layers = 2
        self.single_path_search = bool(single_path_search)
        self.arch_path_keep_prob = float(min(max(arch_path_keep_prob, 0.0), 1.0))
        self.rnn_type = "transformer"
        self.decode_style_names = ("autoregressive", "informer")
        self.decode_style = "auto"
        self.register_parameter(
            "decode_style_alphas", nn.Parameter(0.01 * torch.randn(2))
        )

        self.use_attention_bridge = False
        self.attention_layers = attention_layers
        self.use_learned_memory_pooling = use_learned_memory_pooling
        self.attention_temperature_mult = float(max(attention_temperature_mult, 1e-3))
        self.min_attention_temperature = float(max(min_attention_temperature, 1e-3))
        self.searchable_decomp = SearchableDecomposition(c_in=input_dim)
        self.default_memory_num_queries = int(max(1, memory_num_queries))

        resolved_self_attention_type = str(transformer_self_attention_type).lower()
        resolved_cross_attention_type = str(transformer_cross_attention_type).lower()
        self.transformer_use_moe = bool(transformer_use_moe)
        self.transformer_ffn_variant = str(transformer_ffn_variant).lower()

        self.transformer = LightweightTransformerDecoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=2,
            dropout=dropout,
            self_attention_type=resolved_self_attention_type,
            self_attention_position_mode="auto",
            cross_attention_type=resolved_cross_attention_type,
            cross_attention_position_mode="auto",
            use_moe=self.transformer_use_moe,
            ffn_variant=self.transformer_ffn_variant,
            use_checkpoint=use_checkpoint,
            temperature=self.temperature,
            single_path_search=self.single_path_search,
        )

        self.decoders = nn.ModuleList([self.transformer])
        self.decoder_names = ["transformer"]
        self.rnn_names = ["transformer"]
        self.decoder_name_to_idx = {
            name: idx for idx, name in enumerate(self.decoder_names)
        }
        self.rnn = self.transformer

        self.normalizer = ArchitectureNormalizer(latent_dim)
        self.memory_query_options = (
            list(memory_query_options)
            if memory_query_options is not None
            else [4, 8, 16]
        )
        self.memory_query_options = [
            int(max(1, q)) for q in self.memory_query_options if int(max(1, q)) > 0
        ]
        if not self.memory_query_options:
            self.memory_query_options = [4, 8, 16]
        if self.default_memory_num_queries not in self.memory_query_options:
            self.memory_query_options = sorted(
                set(self.memory_query_options + [self.default_memory_num_queries])
            )
        self.default_memory_query_idx = self.memory_query_options.index(
            self.default_memory_num_queries
        )

        if use_learned_memory_pooling:
            self.memory_pool_bridges = nn.ModuleList(
                [
                    LearnedPoolingBridge(
                        dim=latent_dim,
                        num_queries=num_queries,
                        num_heads=4,
                        dropout=dropout,
                    )
                    for num_queries in self.memory_query_options
                ]
            )
            memory_query_init = 0.01 * torch.randn(len(self.memory_query_options))
            self.register_parameter(
                "memory_query_alphas", nn.Parameter(memory_query_init)
            )
        else:
            self.memory_pool_bridges = None

    def get_decode_style_weights(self) -> torch.Tensor:
        tau = max(float(self.temperature), 1e-3)
        if self.training:
            if self.single_path_search:
                return F.gumbel_softmax(
                    self.decode_style_alphas, tau=tau, hard=True, dim=0
                )
            return F.gumbel_softmax(
                self.decode_style_alphas, tau=tau, hard=False, dim=0
            )
        probs = F.softmax(self.decode_style_alphas / tau, dim=0)
        if self.single_path_search:
            hard = torch.zeros_like(probs)
            hard[int(torch.argmax(probs).item())] = 1.0
            return hard
        return probs

    def get_decode_style_probs(self) -> torch.Tensor:
        return F.softmax(self.decode_style_alphas.detach(), dim=0)

    def resolve_decode_style(self) -> str:
        probs = self.get_decode_style_probs()
        idx = int(torch.argmax(probs).item())
        return self.decode_style_names[idx]

    def freeze_decode_style(self, decode_style: str) -> None:
        resolved = (
            "informer" if str(decode_style).lower() == "informer" else "autoregressive"
        )
        self.decode_style = resolved
        if hasattr(self, "decode_style_alphas"):
            self._parameters.pop("decode_style_alphas", None)
            try:
                delattr(self, "decode_style_alphas")
            except AttributeError:
                pass

    def _get_decoder_weights(self) -> torch.Tensor:
        ref = next(self.parameters())
        return ref.new_ones(1)

    def _get_memory_query_weights(self) -> torch.Tensor | None:
        if not (
            self.use_learned_memory_pooling
            and hasattr(self, "memory_query_alphas")
            and self.memory_pool_bridges is not None
        ):
            return None

        tau = max(float(self.temperature), 1e-3)
        if self.training:
            if self.single_path_search:
                return F.gumbel_softmax(
                    self.memory_query_alphas, tau=tau, hard=True, dim=0
                )
            return F.gumbel_softmax(
                self.memory_query_alphas, tau=tau, hard=False, dim=0
            )
        probs = F.softmax(self.memory_query_alphas / tau, dim=0)
        if self.single_path_search:
            hard = torch.zeros_like(probs)
            hard[int(torch.argmax(probs).item())] = 1.0
            return hard
        return probs

    def _build_shared_memory(
        self,
        memory: torch.Tensor | None,
        encoder_output: torch.Tensor | None,
        encoder_context: torch.Tensor | None,
    ) -> torch.Tensor | None:
        source = encoder_output
        if source is None:
            source = memory
        if source is None:
            source = encoder_context
        if source is None:
            return None

        if isinstance(source, tuple):
            source = source[0]

        if source.dim() == 2:
            source = source.unsqueeze(1)

        if self.memory_pool_bridges is not None:
            memory_weights = self._get_memory_query_weights()
            if memory_weights is None:
                pooled = self.memory_pool_bridges[self.default_memory_query_idx](source)
                return self._resize_memory_queries(
                    pooled, self.default_memory_num_queries
                )

            if self.training and self.single_path_search:
                chosen = int(torch.argmax(memory_weights.detach()).item())
                pooled = self.memory_pool_bridges[chosen](source)
                pooled = self._resize_memory_queries(
                    pooled, self.default_memory_num_queries
                )
                return memory_weights[chosen] * pooled

            pooled_outputs = [
                self._resize_memory_queries(
                    bridge(source), self.default_memory_num_queries
                )
                for bridge in self.memory_pool_bridges
            ]
            return sum(w * out for w, out in zip(memory_weights, pooled_outputs))

        return source

    @staticmethod
    def _resize_memory_queries(
        memory: torch.Tensor, target_queries: int
    ) -> torch.Tensor:
        if memory.dim() != 3:
            return memory
        q = int(memory.size(1))
        target = int(max(1, target_queries))
        if q == target:
            return memory
        return F.interpolate(
            memory.transpose(1, 2),
            size=target,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: torch.Tensor | None = None,
        encoder_context: torch.Tensor | None = None,
        forced_output_idx: int | None = None,
        forced_layer_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        tgt = self.searchable_decomp(tgt, temperature=self.temperature)
        batch_size = tgt.size(0)

        trans_state = SequenceStateAdapter.ensure_rnn_state(
            hidden_state,
            rnn_type="transformer",
            num_layers=self.num_layers,
            batch_size=batch_size,
            hidden_size=self.latent_dim,
            device=tgt.device,
            dtype=tgt.dtype,
        )

        shared_memory = self._build_shared_memory(
            memory, encoder_output, encoder_context
        )
        transformer_memory = (
            shared_memory
            if shared_memory is not None
            else (
                encoder_context
                if encoder_context is not None
                else torch.zeros_like(tgt[:, :1, :])
            )
        )
        trans_out, trans_new_state = self.transformer(
            tgt, transformer_memory, trans_state
        )
        output = self.normalizer.normalize_output(trans_out, "transformer")
        state = self.normalizer.normalize_state(trans_new_state, "transformer")

        return output, state

    def set_temperature(self, temp: float):
        self.temperature = max(float(temp), 1e-3)
        if hasattr(self.transformer, "set_temperature"):
            self.transformer.set_temperature(self.temperature)

    def orthogonal_regularization(self) -> torch.Tensor:
        ref = next(self.parameters(), None)
        if ref is None:
            return torch.tensor(0.0)
        return ref.new_zeros(())


class ArchitectureConverter:
    """Utility class for converting between mixed and fixed architectures"""

    @staticmethod
    def get_best_architecture(
        alphas: torch.Tensor,
        num_options: int = 4,
        arch_names: list[str] | None = None,
    ) -> str:
        arch_names = list(arch_names or ["transformer"])
        num_options = min(max(1, int(num_options)), len(arch_names))
        best_idx = int(torch.argmax(alphas[:num_options]).item())
        return arch_names[best_idx]

    @staticmethod
    def ensure_proper_state_format(
        state,
        rnn_type: str,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
    ):
        return SequenceStateAdapter.ensure_rnn_state(
            state,
            rnn_type=rnn_type,
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
        )

    @staticmethod
    def fix_mixed_weights(mixed_model, temperature: float = 0.01):
        if hasattr(mixed_model, "set_temperature"):
            mixed_model.set_temperature(temperature)

    @staticmethod
    def create_fixed_encoder(mixed_encoder, **kwargs) -> FixedEncoder:
        kwargs.pop("forced_arch_type", None)
        best_type = "transformer"
        self_attention_type = None
        self_attention_position_mode = None
        ffn_mode = None
        if best_type == "transformer":
            self_attention_type = _resolve_searchable_self_attention_type(
                mixed_encoder.transformer
            )
            self_attention_position_mode = _resolve_searchable_self_attention_position(
                mixed_encoder.transformer
            )
            ffn_mode = _resolve_searchable_ffn_mode(mixed_encoder.transformer)
        patch_mode = _resolve_searchable_patch_mode(mixed_encoder.transformer)
        fixed_encoder = FixedEncoder(
            rnn=copy.deepcopy(mixed_encoder.transformer),
            input_dim=mixed_encoder.input_dim,
            latent_dim=mixed_encoder.latent_dim,
            self_attention_type=self_attention_type,
            self_attention_position_mode=self_attention_position_mode,
            patching_mode=patch_mode,
            **kwargs,
        )
        ArchitectureConverter._transfer_encoder_weights(
            mixed_encoder,
            fixed_encoder,
            best_type,
            self_attention_type=self_attention_type,
            self_attention_position_mode=self_attention_position_mode,
            ffn_mode=ffn_mode,
            patch_mode=patch_mode,
        )
        return fixed_encoder

    @staticmethod
    def create_fixed_decoder(mixed_decoder, **kwargs) -> FixedDecoder:
        kwargs.pop("forced_arch_type", None)
        kwargs.pop("use_attention_bridge", None)
        legacy_attention_variant = kwargs.pop("attention_variant", None)
        explicit_cross_attention_type = kwargs.pop("cross_attention_type", None)
        best_type = "transformer"
        self_attention_type = None
        self_attention_position_mode = None
        decode_style = _resolve_searchable_decoder_style(mixed_decoder)
        cross_attention_type = _resolve_searchable_cross_attention_type(
            mixed_decoder.transformer
        )
        cross_attention_position_mode = _resolve_searchable_cross_attention_position(
            mixed_decoder.transformer
        )
        ffn_mode = _resolve_searchable_ffn_mode(mixed_decoder.transformer)
        if explicit_cross_attention_type is not None:
            cross_attention_type = str(explicit_cross_attention_type).lower()
        elif legacy_attention_variant is not None:
            cross_attention_type = str(legacy_attention_variant).lower()
        if best_type == "transformer":
            self_attention_type = _resolve_searchable_self_attention_type(
                mixed_decoder.transformer
            )
            self_attention_position_mode = _resolve_searchable_self_attention_position(
                mixed_decoder.transformer
            )

        fixed_decoder = FixedDecoder(
            rnn_type=best_type,
            input_dim=mixed_decoder.input_dim,
            latent_dim=mixed_decoder.latent_dim,
            self_attention_type=self_attention_type,
            self_attention_position_mode=self_attention_position_mode,
            cross_attention_type=cross_attention_type,
            cross_attention_position_mode=cross_attention_position_mode,
            ffn_mode=ffn_mode,
            decode_style=decode_style,
            **kwargs,
        )
        ArchitectureConverter._transfer_decoder_weights(
            mixed_decoder,
            fixed_decoder,
            best_type,
            self_attention_type=self_attention_type,
            self_attention_position_mode=self_attention_position_mode,
            cross_attention_type=cross_attention_type,
            cross_attention_position_mode=cross_attention_position_mode,
            ffn_mode=ffn_mode,
            decode_style=decode_style,
        )
        return fixed_decoder

    @staticmethod
    def _transfer_encoder_weights(
        mixed_encoder,
        fixed_encoder,
        arch_type: str,
        self_attention_type: str | None = None,
        self_attention_position_mode: str | None = None,
        ffn_mode: str | None = None,
        patch_mode: str | None = None,
    ):
        try:
            if arch_type == "transformer":
                source_rnn = mixed_encoder.transformer
            else:
                raise ValueError(f"Unknown architecture type: {arch_type}")

            fixed_encoder.rnn.load_state_dict(source_rnn.state_dict(), strict=False)
            if (
                self_attention_type is not None
                and arch_type == "transformer"
                and hasattr(fixed_encoder, "rnn")
            ):
                _freeze_transformer_self_attention(
                    fixed_encoder.rnn, self_attention_type
                )
            if (
                self_attention_position_mode is not None
                and arch_type == "transformer"
                and hasattr(fixed_encoder, "rnn")
            ):
                _freeze_transformer_self_attention_position(
                    fixed_encoder.rnn, self_attention_position_mode
                )
            if patch_mode is not None and arch_type == "transformer":
                _freeze_transformer_patch_mode(fixed_encoder.rnn, patch_mode)
            if ffn_mode is not None and arch_type == "transformer":
                _freeze_transformer_ffn_mode(fixed_encoder.rnn, ffn_mode)

            if hasattr(mixed_encoder, "normalizer"):
                fixed_encoder.normalizer = ArchitectureNormalizer(
                    mixed_encoder.latent_dim
                ).to(next(fixed_encoder.parameters()).device)
                fixed_encoder.normalizer.load_state_dict(
                    mixed_encoder.normalizer.state_dict()
                )
            if hasattr(mixed_encoder, "context_proj"):
                fixed_encoder.context_proj = nn.Linear(
                    mixed_encoder.latent_dim,
                    mixed_encoder.latent_dim,
                ).to(next(fixed_encoder.parameters()).device)
                fixed_encoder.context_proj.load_state_dict(
                    mixed_encoder.context_proj.state_dict()
                )
            if hasattr(mixed_encoder, "searchable_decomp"):
                fixed_encoder.searchable_decomp = copy.deepcopy(
                    mixed_encoder.searchable_decomp
                ).to(next(fixed_encoder.parameters()).device)
                if hasattr(fixed_encoder.searchable_decomp, "alpha_logits"):
                    with torch.no_grad():
                        logits = fixed_encoder.searchable_decomp.alpha_logits
                        hard = torch.full_like(logits, -10.0)
                        hard[int(torch.argmax(logits).item())] = 10.0
                        logits.copy_(hard)
                    fixed_encoder.searchable_decomp.alpha_logits.requires_grad_(False)
        except Exception as e:
            print(f"Warning: Could not transfer encoder weights: {e}")

    @staticmethod
    def _transfer_decoder_weights(
        mixed_decoder,
        fixed_decoder,
        arch_type: str,
        self_attention_type: str | None = None,
        self_attention_position_mode: str | None = None,
        cross_attention_type: str | None = None,
        cross_attention_position_mode: str | None = None,
        ffn_mode: str | None = None,
        decode_style: str | None = None,
    ):
        try:
            if arch_type == "transformer":
                source_rnn = mixed_decoder.transformer
            else:
                raise ValueError(f"Unknown architecture type: {arch_type}")

            fixed_decoder.rnn.load_state_dict(source_rnn.state_dict(), strict=False)
            if (
                self_attention_type is not None
                and arch_type == "transformer"
                and hasattr(fixed_decoder, "rnn")
            ):
                _freeze_transformer_self_attention(
                    fixed_decoder.rnn, self_attention_type
                )
            if (
                self_attention_position_mode is not None
                and arch_type == "transformer"
                and hasattr(fixed_decoder, "rnn")
            ):
                _freeze_transformer_self_attention_position(
                    fixed_decoder.rnn, self_attention_position_mode
                )
            if (
                cross_attention_type is not None
                and arch_type == "transformer"
                and hasattr(fixed_decoder, "rnn")
            ):
                _freeze_transformer_cross_attention(
                    fixed_decoder.rnn, cross_attention_type
                )
            if (
                cross_attention_position_mode is not None
                and arch_type == "transformer"
                and hasattr(fixed_decoder, "rnn")
            ):
                _freeze_transformer_cross_attention_position(
                    fixed_decoder.rnn, cross_attention_position_mode
                )
            if decode_style is not None and hasattr(fixed_decoder, "decode_style"):
                fixed_decoder.decode_style = str(decode_style).lower()
            if decode_style is not None and hasattr(fixed_decoder, "rnn"):
                _freeze_transformer_decoder_style(fixed_decoder, decode_style)
            if ffn_mode is not None and arch_type == "transformer":
                _freeze_transformer_ffn_mode(fixed_decoder.rnn, ffn_mode)

            if hasattr(mixed_decoder, "normalizer"):
                fixed_decoder.normalizer = ArchitectureNormalizer(
                    mixed_decoder.latent_dim
                ).to(next(fixed_decoder.parameters()).device)
                fixed_decoder.normalizer.load_state_dict(
                    mixed_decoder.normalizer.state_dict()
                )
            if hasattr(mixed_decoder, "searchable_decomp"):
                fixed_decoder.searchable_decomp = copy.deepcopy(
                    mixed_decoder.searchable_decomp
                ).to(next(fixed_decoder.parameters()).device)
                if hasattr(fixed_decoder.searchable_decomp, "alpha_logits"):
                    with torch.no_grad():
                        logits = fixed_decoder.searchable_decomp.alpha_logits
                        hard = torch.full_like(logits, -10.0)
                        hard[int(torch.argmax(logits).item())] = 10.0
                        logits.copy_(hard)
                    fixed_decoder.searchable_decomp.alpha_logits.requires_grad_(False)

        except Exception as e:
            print(f"Warning: Could not transfer decoder weights: {e}")


class FixedEncoder(BaseFixedSequenceBlock):
    """Simple fixed encoder wrapper for deployment"""

    def __init__(
        self,
        rnn=None,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        self_attention_type: str | None = None,
        self_attention_position_mode: str | None = None,
        ffn_mode: str | None = None,
        patching_mode: str | None = None,
    ):
        self.self_attention_type = (
            str(self_attention_type).lower()
            if self_attention_type is not None
            else None
        )
        self.self_attention_position_mode = (
            str(self_attention_position_mode).lower()
            if self_attention_position_mode is not None
            else "rope"
        )
        self.patching_mode = (
            str(patching_mode).lower() if patching_mode is not None else None
        )
        self.ffn_mode = str(ffn_mode).lower() if ffn_mode is not None else "swiglu"
        super().__init__(
            rnn=rnn,
            rnn_type=rnn_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            transformer_factory=(
                lambda input_dim, latent_dim, num_layers, dropout: (
                    LightweightTransformerEncoder(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        self_attention_type=self.self_attention_type or "sdp",
                        self_attention_position_mode=self.self_attention_position_mode,
                        ffn_variant=self.ffn_mode,
                        patching_mode=self.patching_mode or "direct",
                        enable_patch_search=False,
                    )
                )
            ),
        )
        if self.self_attention_type is not None and hasattr(self, "rnn"):
            _freeze_transformer_self_attention(self.rnn, self.self_attention_type)
        if hasattr(self, "rnn"):
            _freeze_transformer_self_attention_position(
                self.rnn, self.self_attention_position_mode
            )
        if self.patching_mode is not None and hasattr(self, "rnn"):
            _freeze_transformer_patch_mode(self.rnn, self.patching_mode)
        if hasattr(self, "rnn"):
            _freeze_transformer_ffn_mode(self.rnn, self.ffn_mode)
        self.normalizer = None
        self.context_proj = None
        self.searchable_decomp = None

    def forward(self, x: torch.Tensor) -> tuple:
        if self.searchable_decomp is not None:
            x = self.searchable_decomp(x, temperature=0.01)

        if self.rnn_type == "transformer":
            output, context, state = self.rnn(x)
        else:
            raw_output, state = self.rnn(x)
            context = raw_output[:, -1:, :]
            output = raw_output

            if isinstance(self.rnn, nn.GRU):
                h = state
                c = torch.zeros_like(h)
                state = (h, c)

        if self.normalizer is not None:
            output = self.normalizer.normalize_output(output, self.rnn_type)
            state = self.normalizer.normalize_state(state, self.rnn_type)
        if self.context_proj is not None:
            context = self.context_proj(context)

        return output, context, state


class FixedDecoder(BaseFixedSequenceBlock):
    """Simple fixed decoder wrapper for deployment"""

    def __init__(
        self,
        rnn=None,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_attention_bridge: bool = False,
        attention_layers: int = 1,
        attention_variant: str = "sdp",
        self_attention_type: str | None = None,
        self_attention_position_mode: str | None = None,
        cross_attention_type: str | None = None,
        cross_attention_position_mode: str | None = None,
        ffn_mode: str | None = None,
        decode_style: str | None = None,
    ):
        self.self_attention_type = (
            str(self_attention_type).lower()
            if self_attention_type is not None
            else None
        )
        self.self_attention_position_mode = (
            str(self_attention_position_mode).lower()
            if self_attention_position_mode is not None
            else "rope"
        )
        self.cross_attention_type = (
            str(cross_attention_type).lower()
            if cross_attention_type is not None
            else str(attention_variant).lower()
        )
        self.cross_attention_position_mode = (
            str(cross_attention_position_mode).lower()
            if cross_attention_position_mode is not None
            else "rope"
        )
        self.decode_style = (
            str(decode_style).lower() if decode_style is not None else "autoregressive"
        )
        self.ffn_mode = str(ffn_mode).lower() if ffn_mode is not None else "swiglu"
        self.use_attention_bridge = False

        super().__init__(
            rnn=rnn,
            rnn_type=rnn_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            transformer_factory=(
                lambda input_dim, latent_dim, num_layers, dropout: (
                    LightweightTransformerDecoder(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        self_attention_type=self.self_attention_type or "sdp",
                        self_attention_position_mode=self.self_attention_position_mode,
                        cross_attention_type=self.cross_attention_type,
                        cross_attention_position_mode=self.cross_attention_position_mode,
                        ffn_variant=self.ffn_mode,
                    )
                )
            ),
        )
        if self.self_attention_type is not None and hasattr(self, "rnn"):
            _freeze_transformer_self_attention(self.rnn, self.self_attention_type)
        if hasattr(self, "rnn"):
            _freeze_transformer_self_attention_position(
                self.rnn, self.self_attention_position_mode
            )
        if hasattr(self, "rnn"):
            _freeze_transformer_cross_attention(self.rnn, self.cross_attention_type)
            _freeze_transformer_cross_attention_position(
                self.rnn, self.cross_attention_position_mode
            )
            _freeze_transformer_ffn_mode(self.rnn, self.ffn_mode)
        self.normalizer = None
        self.searchable_decomp = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: torch.Tensor = None,
    ) -> tuple:
        if self.searchable_decomp is not None:
            tgt = self.searchable_decomp(tgt, temperature=0.01)

        batch_size = tgt.size(0)
        num_layers = getattr(self.rnn, "num_layers", 1)
        hidden_size = getattr(self.rnn, "hidden_size", self.latent_dim)

        hidden_state = ArchitectureConverter.ensure_proper_state_format(
            hidden_state, self.rnn_type, num_layers, batch_size, hidden_size, tgt.device
        )

        if self.rnn_type == "transformer":
            raw_output, new_state = self.rnn(tgt, memory, hidden_state)
        else:
            raw_output, new_state = self.rnn(tgt, hidden_state)

        output = raw_output
        if self.normalizer is not None:
            output = self.normalizer.normalize_output(output, self.rnn_type)
            normalized_state = self.normalizer.normalize_state(new_state, self.rnn_type)
            if normalized_state is not None:
                new_state = normalized_state

        return output, new_state
