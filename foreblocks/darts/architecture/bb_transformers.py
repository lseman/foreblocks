"""Transformer encoder and decoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .bb_attention import AttentionBridge
from .bb_attention import SelfAttention
from .bb_moe import DARTSFeedForward
from .bb_primitives import RMSNorm


__all__ = [
    "LightweightTransformerEncoder",
    "LightweightTransformerDecoder",
]


def _layer_get(layer, key: str):
    """Safely fetch layer components from dict-like or Module containers."""
    if isinstance(layer, dict):
        return layer.get(key)
    try:
        return layer[key]
    except Exception:
        pass
    if hasattr(layer, key):
        return getattr(layer, key)
    params = getattr(layer, "_parameters", None)
    if isinstance(params, dict) and key in params:
        return params[key]
    modules = getattr(layer, "_modules", None)
    if isinstance(modules, dict) and key in modules:
        return modules[key]
    buffers = getattr(layer, "_buffers", None)
    if isinstance(buffers, dict) and key in buffers:
        return buffers[key]
    return None


class LightweightTransformerEncoder(nn.Module):
    """Improved transformer encoder with better RNN compatibility"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_layers=2,
        dropout=0.1,
        nhead=4,
        max_seq_len=512,
        seq_len: int | None = None,
        causal=False,
        self_attention_type: str = "auto",
        self_attention_position_mode: str = "auto",
        use_moe: bool = False,
        ffn_variant: str | None = None,
        rope_base: float = 500000.0,
        use_checkpoint: bool = False,
        temperature: float = 1.0,
        single_path_search: bool = True,
        enable_patch_search: bool = False,
        patching_mode: str = "direct",
        patch_size: int = 16,
        stride: int | None = None,
    ):
        super().__init__()
        resolved_self_attention_type = str(self_attention_type).lower()
        resolved_self_position_mode = str(self_attention_position_mode).lower()
        resolved_patching_mode = str(patching_mode).lower()
        if resolved_patching_mode == "patch":
            resolved_patching_mode = "patch_16"
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.causal = causal
        self.self_attention_type = resolved_self_attention_type
        resolved_ffn_variant = (
            str(ffn_variant).lower() if ffn_variant is not None else ("moe" if use_moe else "swiglu")
        )
        self.ffn_variant = resolved_ffn_variant
        self.use_moe = resolved_ffn_variant == "moe"
        self.use_checkpoint = use_checkpoint
        self.temperature = max(float(temperature), 1e-3)
        self.single_path_search = bool(single_path_search)
        self.enable_patch_search = bool(enable_patch_search)
        self.patching_mode = resolved_patching_mode
        self.patch_mode_names = (
            "direct",
            "patch_8",
            "patch_16",
            "patch_32",
            "multi_scale_patch",
            "hierarchical",
            "variate_tokens",
        )
        if self.patching_mode not in (*self.patch_mode_names, "auto"):
            self.patching_mode = "direct"
        self.patch_size = max(2, int(patch_size))
        self.default_stride = max(
            1, int(stride if stride is not None else self.patch_size // 2)
        )
        self.patch_sizes = (8, 16, 32)
        self.target_seq_len = int(max(seq_len or max_seq_len, max(self.patch_sizes)))
        self.num_patches_by_size = {
            size: max(1, (self.target_seq_len - size) // max(1, size // 2) + 1)
            for size in self.patch_sizes
        }

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.patch_projs = nn.ModuleDict(
            {str(size): nn.Linear(size, latent_dim, bias=False) for size in self.patch_sizes}
        )
        self.patch_pos_embeds = nn.ParameterDict(
            {
                str(size): nn.Parameter(
                    torch.zeros(1, self.num_patches_by_size[size], latent_dim)
                )
                for size in self.patch_sizes
            }
        )
        for pos in self.patch_pos_embeds.values():
            nn.init.trunc_normal_(pos, std=0.02)
        self.variate_proj = nn.Linear(self.target_seq_len, latent_dim, bias=False)
        self.variate_pos_embed = nn.Parameter(torch.zeros(1, input_dim, latent_dim))
        nn.init.trunc_normal_(self.variate_pos_embed, std=0.02)
        if self.enable_patch_search and self.patching_mode == "auto":
            self.register_parameter(
                "patch_alpha_logits",
                nn.Parameter(0.01 * torch.randn(len(self.patch_mode_names))),
            )

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": SelfAttention(
                            latent_dim,
                            heads=nhead,
                            dropout=dropout,
                            causal=causal,
                            attention_type=resolved_self_attention_type,
                            position_mode=resolved_self_position_mode,
                            rope_base=rope_base,
                            rope_max_seq_len=max_seq_len,
                        ),
                        "ffn": DARTSFeedForward(
                            d_model=latent_dim,
                            expand=4,
                            dropout=dropout,
                            use_moe=self.use_moe,
                            ffn_mode=resolved_ffn_variant,
                            temperature=self.temperature,
                            single_path_search=self.single_path_search,
                        ),
                        "norm1": RMSNorm(latent_dim),
                        "norm2": RMSNorm(latent_dim),
                        "patch_mix": nn.Parameter(torch.randn(len(self.patch_sizes))) if self.patching_mode == "hierarchical" else None,
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(latent_dim)
        self.dropout_p = dropout
        self.state_proj = nn.Linear(latent_dim, latent_dim * 2, bias=False)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-3)
        for layer in self.layers:
            ffn = _layer_get(layer, "ffn")
            if ffn is not None and hasattr(ffn, "set_temperature"):
                ffn.set_temperature(self.temperature)

    def _get_patch_mode_weights(
        self, temperature: float = 1.0, single_path: bool = False
    ) -> torch.Tensor:
        if not self.enable_patch_search or self.patching_mode != "auto":
            mode = self.patching_mode if self.patching_mode in self.patch_mode_names else "direct"
            ref = next(self.parameters())
            weights = ref.new_zeros(len(self.patch_mode_names))
            weights[self.patch_mode_names.index(mode)] = 1.0
            return weights

        tau = max(float(temperature), 1e-3)
        logits = self.patch_alpha_logits
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=bool(single_path), dim=0)
        probs = F.softmax(logits / tau, dim=0)
        if self.single_path_search:
            hard = torch.zeros_like(probs)
            hard[int(torch.argmax(probs).item())] = 1.0
            return hard
        return probs

    def get_patch_mode_probs(self) -> torch.Tensor:
        return self._get_patch_mode_weights(temperature=1.0, single_path=False)

    def resolve_patch_mode(self) -> str:
        weights = self.get_patch_mode_probs()
        idx = int(torch.argmax(weights.detach()).item())
        return self.patch_mode_names[idx]

    def freeze_patch_mode(self, patch_mode: str) -> None:
        resolved = str(patch_mode).lower()
        if resolved == "patch":
            resolved = "patch_16"
        self.patching_mode = (
            resolved if resolved in self.patch_mode_names else "direct"
        )
        if hasattr(self, "patch_alpha_logits"):
            self._parameters.pop("patch_alpha_logits", None)
            try:
                delattr(self, "patch_alpha_logits")
            except AttributeError:
                pass

    def _layer_fn(self, layer, x: torch.Tensor) -> torch.Tensor:
        attn_out = layer["self_attn"](layer["norm1"](x))
        if self.training and self.dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.dropout_p)
        x = x + attn_out
        ffn_out = layer["ffn"](layer["norm2"](x))
        if self.training and self.dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.dropout_p)
        return x + ffn_out

    def _run_encoder_layers(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for layer in self.layers:
            if self.training and self.use_checkpoint:
                x = checkpoint(self._layer_fn, layer, x, use_reentrant=False)
            else:
                x = self._layer_fn(layer, x)
        return self.final_norm(x)

    def _summarize_sequence(
        self, x: torch.Tensor, output_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.size(1) != output_len:
            output = F.interpolate(
                x.transpose(1, 2), size=output_len, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            output = x

        context = output[:, -1:, :]
        pooled = output.mean(dim=1)
        state_proj = self.state_proj(pooled)
        h_state, c_state = state_proj.chunk(2, dim=-1)
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

        return output, context, (h_state, c_state)

    def _build_patch_tokens(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        B, L, C = x.shape
        x_t = x.transpose(1, 2)
        stride = max(1, patch_size // 2)
        if L < patch_size:
            x_t = F.pad(x_t, (0, patch_size - L))

        patches = x_t.unfold(2, patch_size, stride)
        n_patches = patches.size(2)
        # PatchTST-style tokenization: each channel contributes its own patch stream.
        patches = patches.contiguous().view(B * C, n_patches, patch_size)
        tokens = self.patch_projs[str(patch_size)](patches)
        pos_param = self.patch_pos_embeds[str(patch_size)]
        if n_patches != pos_param.size(1):
            pos = F.interpolate(
                pos_param.transpose(1, 2),
                size=n_patches,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = pos_param
        return tokens + pos

    def _encode_channel_independent_patch_mode(
        self, x: torch.Tensor, patch_size: int, output_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, _, C = x.shape
        tokens = self._build_patch_tokens(x, patch_size)
        encoded = self._run_encoder_layers(tokens)
        encoded = encoded.view(B, C, encoded.size(1), self.latent_dim).mean(dim=1)
        return self._summarize_sequence(encoded, output_len)

    def _build_variate_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_var = x.transpose(1, 2)
        if T != self.target_seq_len:
            x_var = F.interpolate(
                x_var, size=self.target_seq_len, mode="linear", align_corners=False
            )
        tokens = self.variate_proj(x_var)
        pos = self.variate_pos_embed
        if pos.size(1) != C:
            pos = F.interpolate(
                pos.transpose(1, 2), size=C, mode="linear", align_corners=False
            ).transpose(1, 2)
        return tokens + pos

    def _encode_mode(
        self, x: torch.Tensor, mode: str, output_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if mode == "direct":
            encoded = self._run_encoder_layers(self.input_proj(x))
            return self._summarize_sequence(encoded, output_len=output_len)
        if mode.startswith("patch_"):
            patch_size = int(mode.split("_", 1)[1])
            return self._encode_channel_independent_patch_mode(
                x, patch_size, output_len=output_len
            )
        if mode == "multi_scale_patch":
            outputs = [
                self._encode_channel_independent_patch_mode(
                    x, size, output_len=output_len
                )
                for size in self.patch_sizes
            ]
            out = sum(item[0] for item in outputs) / float(len(outputs))
            ctx = sum(item[1] for item in outputs) / float(len(outputs))
            h = sum(item[2][0] for item in outputs) / float(len(outputs))
            c = sum(item[2][1] for item in outputs) / float(len(outputs))
            return out, ctx, (h, c)
        if mode == "hierarchical":
            outputs = {
                size: self._encode_channel_independent_patch_mode(
                    x, size, output_len=output_len
                )
                for size in self.patch_sizes
            }
            mix_logits = [
                _layer_get(layer, "patch_mix")
                for layer in self.layers
                if _layer_get(layer, "patch_mix") is not None
            ]
            if mix_logits:
                mix_weights = torch.stack(
                    [F.softmax(logits / self.temperature, dim=0) for logits in mix_logits],
                    dim=0,
                ).mean(dim=0)
            else:
                mix_weights = x.new_full((len(self.patch_sizes),), 1.0 / len(self.patch_sizes))
            out = sum(
                mix_weights[idx] * outputs[size][0]
                for idx, size in enumerate(self.patch_sizes)
            )
            ctx = sum(
                mix_weights[idx] * outputs[size][1]
                for idx, size in enumerate(self.patch_sizes)
            )
            h = sum(
                mix_weights[idx] * outputs[size][2][0]
                for idx, size in enumerate(self.patch_sizes)
            )
            c = sum(
                mix_weights[idx] * outputs[size][2][1]
                for idx, size in enumerate(self.patch_sizes)
            )
            return out, ctx, (h, c)

        if mode == "variate_tokens":
            encoded = self._run_encoder_layers(self._build_variate_tokens(x))
            return self._summarize_sequence(encoded, output_len=output_len)
        encoded = self._run_encoder_layers(self.input_proj(x))
        return self._summarize_sequence(encoded, output_len=output_len)

    def forward(
        self,
        x,
        hidden_state=None,
        *,
        temperature: float = 1.0,
        single_path: bool = False,
    ):
        _, T, _ = x.shape
        tokenizer_weights = self._get_patch_mode_weights(
            temperature=temperature, single_path=single_path
        )
        active_indices = [
            idx
            for idx, weight in enumerate(tokenizer_weights.detach())
            if float(weight.item()) > 0.0
        ]
        if not active_indices:
            active_indices = [0]

        outputs = []
        for idx in active_indices:
            mode = self.patch_mode_names[idx]
            outputs.append((tokenizer_weights[idx], self._encode_mode(x, mode, output_len=T)))

        if len(outputs) == 1:
            _, (out, ctx, state) = outputs[0]
            return out, ctx, state

        output = sum(weight * item[0] for weight, item in outputs)
        context = sum(weight * item[1] for weight, item in outputs)
        h_state = sum(weight * item[2][0] for weight, item in outputs)
        c_state = sum(weight * item[2][1] for weight, item in outputs)
        return output, context, (h_state, c_state)


class LightweightTransformerDecoder(nn.Module):
    """Improved transformer decoder with better compatibility"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_layers=2,
        dropout=0.1,
        nhead=4,
        max_seq_len=512,
        causal=True,
        self_attention_type: str = "auto",
        self_attention_position_mode: str = "auto",
        cross_attention_type: str = "auto",
        cross_attention_position_mode: str = "auto",
        use_moe: bool = False,
        ffn_variant: str | None = None,
        rope_base: float = 500000.0,
        use_checkpoint: bool = False,
        temperature: float = 1.0,
        single_path_search: bool = True,
    ):
        super().__init__()
        resolved_self_attention_type = str(self_attention_type).lower()
        resolved_self_position_mode = str(self_attention_position_mode).lower()
        resolved_cross_attention_type = str(cross_attention_type).lower()
        resolved_cross_position_mode = str(cross_attention_position_mode).lower()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.causal = causal
        self.self_attention_type = resolved_self_attention_type
        self.cross_attention_type = resolved_cross_attention_type
        resolved_ffn_variant = (
            str(ffn_variant).lower() if ffn_variant is not None else ("moe" if use_moe else "swiglu")
        )
        self.ffn_variant = resolved_ffn_variant
        self.use_moe = resolved_ffn_variant == "moe"
        self.use_checkpoint = use_checkpoint
        self.temperature = max(float(temperature), 1e-3)
        self.single_path_search = bool(single_path_search)

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": SelfAttention(
                            latent_dim,
                            heads=nhead,
                            dropout=dropout,
                            causal=causal,
                            attention_type=resolved_self_attention_type,
                            position_mode=resolved_self_position_mode,
                            rope_base=rope_base,
                            rope_max_seq_len=max_seq_len,
                        ),
                        "cross_attn": AttentionBridge(
                            latent_dim,
                            num_heads=nhead,
                            dropout=dropout,
                            attention_type=resolved_cross_attention_type,
                            position_mode=resolved_cross_position_mode,
                        ),
                        "ffn": DARTSFeedForward(
                            d_model=latent_dim,
                            expand=4,
                            dropout=dropout,
                            use_moe=self.use_moe,
                            ffn_mode=resolved_ffn_variant,
                            temperature=self.temperature,
                            single_path_search=self.single_path_search,
                        ),
                        "norm1": RMSNorm(latent_dim),
                        "norm2": RMSNorm(latent_dim),
                        "norm3": RMSNorm(latent_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(latent_dim)
        self.dropout_p = dropout
        self.state_proj = nn.Linear(latent_dim, latent_dim * 2, bias=False)
        self.state_to_token = nn.Linear(latent_dim, latent_dim, bias=False)
        self.state_mix = nn.Parameter(torch.tensor(0.5))

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-3)
        for layer in self.layers:
            ffn = _layer_get(layer, "ffn")
            if ffn is not None and hasattr(ffn, "set_temperature"):
                ffn.set_temperature(self.temperature)

    def _prepare_memory(self, memory_or_hidden, batch_size: int | None = None):
        if memory_or_hidden is None:
            return None

        if isinstance(memory_or_hidden, tuple):
            if len(memory_or_hidden) == 2:
                h, c = memory_or_hidden
                if h.dim() == 3:
                    if h.size(0) == self.num_layers:
                        memory = h.transpose(0, 1)
                    else:
                        memory = h
                else:
                    memory = h.unsqueeze(1)
            else:
                memory = memory_or_hidden[0]
        else:
            if memory_or_hidden.dim() == 3:
                if batch_size is not None and memory_or_hidden.size(0) == batch_size:
                    memory = memory_or_hidden
                elif memory_or_hidden.size(0) == self.num_layers:
                    memory = memory_or_hidden.transpose(0, 1)
                else:
                    memory = memory_or_hidden
            else:
                memory = memory_or_hidden.unsqueeze(1)

        return memory

    def _extract_state_summary(self, hidden_state, batch_size: int, device, dtype):
        if hidden_state is None:
            return None

        if isinstance(hidden_state, tuple) and len(hidden_state) == 2:
            h, _ = hidden_state
        else:
            h = hidden_state

        if h is None:
            return None

        if isinstance(h, torch.Tensor):
            if h.dim() == 3:
                state = h[-1]
            elif h.dim() == 2:
                state = h
            else:
                return None
            return state.to(device=device, dtype=dtype)

        return None

    def forward(self, tgt, memory_or_hidden, hidden_state=None):
        tgt = self.input_proj(tgt)
        batch_size = tgt.size(0)
        target_len = tgt.size(1)
        memory = self._prepare_memory(memory_or_hidden, batch_size=batch_size)

        prev_state = self._extract_state_summary(
            hidden_state,
            batch_size=batch_size,
            device=tgt.device,
            dtype=tgt.dtype,
        )

        if prev_state is not None:
            state_token = self.state_to_token(prev_state).unsqueeze(1)
            tgt = torch.cat([state_token, tgt], dim=1)

        def _decoder_layer(layer, tgt, memory):
            self_attn_out = layer["self_attn"](layer["norm1"](tgt))
            if self.training and self.dropout_p > 0:
                self_attn_out = F.dropout(self_attn_out, p=self.dropout_p)
            tgt = tgt + self_attn_out

            if memory is not None:
                cross_out = layer["cross_attn"](
                    layer["norm2"](tgt),
                    encoder_output=memory,
                )
                if self.training and self.dropout_p > 0:
                    cross_out = F.dropout(cross_out, p=self.dropout_p)
                tgt = tgt + cross_out

            ffn_out = layer["ffn"](layer["norm3"](tgt))
            if self.training and self.dropout_p > 0:
                ffn_out = F.dropout(ffn_out, p=self.dropout_p)
            return tgt + ffn_out

        for layer in self.layers:
            if self.training and self.use_checkpoint:
                tgt = checkpoint(
                    _decoder_layer, layer, tgt, memory, use_reentrant=False
                )
            else:
                tgt = _decoder_layer(layer, tgt, memory)

        tgt = self.final_norm(tgt)
        tgt_out = tgt[:, -target_len:, :]

        last_token = tgt_out[:, -1]
        if prev_state is not None:
            mix = torch.sigmoid(self.state_mix)
            last_token = mix * last_token + (1.0 - mix) * prev_state
        state_proj = self.state_proj(last_token)
        h_state, c_state = state_proj.chunk(2, dim=-1)
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

        return tgt_out, (h_state, c_state)
