"""Decoder execution over references to model-owned parameters and modules.

The runtime is deliberately not an nn.Module: constructing it never changes
checkpoint paths, parameter registration, or optimizer ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.tensors import hard_one_hot

def _teacher_forcing_flags(count: int, ratio: float, enabled: bool) -> list[bool]:
    """Sample CPU control-flow decisions in one RNG dispatch."""
    if not enabled or count <= 0 or ratio <= 0.0:
        return [False] * max(count, 0)
    if ratio >= 1.0:
        return [True] * count
    return (torch.rand(count) < float(ratio)).tolist()


@dataclass
class DecoderRuntime:
    forecast_decoder: nn.Module | None
    output_layer: nn.Module
    forecast_horizon: int
    latent_dim: int
    learned_horizon_queries: torch.Tensor
    _autoformer_proj: nn.Module
    norm_alpha: torch.Tensor
    decoder_query_mode_names: tuple[str, ...]
    decoder_query_alphas: torch.Tensor | None
    decoder_query_mode: str | None
    temperature: float
    variant_gdas: bool
    training: bool

    @staticmethod
    def _ensure_dtype(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() if tensor.dtype != torch.float32 else tensor

    def _get_decoder_query_mode_weights(self) -> torch.Tensor | None:
        logits = getattr(self, "decoder_query_alphas", None)
        if isinstance(logits, torch.Tensor):
            tau = max(float(getattr(self, "temperature", 1.0)), 1e-3)
            if self.training:
                return F.gumbel_softmax(
                    logits,
                    tau=tau,
                    hard=bool(getattr(self, "variant_gdas", False)),
                    dim=0,
                )
            probs = F.softmax(logits / tau, dim=0)
            if bool(getattr(self, "variant_gdas", False)):
                return hard_one_hot(probs)
            return probs

        direct = getattr(self, "decoder_query_mode", None)
        if isinstance(direct, str) and direct in getattr(
            self, "decoder_query_mode_names", ()
        ):
            ref = self.norm_alpha
            weights = ref.new_zeros(len(self.decoder_query_mode_names))
            weights[self.decoder_query_mode_names.index(direct)] = 1.0
            return weights
        return None


    def get_decoder_query_mode_probs(self) -> torch.Tensor | None:
        logits = getattr(self, "decoder_query_alphas", None)
        if isinstance(logits, torch.Tensor):
            return F.softmax(logits.detach(), dim=0)
        weights = self._get_decoder_query_mode_weights()
        if weights is not None:
            return weights.detach()
        return None


    def resolve_decoder_query_mode(self) -> str:
        probs = self.get_decoder_query_mode_probs()
        if probs is None:
            return "repeat_last"
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(self.decoder_query_mode_names):
            return self.decoder_query_mode_names[idx]
        return "repeat_last"


    def _build_parallel_decoder_input(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None = None,
        decoder_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        last_token = x_seq[:, -1:, :]
        horizon = int(self.forecast_horizon)
        history_len = max(1, horizon // 2)
        history_len = min(history_len, int(x_seq.size(1)))
        history_prefix = self._ensure_dtype(x_seq[:, -history_len:, :])
        repeated = last_token.expand(-1, horizon, -1).clone()
        zeros = torch.zeros_like(repeated)
        learned = repeated + self.learned_horizon_queries[:, :horizon, :]

        shifted = repeated
        if (
            decoder_targets is not None
            and decoder_targets.dim() == 3
            and decoder_targets.size(1) > 0
        ):
            shifted = torch.cat([last_token, decoder_targets[:, :-1, :]], dim=1)
            if shifted.size(1) < horizon:
                pad = shifted[:, -1:, :].expand(-1, horizon - shifted.size(1), -1)
                shifted = torch.cat([shifted, pad], dim=1)
            shifted = shifted[:, :horizon, :]

        future_covariates = repeated
        if x_future is not None and x_future.dim() == 3 and x_future.size(1) > 0:
            future_covariates = x_future[:, :horizon, :]
            if future_covariates.size(1) < horizon:
                pad = future_covariates[:, -1:, :].expand(
                    -1, horizon - future_covariates.size(1), -1
                )
                future_covariates = torch.cat([future_covariates, pad], dim=1)

        use_shifted_target = _teacher_forcing_flags(
            1,
            teacher_forcing_ratio,
            self.training and decoder_targets is not None,
        )[0]
        candidates = {
            "repeat_last": repeated,
            "zeros": zeros,
            "learned_horizon_queries": learned,
            "shifted_target": shifted if use_shifted_target else repeated,
            "future_covariate_queries": future_covariates,
        }

        mode_weights = self._get_decoder_query_mode_weights()
        if mode_weights is None or mode_weights.numel() != len(
            self.decoder_query_mode_names
        ):
            return self._ensure_dtype(repeated)

        candidate_stack = torch.stack(
            [
                self._ensure_dtype(candidates[name][:, :horizon, :])
                for name in self.decoder_query_mode_names
            ],
            dim=0,
        )
        future_block = (
            mode_weights.reshape(-1, 1, 1, 1) * candidate_stack
        ).sum(dim=0)
        return self._ensure_dtype(torch.cat([history_prefix, future_block], dim=1))


    def _decode_autoregressive_path(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None = None,
        decoder_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        memory: torch.Tensor | None = None,
        encoder_output: torch.Tensor | None = None,
        decoder_hidden=None,
    ) -> torch.Tensor:
        forecasts = []
        decoder_input = self._ensure_dtype(x_seq[:, -1:, :])
        hidden = decoder_hidden

        if isinstance(hidden, tuple):
            hidden = tuple(self._ensure_dtype(h) for h in hidden)
        elif hidden is not None:
            hidden = self._ensure_dtype(hidden)

        mem = self._ensure_dtype(memory)
        enc_out = self._ensure_dtype(encoder_output)
        teacher_forcing_flags = _teacher_forcing_flags(
            self.forecast_horizon,
            teacher_forcing_ratio,
            self.training and decoder_targets is not None,
        )

        for t in range(self.forecast_horizon):
            out, hidden = self.forecast_decoder(
                decoder_input,
                mem,
                hidden,
                enc_out,
            )
            prediction = self.output_layer(out)
            forecasts.append(prediction.squeeze(1))

            if (
                decoder_targets is not None
                and t < decoder_targets.size(1)
                and teacher_forcing_flags[t]
            ):
                decoder_input = self._ensure_dtype(decoder_targets[:, t : t + 1])
            else:
                decoder_input = self._ensure_dtype(prediction)

        return torch.stack(forecasts, dim=1)


    def _decode_parallel_informer_path(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None,
        decoder_targets: torch.Tensor | None,
        teacher_forcing_ratio: float,
        memory: torch.Tensor,
        encoder_output: torch.Tensor,
        decoder_hidden,
    ) -> torch.Tensor:
        decoder_input = self._build_parallel_decoder_input(
            x_seq,
            x_future=x_future,
            decoder_targets=decoder_targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        hidden = decoder_hidden
        if isinstance(hidden, tuple):
            hidden = tuple(self._ensure_dtype(h) for h in hidden)
        elif hidden is not None:
            hidden = self._ensure_dtype(hidden)

        out, _ = self.forecast_decoder(
            decoder_input,
            self._ensure_dtype(memory),
            hidden,
            self._ensure_dtype(encoder_output),
        )
        prediction = self.output_layer(out)
        return prediction[:, -self.forecast_horizon :, :]


    def _get_decoder_layers(self):
        """Get the underlying transformer layers from forecast_decoder."""
        dec = self.forecast_decoder
        if dec is None:
            return None
        # Direct transformer has .layers
        if hasattr(dec, "layers"):
            return dec.layers
        # MixedDecoder wraps it in .transformer
        if hasattr(dec, "transformer") and hasattr(dec.transformer, "layers"):
            return dec.transformer.layers
        return None


    def _decode_autoformer_path(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None,
        decoder_targets: torch.Tensor | None,
        teacher_foring_ratio: float,
        memory: torch.Tensor,
        encoder_output: torch.Tensor,
        decoder_hidden,
    ) -> torch.Tensor:
        """Autoformer-style decoder with moving-average decomposition (Wu et al. 2021).

        Each decoder layer decomposes the input into trend + seasonal components
        using a moving average, then processes them separately through the
        transformer.
        """
        horizon = self.forecast_horizon

        # Build decoder input: last token repeated horizon times
        decoder_input = x_seq[:, -1:, :].expand(-1, horizon, -1)

        # Get layers from decoder (handles both MixedDecoder and direct transformer)
        layers = self._get_decoder_layers()
        if layers is None:
            return self._decode_autoregressive_path(
                x_seq, None, decoder_targets, teacher_foring_ratio,
                memory, encoder_output, decoder_hidden,
            )

        # Project decoder input to latent dim (transformer layers expect latent_dim)
        out = self._ensure_dtype(decoder_input)
        if out.shape[-1] != self.latent_dim:
            out = self._autoformer_proj(out)

        # Apply decomposition at each decoder layer
        out = self._ensure_dtype(out)
        for layer in layers:
            # Moving-average decomposition
            kernel_size = max(1, horizon // 4)
            pad_len = kernel_size - 1
            if pad_len > 0:
                padded = F.pad(out, (0, 0, pad_len, 0))
                trend = F.avg_pool1d(
                    padded.transpose(1, 2), kernel_size, stride=1
                ).transpose(1, 2)
                seasonal = out - trend
            else:
                trend = out
                seasonal = out * 0

            # Self-attention on seasonal component
            if hasattr(layer, "self_attn"):
                seasonal = layer["norm1"](seasonal)
                seasonal = layer["self_attn"](seasonal)

            # Self-attention on trend component (inner correlation)
            if hasattr(layer, "self_attn_trend"):
                trend = layer["norm1_trend"](trend)
                trend = layer["self_attn_trend"](trend)
            else:
                trend_out = layer["norm2"](trend)
                trend_out = layer["self_attn"](trend_out) if hasattr(layer, "self_attn") else trend_out
                trend = trend + trend_out

            # Combine
            out = seasonal + trend

            # FFN
            if hasattr(layer, "ffn"):
                out = layer["norm3"](out)
                out = layer["ffn"](out)

        prediction = self.output_layer(out)
        return prediction[:, -horizon:, :]

