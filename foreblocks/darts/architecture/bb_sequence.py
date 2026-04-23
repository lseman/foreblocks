"""Sequence block base classes and search infrastructure."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "ArchitectureNormalizer",
    "SearchableDecomposition",
    "SequenceStateAdapter",
    "BaseMixedSequenceBlock",
    "BaseFixedSequenceBlock",
]


class ArchitectureNormalizer(nn.Module):
    """Normalizes outputs from different architectures for compatibility"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.rnn_proj = nn.Linear(latent_dim, latent_dim)
        self.transformer_proj = nn.Linear(latent_dim, latent_dim)
        self.state_norm = nn.LayerNorm(latent_dim)

    def normalize_state(self, state, arch_type: str):
        if state is None:
            return None, None

        if arch_type == "lstm":
            if isinstance(state, tuple) and len(state) == 2:
                h, c = state
                return self.state_norm(h), self.state_norm(c)
            else:
                h = state if not isinstance(state, tuple) else state[0]
                c = torch.zeros_like(h)
                return self.state_norm(h), self.state_norm(c)

        elif arch_type == "gru":
            h = state if not isinstance(state, tuple) else state[0]
            c = torch.zeros_like(h)
            return self.state_norm(h), self.state_norm(c)

        elif arch_type == "transformer":
            if isinstance(state, tuple) and len(state) == 2:
                h, c = state
                return self.state_norm(h), self.state_norm(c)
            else:
                h = state if not isinstance(state, tuple) else state[0]
                c = torch.zeros_like(h) if h is not None else None
                return self.state_norm(h) if h is not None else None, (
                    self.state_norm(c) if c is not None else None
                )

    def normalize_output(self, output: torch.Tensor, arch_type: str) -> torch.Tensor:
        if arch_type in ["lstm", "gru"]:
            return self.rnn_proj(output)
        elif arch_type == "transformer":
            return self.transformer_proj(output)
        return output


class SearchableDecomposition(nn.Module):
    """Searchable decomposition front-end for trend/seasonal mixing."""

    def __init__(self, c_in: int, kernel_size: int = 25):
        super().__init__()
        self.c_in = int(c_in)
        self.kernel_size = int(max(3, kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.alpha_logits = nn.Parameter(torch.randn(4) * 0.01)
        pad = self.kernel_size // 2
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        self.learnable_filter = nn.Conv1d(
            self.c_in,
            self.c_in,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.c_in,
            bias=False,
        )
        nn.init.dirac_(self.learnable_filter.weight)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if x.dim() != 3:
            return x

        tau = max(float(temperature), 1e-3)
        weights = F.softmax(self.alpha_logits / tau, dim=0)

        x_t = x.transpose(1, 2)
        trend = self.avg_pool(x_t).transpose(1, 2)
        seasonal = x - trend
        filtered = self.learnable_filter(x_t).transpose(1, 2)

        return (
            weights[0] * x
            + weights[1] * trend
            + weights[2] * seasonal
            + weights[3] * filtered
        )

    def get_alphas(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=0)


class SequenceStateAdapter:
    """Shared hidden-state adapter for mixed/fixed encoder-decoder blocks."""

    @staticmethod
    def _extract_tensor_dtype(
        state, fallback: torch.dtype = torch.float32
    ) -> torch.dtype:
        if (
            isinstance(state, tuple)
            and len(state) > 0
            and isinstance(state[0], torch.Tensor)
        ):
            return state[0].dtype
        if isinstance(state, torch.Tensor):
            return state.dtype
        return fallback

    @staticmethod
    def _coerce_pair_state(
        state,
        *,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = torch.zeros(
                num_layers, batch_size, hidden_size, device=device, dtype=dtype
            )
            c = torch.zeros(
                num_layers, batch_size, hidden_size, device=device, dtype=dtype
            )
            return h, c

        if isinstance(state, tuple) and len(state) == 2:
            h, c = state
        else:
            h = state
            c = torch.zeros_like(h)

        h = h.to(device=device, dtype=dtype).contiguous()
        c = c.to(device=device, dtype=dtype).contiguous()

        if h.dim() == 2:
            h = h.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
            c = c.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        return h, c

    @staticmethod
    def ensure_rnn_state(
        state,
        rnn_type: str,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        state_dtype = dtype or SequenceStateAdapter._extract_tensor_dtype(state)
        h, c = SequenceStateAdapter._coerce_pair_state(
            state,
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
            dtype=state_dtype,
        )
        if rnn_type == "gru":
            return h
        return (h, c)

    @staticmethod
    def split_mixed_decoder_states(
        hidden_state,
        *,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        pair_state = SequenceStateAdapter.ensure_rnn_state(
            hidden_state,
            rnn_type="lstm",
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
        )
        h, c = pair_state
        lstm_state = (h, c)
        gru_state = h
        trans_state = (h, c)
        return lstm_state, gru_state, trans_state


class BaseMixedSequenceBlock(nn.Module):
    """Mother class for mixed encoder/decoder blocks."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        num_layers: int = 2,
        num_options: int = 3,
        single_path_search: bool = True,
        arch_path_keep_prob: float = 0.85,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.temperature = temperature
        self.num_layers = num_layers
        self.single_path_search = single_path_search
        self.arch_path_keep_prob = float(min(max(arch_path_keep_prob, 0.0), 1.0))
        self._warned_no_grad_sampling = False

        self.lstm = nn.LSTM(
            input_dim,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_dim,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )

        init = 0.01 * torch.randn(num_options)
        self.register_parameter("alphas", nn.Parameter(init))
        layer_offsets = 0.01 * torch.randn(num_layers, num_options)
        self.register_parameter("layer_alpha_offsets", nn.Parameter(layer_offsets))
        self._init_rnn_weights()

    def _init_rnn_weights(self):
        for rnn in [self.lstm, self.gru]:
            for name, param in rnn.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.data.fill_(0)
                    if "lstm" in str(rnn.__class__).lower() and "bias_ih" in name:
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    def _get_layer_arch_logits(self) -> torch.Tensor:
        return self.alphas.unsqueeze(0) + self.layer_alpha_offsets

    def _should_use_stochastic_arch_sampling(self) -> bool:
        if not self.training:
            return False
        if torch.is_grad_enabled():
            return True
        if self.single_path_search and not self._warned_no_grad_sampling:
            warnings.warn(
                "Single-path architecture sampling requested while model is in train() "
                "but gradients are disabled. Falling back to deterministic softmax. "
                "Call eval() for inference to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_no_grad_sampling = True
        return False

    @staticmethod
    def _sample_straight_through_gumbel(
        logits: torch.Tensor, tau: float, dim: int = -1
    ) -> torch.Tensor:
        soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=dim)
        hard_idx = soft.argmax(dim=dim, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(dim, hard_idx, 1.0)
        return hard - soft.detach() + soft

    def _get_arch_weights(self, layer_idx: int | None = None) -> torch.Tensor:
        logits = self._get_layer_arch_logits()
        tau = max(float(self.temperature), 1e-3)
        if self._should_use_stochastic_arch_sampling():
            if self.single_path_search:
                weights = self._sample_straight_through_gumbel(logits, tau=tau, dim=-1)
            else:
                weights = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            weights = F.softmax(logits / tau, dim=-1)

        if (
            self.training
            and self.single_path_search
            and torch.is_grad_enabled()
            and self.arch_path_keep_prob < 1.0
        ):
            keep = torch.full_like(weights, self.arch_path_keep_prob)
            mask = torch.bernoulli(keep)
            if mask.dim() == 2:
                zero_rows = mask.sum(dim=-1, keepdim=True) == 0
                if zero_rows.any():
                    top_idx = weights.argmax(dim=-1, keepdim=True)
                    for r in torch.where(zero_rows.squeeze(-1))[0].tolist():
                        mask[r, top_idx[r, 0]] = 1.0
            elif mask.dim() == 1 and mask.sum().item() == 0:
                mask[weights.argmax(dim=-1)] = 1.0

            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if layer_idx is None:
            return weights

        layer_idx = max(0, min(int(layer_idx), weights.size(0) - 1))
        return weights[layer_idx]

    def _get_output_arch_weights(self) -> torch.Tensor:
        return self._get_arch_weights(layer_idx=self.num_layers - 1)

    def get_layer_alphas(self) -> torch.Tensor:
        return F.softmax(self._get_layer_arch_logits(), dim=-1)

    def get_alphas(self) -> torch.Tensor:
        return self.get_layer_alphas().mean(dim=0)

    def set_temperature(self, temp: float):
        self.temperature = max(float(temp), 1e-3)

    def orthogonal_regularization(self) -> torch.Tensor:
        reg = None
        for rnn in [self.lstm, self.gru]:
            for name, param in rnn.named_parameters():
                if "weight_hh" not in name or param.dim() != 2:
                    continue
                rows, cols = param.shape
                if cols <= 0:
                    continue
                if rows % cols == 0:
                    gates = rows // cols
                    chunks = param.view(gates, cols, cols)
                    for mat in chunks:
                        gram = mat @ mat.t()
                        eye = torch.eye(cols, device=mat.device, dtype=mat.dtype)
                        term = (gram - eye).pow(2).mean()
                        reg = term if reg is None else reg + term
                else:
                    gram = param @ param.t()
                    eye = torch.eye(rows, device=param.device, dtype=param.dtype)
                    term = (gram - eye).pow(2).mean()
                    reg = term if reg is None else reg + term

        if reg is None:
            return self.alphas.new_zeros(())
        return reg


class BaseFixedSequenceBlock(nn.Module):
    """Mother class for fixed encoder/decoder blocks."""

    def __init__(
        self,
        rnn=None,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        transformer_factory=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        rnn_or_type = rnn if rnn is not None else rnn_type
        if rnn_or_type is None:
            raise ValueError("Either 'rnn' or 'rnn_type' must be provided")

        if isinstance(rnn_or_type, str):
            self.rnn_type = rnn_or_type.lower()
            if self.rnn_type == "lstm":
                self.rnn = nn.LSTM(
                    input_dim,
                    latent_dim,
                    num_layers,
                    dropout=dropout if dropout > 0 else 0,
                    batch_first=True,
                )
            elif self.rnn_type == "gru":
                self.rnn = nn.GRU(
                    input_dim,
                    latent_dim,
                    num_layers,
                    dropout=dropout if dropout > 0 else 0,
                    batch_first=True,
                )
            elif self.rnn_type == "transformer":
                if transformer_factory is None:
                    raise ValueError(
                        "transformer_factory must be provided for transformer type"
                    )
                self.rnn = transformer_factory(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            self.rnn = rnn_or_type
            if isinstance(self.rnn, nn.LSTM):
                self.rnn_type = "lstm"
                self.latent_dim = self.rnn.hidden_size
            elif isinstance(self.rnn, nn.GRU):
                self.rnn_type = "gru"
                self.latent_dim = self.rnn.hidden_size
            elif hasattr(self.rnn, "latent_dim"):
                self.rnn_type = "transformer"
                self.latent_dim = self.rnn.latent_dim
            else:
                self.rnn_type = "unknown"

    def get_alphas(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.rnn_type == "lstm":
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        elif self.rnn_type == "gru":
            return torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
        return torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)

    def set_temperature(self, temp: float):
        pass
