from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Assuming this comes from your UI/builder framework
from foreblocks.ui.node_spec import node


# ==============================================================
# Base Interfaces
# ==============================================================
class EncoderBase(nn.Module):
    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, ...]] = None,
        time_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError


class DecoderBase(nn.Module):
    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, ...]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError


# ==============================================================
# LSTM
# ==============================================================
@node(
    type_id="lstm_encoder",
    name="LSTM Encoder",
    category="Encoder",
    color="bg-gradient-to-br from-blue-700 to-blue-800",
    outputs=["encoder"],
    inputs=[],
)
class LSTMEncoder(EncoderBase):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                start = n // 4
                end = n // 2
                param.data[start:end].fill_(1.0)

    def forward(
        self,
        x: Tensor,  # [B, T, input_size]
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
        time_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if time_features is not None:
            # Example: concat on last dim (very common pattern)
            x = torch.cat([x, time_features], dim=-1)

        output, (h, c) = self.lstm(x, hidden)  # output: [B, T, H*num_dirs]
        return output, (h, c)


@node(
    type_id="lstm_decoder",
    name="LSTM Decoder",
    category="Decoder",
    color="bg-gradient-to-br from-green-700 to-green-800",
    outputs=["decoder"],
    inputs=[],
)
class LSTMDecoder(DecoderBase):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Optional projection (very useful in practice)
        self.proj = (
            nn.Linear(hidden_size, output_size)
            if output_size != hidden_size
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        # same as encoder
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: Tensor,  # [B, 1, input_size] or [B, T_dec, input_size]
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # teacher forcing or single-step input

        lstm_out, hidden = self.lstm(x, hidden)  # [B, T_dec, hidden_size]
        out = self.proj(lstm_out[:, -1])  # [B, output_size] — last step

        return out, hidden


# ==============================================================
# GRU (symmetric fixes)
# ==============================================================
@node(
    type_id="gru_encoder",
    name="GRU Encoder",
    category="Encoder",
    color="bg-gradient-to-br from-blue-600 to-blue-900",
)
class GRUEncoder(EncoderBase):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
        time_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if time_features is not None:
            x = torch.cat([x, time_features], dim=-1)

        output, hidden = self.gru(x, hidden)  # output: [B, T, H*num_dirs]
        return output, hidden


@node(
    type_id="gru_decoder",
    name="GRU Decoder",
    category="Decoder",
    color="bg-gradient-to-br from-green-600 to-green-900",
)
class GRUDecoder(DecoderBase):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = (
            nn.Linear(hidden_size, output_size)
            if output_size != hidden_size
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        gru_out, hidden = self.gru(x, hidden)
        out = self.proj(gru_out[:, -1])

        return out, hidden
