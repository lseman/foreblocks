from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==============================================================
# Base Interfaces
# ==============================================================

class EncoderBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError("Subclasses must implement forward method")


class DecoderBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError("Subclasses must implement forward method")


# ==============================================================
# LSTM
# ==============================================================

class LSTMEncoder(EncoderBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

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
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.fill_(0.0)
                    # Forget-gate bias to 1 (LSTM bias is [i, f, g, o])
                    n = param.size(0)
                    param[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # x: [B, T, input_size]
        output, hidden = self.lstm(x, hidden)  # output: [B, T, H*num_dirs]
        return output, hidden


class LSTMDecoder(DecoderBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,  # kept for compatibility; not used (no projection)
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  # not used
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.fill_(0.0)
                    n = param.size(0)
                    param[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Input:
          x: [B, D] or [B, T, D]
        Output:
          y: last hidden state, shape [B, H]
          hidden: (h_T, c_T)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
        elif x.dim() != 3:
            raise ValueError(f"Decoder input must be 2D or 3D, got {tuple(x.shape)}")

        lstm_out, hidden = self.lstm(x, hidden)   # [B, T, H]
        last_out = lstm_out[:, -1, :]             # [B, H]
        return last_out, hidden


# ==============================================================
# GRU
# ==============================================================

class GRUEncoder(EncoderBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

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
        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.fill_(0.0)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # x: [B, T, input_size]
        outputs, hidden = self.gru(x, hidden)  # outputs: [B, T, H*num_dirs]
        return outputs, hidden


class GRUDecoder(DecoderBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,  # kept for compatibility; not used (no projection)
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size  # not used
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.fill_(0.0)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Input:
          x: [B, D] or [B, T, D]
        Output:
          y: last hidden state, shape [B, H]
          hidden: final hidden [num_layers, B, H]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
        elif x.dim() != 3:
            raise ValueError(f"Decoder input must be 2D or 3D, got {tuple(x.shape)}")

        gru_out, hidden = self.gru(x, hidden)  # [B, T, H]
        last_out = gru_out[:, -1, :]           # [B, H]
        return last_out, hidden


# ==============================================================
# Variational Encoder Wrapper (bidi-aware)
# ==============================================================

class VariationalEncoderWrapper(nn.Module):
    """
    Wraps an encoder (LSTM/GRU). Produces:
      - encoder_outputs
      - (z, mu, logvar), where z ~ N(mu, diag(exp(logvar)))
    Uses last layer's hidden for each direction and concatenates directions.
    """
    def __init__(self, base_encoder: nn.Module, latent_dim: int):
        super().__init__()
        self.base_encoder = base_encoder
        self.latent_dim = latent_dim
        self.hidden_size = base_encoder.hidden_size
        self.num_layers = base_encoder.num_layers
        self.num_directions = getattr(base_encoder, "num_directions", 1)

        in_dim = self.hidden_size * self.num_directions
        self.hidden_to_mu = nn.Linear(in_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: Tensor):
        outputs, hidden = self.base_encoder(x)
        # hidden for LSTM: (h, c); for GRU: h
        h_last = hidden[0] if isinstance(hidden, tuple) else hidden  # [L*num_dirs, B, H]

        # Reshape to [L, num_dirs, B, H], take last layer, then concat directions
        LND = h_last.view(self.num_layers, self.num_directions, h_last.size(1), h_last.size(2))
        h_top = LND[-1]                               # [num_dirs, B, H]
        h = h_top.transpose(0, 1).reshape(h_top.size(1), -1)  # [B, H*num_dirs]

        mu = self.hidden_to_mu(h)
        logvar = self.hidden_to_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return outputs, (z, mu, logvar)


# ==============================================================
# Latent-conditioned Decoder Wrapper
# ==============================================================

class LatentConditionedDecoder(nn.Module):
    """
    Initializes the decoder's initial hidden state from a latent vector.
    Works with LSTMDecoder or GRUDecoder that accept (x, hidden).
    No output projection is applied in the base decoders.
    """
    def __init__(
        self,
        base_decoder: nn.Module,
        latent_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size * num_layers)
        self.latent_to_cell = (
            nn.Linear(latent_dim, hidden_size * num_layers)
            if self.rnn_type == "lstm"
            else None
        )

    def forward(self, x: Tensor, latent: Tensor):
        """
        Args:
          x: [B, D] or [B, T, D]
          latent: [B, Z]
        Returns:
          y, hidden from base decoder (y is [B, H], no projection)
        """
        B = latent.size(0)
        h0 = torch.tanh(self.latent_to_hidden(latent)).view(self.num_layers, B, self.hidden_size)
        if self.rnn_type == "lstm":
            if self.latent_to_cell is None:
                raise RuntimeError("lat_to_cell is None for LSTM mode.")
            c0 = torch.tanh(self.latent_to_cell(latent)).view(self.num_layers, B, self.hidden_size)
            hidden0 = (h0, c0)
        else:
            hidden0 = h0

        return self.base_decoder(x, hidden0)


# ==============================================================
# VAE utilities
# ==============================================================

def compute_kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    KL divergence between N(mu, diag(exp(logvar))) and N(0, I), averaged over batch.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss_function(recon_x: Tensor, target: Tensor, mu: Tensor, logvar: Tensor):
    """
    VAE loss = reconstruction loss + KL divergence
    - recon_x and target are expected to be same-shaped tensors.
    - No projection layer is assumed in decoders; adapt target accordingly.
    """
    recon_loss = F.mse_loss(recon_x, target, reduction="mean")
    kl = compute_kl_divergence(mu, logvar)
    return recon_loss + kl, recon_loss, kl
