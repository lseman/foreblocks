import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D
from torch.utils.data import DataLoader, TensorDataset


class FixedOp(nn.Module):
    def __init__(self, selected_op: nn.Module):
        super().__init__()
        self.op = selected_op

    def forward(self, x):
        return self.op(x)


class TransformerOp(nn.Module):
    """Single-layer Transformer encoder block"""

    def __init__(self, input_dim, latent_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, latent_dim)
            if input_dim != latent_dim
            else nn.Identity()
        )

        self.transformer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        return self.norm(out + x)  # Residual


# Define individual operations for the search space
class IdentityOp(nn.Module):
    """Simple projection"""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        # Apply to each time step
        return self.linear(x)


# Alternative implementation that avoids pytorch_wavelets completely
class WaveletOp(nn.Module):
    def __init__(self, input_dim, latent_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        # Use dilated depthwise convolutions
        self.dilated_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        input_dim,
                        input_dim,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        groups=input_dim,
                    ),
                    nn.Conv1d(input_dim, input_dim, kernel_size=1),  # pointwise
                    nn.BatchNorm1d(input_dim),
                    nn.SiLU(),
                )
                for d in [1, 2, 4][:num_scales]
            ]
        )

        # Fuse projection using 1x1 conv
        self.fuse_proj = nn.Conv1d(input_dim * num_scales, latent_dim, kernel_size=1)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        B, L, C = x.shape
        x_conv = x.transpose(1, 2)  # [B, C, L]

        features = [conv(x_conv) for conv in self.dilated_convs]
        features = [
            f[:, :, :L] if f.size(2) > L else F.pad(f, (0, L - f.size(2)))
            for f in features
        ]

        concat = torch.cat(features, dim=1)  # [B, C*num_scales, L]
        out = self.fuse_proj(concat)  # [B, latent_dim, L]
        out = out.transpose(1, 2)  # [B, L, latent_dim]
        return self.norm(out)


class FourierOp(nn.Module):
    """Efficient Fourier-based feature extractor with frequency truncation"""

    def __init__(self, input_dim, latent_dim, seq_length, num_frequencies=None):
        super().__init__()

        self.seq_length = seq_length
        self.num_frequencies = (
            min(seq_length // 2 + 1, 10)
            if num_frequencies is None
            else min(num_frequencies, seq_length // 2 + 1)
        )

        self.freq_proj = nn.Linear(input_dim * 2, latent_dim)  # real + imag per freq

    def forward(self, x):
        B, L, C = x.shape

        # Apply FFT safely
        if x.is_cuda and not (self.seq_length & (self.seq_length - 1) == 0):
            x = x.float()  # avoid float16 cuFFT crash

        x_fft = torch.fft.rfft(x, dim=1)  # [B, F, C]
        x_fft = x_fft[:, : self.num_frequencies, :]  # truncate

        # Project real+imag per frequency, per feature
        real = x_fft.real  # [B, F, C]
        imag = x_fft.imag
        combined = torch.cat([real, imag], dim=-1)  # [B, F, 2C]

        # Project frequency info
        freq_emb = self.freq_proj(combined)  # [B, F, latent]

        # Aggregate across frequencies (e.g. mean/max)
        freq_summary = freq_emb.mean(dim=1)  # [B, latent]

        # Repeat across time
        return freq_summary.unsqueeze(1).repeat(1, L, 1)  # [B, L, latent]


class AttentionOp(nn.Module):
    """Self-attention operation"""

    def __init__(self, input_dim, latent_dim, num_heads=4, dropout=0.1):
        super().__init__()

        # Project input to latent dimension if needed
        self.input_proj = (
            nn.Linear(input_dim, latent_dim)
            if input_dim != latent_dim
            else nn.Identity()
        )

        # Self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        # Project input if needed
        x_proj = self.input_proj(x)

        # Apply self-attention
        attn_output, _ = self.self_attention(x_proj, x_proj, x_proj)

        # Add residual connection and normalization
        return self.norm(x_proj + attn_output)


class TimeConvOp(nn.Module):
    def __init__(self, input_dim, latent_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            input_dim, latent_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.conv(x).transpose(1, 2)  # (B, L, C)
        return self.norm(x)


class MixedOp(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        seq_length,
        available_ops=None,
        drop_prob=0.1,
        normalize_outputs=False,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.normalize_outputs = normalize_outputs
        self.tau = 1.0  # temperature for Gumbel softmax

        self.op_map = {
            "Identity": lambda: IdentityOp(input_dim, latent_dim),
            "Wavelet": lambda: WaveletOp(input_dim, latent_dim),
            "Fourier": lambda: FourierOp(input_dim, latent_dim, seq_length),
            "Attention": lambda: AttentionOp(input_dim, latent_dim),
            "TCN": lambda: TCNOp(input_dim, latent_dim),
            "ResidualMLP": lambda: ResidualMLPOp(input_dim, latent_dim),
            "ConvMixer": lambda: ConvMixerOp(input_dim, latent_dim),
            "Transformer": lambda: TransformerOp(input_dim, latent_dim),
            "GRN": lambda: GRNOp(input_dim, latent_dim),
            "TimeConv": lambda: TimeConvOp(input_dim, latent_dim),
        }

        if not available_ops:
            available_ops = self.op_map.keys()

        self.available_ops = list(available_ops)
        self.ops = nn.ModuleList(
            [self.op_map[op]() for op in self.available_ops if op in self.op_map]
        )
        self.alphas = nn.Parameter(torch.zeros(len(self.ops)), requires_grad=True)

    def gumbel_softmax(self, logits, tau=None, hard=False):
        tau = tau if tau is not None else self.tau
        U = torch.rand_like(logits)
        gumbels = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        y = (logits + gumbels) / tau
        y_soft = F.softmax(y, dim=-1)
        if hard:
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return (y_hard - y_soft).detach() + y_soft
        return y_soft

    def forward(self, x):
        weights = self.gumbel_softmax(self.alphas, hard=not self.training)
        op_outputs = []
        for i, op in enumerate(self.ops):
            if self.training and torch.rand(1).item() < self.drop_prob:
                continue
            try:
                out = op(x)
                if self.normalize_outputs:
                    out = F.layer_norm(out, out.shape[-1:])
                op_outputs.append(weights[i] * out)
            except Exception as e:
                raise RuntimeError(f"Operation {i} ({type(op).__name__}) failed: {e}")
        return sum(op_outputs) if op_outputs else x

    def get_alphas(self):
        return F.softmax(self.alphas, dim=0)

    def log_op_contributions(self):
        probs = F.softmax(self.alphas, dim=0)
        for name, prob in zip(self.available_ops, probs):
            print(f"{name}: {prob.item():.4f}")


class DARTSCell(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        seq_length,
        num_nodes=4,
        initial_search=False,
        selected_ops=None,
        aggregation="mean",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.aggregation = aggregation

        self.input_proj = nn.Linear(input_dim, latent_dim)

        if initial_search:
            self.available_ops = ["Identity", "Attention"]
        elif selected_ops:
            self.available_ops = selected_ops
        else:
            self.available_ops = [
                "Identity",
                "TimeConv",
                "GRN",
                "Wavelet",
                "Fourier",
                "Attention",
                "TCN",
                "ResidualMLP",
                "ConvMixer",
                "Transformer",
            ]

        self.edges = nn.ModuleList(
            [
                MixedOp(latent_dim, latent_dim, seq_length, self.available_ops)
                for i in range(num_nodes)
                for j in range(i + 1, num_nodes)
            ]
        )

    def forward(self, x):
        x_proj = self.input_proj(x)
        nodes = [x_proj]
        edge_idx = 0
        for i in range(1, self.num_nodes):
            inputs = [self.edges[edge_idx + j](nodes[j]) for j in range(i)]
            edge_idx += i
            if self.aggregation == "mean":
                node_output = sum(inputs) / len(inputs)
            elif self.aggregation == "sum":
                node_output = sum(inputs)
            elif self.aggregation == "max":
                node_output = torch.stack(inputs, dim=0).max(dim=0)[0]
            nodes.append(node_output)
        return nodes[-1] + x_proj  # Residual connection

    def get_alphas(self):
        return [edge.get_alphas() for edge in self.edges]


class TimeSeriesDARTS(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        latent_dim=64,
        forecast_horizon=24,
        seq_length=48,
        num_cells=2,
        num_nodes=4,
        dropout=0.1,
        initial_search=False,
        selected_ops=None,
        loss_type="huber",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.forecast_horizon = forecast_horizon

        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # DARTS Cells
        self.cells = nn.ModuleList()
        for _ in range(num_cells):
            self.cells.append(
                DARTSCell(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    seq_length=seq_length,
                    num_nodes=num_nodes,
                    initial_search=initial_search,
                    selected_ops=selected_ops,
                )
            )

        # Projection to keep dim stable
        self.cell_proj = nn.ModuleList(
            [nn.Linear(latent_dim, hidden_dim) for _ in range(num_cells)]
        )

        self.cell_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_cells)]
        )

        self.cell_gate = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
                for _ in range(num_cells)
            ]
        )

        self.forecast_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

        self.forecast_decoder = nn.LSTM(
            input_size=input_dim, hidden_size=latent_dim, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
        )

        self.output_layer = nn.Linear(latent_dim, input_dim)
        self.forecast_norm = nn.LayerNorm(latent_dim)
        self.loss_type = loss_type

        self._init_weights()
        self.gate_layer = nn.Linear(latent_dim * 2, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_normal_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_seq, x_future=None, teacher_forcing_ratio=0.5):
        batch, seq_len, _ = x_seq.shape
        # Create embedded representation while keeping raw input
        x_emb = self.input_embedding(x_seq)

        # Process through cells using raw input
        cell_features = []
        for i, cell in enumerate(self.cells):
            cell_out = cell(x_seq)  # Using raw x_seq
            cell_features.append(cell_out)

        # Combine cell outputs
        cell_features = torch.stack(cell_features, dim=1)
        # mean on dim=1
        cell_features = cell_features.mean(dim=1)
        # print(cell_features.shape)
        cell_features = cell_features.transpose(1, 2)
        cell_features = cell_features.reshape(batch, seq_len, -1)

        # Combine with embeddings via gating mechanism
        gate = torch.sigmoid(self.gate_layer(torch.cat([cell_features, x_emb], dim=-1)))
        combined_features = gate * cell_features + (1 - gate) * x_emb

        # Continue with encoder/decoder as before
        h_enc, (hn_enc, cn_enc) = self.forecast_encoder(combined_features)
        # h_enc, (hn_enc, cn_enc) = self.forecast_encoder(features)
        h_n = hn_enc[-1:].contiguous()
        c_n = cn_enc

        forecasts = []
        decoder_input = x_seq[:, -1:, :]

        for t in range(self.forecast_horizon):
            output, (h_n, c_n) = self.forecast_decoder(decoder_input, (h_n, c_n))
            # output = self.forecast_norm(output)
            output = self.mlp(output)
            prediction = self.output_layer(output)
            forecasts.append(prediction.squeeze(1))

            if (
                self.training
                and x_future is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                decoder_input = x_future[:, t : t + 1, :]
            else:
                decoder_input = prediction

        return torch.stack(forecasts, dim=1)

    def calculate_loss(self, x_seq, x_future, teacher_forcing_ratio=0.5):
        pred = self.forward(x_seq, x_future, teacher_forcing_ratio)
        return F.huber_loss(pred, x_future, delta=0.1)

    def get_alphas(self):
        return [cell.get_alphas() for cell in self.cells]

    def get_alpha_dict(self):
        alpha_map = {}
        for idx, cell in enumerate(self.cells):
            edge_idx = 0
            for i in range(cell.num_nodes):
                for j in range(i + 1, cell.num_nodes):
                    alpha_map[f"cell{idx}_edge_{i}->{j}"] = cell.edges[
                        edge_idx
                    ].get_alphas()
                    edge_idx += 1
        return alpha_map


class TCNOp(nn.Module):
    def __init__(self, input_dim, latent_dim, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            input_dim, latent_dim, kernel_size, padding=dilation, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, L)
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out.transpose(1, 2)


class ResidualMLPOp(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvMixerOp(nn.Module):
    def __init__(self, input_dim, latent_dim, kernel_size=5):
        super().__init__()
        self.depthwise = nn.Conv1d(
            latent_dim,
            latent_dim,
            kernel_size,
            groups=latent_dim,
            padding=kernel_size // 2,
        )
        self.pointwise = nn.Conv1d(latent_dim, latent_dim, 1)
        self.norm = nn.LayerNorm(latent_dim)
        self.input_proj = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        x = self.input_proj(x).transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x).transpose(1, 2)
        return self.norm(x)


class GRNOp(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, latent_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        h = F.elu(self.linear1(x))
        g = self.gate(h)
        y = g * self.linear2(h)
        return self.norm(y + x)
