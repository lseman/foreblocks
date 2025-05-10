# === Core Libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# === Learnable Filtering Block ===
class LearnableFilterBlock(nn.Module):
    def __init__(self, input_size, filter_width=5, mode="conv", residual=True, activation="gelu"):
        super().__init__()
        self.mode = mode
        self.residual = residual
        self.activation = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}.get(activation, nn.GELU())

        if mode == "conv":
            self.filter = nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=filter_width,
                padding=filter_width // 2,
                groups=input_size,
                bias=False
            )
        elif mode == "mlp":
            self.filter = nn.Sequential(
                nn.Linear(input_size, input_size),
                self.activation,
                nn.Linear(input_size, input_size)
            )
        else:
            raise ValueError("mode must be 'conv' or 'mlp'")

    def forward(self, x):
        if self.mode == "conv":
            out = self.filter(x.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.filter(x)
        return x + out if self.residual else out

# === Simple GRU Block ===
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.proj(self.dropout(out).squeeze(1))

