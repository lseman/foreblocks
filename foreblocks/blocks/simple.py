import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.proj(self.dropout(out).squeeze(1))
