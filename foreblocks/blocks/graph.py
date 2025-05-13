import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class SGConv(nn.Module):
    """
    Spectral Graph Convolutional layer for time series.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        k: int = 3,
        include_feature_edges: bool = True,
        learned_adjacency: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.include_feature_edges = include_feature_edges

        self.weight = nn.Parameter(torch.Tensor(k, input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))

        if learned_adjacency:
            self.adj_param = nn.Parameter(torch.randn(input_size, input_size))
        else:
            self.register_parameter('adj_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def _get_graph_filter(self, seq_len: int) -> torch.Tensor:
        device = self.weight.device
        n = seq_len * self.input_size

        # Construct adjacency matrix
        adj = torch.zeros(n, n, device=device)

        if self.include_feature_edges and self.adj_param is not None:
            feature_adj = torch.sigmoid(self.adj_param)

            for i in range(seq_len - 1):
                for f in range(self.input_size):
                    idx = i * self.input_size + f
                    next_idx = (i + 1) * self.input_size + f
                    adj[idx, next_idx] = 1.0
                    adj[next_idx, idx] = 1.0

            for i in range(seq_len):
                for f1 in range(self.input_size):
                    for f2 in range(self.input_size):
                        if f1 != f2:
                            idx1 = i * self.input_size + f1
                            idx2 = i * self.input_size + f2
                            adj[idx1, idx2] = feature_adj[f1, f2]
        else:
            for i in range(seq_len - 1):
                for f in range(self.input_size):
                    idx = i * self.input_size + f
                    next_idx = (i + 1) * self.input_size + f
                    adj[idx, next_idx] = 1.0
                    adj[next_idx, idx] = 1.0

        # Add self-loops
        adj += torch.eye(n, device=device)

        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        d = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        L = torch.eye(n, device=device) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        # Safe spectral norm rescaling
        norm = torch.linalg.norm(L, ord=2)
        norm = norm if norm > 1e-6 else 1.0  # Prevent division by near-zero
        L_scaled = 2.0 * L / norm - torch.eye(n, device=device)

        # Compute Chebyshev polynomials
        cheb_polynomials = [torch.eye(n, device=device), L_scaled]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * L_scaled @ cheb_polynomials[-1] - cheb_polynomials[-2])
        return cheb_polynomials

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        cheb_polynomials = self._get_graph_filter(seq_len)
        x_reshaped = x.reshape(batch_size, seq_len * self.input_size)

        out = torch.zeros(batch_size, seq_len * self.output_size, device=x.device, dtype=x.dtype)

        for k in range(self.k):
            x_k = x_reshaped @ cheb_polynomials[k]
            x_k = x_k.reshape(batch_size, seq_len, self.input_size)
            x_k = x_k @ self.weight[k]
            out += x_k.reshape(batch_size, seq_len * self.output_size)

        out = out.reshape(batch_size, seq_len, self.output_size)
        out = out + self.bias

        # Optional: guard against NaNs in output
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return out
