import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List, Tuple, Union

class SGConv(nn.Module):
    """
    Spectral Graph Convolutional layer for time series.
    
    This layer treats the time series as a graph, with temporal connections
    and optional feature connections. It applies graph convolution in the
    spectral domain for efficient learning of complex temporal relationships.
    
    Inspired by spectral graph theory and its applications to time series.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        k: int = 3,  # Number of polynomial terms (filter order)
        include_feature_edges: bool = True,
        learned_adjacency: bool = True
    ):
        """
        Args:
            input_size: Number of input features
            output_size: Number of output features
            k: Order of the polynomial filter (Chebyshev filter)
            include_feature_edges: Whether to include edges between features
            learned_adjacency: Whether to learn the adjacency matrix
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.include_feature_edges = include_feature_edges
        
        # Parameters for k filter orders
        self.weight = nn.Parameter(torch.Tensor(k, input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        
        # Learnable adjacency matrix if enabled
        if learned_adjacency:
            self.adj_param = nn.Parameter(torch.randn(input_size, input_size))
        else:
            self.register_parameter('adj_param', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Glorot initialization"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def _get_graph_filter(self, seq_len: int) -> torch.Tensor:
        """
        Compute the graph filter based on the adjacency matrix.
        
        Args:
            seq_len: Sequence length for temporal graph
            
        Returns:
            Tensor of Chebyshev polynomials of the normalized Laplacian
        """
        # Form the adjacency matrix based on temporal connections
        if self.include_feature_edges and self.adj_param is not None:
            # Feature adjacency (learned)
            feature_adj = torch.sigmoid(self.adj_param)
            
            # Combine temporal and feature adjacency
            adj = torch.zeros(
                seq_len * self.input_size,
                seq_len * self.input_size,
                device=self.weight.device
            )
            
            # Add temporal connections (each node connects to its neighbors)
            for i in range(seq_len - 1):
                for f in range(self.input_size):
                    idx = i * self.input_size + f
                    next_idx = (i + 1) * self.input_size + f
                    adj[idx, next_idx] = 1.0
                    adj[next_idx, idx] = 1.0
                    
            # Add feature connections within same time step
            for i in range(seq_len):
                for f1 in range(self.input_size):
                    for f2 in range(self.input_size):
                        if f1 != f2:
                            idx1 = i * self.input_size + f1
                            idx2 = i * self.input_size + f2
                            adj[idx1, idx2] = feature_adj[f1, f2]
        else:
            # Simple temporal chain
            adj = torch.zeros(
                seq_len * self.input_size,
                seq_len * self.input_size,
                device=self.weight.device
            )
            
            for i in range(seq_len - 1):
                for f in range(self.input_size):
                    idx = i * self.input_size + f
                    next_idx = (i + 1) * self.input_size + f
                    adj[idx, next_idx] = 1.0
                    adj[next_idx, idx] = 1.0
        
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        d = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        normalized_laplacian = torch.eye(adj.size(0), device=adj.device) - torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        # Rescaled Laplacian with eigenvalues in [-1, 1]
        L_scaled = (2.0 / torch.linalg.norm(normalized_laplacian, ord=2)) * normalized_laplacian - torch.eye(adj.size(0), device=adj.device)
        
        # Chebyshev polynomials
        cheb_polynomials = [torch.eye(adj.size(0), device=adj.device), L_scaled]
        
        for i in range(2, self.k):
            cheb_polynomials.append(2 * torch.mm(L_scaled, cheb_polynomials[i-1]) - cheb_polynomials[i-2])
            
        return cheb_polynomials
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Chebyshev polynomials of the graph Laplacian
        cheb_polynomials = self._get_graph_filter(seq_len)
        
        # Reshape x for graph convolution: [batch, seq_len*input_size]
        x_reshaped = x.reshape(batch_size, seq_len * self.input_size)
        
        # Apply spectral graph convolution
        out = torch.zeros(
            batch_size,
            seq_len * self.output_size,
            device=x.device,
            dtype=x.dtype
        )
        
        for k in range(self.k):
            # Apply k-th Chebyshev polynomial filter
            x_k = torch.matmul(x_reshaped, cheb_polynomials[k])
            x_k = x_k.reshape(batch_size, seq_len, self.input_size)
            
            # Apply k-th order filter weights
            x_k = torch.matmul(x_k, self.weight[k])
            
            # Sum contributions from all orders
            out += x_k.reshape(batch_size, seq_len * self.output_size)
            
        # Reshape output and add bias
        out = out.reshape(batch_size, seq_len, self.output_size)
        out = out + self.bias
        
        return out
