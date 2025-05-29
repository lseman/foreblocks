import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Union, Dict


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism that allows the model to focus on different
    temporal scales and patterns within the data.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Multi-head attention projections
        self.query = nn.Linear(input_dim, hidden_dim * num_heads)
        self.key = nn.Linear(input_dim, hidden_dim * num_heads)
        self.value = nn.Linear(input_dim, hidden_dim * num_heads)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * num_heads, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Attention-weighted output [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.size()

        # Compute query, key, value projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.hidden_dim)

        # Transpose for attention computation [batch_size, num_heads, seq_len, hidden_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)

        # Reshape and project back to original dimension
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output_projection(context)

        return output


class TemporalConvLayer(nn.Module):
    """
    Temporal convolution layer with dilated convolutions to capture
    hierarchical patterns at different time scales.

    This layer uses dilated convolutions to efficiently expand the receptive field,
    allowing the model to capture both short-term dependencies (with small dilation)
    and long-term dependencies (with large dilation) without increasing the number
    of parameters significantly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            dilation: Dilation factor for the convolution
            causal: Whether to use causal convolution
            dropout: Dropout probability
        """
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Calculate padding based on kernel size and dilation
        # For causal convolution, we only pad on the left
        if causal:
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=0, dilation=dilation
            )
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation,
            )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, in_channels]

        Returns:
            Convolved output [batch_size, seq_len, out_channels]
        """
        # Convert to [batch_size, in_channels, seq_len] for 1D convolution
        x_conv = x.transpose(1, 2)

        # Apply causal padding if needed
        if self.causal:
            x_conv = F.pad(x_conv, (self.padding, 0))

        # Apply convolution
        y = self.conv(x_conv)

        # Convert back to [batch_size, seq_len, out_channels]
        y = y.transpose(1, 2)

        # Apply normalization, activation, and dropout
        y = self.layer_norm(y)
        y = self.activation(y)
        y = self.dropout(y)

        return y


class HierarchicalBlock(nn.Module):
    """
    Hierarchical block that combines temporal convolutions at different scales
    with attention mechanisms for adaptive feature extraction.

    This block uses multiple levels of representation to capture patterns
    at different temporal scales and granularities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_levels: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        residual_connections: bool = True,
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for processing
            output_dim: Output dimension
            num_levels: Number of hierarchical levels
            kernel_size: Kernel size for convolutions
            attention_heads: Number of attention heads
            dropout: Dropout probability
            residual_connections: Whether to use residual connections between levels
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        self.residual_connections = residual_connections

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Hierarchical temporal convolutions with increasing dilation
        self.temporal_convs = nn.ModuleList(
            [
                TemporalConvLayer(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_levels)
            ]
        )

        # Hierarchical attention mechanisms
        self.attention_layers = nn.ModuleList(
            [
                HierarchicalAttention(
                    hidden_dim,
                    hidden_dim // 2,
                    num_heads=attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_levels)
            ]
        )

        # Level-specific feature transformation
        self.level_transforms = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_levels)]
        )

        # Gate mechanism for adaptive feature fusion
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_levels)]
        )

        # Output projections - maintains original structure for compatibility
        self.backcast_projection = nn.Linear(hidden_dim, input_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Additional processing for hierarchical embeddings
        self.level_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_levels)]
        )

        # Cross-level attention for information exchange
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Tuple of:
            - Backcast (reconstructed input) [batch_size, seq_len, input_dim]
            - Output embedding [batch_size, seq_len, output_dim]
        """
        # Project input to hidden dimension
        h = self.input_projection(x)

        # Process through hierarchical levels
        level_outputs = []
        current_h = h

        for i in range(self.num_levels):
            # Apply temporal convolution
            conv_out = self.temporal_convs[i](current_h)

            # Apply attention mechanism
            attn_out = self.attention_layers[i](conv_out)

            # Transform features
            level_features = self.level_transforms[i](attn_out)

            # Compute gate for adaptive feature fusion
            gate_input = torch.cat([current_h, level_features], dim=-1)
            gate = torch.sigmoid(self.gates[i](gate_input))

            # Update representation with gated residual connection
            if self.residual_connections:
                current_h = current_h + gate * level_features
            else:
                current_h = gate * level_features

            # Apply normalization
            current_h = self.layer_norm(current_h)

            # Project level output for hierarchical representation
            level_proj = self.level_projections[i](current_h)

            # Store level output
            level_outputs.append(level_proj)

        # Cross-level attention for information exchange between hierarchical levels
        if len(level_outputs) > 1:
            # Stack level outputs [num_levels, batch_size, seq_len, hidden_dim]
            stacked_levels = torch.stack(level_outputs, dim=0)
            batch_size, seq_len, hidden_dim = stacked_levels.shape[1:]

            # Reshape for multi-head attention [batch_size*seq_len, num_levels, hidden_dim]
            reshaped_levels = stacked_levels.permute(1, 2, 0, 3).reshape(
                batch_size * seq_len, self.num_levels, hidden_dim
            )

            # Apply self-attention across levels
            attended_levels, _ = self.cross_level_attention(
                reshaped_levels, reshaped_levels, reshaped_levels
            )

            # Reshape back [num_levels, batch_size, seq_len, hidden_dim]
            attended_levels = attended_levels.reshape(
                batch_size, seq_len, self.num_levels, hidden_dim
            )
            attended_levels = attended_levels.permute(2, 0, 1, 3)

            # Update level outputs with attended information
            level_outputs = [attended_levels[i] for i in range(self.num_levels)]

        # Combine information from all hierarchical levels
        final_representation = sum(level_outputs) / len(level_outputs)
        final_representation = self.dropout(final_representation)

        # Compute backcast for residual connection compatibility
        backcast = self.backcast_projection(final_representation)

        # Compute output embedding
        output_embedding = self.output_projection(final_representation)

        return backcast, output_embedding


class NHA(nn.Module):
    """
    Neural Hierarchical Architecture (NHA) model.

    A deep learning architecture that leverages hierarchical processing to capture
    patterns at multiple temporal scales. This model produces hierarchical embeddings
    suitable for seq2seq frameworks or other downstream tasks.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        num_levels_per_block: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        share_blocks: bool = False,
        pooling: str = "attention",
    ):
        """
        Args:
            input_dim: Input dimension
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden dimension for processing
            num_blocks: Number of hierarchical blocks
            num_levels_per_block: Number of hierarchical levels per block
            kernel_size: Kernel size for convolutions
            attention_heads: Number of attention heads
            dropout: Dropout probability
            share_blocks: Whether to share parameters across blocks
            pooling: How to aggregate sequence information ("attention", "mean", "max")
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.share_blocks = share_blocks
        self.pooling = pooling

        # Create stack of hierarchical blocks
        if share_blocks:
            self.block = HierarchicalBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,  # Output has same dim as hidden layers
                num_levels=num_levels_per_block,
                kernel_size=kernel_size,
                attention_heads=attention_heads,
                dropout=dropout,
            )
            self.blocks = nn.ModuleList([self.block] * num_blocks)
        else:
            self.blocks = nn.ModuleList(
                [
                    HierarchicalBlock(
                        input_dim=input_dim if i == 0 else hidden_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,  # Output has same dim as hidden layers
                        num_levels=num_levels_per_block,
                        kernel_size=kernel_size,
                        attention_heads=attention_heads,
                        dropout=dropout,
                    )
                    for i in range(num_blocks)
                ]
            )

        # Optional temporal attention pooling
        if pooling == "attention":
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1),
            )

        # Final embedding projection
        self.embedding_projection = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence dimension to create a single embedding vector

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Pooled representation [batch_size, hidden_dim]
        """
        if self.pooling == "mean":
            # Mean pooling over sequence dimension
            return torch.mean(x, dim=1)

        elif self.pooling == "max":
            # Max pooling over sequence dimension
            return torch.max(x, dim=1)[0]

        elif self.pooling == "attention":
            # Attention-based pooling
            attention_scores = self.temporal_attention(x)  # [batch_size, seq_len, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            weighted_sum = torch.sum(x * attention_weights, dim=1)
            return weighted_sum

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def forward(
        self, x: torch.Tensor, return_sequence: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            return_sequence: Whether to return sequence embeddings or just the pooled embedding

        Returns:
            If return_sequence=True:
                Tuple of:
                - Sequence embeddings [batch_size, seq_len, embedding_dim]
                - Pooled embedding [batch_size, embedding_dim]
            Else:
                Pooled embedding [batch_size, embedding_dim]
        """
        batch_size, seq_len, _ = x.size()
        current_input = x

        # Process through hierarchical blocks
        for i, block in enumerate(self.blocks):
            # Get block outputs
            backcast, block_output = block(current_input)

            # Use block output directly if not first block, otherwise use residual connection
            if i == 0:
                current_input = block_output
            else:
                current_input = current_input + block_output

        # Final sequence representation
        sequence_embedding = self.embedding_projection(current_input)
        sequence_embedding = self.layer_norm(sequence_embedding)
        sequence_embedding = self.dropout(sequence_embedding)

        # Create pooled embedding
        pooled_embedding = self._temporal_pool(sequence_embedding)

        if return_sequence:
            return sequence_embedding, pooled_embedding
        else:
            return pooled_embedding

    def extract_hierarchical_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract hierarchical features at different levels for interpretation
        or use in downstream tasks.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Dictionary with hierarchical features and embeddings
        """
        batch_size, seq_len, _ = x.size()
        current_input = x

        # Store intermediate values
        results = {"input": x, "block_outputs": [], "backcasts": []}

        # Process through blocks
        for i, block in enumerate(self.blocks):
            # Pass current input through block
            backcast, block_output = block(current_input)

            # Store intermediate values
            results["backcasts"].append(backcast)
            results["block_outputs"].append(block_output)

            # Update input for next block
            if i == 0:
                current_input = block_output
            else:
                current_input = current_input + block_output

        # Final sequence embedding
        sequence_embedding = self.embedding_projection(current_input)
        sequence_embedding = self.layer_norm(sequence_embedding)
        sequence_embedding = self.dropout(sequence_embedding)

        # Create pooled embedding
        pooled_embedding = self._temporal_pool(sequence_embedding)

        results["sequence_embedding"] = sequence_embedding
        results["pooled_embedding"] = pooled_embedding

        return results
