import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Literal
import math
import contextlib

# External
from xformers.ops import memory_efficient_attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LatentCorrelationLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        learnable_alpha: bool = True,
        init_alpha: float = 0.5,
        use_layer_norm: bool = True,
        low_rank: bool = False,
        rank: Optional[int] = None,
        correlation_dropout: float = 0.0,
        cheb_k: int = 3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size or (2 * input_size)
        self.low_rank = low_rank
        self.rank = rank or max(1, input_size // 4)
        self.use_layer_norm = use_layer_norm
        self.cheb_k = max(1, cheb_k)
        self.eps = eps

        # Alpha blending
        if learnable_alpha:
            self.alpha = nn.Parameter(
                torch.tensor(torch.logit(torch.tensor(init_alpha)))
            )
        else:
            self.register_buffer("alpha", torch.tensor(init_alpha))

        # Correlation
        if low_rank:
            scale = 1.0 / (self.rank**0.5)
            self.corr_factors = nn.Parameter(
                torch.randn(2, input_size, self.rank) * scale
            )
        else:
            self.correlation = nn.Parameter(torch.randn(input_size, input_size))

        # Projections
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.output_size)

        # Normalization
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(input_size)
            self.layer_norm2 = nn.LayerNorm(self.hidden_size)
            self.layer_norm3 = nn.LayerNorm(self.output_size)

        # Dropout
        self.dropout = (
            nn.Dropout(correlation_dropout)
            if correlation_dropout > 0
            else nn.Identity()
        )

        # Chebyshev coefficients
        self.cheb_weights = nn.Parameter(torch.ones(self.cheb_k) / self.cheb_k)

        self.reset_parameters()

    def reset_parameters(self):
        if self.low_rank:
            nn.init.orthogonal_(self.corr_factors[0])
            nn.init.orthogonal_(self.corr_factors[1])
        else:
            nn.init.eye_(self.correlation)
            with torch.no_grad():
                self.correlation.data += 0.01 * torch.randn_like(self.correlation)
                self.correlation.data = 0.5 * (
                    self.correlation.data + self.correlation.data.t()
                )

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        nn.init.constant_(self.cheb_weights, 1.0 / self.cheb_k)

    def get_learned_correlation(self) -> torch.Tensor:
        if self.low_rank:
            U, V = self.corr_factors[0], self.corr_factors[1]
            corr = torch.matmul(U, V.T)
            corr = 0.5 * (corr + corr.T)
        else:
            corr = 0.5 * (self.correlation + self.correlation.T)

        corr = torch.tanh(corr)
        return self.dropout(corr) if self.training else corr

    def compute_data_correlation(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_reshaped = x_centered.transpose(1, 2)
        norms = torch.norm(x_reshaped, dim=2, keepdim=True).clamp(min=self.eps)
        x_normalized = x_reshaped / norms
        corr_batch = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
        return corr_batch.mean(dim=0).clamp(min=-1.0, max=1.0).detach()

    def compute_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        A = A.clone()
        A.fill_diagonal_(0.0)
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=self.eps), -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(A.size(0), device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        return L.clamp(min=-2.0, max=2.0)

    def chebyshev_filter(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        cheb_weights = F.softmax(self.cheb_weights, dim=0)

        Tx_0 = x
        if self.cheb_k == 1:
            return cheb_weights[0] * Tx_0

        Tx_1 = torch.matmul(x, L)
        out = cheb_weights[0] * Tx_0 + cheb_weights[1] * Tx_1

        for k in range(2, self.cheb_k):
            Tx_k = 2 * torch.matmul(Tx_1, L) - Tx_0
            Tx_k = Tx_k.clamp(min=-1e2, max=1e2)
            out += cheb_weights[k] * Tx_k
            Tx_0, Tx_1 = Tx_1, Tx_k

        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_layer_norm:
            x = self.layer_norm1(x)

        raw_data_corr = self.compute_data_correlation(x)
        learned_corr = self.get_learned_correlation()
        alpha = torch.sigmoid(self.alpha)
        mixed_corr = alpha * learned_corr + (1 - alpha) * raw_data_corr

        laplacian = self.compute_laplacian(mixed_corr)
        x_filtered = self.chebyshev_filter(x, laplacian)
        x_proj = self.input_proj(x_filtered)

        if self.use_layer_norm:
            x_proj = self.layer_norm2(x_proj)

        x_proj = F.gelu(x_proj)
        out = self.output_proj(x_proj)

        if self.use_layer_norm:
            out = self.layer_norm3(out)

        return out, mixed_corr


from flash_attn import flash_attn_qkvpacked_func


def round_to_supported_head_dim(dim):
    supported_dims = [16, 32, 64, 128]
    return min(supported_dims, key=lambda x: abs(x - dim))


class MessagePassing(nn.Module):
    """
    Generic message passing base class.
    Supports xformers-based multi-head attention over the feature dimension.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        aggregation: str = "sum",
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.num_heads = num_heads
        # self.head_dim = hidden_dim * num_heads

        # Your current calculation
        raw_head_dim = hidden_dim * num_heads
        self.head_dim = round_to_supported_head_dim(raw_head_dim)

        # Adjust num_heads to maintain similar total dimension
        # self.num_heads = max(1, (hidden_dim * num_heads) // self.head_dim)

        # Shared node transformation
        self.message_transform = nn.Linear(input_size, hidden_dim)

        if self.aggregation == "sage":
            self.sage_update = nn.Linear(input_size + hidden_dim, hidden_dim)

        if self.aggregation == "sage_lstm":
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
            )
            self.sage_update = nn.Linear(input_size + hidden_dim, hidden_dim)

        # Projections for attention mode
        if aggregation == "xformers":
            self.q_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)
            self.k_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)
            self.v_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)
            self.bias_proj = nn.Linear(input_size, num_heads * num_heads)
        elif aggregation == "flash":
            self.q_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)
            self.k_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)
            self.v_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)

    def _flash_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Use FlashAttention instead of xFormers.
        """
        B, T, D = messages.shape
        H = self.num_heads
        d_head = self.head_dim

        x_flat = messages.reshape(B * T, D)
        # [B, T, D] → [B, T, H, d_head] → [B, T, 3, H, d_head]
        q = self.q_proj(x_flat).reshape(B * T, H, D, d_head)
        k = self.k_proj(x_flat).reshape(B * T, H, D, d_head)
        v = self.v_proj(x_flat).reshape(B * T, H, D, d_head)

        # FlashAttention expects [B, T, 3, H, d_head]
        qkv = torch.stack([q, k, v], dim=2)

        # Apply FlashAttention (no masking for now)
        out = flash_attn_qkvpacked_func(qkv, causal=False)  # [B, T, H, d_head]
        out = out.permute(0, 2, 1, 3)  # [B*T, D_token, H, d_head]
        out = out.reshape(B, T, D, -1)
        out = out.mean(dim=-1)  # [B, T, D]
        # Flatten: [B, T, H, d_head] → [B, T, D]
        return out.reshape(B, T, D)

    def message(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply shared linear transformation to input features.
        """
        return self.message_transform(h)  # [B, T, hidden_dim]

    def aggregate(
        self,
        messages: torch.Tensor,
        graph: torch.Tensor,
        self_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Aggregate messages based on specified strategy.
        messages: [B, T, hidden_dim]
        graph: [F, F] for sum/mean, or attention bias for xformers
        """
        if self.aggregation == "sum":
            return torch.einsum("bth,hg->btg", messages, graph)

        elif self.aggregation == "mean":
            deg = graph.sum(dim=1) + 1e-10  # avoid div by zero
            norm_graph = graph * (1.0 / deg).unsqueeze(1)
            return torch.einsum("bth,hg->btg", messages, norm_graph)

        elif self.aggregation == "max":
            # Use max pooling over the graph
            return torch.einsum("bth,hg->btg", messages, graph).max(dim=1)[0]

        elif self.aggregation == "sage":
            assert self_features is not None, "SAGE requires self node features"

            # Graph aggregation (mean)
            deg = graph.sum(dim=1) + 1e-10
            norm_graph = graph * (1.0 / deg).unsqueeze(1)
            neighbor_agg = torch.einsum(
                "bth,hg->btg", messages, norm_graph
            )  # [B, T, H]

            # Concatenate self features and aggregated neighbor messages
            concat = torch.cat([self_features, neighbor_agg], dim=-1)  # [B, T, in + H]
            return self.sage_update(concat)  # [B, T, H]

        elif self.aggregation == "sage_lstm":
            assert self_features is not None
            neighbor_sequences = torch.einsum("bth,hg->btg", messages, graph).transpose(
                1, 2
            )
            lstm_out, _ = self.lstm(neighbor_sequences)
            neighbor_agg, _ = torch.max(lstm_out, dim=1)
            neighbor_agg = neighbor_agg.unsqueeze(1).expand(-1, messages.size(1), -1)
            concat = torch.cat([self_features, neighbor_agg], dim=-1)
            return self.sage_update(concat)

        elif self.aggregation == "xformers":
            return self._xformers_aggregate(messages, graph)  # [B, T, H]
        elif self.aggregation == "flash":
            return self._flash_aggregate(messages, graph)
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation}")

    def _xformers_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention over features per time step, with bias shape [B*T, 12, 4, 4].
        """
        B, T, D = messages.shape
        H = self.num_heads
        d_head = self.head_dim

        # Flatten [B, T, D] → [B*T, D]
        x_flat = messages.reshape(B * T, D)
        # print(f"Xformers aggregate: {x_flat.shape=}, {graph.shape=}")

        # Project Q/K/V to [B*T, H, D, d_head]
        q = self.q_proj(x_flat).reshape(B * T, H, D, d_head)
        k = self.k_proj(x_flat).reshape(B * T, H, D, d_head)
        v = self.v_proj(x_flat).reshape(B * T, H, D, d_head)

        # Call xFormers without bias for now
        # q = q.contiguous().half()  # or .bfloat16()
        # k = k.contiguous().half()
        # v = v.contiguous().half()
        # print(f"Xformers aggregate: {q.shape=}, {k.shape=}, {v.shape=}")
        out = memory_efficient_attention(q, k, v)  # [B*T, H, D, d_head]
        # out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        out = out.permute(0, 2, 1, 3)  # [B*T, D_token, H, d_head]
        out = out.reshape(B, T, D, -1)
        # take mean over last dim
        out = out.mean(dim=-1)  # [B, T, D]
        # print(f"Xformers aggregate: {out.shape=}")

        return out

    def forward(self, h: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement forward pass.")

    def update(self, x, agg):
        combined = torch.cat([x, agg], dim=-1)
        return self.norm(self.output_proj(self.update_fn(combined)))


class GraphConv(MessagePassing):
    def __init__(
        self, input_size, output_size, hidden_dim, aggregation="sum", dropout=0.1
    ):
        super().__init__(input_size, hidden_dim, aggregation)
        self.update_fn = nn.Sequential(
            nn.Linear(input_size + hidden_dim, input_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_proj = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, graph):
        msg = self.message(x)  # [B, T, hidden]
        agg = self.aggregate(msg, graph)  # [B, T, hidden]
        return self.update(x, agg)  # [B, T, output_size]


class SageLayer(GraphConv):
    def __init__(self, input_size, hidden_dim):
        super().__init__(input_size, input_size, hidden_dim, aggregation="sage")


class AttGraphConv(MessagePassing):
    def __init__(self, input_size, output_size, hidden_dim, num_heads=4, dropout=0.1):
        print(f"Using {num_heads} attention heads for AttGraphConv")
        super().__init__(input_size, hidden_dim)
        self.num_heads = num_heads
        self.output_proj = nn.Linear(input_size, output_size)
        self.attn_q = nn.Linear(input_size, hidden_dim)
        self.attn_k = nn.Linear(input_size, hidden_dim)
        self.update_fn = nn.Sequential(
            nn.Linear(input_size + hidden_dim, input_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(output_size)

    def compute_attention(self, x):
        # x: [B, T, F]
        q = self.attn_q(x).mean(dim=1)  # [B, hidden]
        k = self.attn_k(x).mean(dim=1)  # [B, hidden]
        scores = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)).squeeze(-1)  # [B, 1]
        alpha = torch.sigmoid(scores).squeeze(-1)  # [B]
        return alpha

    def forward(self, x, graph):
        attn_graph = torch.tanh(graph) * (graph.abs() > 1e-3).float()  # soft mask
        msg = self.message(x)
        agg = self.aggregate(msg, attn_graph)
        combined = torch.cat([x, agg], dim=-1)
        h = self.update_fn(combined)
        return self.norm(self.output_proj(h))


class XFormerAttGraphConv(MessagePassing):
    def __init__(self, input_size, output_size, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__(input_size, hidden_dim)
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(input_size, input_size * self.head_dim * num_heads)
        self.k_proj = nn.Linear(input_size, input_size * self.head_dim * num_heads)
        self.v_proj = nn.Linear(input_size, input_size * self.head_dim * num_heads)

        self.update_fn = nn.Sequential(
            nn.Linear(input_size + self.head_dim, input_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_proj = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F] (input time series)
        graph: [F, F] (correlation/adjacency matrix over features)
        """
        B, T, F = x.shape
        H, D = self.num_heads, self.head_dim

        # Apply graph as soft attention bias over features
        # x: [B, T, F] → [B, F, T] (features as "tokens")
        x_feat = x.reshape(B * T, F)

        # Project to Q, K, V
        q = self.q_proj(x_feat).reshape(B * T, H, F, D).transpose(1, 2)  # [B, H, F, D]
        k = self.k_proj(x_feat).reshape(B * T, H, F, D).transpose(1, 2)  # [B, H, F, D]
        v = self.v_proj(x_feat).reshape(B * T, H, F, D).transpose(1, 2)  # [B, H, F, D]

        # Use xformers efficient attention
        out = memory_efficient_attention(q, k, v)  # [B, H, F, D]
        out = out.permute(0, 2, 1, 3)  # [B*T, D_token, H, d_head]
        out = out.reshape(B, T, D, -1)
        # take mean over last dim
        out = out.mean(dim=-1)  # [B, T, D]
        # Update with residual information
        combined = torch.cat([x, out], dim=-1)  # [B, T, in + hidden]
        updated = self.update_fn(combined)  # [B, T, input_size]
        return self.norm(self.output_proj(updated))  # [B, T, output_size]


class LatentGraphNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Optional[int] = None,
        correlation_hidden_size: Optional[int] = None,
        low_rank: bool = True,
        rank: Optional[int] = None,
        num_passes: int = 1,
        aggregation: str = "sum",
        dropout: float = 0.1,
        residual: bool = True,
        strategy: Literal["vanilla", "attn", "xformers", "sage", "gtat"] = "vanilla",
        jk_mode: Literal["last", "sum", "max", "concat", "lstm", "none"] = "none",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size or max(input_size, output_size)
        self.num_passes = num_passes
        self.residual = residual

        # Latent correlation layer (data + learnable graph)
        self.correlation_layer = LatentCorrelationLayer(
            input_size=input_size,
            output_size=input_size,
            hidden_size=correlation_hidden_size,
            low_rank=low_rank,
            rank=rank,
            correlation_dropout=dropout,
        )

        # Message passing layers
        self.message_passing_layers = nn.ModuleList(
            [
                self._create_layer(
                    strategy, input_size, self.hidden_size, aggregation, dropout
                )
                for _ in range(num_passes)
            ]
        )

        self.jk_mode = jk_mode
        if jk_mode != "none":
            # Jump knowledge module
            self.jump_knowledge = JumpKnowledge(
                mode=jk_mode, hidden_size=self.input_size, output_size=input_size
            )

        if strategy == "gtat":
            self.gdv_encoder = GDVEncoder(gdv_dim=73, topo_dim=input_size)

        self.norm = nn.LayerNorm(output_size)

    def _create_layer(
        self,
        strategy: str,
        input_size: int,
        hidden_size: int,
        aggregation: str,
        dropout: float,
    ) -> nn.Module:
        if strategy == "vanilla":
            return GraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                aggregation=aggregation,
                dropout=dropout,
            )
        elif strategy == "attn":
            return AttGraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                dropout=dropout,
            )
        elif strategy == "xformers":
            return XFormerAttGraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=16,
                dropout=dropout,
            )
        elif strategy == "sage":
            return SageLayer(input_size=input_size, hidden_dim=hidden_size)
        elif strategy == "gtat":
            return GTATLayerWrapper(
                input_size,
                hidden_size,
                topo_dim=input_size,
                hidden_dim=hidden_size,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        corr_features, correlation = self.correlation_layer(x)

        h = corr_features
        outputs: List[torch.Tensor] = []

        topo_embedding = None
        if any(isinstance(l, GTATLayerWrapper) for l in self.message_passing_layers):
            B, T, F = x.shape
            gdv = compute_mock_gdv(F).to(x.device)
            gdv = gdv / (gdv.sum(dim=1, keepdim=True) + 1e-6)
            topo_embedding = self.gdv_encoder(gdv)  # [F, topo_dim]
            topo_embedding = (
                topo_embedding.unsqueeze(0).expand(B, F, -1).clone()
            )  # ✅ shape [B, F, topo_dim]

        for layer in self.message_passing_layers:
            if isinstance(layer, GTATLayerWrapper):
                h = layer(h, correlation, topo_embedding)
            else:
                h = layer(h, correlation)

        if self.jk_mode != "none":
            jk_out = self.jump_knowledge(outputs)

            if self.residual and x.shape[-1] == jk_out.shape[-1]:
                jk_out = jk_out + x
        else:
            jk_out = h
            if self.residual and x.shape[-1] == h.shape[-1]:
                jk_out = h + x

        return jk_out


class JumpKnowledge(nn.Module):
    def __init__(
        self,
        mode: Literal["last", "sum", "max", "concat", "lstm"] = "concat",
        hidden_size: int = None,
        output_size: int = None,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out_proj = None  # lazy initialization

        if self.mode == "lstm":
            assert hidden_size is not None
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.out_proj = (
                nn.Identity()
                if hidden_size == output_size
                else nn.Linear(hidden_size, output_size)
            )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        if self.mode == "last":
            return xs[-1]

        elif self.mode == "sum":
            return torch.stack(xs, dim=0).sum(dim=0)

        elif self.mode == "max":
            return torch.stack(xs, dim=0).max(dim=0)[0]

        elif self.mode == "concat":
            x_cat = torch.cat(xs, dim=-1)  # [B, T, D * num_layers]
            if self.out_proj is None:
                input_dim = x_cat.size(-1)
                self.out_proj = nn.Linear(input_dim, self.output_size).to(x_cat.device)
            return self.out_proj(x_cat)

        elif self.mode == "lstm":
            B, T, D = xs[0].shape
            x_seq = torch.stack(xs, dim=1).reshape(B * T, len(xs), D)
            lstm_out, _ = self.lstm(x_seq)
            final = lstm_out[:, -1, :].reshape(B, T, -1)
            return self.out_proj(final)

        else:
            raise ValueError(f"Unsupported JK mode: {self.mode}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import subprocess
import tempfile
import numpy as np
from typing import Union, Optional


# === GDV via ORCA ===
def compute_gdv_orca(
    G: Union[nx.Graph, nx.DiGraph], orca_path: str = "./orca", graphlet_size: int = 5
) -> np.ndarray:
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)

    with tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as edge_file, tempfile.NamedTemporaryFile("r", delete=False) as out_file:

        node_map = {n: i for i, n in enumerate(G.nodes())}
        for u, v in G.edges():
            edge_file.write(f"{node_map[u]} {node_map[v]}\n")
        edge_file.flush()

        cmd = [orca_path, str(graphlet_size), edge_file.name, out_file.name]
        subprocess.run(cmd, check=True)

        out_lines = out_file.readlines()
        gdv = [list(map(int, line.strip().split())) for line in out_lines]

    return np.array(gdv, dtype=np.float32)


def compute_mock_gdv(num_nodes: int, gdv_dim: int = 73) -> torch.Tensor:
    return torch.randn(num_nodes, gdv_dim)


class GDVEncoder(nn.Module):
    def __init__(self, gdv_dim: int, topo_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(gdv_dim, topo_dim), nn.ReLU(), nn.Linear(topo_dim, topo_dim)
        )

    def forward(self, gdv):
        return self.proj(gdv)


class GTATLayer(nn.Module):
    def __init__(self, feature_dim, topo_dim, hidden_dim):
        super().__init__()
        self.feature_attn = nn.Linear(2 * hidden_dim, 1)
        self.topo_attn = nn.Linear(2 * hidden_dim, 1)

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.topo_proj = nn.Linear(topo_dim, hidden_dim)

    def forward(self, H, T, adj):
        # H: [B, N, F], T: [B, F, F_t], adj: [F, F] or [B, F, F]
        B, N, Fdim = H.shape
        _, Fdim, F_t = T.shape

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)  # [B, F, F]

        # Project features
        H_proj = self.feature_proj(H)  # [B, N, H]
        T_proj = self.topo_proj(T)  # [B, F, H]

        Hi = T_proj.unsqueeze(2).expand(B, Fdim, Fdim, -1)
        Hj = T_proj.unsqueeze(1).expand(B, Fdim, Fdim, -1)
        topo_input = torch.cat([Hi, Hj], dim=-1)
        e_topo = F.leaky_relu(self.topo_attn(topo_input)).squeeze(-1)
        beta = F.softmax(e_topo.masked_fill(adj == 0, -1e4), dim=-1)  # [B, F, F]

        T_out = torch.bmm(beta, T_proj)  # [B, F, H]

        H_out = torch.zeros_like(H_proj)
        for b in range(B):
            for t in range(N):
                Hi = H_proj[b, t].unsqueeze(0).expand(Fdim, -1)
                Hj = T_out[b]
                feat_cat = torch.cat([Hi, Hj], dim=-1)  # [F, 2H]
                e_feat = F.leaky_relu(self.feature_attn(feat_cat)).squeeze(-1)  # [F]
                alpha = F.softmax(e_feat, dim=-1)  # [F]
                H_out[b, t] = torch.matmul(alpha, Hj)

        return H_out, T_out


class GTATLayerWrapper(nn.Module):
    def __init__(self, input_size, output_size, topo_dim, hidden_dim, dropout):
        super().__init__()
        self.gtat_layer = GTATLayer(input_size, topo_dim, hidden_dim)
        self.output_proj = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, graph, topo_embedding):
        h, t = self.gtat_layer(x, topo_embedding, graph)
        out = self.output_proj(h)
        return self.norm(self.dropout(out))


class GTAT(nn.Module):
    def __init__(self, in_dim, gdv_dim, topo_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.gdv_encoder = GDVEncoder(gdv_dim, topo_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GTATLayer(hidden_dim, topo_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, H, adj, gdv=None):
        if gdv is None:
            gdv = compute_mock_gdv(H.size(0))
        if isinstance(gdv, np.ndarray):
            gdv = torch.tensor(gdv, dtype=torch.float32, device=H.device)

        gdv = gdv / (gdv.sum(dim=1, keepdim=True) + 1e-6)
        T = self.gdv_encoder(gdv)
        H = self.input_proj(H)

        for layer in self.layers:
            H, T = layer(H, T, adj)

        return self.output_proj(H)


class GTATIntegrated(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        gdv_dim: int = 73,
        topo_dim: int = 64,
        hidden_size: Optional[int] = None,
        num_passes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size or max(input_size, output_size)
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, output_size)
        self.gdv_encoder = GDVEncoder(gdv_dim, topo_dim)
        self.layers = nn.ModuleList(
            [
                GTATLayerWrapper(
                    self.hidden_size,
                    self.hidden_size,
                    topo_dim,
                    self.hidden_size,
                    dropout,
                )
                for _ in range(num_passes)
            ]
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, gdv: Optional[torch.Tensor] = None
    ):
        if gdv is None:
            gdv = compute_mock_gdv(x.size(-1))  # x.shape: [B, T, F]
        if isinstance(gdv, np.ndarray):
            gdv = torch.tensor(gdv, dtype=torch.float32, device=x.device)

        gdv = gdv / (gdv.sum(dim=1, keepdim=True) + 1e-6)
        topo_embedding = self.gdv_encoder(gdv)
        if topo_embedding.ndim == 2:
            B, T, F = x.shape
            topo_embedding = (
                topo_embedding.unsqueeze(0).expand(B, T, -1).clone()
            )  # [B, F, topo_dim]
        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h, adj, topo_embedding)

        return self.norm(self.output_proj(h))
