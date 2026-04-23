from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

from foreblocks.layers.graph.common import Tensor
from foreblocks.layers.graph.common import ensure_adj
from foreblocks.layers.graph.common import xavier_zero_bias
from foreblocks.layers.graph.latent import CorrelationConfig
from foreblocks.layers.graph.latent import LatentCorrelationLearner
from foreblocks.layers.graph.layers import EdgeCondGCN
from foreblocks.layers.graph.layers import GATConv
from foreblocks.layers.graph.layers import GCNConv
from foreblocks.layers.graph.layers import JumpKnowledge
from foreblocks.layers.graph.layers import SAGEConv
from foreblocks.layers.graph.layers import StochasticDepth
from foreblocks.layers.graph.mtgnn import GraphWaveNetBlock
from foreblocks.layers.graph.norms import make_norm_pair


GraphConvType = Literal["gcn", "sage", "gat", "edge_cond", "graph_wavenet"]
GraphSource = Literal["latent", "external", "static"]
JumpKnowledgeMode = Literal["none", "last", "sum", "max", "concat", "lstm"]
GraphOutputMode = Literal["sequence", "last", "mean", "flatten_nodes"]


class GraphForecastingModel(nn.Module):
    """
    Generic graph forecasting/modeling stack over [B, T, N, F] tensors.

    The model owns graph construction, graph convolution blocks, optional jumping
    knowledge, and output readout. By default it learns a latent graph, runs
    graph convolutions, and returns a [B, T, N, F_out] sequence.
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int,
        out_feat_dim: int | None = None,
        *,
        hidden_size: int | None = None,
        passes: int | None = None,
        num_layers: int | None = None,
        layer: GraphConvType | Sequence[GraphConvType] = "edge_cond",
        conv: GraphConvType | Sequence[GraphConvType] | None = None,
        gat_heads: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.1,
        jk: JumpKnowledgeMode = "none",
        corr_cfg: CorrelationConfig | None = None,
        graph_source: GraphSource = "latent",
        static_adjacency: Tensor | None = None,
        residual: bool = True,
        pre_norm: bool | None = None,
        norm_strategy: str | None = None,
        seq_len: int | None = None,
        horizon: int | None = None,
        output_mode: GraphOutputMode = "sequence",
        flatten: Literal["nodes", "none"] | None = None,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.feat_dim = int(feat_dim)
        self.hidden_size = int(hidden_size or feat_dim)
        self.out_feat_dim = int(out_feat_dim or feat_dim)
        self.graph_source = graph_source
        self.output_mode = "flatten_nodes" if flatten == "nodes" else output_mode
        self.seq_len = seq_len
        self.horizon = horizon
        self.last_adj: Tensor | None = None

        if graph_source not in ("latent", "external", "static"):
            raise ValueError(
                "graph_source must be one of {'latent','external','static'}"
            )
        if self.output_mode not in ("sequence", "last", "mean", "flatten_nodes"):
            raise ValueError(
                "output_mode must be one of {'sequence','last','mean','flatten_nodes'}"
            )
        if horizon is not None and seq_len is None:
            raise ValueError("seq_len is required when horizon is set.")

        self.register_buffer(
            "static_adjacency",
            static_adjacency.detach().clone() if static_adjacency is not None else None,
            persistent=False,
        )

        if graph_source == "static" and static_adjacency is None:
            raise ValueError("static_adjacency is required when graph_source='static'.")

        if graph_source == "latent":
            cfg = corr_cfg or CorrelationConfig(
                num_nodes=num_nodes,
                feat_dim=feat_dim,
                out_feat_dim=self.hidden_size,
                cheb_k=3,
                low_rank=True,
                rank=max(1, num_nodes // 8),
                use_graph_norm=True,
                multi_scale=True,
                num_scales=3,
                use_ema=True,
                adaptive_sparse=True,
                sparsity_ratio=0.3,
                dropout_graph=dropout * 0.5,
            )
            self.corr: LatentCorrelationLearner | None = LatentCorrelationLearner(cfg)
            corr_out = cfg.out_feat_dim or cfg.feat_dim
            if corr_out != self.hidden_size:
                self.input_proj = nn.Linear(corr_out, self.hidden_size)
                xavier_zero_bias(self.input_proj)
            else:
                self.input_proj = nn.Identity()
        else:
            self.corr = None
            if feat_dim != self.hidden_size:
                self.input_proj = nn.Linear(feat_dim, self.hidden_size)
                xavier_zero_bias(self.input_proj)
            else:
                self.input_proj = nn.Identity()

        conv_spec = conv if conv is not None else layer
        layer_count = num_layers if num_layers is not None else passes
        layer_count = 2 if layer_count is None else int(layer_count)
        if layer_count < 1:
            raise ValueError("GraphForecastingModel requires at least one graph layer.")
        self.conv_types = self._expand_conv_types(conv_spec, layer_count)
        if not self.conv_types:
            raise ValueError("GraphForecastingModel requires at least one graph layer.")
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv": self._build_block_conv(
                            conv_type,
                            idx=idx,
                            gat_heads=gat_heads,
                            dropout=dropout,
                            pre_norm=pre_norm,
                            norm_strategy=norm_strategy,
                        ),
                        "sd": StochasticDepth(
                            stochastic_depth * idx / len(self.conv_types)
                        )
                        if stochastic_depth > 0
                        else nn.Identity(),
                    }
                )
                for idx, conv_type in enumerate(self.conv_types)
            ]
        )

        self.jk_on = jk != "none"
        if self.jk_on:
            self.jk = JumpKnowledge(
                mode=jk,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                num_layers_hint=len(self.conv_types) if jk == "concat" else None,
            )

        _, out_pre_norm, out_post_norm = make_norm_pair(
            self.hidden_size,
            self.out_feat_dim,
            use_graph_norm=True,
            norm_strategy=norm_strategy,
            pre_norm=pre_norm,
        )
        out_linear = nn.Linear(self.hidden_size, self.out_feat_dim)
        xavier_zero_bias(out_linear)
        self.out = nn.Sequential(out_pre_norm, out_linear, out_post_norm)

        if horizon is not None:
            self.time_head = nn.Linear(seq_len, horizon)
            xavier_zero_bias(self.time_head)
        else:
            self.time_head = nn.Identity()

        self.residual = residual and self.feat_dim == self.out_feat_dim
        if self.output_mode == "flatten_nodes":
            self.input_size = num_nodes * feat_dim
            self.output_size = num_nodes * self.out_feat_dim
        else:
            self.input_size = None
            self.output_size = None

    @staticmethod
    def _expand_conv_types(
        conv: GraphConvType | Sequence[GraphConvType],
        num_layers: int,
    ) -> list[GraphConvType]:
        if isinstance(conv, str):
            convs = [conv for _ in range(num_layers)]
        else:
            convs = list(conv)
            if not convs:
                raise ValueError("conv sequence must contain at least one layer type.")
        valid = {"gcn", "sage", "gat", "edge_cond", "graph_wavenet"}
        unknown = set(convs) - valid
        if unknown:
            raise ValueError(
                f"Unsupported graph convolution type(s): {sorted(unknown)}"
            )
        return convs

    def _build_block_conv(
        self,
        conv_type: GraphConvType,
        *,
        idx: int,
        gat_heads: int,
        dropout: float,
        pre_norm: bool | None,
        norm_strategy: str | None,
    ) -> nn.Module:
        if conv_type == "graph_wavenet":
            return GraphWaveNetBlock(
                num_nodes=self.num_nodes,
                channels=self.hidden_size,
                conv_channels=self.hidden_size,
                gcn_depth=2,
                prop_alpha=0.05,
                dropout=dropout,
                dilation_factor=2**idx,
                preserve_length=True,
                use_adaptive_graph=self.graph_source != "external",
                use_graph_norm=True,
            )

        in_channels = self.hidden_size
        out_channels = self.hidden_size
        common = {
            "dropout": dropout,
            "pre_norm": pre_norm,
            "norm_strategy": norm_strategy,
        }
        if conv_type == "edge_cond":
            return EdgeCondGCN(in_channels, out_channels, **common)
        if conv_type == "gcn":
            return GCNConv(in_channels, out_channels, **common)
        if conv_type == "sage":
            return SAGEConv(in_channels, out_channels, **common)
        if conv_type == "gat":
            return GATConv(in_channels, out_channels, heads=gat_heads, **common)
        raise ValueError(f"Unsupported graph convolution type: {conv_type}")

    def set_static_adjacency(self, adj: Tensor | None) -> "GraphForecastingModel":
        self.static_adjacency = None if adj is None else adj.detach().clone()
        return self

    def _resolve_graph(
        self,
        x: Tensor,
        adj: Tensor | None,
        edge_index: Tensor | None,
        edge_weight: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if adj is not None or edge_index is not None:
            return self.input_proj(x), adj

        if self.graph_source == "latent":
            if self.corr is None:
                raise RuntimeError("Latent graph learner is not initialized.")
            h, learned_adj = self.corr(x)
            return self.input_proj(h), learned_adj

        if self.graph_source == "static":
            if self.static_adjacency is None:
                raise ValueError("No static adjacency has been set.")
            return self.input_proj(x), self.static_adjacency.to(
                device=x.device,
                dtype=x.dtype,
            )

        raise ValueError(
            "GraphForecastingModel with graph_source='external' requires adj or edge_index."
        )

    def _dense_adj_if_needed(
        self,
        x: Tensor,
        adj: Tensor | None,
        edge_index: Tensor | None,
        edge_weight: Tensor | None,
    ) -> Tensor:
        return ensure_adj(
            adj=adj,
            edge_index=edge_index,
            num_nodes=x.size(2),
            edge_weight=edge_weight,
            batch_size=x.size(0) if adj is None and edge_index is not None else None,
            dtype=x.dtype,
            device=x.device,
        )

    def _apply_conv(
        self,
        conv: nn.Module,
        x: Tensor,
        adj: Tensor | None,
        edge_index: Tensor | None,
        edge_weight: Tensor | None,
    ) -> Tensor:
        if isinstance(conv, EdgeCondGCN):
            dense_adj = self._dense_adj_if_needed(x, adj, edge_index, edge_weight)
            return conv(x, dense_adj)
        if isinstance(conv, GraphWaveNetBlock):
            y, _ = conv(x, adj=adj, edge_index=edge_index, edge_weight=edge_weight)
            return y
        return conv(x, adj=adj, edge_index=edge_index, edge_weight=edge_weight)

    def _apply_time_head(self, x: Tensor) -> Tensor:
        if self.horizon is None:
            return x
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {x.size(1)}."
            )
        x = x.permute(0, 2, 3, 1)
        x = self.time_head(x)
        return x.permute(0, 3, 1, 2).contiguous()

    def _readout(self, x: Tensor) -> Tensor:
        if self.output_mode == "sequence":
            return x
        if self.output_mode == "last":
            return x[:, -1]
        if self.output_mode == "mean":
            return x.mean(dim=1)
        if self.output_mode == "flatten_nodes":
            batch_size, steps, num_nodes, feat_dim = x.shape
            return x.reshape(batch_size, steps, num_nodes * feat_dim)
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")

    def forward(
        self,
        x: Tensor,
        adj: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        return_graph: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        if x.dim() != 4:
            raise ValueError(
                f"Expected x with shape [B, T, N, F], got {tuple(x.shape)}"
            )
        if x.size(2) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.size(2)}.")
        if x.size(3) != self.feat_dim:
            raise ValueError(f"Expected {self.feat_dim} features, got {x.size(3)}.")

        h, graph_adj = self._resolve_graph(x, adj, edge_index, edge_weight)
        conv_adj = graph_adj if graph_adj is not None else adj
        outs: list[Tensor] = []

        for block in self.blocks:
            h_new = self._apply_conv(
                block["conv"],
                h,
                conv_adj,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            if isinstance(block["sd"], StochasticDepth):
                h = block["sd"](h_new, h)
            else:
                h = h_new
            outs.append(h)

        if self.jk_on:
            h = self.jk(outs)

        y = self.out(h)
        if self.residual and y.shape == x.shape:
            y = y + x
        y = self._apply_time_head(y)

        with torch.no_grad():
            self.last_adj = None if conv_adj is None else conv_adj.detach().cpu()

        out = self._readout(y)
        if return_graph:
            return out, conv_adj
        return out


__all__ = [
    "GraphConvType",
    "GraphForecastingModel",
    "GraphOutputMode",
    "GraphSource",
    "JumpKnowledgeMode",
]
