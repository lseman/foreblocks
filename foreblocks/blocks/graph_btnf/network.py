from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .common import Tensor, xavier_zero_bias
from .latent import CorrelationConfig, LatentCorrelationLearner
from .layers import (
    EdgeCondGCN,
    GATConv,
    GCNConv,
    JumpKnowledge,
    SAGEConv,
    StochasticDepth,
)
from .norms import make_norm_pair


class LatentGraphNetwork(nn.Module):
    """
    End-to-end graph network over [B, T, N, F] tensors.
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int,
        out_feat_dim: int,
        passes: int = 2,
        layer: Literal["gcn", "sage", "gat", "edge_cond"] = "edge_cond",
        gat_heads: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.1,
        jk: Literal["none", "last", "sum", "max", "concat", "lstm"] = "none",
        corr_cfg: CorrelationConfig | None = None,
        residual: bool = True,
        pre_norm: bool | None = None,
        norm_strategy: str | None = None,
    ):
        super().__init__()
        self.residual = residual and (feat_dim == out_feat_dim)
        self.jk_on = jk != "none"

        self.corr = LatentCorrelationLearner(
            corr_cfg
            or CorrelationConfig(
                num_nodes=num_nodes,
                feat_dim=feat_dim,
                out_feat_dim=feat_dim,
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
        )

        blocks: list[nn.Module] = []
        sd_rates = [stochastic_depth * i / passes for i in range(passes)]
        for sd_rate in sd_rates:
            if layer == "edge_cond":
                conv = EdgeCondGCN(
                    feat_dim,
                    feat_dim,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    norm_strategy=norm_strategy,
                )
            elif layer == "gcn":
                conv = GCNConv(
                    feat_dim,
                    feat_dim,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    norm_strategy=norm_strategy,
                )
            elif layer == "sage":
                conv = SAGEConv(
                    feat_dim,
                    feat_dim,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    norm_strategy=norm_strategy,
                )
            elif layer == "gat":
                conv = GATConv(
                    feat_dim,
                    feat_dim,
                    heads=gat_heads,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    norm_strategy=norm_strategy,
                )
            else:
                raise ValueError(f"Unsupported layer type: {layer}")

            blocks.append(
                nn.ModuleDict(
                    {
                        "conv": conv,
                        "sd": StochasticDepth(sd_rate)
                        if sd_rate > 0
                        else nn.Identity(),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)
        _, out_pre_norm, out_post_norm = make_norm_pair(
            feat_dim,
            out_feat_dim,
            use_graph_norm=True,
            norm_strategy=norm_strategy,
            pre_norm=pre_norm,
        )
        out_linear = nn.Linear(feat_dim, out_feat_dim)
        xavier_zero_bias(out_linear)
        self.out = nn.Sequential(out_pre_norm, out_linear, out_post_norm)

        if self.jk_on:
            self.jk = JumpKnowledge(
                mode=jk,
                hidden_size=feat_dim,
                output_size=feat_dim,
                num_layers_hint=passes if jk == "concat" else None,
            )

    def forward(self, x: Tensor) -> Tensor:
        h, A = self.corr(x)
        outs: list[Tensor] = []

        for block in self.blocks:
            h_new = block["conv"](h, adj=A)
            if isinstance(block["sd"], StochasticDepth):
                h = block["sd"](h_new, h)
            else:
                h = h_new
            outs.append(h)

        if self.jk_on:
            h = self.jk(outs)
        if self.residual and h.shape[-1] == x.shape[-1]:
            h = h + x
        return self.out(h)


class GraphPreprocessor(nn.Module):
    """
    Plug-and-play graph preprocessor over [B, T, N, F] inputs.
    """

    def __init__(
        self,
        num_nodes: int,
        in_feat_dim: int,
        out_feat_dim: int | None = None,
        passes: int = 2,
        layer: Literal["gcn", "sage", "gat", "edge_cond"] = "edge_cond",
        gat_heads: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.1,
        jk: Literal["none", "last", "sum", "max", "concat", "lstm"] = "none",
        flatten: Literal["nodes", "none"] = "nodes",
        residual: bool = True,
        pre_norm: bool | None = None,
        norm_strategy: str | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim or in_feat_dim
        self.flatten = flatten

        self.graph = LatentGraphNetwork(
            num_nodes=num_nodes,
            feat_dim=in_feat_dim,
            out_feat_dim=self.out_feat_dim,
            passes=passes,
            layer=layer,
            gat_heads=gat_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            jk=jk,
            residual=residual,
            pre_norm=pre_norm,
            norm_strategy=norm_strategy,
        )

        if self.flatten == "nodes":
            self.input_size = num_nodes * in_feat_dim
            self.output_size = num_nodes * self.out_feat_dim
        else:
            self.input_size = None
            self.output_size = None

        self.last_adj: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_full = self.graph(x)

        with torch.no_grad():
            _, A = self.graph.corr(x)
            self.last_adj = A.detach().cpu()

        if self.flatten == "nodes":
            B, T, N, Fout = y_full.shape
            return y_full.reshape(B, T, N * Fout)
        return y_full


__all__ = ["GraphPreprocessor", "LatentGraphNetwork"]
