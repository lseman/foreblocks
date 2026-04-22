import torch

from foreblocks.blocks.graph_btnf.layers import GATConv, GCNConv, SAGEConv
from foreblocks.blocks.graph_btnf.network import LatentGraphNetwork


def _dense_adj_from_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    src, dst = edge_index
    if edge_weight is None:
        adj[dst, src] = 1.0
    else:
        adj[dst, src] = edge_weight
    return adj


def test_gcn_sparse_matches_dense_weighted() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 6)
    edge_index = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 1]])
    edge_weight = torch.tensor([0.7, 1.2, 0.5, 0.9, 1.1], dtype=torch.float32)
    adj = _dense_adj_from_edge_index(edge_index, num_nodes=4, edge_weight=edge_weight)

    layer = GCNConv(6, 8, dropout=0.0)
    layer.eval()

    dense_out = layer(x, adj=adj)
    sparse_out = layer(x, edge_index=edge_index, edge_weight=edge_weight)

    torch.testing.assert_close(dense_out, sparse_out, atol=1e-6, rtol=1e-6)


def test_sage_sparse_matches_dense_weighted() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 2, 4, 5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_weight = torch.tensor([1.0, 0.5, 1.5, 0.8], dtype=torch.float32)
    adj = _dense_adj_from_edge_index(edge_index, num_nodes=4, edge_weight=edge_weight)

    layer = SAGEConv(5, 7, dropout=0.0)
    layer.eval()

    dense_out = layer(x, adj=adj)
    sparse_out = layer(x, edge_index=edge_index, edge_weight=edge_weight)

    torch.testing.assert_close(dense_out, sparse_out, atol=1e-6, rtol=1e-6)


def test_gat_sparse_matches_dense_mask() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 3, 4, 8)
    edge_index = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 1]])
    adj = _dense_adj_from_edge_index(edge_index, num_nodes=4)

    layer = GATConv(8, 8, heads=2, dropout=0.0)
    layer.eval()

    dense_out = layer(x, adj=adj)
    sparse_out = layer(x, edge_index=edge_index)

    torch.testing.assert_close(dense_out, sparse_out, atol=1e-6, rtol=1e-6)


def test_gcn_norm_strategy_matches_legacy_pre_norm_flag() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 2, 4, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    torch.manual_seed(4)
    legacy = GCNConv(6, 8, pre_norm=True)
    torch.manual_seed(4)
    modern = GCNConv(6, 8, norm_strategy="pre_norm")
    legacy.eval()
    modern.eval()

    torch.testing.assert_close(
        legacy(x, edge_index=edge_index),
        modern(x, edge_index=edge_index),
        atol=1e-6,
        rtol=1e-6,
    )


def test_network_accepts_sandwich_norm_strategy() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 3, 4, 6)
    model = LatentGraphNetwork(
        num_nodes=4,
        feat_dim=6,
        out_feat_dim=6,
        passes=2,
        layer="gcn",
        norm_strategy="sandwich_norm",
    )
    y = model(x)
    assert y.shape == x.shape
