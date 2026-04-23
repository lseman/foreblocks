import torch

from foreblocks.models import GraphForecastingModel
from foreblocks.layers.graph.mtgnn import (
    GraphWaveNetBlock,
    MTGNNBlock,
    MTGNNDilatedInception,
    MTGNNGraphConstructor,
    MTGNNMixProp,
)


def test_mtgnn_graph_constructor_respects_topk() -> None:
    torch.manual_seed(0)
    constructor = MTGNNGraphConstructor(num_nodes=6, k=2, embed_dim=8, alpha=3.0)
    adj = constructor()

    assert adj.shape == (6, 6)
    assert torch.all(adj >= 0)
    assert int((adj > 0).sum(dim=-1).max().item()) <= 2


def test_mtgnn_graph_constructor_full_adjacency_is_dense() -> None:
    torch.manual_seed(1)
    constructor = MTGNNGraphConstructor(num_nodes=5, k=2, embed_dim=4, alpha=2.0)
    full_adj = constructor.full_adjacency()
    sparse_adj = constructor()

    assert full_adj.shape == sparse_adj.shape == (5, 5)
    assert torch.count_nonzero(full_adj) >= torch.count_nonzero(sparse_adj)


def test_mtgnn_dilated_inception_preserves_length_when_requested() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 12, 4, 5)
    layer = MTGNNDilatedInception(
        in_channels=5,
        out_channels=9,
        dilation_factor=2,
        preserve_length=True,
    )
    y = layer(x)

    assert y.shape == (3, 12, 4, 9)


def test_mtgnn_dilated_inception_matches_original_temporal_shrink_when_unpadded() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 20, 3, 4)
    layer = MTGNNDilatedInception(
        in_channels=4,
        out_channels=8,
        dilation_factor=2,
        kernel_set=(2, 3, 6, 7),
        preserve_length=False,
    )
    y = layer(x)

    expected_t = 20 - 2 * (7 - 1)
    assert y.shape == (2, expected_t, 3, 8)


def test_mtgnn_mixprop_accepts_batched_adjacency() -> None:
    torch.manual_seed(4)
    x = torch.randn(2, 10, 4, 6)
    adj = torch.rand(2, 4, 4)
    layer = MTGNNMixProp(in_channels=6, out_channels=7, gdep=2, alpha=0.1)

    y = layer(x, adj)

    assert y.shape == (2, 10, 4, 7)


def test_mtgnn_block_returns_residual_and_skip_shapes() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 16, 5, 8)
    adj = torch.rand(5, 5)
    block = MTGNNBlock(
        channels=8,
        conv_channels=12,
        skip_channels=6,
        gcn_depth=2,
        prop_alpha=0.05,
        dropout=0.0,
        dilation_factor=2,
        preserve_length=True,
        use_graph_conv=True,
    )

    y, skip = block(x, adj)

    assert y.shape == x.shape
    assert skip is not None
    assert skip.shape == (2, 16, 5, 6)


def test_graph_wavenet_block_uses_adaptive_graph_when_not_provided() -> None:
    torch.manual_seed(6)
    x = torch.randn(2, 12, 5, 8)
    block = GraphWaveNetBlock(
        num_nodes=5,
        channels=8,
        conv_channels=10,
        skip_channels=4,
        dropout=0.0,
    )

    y, skip = block(x)

    assert y.shape == x.shape
    assert skip is not None
    assert skip.shape == (2, 12, 5, 4)


def test_graph_wavenet_block_accepts_external_sparse_graph() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 10, 4, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_weight = torch.tensor([1.0, 0.5, 1.5, 0.8], dtype=torch.float32)
    block = GraphWaveNetBlock(
        num_nodes=4,
        channels=6,
        conv_channels=6,
        dropout=0.0,
        use_adaptive_graph=False,
    )

    y, skip = block(x, edge_index=edge_index, edge_weight=edge_weight)

    assert y.shape == x.shape
    assert skip is None


def test_graph_forecasting_model_supports_graph_wavenet_conv() -> None:
    torch.manual_seed(8)
    x = torch.randn(2, 8, 4, 3)
    model = GraphForecastingModel(
        num_nodes=4,
        feat_dim=3,
        out_feat_dim=2,
        conv="graph_wavenet",
        graph_source="static",
        static_adjacency=torch.eye(4),
        hidden_size=8,
        seq_len=8,
        horizon=3,
    )

    y = model(x)

    assert y.shape == (2, 3, 4, 2)
