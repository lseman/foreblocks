import torch

from foreblocks.layers.graph.tggc import (
    GraphGegenbauerConv,
    LatentCorrelationLayer,
    TGGCBlock,
    TGGCModern,
    TGGCModernConfig,
    TemporalSpectralFilter,
    normalized_laplacian_from_adjacency,
    symmetric_normalize_adjacency,
)


def test_tggc_symmetric_normalize_adjacency_keeps_shape() -> None:
    adj = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    norm_adj = symmetric_normalize_adjacency(adj)
    lap = normalized_laplacian_from_adjacency(adj)

    assert norm_adj.shape == adj.shape
    assert lap.shape == adj.shape
    torch.testing.assert_close(norm_adj, norm_adj.transpose(0, 1), atol=1e-6, rtol=1e-6)


def test_tggc_latent_correlation_returns_symmetric_graph() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 12, 5, 3)
    layer = LatentCorrelationLayer(
        in_channels=3,
        hidden_size=7,
        attention_dropout=0.0,
        symmetric=True,
    )
    adj = layer(x)

    assert adj.shape == (5, 5)
    torch.testing.assert_close(adj, adj.transpose(0, 1), atol=1e-6, rtol=1e-6)


def test_tggc_temporal_spectral_filter_preserves_shape() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 16, 4, 6)
    layer = TemporalSpectralFilter(
        seq_len=16,
        channels=6,
        num_modes=5,
        mode_select_method="lowest",
        per_channel=True,
    )
    y = layer(x)

    assert y.shape == x.shape


def test_tggc_graph_gegenbauer_conv_accepts_static_and_batched_adjacency() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 10, 4, 5)
    adj = torch.rand(4, 4)
    layer = GraphGegenbauerConv(in_channels=5, out_channels=7, order=3, alpha=1.0)

    y = layer(x, symmetric_normalize_adjacency(adj))
    assert y.shape == (3, 10, 4, 7)

    batched_adj = symmetric_normalize_adjacency(torch.rand(3, 4, 4))
    y_batched = layer(x, batched_adj)
    assert y_batched.shape == (3, 10, 4, 7)


def test_tggc_block_returns_expected_shape() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 24, 6, 8)
    adj = symmetric_normalize_adjacency(torch.rand(6, 6))
    block = TGGCBlock(
        seq_len=24,
        in_channels=8,
        hidden_channels=8,
        order=2,
        coarse_modes=6,
        fine_modes=6,
        use_fine_filter=True,
        dropout=0.0,
    )

    y = block(x, adj)
    assert y.shape == x.shape


def test_tggc_modern_returns_forecast_and_learned_graph() -> None:
    torch.manual_seed(4)
    x = torch.randn(2, 18, 5, 4)
    cfg = TGGCModernConfig(
        num_nodes=5,
        seq_len=18,
        horizon=7,
        in_channels=4,
        out_channels=2,
        hidden_channels=12,
        num_blocks=2,
        coarse_modes=5,
        fine_modes=5,
        graph_hidden_size=10,
        block_dropout=0.0,
    )
    model = TGGCModern(cfg)
    y_hat, adj = model(x)

    assert y_hat.shape == (2, 7, 5, 2)
    assert adj.shape == (5, 5)
