import pytest
import torch

from foreblocks.config import TrainingConfig
from foreblocks.layers.graph.common import add_self_loops
from foreblocks.layers.graph.layers import GATConv, GCNConv, JumpKnowledge, SAGEConv
from foreblocks.models import GraphForecastingModel
from foreblocks.training.trainer import Trainer


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


def test_add_self_loops_batched_adj() -> None:
    adj = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    expected = adj + torch.eye(2, dtype=torch.float32).unsqueeze(0)
    torch.testing.assert_close(add_self_loops(adj), expected)


def test_jumpknowledge_attn_pooling_shape() -> None:
    xs = [torch.randn(2, 3, 4, 5) for _ in range(4)]
    jk = JumpKnowledge(mode="attn", output_size=7)
    out = jk(xs)
    assert out.shape == (2, 3, 4, 7)


def test_conv_residual_connection_matches_identity_when_zeroed() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 6)
    layer = GCNConv(
        6,
        6,
        bias=True,
        activation="none",
        dropout=0.0,
        use_graph_norm=False,
        norm_strategy="none",
        residual=True,
    )
    torch.nn.init.zeros_(layer.lin.weight)
    if layer.lin.bias is not None:
        torch.nn.init.zeros_(layer.lin.bias)
    if isinstance(layer.res_lin, torch.nn.Linear):
        torch.nn.init.zeros_(layer.res_lin.weight)
    adj = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(2, 4, 4)
    torch.testing.assert_close(layer(x, adj=adj), x)


def test_gcn_edge_attr_supports_dense_and_sparse() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 2, 4, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.tensor([0.7, 1.2, 0.5, 0.9], dtype=torch.float32)
    adj = _dense_adj_from_edge_index(edge_index, num_nodes=4, edge_weight=edge_attr)

    layer = GCNConv(
        6,
        8,
        activation="none",
        dropout=0.0,
        use_graph_norm=False,
        norm_strategy="none",
    )
    layer.eval()

    dense_out = layer(x, adj=adj)
    sparse_out = layer(x, edge_index=edge_index, edge_attr=edge_attr)

    torch.testing.assert_close(dense_out, sparse_out, atol=1e-6, rtol=1e-6)


def test_jumpknowledge_lstm_handles_input_dim_mismatch() -> None:
    xs = [torch.randn(2, 3, 4, 5) for _ in range(4)]
    jk = JumpKnowledge(mode="lstm", hidden_size=7, output_size=9)
    out = jk(xs)
    assert out.shape == (2, 3, 4, 9)


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


def test_gcn_sparse_matches_dense_weighted_use_sparse() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 6)
    edge_index = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 1]])
    edge_weight = torch.tensor([0.7, 1.2, 0.5, 0.9, 1.1], dtype=torch.float32)
    adj = _dense_adj_from_edge_index(edge_index, num_nodes=4, edge_weight=edge_weight)

    layer_dense = GCNConv(
        6,
        8,
        dropout=0.0,
        use_graph_norm=False,
        norm_strategy="none",
    )
    layer_sparse = GCNConv(
        6,
        8,
        dropout=0.0,
        use_graph_norm=False,
        norm_strategy="none",
        use_sparse=True,
    )
    layer_dense.eval()
    layer_sparse.eval()
    layer_sparse.load_state_dict(layer_dense.state_dict())

    dense_out = layer_dense(x, adj=adj)
    sparse_out = layer_sparse(x, edge_index=edge_index, edge_weight=edge_weight)

    torch.testing.assert_close(dense_out, sparse_out, atol=1e-6, rtol=1e-6)


def test_sage_fuse_linear_forward() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 2, 4, 5)
    adj = _dense_adj_from_edge_index(
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]), num_nodes=4
    )

    layer = SAGEConv(
        5,
        7,
        dropout=0.0,
        use_graph_norm=False,
        norm_strategy="none",
        fuse_linear=True,
        use_sparse=True,
    )
    layer.eval()

    out = layer(x, adj=adj)
    assert out.shape == (2, 2, 4, 7)


def test_gatv2_forward_shape() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 3, 4, 8)
    adj = torch.eye(4, dtype=torch.float32)

    layer = GATConv(
        8,
        8,
        heads=2,
        dropout=0.0,
        use_gatv2=True,
    )
    layer.eval()

    out = layer(x, adj=adj)
    assert out.shape == (2, 3, 4, 8)


def test_gat_sparse_matches_dense_weighted() -> None:
    torch.manual_seed(11)
    x = torch.randn(2, 3, 4, 8)
    edge_index = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 1]])
    edge_weight = torch.tensor([0.2, 1.5, 0.7, 2.0, 0.4], dtype=torch.float32)
    adj = _dense_adj_from_edge_index(edge_index, num_nodes=4, edge_weight=edge_weight)

    layer = GATConv(8, 8, heads=2, dropout=0.0)
    layer.eval()

    dense_out = layer(x, adj=adj)
    sparse_out = layer(x, edge_index=edge_index, edge_weight=edge_weight)

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
    model = GraphForecastingModel(
        num_nodes=4,
        feat_dim=6,
        out_feat_dim=6,
        passes=2,
        layer="gcn",
        norm_strategy="sandwich_norm",
    )
    y = model(x)
    assert y.shape == x.shape


def test_graph_forecasting_model_supports_mixed_convs_and_jk() -> None:
    torch.manual_seed(6)
    x = torch.randn(2, 5, 4, 3)
    model = GraphForecastingModel(
        num_nodes=4,
        feat_dim=3,
        out_feat_dim=5,
        conv=["gcn", "sage", "gat"],
        gat_heads=1,
        jk="concat",
    )

    y, adj = model(x, return_graph=True)

    assert y.shape == (2, 5, 4, 5)
    assert adj is not None
    assert adj.shape == (4, 4)


def test_graph_forecasting_model_static_graph_horizon() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 6, 4, 3)
    model = GraphForecastingModel(
        num_nodes=4,
        feat_dim=3,
        out_feat_dim=2,
        graph_source="static",
        static_adjacency=torch.eye(4),
        conv="gcn",
        seq_len=6,
        horizon=2,
    )

    y = model(x)

    assert y.shape == (2, 2, 4, 2)


def test_graph_forecasting_model_flatten_nodes_readout() -> None:
    torch.manual_seed(8)
    x = torch.randn(2, 5, 3, 2)
    model = GraphForecastingModel(
        num_nodes=3,
        feat_dim=2,
        out_feat_dim=4,
        passes=1,
        layer="gcn",
        output_mode="flatten_nodes",
    )

    y = model(x)

    assert y.shape == (2, 5, 12)
    assert model.input_size == 6
    assert model.output_size == 12


def test_trainer_accepts_graph_forecasting_dict_batches() -> None:
    torch.manual_seed(9)
    x = torch.randn(4, 5, 3, 2)
    y = torch.randn(4, 2, 3, 1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    model = GraphForecastingModel(
        num_nodes=3,
        feat_dim=2,
        out_feat_dim=1,
        graph_source="external",
        conv="gcn",
        seq_len=5,
        horizon=2,
    )
    trainer = Trainer(
        model,
        config=TrainingConfig(num_epochs=1, use_amp=False, batch_size=4),
        auto_track=False,
    )

    loss, components = trainer.train_epoch([{"X": x, "y": y, "edge_index": edge_index}])

    assert loss > 0
    assert "task_loss" in components


def test_trainer_graph_metrics_and_plot_prediction() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    torch.manual_seed(10)
    x = torch.randn(5, 6, 3, 2)
    y = torch.randn(5, 2, 3, 2)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    model = GraphForecastingModel(
        num_nodes=3,
        feat_dim=2,
        out_feat_dim=2,
        graph_source="external",
        conv="gcn",
        seq_len=6,
        horizon=2,
    )
    trainer = Trainer(
        model,
        config=TrainingConfig(num_epochs=1, use_amp=False, batch_size=5),
        auto_track=False,
    )
    graph_kwargs = {"edge_index": edge_index}

    metrics = trainer.metrics(x, y, batch_size=2, graph_kwargs=graph_kwargs)
    fig = trainer.plot_prediction(
        x,
        y,
        graph_kwargs=graph_kwargs,
        show=False,
        names=["n0", "n1", "n2"],
    )

    assert {"mse", "rmse", "mae", "mape"} <= set(metrics)
    assert fig is not None
    assert len(fig.axes) == 6
    assert fig.axes[0].get_title().startswith("node 0 feature 0")
    assert fig.axes[-1].get_title().startswith("node 2 feature 1")
