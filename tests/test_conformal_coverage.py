import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

from foreblocks.config import TrainingConfig
from foreblocks.core.conformal import ConformalPredictionEngine
from foreblocks.training.trainer import Trainer


class ZeroForecastModel(torch.nn.Module):
    def __init__(self, horizon: int, output_dim: int = 1):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch = x.shape[0]
        if self.output_dim == 1:
            return self.bias.expand(batch, self.horizon)
        return self.bias.expand(batch, self.horizon, self.output_dim)


def test_conformal_engine_reports_elementwise_and_joint_coverage():
    model = ZeroForecastModel(horizon=2)
    engine = ConformalPredictionEngine(method="split", quantile=0.5)

    X_cal = np.zeros((4, 3), dtype=np.float32)
    y_cal = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )
    engine.calibrate(model, X_cal, y_cal, batch_size=8)

    X_test = np.zeros((2, 3), dtype=np.float32)
    y_test = np.array(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=np.float32,
    )

    preds, lower, upper = engine.predict(model, X_test, batch_size=8)
    stats = engine.compute_coverage(model, X_test, y_test, batch_size=8)

    assert preds.shape == (2, 2, 1)
    assert lower.shape == (2, 2, 1)
    assert upper.shape == (2, 2, 1)
    assert stats["coverage"] == 0.5
    assert stats["joint_coverage"] == 0.0
    assert stats["coverage_gap"] == 0.0
    assert stats["joint_coverage_gap"] == -0.5


def test_split_conformal_uses_exact_higher_quantile():
    model = ZeroForecastModel(horizon=1)
    engine = ConformalPredictionEngine(method="split", quantile=0.5)

    X_cal = np.zeros((5, 3), dtype=np.float32)
    y_cal = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    engine.calibrate(model, X_cal, y_cal, batch_size=8)

    np.testing.assert_allclose(engine.radii, np.array([[2.0]], dtype=np.float32))


def test_rolling_conformal_uses_configured_initial_alpha():
    engine = ConformalPredictionEngine(
        method="rolling",
        quantile=0.9,
        rolling_alpha=0.3,
        aci_gamma=0.01,
    )
    assert engine.alpha == 0.3

    model = ZeroForecastModel(horizon=1)
    X_cal = np.zeros((4, 3), dtype=np.float32)
    y_cal = np.ones((4, 1), dtype=np.float32)
    engine.calibrate(model, X_cal, y_cal, batch_size=8)

    assert engine.alpha == 0.3
    engine.reset()
    assert engine.alpha == 0.3


def test_batch_rolling_update_matches_window_level_miss_definition():
    model = ZeroForecastModel(horizon=2)
    X_cal = np.zeros((8, 3), dtype=np.float32)
    y_cal = np.ones((8, 2), dtype=np.float32)
    X_update = np.zeros((2, 3), dtype=np.float32)
    y_update = np.array([[0.0, 10.0], [10.0, 0.0]], dtype=np.float32)

    seq_engine = ConformalPredictionEngine(method="rolling", quantile=0.9, aci_gamma=0.1)
    seq_engine.calibrate(model, X_cal, y_cal, batch_size=8)
    seq_engine.update(model, X_update, y_update, batch_size=8, sequential=True)

    batch_engine = ConformalPredictionEngine(method="rolling", quantile=0.9, aci_gamma=0.1)
    batch_engine.calibrate(model, X_cal, y_cal, batch_size=8)
    batch_engine.update(model, X_update, y_update, batch_size=8, sequential=False)

    assert seq_engine.alpha == 0.01
    assert batch_engine.alpha == 0.01
    np.testing.assert_allclose(batch_engine.radii, seq_engine.radii)


def test_trainer_predict_with_intervals_and_compute_coverage_match_engine():
    model = ZeroForecastModel(horizon=2)
    config = TrainingConfig(
        conformal_enabled=True,
        conformal_method="split",
        conformal_quantile=0.5,
        batch_size=8,
        use_amp=False,
        conformal_local_window=321,
        conformal_tsp_lambda=0.123,
        conformal_tsp_window=456,
        conformal_afocp_internal_feat_hidden=33,
        conformal_afocp_internal_feat_depth=4,
        conformal_afocp_internal_feat_dropout=0.2,
    )
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        auto_track=False,
        device="cpu",
    )
    assert trainer.conformal_engine.local_window == 321
    assert trainer.conformal_engine.tsp_lambda == 0.123
    assert trainer.conformal_engine.tsp_window == 456
    assert trainer.conformal_engine.afocp_internal_feat_hidden == 33
    assert trainer.conformal_engine.afocp_internal_feat_depth == 4
    assert trainer.conformal_engine.afocp_internal_feat_dropout == 0.2

    X_cal = torch.zeros((4, 3), dtype=torch.float32)
    y_cal = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    cal_loader = DataLoader(TensorDataset(X_cal, y_cal), batch_size=4, shuffle=False)
    trainer.calibrate_conformal(cal_loader)

    X_test = torch.zeros((2, 3), dtype=torch.float32)
    y_test = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=torch.float32,
    )

    preds, lower, upper = trainer.predict_with_intervals(X_test, return_tensors=False)
    stats = trainer.compute_coverage(X_test, y_test)

    assert preds.shape == (2, 2, 1)
    assert lower.shape == (2, 2, 1)
    assert upper.shape == (2, 2, 1)
    assert stats["coverage"] == 0.5
    assert stats["joint_coverage"] == 0.0
    assert stats["coverage_gap"] == 0.0
    assert stats["joint_coverage_gap"] == -0.5


def test_trainer_compute_coverage_streaming_reports_elementwise_and_joint_metrics():
    model = ZeroForecastModel(horizon=2)
    config = TrainingConfig(
        conformal_enabled=True,
        conformal_method="split",
        conformal_quantile=0.5,
        batch_size=1,
        use_amp=False,
    )
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        auto_track=False,
        device="cpu",
    )

    X_cal = torch.zeros((4, 3), dtype=torch.float32)
    y_cal = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    trainer.calibrate_conformal(
        DataLoader(TensorDataset(X_cal, y_cal), batch_size=4, shuffle=False)
    )

    X_test = torch.zeros((2, 3), dtype=torch.float32)
    y_test = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    stats = trainer.compute_coverage_streaming(
        DataLoader(TensorDataset(X_test, y_test), batch_size=1, shuffle=False),
        do_update=False,
    )

    assert stats["coverage"] == 0.5
    assert stats["joint_coverage"] == 0.0
    assert stats["coverage_gap"] == 0.0
    assert stats["joint_coverage_gap"] == -0.5
    np.testing.assert_allclose(stats["per_horizon_coverage"], np.array([0.5, 0.5]))
    np.testing.assert_allclose(stats["per_feature_coverage"], np.array([0.5]))


def test_trainer_compute_coverage_streaming_reports_full_joint_metrics_for_multitarget_forecasts():
    model = ZeroForecastModel(horizon=2, output_dim=2)
    config = TrainingConfig(
        conformal_enabled=True,
        conformal_method="split",
        conformal_quantile=0.5,
        batch_size=1,
        use_amp=False,
    )
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        auto_track=False,
        device="cpu",
    )

    X_cal = torch.zeros((4, 3), dtype=torch.float32)
    y_cal = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    trainer.calibrate_conformal(
        DataLoader(TensorDataset(X_cal, y_cal), batch_size=4, shuffle=False)
    )

    X_test = torch.zeros((2, 3), dtype=torch.float32)
    y_test = torch.tensor(
        [
            [[1.0, 3.0], [1.0, 1.0]],
            [[3.0, 1.0], [1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    stats = trainer.compute_coverage_streaming(
        DataLoader(TensorDataset(X_test, y_test), batch_size=1, shuffle=False),
        do_update=False,
    )

    assert stats["coverage"] == 0.75
    assert stats["joint_coverage"] == 0.0
    np.testing.assert_allclose(stats["per_feature_coverage"], np.array([0.75, 0.75]))
    np.testing.assert_allclose(
        stats["per_feature_joint_coverage"], np.array([0.5, 0.5])
    )
    np.testing.assert_allclose(
        stats["per_horizon_all_feature_coverage"], np.array([0.0, 1.0])
    )


def test_plot_intervals_does_not_mutate_adaptive_conformal_state_by_default():
    model = ZeroForecastModel(horizon=2)
    config = TrainingConfig(
        conformal_enabled=True,
        conformal_method="rolling",
        conformal_quantile=0.5,
        batch_size=2,
        use_amp=False,
    )
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        auto_track=False,
        device="cpu",
    )

    X_cal = torch.zeros((4, 3), dtype=torch.float32)
    y_cal = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    trainer.calibrate_conformal(
        DataLoader(TensorDataset(X_cal, y_cal), batch_size=4, shuffle=False)
    )

    alpha_before = trainer.conformal_engine.alpha
    radii_before = trainer.conformal_engine.radii.copy()
    buffer_before = trainer.conformal_engine.residuals_buffer.clone()

    X_val = torch.zeros((2, 3), dtype=torch.float32)
    y_val = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    fig = trainer.plot_intervals(
        X_val=X_val,
        y_val=y_val,
        full_series=torch.zeros((6, 1), dtype=torch.float32),
        show=False,
    )
    fig.clf()

    assert trainer.conformal_engine.alpha == alpha_before
    np.testing.assert_allclose(trainer.conformal_engine.radii, radii_before)
    assert torch.equal(trainer.conformal_engine.residuals_buffer, buffer_before)


def test_plot_intervals_accepts_custom_time_index():
    model = ZeroForecastModel(horizon=2)
    config = TrainingConfig(
        conformal_enabled=True,
        conformal_method="split",
        conformal_quantile=0.5,
        batch_size=2,
        use_amp=False,
    )
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        auto_track=False,
        device="cpu",
    )

    X_cal = torch.zeros((4, 3), dtype=torch.float32)
    y_cal = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    trainer.calibrate_conformal(
        DataLoader(TensorDataset(X_cal, y_cal), batch_size=4, shuffle=False)
    )

    X_val = torch.zeros((2, 3), dtype=torch.float32)
    y_val = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    time_index = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)

    fig = trainer.plot_intervals(
        X_val=X_val,
        y_val=y_val,
        full_series=torch.zeros((6, 1), dtype=torch.float32),
        time_index=time_index,
        show=False,
        show_width_plot=False,
    )

    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), time_index)
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), time_index[3:])
    np.testing.assert_array_equal(ax.lines[2].get_xdata(), np.array([40, 40]))
    assert ax.get_xlabel() == "Time"
    fig.clf()
