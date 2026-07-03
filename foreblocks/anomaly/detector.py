"""foreblocks.anomaly.detector.

Detector orchestration and scoring logic for anomaly detection.

Defines the ForeblocksAnomalyDetector class that orchestrates fitting and
scoring for neural anomaly detection models across multiple detection modes
(forecasting, reconstruction, representation, hybrid). Provides configuration
and result types for modular anomaly detection pipelines.

Core API:
- ForeblocksAnomalyDetector: fit/predict neural anomaly detector for multivariate time-series windows
- AnomalyDetectorConfig: configuration for anomaly detection
- AnomalyResult, AnomalyDecisionResult: detection result types

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from foreblocks.anomaly.modes import (
    AnomalyBlock,
    AnomalyBlockSpec,
    AnomalyBlockStack,
    AnomalyDecisionResult,
    DecisionConfig,
    fit_mode_state,
    list_blocks,
    register_block,
    resolve_block,
    resolve_mode,
)
from foreblocks.anomaly.windows import (
    build_sliding_windows,
    fill_nan_forward,
    map_window_scores,
    robust_threshold,
)


@dataclass
class AnomalyResult:
    """Result from single-block anomaly detection."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_scores: np.ndarray


@dataclass
class AnomalyDetectorConfig:
    detection_mode: Literal[
        "auto", "forecasting", "reconstruction", "representation", "hybrid"
    ] = "auto"
    model_type: Literal[
        "transformer_vae",
        "mlp_vae",
        "omni_anomaly",
        "anomaly_transformer",
        "dagmm",
        "tranad",
        "patch_mamba",
        "i_transformer",
    ] = "transformer_vae"
    block_stack: list[str | AnomalyBlockSpec] | None = None
    decision_strategy: Literal["majority", "weighted", "all", "any"] = "majority"
    window_size: int = 32
    contamination: float = 0.01
    decision_contamination: float = 0.01
    score_align: Literal["end", "center", "all"] = "end"
    d_model: int = 128
    latent_size: int = 32
    hidden_size: int = 128
    n_heads: int | None = None
    n_layers: int = 2
    dim_feedforward: int | None = None
    layer_attention_type: str = "standard"
    projection_size: int = 64
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    beta: float = 0.05
    beta_warmup_epochs: int = 5
    patience: int = 5
    scaler_type: Literal["robust", "standard", "minmax"] = "robust"
    device: str | None = None
    num_workers: int = 0
    use_mixed_precision: bool = True
    gradient_clip: float = 1.0
    seed: int | None = 42
    contrastive_temperature: float = 0.2
    augmentation_noise_std: float = 0.05
    reconstruction_weight: float = 1.0
    forecasting_weight: float = 1.0
    representation_weight: float = 0.25
    association_weight: float = 0.1
    energy_weight: float = 0.1
    covariance_weight: float = 0.005
    gmm_components: int = 4
    decision_weights: dict[str, float] | None = None


class ForeblocksAnomalyDetector:
    """Fit/predict neural anomaly detector for multivariate time-series windows."""

    def __init__(self, config: AnomalyDetectorConfig | None = None, **kwargs) -> None:
        if config is None:
            config = AnomalyDetectorConfig(**kwargs)
        elif kwargs:
            merged = {**config.__dict__, **kwargs}
            config = AnomalyDetectorConfig(**merged)
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_mixed_precision = (
            bool(config.use_mixed_precision) and self.device.type == "cuda"
        )
        self.scaler = self._make_scaler(config.scaler_type)
        self.model: torch.nn.Module | None = None
        self.mode = resolve_mode(config)
        self._block_stack: AnomalyBlockStack | None = None
        self._block_models: dict[str, torch.nn.Module] | None = None
        self.threshold_: float | None = None
        self.n_features_: int | None = None

        # Initialize block stack if configured
        if config.block_stack is not None:
            self._block_stack = AnomalyBlockStack(
                blocks=config.block_stack,
                decision=DecisionConfig(
                    strategy=config.decision_strategy,
                    contamination=config.decision_contamination,
                    weights=config.decision_weights,
                ),
            )

    @staticmethod
    def _make_scaler(kind: str):
        if kind == "standard":
            return StandardScaler()
        if kind == "minmax":
            return MinMaxScaler()
        return RobustScaler()

    @property
    def detection_mode(self) -> str:
        return getattr(self.mode, "name", self.mode.block_type())

    def _windows_from_series(
        self, series: np.ndarray, *, fit_scaler: bool
    ) -> np.ndarray:
        x = fill_nan_forward(series)
        scaled = (
            self.scaler.fit_transform(x) if fit_scaler else self.scaler.transform(x)
        )
        return build_sliding_windows(scaled, self.config.window_size)

    def _loader(self, windows: np.ndarray, *, shuffle: bool) -> DataLoader:
        tensor = torch.from_numpy(windows.astype(np.float32, copy=False))
        return DataLoader(
            TensorDataset(tensor),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=max(0, int(self.config.num_workers)),
            pin_memory=self.device.type == "cuda",
        )

    def fit(
        self, series: np.ndarray, validation_split: float = 0.1
    ) -> "ForeblocksAnomalyDetector":
        if self.config.seed is not None:
            torch.manual_seed(int(self.config.seed))
            np.random.seed(int(self.config.seed))

        windows = self._windows_from_series(series, fit_scaler=True)
        self.n_features_ = int(windows.shape[-1])
        self.model = self.mode.build_model(self.config, self.n_features_).to(
            self.device
        )

        n_val = int(len(windows) * float(np.clip(validation_split, 0.0, 0.8)))
        train_windows = windows[:-n_val] if n_val > 0 else windows
        val_windows = windows[-n_val:] if n_val > 0 else None
        train_loader = self._loader(train_windows, shuffle=True)
        val_loader = (
            self._loader(val_windows, shuffle=False)
            if val_windows is not None
            else None
        )

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_state = None
        best_val = float("inf")
        patience = 0

        for epoch in range(int(self.config.epochs)):
            self.model.train()
            for (batch,) in train_loader:
                batch = batch.to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, enabled=self.use_mixed_precision):
                    loss = self._loss(batch, epoch=epoch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                opt.step()

            val_loss = self._validation_loss(val_loader, epoch=epoch)
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                patience += 1
                if val_loader is not None and patience >= int(self.config.patience):
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Handle block stack path
        if self._block_stack is not None:
            models = self._block_stack.build_models(self.config, self.n_features_)
            self._block_stack.fit(
                models=models,
                windows=windows,
                config=self.config,
                epochs=int(self.config.epochs),
                batch_size=int(self.config.batch_size),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                patience=int(self.config.patience),
                seed=int(self.config.seed) if self.config.seed is not None else 42,
                device=self.device,
                use_mixed_precision=self.use_mixed_precision,
                gradient_clip=self.config.gradient_clip,
                num_workers=self.config.num_workers,
            )
            self._block_models = models
            return self

        fit_mode_state(self.mode, self.model, windows, self)
        window_scores = self.score_windows(windows)
        scores = map_window_scores(
            window_scores,
            series_length=fill_nan_forward(series).shape[0],
            window_size=self.config.window_size,
            align=self.config.score_align,
        )
        self.threshold_ = robust_threshold(
            scores,
            contamination=self.config.contamination,
        )
        return self

    def _loss(self, batch: torch.Tensor, *, epoch: int) -> torch.Tensor:
        assert self.model is not None
        return self.mode.loss(self.model, batch, self.config, epoch)

    def _validation_loss(self, loader: DataLoader | None, *, epoch: int) -> float:
        if loader is None:
            return 0.0
        assert self.model is not None
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device, non_blocking=True)
                total += float(self._loss(batch, epoch=epoch).detach().cpu())
                count += 1
        return total / max(1, count)

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Detector is not fitted.")
        self.model.eval()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in self._loader(windows, shuffle=False):
                batch = batch.to(self.device, non_blocking=True)
                scores.append(self.mode.score_batch(self.model, batch, self.config))
        return np.concatenate(scores, axis=0) if scores else np.empty((0,))

    def decision_scores(self, series: np.ndarray) -> np.ndarray:
        windows = self._windows_from_series(series, fit_scaler=False)
        window_scores = self.score_windows(windows)
        return map_window_scores(
            window_scores,
            series_length=fill_nan_forward(series).shape[0],
            window_size=self.config.window_size,
            align=self.config.score_align,
        )

    def predict(self, series: np.ndarray) -> AnomalyResult | AnomalyDecisionResult:
        if self._block_stack is not None and self._block_models is not None:
            # Block stack path
            windows = self._windows_from_series(series, fit_scaler=False)
            result = self._block_stack.predict(self._block_models, windows, self.config)
            # Convert to series-level scores
            scores = map_window_scores(
                result.window_scores,
                series_length=fill_nan_forward(series).shape[0],
                window_size=self.config.window_size,
                align=self.config.score_align,
            )
            return AnomalyDecisionResult(
                scores=scores,
                labels=result.labels,
                threshold=0.0,
                window_scores=result.window_scores,
                block_scores=result.block_scores,
                block_labels=result.block_labels,
                voting_info=result.voting_info,
            )

        # Single-block path
        if self.threshold_ is None:
            raise RuntimeError("Detector is not fitted.")
        windows = self._windows_from_series(series, fit_scaler=False)
        window_scores = self.score_windows(windows)
        scores = map_window_scores(
            window_scores,
            series_length=fill_nan_forward(series).shape[0],
            window_size=self.config.window_size,
            align=self.config.score_align,
        )
        labels = np.where(np.isfinite(scores) & (scores > self.threshold_), 1, 0)
        return AnomalyResult(
            scores=scores,
            labels=labels.astype(np.int8),
            threshold=float(self.threshold_),
            window_scores=window_scores,
        )

    def fit_predict(
        self, series: np.ndarray, validation_split: float = 0.1
    ) -> AnomalyResult | AnomalyDecisionResult:
        self.fit(series, validation_split=validation_split)
        return self.predict(series)
