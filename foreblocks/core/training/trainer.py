"""Enhanced Trainer with NAS and conformal prediction support.

This module is a thin orchestrator that delegates to focused submodules:

* ``batch_io`` – batch unpacking and device transfer
* ``training_loop`` – ``train_epoch``, ``evaluate``, forward/backward passes
* ``logging`` – MLTracker and MoE logging helpers
* ``conformal_trainer`` – conformal calibration, update, and prediction API
* ``visualization`` – prediction and interval plots

See the submodule docstrings for implementation details.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import tempfile
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from foreblocks.config import TrainingConfig
from foreblocks.core.evaluation.model_evaluator import ModelEvaluator
from foreblocks.core.training import (
    batch_io,
    conformal_trainer as _conf,
    logging as _log,
    training_loop,
    visualization as _viz,
)
from foreblocks.core.training.history import TrainingHistory
from foreblocks.core.training.losses import LossComputer
from foreblocks.core.training.nas import NASHelper


# ── Optional imports ────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None  # type: ignore[assignment]
    ListedColormap = None  # type: ignore[assignment]

try:
    from foreblocks.modules.moe.experts.moe import MoEFeedForwardDMoE
    from foreblocks.modules.moe.experts.moe_logging import (
        MoELogger,
        ReportInputs,
        build_moe_report,
    )
    from foreblocks.modules.moe.ff import FeedForwardBlock
except Exception:
    MoELogger = None  # type: ignore[assignment]
    ReportInputs = None  # type: ignore[assignment]

    def build_moe_report(*args, **kwargs: Any) -> Any:
        raise RuntimeError("MoE logging not available")


# ========================================================================
# Trainer
# ========================================================================

class Trainer:
    """Unified training loop with NAS, conformal prediction, and MoE logging.

    Public API
    ----------
    train() – full training loop with optional validation and early stopping
    evaluate() – quick validation loss
    save() / load() – checkpointing
    calibrate_conformal() / predict_with_intervals() / compute_coverage()
    metrics() / cv() – evaluation helpers
    plot_prediction() / plot_intervals() – visualization
    """

    # ── Device resolution ──────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str | torch.device | None = None) -> torch.device:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        return device

    # ── Initialization ─────────────────────────────────────────────────

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | dict[str, Any] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: Callable | None = None,
        scheduler: Any | None = None,
        device: str | None = None,
        use_wandb: bool = False,
        wandb_config: dict[str, Any] | None = None,
        moe_meta_builder: Callable[..., dict[str, Any] | None] | None = None,
        alpha_optimizer: torch.optim.Optimizer | None = None,
        mltracker: Any | None = None,
        mltracker_uri: str | None = None,
        auto_track: bool = True,
    ) -> None:
        """Initialize the Trainer.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model to train.
        config : TrainingConfig | dict | None
            Training configuration.  If a dict, it is converted to
            ``TrainingConfig`` and updated with the given keys.
        optimizer : torch.optim.Optimizer | None
            Custom optimizer.  Created automatically (AdamW) if *None*.
        criterion : Callable | None
            Deprecated – use ``config.loss_function`` or set via
            ``self.criterion = ...`` after construction.
        scheduler : Any | None
            Deprecated – use ``config.scheduler_type`` instead.
        device : str | None
            Device string ("cpu", "cuda", …).  Auto-detected if *None*.
        use_wandb : bool
            Deprecated – WandB support has been removed.
        wandb_config : dict | None
            Deprecated.
        moe_meta_builder : callable | None
            Custom MoE metadata builder.  The default reads ``time_feat``
            to derive an ``"hour"`` feature.
        alpha_optimizer : torch.optim.Optimizer | None
            Deprecated – NAS alpha optimization is handled internally.
        mltracker : Any | None
            Pre-existing MLTracker instance.  If *None* and *auto_track* is
            True, one is auto-created from the DB path.
        mltracker_uri : str | None
            Path to the MLTracker SQLite DB.  Defaults to
            ``<project_root>/mltracker/mltracker_data``.
        auto_track : bool
            When True and no *mltracker* is provided, auto-create one.
        """
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.use_wandb = use_wandb

        # ── MLTracker DB path ──────────────────────────────────────────
        import os as _os

        if mltracker_uri is None:
            mltracker_uri = _os.environ.get(
                "MLTRACKER_DIR",
                str(Path(__file__).resolve().parents[2] / "mltracker/mltracker_data"),
            )
        self._mltracker_uri = mltracker_uri
        self._last_run_id: Any = None

        # ── Auto-create MLTracker when none is supplied ────────────────
        if mltracker is not None:
            self.mltracker = mltracker
        elif auto_track:
            try:
                from foreblocks.mltracker.mltracker import MLTracker

                self.mltracker = MLTracker(tracking_uri=mltracker_uri)
            except Exception as _mt_err:
                print(f"[MLTracker] Auto-track init failed, tracking disabled: {_mt_err}")
                self.mltracker = None
        else:
            self.mltracker = None

        # ── Config ─────────────────────────────────────────────────────
        if isinstance(config, dict):
            self.config = TrainingConfig()
            self.config.update(**config)
        else:
            self.config = config or TrainingConfig()

        # ── Device & AMP ───────────────────────────────────────────────
        self._amp_enabled = getattr(self.config, "use_amp", False) and self.device.type == "cuda"
        self.scaler: GradScaler | None = GradScaler() if self._amp_enabled else None

        # ── Optimizer ──────────────────────────────────────────────────
        self.nas_helper = NASHelper(self.model, self.config)

        if getattr(self.config, "train_nas", False) and self.nas_helper.has_nas:
            _weight_params = self.nas_helper.get_weight_parameters()
            _alpha_params = self.nas_helper.get_alpha_parameters()
            self._alpha_params = _alpha_params
            self._weight_params = _weight_params
            print(
                f"[NAS] Training with NAS. Found {len(_alpha_params)} architecture parameters."
            )
            self.optimizer = optimizer or torch.optim.AdamW(
                _weight_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self._alpha_params: list[torch.nn.Parameter] = []
            self._weight_params = list(self.model.parameters())
            self.optimizer = optimizer or torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # ── Scheduler ──────────────────────────────────────────────────
        self.scheduler = self._create_scheduler()

        # ── Loss ───────────────────────────────────────────────────────
        self.loss_computer = LossComputer(self.model, self.config, criterion)

        # ── History ────────────────────────────────────────────────────
        self.history = TrainingHistory()

        # ── Current state ──────────────────────────────────────────────
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_model_state: dict[str, Any] | None = None

        # ── MoE logging ────────────────────────────────────────────────
        self.moe_log: MoELogger | None = None
        self.moe_meta_builder = (
            moe_meta_builder if moe_meta_builder is not None else self._default_moe_meta_builder
        )
        self._wire_moe_logger(self.model, None, self._get_step, False)

        # ── Conformal ──────────────────────────────────────────────────
        if getattr(self.config, "conformal_enabled", False):
            self.conformal_engine = self._create_conformal_engine()
        else:
            self.conformal_engine = None

    # ── Conformal engine factory ───────────────────────────────────────

    def _create_conformal_engine(self) -> Any:
        """Create conformal engine with all method-specific parameters from config."""
        from foreblocks.core.training.conformal import ConformalPredictionEngine

        return ConformalPredictionEngine(
            method=getattr(self.config, "conformal_method", "split"),
            quantile=getattr(self.config, "conformal_quantile", 0.9),
            knn_k=getattr(self.config, "conformal_knn_k", 50),
            local_window=getattr(self.config, "conformal_local_window", 5000),
            rolling_alpha=getattr(self.config, "conformal_rolling_alpha", 0.05),
            aci_gamma=getattr(self.config, "conformal_aci_gamma", 0.01),
            agaci_gammas=getattr(self.config, "conformal_agaci_gammas", None),
            enbpi_B=getattr(self.config, "conformal_enbpi_B", 20),
            enbpi_window=getattr(self.config, "conformal_enbpi_window", 500),
            tsp_lambda=getattr(self.config, "conformal_tsp_lambda", 0.01),
            tsp_window=getattr(self.config, "conformal_tsp_window", 5000),
            cptc_window=getattr(self.config, "conformal_cptc_window", 500),
            cptc_tau=getattr(self.config, "conformal_cptc_tau", 1.0),
            cptc_hard_state_filter=getattr(self.config, "conformal_cptc_hard_state_filter", False),
            afocp_feature_dim=getattr(self.config, "conformal_afocp_feature_dim", 128),
            afocp_attn_hidden=getattr(self.config, "conformal_afocp_attn_hidden", 64),
            afocp_window=getattr(self.config, "conformal_afocp_window", 500),
            afocp_tau=getattr(self.config, "conformal_afocp_tau", 1.0),
            afocp_internal_feat_hidden=getattr(self.config, "conformal_afocp_internal_feat_hidden", 256),
            afocp_internal_feat_depth=getattr(self.config, "conformal_afocp_internal_feat_depth", 3),
            afocp_internal_feat_dropout=getattr(self.config, "conformal_afocp_internal_feat_dropout", 0.1),
            afocp_online_lr=getattr(self.config, "conformal_afocp_online_lr", 0.0),
            afocp_online_steps=getattr(self.config, "conformal_afocp_online_steps", 1),
        )

    # ── MoE helpers ────────────────────────────────────────────────────

    @staticmethod
    def _default_moe_meta_builder(
        X: torch.Tensor,
        y: torch.Tensor | None,
        time_feat: torch.Tensor | None,
        epoch: int,
        batch_idx: int,
    ) -> dict[str, Any] | None:
        if time_feat is None:
            return None
        meta: dict[str, Any] = {}
        if time_feat.dtype in (torch.int32, torch.int64) and time_feat.ndim >= 1:
            meta["hour"] = time_feat.view(-1).clamp_min(0).clamp_max(23)
        return meta or None

    def _wire_moe_logger(
        self,
        module: nn.Module,
        moe_logger: MoELogger | None,
        step_getter: Callable[[], int],
        log_latency: bool,
    ) -> None:
        if moe_logger is None:
            return
        for child in module.modules():
            try:
                is_moe = False
                if MoEFeedForwardDMoE is not None and isinstance(child, MoEFeedForwardDMoE):
                    is_moe = True
                if FeedForwardBlock is not None and isinstance(child, FeedForwardBlock) and getattr(child, "use_moe", False):
                    is_moe = True
                    moe_block = getattr(child, "block", None)
                    if moe_block is not None:
                        setattr(moe_block, "moe_logger", moe_logger)
                        setattr(moe_block, "step_getter", step_getter)
                        setattr(moe_block, "log_latency", bool(log_latency))
                        continue
                if is_moe and hasattr(child, "moe_logger"):
                    setattr(child, "moe_logger", moe_logger)
                    setattr(child, "step_getter", step_getter)
                    setattr(child, "log_latency", bool(log_latency))
            except Exception:
                pass

    # ── Loss criterion property ────────────────────────────────────────

    @property
    def criterion(self) -> Any:
        return self.loss_computer.criterion

    @criterion.setter
    def criterion(self, value: Any) -> None:
        self.loss_computer.criterion = value

    # ── Optimizer & scheduler factories ────────────────────────────────

    def _create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> Any | None:
        stype = getattr(self.config, "scheduler_type", None)
        if stype == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(self.config, "lr_step_size", 30),
                gamma=getattr(self.config, "lr_gamma", 0.1),
            )
        if stype == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=getattr(self.config, "lr_gamma", 0.1),
                patience=max(1, getattr(self.config, "patience", 10) // 2),
                min_lr=getattr(self.config, "min_lr", 1e-6),
            )
        if stype == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, getattr(self.config, "num_epochs", 100)),
                eta_min=getattr(self.config, "min_lr", 1e-6),
            )
        return None

    # ── Training infrastructure helpers ────────────────────────────────

    @contextlib.contextmanager
    def _amp_context(self) -> Any:
        if self._amp_enabled:
            with autocast("cuda"):
                yield
        else:
            yield

    def _step_scheduler(self, train_loss: float, val_loss: float | None = None) -> None:
        """Step scheduler using the correct API for metric-aware schedulers."""
        if self.scheduler is None:
            return
        metric = val_loss if val_loss is not None else train_loss
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    # ── Step getter (used by MoE logging) ──────────────────────────────

    def _get_step(self) -> int:
        return self.global_step

    # ── Alpha optimizer & parameter separation ─────────────────────────

    def _separate_alpha_optimizer(self) -> None:
        """Separate architecture parameters (α) from weights (θ)."""
        if not self.nas_helper.has_nas:
            return
        self._alpha_optimizer, self._weight_params, self._alpha_params = self.nas_helper._setup_optimizer(self.optimizer, self.model)

    def _get_alpha_optimizer(self) -> torch.optim.Optimizer | None:
        """Return the alpha optimizer (or the main optimizer if NAS is disabled)."""
        return getattr(self, "_alpha_optimizer", None) or self.optimizer

    @property
    def weight_params(self) -> list[torch.nn.Parameter]:
        return self._weight_params

    @property
    def alpha_params(self) -> list[torch.nn.Parameter]:
        return self._alpha_params

    @property
    def alpha_optimizer(self) -> torch.optim.Optimizer | None:
        """Return the NAS alpha optimizer, or the main optimizer if NAS is disabled."""
        return getattr(self, "_alpha_optimizer", None)

    @alpha_optimizer.setter
    def alpha_optimizer(self, value: torch.optim.Optimizer | None) -> None:
        self._alpha_optimizer = value

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> tuple[float, dict[str, float]]:
        """Train for one epoch (thin wrapper around ``training_loop.train_epoch``).

        Returns ``(total_loss, avg_components)``.
        """
        _global_step = {"step": self.global_step}
        train_loss, components, batches = training_loop.train_epoch(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            loss_computer=self.loss_computer,
            optimizer=self.optimizer,
            global_step_ref=_global_step,
            nas_helper=self.nas_helper,
            scaler=self.scaler,
            amp_context=self._amp_context,
            moe_log=self.moe_log,
            moe_meta_builder=self.moe_meta_builder,
            current_epoch=self.current_epoch,
            forward_pass_fn=training_loop.forward_pass,
            backward_step_fn=training_loop.backward_step,
            device=self.device,
        )
        self.global_step = _global_step["step"]
        return train_loss, components

    # ── Training loop ──────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        callbacks: list[Any] | None = None,
        epochs: int | None = None,
        moe_report_outdir: str | None = None,
        run_name: str | None = None,
    ) -> TrainingHistory:
        """Run the full training loop with optional validation and early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader | None
            Optional validation loader.
        callbacks : list[Any] | None
            List of callback objects with ``on_epoch_begin`` / ``on_epoch_end``.
        epochs : int | None
            Override ``self.config.num_epochs``.
        moe_report_outdir : str | None
            Directory for MoE expert reports.
        run_name : str | None
            Optional name for the MLTracker run.
        """
        callbacks = callbacks or []
        num_epochs = epochs if epochs is not None else self.config.num_epochs

        run_context, run_name = _log.init_mltracker_run_context(self.mltracker, run_name)

        with run_context:
            if self.mltracker and self.mltracker._active_run:
                self._last_run_id = self.mltracker._active_run
            _log.log_mltracker_params(self.mltracker, self.config)
            _log.log_mltracker_model_info(self.mltracker, self.model, self.device)

            completed_epochs = 0
            stopped_early = False

            with tqdm(range(num_epochs), desc="Training", unit="epoch") as pbar:
                for epoch in pbar:
                    self.current_epoch = epoch

                    for cb in callbacks:
                        if hasattr(cb, "on_epoch_begin"):
                            cb.on_epoch_begin(self, epoch)

                    _global_step = {"step": self.global_step}
                    train_loss, components, batches = training_loop.train_epoch(
                        model=self.model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        config=self.config,
                        loss_computer=self.loss_computer,
                        optimizer=self.optimizer,
                        global_step_ref=_global_step,
                        nas_helper=self.nas_helper,
                        scaler=self.scaler,
                        amp_context=self._amp_context,
                        moe_log=self.moe_log,
                        moe_meta_builder=self.moe_meta_builder,
                        forward_pass_fn=training_loop.forward_pass,
                        backward_step_fn=training_loop.backward_step,
                        device=self.device,
                    )
                    self.global_step = _global_step["step"]

                    val_loss = self.evaluate(val_loader) if val_loader else None

                    lr = self.optimizer.param_groups[0]["lr"]
                    model_info = self.model.get_model_size() if hasattr(self.model, "get_model_size") else None

                    alpha_info = None
                    if getattr(self.config, "train_nas", False) and self.nas_helper.has_nas:
                        alpha_info = self.nas_helper.collect_alpha_report()

                    self.history.record_epoch(train_loss, val_loss, lr, components, model_info, alpha_info)

                    if val_loader:
                        if val_loss + getattr(self.config, "min_delta", 0) < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.epochs_without_improvement = 0
                            self.best_model_state = copy.deepcopy(self.model.state_dict())
                        else:
                            self.epochs_without_improvement += 1
                        if self.epochs_without_improvement >= getattr(self.config, "patience", 10):
                            print(f"\nEarly stopping at epoch {epoch + 1}")
                            completed_epochs = epoch + 1
                            stopped_early = True
                            break

                    pbar.set_postfix({"train": train_loss, "val": val_loss, "lr": lr})
                    self._step_scheduler(train_loss, val_loss)

                    for cb in callbacks:
                        if hasattr(cb, "on_epoch_end"):
                            cb.on_epoch_end(self, epoch, {
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "lr": lr,
                            })

                    _log.log_mltracker_metrics(
                        self.mltracker, epoch, train_loss, lr, components, val_loss,
                    )
                    completed_epochs = epoch + 1

            _log.log_mltracker_final(self.mltracker, completed_epochs, stopped_early, self.best_val_loss)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        _log.log_model_to_last_run(self.mltracker, self._last_run_id, self.model, model_name="model")

        if getattr(self.config, "conformal_enabled", False) and self.conformal_engine is not None:
            print("\n[Conformal] Engine ready. Call calibrate_conformal(cal_loader) with held-out data.")

        return self.history

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on *dataloader* and return mean loss."""
        return training_loop.evaluate(
            self.model, dataloader, self.device, self._amp_context,
            self.moe_log, self.moe_meta_builder,
        )

    # ── Conformal API ──────────────────────────────────────────────────

    def calibrate_conformal(
        self,
        cal_loader: DataLoader,
        state_model: Any = None,
        feature_extractor: Any = None,
        jackknife_cv_models: Any = None,
        jackknife_cv_indices: Any = None,
        enbpi_member_models: Any = None,
        enbpi_boot_indices: Any = None,
    ) -> None:
        """Calibrate conformal engine with held-out calibration data."""
        _conf.calibrate_conformal(
            self, cal_loader, state_model, feature_extractor,
            jackknife_cv_models, jackknife_cv_indices,
            enbpi_member_models, enbpi_boot_indices,
        )

    def update_conformal(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        state_model: Any = None,
        feature_extractor: Any = None,
        sequential: bool = True,
    ) -> None:
        """Online update for adaptive conformal methods."""
        _conf.update_conformal(self, X_new, y_new, state_model, feature_extractor, sequential)

    def predict_with_intervals(
        self,
        X: torch.Tensor,
        return_tensors: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with conformal intervals."""
        return _conf.predict_with_intervals(self, X, return_tensors)

    def compute_coverage(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Empirical coverage and basic interval stats."""
        return _conf.compute_coverage(self, X, y)

    def predict_with_intervals_streaming(
        self,
        dataloader: DataLoader,
        do_update: bool = True,
        return_numpy: bool = True,
        sequential: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Streaming (rolling) prediction over a DataLoader."""
        return _conf.predict_with_intervals_streaming(self, dataloader, do_update, return_numpy, sequential)

    def compute_coverage_streaming(
        self,
        dataloader: DataLoader,
        do_update: bool = True,
        sequential: bool | None = None,
    ) -> dict[str, Any]:
        """Coverage diagnostics for streaming/rolling evaluation."""
        return _conf.compute_coverage_streaming(self, dataloader, do_update, sequential)

    # ── Saving / loading ───────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save a checkpoint (model, optimizer, config, history, conformal state)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else dict(self.config),
            "history": {
                "train_losses": self.history.train_losses,
                "val_losses": self.history.val_losses,
                "learning_rates": self.history.learning_rates,
                "alpha_values": self.history.alpha_values,
            },
        }
        if self.alpha_optimizer is not None:
            save_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        if self.conformal_engine is not None and getattr(self.conformal_engine, "radii", None) is not None:
            save_dict["conformal_radii"] = self.conformal_engine.radii
            save_dict["conformal_method"] = self.conformal_engine.method

        torch.save(save_dict, path)

    def load(self, path: str | Path) -> None:
        """Load a checkpoint (model, optimizer, config, conformal state)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "alpha_optimizer_state_dict" in checkpoint and self.alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        if "config" in checkpoint:
            if hasattr(self.config, "update"):
                self.config.update(**checkpoint["config"])
            elif hasattr(self.config, "__dict__"):
                self.config.__dict__.update(checkpoint["config"])

        if "conformal_radii" in checkpoint and self.conformal_engine is not None:
            self.conformal_engine.radii = checkpoint["conformal_radii"]

    # ── Model utilities ────────────────────────────────────────────────

    @staticmethod
    def _infer_num_experts(model: nn.Module) -> int | None:
        """Walk the model tree to find the first ``num_experts`` attribute."""
        for m in model.modules():
            if hasattr(m, "num_experts"):
                try:
                    ne = int(getattr(m, "num_experts"))
                    if ne > 0:
                        return ne
                except Exception:
                    pass
        return None

    # ── Evaluation wrappers ────────────────────────────────────────────

    def metrics(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        batch_size: int = 256,
        graph_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Compute evaluation metrics via ``ModelEvaluator``."""
        evaluator = ModelEvaluator(self)
        result = evaluator.compute_metrics(X_val, y_val, batch_size, graph_kwargs=graph_kwargs)
        _log.log_to_last_run(self.mltracker, self._last_run_id, result, step=None, prefix="eval/")
        return result

    def cv(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_windows: int,
        horizon: int,
        step_size: int | None = None,
        batch_size: int = 256,
        graph_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """K-fold cross-validation via ``ModelEvaluator``."""
        evaluator = ModelEvaluator(self)
        return evaluator.cross_validation(X, y, n_windows, horizon, step_size, batch_size, graph_kwargs=graph_kwargs)

    # ── Visualization ──────────────────────────────────────────────────

    def plot_prediction(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        graph_kwargs: dict[str, Any] | None = None,
        full_series: torch.Tensor | None = None,
        offset: int = 0,
        stride: int = 1,
        figsize: tuple[int, int] = (12, 4),
        show: bool = True,
        names: str | list | None = None,
        pred_color: str = "orange",
        series_color: str = "blue",
        save_path: str | None = None,
    ) -> plt.Figure:  # type: ignore[name-defined]
        """Plot model predictions against actual values."""
        _viz._require_matplotlib()
        fig = _viz.plot_prediction(
            self, X_val, y_val, graph_kwargs, full_series, offset, stride,
            figsize, show, names, pred_color, series_color, save_path,
        )
        if self.mltracker and self._last_run_id:
            _log.log_figure_to_last_run(self.mltracker, self._last_run_id, fig)
        return fig

    def plot_intervals(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        full_series: torch.Tensor | None = None,
        time_index: Sequence[Any] | None = None,
        offset: int = 0,
        stride: int = 1,
        figsize: tuple[int, int] = (14, 5),
        show: bool = True,
        names: str | list | None = None,
        interval_alpha: float = 0.25,
        pred_color: str = "blue",
        interval_color: str = "blue",
        aggregation: str = "envelope",
        show_width_plot: bool = True,
        min_count: int = 1,
        do_update: bool = False,
    ) -> plt.Figure:  # type: ignore[name-defined]
        """Plot predictions with conformal intervals."""
        _viz._require_matplotlib()

        if self.conformal_engine is None or getattr(self.conformal_engine, "radii", None) is None:
            raise RuntimeError("Conformal engine not calibrated. Call calibrate_conformal() first.")

        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
        preds, lower, upper, y_stream = self.predict_with_intervals_streaming(
            val_loader, do_update=do_update, return_numpy=True,
        )

        N, H, D = preds.shape
        seq_len = X_val.shape[1]

        if full_series is None:
            raise ValueError("full_series must be provided for time-aligned plotting.")

        series = full_series.detach().cpu().numpy() if isinstance(full_series, torch.Tensor) else full_series
        if series.ndim == 1:
            series = series[:, None]

        T, S_dim = series.shape
        D_plot = min(D, S_dim)
        names = names or [f"Feature {i}" for i in range(D_plot)]

        starts = offset + seq_len + np.arange(N) * stride
        coverage_end = min(int(starts[-1] + H), T)
        if time_index is None:
            xs = np.arange(coverage_end)
            first_forecast_x = offset + seq_len
            xlabel = "Time Step"
        else:
            xs_full = np.asarray(time_index)
            if xs_full.ndim != 1:
                raise ValueError("time_index must be 1-dimensional.")
            if len(xs_full) < coverage_end:
                raise ValueError(f"time_index must have at least {coverage_end} elements, got {len(xs_full)}.")
            if offset + seq_len >= len(xs_full):
                raise ValueError(f"time_index must include the first forecast boundary at {offset + seq_len}.")
            xs = xs_full[:coverage_end]
            first_forecast_x = xs_full[offset + seq_len]
            xlabel = "Time"

        count = np.zeros((T,))

        # Initialize based on aggregation method
        if aggregation == "envelope":
            agg_pred = np.zeros((T, D_plot))
            agg_low = np.full((T, D_plot), np.inf)
            agg_up = np.full((T, D_plot), -np.inf)
            for k in range(N):
                start = int(starts[k])
                if start >= T:
                    continue
                end = min(start + H, T)
                h = end - start
                if h <= 0:
                    continue
                for j in range(D_plot):
                    pred_col = j if D > j else 0
                    agg_pred[start:end, j] += preds[k, :h, pred_col]
                    agg_low[start:end, j] = np.minimum(agg_low[start:end, j], lower[k, :h, pred_col])
                    agg_up[start:end, j] = np.maximum(agg_up[start:end, j], upper[k, :h, pred_col])
                count[start:end] += 1
            have = count >= min_count
            mean_pred = np.zeros_like(agg_pred)
            mean_pred[have] = agg_pred[have] / count[have, None]
            mean_low = np.where(agg_low == np.inf, 0, agg_low)
            mean_up = np.where(agg_up == -np.inf, 0, agg_up)

        elif aggregation == "last":
            mean_pred = np.full((T, D_plot), np.nan)
            mean_low = np.full((T, D_plot), np.nan)
            mean_up = np.full((T, D_plot), np.nan)
            for k in range(N):
                start = int(starts[k])
                if start >= T:
                    continue
                end = min(start + H, T)
                h = end - start
                if h <= 0:
                    continue
                for j in range(D_plot):
                    pred_col = j if D > j else 0
                    mean_pred[start:end, j] = preds[k, :h, pred_col]
                    mean_low[start:end, j] = lower[k, :h, pred_col]
                    mean_up[start:end, j] = upper[k, :h, pred_col]
                count[start:end] += 1
            have = (~np.isnan(mean_pred[:, 0])) & (count >= min_count)

        elif aggregation == "min_width":
            mean_pred = np.full((T, D_plot), np.nan)
            mean_low = np.full((T, D_plot), np.nan)
            mean_up = np.full((T, D_plot), np.nan)
            min_width = np.full((T, D_plot), np.inf)
            for k in range(N):
                start = int(starts[k])
                if start >= T:
                    continue
                end = min(start + H, T)
                h = end - start
                if h <= 0:
                    continue
                for j in range(D_plot):
                    pred_col = j if D > j else 0
                    width_k = upper[k, :h, pred_col] - lower[k, :h, pred_col]
                    for t_idx, t in enumerate(range(start, end)):
                        if width_k[t_idx] < min_width[t, j]:
                            min_width[t, j] = width_k[t_idx]
                            mean_pred[t, j] = preds[k, t_idx, pred_col]
                            mean_low[t, j] = lower[k, t_idx, pred_col]
                            mean_up[t, j] = upper[k, t_idx, pred_col]
                count[start:end] += 1
            have = (~np.isnan(mean_pred[:, 0])) & (count >= min_count)

        else:  # "mean"
            acc_pred = np.zeros((T, D_plot))
            acc_low = np.zeros((T, D_plot))
            acc_up = np.zeros((T, D_plot))
            for k in range(N):
                start = int(starts[k])
                if start >= T:
                    continue
                end = min(start + H, T)
                h = end - start
                if h <= 0:
                    continue
                for j in range(D_plot):
                    pred_col = j if D > j else 0
                    acc_pred[start:end, j] += preds[k, :h, pred_col]
                    acc_low[start:end, j] += lower[k, :h, pred_col]
                    acc_up[start:end, j] += upper[k, :h, pred_col]
                count[start:end] += 1
            have = count >= min_count
            mean_pred = np.zeros_like(acc_pred)
            mean_low = np.zeros_like(acc_low)
            mean_up = np.zeros_like(acc_up)
            for j in range(D_plot):
                mean_pred[have, j] = acc_pred[have, j] / count[have]
                mean_low[have, j] = acc_low[have, j] / count[have]
                mean_up[have, j] = acc_up[have, j] / count[have]

        interval_widths = mean_up - mean_low
        n_rows = D_plot + (1 if show_width_plot else 0)

        fig, axes = plt.subplots(n_rows, 1, figsize=(figsize[0], figsize[1] * n_rows), sharex=True)
        axes = np.atleast_1d(axes)

        for j in range(D_plot):
            ax = axes[j]
            ax.plot(xs, series[:coverage_end, j], label=f"Actual {names[j]}", alpha=0.8, linewidth=1)
            mask = have[:coverage_end]
            if mask.any():
                yp = mean_pred[:coverage_end, j]
                yl = mean_low[:coverage_end, j]
                yu = mean_up[:coverage_end, j]
                ax.plot(xs[mask], yp[mask], label=f"Predicted {names[j]}", linestyle="--", color=pred_color, linewidth=1)
                ax.fill_between(xs[mask], yl[mask], yu[mask], color=interval_color, alpha=interval_alpha, label=f"Interval ({aggregation})")
            ax.axvline(first_forecast_x, color="gray", linestyle="--", alpha=0.5, label="First forecast")
            ax.set_title(f"{names[j]} — Forecast with Conformal Intervals")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

        if show_width_plot:
            ax_width = axes[-1]
            for j in range(D_plot):
                mask = have[:coverage_end]
                widths_j = interval_widths[:coverage_end, j]
                ax_width.plot(xs[mask], widths_j[mask], label=f"Width {names[j]}", alpha=0.8, linewidth=1)
            ax_width.axvline(first_forecast_x, color="gray", linestyle="--", alpha=0.5)
            ax_width.set_ylabel("Interval Width")
            ax_width.set_title("Adaptive Interval Widths Over Time")
            ax_width.legend(loc="upper left", fontsize=8)
            ax_width.grid(True, alpha=0.3)

        axes[-1].set_xlabel(xlabel)
        plt.tight_layout()

        if show:
            plt.show()
        return fig

    def plot_violation_heatmap_streaming(
        self,
        dataloader: DataLoader,
        feature: int = 0,
        do_update: bool = True,
        figsize: tuple[int, int] = (10, 4),
        show: bool = True,
        sequential: bool | None = None,
    ) -> plt.Figure:  # type: ignore[name-defined]
        """Heatmap of conformal misses (outside interval) for streaming evaluation."""
        _viz._require_matplotlib()

        if self.conformal_engine is None or getattr(self.conformal_engine, "radii", None) is None:
            raise RuntimeError("Conformal engine not calibrated. Call calibrate_conformal() first.")

        preds, L, U, y_true = self.predict_with_intervals_streaming(
            dataloader, do_update=do_update, return_numpy=True, sequential=sequential,
        )

        N, H, D = L.shape
        j = int(feature)
        if j < 0 or j >= D:
            raise ValueError(f"feature index out of range: {j} (D={D})")

        covered = (y_true >= L) & (y_true <= U)
        miss = ~covered[:, :, j]

        fig, ax = plt.subplots(figsize=figsize)
        binary_cmap = ListedColormap(["white", "black"])
        im = ax.imshow(miss.astype(float), aspect="auto", interpolation="nearest", cmap=binary_cmap, vmin=0, vmax=1)

        ax.set_xlabel("Horizon")
        ax.set_ylabel("Window index (stream order)")
        ax.set_title(f"Conformal Misses — feature={j}")
        ax.set_xticks(np.arange(H))
        ax.set_xticklabels([str(h + 1) for h in range(H)])

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(["Covered", "Miss"])

        plt.tight_layout()
        if show:
            plt.show()
        return fig
