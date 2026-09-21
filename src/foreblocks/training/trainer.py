"""foreblocks.training.trainer.

Unified training loop with NAS, conformal prediction, and MoE logging.

The ``Trainer`` orchestrates full training with optional validation, early stopping,
checkpointing, and conformal prediction calibration. It integrates NAS alpha
optimization, MoE expert logging, and provides visualization helpers for
predictions and conformal intervals.

Core API:
- Trainer: unified training loop with NAS, conformal prediction, and MoE logging
- Trainer.train(): full training loop with optional validation and early stopping
- Trainer.calibrate_conformal() / predict_with_intervals() / compute_coverage(): conformal prediction API
- Trainer.plot_prediction() / plot_intervals(): visualization

"""

from __future__ import annotations

import contextlib
import copy
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from foreblocks.training.config import TrainingConfig
from foreblocks.evaluation import visualization as _viz
from foreblocks.evaluation.model_evaluator import ModelEvaluator
from foreblocks.training.conformal import workflows as _conf
from foreblocks.training.execution import epochs as training_loop
from foreblocks.training.losses import LossComputer
from foreblocks.training.optimization.nas import NASHelper
from foreblocks.training.state.checkpoint import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from foreblocks.training.state.history import TrainingHistory
from foreblocks.training.telemetry import mltracker as _log

# ── Optional imports ────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None  # type: ignore[assignment]
    ListedColormap = None  # type: ignore[assignment]

try:
    from foreblocks.nn.moe.layer import MoEFeedForwardDMoE
    from foreblocks.nn.moe.logging import (
        MoELogger,
        ReportInputs,
        build_moe_report,
    )
    from foreblocks.nn.moe.feedforward import FeedForwardBlock
except Exception:
    MoELogger = None  # type: ignore[assignment]
    ReportInputs = None  # type: ignore[assignment]

    def build_moe_report(*args, **kwargs: Any) -> Any:
        raise RuntimeError("MoE logging not available")


# ========================================================================
# Trainer
# ========================================================================


class Trainer:
    # ── Device resolution ──────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str | torch.device | None = None) -> torch.device:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        return device

    # ── Initialization ─────────────────────────────────────────────────

    @staticmethod
    def _seed_everything(seed: int | None, deterministic: bool = False) -> None:
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

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
                from mltracker.mltracker import MLTracker

                self.mltracker = MLTracker(tracking_uri=mltracker_uri)
            except Exception as _mt_err:
                print(
                    f"[MLTracker] Auto-track init failed, tracking disabled: {_mt_err}"
                )
                self.mltracker = None
        else:
            self.mltracker = None

        # ── Config ─────────────────────────────────────────────────────
        if isinstance(config, dict):
            self.config = TrainingConfig()
            self.config.update(**config)
        else:
            self.config = config or TrainingConfig()

        self._seed_everything(
            self.config.seed,
            deterministic=self.config.deterministic,
        )

        # ── Device & AMP ───────────────────────────────────────────────
        self._amp_enabled = (
            getattr(self.config, "use_amp", False) and self.device.type == "cuda"
        )
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
            # LLRD only applies to weight params, not alpha params
            if getattr(self.config, "use_llrd", False):
                from foreblocks.training.optimization.llrd import (
                    get_llrd_param_groups,
                )

                param_groups = get_llrd_param_groups(
                    self.model,
                    base_lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    decay=getattr(self.config, "llrd_decay", 0.9),
                )
                # Filter to only include NAS weight params
                weight_param_ids = {id(p) for p in _weight_params}
                param_groups = [
                    {
                        **g,
                        "params": [p for p in g["params"] if id(p) in weight_param_ids],
                    }
                    for g in param_groups
                ]
                param_groups = [g for g in param_groups if g["params"]]
            else:
                param_groups = _weight_params
            self.optimizer = optimizer or torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self._alpha_params: list[torch.nn.Parameter] = []
            self._weight_params = list(self.model.parameters())
            # LLRD: build param groups by layer depth
            if getattr(self.config, "use_llrd", False):
                from foreblocks.training.optimization.llrd import (
                    get_llrd_param_groups,
                )

                param_groups = get_llrd_param_groups(
                    self.model,
                    base_lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    decay=getattr(self.config, "llrd_decay", 0.9),
                )
            else:
                param_groups = self.model.parameters()
            self.optimizer = optimizer or torch.optim.AdamW(
                param_groups,
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
            moe_meta_builder
            if moe_meta_builder is not None
            else self._default_moe_meta_builder
        )
        self._wire_moe_logger(self.model, None, self._get_step, False)

        # ── Conformal ──────────────────────────────────────────────────
        if getattr(self.config, "conformal_enabled", False):
            self.conformal_engine = self._create_conformal_engine()
        else:
            self.conformal_engine = None

    # ── Conformal engine factory ───────────────────────────────────────

    def _create_conformal_engine(self) -> Any:
        from foreblocks.training.conformal.engine import ConformalPredictionEngine

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
            cptc_hard_state_filter=getattr(
                self.config, "conformal_cptc_hard_state_filter", False
            ),
            afocp_feature_dim=getattr(self.config, "conformal_afocp_feature_dim", 128),
            afocp_attn_hidden=getattr(self.config, "conformal_afocp_attn_hidden", 64),
            afocp_window=getattr(self.config, "conformal_afocp_window", 500),
            afocp_tau=getattr(self.config, "conformal_afocp_tau", 1.0),
            afocp_internal_feat_hidden=getattr(
                self.config, "conformal_afocp_internal_feat_hidden", 256
            ),
            afocp_internal_feat_depth=getattr(
                self.config, "conformal_afocp_internal_feat_depth", 3
            ),
            afocp_internal_feat_dropout=getattr(
                self.config, "conformal_afocp_internal_feat_dropout", 0.1
            ),
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
                if MoEFeedForwardDMoE is not None and isinstance(
                    child, MoEFeedForwardDMoE
                ):
                    is_moe = True
                if (
                    FeedForwardBlock is not None
                    and isinstance(child, FeedForwardBlock)
                    and getattr(child, "use_moe", False)
                ):
                    is_moe = True
                    moe_block = getattr(child, "block", None)
                    if moe_block is not None:
                        moe_block.moe_logger = moe_logger
                        moe_block.step_getter = step_getter
                        moe_block.log_latency = bool(log_latency)
                        continue
                if is_moe and hasattr(child, "moe_logger"):
                    child.moe_logger = moe_logger
                    child.step_getter = step_getter
                    child.log_latency = bool(log_latency)
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
        if stype == "warmup_cosine":
            from foreblocks.training.optimization.llrd import WarmupCosineLR

            # Compute warmup_steps: either explicit or from ratio
            warmup_steps = getattr(self.config, "warmup_steps", 0)
            if warmup_steps == 0:
                warmup_ratio = getattr(self.config, "warmup_ratio", 0.0)
                steps_per_epoch = getattr(self.config, "steps_per_epoch", None)
                num_epochs = getattr(self.config, "num_epochs", 100)
                if warmup_ratio > 0 and steps_per_epoch:
                    warmup_steps = int(warmup_ratio * num_epochs * steps_per_epoch)
            # Total steps for cosine annealing
            steps_per_epoch = getattr(self.config, "steps_per_epoch", None)
            num_epochs = getattr(self.config, "num_epochs", 100)
            if steps_per_epoch:
                total_steps = num_epochs * steps_per_epoch
            else:
                # Fallback: treat num_epochs as total steps (documented limitation)
                total_steps = num_epochs
            return WarmupCosineLR(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=getattr(self.config, "min_lr", 1e-6)
                / getattr(self.config, "learning_rate", 1e-3),
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
        if self.scheduler is None:
            return
        # WarmupCosineLR is stepped per-optimizer-step, not per-epoch
        from foreblocks.training.optimization.llrd import WarmupCosineLR

        if isinstance(self.scheduler, WarmupCosineLR):
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
        if not self.nas_helper.has_nas:
            return
        self._alpha_optimizer, self._weight_params, self._alpha_params = (
            self.nas_helper._setup_optimizer(self.optimizer, self.model)
        )

    def _get_alpha_optimizer(self) -> torch.optim.Optimizer | None:
        return getattr(self, "_alpha_optimizer", None) or self.optimizer

    @property
    def weight_params(self) -> list[torch.nn.Parameter]:
        return self._weight_params

    @property
    def alpha_params(self) -> list[torch.nn.Parameter]:
        return self._alpha_params

    @property
    def alpha_optimizer(self) -> torch.optim.Optimizer | None:
        return getattr(self, "_alpha_optimizer", None)

    @alpha_optimizer.setter
    def alpha_optimizer(self, value: torch.optim.Optimizer | None) -> None:
        self._alpha_optimizer = value

    def train_epoch(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> tuple[float, dict[str, float]]:
        _global_step = {"step": self.global_step}
        # Pass step-level scheduler (only WarmupCosineLR for now)
        from foreblocks.training.optimization.llrd import WarmupCosineLR

        step_scheduler = (
            self.scheduler if isinstance(self.scheduler, WarmupCosineLR) else None
        )
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
            scheduler=step_scheduler,
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
        callbacks = callbacks or []
        num_epochs = epochs if epochs is not None else self.config.num_epochs

        # Reset at the run boundary so RandomSampler/DataLoader worker base
        # seeds, dropout, and other stochastic training paths are repeatable.
        self._seed_everything(
            self.config.seed,
            deterministic=self.config.deterministic,
        )

        run_context, run_name = _log.init_mltracker_run_context(
            self.mltracker, run_name
        )

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
                    # Pass step-level scheduler (only WarmupCosineLR for now)
                    from foreblocks.training.optimization.llrd import (
                        WarmupCosineLR,
                    )

                    step_scheduler = (
                        self.scheduler
                        if isinstance(self.scheduler, WarmupCosineLR)
                        else None
                    )
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
                        scheduler=step_scheduler,
                    )
                    self.global_step = _global_step["step"]

                    val_loss = self.evaluate(val_loader) if val_loader else None

                    lr = self.optimizer.param_groups[0]["lr"]
                    model_info = (
                        self.model.get_model_size()
                        if hasattr(self.model, "get_model_size")
                        else None
                    )

                    alpha_info = None
                    if (
                        getattr(self.config, "train_nas", False)
                        and self.nas_helper.has_nas
                    ):
                        alpha_info = self.nas_helper.collect_alpha_report()

                    self.history.record_epoch(
                        train_loss, val_loss, lr, components, model_info, alpha_info
                    )

                    if val_loader:
                        if (
                            val_loss + getattr(self.config, "min_delta", 0)
                            < self.best_val_loss
                        ):
                            self.best_val_loss = val_loss
                            self.epochs_without_improvement = 0
                            self.best_model_state = copy.deepcopy(
                                self.model.state_dict()
                            )
                        else:
                            self.epochs_without_improvement += 1
                        if self.epochs_without_improvement >= getattr(
                            self.config, "patience", 10
                        ):
                            print(f"\nEarly stopping at epoch {epoch + 1}")
                            completed_epochs = epoch + 1
                            stopped_early = True
                            break

                    pbar.set_postfix({"train": train_loss, "val": val_loss, "lr": lr})
                    self._step_scheduler(train_loss, val_loss)

                    for cb in callbacks:
                        if hasattr(cb, "on_epoch_end"):
                            cb.on_epoch_end(
                                self,
                                epoch,
                                {
                                    "epoch": epoch,
                                    "train_loss": train_loss,
                                    "val_loss": val_loss,
                                    "lr": lr,
                                },
                            )

                    _log.log_mltracker_metrics(
                        self.mltracker,
                        epoch,
                        train_loss,
                        lr,
                        components,
                        val_loss,
                    )
                    completed_epochs = epoch + 1

            _log.log_mltracker_final(
                self.mltracker, completed_epochs, stopped_early, self.best_val_loss
            )

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        _log.log_model_to_last_run(
            self.mltracker, self._last_run_id, self.model, model_name="model"
        )

        if (
            getattr(self.config, "conformal_enabled", False)
            and self.conformal_engine is not None
        ):
            print(
                "\n[Conformal] Engine ready. Call calibrate_conformal(cal_loader) with held-out data."
            )

        return self.history

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, dataloader: DataLoader) -> float:
        return training_loop.evaluate(
            self.model,
            dataloader,
            self.device,
            self._amp_context,
            self.moe_log,
            self.moe_meta_builder,
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
        _conf.calibrate_conformal(
            self,
            cal_loader,
            state_model,
            feature_extractor,
            jackknife_cv_models,
            jackknife_cv_indices,
            enbpi_member_models,
            enbpi_boot_indices,
        )

    def update_conformal(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        state_model: Any = None,
        feature_extractor: Any = None,
        sequential: bool = True,
    ) -> None:
        _conf.update_conformal(
            self, X_new, y_new, state_model, feature_extractor, sequential
        )

    def predict_with_intervals(
        self,
        X: torch.Tensor,
        return_tensors: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        return _conf.predict_with_intervals(self, X, return_tensors)

    def compute_coverage(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        return _conf.compute_coverage(self, X, y)

    def predict_with_intervals_streaming(
        self,
        dataloader: DataLoader,
        do_update: bool = True,
        return_numpy: bool = True,
        sequential: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _conf.predict_with_intervals_streaming(
            self, dataloader, do_update, return_numpy, sequential
        )

    def compute_coverage_streaming(
        self,
        dataloader: DataLoader,
        do_update: bool = True,
        sequential: bool | None = None,
    ) -> dict[str, Any]:
        return _conf.compute_coverage_streaming(self, dataloader, do_update, sequential)

    # ── Saving / loading ───────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        save_trainer_checkpoint(self, path)

    def load(self, path: str | Path) -> None:
        load_trainer_checkpoint(self, path)

    # ── Model utilities ────────────────────────────────────────────────

    @staticmethod
    def _infer_num_experts(model: nn.Module) -> int | None:
        for m in model.modules():
            if hasattr(m, "num_experts"):
                try:
                    ne = int(m.num_experts)
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
        evaluator = ModelEvaluator(self)
        result = evaluator.compute_metrics(
            X_val, y_val, batch_size, graph_kwargs=graph_kwargs
        )
        _log.log_to_last_run(
            self.mltracker, self._last_run_id, result, step=None, prefix="eval/"
        )
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
        evaluator = ModelEvaluator(self)
        return evaluator.cross_validation(
            X, y, n_windows, horizon, step_size, batch_size, graph_kwargs=graph_kwargs
        )

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
        _viz._require_matplotlib()
        fig = _viz.plot_prediction(
            self,
            X_val,
            y_val,
            graph_kwargs,
            full_series,
            offset,
            stride,
            figsize,
            show,
            names,
            pred_color,
            series_color,
            save_path,
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
        _viz._require_matplotlib()
        return _viz.plot_intervals(
            self,
            X_val,
            y_val,
            full_series,
            time_index,
            offset,
            stride,
            figsize,
            show,
            names,
            interval_alpha,
            pred_color,
            interval_color,
            aggregation,
            show_width_plot,
            min_count,
            do_update,
        )

    def plot_violation_heatmap_streaming(
        self,
        dataloader: DataLoader,
        feature: int = 0,
        do_update: bool = True,
        figsize: tuple[int, int] = (10, 4),
        show: bool = True,
        sequential: bool | None = None,
    ) -> plt.Figure:  # type: ignore[name-defined]
        _viz._require_matplotlib()
        return _viz.plot_violation_heatmap_streaming(
            self,
            dataloader,
            feature,
            do_update,
            figsize,
            show,
            sequential,
        )
