"""
Enhanced Trainer with NAS support for HeadComposer
"""

import contextlib
import copy
import datetime
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from foreblocks.ui.node_spec import node

# Optional: import your MoE classes and HeadComposer
try:
    from foreblocks.tf.experts.moe import FeedForwardBlock, MoEFeedForwardDMoE
    from foreblocks.tf.experts.moe_logging import (
        MoELogger,
        ReportInputs,
        build_moe_report,
    )
except Exception:
    MoELogger = None
    ReportInputs = None

    def build_moe_report(*args, **kwargs):
        raise RuntimeError("MoE logging not available")


from foreblocks.aux.utils import (
    LossComputer,
    NASHelper,
    TrainingConfig,
    TrainingHistory,
)
from foreblocks.evaluation.model_evaluator import ModelEvaluator

# ============================================================
# Conformal helpers (Trainer-side only)
# ============================================================


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_y_shape_like_intervals(y_np: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """
    Make y shape match [N,H,D] like lower/upper.
    Accepts y as [N,H], [N,H,1], [N,H,D].
    """
    if y_np.ndim == 2:
        y_np = y_np[:, :, None]  # [N,H,1]
    if y_np.ndim != 3:
        raise ValueError(f"y must have shape [N,H] or [N,H,D], got {y_np.shape}")
    if y_np.shape[-1] == 1 and lower.shape[-1] > 1:
        y_np = np.repeat(y_np, lower.shape[-1], axis=-1)
    if y_np.shape != lower.shape:
        raise ValueError(f"Shape mismatch: y {y_np.shape} vs intervals {lower.shape}")
    return y_np


def _collect_xy_from_loader(cal_loader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (X, y) from a DataLoader.

    IMPORTANT: if your dataset yields (X, y, time_feat), we intentionally ignore time_feat
    because you requested to keep using model(X) only.
    """
    Xc, Yc = [], []
    for batch in cal_loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError(
                "Calibration loader must yield (X, y) or (X, y, ...) tuples."
            )
        xb, yb = batch[0], batch[1]
        Xc.append(xb)
        Yc.append(yb)

    Xc_t = (
        torch.cat(Xc, dim=0)
        if torch.is_tensor(Xc[0])
        else torch.tensor(np.concatenate(Xc, axis=0))
    )
    Yc_t = (
        torch.cat(Yc, dim=0)
        if torch.is_tensor(Yc[0])
        else torch.tensor(np.concatenate(Yc, axis=0))
    )

    # Keep as numpy for your engine API
    return Xc_t.detach().cpu().numpy(), Yc_t.detach().cpu().numpy()


@node(
    type_id="trainer",
    name="Trainer",
    category="Training",
    inputs=["X_train", "Y_train", "model"],
    outputs=["trained_model"],
    infer=True,
)
class Trainer:
    """Trainer for time series models with optional MoE logging and NAS support"""

    Config = TrainingConfig

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[Callable] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        moe_meta_builder: Optional[
            Callable[
                [
                    torch.Tensor,
                    Optional[torch.Tensor],
                    Optional[torch.Tensor],
                    int,
                    int,
                ],
                Optional[Dict[str, Any]],
            ]
        ] = None,
        alpha_optimizer: Optional[torch.optim.Optimizer] = None,
        mltracker: Optional[Any] = None,  # Pass an existing MLTracker instance
        mltracker_uri: Optional[
            str
        ] = None,  # DB directory; defaults to <project_root>/mltracker_data
        auto_track: bool = True,  # Auto-create MLTracker when none is supplied
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_wandb = use_wandb

        # ------------------------------------------------------------------
        # Resolve the DB path to an absolute location anchored at the project
        # root (3 levels up from this file: training/ → foreblocks/ → root).
        # This ensures the same DB is used regardless of notebook CWD.
        # Priority: explicit argument > MLTRACKER_DIR env var > package-relative default
        # ------------------------------------------------------------------
        import os as _os
        from pathlib import Path as _Path

        if mltracker_uri is None:
            mltracker_uri = _os.environ.get(
                "MLTRACKER_DIR",
                str(
                    _Path(__file__).resolve().parent.parent / "mltracker/mltracker_data"
                ),
            )
            print(f"[MLTracker] Resolved mltracker_uri to: {mltracker_uri}")

        # ------------------------------------------------------------------
        # Auto-tracking: create an MLTracker from the DB path when none is
        # provided and auto_track=True.  The MLTracker writes directly to
        # SQLite — no API server required.
        # ------------------------------------------------------------------
        if mltracker is not None:
            self.mltracker = mltracker
        elif auto_track:
            try:
                from foreblocks.mltracker.mltracker import MLTracker

                self.mltracker = MLTracker(tracking_uri=mltracker_uri)
            except Exception as _mt_err:
                print(
                    f"[MLTracker] Auto-track init failed, tracking disabled: {_mt_err}"
                )
                self.mltracker = None
        else:
            self.mltracker = None
        self._mltracker_uri = mltracker_uri
        self._last_run_id: Optional[str] = None  # set each time train() starts a run

        # -----------------------------------------
        # Config init
        # -----------------------------------------
        if isinstance(config, dict):
            self.config = TrainingConfig()
            self.config.update(**config)
        else:
            self.config = config or TrainingConfig()

        # -----------------------------------------
        # NAS setup
        # -----------------------------------------
        self.nas_helper = NASHelper(self.model, self.config)

        if self.config.train_nas and self.nas_helper.has_nas:
            weight_params = self.nas_helper.get_weight_parameters()
            alpha_params = self.nas_helper.get_alpha_parameters()
            self.alpha_params = alpha_params
            self.weight_params = weight_params
            print(
                f"[NAS] Training with NAS. Found {len(alpha_params)} architecture parameters."
            )

            self.optimizer = optimizer or torch.optim.AdamW(
                weight_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            self.alpha_optimizer = alpha_optimizer or torch.optim.Adam(
                alpha_params,
                lr=self.config.nas_alpha_lr,
                weight_decay=self.config.nas_alpha_weight_decay,
            )
        else:
            self.optimizer = optimizer or self._create_optimizer()
            self.alpha_optimizer = None

        # -----------------------------------------
        # Scheduler, loss, AMP, history
        # -----------------------------------------
        self.scheduler = scheduler or self._create_scheduler()
        self.loss_computer = LossComputer(self.model, self.config, criterion)
        self.scaler = GradScaler() if self.config.use_amp else None

        self.history = TrainingHistory()
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.current_epoch = 0

        # -----------------------------------------
        # MoE logging setup
        # -----------------------------------------
        self.global_step = 0
        self.moe_log: Optional[MoELogger] = None
        self.moe_meta_builder = moe_meta_builder or self._default_moe_meta_builder

        if self.config.moe_logging and MoELogger is not None:
            self.moe_log = MoELogger()
            self._wire_moe_logger(
                self.model,
                self.moe_log,
                lambda: self.global_step,
                self.config.moe_log_latency,
            )

        # -----------------------------------------
        # NEW — Conformal Prediction (FIXED)
        # -----------------------------------------
        self.conformal_engine = None
        if self.config.conformal_enabled:
            self.conformal_engine = self._create_conformal_engine()

    # =====================================================================
    # NEW — Conformal Engine Factory (passes all parameters)
    # =====================================================================
    def _create_conformal_engine(self):
        """Create conformal engine with all method-specific parameters from config."""
        from foreblocks.core.conformal import ConformalPredictionEngine

        return ConformalPredictionEngine(
            method=getattr(self.config, "conformal_method", "split"),
            quantile=getattr(self.config, "conformal_quantile", 0.9),
            # Local method
            knn_k=getattr(self.config, "conformal_knn_k", 50),
            # Rolling/ACI
            rolling_alpha=getattr(self.config, "conformal_rolling_alpha", 0.05),
            aci_gamma=getattr(self.config, "conformal_aci_gamma", 0.01),
            # AgACI
            agaci_gammas=getattr(self.config, "conformal_agaci_gammas", None),
            # EnbPI
            enbpi_B=getattr(self.config, "conformal_enbpi_B", 20),
            enbpi_window=getattr(self.config, "conformal_enbpi_window", 500),
            # CPTC
            cptc_window=getattr(self.config, "conformal_cptc_window", 500),
            cptc_tau=getattr(self.config, "conformal_cptc_tau", 1.0),
            cptc_hard_state_filter=getattr(
                self.config, "conformal_cptc_hard_state_filter", False
            ),
            # AFOCP
            afocp_feature_dim=getattr(self.config, "conformal_afocp_feature_dim", 128),
            afocp_attn_hidden=getattr(self.config, "conformal_afocp_attn_hidden", 64),
            afocp_window=getattr(self.config, "conformal_afocp_window", 500),
            afocp_tau=getattr(self.config, "conformal_afocp_tau", 1.0),
            afocp_online_lr=getattr(self.config, "conformal_afocp_online_lr", 0.0),
            afocp_online_steps=getattr(self.config, "conformal_afocp_online_steps", 1),
        )

    @staticmethod
    def _default_moe_meta_builder(
        X: torch.Tensor,
        y: Optional[torch.Tensor],
        time_feat: Optional[torch.Tensor],
        epoch: int,
        batch_idx: int,
    ) -> Optional[Dict[str, Any]]:
        if time_feat is None:
            return None
        meta = {}
        if time_feat.dtype in (torch.int32, torch.int64) and time_feat.ndim >= 1:
            meta["hour"] = time_feat.view(-1).clamp_min(0).clamp_max(23)
        return meta or None

    def _wire_moe_logger(
        self,
        module: nn.Module,
        moe_logger: "MoELogger",
        step_getter: Callable[[], int],
        log_latency: bool,
    ):
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

    @property
    def criterion(self):
        return self.loss_computer.criterion

    @criterion.setter
    def criterion(self, value):
        self.loss_computer.criterion = value

    def _create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> Optional[Any]:
        if self.config.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        return None

    @contextlib.contextmanager
    def _amp_context(self):
        if self.config.use_amp and self.device == "cuda":
            with autocast("cuda"):
                yield
        else:
            yield

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Normalize batch formats to (X, y, time_feat)."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], None
            if len(batch) >= 1:
                y = batch[1] if len(batch) > 1 else None
                time_feat = batch[2] if len(batch) > 2 else None
                return batch[0], y, time_feat
            return batch, None, None
        return batch, None, None

    def _move_batch_to_device(
        self,
        X: Any,
        y: Optional[Any],
        time_feat: Optional[Any] = None,
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Move available tensors in a batch to the trainer device."""
        if torch.is_tensor(X):
            X = X.to(self.device)
        if torch.is_tensor(y):
            y = y.to(self.device)
        if torch.is_tensor(time_feat):
            time_feat = time_feat.to(self.device)
        return X, y, time_feat

    def _forward_pass(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        time_feat: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        aux = {}
        meta = (
            self.moe_meta_builder(X, y, time_feat, self.current_epoch, batch_idx)
            if self.moe_log is not None
            else None
        )

        if hasattr(self.model, "get_distillation_info"):
            distill_info = self.model.get_distillation_info()
            if distill_info.get("distillation_enabled", False):
                try:
                    result = self.model(
                        X,
                        y,
                        time_feat,
                        self.current_epoch,
                        return_teacher_outputs=True,
                        meta=meta,
                    )
                except TypeError:
                    result = self.model(
                        X, y, time_feat, self.current_epoch, return_teacher_outputs=True
                    )
                if isinstance(result, tuple) and len(result) == 2:
                    outputs, aux["teacher_outputs"] = result[0], result[1]
                else:
                    outputs = result[0] if isinstance(result, tuple) else result
                return outputs, aux

        try:
            result = self.model(X, y, time_feat, self.current_epoch, meta=meta)
        except TypeError:
            try:
                result = self.model(X, y, time_feat, self.current_epoch)
            except TypeError:
                result = self.model(X)

        outputs = result[0] if isinstance(result, tuple) else result
        return outputs, aux

    def _backward_step(
        self,
        loss: torch.Tensor,
        batch_idx: int,
        total_batches: int,
        optimizer: torch.optim.Optimizer,
    ):
        """Backward step with gradient accumulation."""
        loss = loss / self.config.gradient_accumulation_steps

        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            if (
                (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                or batch_idx + 1 == total_batches
            ):
                if self.config.gradient_clip_val:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_val
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (
                (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                or batch_idx + 1 == total_batches
            ):
                if self.config.gradient_clip_val:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_val
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    def _alpha_step(self, dataloader: DataLoader, num_steps: int = 1) -> float:
        assert self.alpha_optimizer is not None

        total_loss = 0.0

        weight_params = self.weight_params
        alpha_params = self.alpha_params

        # Freeze θ
        for p in weight_params:
            p.requires_grad_(False)
        for p in alpha_params:
            p.requires_grad_(True)

        prev_mode = self.model.training
        self.model.eval()

        dataloader_iter = iter(dataloader)

        try:
            for step in range(num_steps):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                X, y, time_feat = self._unpack_batch(batch)
                X, y, time_feat = self._move_batch_to_device(X, y, time_feat)

                with torch.set_grad_enabled(True):
                    with self._amp_context():
                        outputs, aux = self._forward_pass(X, y, time_feat, 0)

                        if y is None:
                            target = outputs.detach().clone()
                        else:
                            target = y
                        target.requires_grad_(False)

                        loss = self.loss_computer.compute(outputs, target, aux)

                # --- DEBUG / FAIL FAST (add inside _alpha_step right before backward) ---
                if not alpha_params:
                    raise RuntimeError(
                        "[NAS] alpha_params is empty. NASHelper did not find architecture parameters."
                    )

                # ensure at least one alpha param requires grad
                if not any(p.requires_grad for p in alpha_params):
                    names = []
                    for n, p in self.model.named_parameters():
                        if p in set(alpha_params):
                            names.append((n, p.requires_grad))
                    raise RuntimeError(
                        f"[NAS] No alpha params require grad. Found: {names[:20]}"
                    )

                # outputs/loss connectivity checks
                if not torch.is_tensor(loss):
                    raise RuntimeError(
                        f"[NAS] LossComputer.compute returned non-tensor: {type(loss)}"
                    )

                if not outputs.requires_grad:
                    # This is the key symptom: forward does not depend on alpha (since theta is frozen)
                    raise RuntimeError(
                        "[NAS] outputs.requires_grad is False during alpha step. "
                        "This means the forward pass is not connected to alpha parameters "
                        "(likely using argmax/.item()/hard indexing on alphas, or alphas unused)."
                    )

                if not loss.requires_grad:
                    raise RuntimeError(
                        "[NAS] loss.requires_grad is False during alpha step. "
                        "Either outputs are detached, LossComputer detaches internally, "
                        "or alphas are not used differentiably in forward."
                    )
                # --- end DEBUG ---

                self.alpha_optimizer.zero_grad(set_to_none=True)
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.alpha_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.alpha_optimizer.step()

                for p in weight_params:
                    p.grad = None

                total_loss += float(loss.detach())

        finally:
            self.model.train(prev_mode)
            for p in weight_params:
                p.requires_grad_(True)

        return total_loss / max(num_steps, 1)

    def train_epoch(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with optional NAS alpha optimization."""
        self.model.train()
        total_loss = 0.0
        all_components: Dict[str, List[float]] = {}

        do_nas = (
            self.config.train_nas
            and self.nas_helper.has_nas
            and self.current_epoch >= self.config.nas_warmup_epochs
            and (val_loader is not None or not self.config.nas_use_val_for_alpha)
        )

        if do_nas and self.config.nas_use_val_for_alpha:
            alpha_loss = self._alpha_step(
                val_loader, num_steps=self.config.nas_alternate_steps
            )
            all_components.setdefault("alpha_loss", []).append(alpha_loss)

        if do_nas:
            for p in self.nas_helper.get_alpha_parameters():
                p.requires_grad_(False)

        try:
            for batch_idx, batch in enumerate(train_loader):
                X, y, time_feat = self._unpack_batch(batch)
                X, y, time_feat = self._move_batch_to_device(X, y, time_feat)

                with self._amp_context():
                    outputs, aux = self._forward_pass(X, y, time_feat, batch_idx)
                    if y is not None and outputs.ndim == 4 and y.ndim == 3:
                        y = y.unsqueeze(-1)
                    target = y if y is not None else outputs.detach() * 0
                    loss = self.loss_computer.compute(outputs, target, aux)

                self._backward_step(loss, batch_idx, len(train_loader), self.optimizer)
                total_loss += float(loss.detach())
                self.global_step += 1

                for k, v in self.loss_computer.components.items():
                    all_components.setdefault(k, []).append(v)
        finally:
            if do_nas:
                for p in self.nas_helper.get_alpha_parameters():
                    p.requires_grad_(True)

        avg_components = {k: float(np.mean(v)) for k, v in all_components.items()}
        return total_loss / max(1, len(train_loader)), avg_components

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        if dataloader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in dataloader:
            X, y, _ = self._unpack_batch(batch)
            X, y, _ = self._move_batch_to_device(X, y, None)

            with self._amp_context():
                try:
                    result = self.model(X)
                except TypeError:
                    result = self.model(X, y, None, self.current_epoch)
                outputs = result[0] if isinstance(result, tuple) else result

                if y is None:
                    continue
                loss = nn.MSELoss()(outputs, y)
                bs = X.size(0)
                total_loss += float(loss) * bs
                n += bs

        return (total_loss / max(n, 1)) if n > 0 else float("nan")

    def _init_mltracker_run_context(
        self, run_name: Optional[str]
    ) -> Tuple[Any, Optional[str]]:
        run_context = contextlib.nullcontext()
        if not self.mltracker:
            return run_context, run_name

        try:
            exp_name = getattr(self.config, "experiment_name", "default_experiment")
            if not run_name:
                run_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_context = self.mltracker.run(
                experiment_name=exp_name, run_name=run_name
            )
        except Exception as e:
            print(f"[MLTracker] Warning: Failed to initialize run context: {e}")
        return run_context, run_name

    def _get_mltracker_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if hasattr(self.config, "__dict__"):
            params.update(self.config.__dict__)
        elif isinstance(self.config, dict):
            params.update(self.config)
        return params

    def _log_mltracker_params(self):
        if not self.mltracker:
            return
        try:
            self.mltracker.log_params(self._get_mltracker_params())
        except Exception as e:
            print(f"[MLTracker] Warning: Failed to log params: {e}")

    @staticmethod
    def _build_mltracker_metrics(
        train_loss: float,
        lr: float,
        components: Dict[str, float],
        val_loss: Optional[float],
    ) -> Dict[str, float]:
        metrics = {"train_loss": train_loss, "lr": lr}
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        for k, v in components.items():
            metrics[f"comp/{k}"] = float(v)
        return metrics

    def _log_mltracker_metrics(
        self,
        epoch: int,
        train_loss: float,
        lr: float,
        components: Dict[str, float],
        val_loss: Optional[float],
    ):
        if not self.mltracker:
            return
        try:
            metrics = self._build_mltracker_metrics(
                train_loss, lr, components, val_loss
            )
            self.mltracker.log_metrics(metrics, step=epoch)
        except Exception as e:
            print(f"[MLTracker] Warning: Failed to log metrics: {e}")

    def _log_mltracker_model_info(self):
        """Log model architecture metadata as params and system/git info as tags."""
        if not self.mltracker:
            return
        try:
            # Model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self.mltracker.log_params(
                {
                    "model/class": type(self.model).__name__,
                    "model/total_params": total_params,
                    "model/trainable_params": trainable_params,
                    "model/device": str(self.device),
                }
            )
        except Exception as e:
            print(f"[MLTracker] Warning: Failed to log model info: {e}")
        # System + git tags (best-effort)
        try:
            from foreblocks.mltracker.mltracker import _maybe_git_info, _sys_info

            self.mltracker.set_tags({f"sys:{k}": v for k, v in _sys_info().items()})
            git = _maybe_git_info()
            if git:
                self.mltracker.set_tags({f"git:{k}": v for k, v in git.items()})
        except Exception:
            pass

    def _log_mltracker_final(
        self,
        total_epochs: int,
        stopped_early: bool,
    ):
        """Log end-of-training summary metrics and tags."""
        if not self.mltracker:
            return
        try:
            summary: Dict[str, Any] = {"epochs_completed": total_epochs}
            if self.best_val_loss < float("inf"):
                summary["best_val_loss"] = self.best_val_loss
            self.mltracker.log_metrics(summary, step=total_epochs)
            self.mltracker.set_tags(
                {
                    "trainer/early_stopped": str(stopped_early),
                    "trainer/device": str(self.device),
                    "trainer/amp": str(self.config.use_amp),
                }
            )
        except Exception as e:
            print(f"[MLTracker] Warning: Failed to log final summary: {e}")

    def _log_to_last_run(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics to the most recently finished training run without re-opening it."""
        if not self.mltracker or not self._last_run_id:
            return
        _prev = self.mltracker._active_run
        try:
            self.mltracker._active_run = self._last_run_id
            prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
            self.mltracker.log_metrics(prefixed, step=step if step is not None else 0)
        except Exception as e:
            print(f"[MLTracker] Warning: Failed to log eval metrics: {e}")
        finally:
            self.mltracker._active_run = _prev

    def _log_model_to_last_run(self, model_name: str = "model") -> None:
        """Log model artifacts (including architecture) to the most recent run."""
        if not self.mltracker or not self._last_run_id:
            return
        _prev = self.mltracker._active_run
        try:
            self.mltracker._active_run = self._last_run_id
            self.mltracker.log_model(self.model, model_name=model_name)
        except Exception as e:
            print(f"[MLTracker] Warning: Failed to log model artifacts: {e}")
        finally:
            self.mltracker._active_run = _prev

    # =====================================================================
    # TRAIN LOOP — REMOVED AUTO-CALIBRATION (data leakage)
    # =====================================================================
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Any]] = None,
        epochs: Optional[int] = None,
        moe_report_outdir: Optional[str] = None,
        run_name: Optional[str] = None,  # New: Optional run name for MLTracker
    ) -> TrainingHistory:
        callbacks = callbacks or []
        num_epochs = epochs if epochs is not None else self.config.num_epochs

        run_context, run_name = self._init_mltracker_run_context(run_name)

        with run_context:
            # capture run id so metrics() / plot_prediction() can log back to it
            if self.mltracker and self.mltracker._active_run:
                self._last_run_id = self.mltracker._active_run
            self._log_mltracker_params()
            self._log_mltracker_model_info()

            with tqdm(range(num_epochs), desc="Training", unit="epoch") as pbar:
                for epoch in pbar:
                    self.current_epoch = epoch

                    for cb in callbacks:
                        if hasattr(cb, "on_epoch_begin"):
                            cb.on_epoch_begin(self, epoch)

                    train_loss, components = self.train_epoch(train_loader, val_loader)
                    val_loss = self.evaluate(val_loader) if val_loader else None

                    lr = self.optimizer.param_groups[0]["lr"]
                    model_info = (
                        self.model.get_model_size()
                        if hasattr(self.model, "get_model_size")
                        else None
                    )

                    alpha_info = None
                    if self.config.train_nas and self.nas_helper.has_nas:
                        alpha_info = self.nas_helper.collect_alpha_report()

                    self.history.record_epoch(
                        train_loss, val_loss, lr, components, model_info, alpha_info
                    )

                    if val_loader:
                        if val_loss + self.config.min_delta < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.epochs_without_improvement = 0
                            self.best_model_state = copy.deepcopy(
                                self.model.state_dict()
                            )
                        else:
                            self.epochs_without_improvement += 1
                        if self.epochs_without_improvement >= self.config.patience:
                            print(f"\\nEarly stopping at epoch {epoch + 1}")
                            break

                    pbar.set_postfix({"train": train_loss, "val": val_loss, "lr": lr})

                    if self.scheduler:
                        self.scheduler.step(val_loss if val_loader else train_loss)

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

                    self._log_mltracker_metrics(
                        epoch=epoch,
                        train_loss=train_loss,
                        lr=lr,
                        components=components,
                        val_loss=val_loss,
                    )

            self._log_mltracker_final(
                total_epochs=epoch + 1,
                stopped_early=self.epochs_without_improvement >= self.config.patience
                if val_loader
                else False,
            )

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        # Persist final model + architecture artifacts to the completed run.
        self._log_model_to_last_run(model_name="model")

        # NOTE: Auto-calibration REMOVED — call calibrate_conformal() manually
        # with held-out data to avoid data leakage
        if self.config.conformal_enabled and self.conformal_engine is not None:
            print(
                "\n[Conformal] Engine ready. Call calibrate_conformal(cal_loader) with held-out data."
            )

        return self.history

    # =====================================================================
    # Conformal Calibration (FIXED)
    # =====================================================================

    def calibrate_conformal(
        self,
        cal_loader,
        state_model: Optional[Callable] = None,
        feature_extractor: Optional[nn.Module] = None,
        jackknife_cv_models: Optional[Sequence[nn.Module]] = None,
        jackknife_cv_indices: Optional[Sequence[np.ndarray]] = None,
        enbpi_member_models: Optional[Sequence[nn.Module]] = None,
        enbpi_boot_indices: Optional[np.ndarray] = None,
    ):
        """
        Calibrate conformal engine with held-out calibration data.

        Contract (since we keep model(xb)):
        - self.model must support forward(X) for inference.
        - Output must be [N,H,D] (or [N,H], which engine should handle consistently).
        """
        if self.conformal_engine is None:
            raise RuntimeError("Conformal prediction not enabled in config.")
        method = self.conformal_engine.method

        # Method-specific hard requirements
        if method == "cptc" and state_model is None:
            raise ValueError("CPTC requires `state_model`.")
        if (
            method == "enbpi"
            and enbpi_member_models is not None
            and enbpi_boot_indices is None
        ):
            raise ValueError("EnbPI with member models requires `enbpi_boot_indices`.")

        # If method is AFOCP and no feature_extractor was passed, engine will create an internal one.
        if method == "afocp" and feature_extractor is None:
            warnings.warn(
                "AFOCP without a feature_extractor will use an internal (untrained) DefaultFeatureExtractor. "
                "This is valid but may be weaker than a pretrained extractor."
            )

        Xc, Yc = _collect_xy_from_loader(cal_loader)
        print(f"[Conformal] Calibrating with {len(Xc)} samples using method='{method}'")

        self.conformal_engine.calibrate(
            model=self.model,  # IMPORTANT: engine will call model(xb)
            X_cal=Xc,
            y_cal=Yc,
            device=self.device,
            batch_size=int(getattr(self.config, "batch_size", 256)),
            state_model=state_model,
            feature_extractor=feature_extractor,
            enbpi_member_models=enbpi_member_models,
            enbpi_boot_indices=enbpi_boot_indices,
            jackknife_cv_models=jackknife_cv_models,
            jackknife_cv_indices=jackknife_cv_indices,
        )

        print(
            f"[Conformal] Calibration completed. Radii shape: {self.conformal_engine.radii.shape}"
        )

    def update_conformal(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        state_model: Optional[Callable] = None,
        feature_extractor: Optional[nn.Module] = None,
        sequential: bool = True,  # Process point-by-point within batch
    ):
        """
        Online update for adaptive methods.

        Args:
            sequential: If True, update point-by-point within the batch (required for
                    ACI/AgACI correctness). If False, batch update (faster but
                    may be approximate for adaptive methods).
        """
        if self.conformal_engine is None:
            raise RuntimeError("Conformal prediction not enabled in config.")
        if self.conformal_engine.radii is None:
            raise RuntimeError(
                "Conformal engine not calibrated. Call calibrate_conformal() first."
            )

        X_np = _as_numpy(X_new)
        y_np = _as_numpy(y_new)

        method = self.conformal_engine.method

        # For adaptive methods, we should update sequentially
        if sequential and method in ("aci", "agaci", "rolling"):
            for i in range(len(X_np)):
                self.conformal_engine.update(
                    model=self.model,
                    X_new=X_np[i : i + 1],
                    y_new=y_np[i : i + 1],
                    device=self.device,
                    batch_size=1,
                    state_model=state_model,
                    feature_extractor=feature_extractor,
                )
        else:
            self.conformal_engine.update(
                model=self.model,
                X_new=X_np,
                y_new=y_np,
                device=self.device,
                batch_size=int(getattr(self.config, "batch_size", 256)),
                state_model=state_model,
                feature_extractor=feature_extractor,
            )

    def predict_with_intervals(
        self,
        X: torch.Tensor,
        return_tensors: bool = False,
    ):
        """
        Predict with conformal intervals.

        Returns:
        preds, lower, upper with shape [N,H,D] (or [N,H,1] if single target dim).
        """
        if self.conformal_engine is None:
            raise RuntimeError("Conformal prediction not enabled in config.")
        if self.conformal_engine.radii is None:
            raise RuntimeError(
                "Conformal engine not calibrated. Call calibrate_conformal() first."
            )

        X_np = _as_numpy(X)

        preds, lower, upper = self.conformal_engine.predict(
            model=self.model,  # engine calls model(xb)
            X=X_np,
            device=self.device,
            batch_size=int(getattr(self.config, "batch_size", 256)),
        )

        # Normalize to [N,H,D] at the Trainer boundary for downstream consistency
        if preds.ndim == 2:
            preds = preds[:, :, None]
            lower = lower[:, :, None]
            upper = upper[:, :, None]

        if return_tensors:
            return (
                torch.from_numpy(preds),
                torch.from_numpy(lower),
                torch.from_numpy(upper),
            )
        return preds, lower, upper

    def compute_coverage(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Empirical coverage and basic interval stats. Shape-safe.
        """
        preds, lower, upper = self.predict_with_intervals(X, return_tensors=False)

        y_np = _as_numpy(y)
        y_np = _ensure_y_shape_like_intervals(y_np, lower)

        covered = (y_np >= lower) & (y_np <= upper)
        widths = upper - lower

        return {
            "coverage": float(covered.mean()),
            "target_coverage": float(self.conformal_engine.q),
            "coverage_gap": float(covered.mean() - self.conformal_engine.q),
            "mean_interval_width": float(widths.mean()),
            "std_interval_width": float(widths.std()),
            "min_interval_width": float(widths.min()),
            "max_interval_width": float(widths.max()),
        }

    # =====================================================================
    # Saving, loading
    # =====================================================================
    def save(self, path: str):
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "history": {
                "train_losses": self.history.train_losses,
                "val_losses": self.history.val_losses,
                "learning_rates": self.history.learning_rates,
                "alpha_values": self.history.alpha_values,
            },
        }
        if self.alpha_optimizer is not None:
            save_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        # Save conformal state if calibrated
        if (
            self.conformal_engine is not None
            and self.conformal_engine.radii is not None
        ):
            save_dict["conformal_radii"] = self.conformal_engine.radii
            save_dict["conformal_method"] = self.conformal_engine.method

        torch.save(save_dict, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            "alpha_optimizer_state_dict" in checkpoint
            and self.alpha_optimizer is not None
        ):
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )
        if "config" in checkpoint:
            self.config.update(**checkpoint["config"])

        # Restore conformal state
        if "conformal_radii" in checkpoint and self.conformal_engine is not None:
            self.conformal_engine.radii = checkpoint["conformal_radii"]

    @staticmethod
    def _infer_num_experts(model: nn.Module) -> Optional[int]:
        for m in model.modules():
            if hasattr(m, "num_experts"):
                try:
                    ne = int(getattr(m, "num_experts"))
                    if ne > 0:
                        return ne
                except Exception:
                    pass
        return None

    # =====================================================================
    # Visualization
    # =====================================================================
    def plot_prediction(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        full_series: Optional[torch.Tensor] = None,
        offset: int = 0,
        stride: int = 1,
        figsize: Tuple[int, int] = (12, 4),
        show: bool = True,
        names: Optional[Union[str, list]] = None,
        pred_color: str = "orange",
        series_color: str = "blue",
    ) -> plt.Figure:
        evaluator = ModelEvaluator(self)
        predictions = evaluator.predict(X_val)
        N, H = predictions.shape[0], predictions.shape[1]
        D = predictions.shape[2] if predictions.ndim >= 3 else 1

        if full_series is not None:
            series = (
                full_series.detach().cpu().numpy()
                if isinstance(full_series, torch.Tensor)
                else full_series
            )
            if series.ndim == 1:
                series = series[:, None]
            T, S_dim = series.shape
            D_plot = S_dim
            names = names or [f"Feature {i}" for i in range(D_plot)]
            seq_len = X_val.shape[1]
            starts = offset + seq_len + np.arange(N) * stride
            coverage_end = min(T, int(starts[-1] + H)) if N > 0 else 0

            fig, axes = plt.subplots(
                D_plot, 1, figsize=(figsize[0], figsize[1] * D_plot), sharex=True
            )
            axes = np.atleast_1d(axes)
            pred_np = (
                predictions.detach().cpu().numpy()
                if isinstance(predictions, torch.Tensor)
                else predictions
            )

            for j in range(D_plot):
                ax = axes[j]
                acc = np.zeros(T)
                cnt = np.zeros(T)
                for k in range(N):
                    s = int(starts[k])
                    if s >= T:
                        continue
                    e = min(s + H, T)
                    if e > s:
                        pred_col = j if D > j else 0
                        acc[s:e] += pred_np[k, : e - s, pred_col]
                        cnt[s:e] += 1
                have = cnt > 0
                mean_pred = np.zeros(T)
                mean_pred[have] = acc[have] / cnt[have]
                x = np.arange(coverage_end)
                ax.plot(series[:coverage_end, j], label=f"Actual {names[j]}", alpha=0.8)
                if have[:coverage_end].any():
                    ax.plot(
                        x[have[:coverage_end]],
                        mean_pred[:coverage_end][have[:coverage_end]],
                        label=f"Predicted {names[j]}",
                        linestyle="--",
                        color=pred_color,
                    )
                ax.axvline(
                    offset + seq_len,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    label="First forecast",
                )
                ax.set_title(f"{names[j]}: Prediction vs Actual")
                ax.legend(loc="upper left")
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time Step")
            plt.tight_layout()
            if show:
                plt.show()
            # Save plot as artifact in the last training run
            if self.mltracker and self._last_run_id:
                try:
                    import os as _os
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp:
                        tmp_path = tmp.name
                    fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
                    _prev = self.mltracker._active_run
                    self.mltracker._active_run = self._last_run_id
                    self.mltracker.log_artifact(tmp_path, artifact_path="plots")
                    self.mltracker._active_run = _prev
                    _os.unlink(tmp_path)
                except Exception:
                    pass  # artifact logging is best-effort
            return fig

        pred_np = (
            predictions.detach().cpu().numpy()
            if isinstance(predictions, torch.Tensor)
            else predictions
        )
        y_np = (
            y_val.detach().cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
        )
        pred_mean = pred_np.mean(axis=0)
        y_mean = y_np.mean(axis=0)
        if pred_mean.ndim == 1:
            pred_mean = pred_mean[:, None]
            y_mean = y_mean[:, None]
        D_plot = pred_mean.shape[1]
        names = names or [f"Feature {i}" for i in range(D_plot)]

        fig, axes = plt.subplots(
            D_plot, 1, figsize=(figsize[0], figsize[1] * D_plot), sharex=True
        )
        axes = np.atleast_1d(axes)
        for j in range(D_plot):
            ax = axes[j]
            horizon = np.arange(len(pred_mean))
            ax.plot(
                horizon, y_mean[:, j], label=f"Actual {names[j]}", marker="o", alpha=0.7
            )
            ax.plot(
                horizon,
                pred_mean[:, j],
                label=f"Predicted {names[j]}",
                marker="s",
                linestyle="--",
                alpha=0.7,
            )
            ax.set_title(f"{names[j]}: Average Forecast")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Forecast Horizon")
        plt.tight_layout()
        if show:
            plt.show()
        # Save plot as artifact in the last training run
        if self.mltracker and self._last_run_id:
            try:
                import os as _os
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
                _prev = self.mltracker._active_run
                self.mltracker._active_run = self._last_run_id
                self.mltracker.log_artifact(tmp_path, artifact_path="plots")
                self.mltracker._active_run = _prev
                _os.unlink(tmp_path)
            except Exception as _e:
                pass  # artifact logging is best-effort
        return fig

    def metrics(
        self, X_val: torch.Tensor, y_val: torch.Tensor, batch_size: int = 256
    ) -> Dict[str, float]:
        evaluator = ModelEvaluator(self)
        result = evaluator.compute_metrics(X_val, y_val, batch_size)
        # Log each metric back to the last training run (prefixed with "eval/")
        self._log_to_last_run(result, prefix="eval/")
        return result

    def cv(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_windows: int,
        horizon: int,
        step_size: Optional[int] = None,
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        evaluator = ModelEvaluator(self)
        return evaluator.cross_validation(
            X, y, n_windows, horizon, step_size, batch_size
        )

    def plot_intervals(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        full_series: Optional[torch.Tensor] = None,
        offset: int = 0,
        stride: int = 1,
        figsize: Tuple[int, int] = (14, 5),
        show: bool = True,
        names: Optional[Union[str, list]] = None,
        interval_alpha: float = 0.25,
        pred_color: str = "blue",
        interval_color: str = "blue",
        aggregation: str = "envelope",
        show_width_plot: bool = True,
        min_count: int = 1,  # NEW: minimum overlapping windows to include a point
    ) -> plt.Figure:
        """
        Plot predictions with conformal intervals.

        Args:
            aggregation: How to aggregate overlapping intervals
                - "envelope": Use min(lower), max(upper) - most conservative
                - "mean": Average all overlapping intervals
                - "last": Use the most recent prediction for each point
                - "min_width": Use the interval with smallest width
            show_width_plot: If True, add subplot showing interval widths over time
            min_count: Minimum number of overlapping windows required to plot a point.
                       Set higher (e.g., H//2) to remove edge effects at forecast boundaries.
        """
        if self.conformal_engine is None or self.conformal_engine.radii is None:
            raise RuntimeError(
                "Conformal engine not calibrated. Call calibrate_conformal() first."
            )

        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=256, shuffle=False
        )
        # Get predictions with intervals
        preds, lower, upper, y_stream = self.predict_with_intervals_streaming(
            val_loader
        )

        N, H, D = preds.shape
        seq_len = X_val.shape[1]

        if full_series is None:
            raise ValueError("full_series must be provided for time-aligned plotting.")

        # Convert series
        series = (
            full_series.detach().cpu().numpy()
            if isinstance(full_series, torch.Tensor)
            else full_series
        )
        if series.ndim == 1:
            series = series[:, None]

        T, S_dim = series.shape
        D_plot = min(D, S_dim)
        names = names or [f"Feature {i}" for i in range(D_plot)]

        # Rolling alignment
        starts = offset + seq_len + np.arange(N) * stride
        coverage_end = min(int(starts[-1] + H), T)

        # Track count for all aggregation methods (needed for min_count filter)
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
                    agg_low[start:end, j] = np.minimum(
                        agg_low[start:end, j], lower[k, :h, pred_col]
                    )
                    agg_up[start:end, j] = np.maximum(
                        agg_up[start:end, j], upper[k, :h, pred_col]
                    )
                count[start:end] += 1

            have = count >= min_count  # CHANGED: use min_count threshold
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

            have = (~np.isnan(mean_pred[:, 0])) & (count >= min_count)  # CHANGED

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

            have = (~np.isnan(mean_pred[:, 0])) & (count >= min_count)  # CHANGED

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

            have = count >= min_count  # CHANGED: use min_count threshold
            mean_pred = np.zeros_like(acc_pred)
            mean_low = np.zeros_like(acc_low)
            mean_up = np.zeros_like(acc_up)

            for j in range(D_plot):
                mean_pred[have, j] = acc_pred[have, j] / count[have]
                mean_low[have, j] = acc_low[have, j] / count[have]
                mean_up[have, j] = acc_up[have, j] / count[have]

        # Compute interval widths for the width subplot
        interval_widths = mean_up - mean_low

        # Determine subplot layout
        n_rows = D_plot + (1 if show_width_plot else 0)

        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(figsize[0], figsize[1] * n_rows),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        xs = np.arange(coverage_end)

        # Plot each feature
        for j in range(D_plot):
            ax = axes[j]

            # Actual series
            ax.plot(
                xs,
                series[:coverage_end, j],
                label=f"Actual {names[j]}",
                alpha=0.8,
                linewidth=1,
            )

            # Predictions + intervals
            mask = have[:coverage_end]

            if mask.any():
                yp = mean_pred[:coverage_end, j]
                yl = mean_low[:coverage_end, j]
                yu = mean_up[:coverage_end, j]

                ax.plot(
                    xs[mask],
                    yp[mask],
                    label=f"Predicted {names[j]}",
                    linestyle="--",
                    color=pred_color,
                    linewidth=1,
                )

                ax.fill_between(
                    xs[mask],
                    yl[mask],
                    yu[mask],
                    color=interval_color,
                    alpha=interval_alpha,
                    label=f"Interval ({aggregation})",
                )

            ax.axvline(
                offset + seq_len,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="First forecast",
            )

            ax.set_title(f"{names[j]} — Forecast with Conformal Intervals")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Add interval width subplot
        if show_width_plot:
            ax_width = axes[-1]

            for j in range(D_plot):
                mask = have[:coverage_end]
                widths_j = interval_widths[:coverage_end, j]
                ax_width.plot(
                    xs[mask],
                    widths_j[mask],
                    label=f"Width {names[j]}",
                    alpha=0.8,
                    linewidth=1,
                )

            ax_width.axvline(
                offset + seq_len,
                color="gray",
                linestyle="--",
                alpha=0.5,
            )

            ax_width.set_ylabel("Interval Width")
            ax_width.set_title("Adaptive Interval Widths Over Time")
            ax_width.legend(loc="upper left", fontsize=8)
            ax_width.grid(True, alpha=0.3)

            # Add statistics annotation (only for valid points)
            valid_widths = interval_widths[have]
            if valid_widths.size > 0:
                stats_text = f"Width: μ={valid_widths.mean():.2f}, σ={valid_widths.std():.2f}, CV={valid_widths.std() / valid_widths.mean():.2f}"
                ax_width.text(
                    0.98,
                    0.95,
                    stats_text,
                    transform=ax_width.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def predict_with_intervals_streaming(
        self,
        dataloader: DataLoader,
        do_update: bool = True,
        return_numpy: bool = True,
        sequential: Optional[bool] = None,  # ADD THIS
    ):
        """
        Streaming (rolling) prediction over a DataLoader.

        - Assumes dataloader yields (X, y) or (X, y, ...)
        - Predicts intervals batch-by-batch in chronological order (so dataloader must have shuffle=False).
        - If do_update=True, performs online update AFTER predicting each batch (leakage-safe).
        - If sequential=None, auto-enables for ACI methods (rolling/agaci).

        Returns:
            preds, lower, upper, y_true   (all concatenated, shapes [N,H,D])
        """
        if self.conformal_engine is None or self.conformal_engine.radii is None:
            raise RuntimeError(
                "Conformal engine not calibrated. Call calibrate_conformal() first."
            )

        self.model.eval()

        preds_all, low_all, up_all, y_all = [], [], [], []

        for batch in dataloader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Dataloader must yield (X, y) or (X, y, ...) tuples.")
            xb, yb = batch[0], batch[1]

            # 1) Predict intervals (no leakage)
            preds_b, low_b, up_b = self.predict_with_intervals(xb, return_tensors=False)

            preds_all.append(preds_b)
            low_all.append(low_b)
            up_all.append(up_b)

            y_np = _as_numpy(yb)
            if y_np.ndim == 2:
                y_np = y_np[:, :, None]
            y_all.append(y_np)

            # 2) Online update AFTER observing label (this is the "rolling" part)
            if do_update:
                self.update_conformal(xb, yb, sequential=sequential)

        preds = np.concatenate(preds_all, axis=0)
        low = np.concatenate(low_all, axis=0)
        up = np.concatenate(up_all, axis=0)
        y_true = np.concatenate(y_all, axis=0)

        if return_numpy:
            return preds, low, up, y_true
        else:
            return (
                torch.from_numpy(preds),
                torch.from_numpy(low),
                torch.from_numpy(up),
                torch.from_numpy(y_true),
            )

    def plot_violation_heatmap_streaming(
        self,
        dataloader: DataLoader,
        feature: int = 0,
        do_update: bool = True,
        figsize: Tuple[int, int] = (10, 4),
        show: bool = True,
        sequential: Optional[bool] = None,
    ) -> plt.Figure:
        """
        Heatmap of conformal misses (outside interval) for streaming/rolling evaluation.

        Rows: window index n (batch-concatenated sample index)
        Cols: horizon h
        Value: 1 if y_true is outside [L, U], else 0

        IMPORTANT:
        - dataloader must have shuffle=False
        - do_update=True makes it a true rolling (online) conformal evaluation
        - sequential=None auto-enables for ACI methods
        """

        if self.conformal_engine is None or self.conformal_engine.radii is None:
            raise RuntimeError(
                "Conformal engine not calibrated. Call calibrate_conformal() first."
            )

        preds, L, U, y_true = self.predict_with_intervals_streaming(
            dataloader=dataloader,
            do_update=do_update,
            return_numpy=True,
            sequential=sequential,
        )

        N, H, D = L.shape
        j = int(feature)
        if j < 0 or j >= D:
            raise ValueError(f"feature index out of range: {j} (D={D})")

        miss = (y_true[:, :, j] < L[:, :, j]) | (y_true[:, :, j] > U[:, :, j])

        # Compute coverage statistics
        overall_coverage = 1.0 - miss.mean()
        per_horizon_coverage = 1.0 - miss.mean(axis=0)

        fig, ax = plt.subplots(figsize=figsize)

        # Binary colormap: green for covered (0), red for miss (1)
        binary_cmap = ListedColormap(["white", "black"])  # white=covered, black=miss

        im = ax.imshow(
            miss.astype(float),
            aspect="auto",
            interpolation="nearest",
            cmap=binary_cmap,
            vmin=0,
            vmax=1,
        )

        ax.set_xlabel("Horizon")
        ax.set_ylabel("Window index (stream order)")

        target = self.conformal_engine.q
        ax.set_title(f"Conformal Misses — feature={j}, (target={target:.0%})")

        ax.set_xticks(np.arange(H))
        ax.set_xticklabels([str(h + 1) for h in range(H)])

        # Binary colorbar with discrete ticks and labels
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0.25, 0.75])  # Center of each color band
        cbar.set_ticklabels(["Covered", "Miss"])

        # Add per-horizon coverage annotation at bottom
        # coverage_text = "  ".join([f"h{h+1}:{c:.0%}" for h, c in enumerate(per_horizon_coverage)])
        # fig.text(
        #     0.5, 0.02,
        #     f"Per-horizon coverage: {coverage_text}",
        #     ha='center',
        #     fontsize=9,
        #     style='italic',
        # )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Make room for coverage text

        if show:
            plt.show()
        return fig
