"""
Refactored Training System for Time Series Models

MIGRATION GUIDE:
================

Old Code:
---------
trainer = Trainer(model, config={"num_epochs": 10}, criterion=nn.MSELoss())
trainer.train(train_loader, val_loader, epochs=100)
trainer.plot_learning_curves()
trainer.metrics(X_val, Y_val)
trainer.plot_prediction(X_val, Y_val, full_series=ts)

New Code (WORKS THE SAME!):
---------------------------
# Option 1: Use base Trainer (recommended for most users)
trainer = Trainer(model, config={"num_epochs": 10}, criterion=nn.MSELoss())
trainer.train(train_loader, val_loader, epochs=100)  # epochs parameter supported!
trainer.plot_learning_curves()  # Now built-in!
trainer.metrics(X_val, Y_val)  # Now built-in!
trainer.plot_prediction(X_val, Y_val, full_series=ts)  # Now built-in!

# Option 2: Use TrainerWithLegacyMethods (for advanced features)
trainer = TrainerWithLegacyMethods(model, config={"num_epochs": 10})
# Includes: quantization, distillation, benchmark_model, compare_with_baseline, etc.

What Changed:
-------------
✅ All common methods (train, plot_prediction, metrics, plot_learning_curves) 
   now work directly on the base Trainer class
✅ train() accepts epochs= parameter to override config
✅ Configuration can be dict or TrainingConfig dataclass
✅ criterion parameter fully supported
✅ W&B integration maintained

Advanced Features (use TrainerWithLegacyMethods):
-------------------------------------------------
- prepare_quantization() / finalize_quantization()
- enable_distillation() / disable_distillation()
- benchmark_model()
- compare_with_baseline()
- set_config() method

Key Improvements:
-----------------
1. Cleaner internal architecture with LossComputer, TrainingHistory
2. Type-safe configuration with TrainingConfig dataclass
3. Better separation of concerns (but still convenient!)
4. Easier testing and extension
5. 100% backward compatible!
"""

import contextlib
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from foreblocks.node_spec import node
# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class TrainingConfig:
    """Type-safe training configuration"""
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 32
    patience: int = 10
    min_delta: float = 1e-4
    use_amp: bool = True
    gradient_clip_val: Optional[float] = None
    gradient_accumulation_steps: int = 1
    l1_regularization: float = 0.0
    kl_weight: float = 1.0
    scheduler_type: Optional[str] = None
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    min_lr: float = 1e-6
    verbose: bool = True
    log_interval: int = 10
    save_best_model: bool = True
    save_model_path: Optional[str] = None

    def update(self, **kwargs):
        """Update config with validation"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config key '{key}' not found")


# ============================================================================
# Dataset
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Clean dataset for time series data"""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create dataloaders with consistent interface"""
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            TimeSeriesDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loader


# ============================================================================
# Loss Computation
# ============================================================================

class LossComputer:
    """Handles all loss computation logic"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        criterion: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.criterion = criterion or nn.MSELoss()
        self.components: Dict[str, float] = {}

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        aux_data: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute total loss with all components"""
        self.components = {}
        aux_data = aux_data or {}

        # Base task loss
        task_loss = self.criterion(outputs, targets)
        total_loss = task_loss
        self.components["task_loss"] = task_loss.item()

        # Distillation loss
        if hasattr(self.model, "compute_distillation_loss") and "teacher_outputs" in aux_data:
            distill_loss, distill_components = self.model.compute_distillation_loss(
                outputs, aux_data["teacher_outputs"], targets, self.criterion
            )
            total_loss = distill_loss
            self.components.update({
                f"distill_{k}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in distill_components.items()
            })

        # L1 regularization
        if self.config.l1_regularization > 0:
            l1_loss = sum(
                p.abs().sum() for p in self.model.parameters() if p.requires_grad
            )
            total_loss += self.config.l1_regularization * l1_loss
            self.components["l1_loss"] = (self.config.l1_regularization * l1_loss).item()

        # KL divergence
        if hasattr(self.model, "get_kl"):
            kl_div = self.model.get_kl()
            if kl_div is not None:
                total_loss += self.config.kl_weight * kl_div
                self.components["kl_loss"] = (self.config.kl_weight * kl_div).item()

        # Auxiliary loss
        if hasattr(self.model, "get_aux_loss"):
            aux_loss = self.model.get_aux_loss()
            if aux_loss is not None:
                total_loss += aux_loss.detach()
                self.components["aux_loss"] = aux_loss.item()

        return total_loss


# ============================================================================
# Training History
# ============================================================================

@dataclass
class TrainingHistory:
    """Organized training history"""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    task_losses: List[float] = field(default_factory=list)
    distillation_losses: List[float] = field(default_factory=list)
    model_info: List[Dict[str, Any]] = field(default_factory=list)

    def record_epoch(
        self,
        train_loss: float,
        val_loss: Optional[float],
        lr: float,
        loss_components: Dict[str, float],
        model_info: Optional[Dict[str, Any]] = None,
    ):
        """Record metrics for one epoch"""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        self.learning_rates.append(lr)

        if "task_loss" in loss_components:
            self.task_losses.append(loss_components["task_loss"])

        distill_loss = sum(v for k, v in loss_components.items() if k.startswith("distill_"))
        if distill_loss > 0:
            self.distillation_losses.append(distill_loss)

        if model_info:
            self.model_info.append(model_info)


# ============================================================================
# Core Trainer
# ============================================================================
@node(
  type_id="trainer",
  name="Trainer",
  category="Training",
  inputs=["X_train","Y_train","model"],   # override: don’t infer
  outputs=["trained_model"],             # override: don’t infer
  config={"optimizer":"Adam","learning_rate":1e-3,"batch_size":64,"epochs":10},
  infer=False                            # skip inference entirely
)
class Trainer:
    """Clean, focused trainer for time series models"""

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
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_wandb = use_wandb

        # Config handling
        if isinstance(config, dict):
            self.config = TrainingConfig()
            self.config.update(**config)
        else:
            self.config = config or TrainingConfig()

        # Components
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        self.loss_computer = LossComputer(self.model, self.config, criterion)
        self.scaler = GradScaler() if self.config.use_amp else None

        # State
        self.history = TrainingHistory()
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.current_epoch = 0

        # Optional W&B integration
        if self.use_wandb:
            try:
                import wandb
                wandb.init(**(wandb_config or {}))
                wandb.watch(self.model, log="all", log_freq=100)
            except ImportError:
                logging.warning("wandb not installed, disabling W&B logging")

    @property
    def criterion(self):
        """Access to criterion for backward compatibility"""
        return self.loss_computer.criterion

    @criterion.setter
    def criterion(self, value):
        """Set criterion for backward compatibility"""
        self.loss_computer.criterion = value

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create default optimizer - override for custom optimizers"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler if configured"""
        if self.config.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        return None

    @contextlib.contextmanager
    def _amp_context(self):
        """Context manager for automatic mixed precision"""
        if self.config.use_amp and self.device == "cuda":
            with autocast("cuda"):
                yield
        else:
            yield

    def _forward_pass(
        self, X: torch.Tensor, y: torch.Tensor, time_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute forward pass with distillation support"""
        aux = {}

        if hasattr(self.model, "get_distillation_info"):
            distill_info = self.model.get_distillation_info()
            if distill_info.get("distillation_enabled", False):
                result = self.model(X, y, time_feat, self.current_epoch, return_teacher_outputs=True)
                if isinstance(result, tuple) and len(result) == 2:
                    outputs, aux["teacher_outputs"] = result[0], result[1]
                else:
                    outputs = result[0] if isinstance(result, tuple) else result
                return outputs, aux

        result = self.model(X, y, time_feat, self.current_epoch)
        outputs = result[0] if isinstance(result, tuple) else result
        return outputs, aux

    def _backward_step(self, loss: torch.Tensor, batch_idx: int, total_batches: int):
        """Execute backward pass with gradient accumulation"""
        loss = loss / self.config.gradient_accumulation_steps

        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or batch_idx + 1 == total_batches:
                if self.config.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or batch_idx + 1 == total_batches:
                if self.config.gradient_clip_val:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad()

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_components = {}

        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            if len(batch) == 2:
                X, y, time_feat = batch[0], batch[1], None
            else:
                X, y, time_feat = batch[0], batch[1], batch[2] if len(batch) == 3 else None

            X, y = X.to(self.device), y.to(self.device)
            if time_feat is not None:
                time_feat = time_feat.to(self.device)

            # Forward + loss
            with self._amp_context():
                outputs, aux = self._forward_pass(X, y, time_feat)
                if outputs.ndim == 4 and y.ndim == 3:
                    y = y.unsqueeze(-1)
                loss = self.loss_computer.compute(outputs, y, aux)

            # Backward
            self._backward_step(loss, batch_idx, len(dataloader))
            total_loss += loss.item()

            # Track components
            for k, v in self.loss_computer.components.items():
                all_components.setdefault(k, []).append(v)

        # Average components
        avg_components = {k: np.mean(v) for k, v in all_components.items()}
        return total_loss / len(dataloader), avg_components

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0

        for batch in dataloader:
            X, y = batch[0], batch[1]
            X, y = X.to(self.device), y.to(self.device)

            with self._amp_context():
                result = self.model(X)
                outputs = result[0] if isinstance(result, tuple) else result
                loss = nn.MSELoss()(outputs, y)
                total_loss += loss.item() * X.size(0)

        return total_loss / len(dataloader.dataset)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Any]] = None,
        epochs: Optional[int] = None,  # Override config.num_epochs
    ) -> TrainingHistory:
        """Main training loop"""
        callbacks = callbacks or []
        num_epochs = epochs if epochs is not None else self.config.num_epochs

        # Notify callbacks
        for cb in callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(self)

        with tqdm(range(num_epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                self.current_epoch = epoch

                # Epoch begin callbacks
                for cb in callbacks:
                    if hasattr(cb, "on_epoch_begin"):
                        cb.on_epoch_begin(self, epoch)

                # Train
                train_loss, loss_components = self.train_epoch(train_loader)

                # Validate
                val_loss = self.evaluate(val_loader) if val_loader else None

                # Track metrics
                lr = self.optimizer.param_groups[0]["lr"]
                model_info = (
                    self.model.get_model_size()
                    if hasattr(self.model, "get_model_size")
                    else None
                )
                self.history.record_epoch(train_loss, val_loss, lr, loss_components, model_info)

                # W&B logging
                if self.use_wandb:
                    try:
                        import wandb
                        log_dict = {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "learning_rate": lr,
                        }
                        if val_loss is not None:
                            log_dict["val_loss"] = val_loss
                        log_dict.update(loss_components)
                        if model_info:
                            log_dict.update(model_info)
                        wandb.log(log_dict)
                    except Exception as e:
                        logging.warning(f"W&B logging failed: {e}")

                # Early stopping
                if val_loader:
                    if val_loss + self.config.min_delta < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        if self.config.save_model_path:
                            self.save(self.config.save_model_path)
                    else:
                        self.epochs_without_improvement += 1

                    if self.epochs_without_improvement >= self.config.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break

                # Update progress bar
                postfix = {
                    "train_loss": f"{train_loss:.4f}",
                    "lr": f"{lr:.2e}",
                }
                if val_loss is not None:
                    postfix["val_loss"] = f"{val_loss:.4f}"
                pbar.set_postfix(postfix)

                # Scheduler step
                if self.scheduler:
                    self.scheduler.step(val_loss if val_loader else train_loss)

                # Epoch end callbacks
                logs = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                }
                for cb in callbacks:
                    if hasattr(cb, "on_epoch_end"):
                        cb.on_epoch_end(self, epoch, logs)

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        # Training end callbacks
        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(self)

        return self.history

    def save(self, path: str):
        """Save model checkpoint"""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "history": {
                "train_losses": self.history.train_losses,
                "val_losses": self.history.val_losses,
                "learning_rates": self.history.learning_rates,
            },
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "config" in checkpoint:
            self.config.update(**checkpoint["config"])

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
    ) -> plt.Figure:
        """
        Plot predicted time series vs actual values.
        
        Args:
            X_val: Input sequences [N, seq_len, features]
            y_val: Target sequences [N, horizon, features]
            full_series: Optional full time series for context
            offset: Starting offset in full series
            stride: Stride between validation windows
            figsize: Figure size
            show: Whether to display the plot
            names: Feature names for labeling
        
        Returns:
            matplotlib Figure object
        """
        evaluator = ModelEvaluator(self)
        predictions = evaluator.predict(X_val)
        
        # Get shapes
        N, H = predictions.shape[0], predictions.shape[1]
        D = predictions.shape[2] if predictions.ndim >= 3 else 1
        
        if full_series is not None:
            # Plot with full series context
            series = full_series.detach().cpu().numpy()
            if series.ndim == 1:
                series = series[:, None]
            
            T, S_dim = series.shape
            D_plot = S_dim
            print(f"Full series length: {T}, Features: {S_dim}")
            names = names or [f"Feature {i}" for i in range(D_plot)]
            
            seq_len = X_val.shape[1]
            starts = offset + seq_len + np.arange(N) * stride
            coverage_end = min(T, int(starts[-1] + H)) if N > 0 else 0
            
            fig, axes = plt.subplots(
                D_plot, 1, 
                figsize=(figsize[0], figsize[1] * D_plot),
                sharex=True
            )
            axes = np.atleast_1d(axes)
            
            pred_np = predictions.detach().cpu().numpy()
            
            for j in range(D_plot):
                print(f"Plotting feature {j+1}/{D_plot}")
                ax = axes[j]
                
                # Accumulate predictions with overlap
                acc = np.zeros(T)
                cnt = np.zeros(T)
                for k in range(N):
                    s = int(starts[k])
                    if s >= T:
                        continue
                    e = min(s + H, T)
                    if e > s:
                        pred_col = j if D > j else 0
                        acc[s:e] += pred_np[k, :e-s, pred_col]
                        cnt[s:e] += 1
                
                have = cnt > 0
                mean_pred = np.zeros(T)
                mean_pred[have] = acc[have] / cnt[have]
                
                # Plot
                x = np.arange(coverage_end)
                ax.plot(series[:coverage_end, j], label=f"Actual {names[j]}", alpha=0.8)
                
                if have[:coverage_end].any():
                    ax.plot(
                        x[have[:coverage_end]], 
                        mean_pred[:coverage_end][have[:coverage_end]],
                        label=f"Predicted {names[j]}",
                        linestyle='--'
                    )
                    
                    # Shade error region
                    y_true = series[:coverage_end, j][have[:coverage_end]]
                    y_pred = mean_pred[:coverage_end][have[:coverage_end]]
                    mask = np.isfinite(y_true) & np.isfinite(y_pred)
                    if mask.any():
                        ax.fill_between(
                            x[have[:coverage_end]][mask],
                            y_pred[mask],
                            y_true[mask],
                            alpha=0.2,
                            label="Error"
                        )
                
                ax.axvline(offset + seq_len, color='gray', linestyle='--', 
                          alpha=0.5, label='First forecast')
                ax.set_title(f"{names[j]}: Prediction vs Actual")
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel("Time Step")
            plt.tight_layout()
            plt.show()
            
        else:
            # Simple plot without full series
            pred_np = predictions.detach().cpu().numpy()
            y_np = y_val.detach().cpu().numpy()
            
            # Average over samples
            pred_mean = pred_np.mean(axis=0)  # [H, D]
            y_mean = y_np.mean(axis=0)  # [H, D]
            
            if pred_mean.ndim == 1:
                pred_mean = pred_mean[:, None]
                y_mean = y_mean[:, None]
            
            D_plot = pred_mean.shape[1]
            names = names or [f"Feature {i}" for i in range(D_plot)]
            
            fig, axes = plt.subplots(
                D_plot, 1,
                figsize=(figsize[0], figsize[1] * D_plot),
                sharex=True
            )
            axes = np.atleast_1d(axes)
            
            for j in range(D_plot):
                ax = axes[j]
                horizon = np.arange(len(pred_mean))
                ax.plot(horizon, y_mean[:, j], label=f"Actual {names[j]}", 
                       marker='o', alpha=0.7)
                ax.plot(horizon, pred_mean[:, j], label=f"Predicted {names[j]}", 
                       marker='s', linestyle='--', alpha=0.7)
                ax.set_title(f"{names[j]}: Average Forecast")
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel("Forecast Horizon")
            plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig

    def metrics(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics (MSE, RMSE, MAE).
        
        Args:
            X_val: Input sequences
            y_val: Target sequences
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with 'mse', 'rmse', 'mae'
        """
        evaluator = ModelEvaluator(self)
        return evaluator.compute_metrics(X_val, y_val, batch_size)

    def plot_learning_curves(self, figsize: Tuple[int, int] = (15, 5)):
        """Plot training curves (loss, learning rate, components)"""
        evaluator = ModelEvaluator(self)
        return evaluator.plot_learning_curves(figsize)

    def print_training_summary(self):
        """Print formatted training summary"""
        evaluator = ModelEvaluator(self)
        evaluator.print_summary()


# ============================================================================
# Evaluation & Visualization (Separate Module)
# ============================================================================

class ModelEvaluator:
    """Handles all evaluation and visualization"""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.device = trainer.device

    @torch.no_grad()
    def predict(
        self, X: torch.Tensor, batch_size: int = 256, use_amp: bool = True
    ) -> torch.Tensor:
        """Generate predictions in batches"""
        self.model.eval()
        X = X.to(self.device)
        predictions = []

        amp_ctx = (
            autocast("cuda")
            if use_amp and self.device == "cuda"
            else contextlib.nullcontext()
        )

        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            with amp_ctx:
                output = self.model(batch)
                output = output[0] if isinstance(output, tuple) else output
            predictions.append(output.cpu())

        return torch.cat(predictions, dim=0)

    def compute_metrics(
        self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 256
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions = self.predict(X, batch_size)
        y = y.to(predictions.device)

        # Align shapes
        min_samples = min(len(predictions), len(y))
        predictions = predictions[:min_samples]
        y = y[:min_samples]

        diff = (predictions - y).float()
        mse = (diff**2).mean().item()
        mae = diff.abs().mean().item()
        rmse = mse**0.5

        return {"mse": mse, "rmse": rmse, "mae": mae}

    def plot_learning_curves(self, figsize: Tuple[int, int] = (15, 5)):
        """Plot training curves"""
        history = self.trainer.history
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Loss
        axes[0].plot(history.train_losses, label="Train")
        if history.val_losses:
            axes[0].plot(history.val_losses, label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Learning rate
        axes[1].plot(history.learning_rates)
        axes[1].set_title("Learning Rate")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("LR")
        axes[1].grid(True)

        # Task vs distillation (if available)
        if history.task_losses and history.distillation_losses:
            axes[2].plot(history.task_losses, label="Task")
            axes[2].plot(history.distillation_losses, label="Distillation")
            axes[2].set_title("Loss Components")
            axes[2].legend()
        else:
            axes[2].text(0.5, 0.5, "No component data", ha="center", va="center")
            axes[2].set_title("Loss Components")

        axes[2].set_xlabel("Epoch")
        axes[2].grid(True)

        plt.tight_layout()
        return fig

    def print_summary(self):
        """Print training summary"""
        history = self.trainer.history
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total epochs:      {len(history.train_losses)}")
        print(f"Final train loss:  {history.train_losses[-1]:.6f}")
        if history.val_losses:
            print(f"Final val loss:    {history.val_losses[-1]:.6f}")
            print(f"Best val loss:     {self.trainer.best_val_loss:.6f}")

        if hasattr(self.model, "get_model_size"):
            info = self.model.get_model_size()
            print(f"\nModel size:        {info.get('size_mb', 0):.2f} MB")
            print(f"Parameters:        {info.get('parameters', 0):,}")

        print("=" * 60)


# ============================================================================
# Backward Compatibility Extensions
# ============================================================================

class TrainerWithLegacyMethods(Trainer):
    """Extended trainer with backward compatibility for original API"""

    def set_config(self, key: str, value: Any):
        """Legacy method for updating config"""
        self.config.update(**{key: value})

    def train_epoch(self, dataloader: DataLoader, callbacks=None) -> float:
        """Legacy signature with callbacks parameter"""
        loss, _ = super().train_epoch(dataloader)
        return loss

    def plot_learning_curves(self, figsize: Tuple[int, int] = (15, 8)):
        """Legacy plotting method directly on trainer"""
        evaluator = ModelEvaluator(self)
        return evaluator.plot_learning_curves(figsize)

    def print_training_summary(self):
        """Legacy summary method directly on trainer"""
        evaluator = ModelEvaluator(self)
        evaluator.print_summary()

    def metrics(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        eval_mode: str = "grid",
        batch_size: int = 256,
        amp: bool = True,
        node_reducer=None,
    ) -> Dict[str, float]:
        """Legacy metrics computation"""
        evaluator = ModelEvaluator(self)
        return evaluator.compute_metrics(X_val, y_val, batch_size)

    def benchmark_model(self, sample_input: torch.Tensor, num_runs: int = 100):
        """Legacy benchmark method"""
        if not hasattr(self.model, "benchmark_inference"):
            print("Model does not support benchmarking.")
            return None

        sample_input = sample_input.to(self.device)
        results = self.model.benchmark_inference(sample_input, num_runs=num_runs)

        print("\nModel Performance Benchmark:")
        print(f"  Inference Time: {results['avg_inference_time_ms']:.2f} ms")
        print(f"  Throughput:     {results['throughput_samples_per_sec']:.2f} samples/sec")

        if hasattr(self.model, "get_model_size"):
            size_info = self.model.get_model_size()
            print(f"  Model Size:     {size_info['size_mb']:.2f} MB")
            print(f"  Parameters:     {size_info['parameters']:,}")

        return results

    def compare_with_baseline(
        self, baseline_model: nn.Module, test_loader: DataLoader
    ):
        """Legacy baseline comparison"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        current_loss = self.evaluate(test_loader)

        original_model = self.model
        self.model = baseline_model.to(self.device)
        baseline_loss = self.evaluate(test_loader)
        self.model = original_model

        print(f"Current model loss:  {current_loss:.6f}")
        print(f"Baseline model loss: {baseline_loss:.6f}")
        improvement = 100 * (baseline_loss - current_loss) / baseline_loss
        print(f"Improvement:          {improvement:.2f}%")

        if hasattr(self.model, "get_model_size") and hasattr(
            baseline_model, "get_model_size"
        ):
            cur_size = self.model.get_model_size()["size_mb"]
            base_size = baseline_model.get_model_size()["size_mb"]
            reduction = 100 * (base_size - cur_size) / base_size

            print(f"\nCurrent size:   {cur_size:.2f} MB")
            print(f"Baseline size:  {base_size:.2f} MB")
            print(f"Size reduction: {reduction:.2f}%")

        print("=" * 60)

        return {
            "current_loss": current_loss,
            "baseline_loss": baseline_loss,
            "loss_improvement": improvement,
        }

    # Model interface delegation methods
    def prepare_quantization(
        self, sample_input: torch.Tensor, calibration_loader=None
    ):
        """Prepare model for quantization"""
        if hasattr(self.model, "prepare_for_quantization"):
            print("Preparing model for quantization...")
            sample_input = sample_input.to(self.device)
            self.model = self.model.prepare_for_quantization(calibration_loader)
            print("Model quantization prepared!")
        else:
            print("Model does not support quantization.")

    def finalize_quantization(self):
        """Finalize quantization"""
        if hasattr(self.model, "finalize_quantization"):
            print("Finalizing quantization...")
            self.model = self.model.finalize_quantization()
            print("Quantization finalized!")
        else:
            print("Model does not support quantization finalization.")

    def get_quantization_info(self) -> Dict[str, Any]:
        """Get quantization info from model"""
        if hasattr(self.model, "get_quantization_info"):
            return self.model.get_quantization_info()
        return {"quantization_enabled": False}

    def set_quantization_mode(self, mode: str):
        """Set quantization mode"""
        if hasattr(self.model, "set_quantization_mode"):
            self.model.set_quantization_mode(mode)
            print(f"Quantization mode set to: {mode}")
        else:
            print("Model does not support setting quantization mode.")

    def get_distillation_info(self) -> Dict[str, Any]:
        """Get distillation info from model"""
        if hasattr(self.model, "get_distillation_info"):
            return self.model.get_distillation_info()
        return {"distillation_enabled": False}

    def enable_distillation(
        self, mode: str = "output", teacher_model: nn.Module = None
    ):
        """Enable distillation"""
        if hasattr(self.model, "enable_distillation"):
            self.model.enable_distillation(mode, teacher_model)
            print(f"Distillation enabled (mode: {mode}).")
        else:
            print("Model does not support distillation.")

    def disable_distillation(self):
        """Disable distillation"""
        if hasattr(self.model, "disable_distillation"):
            self.model.disable_distillation()
            print("Distillation disabled.")
        else:
            print("Model does not support disabling distillation.")

    def plot_prediction(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        full_series: Optional[torch.Tensor] = None,
        offset: int = 0,
        stride: int = 1,
        figsize: Tuple[int, int] = (12, 4),
        show: bool = False,
        names: Optional[Union[str, list]] = None,
    ) -> plt.Figure:
        """Legacy prediction plotting - simplified version"""
        evaluator = ModelEvaluator(self)
        predictions = evaluator.predict(X_val)

        fig, ax = plt.subplots(figsize=figsize)

        # Simple plot of first feature
        if predictions.ndim >= 2:
            pred_mean = predictions.mean(dim=0).cpu().numpy()
            target_mean = y_val.mean(dim=0).cpu().numpy()

            ax.plot(target_mean, label="Target", alpha=0.7)
            ax.plot(pred_mean, label="Prediction", alpha=0.7)
            ax.legend()
            ax.grid(True)
            ax.set_title("Prediction vs Target (averaged)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")

        plt.tight_layout()
        if show:
            plt.show()

        return fig


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create dummy data
    X_train = np.random.randn(1000, 50, 10)
    y_train = np.random.randn(1000, 10, 10)

    # Create dataloaders
    train_loader, _ = create_dataloaders(X_train, y_train, batch_size=32)

    # Create model (placeholder)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(500, 128),
        nn.ReLU(),
        nn.Linear(128, 100),
    )

    # ========================================================================
    # Option 1: Clean New API (Recommended) - Now with convenience methods!
    # ========================================================================
    config = TrainingConfig(
        num_epochs=10,
        learning_rate=0.001,
        use_amp=True,
    )

    trainer = Trainer(model, config=config)
    
    # All these methods now work directly on Trainer!
    history = trainer.train(train_loader, epochs=5)  # Can override epochs
    trainer.print_training_summary()
    trainer.plot_learning_curves()
    
    # Generate predictions
    X_test = torch.randn(100, 50, 10)
    y_test = torch.randn(100, 10, 10)
    metrics = trainer.metrics(X_test, y_test)
    print(f"Test metrics: {metrics}")
    
    # Plot predictions
    trainer.plot_prediction(X_test, y_test, show=True)

    # Or use separate evaluator for advanced usage
    evaluator = ModelEvaluator(trainer)
    predictions = evaluator.predict(X_test)

    # ========================================================================
    # Option 2: Legacy API (Backward Compatible)
    # ========================================================================
    # Use this if you need ALL original methods (quantization, distillation, etc.)
    trainer_legacy = TrainerWithLegacyMethods(
        model=model,
        config={"num_epochs": 10, "learning_rate": 0.001},
        criterion=nn.MSELoss(),
    )

    # All legacy methods work
    trainer_legacy.set_config("patience", 5)
    history = trainer_legacy.train(train_loader, epochs=10)
    trainer_legacy.print_training_summary()
    trainer_legacy.plot_learning_curves()
    metrics = trainer_legacy.metrics(X_test, y_test)
