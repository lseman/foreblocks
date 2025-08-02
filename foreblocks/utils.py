import contextlib
import copy
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .third_party.vsgd import *


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset for time series data"""

    def __init__(self, X, y=None):
        """
        Initialize dataset

        Args:
            X: Input sequences of shape [n_sequences, seq_len, n_features]
            y: Target sequences of shape [n_sequences, horizon, n_features]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=32):
    """
    Create PyTorch DataLoaders for training and validation

    Args:
        X_train: Training input sequences
        y_train: Training target sequences
        X_val: Validation input sequences
        y_val: Validation target sequences
        batch_size: Batch size

    Returns:
        train_dataloader: DataLoader for training
        val_dataloader: DataLoader for validation (if validation data provided)
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Create validation dataloader if validation data provided
    if X_val is not None and y_val is not None:
        val_dataset = TimeSeriesDataset(X_val, y_val)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        return train_dataloader, val_dataloader

    return train_dataloader, None


class Trainer:
    """
    Clean trainer with automatic handling of AMP, distillation, and quantization.
    Works with BaseForecastingModel, ForecastingModel, and QuantizedForecastingModel.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
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

        self.config = self._default_config()
        if config:
            self.config.update(config)

        self.optimizer = optimizer or self._get_optimizer()
        self.criterion = criterion or self._get_criterion()
        self.scheduler = scheduler or self._get_scheduler()
        self.scaler = GradScaler() if self.config["use_amp"] else None

        self._init_tracking()

        if self.use_wandb:
            import wandb
            wandb.init(**(wandb_config or {}))
            wandb.watch(self.model, log="all", log_freq=100)

    def _default_config(self):
        return {
            "num_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "patience": 10,
            "min_delta": 1e-4,
            "use_amp": True,
            "gradient_clip_val": None,
            "scheduler_type": None,
            "min_lr": 1e-6,
            "lr_step_size": 30,
            "lr_gamma": 0.1,
            "verbose": True,
            "log_interval": 10,
            "save_best_model": True,
            "save_model_path": None,
            "gradient_accumulation_steps": 1,
            "l1_regularization": 0.0,
            "kl_weight": 1.0,
        }

    def set_config(self, key: str, value: Any):
        if key in self.config:
            self.config[key] = value
        else:
            raise KeyError(f"Config key '{key}' not found.")

    def _init_tracking(self):
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
            "task_losses": [],
            "distillation_losses": [],
            "model_info": []
        }
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def _get_optimizer(self):
        logging.warning("Using custom VSGD optimizer.")
        return VSGD(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            ghattg=30.0,
            ps=1e-8,
            tau1=0.81,
            tau2=0.9,
            eps=1e-8,
        )

    def _get_criterion(self):
        return nn.MSELoss()

    def _get_scheduler(self):
        if self.config["scheduler_type"] == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config["lr_step_size"],
                gamma=self.config["lr_gamma"],
            )
        return None

    def _extract_batch_data(self, batch):
        if len(batch) == 2:
            return batch[0], batch[1], None
        elif len(batch) == 3:
            return batch[0], batch[1], batch[2]
        else:
            raise ValueError(f"Expected 2 or 3 elements in batch, got {len(batch)}")

    def _extract_model_info(self):
        info = {}
        if hasattr(self.model, 'get_model_size'):
            info.update(self.model.get_model_size())

        if hasattr(self.model, 'get_distillation_info'):
            distill_info = self.model.get_distillation_info()
            if distill_info.get("distillation_enabled", False):
                info.update({
                    "distillation_mode": distill_info.get("distillation_mode"),
                    "has_teacher": distill_info.get("has_teacher"),
                })

        if hasattr(self.model, 'get_quantization_info'):
            quant_info = self.model.get_quantization_info()
            if quant_info.get("quantization_enabled", False):
                info.update({
                    "quantization_mode": quant_info.get("quantization_mode"),
                    "is_quantized": quant_info.get("is_quantized"),
                })

        return info

    def _log_features(self, epoch, train_loss, val_loss, lr):
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr,
        }
        if self.history["task_losses"]:
            log_dict["task_loss"] = self.history["task_losses"][-1]
        if self.history["distillation_losses"]:
            log_dict["distillation_loss"] = self.history["distillation_losses"][-1]

        log_dict.update(self._extract_model_info())
        return log_dict

    def _forward_pass(self, X, y, time_feat=None):
        if hasattr(self.model, 'get_distillation_info') and self.model.get_distillation_info().get("distillation_enabled", False):
            result = self.model(X, y, time_feat, self.current_epoch, return_teacher_outputs=True)
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], {"teacher_outputs": result[1]}
            return result[0] if isinstance(result, tuple) else result, {}

        result = self.model(X, y, time_feat, self.current_epoch)
        return result if isinstance(result, tuple) else (result, {})

    def _compute_loss(self, outputs, targets, aux: Optional[Dict[str, torch.Tensor]] = None):
        aux = aux or {}
        base_loss = self.criterion(outputs, targets)
        total_loss = base_loss
        loss_components = {"task_loss": base_loss.item()}

        if hasattr(self.model, 'compute_distillation_loss') and "teacher_outputs" in aux:
            distillation_loss, distill_components = self.model.compute_distillation_loss(
                outputs, aux["teacher_outputs"], targets, self.criterion
            )
            total_loss = distillation_loss
            loss_components.update({f"distill_{k}": v.item() if isinstance(v, torch.Tensor) else v
                                    for k, v in distill_components.items()})

        l1_weight = self.config.get("l1_regularization", 0.0)
        if l1_weight > 0:
            l1 = sum(torch.sum(torch.abs(p)) for p in self.model.parameters() if p.requires_grad)
            total_loss += l1_weight * l1
            loss_components["l1_loss"] = (l1_weight * l1).item()

        if hasattr(self.model, "get_kl"):
            kl_div = self.model.get_kl()
            if kl_div is not None:
                kl_weight = self.config.get("kl_weight", 1.0)
                total_loss += kl_weight * kl_div
                loss_components["kl_loss"] = (kl_weight * kl_div).item()

        self.last_loss_components = loss_components
        return total_loss

    def _step_optimizer(self, loss, batch_idx, total_batches):
        grad_acc = self.config["gradient_accumulation_steps"]
        loss = loss / grad_acc

        if self.config["use_amp"]:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1 == total_batches):
                if self.config["gradient_clip_val"]:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip_val"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1 == total_batches):
                if self.config["gradient_clip_val"]:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip_val"])
                self.optimizer.step()
                self.optimizer.zero_grad()

    def train_epoch(self, dataloader, callbacks=None):
        self.model.train()
        total_loss = 0.0
        epoch_loss_components = {}

        for batch_idx, batch in enumerate(dataloader):
            X, y, time_feat = self._extract_batch_data(batch)
            X, y = X.to(self.device), y.to(self.device)
            if time_feat is not None:
                time_feat = time_feat.to(self.device)

            with (autocast("cuda") if self.config["use_amp"] else contextlib.nullcontext()):
                outputs, aux = self._forward_pass(X, y, time_feat)
                loss = self._compute_loss(outputs, y, aux)

            self._step_optimizer(loss, batch_idx, len(dataloader))
            total_loss += loss.item()

            for k, v in self.last_loss_components.items():
                epoch_loss_components.setdefault(k, []).append(v)

        # Average tracked loss components
        avg_loss = {k: np.mean(v) for k, v in epoch_loss_components.items()}
        if "task_loss" in avg_loss:
            self.history["task_losses"].append(avg_loss["task_loss"])
        if any(k.startswith("distill_") for k in avg_loss):
            distill_loss = sum(v for k, v in avg_loss.items() if k.startswith("distill_"))
            self.history["distillation_losses"].append(distill_loss)

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                X, y, _ = self._extract_batch_data(batch)
                X, y = X.to(self.device), y.to(self.device)
                with (autocast("cuda") if self.config["use_amp"] else contextlib.nullcontext()):
                    result = self.model(X)
                    outputs = result[0] if isinstance(result, tuple) else result
                    loss = self.criterion(outputs, y)
                    total_loss += loss.item() * X.size(0)

        return total_loss / len(dataloader.dataset)

    def train(self, train_loader, val_loader=None, callbacks=None, epochs=None):
        self._init_tracking()
        num_epochs = epochs or self.config["num_epochs"]

        with tqdm(range(num_epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                self.current_epoch = epoch
                train_loss = self.train_epoch(train_loader)
                self.history["train_losses"].append(train_loss)

                val_loss = None
                if val_loader:
                    val_loss = self.evaluate(val_loader)
                    self.history["val_losses"].append(val_loss)

                lr = self.optimizer.param_groups[0]["lr"]
                self.history["learning_rates"].append(lr)

                if hasattr(self.model, 'get_model_size'):
                    self.history["model_info"].append({"epoch": epoch, **self.model.get_model_size()})

                log_dict = self._log_features(epoch, train_loss, val_loss, lr)
                if self.use_wandb:
                    import wandb
                    wandb.log(log_dict)

                if val_loader:
                    if val_loss + self.config["min_delta"] < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        if self.config["save_model_path"]:
                            self.save(self.config["save_model_path"])
                    else:
                        self.epochs_without_improvement += 1

                    if self.epochs_without_improvement >= self.config["patience"]:
                        print("Early stopping triggered.")
                        break

                postfix_dict = {
                    "epoch": epoch + 1,
                    "train_loss": f"{train_loss:.4f}",
                    "lr": f"{lr:.2e}",
                }
                if val_loss is not None:
                    postfix_dict["val_loss"] = f"{self.val_loss:.4f}"
                if log_dict.get("is_quantized"):
                    postfix_dict["quant"] = "✓"
                if log_dict.get("distillation_mode") not in [None, "none"]:
                    postfix_dict["distill"] = "✓"
                pbar.set_postfix(postfix_dict)


                if self.scheduler:
                    self.scheduler.step(val_loss if val_loader else train_loss)

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history


    ## Save and Load Methods

    def save(self, path):
        """Save model and training state with automatic feature detection"""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
        }
        
        # Add model info if available (delegate to model)
        if hasattr(self.model, 'get_model_size'):
            save_dict["model_info"] = self.model.get_model_size()
        if hasattr(self.model, 'get_quantization_info'):
            save_dict["quantization_info"] = self.model.get_quantization_info()
        if hasattr(self.model, 'get_distillation_info'):
            save_dict["distillation_info"] = self.model.get_distillation_info()
            
        torch.save(save_dict, path)

    def load(self, path):
        """Load model and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {})
        self.config.update(checkpoint.get("config", {}))

    ## Benchmarking Methods
    
    def benchmark_model(self, sample_input: torch.Tensor, num_runs: int = 100):
        """Benchmark model performance (delegated to model)"""
        if not hasattr(self.model, 'benchmark_inference'):
            print("Model does not support benchmarking.")
            return None

        sample_input = sample_input.to(self.device)
        results = self.model.benchmark_inference(sample_input, num_runs=num_runs)

        print("\nModel Performance Benchmark:")
        print(f"  Inference Time: {results['avg_inference_time_ms']:.2f} ms")
        print(f"  Throughput:     {results['throughput_samples_per_sec']:.2f} samples/sec")

        if hasattr(self.model, 'get_model_size'):
            size_info = self.model.get_model_size()
            print(f"  Model Size:     {size_info['size_mb']:.2f} MB")
            print(f"  Parameters:     {size_info['parameters']:,}")

        return results

    def plot_learning_curves(self, figsize=(15, 8)):
        """Plot loss, learning rate, distillation, and model size over epochs"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Loss curves
        axes[0, 0].plot(self.history["train_losses"], label="Train")
        if self.history["val_losses"]:
            axes[0, 0].plot(self.history["val_losses"], label="Validation")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Learning rate
        axes[0, 1].plot(self.history["learning_rates"], label="LR")
        axes[0, 1].set_title("Learning Rate")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("LR")
        axes[0, 1].grid(True)

        # Task vs distillation loss
        if self.history["task_losses"] and self.history["distillation_losses"]:
            axes[1, 0].plot(self.history["task_losses"], label="Task")
            axes[1, 0].plot(self.history["distillation_losses"], label="Distillation")
            axes[1, 0].set_title("Task vs Distillation Loss")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, "No distillation data", ha='center', va='center')
            axes[1, 0].set_title("Distillation Loss")

        # Model size
        if self.history["model_info"]:
            epochs = [info["epoch"] for info in self.history["model_info"]]
            sizes = [info.get("size_mb", 0) for info in self.history["model_info"]]
            axes[1, 1].plot(epochs, sizes, label="Size (MB)")
            axes[1, 1].set_title("Model Size")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("MB")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, "No model size data", ha='center', va='center')
            axes[1, 1].set_title("Model Size")

        plt.tight_layout()
        plt.show()


    def print_training_summary(self):
        """Print formatted training summary with distillation and quantization info"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)

        print(f"Total epochs:      {len(self.history['train_losses'])}")
        print(f"Final train loss:  {self.history['train_losses'][-1]:.6f}")
        if self.history["val_losses"]:
            print(f"Final val loss:    {self.history['val_losses'][-1]:.6f}")
            print(f"Best val loss:     {self.best_val_loss:.6f}")

        model_type = "BaseForecastingModel"
        if getattr(self.model, 'get_distillation_info', None):
            info = self.model.get_distillation_info()
            if info.get("distillation_enabled", False):
                model_type = "ForecastingModel (with distillation)"

        if getattr(self.model, 'get_quantization_info', None):
            info = self.model.get_quantization_info()
            if info.get("quantization_enabled", False):
                model_type = "QuantizedForecastingModel"

        print(f"\nMODEL TYPE: {model_type}")

        for label, getter in [
            ("DISTILLATION INFO", "get_distillation_info"),
            ("MODEL INFO", "get_model_size"),
            ("QUANTIZATION INFO", "get_quantization_info")
        ]:
            if hasattr(self.model, getter):
                info = getattr(self.model, getter)()
                if info:
                    print(f"\n{label}:")
                    for k, v in info.items():
                        print(f"  {k}: {v}")

        print("="*60)


    def compare_with_baseline(self, baseline_model: nn.Module, test_loader: torch.utils.data.DataLoader):
        """Compare current model with a baseline model on loss and size"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        current_loss = self.evaluate(test_loader)

        original_model = self.model
        self.model = baseline_model.to(self.device)
        baseline_loss = self.evaluate(test_loader)
        self.model = original_model

        print(f"Current model loss:  {current_loss:.6f}")
        print(f"Baseline model loss: {baseline_loss:.6f}")
        improvement = 100 * (baseline_loss - current_loss) / baseline_loss
        print(f"Improvement:          {improvement:.2f}%")

        if hasattr(self.model, 'get_model_size') and hasattr(baseline_model, 'get_model_size'):
            cur_size = self.model.get_model_size()["size_mb"]
            base_size = baseline_model.get_model_size()["size_mb"]
            reduction = 100 * (base_size - cur_size) / base_size

            print(f"\nCurrent size:   {cur_size:.2f} MB")
            print(f"Baseline size:  {base_size:.2f} MB")
            print(f"Size reduction: {reduction:.2f}%")

        print("="*60)

        return {
            "current_loss": current_loss,
            "baseline_loss": baseline_loss,
            "loss_improvement": improvement,
        }

    def _batched_forecast(self, X_val: torch.Tensor, batch_size: int = 256):
        """
        Generate batched forecasts aligned for time series evaluation or plotting.
        Returns: forecast tensor of shape [T + target_len - 1, output_size]
        """
        self.model.eval()
        X_val = X_val.to(self.device)
        N = X_val.shape[0]

        # Dummy output to infer shape
        with torch.no_grad():
            dummy = self.model(X_val[:1])
            dummy = dummy[0] if isinstance(dummy, tuple) else dummy
            if dummy.ndim == 3:
                output_len, output_size = dummy.shape[1:]
            elif dummy.ndim == 2:
                output_len, output_size = 1, dummy.shape[1]
            else:
                output_len, output_size = 1, dummy.shape[0]

        forecast = torch.zeros(N + output_len - 1, output_size, device=self.device)
        count = torch.zeros_like(forecast)

        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = X_val[i:i + batch_size]
                with autocast("cuda", dtype=torch.float16):
                    preds = self.model(batch)
                    preds = preds[0] if isinstance(preds, tuple) else preds

                for j in range(preds.shape[0]):
                    pred = preds[j]
                    start = i + j

                    # Normalize shape
                    if pred.ndim == 3:
                        pred = pred.squeeze(0)
                    if pred.ndim == 2:
                        end = start + pred.shape[0]
                        forecast[start:end] += pred
                        count[start:end] += 1
                    elif pred.ndim == 1:
                        forecast[start] += pred
                        count[start] += 1
                    else:
                        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

        return forecast / count.clamp(min=1.0)

    def metrics(self, X_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, float]:
        """
        Compute error metrics (MSE, RMSE, MAE) over full validation forecast.
        Args:
            X_val: [N, seq_len, input_size]
            y_val: [N, target_len, output_size]
        Returns:
            Dict with 'mse', 'rmse', 'mae'
        """
        self.model.eval()
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        N, target_len, output_size = y_val.shape

        forecast = self._batched_forecast(X_val)  # [T, D]

        # Align ground truth
        truth = torch.zeros_like(forecast)
        count = torch.zeros_like(forecast)

        for i in range(N):
            truth[i:i + target_len] += y_val[i]
            count[i:i + target_len] += 1

        truth /= count.clamp(min=1.0)

        metrics = self._compute_metrics(forecast, truth)

        print("\nValidation Forecast Error Metrics:")
        for k, v in metrics.items():
            print(f"  {k.upper():<5} = {v:.6f}")

        return metrics

    def _compute_metrics(
        self,
        prediction: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute per-feature MSE, RMSE, MAE and return overall average.
        """
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        mse = np.mean((prediction - target) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(prediction - target), axis=0)

        return {
            "mse": float(np.mean(mse)),
            "rmse": float(np.mean(rmse)),
            "mae": float(np.mean(mae)),
        }


    def plot_prediction(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        full_series: Optional[torch.Tensor] = None,
        offset: int = 0,
        figsize: Tuple[int, int] = (12, 4),
        show: bool = False,
        names: Optional[Union[str, list]] = None,
    ) -> plt.Figure:
        """
        Plot predicted sequences and optionally overlay on full time series.

        Args:
            X_val: [N, seq_len, input_size]
            y_val: [N, target_len, output_size]
            full_series: (Optional) Original full time series
            offset: Offset for forecast alignment in full_series
            figsize: Size of each subplot
            show: Whether to call plt.show()
            names: Optional list of feature names

        Returns:
            Matplotlib figure
        """
        self.model.eval()
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        forecast = self._batched_forecast(X_val).cpu().numpy()
        target_len, output_size = y_val.shape[1:]

        if full_series is not None:
            full_series = full_series.cpu().numpy()
            last_dim = full_series.shape[-1] if full_series.ndim > 1 else 1

            fig, axes = plt.subplots(
                last_dim, 1, figsize=(figsize[0], figsize[1] * last_dim), sharex=True
            )
            axes = np.atleast_1d(axes)
            forecast_start = offset + X_val.shape[1]

            for i in range(last_dim):
                name = names[i] if names else f"Feature {i}"
                series = full_series[:, i] if full_series.ndim > 1 else full_series
                pred = forecast[:, i] if forecast.ndim > 1 else forecast

                end = min(forecast_start + len(pred), len(series))
                forecast_plot = pred[: end - forecast_start]

                ax = axes[i]
                ax.plot(series, label=f"Original {name}", alpha=0.5)
                ax.plot(
                    np.arange(forecast_start, end),
                    forecast_plot,
                    label=f"Forecast {name}",
                    color="orange",
                )
                if len(series) >= end:
                    ax.fill_between(
                        np.arange(forecast_start, end),
                        forecast_plot,
                        series[forecast_start:end],
                        color="red",
                        alpha=0.2,
                        label="Forecast Error",
                    )
                ax.axvline(forecast_start, color="gray", linestyle="--", label="Forecast Start")
                ax.set_title(f"{name}: Full Series Forecast")
                ax.legend(loc="upper left")
                ax.grid(True)

            axes[last_dim // 2].set_ylabel("Value")
            plt.xlabel("Time Step")
            plt.tight_layout()

        else:
            fig, ax = plt.subplots(figsize=figsize)
            if forecast.ndim == 1:
                ax.plot(forecast, label="Forecast", color="orange")
            else:
                for i in range(forecast.shape[1]):
                    name = names[i] if names else f"Feature {i}"
                    ax.plot(forecast[:, i], label=f"Forecast {name}")
            ax.set_title("Forecast (Validation)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend(loc="upper left")
            ax.grid(True)

        if show:
            plt.show()
        return fig

    # ==================== MODEL INTERFACE METHODS ====================
    # These methods provide a clean interface to the model's features
    
    def prepare_quantization(self, sample_input: torch.Tensor, calibration_loader=None):
        if hasattr(self.model, "prepare_for_quantization"):
            print("Preparing model for quantization...")
            sample_input = sample_input.to(self.device)
            self.model = self.model.prepare_for_quantization(calibration_loader)
            print("Model quantization prepared!")
        else:
            print("Model does not support quantization.")

    def finalize_quantization(self):
        if hasattr(self.model, "finalize_quantization"):
            print("Finalizing quantization...")
            self.model = self.model.finalize_quantization()
            print("Quantization finalized!")
        else:
            print("Model does not support quantization finalization.")

    def get_quantization_info(self) -> Dict[str, Any]:
        if hasattr(self.model, "get_quantization_info"):
            return self.model.get_quantization_info()
        return {"quantization_enabled": False}

    def set_quantization_mode(self, mode: str):
        if hasattr(self.model, "set_quantization_mode"):
            self.model.set_quantization_mode(mode)
            print(f"Quantization mode set to: {mode}")
        else:
            print("Model does not support setting quantization mode.")

    def get_distillation_info(self) -> Dict[str, Any]:
        if hasattr(self.model, "get_distillation_info"):
            return self.model.get_distillation_info()
        return {"distillation_enabled": False}

    def enable_distillation(self, mode: str = "output", teacher_model: nn.Module = None):
        if hasattr(self.model, "enable_distillation"):
            self.model.enable_distillation(mode, teacher_model)
            print(f"Distillation enabled (mode: {mode}).")
        else:
            print("Model does not support distillation.")

    def disable_distillation(self):
        if hasattr(self.model, "disable_distillation"):
            self.model.disable_distillation()
            print("Distillation disabled.")
        else:
            print("Model does not support disabling distillation.")
