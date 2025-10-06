import contextlib
import copy
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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

        aux_loss = self.model.get_aux_loss() if hasattr(self.model, "get_aux_loss") else None
        # print(f"Auxiliary loss: {aux_loss}")

        loss_components["aux_loss"] = aux_loss.item() if aux_loss is not None else 0.0
        total_loss += loss_components["aux_loss"]
        
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
                if outputs.ndim == 4 and y.ndim == 3:
                    y = y.unsqueeze(-1)
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
        cbs = list(callbacks or [])

        # notify
        for cb in cbs:
            if hasattr(cb, "on_train_begin"): cb.on_train_begin(self)

        with tqdm(range(num_epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                self.current_epoch = epoch
                for cb in cbs:
                    if hasattr(cb, "on_epoch_begin"): cb.on_epoch_begin(self, epoch)

                # ---- training step
                train_loss = self.train_epoch(train_loader)
                self.history.setdefault("train_losses", []).append(train_loss)

                # ---- validation (optional)
                val_loss = None
                if val_loader:
                    val_loss = self.evaluate(val_loader)
                    self.history.setdefault("val_losses", []).append(val_loss)

                # ---- lr + optional model info
                lr = self.optimizer.param_groups[0]["lr"]
                self.history.setdefault("learning_rates", []).append(lr)

                if hasattr(self.model, "get_model_size"):
                    self.history.setdefault("model_info", []).append(
                        {"epoch": epoch, **self.model.get_model_size()}
                    )

                # ---- your existing feature logging (kept)
                log_dict = self._log_features(epoch, train_loss, val_loss, lr)
                if self.use_wandb:
                    import wandb
                    wandb.log(log_dict)

                # ---- early stopping / best model
                if val_loader:
                    if val_loss + self.config["min_delta"] < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        if self.config.get("save_model_path"):
                            self.save(self.config["save_model_path"])
                    else:
                        self.epochs_without_improvement += 1

                    if self.epochs_without_improvement >= self.config["patience"]:
                        print("Early stopping triggered.")
                        # still call callbacks' on_train_end after loop
                        break

                # ---- progress bar
                postfix_dict = {
                    "epoch": epoch + 1,
                    "train_loss": f"{train_loss:.4f}",
                    "lr": f"{lr:.2e}",
                }
                if val_loss is not None:
                    postfix_dict["val_loss"] = f"{val_loss:.4f}"  # <-- fixed
                if log_dict.get("is_quantized"):
                    postfix_dict["quant"] = "✓"
                if log_dict.get("distillation_mode") not in [None, "none"]:
                    postfix_dict["distill"] = "✓"
                pbar.set_postfix(postfix_dict)

                # ---- scheduler
                if self.scheduler:
                    self.scheduler.step(val_loss if val_loader else train_loss)

                # ---- CALLBACK: per-epoch
                logs = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    **({"val_loss": float(val_loss)} if val_loss is not None else {}),
                    "lr": float(lr),
                    # Optional extras if you like:
                    "best_val_loss": float(getattr(self, "best_val_loss", val_loss or train_loss)),
                }
                for cb in cbs:
                    if hasattr(cb, "on_epoch_end"): cb.on_epoch_end(self, epoch, logs)

        # restore best weights if tracked
        if getattr(self, "best_model_state", None):
            self.model.load_state_dict(self.best_model_state)

        # notify end
        for cb in cbs:
            if hasattr(cb, "on_train_end"): cb.on_train_end(self)

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
        
    # =========================
    # Trainer helper utilities
    # =========================
    def _amp_ctx(self, amp: bool):
        from contextlib import nullcontext
        try:
            from torch.cuda.amp import autocast
        except Exception:
            autocast = None
        use_amp = amp and getattr(self.device, "type", "cpu") == "cuda" and autocast is not None
        return autocast("cuda") if use_amp else nullcontext()

    def _shape_info_from_dummy(self, dummy):
        """
        Returns (has_time_axis: bool, L: int, tail_shape: tuple[int,...]).
        Assumptions:
        - If dummy.ndim >= 3, first dim after batch is time (L) -> [B,L,...]
        - If dummy.ndim == 2, no explicit time -> [B,D]
        """
        if dummy.ndim >= 3:          # [B, L, ...]  e.g. [B,L,D] or [B,L,N,D]
            return True, dummy.shape[1], tuple(dummy.shape[2:])
        if dummy.ndim == 2:          # [B, D]
            return False, 1, (dummy.shape[1],)
        if dummy.ndim == 1:          # [D] (rare)
            return False, 1, (dummy.shape[0],)
        raise ValueError(f"Unsupported prediction rank {dummy.ndim} with shape {tuple(dummy.shape)}")

    def _node_reduce(self, x, node_reducer):
        """Apply a node reducer if provided (expects node dim at -2 when present)."""
        return x if node_reducer is None else node_reducer(x)

    def _alloc_forecast_buffers(self, T_total: int, L: int, tail_shape: tuple, has_time: bool, device):
        H = T_total + (L - 1 if has_time else 0)
        import torch
        if has_time:
            forecast = torch.zeros((H,) + tail_shape, device=device)
        else:
            forecast = torch.zeros((T_total,) + tail_shape, device=device)
        count = torch.zeros_like(forecast)
        return forecast, count

    def _align_targets_to_forecast(self, y_val, forecast_tail: tuple):
        """
        Normalize y_val to match forecast tail exactly.
        forecast_tail is () or (D,) or (N,D). Accepted y_val:
        - [N, L, D]
        - [N, L, N, D]
        - [N, L, N*D] (packed)
        Returns y_val with tail matching forecast tail.
        """
        assert y_val.dim() in (3, 4), f"Unsupported y_val shape {tuple(y_val.shape)}"

        N, L = y_val.shape[0], y_val.shape[1]
        tail_ndim = len(forecast_tail)

        if y_val.dim() == 4:
            # [N, L, Nn, D]
            if tail_ndim == 2:
                Nn_f, Dout_f = forecast_tail
                Nn_t, Dout_t = y_val.shape[2], y_val.shape[3]
                if Nn_t == 1 and Nn_f > 1:
                    y_val = y_val.expand(N, L, Nn_f, Dout_t)
                    Nn_t = Nn_f
                if (Nn_t, Dout_t) != (Nn_f, Dout_f):
                    raise RuntimeError(f"Targets tail {(Nn_t, Dout_t)} doesn't match forecast tail {(Nn_f, Dout_f)}")
                return y_val
            if tail_ndim == 1:
                # reduce nodes by mean to get [N,L,D]
                return y_val.mean(dim=2)
            raise RuntimeError(f"Unexpected forecast tail {forecast_tail}")

        # y_val is [N, L, K]
        K = y_val.shape[2]
        if tail_ndim == 2:
            Nn_f, Dout_f = forecast_tail
            if K == Dout_f:
                return y_val.unsqueeze(2).expand(N, L, Nn_f, Dout_f)          # broadcast across nodes
            if K == Nn_f * Dout_f:
                return y_val.view(N, L, Nn_f, Dout_f).contiguous()            # packed -> reshape
            if K == Nn_f and Dout_f == 1:
                return y_val.unsqueeze(-1)                                    # scalar per node
            raise RuntimeError(
                f"Cannot align y_val [N,L,{K}] to forecast tail (N={Nn_f},D={Dout_f}). "
                "Use [N,L,D], [N,L,N,D], or packed [N,L,N*D]."
            )
        if tail_ndim == 1:
            Dout_f = forecast_tail[0]
            if K != Dout_f:
                raise RuntimeError(f"Cannot align y_val [N,L,{K}] to forecast tail (D={Dout_f}).")
            return y_val

        raise RuntimeError(f"Unexpected forecast tail {forecast_tail}")

    def _overlap_add_truth(self, y_val, forecast_shape: tuple):
        """
        Build overlapped ground-truth aligned with forecast.
        forecast_shape: (T, ...) where ... = (D) or (N,D)
        y_val must be normalized to [N,L,...] with same tail.
        """
        import torch
        T = forecast_shape[0]
        tail = forecast_shape[1:]
        N, L = y_val.shape[0], y_val.shape[1]
        truth = torch.zeros(forecast_shape, device=y_val.device, dtype=y_val.dtype)
        count = torch.zeros_like(truth)
        for i in range(N):
            s, e = i, i + L
            truth[s:e] += y_val[i]
            count[s:e] += 1
        truth = truth / count.clamp_min(1)
        return truth


    def _batched_forecast(
        self,
        X_val: torch.Tensor,
        batch_size: int = 256,
        *,
        amp: bool = True,
        node_reducer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_mode: str = "grid",   # "grid" -> [T, L, ...], "overlap_add" -> [T+L-1, ...]
    ):
        """
        Generate batched forecasts aligned for evaluation/plotting.

        Expected per-example prediction (after removing batch):
        - [L, D]        -> grid: [T, L, D]          | overlap_add: [T + L - 1, D]
        - [L, N, D]     -> grid: [T, L, N, D]       | overlap_add: [T + L - 1, N, D]
        - [D]           -> grid: [T, 1, D]          | overlap_add: [T, D]
        - [N, D]        -> grid: [T, 1, N, D]       | overlap_add: [T, N, D]
        """
        from contextlib import nullcontext

        assert return_mode in {"grid", "overlap_add"}

        self.model.eval()
        device = self.device
        X_val = X_val.to(device)
        T_total = X_val.shape[0]

        # Infer layout from a dummy forward
        with torch.no_grad():
            dummy = self.model(X_val[:1])
            dummy = dummy[0] if isinstance(dummy, tuple) else dummy
            has_time, L, tail_shape = self._shape_info_from_dummy(dummy)

        # Ensure we always have a time axis on preds
        if not has_time:
            # Treat scalars/instant preds as L=1
            L = 1
            tail_shape = tuple(dummy.shape)  # e.g., (D) or (N, D)

        # Discover reduced tail (if node_reducer is provided)
        tmp = torch.zeros((L,) + tail_shape, device=device)
        tmp = self._node_reduce(tmp, node_reducer)
        reduced_tail = tuple(tmp.shape[1:]) if tmp.ndim >= 2 else ()

        # Allocate grid buffers: [T, L, ...]
        grid_shape = (T_total, L) + reduced_tail
        forecast_grid = torch.zeros(grid_shape, device=device)
        count_grid    = torch.zeros(grid_shape, device=device)

        amp_ctx = self._amp_ctx(amp) if hasattr(self, "_amp_ctx") else nullcontext()

        with torch.no_grad():
            with amp_ctx:
                for i in range(0, T_total, batch_size):
                    batch = X_val[i:i + batch_size].to(device)
                    preds = self.model(batch)
                    preds = preds[0] if isinstance(preds, tuple) else preds
                    # preds: [B, L, ...] (or [B, ...] -> will expand to [B, 1, ...] below)
                    B = preds.shape[0]

                    for j in range(B):
                        pred = preds[j]  # [L, ...] or [...]
                        if pred.dim() == len(reduced_tail):  # no time dim -> add it
                            pred = pred.unsqueeze(0)  # [1, ...]
                        # Optional node reduction
                        pred = self._node_reduce(pred, node_reducer)  # [L, reduced_tail...]

                        L_eff = pred.shape[0]
                        s = i + j
                        if s >= T_total:
                            break  # safety

                        # Clip horizon if it would run past the end of T
                        L_clip = min(L_eff, L)

                        # Accumulate into the grid: put the L-step forecast starting at time s
                        # Grid semantics: slot the horizon along dim=1
                        forecast_grid[s, :L_clip] += pred[:L_clip]
                        count_grid[s, :L_clip]    += 1

        # Avoid divide-by-zero
        forecast_grid = forecast_grid / count_grid.clamp_min(1)

        if return_mode == "grid":
            # [T, L, ...] — keep per-horizon dimension explicit
            return forecast_grid

        # Convert grid -> overlap-add series: [T+L-1, ...]
        # We slide each row t's horizon slice onto the absolute timeline.
        out_len = T_total + L - 1
        oa = torch.zeros((out_len,) + reduced_tail, device=device)
        oc = torch.zeros((out_len,) + reduced_tail, device=device)

        for t in range(T_total):
            # place forecast_grid[t, h] at absolute time t+h
            # vectorized scatter-add over horizon
            h_len = L
            idx = torch.arange(t, t + h_len, device=device)
            oa[idx] += forecast_grid[t]
            oc[idx] += (count_grid[t] > 0).to(oa.dtype)

        return oa / oc.clamp_min(1)

    def _grid_to_overlap_add(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Convert a grid [T, L, ...] into an overlap-add series [T+L-1, ...].
        """
        assert grid.dim() >= 2, f"grid must be [T, L, ...], got {tuple(grid.shape)}"
        device = grid.device
        T, L = grid.shape[:2]
        tail = tuple(grid.shape[2:])
        out_len = T + L - 1

        series = torch.zeros((out_len,) + tail, device=device, dtype=grid.dtype)
        count  = torch.zeros((out_len,) + tail, device=device, dtype=grid.dtype)

        for t in range(T):
            idx = torch.arange(t, t + L, device=device)
            series[idx] += grid[t]                  # add horizon slice at absolute indices
            count[idx]  += (grid[t] == grid[t]).to(grid.dtype)  # add ones (handles broadcasting)

        return series / count.clamp_min(1)


    def metrics(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        *,
        eval_mode: str = "grid",          # "grid" -> compare [T, L, ...]; "overlap_add" -> old [T+L-1, ...]
        batch_size: int = 256,
        amp: bool = True,
        node_reducer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute MSE / RMSE / MAE over validation forecast.

        New default (eval_mode="grid"):
        - forecast: [T, L, ...] from _batched_forecast(return_mode="grid")
        - targets:  [T, L, ...] aligned via _align_targets_to_forecast(..., tail)

        Legacy (eval_mode="overlap_add"):
        - forecast: [T+L-1, ...] via grid -> overlap-add
        - targets:  [T+L-1, ...] via overlap-adding y_val
        """
        self.model.eval()

        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        # 1) Get forecast grid
        forecast = self._batched_forecast(
            X_val,
            batch_size=batch_size,
            amp=amp,
            node_reducer=node_reducer,
            return_mode="grid",          # <- key change
        )  # [T, L, ...]
        if forecast.dim() < 2:
            raise RuntimeError(f"Expected forecast grid [T, L, ...], got {tuple(forecast.shape)}")

        T, L = forecast.shape[:2]
        tail = tuple(forecast.shape[2:])   # tail beyond [T, L]
        
        # 2) Align targets to the forecast tail (beyond L)
        #    Your helper already expects shapes like [T, L, D] / [T, L, N, D] / packed.
        y_norm = self._align_targets_to_forecast(y_val, tail)  # -> [T, L, ...]
        if y_norm.shape[:2] != (T, L):
            # if targets are shorter, clip; if longer, crop
            T2, L2 = y_norm.shape[:2]
            T_use, L_use = min(T, T2), min(L, L2)
            forecast = forecast[:T_use, :L_use]
            y_norm   = y_norm[:T_use, :L_use]

        if eval_mode == "grid":
            # 3) Elementwise comparison on the grid
            diff = forecast - y_norm
            mse  = (diff ** 2).mean().item()
            mae  = diff.abs().mean().item()
            rmse = mse ** 0.5
            return {"mse": mse, "rmse": rmse, "mae": mae}

        elif eval_mode == "overlap_add":
            # 3b) Convert both to overlap-add series and compare (legacy behavior)
            series_f = self._grid_to_overlap_add(forecast)      # [T+L-1, ...]
            series_t = self._grid_to_overlap_add(y_norm)        # [T+L-1, ...]
            # align lengths (clip to common)
            Lf, Lt = series_f.shape[0], series_t.shape[0]
            Lc = min(Lf, Lt)
            series_f = series_f[:Lc]
            series_t = series_t[:Lc]
            diff = series_f - series_t
            mse  = (diff ** 2).mean().item()
            mae  = diff.abs().mean().item()
            rmse = mse ** 0.5
            return {"mse": mse, "rmse": rmse, "mae": mae}

        else:
            raise ValueError(f"Unknown eval_mode={eval_mode}. Use 'grid' or 'overlap_add'.")


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
