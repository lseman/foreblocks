from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .third_party.vsgd import *

import copy
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
    Clean trainer focused purely on training logic.
    All model-specific features (quantization, distillation) are handled by the model itself.
    Works seamlessly with BaseForecastingModel, ForecastingModel, and QuantizedForecastingModel.
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
            "model_info": []  # General model info (size, etc.)
        }
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def _get_optimizer(self):
        print("Warning: Using custom VSGD optimizer.")
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
        t = self.config["scheduler_type"]
        if t == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config["lr_step_size"],
                gamma=self.config["lr_gamma"],
            )
        return None

    def _forward_pass(self, X, y, time_feat=None):
        """Enhanced forward pass with automatic distillation support"""
        # Check if model supports distillation
        if hasattr(self.model, 'get_distillation_info'):
            distill_info = self.model.get_distillation_info()
            if distill_info.get('distillation_enabled', False):
                # Model has distillation - try to get teacher outputs
                result = self.model(X, y, time_feat, self.current_epoch, return_teacher_outputs=True)
                if isinstance(result, tuple) and len(result) == 2:
                    outputs, teacher_outputs = result
                    return outputs, {"teacher_outputs": teacher_outputs}
                else:
                    # Fallback if teacher outputs not available
                    outputs = result[0] if isinstance(result, tuple) else result
                    return outputs, {}
        
        # Regular forward pass (BaseForecastingModel or no distillation)
        result = self.model(X, y, time_feat, self.current_epoch)
        if isinstance(result, tuple):
            outputs, aux = result
        else:
            outputs, aux = result, {}
        return outputs, aux

    def _compute_loss(self, outputs, targets, aux: Optional[Dict[str, torch.Tensor]] = None):
        """Enhanced loss computation with automatic distillation support"""
        if aux is None:
            aux = {}
            
        # Base task loss
        base_loss = self.criterion(outputs, targets)
        total_loss = base_loss
        
        # Track individual loss components
        loss_components = {"task_loss": base_loss.item()}
        
        # Knowledge distillation loss (delegate to model if available)
        if (hasattr(self.model, 'compute_distillation_loss') and
            "teacher_outputs" in aux):
            
            teacher_outputs = aux["teacher_outputs"]
            distillation_loss, distill_components = self.model.compute_distillation_loss(
                outputs, teacher_outputs, targets, self.criterion
            )
            
            # Use the model's own distillation alpha
            total_loss = distillation_loss
            
            # Track distillation components
            loss_components.update({
                f"distill_{k}": v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in distill_components.items()
            })

        # Add auxiliary loss if present
        if "aux_loss" in aux:
            aux_weight = self.config.get("aux_loss_weight", 0.01)
            total_loss += aux_weight * aux["aux_loss"]
            loss_components["aux_loss"] = aux["aux_loss"].item()

        # L1 regularization
        l1_weight = self.config.get("l1_regularization", 0.0)
        if l1_weight > 0:
            l1 = sum(
                torch.sum(torch.abs(p))
                for p in self.model.parameters()
                if p.requires_grad
            )
            total_loss += l1_weight * l1
            loss_components["l1_loss"] = (l1_weight * l1).item()

        # KL divergence (delegate to model)
        if hasattr(self.model, "get_kl"):
            kl_div = self.model.get_kl()
            if kl_div is not None:
                kl_weight = self.config.get("kl_weight", 1.0)
                total_loss += kl_weight * kl_div
                loss_components["kl_loss"] = (kl_weight * kl_div).item()

        # Store loss components for logging
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
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip_val"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1 == total_batches):
                if self.config["gradient_clip_val"]:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip_val"]
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

    def train_epoch(self, dataloader, callbacks=None):
        """Clean training epoch with automatic feature detection"""
        self.model.train()
        
        total_loss = 0.0
        epoch_loss_components = {}
        
        for batch_idx, data in enumerate(dataloader):
            # Handle different data formats
            if len(data) == 2:
                X, y = data
                time_feat = None
            elif len(data) == 3:
                X, y, time_feat = data
            else:
                raise ValueError(f"Expected 2 or 3 elements in batch, got {len(data)}")
                
            X, y = X.to(self.device), y.to(self.device)
            if time_feat is not None:
                time_feat = time_feat.to(self.device)
                
            with (autocast("cuda") if self.config["use_amp"] else contextlib.nullcontext()):
                outputs, aux = self._forward_pass(X, y, time_feat)
                loss = self._compute_loss(outputs, y, aux)
                
            self._step_optimizer(loss, batch_idx, len(dataloader))
            total_loss += loss.item()
            
            # Accumulate loss components
            if hasattr(self, 'last_loss_components'):
                for key, value in self.last_loss_components.items():
                    if key not in epoch_loss_components:
                        epoch_loss_components[key] = []
                    epoch_loss_components[key].append(value)
        
        # Average loss components for the epoch
        avg_loss_components = {}
        for key, values in epoch_loss_components.items():
            avg_loss_components[key] = np.mean(values)
            
        # Store in history
        if "task_loss" in avg_loss_components:
            self.history["task_losses"].append(avg_loss_components["task_loss"])
        if any(k.startswith("distill_") for k in avg_loss_components):
            distill_loss = sum(v for k, v in avg_loss_components.items() if k.startswith("distill_"))
            self.history["distillation_losses"].append(distill_loss)
        
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Clean evaluation"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                with (autocast("cuda") if self.config["use_amp"] else contextlib.nullcontext()):
                    result = self.model(X)
                    if isinstance(result, tuple):
                        outputs, _ = result
                    else:
                        outputs = result
                    loss = self.criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
                
        return total_loss / len(dataloader.dataset)

    def train(self, train_loader, val_loader=None, callbacks=None, epochs=None):
        """Clean training loop with automatic feature detection"""
        self._init_tracking()
        num_epochs = self.config["num_epochs"]
        if epochs is not None:
            num_epochs = epochs

        # Detect model capabilities
        has_distillation = hasattr(self.model, 'get_distillation_info')
        has_quantization = hasattr(self.model, 'get_quantization_info')

        with tqdm(range(num_epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                self.current_epoch = epoch
                train_loss = self.train_epoch(train_loader, callbacks)
                self.history["train_losses"].append(train_loss)
                
                val_loss = None
                if val_loader:
                    val_loss = self.evaluate(val_loader)
                    self.history["val_losses"].append(val_loss)

                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["learning_rates"].append(current_lr)
                
                # Log model info (delegate to model)
                if hasattr(self.model, 'get_model_size'):
                    model_info = self.model.get_model_size()
                    self.history["model_info"].append({
                        "epoch": epoch,
                        **model_info
                    })

                # Enhanced logging with automatic feature detection
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                }
                
                # Add distillation info if available
                if has_distillation:
                    distill_info = self.model.get_distillation_info()
                    if distill_info.get('distillation_enabled', False):
                        log_dict["distillation_mode"] = distill_info.get('distillation_mode', 'none')
                        log_dict["has_teacher"] = distill_info.get('has_teacher', False)
                
                # Add quantization info if available
                if has_quantization:
                    quant_info = self.model.get_quantization_info()
                    if quant_info.get('quantization_enabled', False):
                        log_dict["quantization_mode"] = quant_info.get('quantization_mode', 'none')
                        log_dict["is_quantized"] = quant_info.get('is_quantized', False)
                
                # Add task and distillation losses if available
                if self.history["task_losses"]:
                    log_dict["task_loss"] = self.history["task_losses"][-1]
                if self.history["distillation_losses"]:
                    log_dict["distillation_loss"] = self.history["distillation_losses"][-1]

                if self.use_wandb:
                    wandb.log(log_dict)

                # Early stopping logic
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

                # Update progress bar with automatic feature detection
                pbar_dict = {
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loader else "N/A",
                    "lr": current_lr,
                }
                
                # Add distillation weight to progress bar if available
                if has_distillation:
                    distill_info = self.model.get_distillation_info()
                    if distill_info.get('distillation_enabled', False):
                        pbar_dict["distill"] = "✓"
                
                # Add quantization status to progress bar if available
                if has_quantization:
                    quant_info = self.model.get_quantization_info()
                    if quant_info.get('quantization_enabled', False):
                        pbar_dict["quant"] = "✓"
                
                pbar.set_postfix(pbar_dict)

                if self.scheduler:
                    self.scheduler.step(val_loss if val_loader else train_loss)

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history

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

    def benchmark_model(self, sample_input: torch.Tensor, num_runs: int = 100):
        """Benchmark model performance (delegate to model)"""
        if not hasattr(self.model, 'benchmark_inference'):
            print("Model does not support benchmarking")
            return None
            
        sample_input = sample_input.to(self.device)
        results = self.model.benchmark_inference(sample_input, num_runs=num_runs)
        
        print("\nModel Performance Benchmark:")
        print(f"  Average inference time: {results['avg_inference_time_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
        
        if hasattr(self.model, 'get_model_size'):
            size_info = self.model.get_model_size()
            print(f"  Model size: {size_info['size_mb']:.2f} MB")
            print(f"  Parameters: {size_info['parameters']:,}")
        
        return results

    def plot_learning_curves(self, figsize=(15, 8)):
        """Plot learning curves with automatic feature detection"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Loss curves
        axes[0, 0].plot(self.history["train_losses"], label="Train Loss")
        if self.history["val_losses"]:
            axes[0, 0].plot(self.history["val_losses"], label="Val Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training & Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.history["learning_rates"], label="Learning Rate")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("LR")
        axes[0, 1].set_title("Learning Rate Schedule")
        axes[0, 1].grid(True)
        
        # Distillation losses (if available)
        if self.history["task_losses"] and self.history["distillation_losses"]:
            axes[1, 0].plot(self.history["task_losses"], label="Task Loss")
            axes[1, 0].plot(self.history["distillation_losses"], label="Distillation Loss")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_title("Task vs Distillation Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, "No distillation data", ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Distillation Loss")
        
        # Model size over time
        if self.history["model_info"]:
            epochs = [info["epoch"] for info in self.history["model_info"]]
            sizes = [info.get("size_mb", 0) for info in self.history["model_info"]]
            
            axes[1, 1].plot(epochs, sizes, label="Model Size (MB)")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Model Size (MB)")
            axes[1, 1].set_title("Model Size Over Training")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, "No model size data", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Model Size")
        
        plt.tight_layout()
        plt.show()

    def print_training_summary(self):
        """Print training summary with automatic feature detection"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        # Basic info
        print(f"Total epochs: {len(self.history['train_losses'])}")
        print(f"Final train loss: {self.history['train_losses'][-1]:.6f}")
        if self.history['val_losses']:
            print(f"Final val loss: {self.history['val_losses'][-1]:.6f}")
            print(f"Best val loss: {self.best_val_loss:.6f}")
        
        # Model type detection
        model_type = "BaseForecastingModel"
        if hasattr(self.model, 'get_distillation_info'):
            distill_info = self.model.get_distillation_info()
            if distill_info.get('distillation_enabled', False):
                model_type = "ForecastingModel (with distillation)"
        
        if hasattr(self.model, 'get_quantization_info'):
            quant_info = self.model.get_quantization_info()
            if quant_info.get('quantization_enabled', False):
                model_type = "QuantizedForecastingModel"
        
        print(f"\nMODEL TYPE: {model_type}")
        
        # Distillation info (if available)
        if hasattr(self.model, 'get_distillation_info'):
            distill_info = self.model.get_distillation_info()
            if distill_info.get('distillation_enabled', False):
                print(f"\nDISTILLATION INFO:")
                for key, value in distill_info.items():
                    print(f"  {key}: {value}")
        
        # Model info (delegate to model)
        if hasattr(self.model, 'get_model_size'):
            model_info = self.model.get_model_size()
            print(f"\nMODEL INFO:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
        
        # Quantization info (if available)
        if hasattr(self.model, 'get_quantization_info'):
            quant_info = self.model.get_quantization_info()
            if quant_info.get('quantization_enabled', False):
                print(f"\nQUANTIZATION INFO:")
                for key, value in quant_info.items():
                    print(f"  {key}: {value}")
        
        print("="*60)

    def compare_with_baseline(self, baseline_model: nn.Module, test_loader: torch.utils.data.DataLoader):
        """Compare current model with baseline model"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Evaluate both models
        current_loss = self.evaluate(test_loader)
        
        # Temporarily switch to baseline
        original_model = self.model
        self.model = baseline_model.to(self.device)
        baseline_loss = self.evaluate(test_loader)
        self.model = original_model
        
        print(f"Current model loss: {current_loss:.6f}")
        print(f"Baseline model loss: {baseline_loss:.6f}")
        print(f"Improvement: {((baseline_loss - current_loss) / baseline_loss * 100):.2f}%")
        
        # Size comparison (delegate to models)
        if hasattr(self.model, 'get_model_size') and hasattr(baseline_model, 'get_model_size'):
            current_size = self.model.get_model_size()
            baseline_size = baseline_model.get_model_size()
            
            print(f"\nCurrent model size: {current_size['size_mb']:.2f} MB")
            print(f"Baseline model size: {baseline_size['size_mb']:.2f} MB")
            print(f"Size reduction: {((baseline_size['size_mb'] - current_size['size_mb']) / baseline_size['size_mb'] * 100):.2f}%")
        
        print("="*60)
        
        return {
            "current_loss": current_loss,
            "baseline_loss": baseline_loss,
            "loss_improvement": ((baseline_loss - current_loss) / baseline_loss * 100),
        }

    def _batched_forecast(self, X_val: torch.Tensor, batch_size: int = 256):
        """
        Generate batched forecasts aligned for time series evaluation or plotting.

        Returns:
            forecast: Tensor of shape [T + target_len - 1, output_size]
        """
        self.model.eval()
        X_val = X_val.to(self.device)
        N = X_val.shape[0]

        # Run a dummy forward pass to determine output shape
        with torch.no_grad():
            dummy_out = self.model(X_val[0:1])
            if isinstance(dummy_out, tuple):
                dummy_out = dummy_out[0]
            if dummy_out.dim() == 3:
                output_len, output_size = dummy_out.shape[1], dummy_out.shape[2]
            elif dummy_out.dim() == 2:
                output_len, output_size = 1, dummy_out.shape[1]
            else:
                output_len, output_size = 1, dummy_out.shape[0]

        forecast = torch.zeros(N + output_len - 1, output_size, device=self.device)
        count = torch.zeros_like(forecast)

        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = X_val[i : i + batch_size]
                with autocast("cuda", dtype=torch.float16):
                    outputs = self.model(batch)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    for j in range(outputs.shape[0]):
                        pred = outputs[j]  # shape: [T, D], [1, D], or [D]
                        start = i + j
                        if pred.dim() == 3:  # [1, T, D]
                            pred = pred.squeeze(0)
                        if pred.dim() == 2:
                            if pred.shape[0] == 1:
                                forecast[start] += pred.squeeze(0)
                                count[start] += 1
                            else:
                                forecast[start : start + pred.shape[0]] += pred
                                count[start : start + pred.shape[0]] += 1
                        elif pred.dim() == 1:
                            forecast[start] += pred
                            count[start] += 1
                        else:
                            raise ValueError(
                                f"Unexpected prediction shape: {pred.shape}"
                            )

        return forecast / count.clamp(min=1.0)

    def metrics(self, X_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, float]:
        """
        Compute error metrics (MSE, RMSE, MAE) over the full validation set prediction.

        Args:
            X_val: Tensor of shape [N, seq_len, input_size]
            y_val: Tensor of shape [N, target_len, output_size]

        Returns:
            Dictionary with 'mse', 'rmse', and 'mae'
        """
        self.model.eval()
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        N, target_len, output_size = y_val.shape
        forecast = self._batched_forecast(X_val)  # shape [T, D]

        # Reconstruct ground truth for comparison
        aligned_truth = torch.zeros_like(forecast)
        truth_count = torch.zeros_like(forecast)

        for i in range(N):
            aligned_truth[i : i + target_len] += y_val[i]
            truth_count[i : i + target_len] += 1

        aligned_truth = aligned_truth / torch.clamp(truth_count, min=1.0)

        # Compute metrics per feature then average
        metrics = self._compute_metrics(forecast, aligned_truth)

        print("\nValidation Forecast Error Metrics:")
        for k, v in metrics.items():
            print(f"  {k.upper():<5} = {v:.6f}")

        return metrics

    def _compute_metrics(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute per-feature error metrics between prediction and target:
        MSE, RMSE, MAE. Returns overall mean across features.
        """
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        # shape: [T, output_size] or [T, F]
        mse_per_feat = np.mean((prediction - target) ** 2, axis=0)
        rmse_per_feat = np.sqrt(mse_per_feat)
        mae_per_feat = np.mean(np.abs(prediction - target), axis=0)

        return {
            "mse": float(np.mean(mse_per_feat)),
            "rmse": float(np.mean(rmse_per_feat)),
            "mae": float(np.mean(mae_per_feat)),
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
        Plot predicted sequence over the validation data, aligned to form a full series forecast.
        Creates one subplot for each feature in the last dimension.

        Args:
            X_val: Tensor of shape [N, seq_len, input_size]
            y_val: Tensor of shape [N, target_len, output_size]
            full_series: (Optional) Original full time series for reference
            offset: (Optional) Index offset for where the validation data starts in the full series
            figsize: (Optional) Figure size as (width, height) in inches
            show: (Optional) Whether to display the plot with plt.show()
            names: (Optional) Names for features

        Returns:
            matplotlib Figure object
        """
        self.model.eval()
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        target_len = y_val.shape[1]
        output_size = y_val.shape[2]
        forecast = self._batched_forecast(X_val).cpu().numpy()

        # If full_series is provided
        if full_series is not None:
            full_series = full_series.cpu().numpy()
            last_dim_size = full_series.shape[-1] if full_series.ndim > 1 else 1
            fig, axes = plt.subplots(
                last_dim_size,
                1,
                figsize=(figsize[0], figsize[1] * last_dim_size),
                sharex=True,
            )

            if last_dim_size == 1:
                axes = [axes]

            forecast_start = offset + X_val.shape[1]

            for i in range(last_dim_size):
                # Extract feature series
                if full_series.ndim == 3:
                    feature_series = full_series[:, 0, i]
                elif full_series.ndim == 2:
                    feature_series = full_series[:, i]
                else:
                    feature_series = full_series

                # Plot original
                axes[i].plot(
                    np.arange(len(feature_series)),
                    feature_series,
                    label=f"Original {names[i] if names else f'Feature {i}'}",
                    alpha=0.5,
                )

                # Plot clipped forecast
                feature_forecast = forecast[:, i] if forecast.ndim > 1 else forecast
                end_idx = min(
                    forecast_start + len(feature_forecast), len(feature_series)
                )
                forecast_range = slice(forecast_start, end_idx)
                forecast_plot = feature_forecast[: end_idx - forecast_start]

                axes[i].plot(
                    np.arange(forecast_range.start, forecast_range.stop),
                    forecast_plot,
                    label=f"Forecast {names[i] if names else f'Feature {i}'}",
                    color="orange",
                )

                # Optional error shading
                if len(feature_series) >= end_idx:
                    axes[i].fill_between(
                        np.arange(forecast_range.start, forecast_range.stop),
                        forecast_plot,
                        feature_series[forecast_range],
                        color="red",
                        alpha=0.2,
                        label="Forecast Error",
                    )

                axes[i].axvline(
                    x=forecast_start,
                    color="gray",
                    linestyle="--",
                    label="Forecast Start",
                )
                axes[i].set_title(f"Feature {i}: Full Series with Forecast")
                axes[i].legend(loc="upper left")
                axes[i].grid(True)

            plt.xlabel("Time Step")
            axes[last_dim_size // 2].set_ylabel("Value")
            plt.tight_layout()

        else:
            # No full_series provided
            fig, ax = plt.subplots(figsize=figsize)
            if forecast.ndim > 1:
                for i in range(forecast.shape[1]):
                    ax.plot(forecast[:, i], label=f"Forecast (Feature {i})")
            else:
                ax.plot(forecast, label="Forecast", color="orange")
            ax.set_title("Validation Prediction")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend(loc="upper left")
            ax.grid(True)

        if show:
            plt.show()

        return fig

    # ==================== MODEL INTERFACE METHODS ====================
    # These methods provide a clean interface to the model's features
    
    def prepare_quantization(self, sample_input: torch.Tensor, calibration_loader: Optional[torch.utils.data.DataLoader] = None):
        """
        Prepare model for quantization (delegates to model)
        
        Args:
            sample_input: Sample input tensor for model preparation
            calibration_loader: DataLoader for calibration (PTQ only)
        """
        if hasattr(self.model, 'prepare_for_quantization'):
            print("Preparing model for quantization...")
            sample_input = sample_input.to(self.device)
            self.model = self.model.prepare_for_quantization(calibration_loader)
            print("Model quantization prepared!")
        else:
            print("Model does not support quantization")

    def finalize_quantization(self):
        """Finalize quantization (delegates to model)"""
        if hasattr(self.model, 'finalize_quantization'):
            print("Finalizing quantization...")
            self.model = self.model.finalize_quantization()
            print("Quantization finalized!")
        else:
            print("Model does not support quantization finalization")

    def get_quantization_info(self) -> Dict[str, Any]:
        """Get quantization info (delegates to model)"""
        if hasattr(self.model, 'get_quantization_info'):
            return self.model.get_quantization_info()
        return {"quantization_enabled": False}

    def set_quantization_mode(self, mode: str):
        """Set quantization mode (delegates to model)"""
        if hasattr(self.model, 'set_quantization_mode'):
            self.model.set_quantization_mode(mode)
            print(f"Quantization mode set to: {mode}")
        else:
            print("Model does not support quantization mode setting")

    def get_distillation_info(self) -> Dict[str, Any]:
        """Get distillation info (delegates to model)"""
        if hasattr(self.model, 'get_distillation_info'):
            return self.model.get_distillation_info()
        return {"distillation_enabled": False}

    def enable_distillation(self, mode: str = "output", teacher_model: nn.Module = None):
        """Enable distillation (delegates to model)"""
        if hasattr(self.model, 'enable_distillation'):
            self.model.enable_distillation(mode, teacher_model)
            print(f"Distillation enabled: {mode}")
        else:
            print("Model does not support distillation")

    def disable_distillation(self):
        """Disable distillation (delegates to model)"""
        if hasattr(self.model, 'disable_distillation'):
            self.model.disable_distillation()
            print("Distillation disabled")
        else:
            print("Model does not support distillation")

