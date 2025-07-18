import contextlib
import copy
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
        # Quantization parameters
        quantization_config: Optional[Dict[str, Any]] = None,
        # Distillation parameters
        distillation_config: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_wandb = use_wandb

        self.config = self._default_config()
        if config:
            self.config.update(config)

        # Setup quantization
        self.quantization_config = quantization_config or {}
        self._setup_quantization()

        # Setup distillation
        self.distillation_config = distillation_config or {}
        self._setup_distillation()

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
            "teacher_forcing_ratio": 0.5,
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
            # Quantization training specific
            "qat_start_epoch": 10,  # Start QAT after this epoch
            "qat_freeze_bn_delay": 5,  # Freeze BN stats after this many QAT epochs
            # Distillation specific
            "distillation_start_epoch": 0,  # Start distillation from this epoch
            "distillation_warmup_epochs": 5,  # Gradually increase distillation weight
        }

    def _setup_quantization(self):
        """Setup quantization configuration"""
        self.quantization_enabled = len(self.quantization_config) > 0
        
        if not self.quantization_enabled:
            self.quantization_mode = None
            return
            
        # Default quantization config
        default_quant_config = {
            "mode": "qat",  # "ptq", "qat", or "dynamic"
            "backend": "fbgemm",
            "bit_width": 8,
            "calibration_batches": 100,
            "enable_fake_quantization": True,
        }
        
        # Update with user config
        for key, value in default_quant_config.items():
            if key not in self.quantization_config:
                self.quantization_config[key] = value
        
        self.quantization_mode = self.quantization_config["mode"]
        self.calibration_data = []
        self.quantization_prepared = False
        
        print(f"Quantization enabled: {self.quantization_mode}")

    def _setup_distillation(self):
        """Setup knowledge distillation configuration"""
        self.distillation_enabled = len(self.distillation_config) > 0
        
        if not self.distillation_enabled:
            return
            
        # Default distillation config
        default_distill_config = {
            "mode": "output",  # "output", "feature", "attention", "comprehensive"
            "teacher_model": None,
            "temperature": 4.0,
            "alpha": 0.7,
            "feature_layers": [],
            "attention_layers": [],
            "teacher_model_path": None,
        }
        
        # Update with user config
        for key, value in default_distill_config.items():
            if key not in self.distillation_config:
                self.distillation_config[key] = value
        
        self.distillation_mode = self.distillation_config["mode"]
        self.teacher_model = None
        self.distillation_weight = 0.0  # Will be adjusted during training
        
        # Load teacher model if path provided
        if self.distillation_config["teacher_model_path"]:
            self._load_teacher_model()
        elif self.distillation_config["teacher_model"]:
            self.teacher_model = self.distillation_config["teacher_model"]
        
        # Setup distillation in student model
        if hasattr(self.model, 'enable_distillation') and self.teacher_model is not None:
            self.model.enable_distillation(
                mode=self.distillation_mode,
                teacher_model=self.teacher_model
            )
            self.model.distillation_temperature = self.distillation_config["temperature"]
            self.model.distillation_alpha = self.distillation_config["alpha"]
            
        print(f"Distillation enabled: {self.distillation_mode}")

    def _load_teacher_model(self):
        """Load teacher model from saved checkpoint"""
        try:
            checkpoint = torch.load(self.distillation_config["teacher_model_path"], map_location=self.device)
            
            # Create teacher model (assuming same architecture for now)
            self.teacher_model = copy.deepcopy(self.model)
            
            # Load teacher weights
            if "model_state_dict" in checkpoint:
                self.teacher_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.teacher_model.load_state_dict(checkpoint)
            
            # Set teacher to eval mode
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
                
            print(f"Teacher model loaded from {self.distillation_config['teacher_model_path']}")
            
        except Exception as e:
            print(f"Failed to load teacher model: {e}")
            self.distillation_enabled = False

    def prepare_quantization(self, sample_input: torch.Tensor, calibration_loader: Optional[torch.utils.data.DataLoader] = None):
        """
        Prepare model for quantization
        
        Args:
            sample_input: Sample input tensor for model tracing
            calibration_loader: DataLoader for calibration (PTQ only)
        """
        if not self.quantization_enabled or not hasattr(self.model, 'prepare_for_quantization'):
            return
            
        print(f"Preparing model for {self.quantization_mode} quantization...")
        
        if self.quantization_mode == "ptq":
            # For PTQ, we need calibration data
            if calibration_loader is None:
                print("Warning: No calibration data provided for PTQ. Using dummy calibration.")
                calibration_loader = self._create_dummy_calibration_loader(sample_input)
            
            # Prepare and calibrate
            self.model = self.model.prepare_for_quantization(sample_input, calibration_loader)
            self.quantization_prepared = True
            
        elif self.quantization_mode == "qat":
            # For QAT, we prepare but don't quantize yet
            self.model = self.model.prepare_for_quantization(sample_input)
            self.quantization_prepared = True
            
        elif self.quantization_mode == "dynamic":
            # Dynamic quantization can be applied immediately
            self.model = self.model.prepare_for_quantization(sample_input)
            self.quantization_prepared = True

    def _create_dummy_calibration_loader(self, sample_input: torch.Tensor):
        """Create dummy calibration loader for PTQ"""
        dummy_dataset = torch.utils.data.TensorDataset(sample_input.repeat(10, 1, 1))
        return torch.utils.data.DataLoader(dummy_dataset, batch_size=1)

    def finalize_quantization(self):
        """Finalize quantization after QAT training"""
        if not self.quantization_enabled or not self.quantization_prepared:
            return
            
        if self.quantization_mode == "qat" and hasattr(self.model, 'finalize_quantization'):
            print("Finalizing QAT quantization...")
            self.model = self.model.finalize_quantization()
            print("Model quantization finalized!")

    def _get_distillation_weight(self, epoch: int) -> float:
        """Get distillation weight based on current epoch"""
        if not self.distillation_enabled:
            return 0.0
            
        start_epoch = self.config["distillation_start_epoch"]
        warmup_epochs = self.config["distillation_warmup_epochs"]
        
        if epoch < start_epoch:
            return 0.0
        elif epoch < start_epoch + warmup_epochs:
            # Linear warmup
            progress = (epoch - start_epoch) / warmup_epochs
            return progress * self.distillation_config["alpha"]
        else:
            return self.distillation_config["alpha"]

    def _should_start_qat(self, epoch: int) -> bool:
        """Check if QAT should start at this epoch"""
        return (self.quantization_enabled and 
                self.quantization_mode == "qat" and 
                epoch >= self.config["qat_start_epoch"] and
                not self.quantization_prepared)

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
            "distillation_losses": [],
            "task_losses": [],
            "quantization_info": []
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
        """Enhanced forward pass with distillation support"""
        if self.distillation_enabled and hasattr(self.model, 'forward'):
            # Get both student and teacher outputs for distillation
            result = self.model(X, y, time_feat, self.current_epoch, return_teacher_outputs=True)
            if isinstance(result, tuple) and len(result) == 2:
                outputs, teacher_outputs = result
                return outputs, {"teacher_outputs": teacher_outputs}
            else:
                # Fallback if teacher outputs not available
                outputs = result[0] if isinstance(result, tuple) else result
                return outputs, {}
        else:
            # Regular forward pass
            result = self.model(X, y, time_feat, self.current_epoch)
            if isinstance(result, tuple):
                outputs, aux = result
            else:
                outputs, aux = result, {}
            return outputs, aux

    def _compute_loss(self, outputs, targets, aux: Optional[Dict[str, torch.Tensor]] = None):
        """Enhanced loss computation with distillation support"""
        if aux is None:
            aux = {}
            
        # Base task loss
        base_loss = self.criterion(outputs, targets)
        total_loss = base_loss
        
        # Track individual loss components
        loss_components = {"task_loss": base_loss.item()}
        
        # Knowledge distillation loss
        if (self.distillation_enabled and 
            hasattr(self.model, 'compute_distillation_loss') and
            "teacher_outputs" in aux):
            
            teacher_outputs = aux["teacher_outputs"]
            distillation_loss, distill_components = self.model.compute_distillation_loss(
                outputs, teacher_outputs, targets, self.criterion
            )
            
            # Use current epoch's distillation weight
            distill_weight = self._get_distillation_weight(self.current_epoch)
            total_loss = (1 - distill_weight) * base_loss + distill_weight * distillation_loss
            
            # Track distillation components
            loss_components.update({
                f"distill_{k}": v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in distill_components.items()
            })
            loss_components["distillation_weight"] = distill_weight

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

        # KL divergence (e.g., for VAEs)
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
        """Enhanced training epoch with quantization and distillation support"""
        self.model.train()
        
        # Check if we should start QAT
        if self._should_start_qat(self.current_epoch):
            print(f"Starting QAT at epoch {self.current_epoch}")
            sample_batch = next(iter(dataloader))
            sample_input = sample_batch[0][:1].to(self.device)  # Take first sample
            self.prepare_quantization(sample_input)
            
            # Re-initialize optimizer for QAT
            self.optimizer = self._get_optimizer()
            
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
        """Enhanced evaluation with quantization support"""
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
        """Enhanced training with quantization and distillation support"""
        self._init_tracking()
        num_epochs = self.config["num_epochs"]
        if epochs is not None:
            num_epochs = epochs

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
                
                # Log quantization info
                if self.quantization_enabled and hasattr(self.model, 'get_model_size'):
                    model_info = self.model.get_model_size()
                    self.history["quantization_info"].append({
                        "epoch": epoch,
                        "is_quantized": model_info.get("is_quantized", False),
                        "model_size_mb": model_info.get("size_mb", 0),
                    })

                # Enhanced logging with distillation info
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                }
                
                # Add distillation weight to logs
                if self.distillation_enabled:
                    log_dict["distillation_weight"] = self._get_distillation_weight(epoch)
                
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

                # Update progress bar
                pbar_dict = {
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loader else "N/A",
                    "lr": current_lr,
                }
                if self.distillation_enabled:
                    pbar_dict["distill_w"] = f"{self._get_distillation_weight(epoch):.3f}"
                
                pbar.set_postfix(pbar_dict)

                if self.scheduler:
                    self.scheduler.step(val_loss if val_loader else train_loss)

        # Finalize quantization after training
        if self.quantization_mode == "qat":
            self.finalize_quantization()

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def save(self, path):
        """Enhanced save with quantization and distillation info"""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
            "quantization_config": self.quantization_config,
            "distillation_config": self.distillation_config,
        }
        
        # Add model size info if available
        if hasattr(self.model, 'get_model_size'):
            save_dict["model_info"] = self.model.get_model_size()
            
        torch.save(save_dict, path)

    def load(self, path):
        """Enhanced load with quantization and distillation info"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {})
        self.config.update(checkpoint.get("config", {}))
        
        # Restore quantization config
        if "quantization_config" in checkpoint:
            self.quantization_config = checkpoint["quantization_config"]
            self._setup_quantization()
            
        # Restore distillation config
        if "distillation_config" in checkpoint:
            self.distillation_config = checkpoint["distillation_config"]
            self._setup_distillation()

    def benchmark_model(self, sample_input: torch.Tensor, num_runs: int = 100):
        """Benchmark model performance with quantization comparison"""
        if not hasattr(self.model, 'benchmark_inference'):
            print("Model does not support benchmarking")
            return None
            
        sample_input = sample_input.to(self.device)
        results = self.model.benchmark_inference(sample_input, num_runs=num_runs)
        
        print("\nModel Performance Benchmark:")
        print(f"  Average inference time: {results['avg_inference_time_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"  Quantization mode: {results['quantization_mode']}")
        print(f"  Is quantized: {results['is_quantized']}")
        
        if hasattr(self.model, 'get_model_size'):
            size_info = self.model.get_model_size()
            print(f"  Model size: {size_info['size_mb']:.2f} MB")
            print(f"  Parameters: {size_info['parameters']:,}")
        
        return results

    def plot_learning_curves(self, figsize=(15, 10)):
        """Enhanced learning curves with distillation info"""
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
        
        # Distillation losses
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
        
        # Model size over time (for quantization)
        if self.history["quantization_info"]:
            epochs = [info["epoch"] for info in self.history["quantization_info"]]
            sizes = [info["model_size_mb"] for info in self.history["quantization_info"]]
            is_quantized = [info["is_quantized"] for info in self.history["quantization_info"]]
            
            axes[1, 1].plot(epochs, sizes, label="Model Size (MB)")
            
            # Mark quantization points
            for i, (epoch, quantized) in enumerate(zip(epochs, is_quantized)):
                if quantized:
                    axes[1, 1].axvline(x=epoch, color='red', linestyle='--', alpha=0.7)
                    if i == 0 or not is_quantized[i-1]:  # First quantization point
                        axes[1, 1].text(epoch, sizes[i], 'Quantized', rotation=90, va='bottom')
            
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Model Size (MB)")
            axes[1, 1].set_title("Model Size Over Training")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, "No quantization data", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Model Size")
        
        plt.tight_layout()
        plt.show()

    def get_distillation_info(self) -> Dict[str, Any]:
        """Get information about distillation setup"""
        if not self.distillation_enabled:
            return {"distillation_enabled": False}
            
        info = {
            "distillation_enabled": True,
            "distillation_mode": self.distillation_mode,
            "has_teacher": self.teacher_model is not None,
            "temperature": self.distillation_config["temperature"],
            "alpha": self.distillation_config["alpha"],
            "current_weight": self._get_distillation_weight(self.current_epoch),
        }
        
        if hasattr(self.model, 'get_distillation_info'):
            info.update(self.model.get_distillation_info())
            
        return info

    def get_quantization_info(self) -> Dict[str, Any]:
        """Get information about quantization setup"""
        if not self.quantization_enabled:
            return {"quantization_enabled": False}
            
        info = {
            "quantization_enabled": True,
            "quantization_mode": self.quantization_mode,
            "quantization_prepared": self.quantization_prepared,
            "backend": self.quantization_config["backend"],
            "bit_width": self.quantization_config["bit_width"],
        }
        
        if hasattr(self.model, 'get_model_size'):
            info.update(self.model.get_model_size())
            
        return info

    def print_training_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        # Basic info
        print(f"Total epochs: {len(self.history['train_losses'])}")
        print(f"Final train loss: {self.history['train_losses'][-1]:.6f}")
        if self.history['val_losses']:
            print(f"Final val loss: {self.history['val_losses'][-1]:.6f}")
            print(f"Best val loss: {self.best_val_loss:.6f}")
        
        # Distillation info
        if self.distillation_enabled:
            print("\nDISTILLATION INFO:")
            distill_info = self.get_distillation_info()
            for key, value in distill_info.items():
                print(f"  {key}: {value}")
        
        # Quantization info
        if self.quantization_enabled:
            print("\nQUANTIZATION INFO:")
            quant_info = self.get_quantization_info()
            for key, value in quant_info.items():
                print(f"  {key}: {value}")
        
        # Model size comparison
        if hasattr(self.model, 'get_model_size'):
            model_info = self.model.get_model_size()
            print(f"\nMODEL SIZE:")
            print(f"  Parameters: {model_info['parameters']:,}")
            print(f"  Model size: {model_info['size_mb']:.2f} MB")
            print(f"  Is quantized: {model_info['is_quantized']}")
        
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
        
        # Size comparison
        if hasattr(self.model, 'get_model_size') and hasattr(baseline_model, 'get_model_size'):
            current_size = self.model.get_model_size()
            baseline_size = baseline_model.get_model_size()
            
            print(f"\nCurrent model size: {current_size['size_mb']:.2f} MB")
            print(f"Baseline model size: {baseline_size['size_mb']:.2f} MB")
            print(f"Size reduction: {((baseline_size['size_mb'] - current_size['size_mb']) / baseline_size['size_mb'] * 100):.2f}%")
        
        # Speed comparison
        if hasattr(self.model, 'benchmark_inference'):
            sample_batch = next(iter(test_loader))
            sample_input = sample_batch[0][:1].to(self.device)
            
            current_bench = self.model.benchmark_inference(sample_input, num_runs=50)
            baseline_bench = baseline_model.benchmark_inference(sample_input, num_runs=50)
            
            print(f"\nCurrent model inference: {current_bench['avg_inference_time_ms']:.2f} ms")
            print(f"Baseline model inference: {baseline_bench['avg_inference_time_ms']:.2f} ms")
            speed_improvement = ((baseline_bench['avg_inference_time_ms'] - current_bench['avg_inference_time_ms']) / baseline_bench['avg_inference_time_ms'] * 100)
            print(f"Speed improvement: {speed_improvement:.2f}%")
        
        print("="*60)
        
        return {
            "current_loss": current_loss,
            "baseline_loss": baseline_loss,
            "loss_improvement": ((baseline_loss - current_loss) / baseline_loss * 100),
            "speed_improvement": speed_improvement if 'speed_improvement' in locals() else None
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