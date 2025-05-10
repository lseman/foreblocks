
import contextlib
import time
import copy
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from .vsgd import *

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
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[Callable] = None,
                 scheduler: Optional[Any] = None,
                 device: Optional[str] = None,
                 use_wandb: bool = False,
                 wandb_config: Optional[Dict[str, Any]] = None):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.use_wandb = use_wandb

        self.config = self._default_config()
        if config:
            self.config.update(config)

        self.optimizer = optimizer or self._get_optimizer()
        self.criterion = criterion or self._get_criterion()
        self.scheduler = scheduler or self._get_scheduler()
        self.scaler = GradScaler() if self.config['use_amp'] else None

        self._init_tracking()

        if self.use_wandb:
            import wandb
            wandb.init(**(wandb_config or {}))
            wandb.watch(self.model, log="all", log_freq=100)

    def _default_config(self):
        return {
            "num_epochs": 100, "learning_rate": 0.001, "weight_decay": 0.0,
            "patience": 10, "min_delta": 1e-4, "use_amp": True,
            "gradient_clip_val": None, "teacher_forcing_ratio": 0.5,
            "scheduler_type": None, "min_lr": 1e-6, "lr_step_size": 30, "lr_gamma": 0.1,
            "verbose": True, "log_interval": 10, "save_best_model": True,
            "save_model_path": None, "gradient_accumulation_steps": 1,
            "l1_regularization": 0.0, "kl_weight": 1.0
        }

    def set_config(self, key: str, value: Any):
        if key in self.config:
            self.config[key] = value
        else:
            raise KeyError(f"Config key '{key}' not found.")

    def _init_tracking(self):
        self.history = {"train_losses": [], "val_losses": [], "learning_rates": []}
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def _get_optimizer(self):
        print("Warning: Using custom VSGD optimizer.")
        #return torch.optim.Adam(self.model.parameters(),
        #                        lr=self.config['learning_rate'],
        #                        weight_decay=self.config['weight_decay'])
        return VSGD(self.model.parameters(),
                    lr=self.config['learning_rate'],
                    weight_decay=self.config['weight_decay'],
                    ghattg=30.0,
                    ps=1e-8,
                    tau1=0.81,
                    tau2=0.9,
                    eps=1e-8)

    def _get_criterion(self):
        return nn.MSELoss()

    def _get_scheduler(self):
        t = self.config["scheduler_type"]
        if t == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config['lr_step_size'], gamma=self.config['lr_gamma']
            )
        return None

    def _forward_pass(self, X, y):
        return self.model(X, y)

    def _compute_loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        if self.config["l1_regularization"] > 0:
            l1 = sum(torch.sum(torch.abs(p)) for p in self.model.parameters())
            loss += self.config["l1_regularization"] * l1
        if hasattr(self.model, 'kl_divergence'):
            loss += self.config["kl_weight"] * self.model.kl_divergence()
        return loss

    def _step_optimizer(self, loss, batch_idx, total_batches):
        grad_acc = self.config["gradient_accumulation_steps"]
        loss = loss / grad_acc
        if self.config['use_amp']:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1 == total_batches):
                if self.config['gradient_clip_val']:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1 == total_batches):
                if self.config['gradient_clip_val']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])
                self.optimizer.step()
                self.optimizer.zero_grad()

    def train_epoch(self, dataloader, callbacks=None):
        self.model.train()
        total_loss = 0.0
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            with autocast('cuda') if self.config['use_amp'] else contextlib.nullcontext():
                outputs = self._forward_pass(X, y)
                loss = self._compute_loss(outputs, y)
            self._step_optimizer(loss, batch_idx, len(dataloader))
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                with autocast('cuda') if self.config['use_amp'] else contextlib.nullcontext():
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
        return total_loss / len(dataloader.dataset)

    def train(self, train_loader, val_loader=None, callbacks=None, epochs=None):
        from tqdm import tqdm
        self._init_tracking()
        num_epochs = self.config['num_epochs']
        if epochs is not None:
            num_epochs = epochs

        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader, callbacks)
            self.history['train_losses'].append(train_loss)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}", end='')
                if val_loss is not None:
                    print(f", Val Loss = {val_loss:.4f}")
                else:
                    print("")
            val_loss = None
            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.history['val_losses'].append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr
                })

            if val_loader:
                if val_loss + self.config['min_delta'] < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    if self.config['save_model_path']:
                        self.save(self.config['save_model_path'])
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.config['patience']:
                    print("Early stopping triggered.")
                    break

            if self.scheduler:
                self.scheduler.step(val_loss if val_loader else train_loss)

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)


    def plot_learning_curves(self, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.plot(self.history['train_losses'], label='Train Loss')
        if self.history['val_losses']:
            plt.plot(self.history['val_losses'], label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.show()

        plt.figure(figsize=figsize)
        plt.plot(self.history['learning_rates'], label='Learning Rate')
        plt.xlabel('Epoch'); plt.ylabel('LR'); plt.grid(True)
        plt.tight_layout(); plt.show()

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

        target_len = y_val.shape[1]
        output_size = y_val.shape[2]
        forecast = torch.zeros((X_val.shape[0] + target_len - 1, output_size), device=self.device)
        count = torch.zeros_like(forecast)

        with torch.no_grad():
            for i in range(X_val.shape[0]):
                x = X_val[i].unsqueeze(0)  # [1, seq_len, input_size]
                pred = self.model(x).squeeze(0)  # [target_len, output_size]
                forecast[i:i + target_len] += pred
                count[i:i + target_len] += 1

        forecast = (forecast / count).squeeze()
        
        # Reconstruct ground truth for comparison
        aligned_truth = torch.zeros_like(forecast)
        truth_count = torch.zeros_like(forecast)

        y_val = y_val.squeeze()
        for i in range(y_val.shape[0]):
            aligned_truth[i:i + target_len] += y_val[i]
            truth_count[i:i + target_len] += 1

        aligned_truth = aligned_truth / torch.clamp(truth_count, min=1.0)

        # Compute metrics using existing helper
        metrics = self._compute_metrics(forecast, aligned_truth)

        print("\nValidation Forecast Error Metrics:")
        for k, v in metrics.items():
            print(f"  {k.upper():<5} = {v:.6f}")

        return metrics
    
    def plot_prediction(self, X_val: torch.Tensor, y_val: torch.Tensor, 
                    full_series: Optional[torch.Tensor] = None, offset: int = 0, 
                    figsize: Tuple[int, int] = (12, 8), show: bool = False) -> plt.Figure:
        """
        Plot predicted sequence over the validation data, aligned to form a full series forecast.
        
        Args:
            X_val: Tensor of shape [N, seq_len, input_size]
            y_val: Tensor of shape [N, target_len, output_size]
            full_series: (Optional) Original full time series for reference
            offset: (Optional) Index offset for where the validation data starts in the full series
            figsize: (Optional) Figure size as (width, height) in inches
            show: (Optional) Whether to display the plot with plt.show()
            
        Returns:
            matplotlib Figure object for further customization or saving
        """
        self.model.eval()
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        target_len = y_val.shape[1]
        output_size = y_val.shape[2]
        
        forecast = torch.zeros((X_val.shape[0] + target_len - 1, output_size), device=self.device)
        count = torch.zeros_like(forecast)
        
        with torch.no_grad():
            for i in range(X_val.shape[0]):
                x = X_val[i].unsqueeze(0)  # [1, seq_len, input_size]
                pred = self.model(x).squeeze(0)  # [target_len, output_size]
                forecast[i:i + target_len] += pred
                count[i:i + target_len] += 1
        
        forecast = (forecast / count).squeeze().cpu().numpy()
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        if full_series is not None:
            full_series = full_series.squeeze().cpu().numpy()
            forecast_start = offset + X_val.shape[1]
            
            ax.plot(np.arange(len(full_series)), full_series, label="Original", alpha=0.5)
            ax.plot(np.arange(forecast_start, forecast_start + len(forecast)), 
                    forecast, label="Forecast", color="orange")
            ax.axvline(x=forecast_start, color="gray", linestyle="--", label="Forecast Start")
            
            ax.set_title("Full Series with Forecast")
        else:
            ax.plot(forecast, label="Forecast", color="orange")
            ax.set_title("Validation Prediction")
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Only show if requested
        if show:
            plt.show()
        
        return fig

    def _compute_metrics(self, prediction: Union[np.ndarray, torch.Tensor],
                        target: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Compute error metrics between prediction and target:
        MSE, RMSE, MAE.

        Args:
            prediction: Forecasted values as NumPy array or Torch tensor
            target: Ground truth values as NumPy array or Torch tensor

        Returns:
            Dictionary with keys 'mse', 'rmse', and 'mae'
        """
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        mse = np.mean((prediction - target) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(prediction - target))

        return {"mse": mse, "rmse": rmse, "mae": mae}
