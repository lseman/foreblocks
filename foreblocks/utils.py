
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
                 device: Optional[str] = None):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.config = self._default_config()
        if config:
            self.config.update(config)

        self.optimizer = optimizer or self._get_optimizer()
        self.criterion = criterion or self._get_criterion()
        self.scheduler = scheduler or self._get_scheduler()
        self.scaler = GradScaler() if self.config['use_amp'] else None

        self._init_tracking()

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

    def _init_tracking(self):
        self.history = {"train_losses": [], "val_losses": [], "learning_rates": []}
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(),
                                 lr=self.config['learning_rate'],
                                 weight_decay=self.config['weight_decay'])

    def _get_criterion(self):
        return nn.MSELoss()

    def _get_scheduler(self):
        t = self.config["scheduler_type"]
        if t == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config['lr_step_size'], gamma=self.config['lr_gamma']
            )
        return None

    def _run_callbacks(self, hook, callbacks, *args):
        for cb in callbacks or []:
            fn = getattr(cb, hook, None)
            if callable(fn):
                fn(*args)

    def _forward_pass(self, X, y):
        if hasattr(self.model, 'forward_with_teacher_forcing'):
            return self.model.forward_with_teacher_forcing(X, y, self.config['teacher_forcing_ratio'])
        return self.model(X)

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
            self._run_callbacks('on_batch_start', callbacks, self, batch_idx)
            with autocast('cuda') if self.config['use_amp'] else contextlib.nullcontext():
                outputs = self._forward_pass(X, y)
                loss = self._compute_loss(outputs, y)

            self._step_optimizer(loss, batch_idx, len(dataloader))
            total_loss += loss.item()
            self._run_callbacks('on_batch_end', callbacks, self, batch_idx, {"loss": loss.item()})

        return total_loss / len(dataloader)

    def evaluate(self, dataloader, callbacks=None):
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

    def train(self, train_loader, val_loader=None, callbacks=None):
            from tqdm import tqdm
            self._init_tracking()
            num_epochs = self.config['num_epochs']
            for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
                self.current_epoch = epoch
                train_loss = self.train_epoch(train_loader, callbacks)
                self.history['train_losses'].append(train_loss)

                val_loss = None
                if val_loader:
                    val_loss = self.evaluate(val_loader, callbacks)
                    self.history['val_losses'].append(val_loss)

                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}", end='')
                    if val_loss is not None:
                        print(f", Val Loss = {val_loss:.4f}")
                    else:
                        print()

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