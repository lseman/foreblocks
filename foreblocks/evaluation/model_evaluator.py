

# ============================================================================
# Model Evaluator (unchanged from original)
# ============================================================================

from __future__ import annotations
import torch
import contextlib
from torch.amp import autocast
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple
import numpy as np

class ModelEvaluator:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.device = trainer.device

    @torch.no_grad()
    def predict(self, X: torch.Tensor, batch_size: int = 256, use_amp: bool = True) -> torch.Tensor:
        self.model.eval()
        X = X.to(self.device)
        predictions = []
        amp_ctx = (autocast("cuda") if use_amp and self.device == "cuda" else contextlib.nullcontext())

        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            with amp_ctx:
                output = self.model(batch)
                output = output[0] if isinstance(output, tuple) else output
                predictions.append(output.cpu())

        return torch.cat(predictions, dim=0)

    def cross_validation(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_windows: int,
        horizon: int,
        step_size: Optional[int] = None,
        batch_size: int = 256
    ) -> Dict[str, Any]:
        if step_size is None:
            step_size = horizon

        print(f"Cross-validation: {n_windows} windows, horizon={horizon}, step_size={step_size}")

        window_metrics = []
        all_predictions = []
        all_targets = []

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + horizon

            if end_idx > len(X):
                break

            X_window = X[start_idx:end_idx]
            y_window = y[start_idx:end_idx]

            preds = self.predict(X_window, batch_size)

            all_predictions.append(preds)
            all_targets.append(y_window)

            metrics = self._compute_window_metrics(preds, y_window)
            metrics['window'] = i
            metrics['start_idx'] = start_idx
            metrics['end_idx'] = end_idx
            window_metrics.append(metrics)

            print(f"  Window {i+1}/{n_windows}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

        all_preds_cat = torch.cat(all_predictions, dim=0)
        all_targets_cat = torch.cat(all_targets, dim=0)

        overall_metrics = self._compute_window_metrics(all_preds_cat, all_targets_cat)

        return {
            'window_metrics': window_metrics,
            'overall': overall_metrics,
            'predictions': all_preds_cat,
            'targets': all_targets_cat,
            'n_windows': len(window_metrics),
            'total_points': len(all_preds_cat)
        }

    def _compute_window_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        targets = targets.to(predictions.device)
        diff = (predictions - targets).float()

        mse = (diff**2).mean().item()
        mae = diff.abs().mean().item()
        rmse = mse**0.5

        mape = (diff.abs() / (targets.abs() + 1e-8)).mean().item() * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    def compute_metrics(self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> Dict[str, float]:
        predictions = self.predict(X, batch_size)
        return self._compute_window_metrics(predictions, y)

    def plot_cv_results(self, cv_results: Dict[str, Any], figsize: Tuple[int, int] = (15, 8)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        window_metrics = cv_results['window_metrics']
        windows = [m['window'] for m in window_metrics]

        metrics_to_plot = ['mae', 'rmse', 'mape']
        for idx, metric in enumerate(metrics_to_plot):
            row, col = idx // 2, idx % 2
            values = [m[metric] for m in window_metrics]
            axes[row, col].plot(windows, values, 'o-', label=f'Per-window {metric.upper()}')
            axes[row, col].axhline(cv_results['overall'][metric],
                                  color='r', linestyle='--',
                                  label=f'Overall {metric.upper()}')
            axes[row, col].set_xlabel('Window')
            axes[row, col].set_ylabel(metric.upper())
            axes[row, col].set_title(f'{metric.upper()} across windows')
            axes[row, col].legend()
            axes[row, col].grid(True)

        axes[1, 1].axis('off')
        summary = f"""Cross-Validation Summary

Total Windows: {cv_results['n_windows']}
Total Points: {cv_results['total_points']}

Overall Metrics:
  MAE:  {cv_results['overall']['mae']:.4f}
  RMSE: {cv_results['overall']['rmse']:.4f}
  MAPE: {cv_results['overall']['mape']:.2f}%

Per-Window Stats:
  MAE:  {np.mean([m['mae'] for m in window_metrics]):.4f} ± {np.std([m['mae'] for m in window_metrics]):.4f}
  RMSE: {np.mean([m['rmse'] for m in window_metrics]):.4f} ± {np.std([m['rmse'] for m in window_metrics]):.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary, fontsize=10, family='monospace', va='center')

        plt.tight_layout()
        return fig

    def plot_learning_curves(self, figsize: Tuple[int, int] = (15, 5)):
        history = self.trainer.history
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].plot(history.train_losses, label="Train")
        if history.val_losses:
            axes[0].plot(history.val_losses, label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.learning_rates)
        axes[1].set_title("Learning Rate")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("LR")
        axes[1].grid(True)

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