# ─── Standard Library ─────────────────────────────────────────────
import concurrent.futures
import copy
import logging
import random
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum


import matplotlib.patches as patches

# ─── Third-Party Libraries ────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ─── Local Imports ─────────────────────────────────────────────────
from .darts import *
from .darts_metrics import *

import higher
# Optional: configure a custom logger
logger = logging.getLogger("NASLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class RegularizationType(Enum):
    """Types of regularization for architecture search"""
    ENTROPY = "entropy"
    KL_DIVERGENCE = "kl_divergence"
    L2_NORM = "l2_norm"
    DIVERSITY = "diversity"
    SPARSITY = "sparsity"
    EFFICIENCY = "efficiency"

class ArchitectureRegularizer:
    """Helper class for different types of architecture regularization"""
    
    def __init__(self, reg_types: List[RegularizationType], weights: List[float] = None):
        self.reg_types = reg_types
        self.weights = weights or [1.0] * len(reg_types)
        assert len(self.reg_types) == len(self.weights), "Number of weights must match number of regularization types"
    
    def compute_regularization(self, model: nn.Module, arch_params: List[torch.Tensor], 
                             epoch: int = 0, total_epochs: int = 100) -> Dict[str, torch.Tensor]:
        """Compute all specified regularization terms"""
        reg_losses = {}
        total_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for reg_type, weight in zip(self.reg_types, self.weights):
            reg_loss = self._compute_single_regularization(
                model, arch_params, reg_type, epoch, total_epochs
            )
            reg_losses[reg_type.value] = reg_loss
            total_reg += weight * reg_loss
        
        reg_losses['total'] = total_reg
        return reg_losses
    
    def _compute_single_regularization(self, model: nn.Module, arch_params: List[torch.Tensor], 
                                     reg_type: RegularizationType, epoch: int, total_epochs: int) -> torch.Tensor:
        """Compute a single type of regularization"""
        device = next(model.parameters()).device
        
        if reg_type == RegularizationType.ENTROPY:
            return self._entropy_regularization(arch_params)
        elif reg_type == RegularizationType.KL_DIVERGENCE:
            return self._kl_divergence_regularization(arch_params)
        elif reg_type == RegularizationType.L2_NORM:
            return self._l2_norm_regularization(arch_params)
        elif reg_type == RegularizationType.DIVERSITY:
            return self._diversity_regularization(model)
        elif reg_type == RegularizationType.SPARSITY:
            return self._sparsity_regularization(arch_params, epoch, total_epochs)
        elif reg_type == RegularizationType.EFFICIENCY:
            return self._efficiency_regularization(model)
        else:
            return torch.tensor(0.0, device=device)
    
    def _entropy_regularization(self, arch_params: List[torch.Tensor]) -> torch.Tensor:
        """Entropy regularization to encourage exploration"""
        total_entropy = torch.tensor(0.0, device=arch_params[0].device)
        
        for param in arch_params:
            if param.dim() >= 1:
                # Apply softmax along the last dimension
                probs = F.softmax(param.view(-1, param.size(-1)), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                # Encourage exploration by penalizing low entropy
                total_entropy += (1.0 - entropy / np.log(param.size(-1)))
        
        return total_entropy
    
    def _kl_divergence_regularization(self, arch_params: List[torch.Tensor]) -> torch.Tensor:
        """KL divergence from uniform distribution to encourage diversity"""
        total_kl = torch.tensor(0.0, device=arch_params[0].device)
        
        for param in arch_params:
            if param.dim() >= 1:
                # Reshape to 2D for easier processing
                flat_param = param.view(-1, param.size(-1))
                probs = F.softmax(flat_param, dim=-1)
                
                # Uniform distribution target
                uniform = torch.ones_like(probs) / probs.size(-1)
                
                # KL divergence: KL(P||Q) = sum(P * log(P/Q))
                kl_div = (probs * torch.log(probs / (uniform + 1e-8) + 1e-8)).sum(dim=-1).mean()
                total_kl += kl_div
        
        return total_kl
    
    def _l2_norm_regularization(self, arch_params: List[torch.Tensor]) -> torch.Tensor:
        """L2 norm regularization on architecture parameters"""
        total_l2 = torch.tensor(0.0, device=arch_params[0].device)
        
        for param in arch_params:
            total_l2 += torch.norm(param, p=2)
        
        return total_l2
    
    def _diversity_regularization(self, model: nn.Module) -> torch.Tensor:
        """Encourage diversity across different parts of the architecture"""
        diversity_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Collect all architecture weights
        all_weights = []
        for module in model.modules():
            if hasattr(module, 'get_alphas'):
                try:
                    alphas = module.get_alphas()
                    if alphas.numel() > 0:
                        all_weights.append(F.softmax(alphas, dim=-1))
                except:
                    continue
        
        if len(all_weights) >= 2:
            # Compute pairwise diversity (negative cosine similarity)
            for i in range(len(all_weights)):
                for j in range(i + 1, len(all_weights)):
                    w1, w2 = all_weights[i], all_weights[j]
                    
                    # Handle different sizes by truncating to smaller size
                    min_size = min(w1.size(0), w2.size(0))
                    w1_trunc = w1[:min_size]
                    w2_trunc = w2[:min_size]
                    
                    # Cosine similarity
                    cos_sim = F.cosine_similarity(w1_trunc, w2_trunc, dim=0)
                    # Encourage diversity (low similarity)
                    diversity_loss += cos_sim
        
        return diversity_loss
    
    def _sparsity_regularization(self, arch_params: List[torch.Tensor], 
                               epoch: int, total_epochs: int) -> torch.Tensor:
        """Sparsity regularization that increases over time"""
        sparsity_loss = torch.tensor(0.0, device=arch_params[0].device)
        
        # Increase sparsity pressure over time
        sparsity_weight = min(1.0, epoch / (total_epochs * 0.8))
        
        for param in arch_params:
            if param.dim() >= 1:
                probs = F.softmax(param.view(-1, param.size(-1)), dim=-1)
                # L1 penalty on probabilities to encourage sparsity
                sparsity_loss += sparsity_weight * probs.sum()
        
        return sparsity_loss
    
    def _efficiency_regularization(self, model: nn.Module) -> torch.Tensor:
        """Efficiency regularization based on operation complexity"""
        efficiency_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Operation efficiency scores (lower is more efficient)
        op_costs = {
            "Identity": 0.0,
            "ResidualMLP": 0.2,
            "TimeConv": 0.3,
            "TCN": 0.5,
            "ConvMixer": 0.4,
            "Fourier": 0.6,
            "Wavelet": 0.6,
            "GRN": 0.4,
            "MultiScaleConv": 0.7,
            "PyramidConv": 0.8,
        }
        
        for module in model.modules():
            if hasattr(module, 'get_alphas') and hasattr(module, 'available_ops'):
                try:
                    alphas = module.get_alphas()
                    probs = F.softmax(alphas, dim=-1)
                    
                    # Compute expected cost
                    for i, op_name in enumerate(module.available_ops):
                        if i < len(probs):
                            cost = op_costs.get(op_name, 0.5)  # Default medium cost
                            efficiency_loss += probs[i] * cost
                except:
                    continue
        
        return efficiency_loss

class TemperatureScheduler:
    """Advanced temperature scheduling for architecture search"""
    
    def __init__(self, initial_temp: float = 2.0, final_temp: float = 0.1, 
                 schedule_type: str = "cosine", warmup_epochs: int = 5):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
    
    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """Get temperature for current epoch"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.initial_temp
        
        progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        
        if self.schedule_type == "cosine":
            temp = self.final_temp + (self.initial_temp - self.final_temp) * \
                   (1 + np.cos(np.pi * progress)) / 2
        elif self.schedule_type == "exponential":
            decay_rate = np.log(self.final_temp / self.initial_temp) / (total_epochs - self.warmup_epochs)
            temp = self.initial_temp * np.exp(decay_rate * (epoch - self.warmup_epochs))
        elif self.schedule_type == "linear":
            temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress
        elif self.schedule_type == "step":
            # Step decay at specific epochs
            if progress < 0.3:
                temp = self.initial_temp
            elif progress < 0.7:
                temp = self.initial_temp * 0.5
            else:
                temp = self.final_temp
        else:
            temp = self.initial_temp
        
        return max(temp, self.final_temp)

class DARTSTrainer:
    """
    Comprehensive DARTS trainer with search, training, and evaluation capabilities.

    This class encapsulates the entire DARTS workflow:
    - Architecture search with zero-cost metrics
    - DARTS training with mixed operations
    - Final model training with fixed architecture
    - Multi-fidelity search strategies
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [32, 64, 128],
        forecast_horizon: int = 6,
        seq_length: int = 12,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        all_ops: Optional[List[str]] = None,
    ):
        """
        Initialize DARTS trainer.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of possible hidden dimensions
            forecast_horizon: Number of steps to forecast
            seq_length: Input sequence length
            device: Training device
            all_ops: List of operations to search over
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.forecast_horizon = forecast_horizon
        self.seq_length = seq_length
        self.device = device
        self.use_gumbel = True  # Use Gumbel-softmax for sharper separation

        self.all_ops = all_ops or [
            "Identity",
            "TimeConv",
            "GRN",
            "Wavelet",
            "Fourier",
            "TCN",
            "ResidualMLP",
            "ConvMixer",
            "MultiScaleConv",
            "PyramidConv",
        ]

        # Training history
        self.search_history = []
        self.training_history = []

        print(f"🚀 DARTSTrainer initialized on {device}")
        print(f"   Input dim: {input_dim}, Forecast horizon: {forecast_horizon}")
        print(f"   Available operations: {len(self.all_ops)}")

    def _get_loss_function(self, loss_type: str):
        """Get loss function by name."""
        loss_functions = {
            "huber": lambda p, t: F.huber_loss(p, t, delta=0.1),
            "mse": F.mse_loss,
            "mae": F.l1_loss,
            "smooth_l1": F.smooth_l1_loss,
        }
        return loss_functions.get(loss_type, loss_functions["huber"])

    def _create_progress_bar(self, iterable, desc: str, leave: bool = True, **kwargs):
        """Create standardized progress bar."""
        # Set default unit if not provided
        if "unit" not in kwargs:
            kwargs["unit"] = "batch"
        return tqdm(iterable, desc=desc, leave=leave, **kwargs)

    def _evaluate_model(
        self, model: nn.Module, dataloader, loss_type: str = "huber"
    ) -> float:
        """Evaluate model on given dataloader."""
        model.eval()
        loss_fn = self._get_loss_function(loss_type)
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, *_ in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                preds = model(batch_x)
                total_loss += loss_fn(preds, batch_y).item()

        return total_loss / len(dataloader)

    def _compute_metrics(
        self, preds: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive regression metrics."""
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))
        mape = np.mean(np.abs((preds - targets) / (np.abs(targets) + 1e-8))) * 100

        # R² score
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2_score": r2_score,
        }

    def _plot_training_curve(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training Progress",
        save_path: str = None,
    ):
        """Plot and save training curves."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss", alpha=0.8)
        plt.plot(
            epochs, val_losses, "r-", linewidth=2, label="Validation Loss", alpha=0.8
        )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.7)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Training curve saved to {save_path}")

        plt.close()

    def evaluate_zero_cost_metrics(
        self, model: nn.Module, dataloader, max_samples: int = 32, num_batches: int = 1
    ) -> Dict[str, Any]:
        """Evaluate model using zero-cost metrics."""

        def create_custom_config(
            max_samples: int = 32,
            max_outputs: int = 10,
            timeout_seconds: float = 30.0,
            enable_mixed_precision: bool = False,
        ) -> Config:
            return Config(max_samples=max_samples, max_outputs=max_outputs)

        config = create_custom_config(
            max_samples=max_samples,
            max_outputs=10,
            timeout_seconds=30.0,
            enable_mixed_precision=True,
        )

        nas_evaluator = ZeroCostNAS(config=config)
        results = nas_evaluator.evaluate_model(
            model, dataloader, self.device, num_batches=num_batches
        )

        return results

    def train_darts_model(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        epochs: int = 50,
        arch_learning_rate: float = 1e-2,
        model_learning_rate: float = 1e-3,
        arch_weight_decay: float = 1e-3,
        model_weight_decay: float = 1e-4,
        patience: int = 10,
        loss_type: str = "huber",
        use_swa: bool = False,
        warmup_epochs: int = 2,
        architecture_update_freq: int = 3,
        diversity_check_freq: int = 1,
        progressive_shrinking: bool = True,
        use_bilevel_optimization: bool = True,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        verbose: bool = True,
        regularization_types: Optional[List[str]] = None,
        regularization_weights: Optional[List[float]] = None,
        temperature_schedule: str = "cosine",
    ) -> Dict[str, Any]:
        """Simplified DARTS training with essential features"""

        model = model.to(self.device)
        start_time = time.time()

        # Model compilation (if available)
        # if hasattr(torch, 'compile'):
        #     try:
        #         model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
        #         if verbose:
        #             print("✓ Model compiled")
        #     except Exception as e:
        #         if verbose:
        #             print(f"Warning: Compilation failed ({e})")

        # Separate architecture and model parameters
        arch_params, model_params = [], []
        for name, param in model.named_parameters():
            #print(f"🔍 Parameter: {name} ({param.numel()})")
            if any(arch_name in name for arch_name in ["alphas", "arch_", "alpha_"]):
                arch_params.append(param)
                # print(f"🔍 Architecture param: {name} ({param.numel()})")
            else:
                model_params.append(param)

        # Add encoder/decoder alphas
        for module_name in ["forecast_encoder", "forecast_decoder"]:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if hasattr(module, "alphas"):
                    arch_params.append(module.alphas)
                if hasattr(module, "attention_alphas"):
                    arch_params.append(module.attention_alphas)

        if verbose:
            print(f"📊 Architecture params: {len(arch_params)}, Model params: {len(model_params)}")

        # Setup optimizers with fused operations if available
        try:
            arch_optimizer = torch.optim.Adam(
                arch_params, lr=arch_learning_rate, betas=(0.5, 0.999), 
                weight_decay=arch_weight_decay, fused=True
            )
            model_optimizer = torch.optim.Adam(
                model_params, lr=model_learning_rate, 
                weight_decay=model_weight_decay, fused=True
            )
        except:
            arch_optimizer = torch.optim.Adam(
                arch_params, lr=arch_learning_rate, betas=(0.5, 0.999), 
                weight_decay=arch_weight_decay
            )
            model_optimizer = torch.optim.AdamW(
                model_params, lr=model_learning_rate, weight_decay=model_weight_decay
            )

        # Learning rate schedulers
        arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            arch_optimizer, T_max=epochs, eta_min=arch_learning_rate * 0.01
        )
        model_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model_optimizer, max_lr=model_learning_rate, epochs=epochs,
            steps_per_epoch=len(train_loader) // gradient_accumulation_steps,
            pct_start=0.3, anneal_strategy="cos"
        )

        # Loss function and data loaders
        loss_fn = self._get_loss_function(loss_type)
        if use_bilevel_optimization:
            train_arch_loader, train_model_loader = self._create_bilevel_loaders(train_loader)
            val_arch_iter = iter(val_loader)
        else:
            train_model_loader = train_loader
            val_arch_iter = iter(val_loader)

        # SWA setup
        swa_model, swa_start = None, None
        if use_swa:
            swa_start = max(epochs // 2, warmup_epochs + 5)
            swa_model = torch.optim.swa_utils.AveragedModel(model).to(self.device)

        # Mixed precision
        scaler = GradScaler(enabled=use_amp and self.device.startswith('cuda'))

        # Training state
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        train_losses, val_losses, alpha_values = [], [], []
        diversity_scores = []

        if verbose:
            print(f"🔍 Training DARTS for {epochs} epochs")
            print(f"   Arch LR: {arch_learning_rate}, Model LR: {model_learning_rate}")
            print(f"   Bilevel: {use_bilevel_optimization}, SWA: {use_swa}, AMP: {use_amp}")
            print("-" * 60)

        # Main training loop
        epoch_pbar = self._create_progress_bar(range(epochs), "DARTS", unit="epoch") if verbose else range(epochs)
        # Setup regularization
        if regularization_types is None:
            regularization_types = ["kl_divergence", "efficiency"]
        if regularization_weights is None:
            regularization_weights = [0.05, 0.01]
        
        reg_types = [RegularizationType(rt) for rt in regularization_types]
        regularizer = ArchitectureRegularizer(reg_types, regularization_weights)
        
        # Setup temperature scheduler
        temp_scheduler = TemperatureScheduler(
            initial_temp=2.0, final_temp=0.1, 
            schedule_type=temperature_schedule, 
            warmup_epochs=warmup_epochs
        )
        for epoch in epoch_pbar:
            model.train()
            
            # Dynamic temperature
            current_temperature = temp_scheduler.get_temperature(epoch, epochs)

            if hasattr(model, "set_temperature"):
                model.set_temperature(current_temperature)

            # Track alphas every 5 epochs
            if epoch % 5 == 0:
                alpha_values.append(self._extract_alpha_values(model))

            # Architecture updates
            if epoch >= warmup_epochs and epoch % architecture_update_freq == 0:
                for _ in range(2):  # 2 architecture steps
                    try:
                        if use_bilevel_optimization:
                            arch_batch = next(iter(train_arch_loader))
                        else:
                            arch_batch = next(val_arch_iter)
                    except StopIteration:
                        val_arch_iter = iter(val_loader)
                        arch_batch = next(val_arch_iter)

                    arch_x, arch_y = arch_batch[0].to(self.device), arch_batch[1].to(self.device)
                    arch_optimizer.zero_grad()

                    with autocast("cuda" if self.device.startswith("cuda") else "cpu", enabled=use_amp):
                        arch_preds = model(arch_x)
                        arch_loss = loss_fn(arch_preds, arch_y)
                        
                        # Simple regularization
                        reg_losses = regularizer.compute_regularization(
                            model, arch_params, epoch, epochs
                        )
                        
                        total_arch_loss = arch_loss + reg_losses['total']

                    scaler.scale(total_arch_loss).backward()
                    scaler.unscale_(arch_optimizer)
                    torch.nn.utils.clip_grad_norm_(arch_params, max_norm=3.0)
                    scaler.step(arch_optimizer)
                    scaler.update()

                arch_scheduler.step()
                # print("Architecture gradients:")
                # for name, param in model.named_parameters():
                #     if any(param is p for p in arch_params):  # ✅ identity check
                #         if param.grad is not None:
                #             print(f"{name} grad norm: {param.grad.norm().item():.4e}")
                #         else:
                #             print(f"{name} grad: None")

            # Model parameter updates
            epoch_train_loss = 0.0
            batch_pbar = (
                self._create_progress_bar(enumerate(train_model_loader), f"Epoch {epoch+1:3d}", 
                                        leave=False, total=len(train_model_loader))
                if verbose else enumerate(train_model_loader)
            )

            model_optimizer.zero_grad()
            for batch_idx, (batch_x, batch_y, *_) in batch_pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                with autocast("cuda" if self.device.startswith("cuda") else "cpu", enabled=use_amp):
                    preds = model(batch_x)
                    loss = loss_fn(preds, batch_y) / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(model_optimizer)
                    torch.nn.utils.clip_grad_norm_(model_params, max_norm=5.0)
                    scaler.step(model_optimizer)
                    scaler.update()
                    model_scheduler.step()
                    model_optimizer.zero_grad()

                epoch_train_loss += loss.item() * gradient_accumulation_steps

                if verbose and hasattr(batch_pbar, 'set_postfix'):
                    batch_pbar.set_postfix({
                        "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                        "avg": f"{epoch_train_loss/(batch_idx+1):.4f}",
                    })

            if verbose and hasattr(batch_pbar, 'close'):
                batch_pbar.close()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_pbar = self._create_progress_bar(val_loader, "Val", leave=False) if verbose else val_loader
                
                for batch_data in val_pbar:
                    batch_x, batch_y = batch_data[0].to(self.device), batch_data[1].to(self.device)
                    
                    with autocast("cuda" if self.device.startswith("cuda") else "cpu", enabled=use_amp):
                        preds = model(batch_x)
                        val_loss += loss_fn(preds, batch_y).item()
                    
                    if verbose and hasattr(val_pbar, 'set_postfix'):
                        val_pbar.set_postfix({"val_loss": f"{val_loss/(len(val_loader)):.4f}"})
                
                if verbose and hasattr(val_pbar, 'close'):
                    val_pbar.close()

            avg_train_loss = epoch_train_loss / len(train_model_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Architecture health check
            if epoch % diversity_check_freq == 0 and hasattr(model, "validate_architecture_health"):
                health = model.validate_architecture_health()
                diversity_scores.append({
                    "epoch": epoch,
                    "health_score": health["health_score"],
                    "avg_identity_dominance": health["avg_identity_dominance"],
                    "issues": len(health["issues"]),
                })

                if health["health_score"] < 0.4:
                    if verbose:
                        print(f"\n⚠️ Architecture health low ({health['health_score']:.3f})")
                    if hasattr(model, "apply_architecture_fixes"):
                        model.apply_architecture_fixes()

            # Progressive shrinking
            if progressive_shrinking and epoch > epochs * 0.6:
                if hasattr(model, "prune_weak_operations"):
                    threshold = 0.1 + 0.1 * (epoch - epochs * 0.6) / (epochs * 0.4)
                    model.prune_weak_operations(threshold=threshold)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
                
                if use_swa and swa_model and epoch >= swa_start:
                    swa_model.update_parameters(model)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            # Progress update
            if verbose and hasattr(epoch_pbar, 'set_postfix'):
                epoch_pbar.set_postfix({
                    "train": f"{avg_train_loss:.4f}",
                    "val": f"{avg_val_loss:.4f}",
                    "best": f"{best_val_loss:.4f}",
                    "patience": f"{patience_counter}/{patience}",
                })

        if verbose and hasattr(epoch_pbar, 'close'):
            epoch_pbar.close()

        training_time = time.time() - start_time

        # SWA finalization
        if use_swa and swa_model and epoch >= swa_start:
            if verbose:
                print("\n🔄 Finalizing SWA...")
            try:
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=self.device)
                swa_val_loss = self._evaluate_model(swa_model, val_loader, loss_type)
                if swa_val_loss < best_val_loss:
                    if verbose:
                        print("✓ SWA model is better")
                    best_state = {k: v.detach().clone() for k, v in swa_model.state_dict().items()}
                    best_val_loss = swa_val_loss
            except Exception as e:
                if verbose:
                    print(f"Warning: SWA failed ({e})")

        # Load best model
        try:
            model.load_state_dict(best_state)
        except RuntimeError as e:
            if "Missing key" in str(e):
                # Handle missing buffers
                filtered_state = {k: v for k, v in best_state.items() 
                                if not k.startswith('_forecast_buffer') and not k.startswith('_context_buffer')}
                model.load_state_dict(filtered_state, strict=False)
            else:
                raise e

        # Ensure float32
        if hasattr(model, 'ensure_float32_dtype'):
            model.ensure_float32_dtype()
        else:
            model = model.float()

        # Final results
        #final_architecture = self._derive_final_architecture(model)
        final_metrics = self._compute_final_metrics(model, val_loader)

        if verbose:
            print(f"\n🎯 Training completed in {training_time:.1f}s")
            print(f"Best Val Loss: {best_val_loss:.6f}")
            print(f"MSE: {final_metrics['mse']:.6f} | MAE: {final_metrics['mae']:.6f}")

        results = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "alpha_values": alpha_values,
            "diversity_scores": diversity_scores,
            "final_architecture": model,
            "best_val_loss": best_val_loss,
            "training_time": training_time,
            "final_metrics": final_metrics,
        }

        self.training_history.append(results)
        return results

    def _compute_architecture_regularization(
        self, model, epoch, epochs, temperature, alpha_values
    ):
        """Consolidated architecture regularization computation"""
        reg_loss = 0.0

        # Dynamic weights
        entropy_weight = 0.1 * (1.0 - epoch / epochs)
        identity_weight = 0.2
        smoothness_weight = 0.05 * (epoch / epochs)
        balance_weight = 0.1 * max(0, 1.0 - 2 * epoch / epochs)

        # Entropy regularization
        entropy_reg = self._calculate_entropy_regularization(model, temperature)
        reg_loss += entropy_weight * entropy_reg

        # Identity penalty
        identity_penalty = self._calculate_identity_penalty(model)
        reg_loss += identity_weight * identity_penalty

        # Smoothness penalty
        if len(alpha_values) > 1:
            smoothness_penalty = self._calculate_smoothness_penalty(model, alpha_values)
            reg_loss += smoothness_weight * smoothness_penalty

        # Operation balance
        balance_penalty = self._calculate_operation_balance_penalty(model)
        reg_loss += balance_weight * balance_penalty

        return reg_loss

    # === HELPER METHODS ===

    def _print_architecture_summary(self, architecture):
        """Print a summary of the final architecture"""
        for cell_name, cell_arch in architecture.items():
            if isinstance(cell_arch, dict) and "edge_0" in cell_arch:
                for edge_name, edge_info in cell_arch.items():
                    op_name = edge_info.get("operation", "Unknown")
                    weight = edge_info.get("weight", 0.0)
                    print(
                        f"   {cell_name.title()}, {edge_name.title()}: {op_name} (weight: {weight:.3f})"
                    )
            elif isinstance(cell_arch, dict) and "type" in cell_arch:
                # Encoder/decoder info
                op_type = cell_arch.get("type", "Unknown")
                weight = cell_arch.get("weight", 0.0)
                print(
                    f"   → Fixing {cell_name.title()}: {op_type.upper()} (weight: {weight:.3f})"
                )

    def _create_bilevel_loaders(self, train_loader):
        """Create separate loaders for bilevel optimization"""
        # Split training data: 70% for model weights, 30% for architecture
        dataset = train_loader.dataset
        train_size = int(0.7 * len(dataset))
        arch_size = len(dataset) - train_size

        train_dataset, arch_dataset = torch.utils.data.random_split(
            dataset, [train_size, arch_size]
        )

        train_model_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            #num_workers=train_loader.num_workers,
            #pin_memory=train_loader.pin_memory,
        )

        train_arch_loader = torch.utils.data.DataLoader(
            arch_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            #num_workers=train_loader.num_workers,
            #pin_memory=train_loader.pin_memory,
        )

        return train_arch_loader, train_model_loader

    def _extract_alpha_values(self, model):
        """Extract current alpha values for tracking"""
        current_alphas = []
        if hasattr(model, "cells"):
            for i, cell in enumerate(model.cells):
                if hasattr(cell, "edges"):
                    for j, edge in enumerate(cell.edges):
                        if hasattr(edge, "alphas"):
                            alphas = (
                                F.softmax(edge.alphas, dim=-1).detach().cpu().numpy()
                            )
                            current_alphas.append((f"cell_{i}_edge_{j}", alphas))

        # Add encoder/decoder alphas
        if hasattr(model, "forecast_encoder") and hasattr(
            model.forecast_encoder, "alphas"
        ):
            alphas = (
                F.softmax(model.forecast_encoder.alphas, dim=-1).detach().cpu().numpy()
            )
            current_alphas.append(("encoder", alphas))

        if hasattr(model, "forecast_decoder") and hasattr(
            model.forecast_decoder, "alphas"
        ):
            alphas = (
                F.softmax(model.forecast_decoder.alphas, dim=-1).detach().cpu().numpy()
            )
            current_alphas.append(("decoder", alphas))

        # Add attention alphas for attention bridge selection
        if hasattr(model, "forecast_decoder") and hasattr(
            model.forecast_decoder, "attention_alphas"
        ):
            alphas = (
                F.softmax(model.forecast_decoder.attention_alphas, dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            current_alphas.append(("attention_bridge", alphas))

        return current_alphas

    def _calculate_entropy_regularization(self, model, temperature):
        """Calculate entropy regularization to encourage diversity"""
        total_entropy = 0.0
        num_edges = 0

        # Cell edges
        if hasattr(model, "cells"):
            for cell in model.cells:
                if hasattr(cell, "edges"):
                    for edge in cell.edges:
                        if hasattr(edge, "alphas"):
                            probs = F.softmax(edge.alphas / temperature, dim=-1)
                            entropy = -(probs * torch.log(probs + 1e-8)).sum()
                            total_entropy += entropy
                            num_edges += 1

        # Encoder/decoder alphas
        for module_name in ["forecast_encoder", "forecast_decoder"]:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if hasattr(module, "alphas"):
                    probs = F.softmax(module.alphas / temperature, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    total_entropy += entropy
                    num_edges += 1

        # Attention bridge alphas
        if hasattr(model, "forecast_decoder") and hasattr(
            model.forecast_decoder, "attention_alphas"
        ):
            probs = F.softmax(
                model.forecast_decoder.attention_alphas / temperature, dim=-1
            )
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            total_entropy += entropy
            num_edges += 1

        return -total_entropy / max(num_edges, 1)  # Negative to encourage high entropy

    def _calculate_identity_penalty(self, model):
        """Penalize Identity operation dominance"""
        total_penalty = 0.0
        num_edges = 0

        if hasattr(model, "cells"):
            for cell in model.cells:
                if hasattr(cell, "edges"):
                    for edge in cell.edges:
                        if (
                            hasattr(edge, "available_ops")
                            and "Identity" in edge.available_ops
                        ):
                            identity_idx = edge.available_ops.index("Identity")
                            probs = F.softmax(edge.alphas, dim=-1)
                            identity_prob = probs[identity_idx]

                            # Quadratic penalty when Identity > 0.5
                            if identity_prob > 0.5:
                                penalty = (identity_prob - 0.5) ** 2
                                total_penalty += penalty

                            num_edges += 1

        return total_penalty / max(num_edges, 1)

    def _calculate_smoothness_penalty(self, model, alpha_history):
        """Penalize rapid changes in architecture"""
        if len(alpha_history) < 2:
            return torch.tensor(0.0).to(
                model.device if hasattr(model, "device") else "cpu"
            )

        current_alphas = alpha_history[-1]
        previous_alphas = alpha_history[-2]

        total_diff = 0.0
        num_comparisons = 0

        # Compare current vs previous alpha values
        for (name1, alphas1), (name2, alphas2) in zip(current_alphas, previous_alphas):
            if name1 == name2:  # Same edge
                diff = np.sum((alphas1 - alphas2) ** 2)
                total_diff += diff
                num_comparisons += 1

        return torch.tensor(total_diff / max(num_comparisons, 1)).to(
            model.device if hasattr(model, "device") else "cpu"
        )

    def _calculate_operation_balance_penalty(self, model):
        """Encourage balanced exploration of all operations"""
        operation_counts = {}
        total_weight = 0.0

        if hasattr(model, "cells"):
            for cell in model.cells:
                if hasattr(cell, "edges"):
                    for edge in cell.edges:
                        if hasattr(edge, "available_ops") and hasattr(edge, "alphas"):
                            probs = F.softmax(edge.alphas, dim=-1)
                            for op_name, prob in zip(edge.available_ops, probs):
                                operation_counts[op_name] = (
                                    operation_counts.get(op_name, 0) + prob.item()
                                )
                                total_weight += prob.item()

        if not operation_counts:
            return torch.tensor(0.0)

        # Calculate variance in operation usage
        avg_weight = total_weight / len(operation_counts)
        variance = sum(
            (weight - avg_weight) ** 2 for weight in operation_counts.values()
        )
        variance /= len(operation_counts)

        return torch.tensor(variance).to(
            model.device if hasattr(model, "device") else "cpu"
        )

    def _derive_final_architecture(self, model):
        """Derive the final discrete architecture"""
        if hasattr(model, "derive_discrete_architecture"):
            return model.derive_discrete_architecture(threshold=0.3)

        # Fallback implementation
        architecture = {}

        if hasattr(model, "cells"):
            for i, cell in enumerate(model.cells):
                if hasattr(cell, "edges"):
                    cell_arch = {}
                    for j, edge in enumerate(cell.edges):
                        if hasattr(edge, "available_ops") and hasattr(edge, "alphas"):
                            weights = F.softmax(edge.alphas, dim=-1)
                            max_idx = weights.argmax().item()
                            max_weight = weights.max().item()

                            cell_arch[f"edge_{j}"] = {
                                "operation": edge.available_ops[max_idx],
                                "weight": max_weight,
                            }
                    architecture[f"cell_{i}"] = cell_arch

        return architecture

    def _compute_final_metrics(self, model: nn.Module, val_loader) -> Dict[str, float]:
        """Compute final metrics on validation set."""
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y, *_ in self._create_progress_bar(
                val_loader, "Computing metrics", leave=False
            ):
                batch_x, batch_y = batch_x.to(self.device).float(), batch_y.to(
                    self.device
                )
                all_preds.append(model(batch_x).cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        preds_flat = np.concatenate(all_preds).reshape(-1)
        targets_flat = np.concatenate(all_targets).reshape(-1)

        return self._compute_metrics(preds_flat, targets_flat)
    
    def derive_final_architecture(self, model: nn.Module) -> nn.Module:
        """
        Create optimized model with fixed operations based on search results.

        Args:
            model: Trained DARTS model

        Returns:
            Optimized model with fixed architecture
        """
        new_model = copy.deepcopy(model)

        print("🔧 Deriving final architecture...")

        # Replace mixed operations with fixed ones
        if hasattr(new_model, 'cells'):
            for cell_idx, cell in enumerate(new_model.cells):
                new_edges = nn.ModuleList()
                for edge_idx, edge in enumerate(cell.edges):
                    weights = F.softmax(edge.alphas, dim=-1)
                    top_op_idx = weights.argmax().item()
                    top_op = edge.ops[top_op_idx]

                    print(
                        f"   Cell {cell_idx}, Edge {edge_idx}: {type(top_op).__name__} "
                        f"(weight: {weights[top_op_idx]:.3f})"
                    )

                    fixed_edge = FixedOp(top_op)
                    new_edges.append(fixed_edge)
                cell.edges = new_edges

        device = next(new_model.parameters()).device

        # Fix encoder
        if hasattr(new_model, "forecast_encoder") and hasattr(new_model.forecast_encoder, 'alphas'):
            try:
                encoder_weights = F.softmax(new_model.forecast_encoder.alphas, dim=-1)
                top_idx = encoder_weights.argmax().item()
                
                # Get the encoder based on the architecture
                if hasattr(new_model.forecast_encoder, 'encoders'):
                    top_encoder = new_model.forecast_encoder.encoders[top_idx]
                elif hasattr(new_model.forecast_encoder, 'lstm') and top_idx == 0:
                    top_encoder = new_model.forecast_encoder.lstm
                elif hasattr(new_model.forecast_encoder, 'gru') and top_idx == 1:
                    top_encoder = new_model.forecast_encoder.gru
                elif hasattr(new_model.forecast_encoder, 'transformer') and top_idx == 2:
                    top_encoder = new_model.forecast_encoder.transformer
                else:
                    raise ValueError(f"Could not find encoder for index {top_idx}")

                encoder_names = getattr(new_model.forecast_encoder, 'encoder_names', ['lstm', 'gru', 'transformer'])
                encoder_type = encoder_names[top_idx] if top_idx < len(encoder_names) else 'unknown'

                print(
                    f"   → Fixing Forecast Encoder: {type(top_encoder).__name__} "
                    f"(weight: {encoder_weights[top_idx]:.3f})"
                )

                # Create fixed encoder using ArchitectureConverter
                new_model.forecast_encoder = ArchitectureConverter.create_fixed_encoder(
                    new_model.forecast_encoder
                ).to(device)

            except Exception as e:
                print(f"Warning: Could not fix encoder architecture: {e}")
                print("Falling back to weight fixing...")
                ArchitectureConverter.fix_mixed_weights(new_model.forecast_encoder)

        # Fix decoder
        if hasattr(new_model, "forecast_decoder") and hasattr(new_model.forecast_decoder, 'alphas'):
            try:
                decoder_weights = F.softmax(new_model.forecast_decoder.alphas, dim=-1)
                top_idx = decoder_weights.argmax().item()
                
                # Get the decoder based on the architecture
                if hasattr(new_model.forecast_decoder, 'decoders'):
                    top_decoder = new_model.forecast_decoder.decoders[top_idx]
                elif hasattr(new_model.forecast_decoder, 'lstm') and top_idx == 0:
                    top_decoder = new_model.forecast_decoder.lstm
                elif hasattr(new_model.forecast_decoder, 'gru') and top_idx == 1:
                    top_decoder = new_model.forecast_decoder.gru
                elif hasattr(new_model.forecast_decoder, 'transformer') and top_idx == 2:
                    top_decoder = new_model.forecast_decoder.transformer
                else:
                    raise ValueError(f"Could not find decoder for index {top_idx}")

                decoder_names = getattr(new_model.forecast_decoder, 'rnn_names', 
                                    getattr(new_model.forecast_decoder, 'decoder_names', ['lstm', 'gru', 'transformer']))
                top_decoder_type = decoder_names[top_idx] if top_idx < len(decoder_names) else 'unknown'

                # Infer latent_dim safely
                if hasattr(top_decoder, "latent_dim"):
                    latent_dim = top_decoder.latent_dim
                elif hasattr(top_decoder, "hidden_size"):  # for LSTM/GRU
                    latent_dim = top_decoder.hidden_size
                elif hasattr(new_model.forecast_decoder, "latent_dim"):
                    latent_dim = new_model.forecast_decoder.latent_dim
                else:
                    print("Warning: Could not extract latent_dim, using default 64")
                    latent_dim = 64

                print(
                    f"   → Fixing Forecast Decoder: {type(top_decoder).__name__} "
                    f"(weight: {decoder_weights[top_idx]:.3f})"
                )

                # Handle attention bridge selection
                attention_choice = "no_attention"
                max_att_idx = 0
                
                use_attention = getattr(new_model, 'use_attention_bridge', False)
                
                if use_attention and hasattr(new_model.forecast_decoder, "attention_alphas"):
                    try:
                        attention_weights = F.softmax(
                            new_model.forecast_decoder.attention_alphas, dim=0
                        )
                        max_idx = attention_weights.argmax().item()

                        if max_idx == len(attention_weights) - 1:
                            attention_choice = "no_attention"
                        else:
                            attention_choice = f"attention_layer_{max_idx}"
                            max_att_idx = max_idx

                        print("   → Using Attention Bridge:", attention_choice)
                    except Exception as e:
                        print(f"Warning: Could not determine attention choice: {e}")
                        attention_choice = "attention" if use_attention else "no_attention"

                # Get attention bridges safely
                attention_bridges = None
                if hasattr(new_model.forecast_decoder, "attention_bridges"):
                    attention_bridges = new_model.forecast_decoder.attention_bridges
                elif hasattr(new_model.forecast_decoder, "attention_bridge"):
                    # Handle single attention bridge case
                    attention_bridges = [new_model.forecast_decoder.attention_bridge]

                # Create fixed decoder using ArchitectureConverter
                use_attention_final = use_attention and attention_choice != "no_attention"
                
                new_model.forecast_decoder = ArchitectureConverter.create_fixed_decoder(
                    new_model.forecast_decoder,
                    use_attention_bridge=use_attention_final
                ).to(device)

                # Handle attention bridges assignment safely
                if use_attention_final and attention_bridges is not None:
                    try:
                        if isinstance(attention_bridges, (list, nn.ModuleList)) and len(attention_bridges) > max_att_idx:
                            if hasattr(new_model.forecast_decoder, 'attention_bridges'):
                                new_model.forecast_decoder.attention_bridges = nn.ModuleList([attention_bridges[max_att_idx]])
                            elif hasattr(new_model.forecast_decoder, 'attention_bridge'):
                                # Transfer weights to the single attention bridge
                                new_model.forecast_decoder.attention_bridge.load_state_dict(
                                    attention_bridges[max_att_idx].state_dict()
                                )
                        else:
                            print("Warning: Could not assign attention bridges - index out of range or invalid format")
                    except Exception as e:
                        print(f"Warning: Could not assign attention bridges: {e}")
                else:
                    # Ensure no attention bridges are set
                    if hasattr(new_model.forecast_decoder, 'attention_bridges'):
                        new_model.forecast_decoder.attention_bridges = None

            except Exception as e:
                print(f"Warning: Could not fix decoder architecture: {e}")
                print("Falling back to weight fixing...")
                ArchitectureConverter.fix_mixed_weights(new_model.forecast_decoder)

        print("✓ Architecture derivation completed")
        return new_model

    def train_final_model(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 50,
        loss_type: str = "huber",
        use_onecycle: bool = True,
        swa_start_ratio: float = 0.33,
        grad_clip_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Train final model with fixed architecture.

        Args:
            model: Model with fixed architecture
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            loss_type: Loss function type
            use_onecycle: Whether to use OneCycle learning rate scheduler
            swa_start_ratio: When to start SWA (as fraction of total epochs)
            grad_clip_norm: Gradient clipping norm

        Returns:
            Dictionary containing training results
        """
        model = model.to(self.device)

        # Setup training components
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler setup
        if use_onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1000,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )

        # SWA and mixed precision setup
        swa_model = torch.optim.swa_utils.AveragedModel(model).to(self.device)
        scaler = GradScaler()
        loss_fn = self._get_loss_function(loss_type)

        # Training state
        best_val_loss, patience_counter, best_state = float("inf"), 0, None
        train_losses, val_losses = [], []
        swa_start = int(epochs * swa_start_ratio)

        print(f"🚀 Training final model for {epochs} epochs")
        print(f"   Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        print(
            f"   Loss function: {loss_type}, Scheduler: {'OneCycle' if use_onecycle else 'CosineAnnealing'}"
        )
        print(f"   SWA starts at epoch {swa_start}, Patience: {patience}")
        print("-" * 70)

        start_time = time.time()
        epoch_pbar = self._create_progress_bar(
            range(epochs), "Final Training", unit="epoch"
        )

        for epoch in epoch_pbar:
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_train_batches = len(train_loader)

            train_pbar = self._create_progress_bar(
                train_loader,
                f"Epoch {epoch+1:3d} Train",
                leave=False,
                total=num_train_batches,
            )

            for batch_idx, (batch_x, batch_y, *_) in enumerate(train_pbar):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()

                with autocast("cuda" if self.device.startswith("cuda") else "cpu"):
                    preds = model(batch_x)
                    loss = loss_fn(preds, batch_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()

                if use_onecycle:
                    scheduler.step()

                epoch_train_loss += loss.item()

                train_pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{epoch_train_loss/(batch_idx+1):.4f}",
                        "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
                    }
                )

            train_pbar.close()

            if not use_onecycle:
                scheduler.step()

            avg_train_loss = epoch_train_loss / num_train_batches
            train_losses.append(avg_train_loss)

            # Validation phase
            avg_val_loss = self._evaluate_model(model, val_loader, loss_type)
            val_losses.append(avg_val_loss)

            # SWA update
            swa_updated = False
            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_updated = True

            # Update main progress bar
            postfix_dict = {
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "best_val": f"{best_val_loss:.4f}",
                "patience": f"{patience_counter}/{patience}",
            }

            if swa_updated:
                postfix_dict["swa"] = "✓"

            epoch_pbar.set_postfix(postfix_dict)

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    epoch_pbar.set_description(f"Early stopping at epoch {epoch+1}")
                    break

        epoch_pbar.close()

        # Finalize SWA model
        swa_used = self._finalize_swa(
            model,
            swa_model,
            val_loader,
            train_loader,
            epoch,
            swa_start,
            loss_type,
            best_val_loss,
            best_state,
        )

        if swa_used:
            # strip .module
            if any(k.startswith("module.") for k in best_state.keys()):
                best_state = {k.replace("module.", "", 1): v for k, v in best_state.items()}
                
        # Evaluate on test set
        model.load_state_dict(best_state, strict=False)
        test_results = self._evaluate_test_set(model, test_loader, loss_type)
        training_time = time.time() - start_time

        # Final results
        results = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_loss": test_results["test_loss"],
            "training_time": training_time,
            "final_metrics": test_results["metrics"],
            "training_info": {
                "epochs_completed": epoch + 1,
                "swa_used": swa_used,
                "final_lr": optimizer.param_groups[0]["lr"],
                "best_val_loss": best_val_loss,
            },
        }

        self._print_final_results(results)
        self.training_history.append(results)

        return results

    def _finalize_swa(
        self,
        model,
        swa_model,
        val_loader,
        train_loader,
        epoch,
        swa_start,
        loss_type,
        best_val_loss,
        best_state,
    ):
        """Finalize SWA model and determine whether to use it."""
        if epoch < swa_start:
            return False

        print("\\n🔄 Finalizing SWA model...")

        try:
            bn_update_pbar = self._create_progress_bar(
                train_loader, "Updating BN", leave=False
            )
            torch.optim.swa_utils.update_bn(
                bn_update_pbar, swa_model, device=self.device
            )
            bn_update_pbar.close()
        except Exception as e:
            print(f"Warning: Standard BN update failed ({e}), using fallback...")
            swa_model.train()
            with torch.no_grad():
                for batch_x, *_ in self._create_progress_bar(
                    train_loader, "Fallback BN", leave=False
                ):
                    swa_model(batch_x.to(self.device))

        # Evaluate SWA model
        swa_val_loss = self._evaluate_model(swa_model, val_loader, loss_type)
        print(f"SWA validation loss: {swa_val_loss:.6f} vs Best: {best_val_loss:.6f}")

        if swa_val_loss < best_val_loss:
            print("✓ Using SWA model (better performance)")
            best_state.update(
                {k: v.cpu().clone() for k, v in swa_model.state_dict().items()}
            )
            return True
        else:
            print("✓ Keeping original best model")
            return False

    def _evaluate_test_set(self, model, test_loader, loss_type):
        """Evaluate model on test set and compute comprehensive metrics."""
        print("\\n📊 Evaluating on test set...")
        model.eval()
        loss_fn = self._get_loss_function(loss_type)

        test_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            test_pbar = self._create_progress_bar(test_loader, "Test Evaluation")

            for batch_x, batch_y, *_ in test_pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                with autocast("cuda" if self.device.startswith("cuda") else "cpu"):
                    preds = model(batch_x)
                    batch_test_loss = loss_fn(preds, batch_y).item()

                test_loss += batch_test_loss
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

                test_pbar.set_postfix({"test_loss": f"{batch_test_loss:.4f}"})

            test_pbar.close()

        test_loss /= len(test_loader)
        preds_flat = np.concatenate(all_preds).reshape(-1)
        targets_flat = np.concatenate(all_targets).reshape(-1)

        return {
            "test_loss": test_loss,
            "metrics": self._compute_metrics(preds_flat, targets_flat),
        }

    def _print_final_results(self, results: Dict[str, Any]):
        """Prints the final model's results in a professional, aligned format."""
        metrics = results["final_metrics"]
        info = results["training_info"]

        logger.info("\n" + "=" * 70)
        logger.info("🏁 FINAL MODEL TRAINING COMPLETED")
        logger.info("=" * 70)
        logger.info(
            f"{'Training duration:':<30} {results['training_time']:.1f} seconds  ({results['training_time']/60:.1f} minutes)"
        )
        logger.info(f"{'Total epochs:':<30} {info['epochs_completed']}")
        logger.info(
            f"{'Checkpoint used:':<30} {'SWA' if info.get('swa_used', False) else 'Best model'}"
        )
        logger.info(f"{'Final learning rate:':<30} {info['final_lr']:.2e}")
        logger.info("-" * 70)
        logger.info("📊 PERFORMANCE METRICS:")
        logger.info(f"{'Test Loss:':<30} {results['test_loss']:.6f}")
        logger.info(f"{'Mean Squared Error (MSE):':<30} {metrics['mse']:.6f}")
        logger.info(f"{'Root Mean Squared Error (RMSE):':<30} {metrics['rmse']:.6f}")
        logger.info(f"{'Mean Absolute Error (MAE):':<30} {metrics['mae']:.6f}")
        logger.info(f"{'Mean Absolute % Error (MAPE):':<30} {metrics['mape']:.2f}%")
        logger.info(f"{'R² Score:':<30} {metrics['r2_score']:.4f}")
        logger.info("=" * 70 + "\n")

    def multi_fidelity_search(
        self,
        train_loader,
        val_loader,
        test_loader,
        num_candidates: int = 10,
        search_epochs: int = 10,
        final_epochs: int = 100,
        max_samples: int = 32,
        top_k: int = 5,
        max_workers: int = None,  # New parameter for controlling parallelism
    ) -> Dict[str, Any]:
        """
        Multi-fidelity architecture search using zero-cost metrics with parallel Phase 1.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            num_candidates: Number of candidates to generate
            search_epochs: Epochs for DARTS search phase
            final_epochs: Epochs for final training
            max_samples: Max samples for zero-cost evaluation
            top_k: Number of top candidates to train
            max_workers: Maximum number of parallel workers (None = auto-detect)

        Returns:
            Dictionary containing search results
        """
        print("🔍 Starting multi-fidelity DARTS search...")
        print(f"   Candidates: {num_candidates}, Search epochs: {search_epochs}")
        print(f"   Final epochs: {final_epochs}, Top-k: {top_k}")
        print(f"   Max workers: {max_workers or 'auto-detect'}")
        print("-" * 60)

        # Phase 1: Generate and evaluate candidates with zero-cost metrics (PARALLELIZED)
        print("\n📋 Phase 1: Generating and evaluating candidates in parallel...")

        def generate_and_evaluate_candidate(candidate_id: int) -> Dict[str, Any]:
            """Generate and evaluate a single candidate."""
            try:
                # Random architecture generation
                num_ops = random.randint(2, len(self.all_ops))
                selected_ops = ["Identity"] + random.sample(
                    [op for op in self.all_ops if op != "Identity"], num_ops - 1
                )

                config = {
                    "selected_ops": selected_ops,
                    "hidden_dim": random.choice(self.hidden_dims),
                    "num_cells": random.randint(1, 2),
                    "num_nodes": random.randint(2, 4),
                }

                # Create model (each worker gets its own model instance)
                model = TimeSeriesDARTS(
                    input_dim=self.input_dim,
                    hidden_dim=config["hidden_dim"],
                    latent_dim=config["hidden_dim"],
                    forecast_horizon=self.forecast_horizon,
                    seq_length=self.seq_length,
                    num_cells=config["num_cells"],
                    num_nodes=config["num_nodes"],
                    selected_ops=config["selected_ops"],
                ).to(self.device)

                # Evaluate with zero-cost metrics
                metrics = self.evaluate_zero_cost_metrics(model, val_loader, max_samples)
                score = metrics["aggregate_score"]

                return {
                    "candidate_id": candidate_id,
                    "model": model,
                    "metrics": metrics,
                    "score": score,
                    "success": True,
                    **config,
                }

            except Exception as e:
                return {
                    "candidate_id": candidate_id,
                    "model": None,
                    "metrics": {"aggregate_score": 0.0},
                    "score": 0.0,
                    "success": False,
                    "error": str(e),
                }

        # ─────────────────────────────────────────────────────────────────────────────

        candidates = []
        completed_count = 0
        lock = threading.Lock()

        def update_progress(future):
            nonlocal completed_count
            result = future.result()
            with lock:
                completed_count += 1
                cid = result.get("candidate_id", -1)
                if result["success"]:
                    print(
                        f"✓ Candidate {completed_count}/{num_candidates} "
                        f"(ID {cid}) | Score: {result['score']:.4f} | "
                        f"Ops: {len(result.get('selected_ops', []))} | "
                        f"Hidden: {result.get('hidden_dim', 'N/A')}"
                    )
                else:
                    print(f"✗ Candidate {completed_count}/{num_candidates} (ID {cid}) - Failed")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_and_evaluate_candidate, i): i
                for i in range(num_candidates)
            }

            for future in futures:
                future.add_done_callback(update_progress)

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result["success"]:
                        candidates.append(result)
                except Exception as e:
                    print(f"⚠️ Unexpected error in future: {e}")

        print(f"\n📊 Phase 1 completed: {len(candidates)}/{num_candidates} candidates successful")

        # Phase 2: Select top candidates
        print(f"\n🏆 Phase 2: Selecting top {top_k} candidates...")
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_k = min(top_k, len(candidates))
        top_candidates = candidates[:top_k]

        print("Top candidates:")
        for i, c in enumerate(top_candidates):
            print(
                f"  {i+1}: Score={c['score']:.4f}, Ops={len(c['selected_ops'])}, "
                f"Hidden={c['hidden_dim']}, Arch={c['num_cells']}x{c['num_nodes']}"
            )

        # Phase 3: Short DARTS training for top candidates
        print(f"\n🔧 Phase 3: Training top {top_k} candidates...")
        trained_candidates = []

        for i, candidate in enumerate(top_candidates):
            print(f"\nTraining candidate {i+1}/{top_k}...")

            # Quick DARTS training
            search_results = self.train_darts_model(
                model=candidate["model"],
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=search_epochs,
                use_swa=False,
            )

            # Derive and evaluate architecture
            derived_model = self.derive_final_architecture(search_results["model"])
            val_loss = self._evaluate_model(derived_model, val_loader)

            trained_candidates.append(
                {
                    "model": derived_model,
                    "val_loss": val_loss,
                    "candidate": candidate,
                    "search_results": search_results,
                }
            )

            print(f"   Validation loss: {val_loss:.6f}")

        # Phase 4: Select best candidate
        print(f"\n🎯 Phase 4: Selecting best candidate...")
        best_candidate = min(trained_candidates, key=lambda x: x["val_loss"])

        print(f"Best candidate:")
        print(f"   Validation loss: {best_candidate['val_loss']:.6f}")
        print(f"   Operations: {best_candidate['candidate']['selected_ops']}")
        print(
            f"   Architecture: {best_candidate['candidate']['num_cells']}x{best_candidate['candidate']['num_nodes']}"
        )
        print(f"   Hidden dim: {best_candidate['candidate']['hidden_dim']}")

        # Phase 5: Train final model
        print(f"\n🚀 Phase 5: Training final model...")
        final_model = copy.deepcopy(best_candidate["model"])

        final_results = self.train_final_model(
            model=final_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=final_epochs,
            learning_rate=5e-4,
            weight_decay=1e-5,
        )

        # Plot training curve
        self._plot_training_curve(
            final_results["train_losses"],
            final_results["val_losses"],
            title="Final Model Training Progress",
            save_path="final_model_training.pdf",
        )

        # Store search results
        search_summary = {
            "final_model": final_results["model"],
            "candidates": candidates,
            "top_candidates": top_candidates,
            "trained_candidates": trained_candidates,
            "best_candidate": best_candidate,
            "final_results": final_results,
            "search_config": {
                "num_candidates": num_candidates,
                "search_epochs": search_epochs,
                "final_epochs": final_epochs,
                "top_k": top_k,
            },
        }

        self.search_history.append(search_summary)

        self.final_model = final_results["model"]

        print("\n✅ Multi-fidelity search completed!")
        return search_summary

    def get_search_summary(self) -> str:
        """Get a summary of all searches performed."""
        if not self.search_history:
            return "No searches performed yet."

        summary = []
        summary.append("🔍 DARTS SEARCH SUMMARY")
        summary.append("=" * 50)

        for i, search in enumerate(self.search_history):
            final_metrics = search["final_results"]["final_metrics"]
            config = search["search_config"]

            summary.append(f"\nSearch {i+1}:")
            summary.append(f"  Candidates evaluated: {config['num_candidates']}")
            summary.append(f"  Final test RMSE: {final_metrics['rmse']:.6f}")
            summary.append(f"  Final R² score: {final_metrics['r2_score']:.4f}")
            summary.append(
                f"  Training time: {search['final_results']['training_time']:.1f}s"
            )

        summary.append("\n" + "=" * 50)
        return "\n".join(summary)

    def get_training_summary(self) -> str:
        """Get a summary of all training sessions."""
        if not self.training_history:
            return "No training sessions completed yet."

        summary = []
        summary.append("🚀 TRAINING SUMMARY")
        summary.append("=" * 40)

        for i, training in enumerate(self.training_history):
            if "final_metrics" in training:
                metrics = training["final_metrics"]
                summary.append(f"\nSession {i+1}:")
                summary.append(
                    f"  Best val loss: {training.get('best_val_loss', 'N/A')}"
                )
                summary.append(f"  RMSE: {metrics.get('rmse', 'N/A')}")
                summary.append(f"  R² score: {metrics.get('r2_score', 'N/A')}")
                summary.append(
                    f"  Training time: {training.get('training_time', 'N/A')}s"
                )

        summary.append("\n" + "=" * 40)
        return "\n".join(summary)

    def save_best_model(self, filepath: str = "best_darts_model.pth"):
        """Save the best model from search history."""
        if not self.search_history:
            print("❌ No search history available to save.")
            return

        best_search = min(
            self.search_history,
            key=lambda x: x["final_results"]["final_metrics"]["rmse"],
        )

        torch.save(
            {
                "model_state_dict": best_search["final_model"].state_dict(),
                "final_metrics": best_search["final_results"]["final_metrics"],
                "training_info": best_search["final_results"]["training_info"],
                "search_config": best_search["search_config"],
            },
            filepath,
        )

        print(f"💾 Best model saved to {filepath}")
        print(f"   RMSE: {best_search['final_results']['final_metrics']['rmse']:.6f}")
        print(
            f"   R² Score: {best_search['final_results']['final_metrics']['r2_score']:.4f}"
        )

    def load_model(self, filepath: str, model_class):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        # You'll need to reconstruct the model architecture first
        # This is a placeholder - you'd need the actual architecture info
        print(f"📂 Loading model from {filepath}")
        print(f"   Saved RMSE: {checkpoint['final_metrics']['rmse']:.6f}")
        print(f"   Saved R² Score: {checkpoint['final_metrics']['r2_score']:.4f}")

        return checkpoint

    def plot_alpha_evolution(
        self, alpha_values: List, save_path: str = "alpha_evolution.png"
    ):
        """Plot the evolution of architecture parameters during search."""
        if not alpha_values:
            print("❌ No alpha values to plot.")
            return

        # Extract alpha evolution for first few edges
        num_epochs = len(alpha_values)
        num_edges_to_plot = min(4, len(alpha_values[0]))  # Plot first 4 edges

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for edge_idx in range(num_edges_to_plot):
            if edge_idx < len(alpha_values[0]):
                ax = axes[edge_idx]

                # Get alpha values for this edge across epochs
                edge_alphas = []
                for epoch_alphas in alpha_values:
                    if edge_idx < len(epoch_alphas):
                        cell_idx, edge_in_cell, alphas = epoch_alphas[edge_idx]
                        edge_alphas.append(alphas)

                if edge_alphas:
                    edge_alphas = np.array(edge_alphas)
                    epochs = range(len(edge_alphas))

                    # Plot each operation's alpha
                    for op_idx in range(edge_alphas.shape[1]):
                        ax.plot(
                            epochs,
                            edge_alphas[:, op_idx],
                            label=f"Op {op_idx}",
                            linewidth=2,
                            alpha=0.8,
                        )

                    ax.set_title(f"Edge {edge_idx} Alpha Evolution", fontweight="bold")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Alpha Weight")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Alpha evolution plot saved to {save_path}")

    def plot_architecture(self, *, candidate: dict, save_path: str = "arch.png"):

        arquitetura = self.parse_model_architecture(candidate["model"])
        self.draw_darts_architecture(arquitetura, save_path=save_path)

    def draw_darts_architecture(
        self, model_info=None, save_path="darts_architecture.png"
    ):
        """
        Draw a beautiful DARTS architecture diagram based on actual model structure.

        Args:
            model_info: Dictionary containing model architecture details
        """
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 12)
        ax.axis("off")


        # Beautiful color scheme
        colors = {
            "input_output": "#B8E6B8",
            "embedding": "#A8D0F0",
            "cell1": "#F4E4E4",
            "cell1_ops": "#F5C2C7",
            "cell1_processing": "#F5C2C7",
            "cell2": "#FFF3CD",
            "cell2_ops": "#FFE69C",
            "forecast": "#E9ECEF",
            "forecast_ops": "#CED4DA",
        }

        # Helper functions for beautiful boxes
        def add_beautiful_box(
            x,
            y,
            w,
            h,
            label,
            color,
            fontsize=10,
            fontweight="normal",
            edge_color="#333333",
            linewidth=1.5,
            corner_radius=0.05,
        ):
            """Add a beautiful rounded rectangle with shadow effect"""
            # Add subtle shadow
            shadow = FancyBboxPatch(
                (x + 0.02, y - 0.02),
                w,
                h,
                boxstyle=f"round,pad=0.02,rounding_size={corner_radius}",
                facecolor="#00000015",
                edgecolor="none",
                zorder=1,
            )
            ax.add_patch(shadow)

            # Main box
            box = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0.02,rounding_size={corner_radius}",
                edgecolor=edge_color,
                facecolor=color,
                linewidth=linewidth,
                zorder=2,
            )
            ax.add_patch(box)

            # Text with better typography
            ax.text(
                x + w / 2,
                y + h / 2,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                weight=fontweight,
                color="#2C3E50",
                zorder=3,
            )

        def draw_curved_arrow(
            start_x,
            start_y,
            end_x,
            end_y,
            style="->",
            linewidth=2,
            color="#2C3E50",
            curve_strength=0.3,
        ):
            """Draw a beautiful curved arrow"""
            if abs(start_x - end_x) > abs(start_y - end_y):
                # Horizontal curve
                mid_x = (start_x + end_x) / 2
                control_y = start_y + curve_strength * (end_y - start_y)
                control_x = mid_x
            else:
                # Vertical curve
                mid_y = (start_y + end_y) / 2
                control_x = start_x + curve_strength * (end_x - start_x)
                control_y = mid_y

            # Create curved path
            from matplotlib.path import Path

            verts = [(start_x, start_y), (control_x, control_y), (end_x, end_y)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)

            # Draw path
            patch = patches.PathPatch(
                path, facecolor="none", edgecolor=color, linewidth=linewidth, zorder=2
            )
            ax.add_patch(patch)

            # Add arrowhead
            if style == "->":
                dx = end_x - start_x
                dy = end_y - start_y
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    arrow_size = 0.15
                    ax.arrow(
                        end_x - arrow_size * dx_norm,
                        end_y - arrow_size * dy_norm,
                        arrow_size * dx_norm,
                        arrow_size * dy_norm,
                        head_width=0.08,
                        head_length=0.08,
                        fc=color,
                        ec=color,
                        zorder=3,
                    )

        def draw_dashed_connection(
            start_x, start_y, end_x, end_y, color="#7F8C8D", alpha=0.6
        ):
            """Draw beautiful dashed connections"""
            ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                "--",
                color=color,
                linewidth=1.5,
                alpha=alpha,
                zorder=1,
            )

        # Extract values from model_info
        input_feat = model_info["input_features"]
        embed_dim = model_info["embedding_dim"]
        output_feat = model_info["output_features"]
        cells = model_info["cells"]

        # 1. Input Features (top left)
        add_beautiful_box(
            0.5,
            9.5,
            2.5,
            1,
            f"Input Features ({input_feat})",
            colors["input_output"],
            fontsize=11,
            fontweight="bold",
        )

        # 2. Input Embedding (below input)
        add_beautiful_box(
            0.5,
            8,
            2.5,
            1,
            f"Input Embedding\nLinear({input_feat} → {embed_dim})",
            colors["embedding"],
            fontsize=10,
            fontweight="bold",
        )

        # Curved arrow from Input to Embedding
        draw_curved_arrow(1.75, 9.5, 1.75, 9, linewidth=2.5, color="#2980B9")

        # 3. DARTS Cells (better positioned)
        # Dynamically compute vertical positions for all DARTS cells
        num_cells = len(cells)
        cell_spacing = 4.0  # Vertical spacing between cells
        total_height = (num_cells - 1) * cell_spacing
        base_y = (
            6.5 - total_height / 2
        )  # Center around y=6.5 (midpoint of forecast box)

        cell_positions = [(4.5, base_y + i * cell_spacing) for i in range(num_cells)]

        cell_colors = [colors["cell1"], colors["cell2"]]
        cell_op_colors = [colors["cell1_ops"], colors["cell2_ops"]]

        for cell_idx, cell in enumerate(cells):
            cell_x, cell_y = cell_positions[cell_idx]

            # Main cell container with beautiful styling
            add_beautiful_box(
                cell_x,
                cell_y,
                5,
                3.2,
                "",
                cell_colors[cell_idx],
                corner_radius=0.08,
                linewidth=2,
            )

            # Cell title
            add_beautiful_box(
                cell_x + 0.1,
                cell_y + 2.7,
                4.8,
                0.4,
                f"DARTS Cell {cell_idx + 1}",
                cell_colors[cell_idx],
                fontsize=13,
                fontweight="bold",
            )

            # Operations (left side with better spacing)
            ops = cell["ops"][:5]  # Show up to 5 operations
            op_width = 2
            op_height = 0.35
            for i, op in enumerate(ops):
                op_y = cell_y + 2.2 - i * 0.4
                add_beautiful_box(
                    cell_x + 0.2,
                    op_y,
                    op_width,
                    op_height,
                    op,
                    cell_op_colors[cell_idx],
                    fontsize=9,
                    corner_radius=0.03,
                    linewidth=1,
                )

                # Small connection line to processing box
                line_start_x = cell_x + 0.2 + op_width
                line_end_x = cell_x + 2.6
                ax.plot(
                    [line_start_x, line_end_x],
                    [op_y + op_height / 2, op_y + op_height / 2],
                    "-",
                    color=cell_op_colors[cell_idx],
                    linewidth=2,
                    alpha=0.7,
                )

            # Cell processing box (right side)
            add_beautiful_box(
                cell_x + 2.6,
                cell_y + 0.6,
                2.2,
                1.8,
                "Cell Processing\nProj + Norm + Gate",
                cell_op_colors[cell_idx],
                fontsize=10,
                fontweight="bold",
                corner_radius=0.05,
            )

        # 4. Beautiful dashed connections from embedding to cells
        embed_center_x = 1.75
        embed_center_y = 8.5

        for cell_idx, cell in enumerate(cells):
            cell_x, cell_y = cell_positions[cell_idx]
            ops = cell["ops"][:5]

            for i, op in enumerate(ops):
                op_y = cell_y + 2.2 - i * 0.4 + 0.175  # Center of op box
                draw_dashed_connection(3, embed_center_y, cell_x + 0.2, op_y)

        # 5. Forecasting Module (well-positioned)
        forecast_x, forecast_y = 11, 6
        add_beautiful_box(
            forecast_x,
            forecast_y,
            4.5,
            3.8,
            "",
            colors["forecast"],
            corner_radius=0.08,
            linewidth=2,
        )

        # Forecasting title
        add_beautiful_box(
            forecast_x + 0.1,
            forecast_y + 3.3,
            4.3,
            0.4,
            "Forecasting Module",
            colors["forecast"],
            fontsize=13,
            fontweight="bold",
        )

        # Forecasting operations with actual model info
        encoder_type = model_info["forecast_encoder"]
        decoder_type = model_info["forecast_decoder"]
        mlp_dims = model_info["mlp_dims"]

        forecast_ops = [
            f"{encoder_type} Encoder ({embed_dim} → {embed_dim})",
            f"{decoder_type} Decoder ({input_feat} → {embed_dim})",
            f"MLP ({' → '.join(map(str, mlp_dims))})",
            f"Output Layer ({embed_dim} → {output_feat})",
        ]

        for i, op in enumerate(forecast_ops):
            add_beautiful_box(
                forecast_x + 0.2,
                forecast_y + 2.7 - i * 0.6,
                4.1,
                0.5,
                op,
                colors["forecast_ops"],
                fontsize=10,
                corner_radius=0.03,
                linewidth=1,
            )

        # 6. Output (far right)
        add_beautiful_box(
            16.5,
            7.5,
            2.5,
            1,
            f"Output ({output_feat})",
            colors["input_output"],
            fontsize=11,
            fontweight="bold",
        )

        # 7. Beautiful curved arrows from cells to forecasting
        for cell_idx, cell in enumerate(cells):
            cell_x, cell_y = cell_positions[cell_idx]
            start_x = cell_x + 5
            start_y = cell_y + 1.6
            draw_curved_arrow(
                start_x,
                start_y,
                forecast_x,
                forecast_y + 1.9,
                linewidth=3,
                color="#27AE60",
                curve_strength=0.2,
            )

        # 8. Arrow from forecasting to output
        draw_curved_arrow(
            forecast_x + 4.5,
            forecast_y + 1.9,
            16.5,
            8,
            linewidth=3,
            color="#8E44AD",
            curve_strength=0.1,
        )

        # 9. Beautiful Legend (repositioned)
        legend_x, legend_y = 11.5, 0.5
        add_beautiful_box(
            legend_x,
            legend_y,
            3.5,
            3.2,
            "",
            "white",
            edge_color="#34495E",
            linewidth=2,
            corner_radius=0.06,
        )

        ax.text(
            legend_x + 1.75,
            legend_y + 2.8,
            "Legend",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="#2C3E50",
        )

        # Base legend items
        legend_items = [
            ("Input/Output", colors["input_output"]),
            ("Input Embedding", colors["embedding"]),
        ]

        # Add DARTS cell legend entries dynamically
        cell_colors_all = [
            colors["cell1"],
            colors["cell2"],
        ]  # Extend if more cells expected
        for i in range(len(cells)):
            color = cell_colors_all[i % len(cell_colors_all)]  # Cycle colors if needed
            legend_items.append((f"DARTS Cell {i + 1}", color))

        # Add Forecasting module
        legend_items.append(("Forecasting Module", colors["forecast"]))

        for i, (label, color) in enumerate(legend_items):
            legend_item_y = legend_y + 2.3 - i * 0.35
            add_beautiful_box(
                legend_x + 0.15, legend_item_y, 0.3, 0.25, "", color, corner_radius=0.02
            )
            ax.text(
                legend_x + 0.6,
                legend_item_y + 0.125,
                label,
                ha="left",
                va="center",
                fontsize=10,
                color="#2C3E50",
            )

        # Set beautiful background
        fig.patch.set_facecolor("#FAFAFA")

        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="#FAFAFA",
            edgecolor="none",
            pad_inches=0.2,
        )
        plt.show()
        print(f"Saved beautiful architecture diagram to {save_path}")

    def parse_model_architecture(self, model_dict):
        """
        Parse the TimeSeriesDARTS model dictionary to extract architecture details.

        Args:
            model_dict: Dictionary containing the PyTorch model (e.g., checkpoint['model'])

        Returns:
            Dictionary with architecture information for visualization
        """
        model_info = {}

        # Get the actual model from the dictionary
        if isinstance(model_dict, dict) and "model" in model_dict:
            model = model_dict["model"]
        else:
            model = model_dict

        # Extract input features and embedding dimension from input_embedding
        try:
            input_embedding = model.input_embedding
            # Find the first Linear layer
            for layer in input_embedding:
                if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                    model_info["input_features"] = layer.in_features
                    model_info["embedding_dim"] = layer.out_features
                    break
        except:
            model_info["input_features"] = 3
            model_info["embedding_dim"] = 32

        # Extract operations from DARTS cells
        cells = []
        try:
            for cell_idx, cell in enumerate(model.cells):
                ops = []

                # Traverse the cell structure to find operations
                if hasattr(cell, "edges"):
                    for edge in cell.edges:
                        if hasattr(edge, "ops"):
                            for op_idx, op in enumerate(edge.ops):
                                op_name = type(op).__name__
                                if op_name not in ops:
                                    ops.append(op_name)

                if ops:
                    cells.append({"ops": ops})
        except:
            # Fallback to default if parsing fails
            cells = [
                {
                    "ops": [
                        "IdentityOp",
                        "GRNOp",
                        "TimeConvOp",
                        "WaveletOp",
                    ]
                }
            ]

        model_info["cells"] = cells

        # Extract forecast encoder and decoder info
        try:
            forecast_encoder = model.forecast_encoder
            model_info["forecast_encoder"] = type(forecast_encoder).__name__
            if hasattr(forecast_encoder, "input_size"):
                encoder_input_dim = forecast_encoder.input_size
            if hasattr(forecast_encoder, "hidden_size"):
                encoder_hidden_dim = forecast_encoder.hidden_size
        except:
            model_info["forecast_encoder"] = "GRU"
            encoder_input_dim = 32
            encoder_hidden_dim = 32

        try:
            forecast_decoder = model.forecast_decoder
            model_info["forecast_decoder"] = type(forecast_decoder).__name__
            if hasattr(forecast_decoder, "input_size"):
                decoder_input_dim = forecast_decoder.input_size
            if hasattr(forecast_decoder, "hidden_size"):
                decoder_hidden_dim = forecast_decoder.hidden_size
        except:
            model_info["forecast_decoder"] = "GRU"
            decoder_input_dim = 3
            decoder_hidden_dim = 32

        # Extract MLP dimensions
        try:
            mlp = model.mlp
            mlp_dims = []
            for layer in mlp:
                if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                    if not mlp_dims:  # First layer
                        mlp_dims.append(layer.in_features)
                    mlp_dims.append(layer.out_features)
            model_info["mlp_dims"] = mlp_dims if mlp_dims else [32, 64, 32]
        except:
            model_info["mlp_dims"] = [32, 64, 32]

        # Extract output features
        try:
            output_layer = model.output_layer
            model_info["output_features"] = output_layer.out_features
        except:
            model_info["output_features"] = 3

        return model_info

    def _batched_forecast(self, X_val: torch.Tensor, batch_size: int = 256):
        """
        Generate batched forecasts aligned for time series evaluation or plotting.

        Returns:
            forecast: Tensor of shape [T + target_len - 1, output_size]
        """
        model = self.final_model
        model.eval()
        device = next(model.parameters()).device
        X_val = X_val.to(device)
        N = X_val.shape[0]

        # Run a dummy forward pass to determine output shape
        with torch.no_grad():
            dummy_out = model(X_val[0:1])
            if isinstance(dummy_out, tuple):
                dummy_out = dummy_out[0]
            if dummy_out.dim() == 3:
                output_len, output_size = dummy_out.shape[1], dummy_out.shape[2]
            elif dummy_out.dim() == 2:
                output_len, output_size = 1, dummy_out.shape[1]
            else:
                output_len, output_size = 1, dummy_out.shape[0]

        forecast = torch.zeros(N + output_len - 1, output_size, device=device)
        count = torch.zeros_like(forecast)

        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = X_val[i : i + batch_size]
                with autocast("cuda", dtype=torch.float16):
                    outputs = model(batch)
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

    def plot_prediction(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        full_series: Optional[torch.Tensor] = None,
        offset: int = 0,
        figsize: Tuple[int, int] = (12, 4),
        show: bool = False,
        device: Optional[torch.device] = None,
        names: Optional[List[str]] = None,
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

        model = self.final_model
        model.eval()  # Set model to evaluation mode
        X_val = X_val.to(device)
        y_val = y_val.to(device)
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
                    label=f"Original - {names[i] if names else f'Feature {i}'}",
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
                    label=f"Forecast - {names[i] if names else f'Feature {i}'}",
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
                axes[i].set_title(f"{names[i] if names else f'Feature {i}'} Forecast")
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
                    ax.plot(forecast[:, i], label=f"Forecast {names[i] if names else f'Feature {i}'}")
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
