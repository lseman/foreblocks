"""
Enhanced Trainer with NAS support for HeadComposer

Key additions:
1. train_nas flag in TrainingConfig
2. Separate optimizers for architecture (α) and weights
3. Two-step optimization loop
4. Alpha reporting and discretization utilities
"""

import contextlib
import copy
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from foreblocks.core.heads.head_helper import HeadComposer
from foreblocks.core.heads.head_helper import HeadSpec
from foreblocks.ui.node_spec import node


# Optional: import your MoE classes and HeadComposer
try:
    from foreblocks.tf.experts.moe import FeedForwardBlock
    from foreblocks.tf.experts.moe import MoEFeedForwardDMoE
    from foreblocks.tf.experts.moe_logging import MoELogger
    from foreblocks.tf.experts.moe_logging import ReportInputs
    from foreblocks.tf.experts.moe_logging import build_moe_report
except Exception:
    MoELogger = None
    ReportInputs = None
    def build_moe_report(*args, **kwargs):
        raise RuntimeError("MoE logging not available")

# Try to import HeadComposer for NAS detection

from foreblocks.evaluation.model_evaluator import ModelEvaluator


# ============================================================================
# Configuration Management
# ============================================================================
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dataclasses import dataclass
from typing import Any, Dict, Optional


from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Type-safe training configuration with NAS + Conformal support."""

    # -------------------------------------------------
    # Core training
    # -------------------------------------------------
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

    # Scheduler
    scheduler_type: Optional[str] = None
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    min_lr: float = 1e-6

    # Logging / saving
    verbose: bool = True
    log_interval: int = 10
    save_best_model: bool = True
    save_model_path: Optional[str] = None
    experiment_name: str = "default_experiment" # New: for MLTracker

    # -------------------------------------------------
    # MoE logging toggles
    # -------------------------------------------------
    moe_logging: bool = False
    moe_log_latency: bool = False
    moe_condition_name: Optional[str] = None
    moe_condition_cardinality: Optional[int] = None

    # -------------------------------------------------
    # NAS training toggles (HeadComposer alphas)
    # -------------------------------------------------
    train_nas: bool = False
    nas_alpha_lr: float = 3e-4
    nas_alpha_weight_decay: float = 1e-3
    nas_warmup_epochs: int = 5
    nas_alternate_steps: int = 1
    nas_use_val_for_alpha: bool = True
    nas_discretize_at_end: bool = True
    nas_discretize_threshold: float = 0.5
    nas_log_alphas: bool = True

    # -------------------------------------------------
    # Conformal Prediction Configuration
    # -------------------------------------------------
    # Master switch
    conformal_enabled: bool = False

    # Method selection
    # Options: "split", "local", "jackknife", "quantile", "tsp", 
    #          "rolling", "agaci", "enbpi", "cptc", "afocp"
    conformal_method: str = "split"

    # Target coverage (e.g., 0.9 = 90% coverage)
    conformal_quantile: float = 0.9

    # --- Local Method ---
    conformal_knn_k: int = 50
    conformal_local_window: int = 5000

    # --- Rolling / ACI ---
    conformal_aci_gamma: float = 0.01
    conformal_rolling_alpha: float = 0.1

    # --- AgACI ---
    conformal_agaci_gammas: Optional[List[float]] = None  # Default: [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    # --- EnbPI ---
    conformal_enbpi_B: int = 20
    conformal_enbpi_window: int = 500

    # --- TSP (Time Series) ---
    conformal_tsp_lambda: float = 0.01
    conformal_tsp_window: int = 5000

    # --- CPTC (State-Aware) ---
    conformal_cptc_window: int = 500
    conformal_cptc_tau: float = 1.0
    conformal_cptc_hard_state_filter: bool = False

    # --- AFOCP (Attention-Based) ---
    conformal_afocp_feature_dim: int = 128
    conformal_afocp_attn_hidden: int = 64
    conformal_afocp_window: int = 500
    conformal_afocp_tau: float = 1.0
    conformal_afocp_internal_feat_hidden: int = 256
    conformal_afocp_internal_feat_depth: int = 3
    conformal_afocp_internal_feat_dropout: float = 0.1
    conformal_afocp_online_lr: float = 0.0
    conformal_afocp_online_steps: int = 1

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.conformal_agaci_gammas is None:
            self.conformal_agaci_gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    def update(self, **kwargs):
        """Safe update: only allow known keys."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config key '{key}' not found in TrainingConfig")

    def get_conformal_params(self) -> dict:
        """Extract conformal parameters for engine initialization."""
        return {
            "method": self.conformal_method,
            "quantile": self.conformal_quantile,
            # Local
            "knn_k": self.conformal_knn_k,
            "local_window": self.conformal_local_window,
            # Rolling/ACI
            "aci_gamma": self.conformal_aci_gamma,
            # AgACI
            "agaci_gammas": self.conformal_agaci_gammas,
            # EnbPI
            "enbpi_B": self.conformal_enbpi_B,
            "enbpi_window": self.conformal_enbpi_window,
            # TSP
            "tsp_lambda": self.conformal_tsp_lambda,
            "tsp_window": self.conformal_tsp_window,
            # CPTC
            "cptc_window": self.conformal_cptc_window,
            "cptc_tau": self.conformal_cptc_tau,
            "cptc_hard_state_filter": self.conformal_cptc_hard_state_filter,
            # AFOCP
            "afocp_feature_dim": self.conformal_afocp_feature_dim,
            "afocp_attn_hidden": self.conformal_afocp_attn_hidden,
            "afocp_window": self.conformal_afocp_window,
            "afocp_tau": self.conformal_afocp_tau,
            "afocp_internal_feat_hidden": self.conformal_afocp_internal_feat_hidden,
            "afocp_internal_feat_depth": self.conformal_afocp_internal_feat_depth,
            "afocp_internal_feat_dropout": self.conformal_afocp_internal_feat_dropout,
            "afocp_online_lr": self.conformal_afocp_online_lr,
            "afocp_online_steps": self.conformal_afocp_online_steps,
        }

# ============================================================================
# Dataset (unchanged)
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Clean dataset for time series data"""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, time_feat: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.t = torch.tensor(time_feat, dtype=torch.long) if time_feat is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None and self.t is None:
            return self.X[idx]
        if self.t is None:
            return self.X[idx], self.y[idx]
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx], self.t[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    time_feat_train: Optional[np.ndarray] = None,
    time_feat_val: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:

    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train, time_feat_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            TimeSeriesDataset(X_val, y_val, time_feat_val),
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loader


# ============================================================================
# Loss Computation (unchanged)
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
        self.components = {}
        aux_data = aux_data or {}

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

        # L1
        if self.config.l1_regularization > 0:
            l1_loss = sum(p.abs().sum() for p in self.model.parameters() if p.requires_grad)
            total_loss += self.config.l1_regularization * l1_loss
            self.components["l1_loss"] = (self.config.l1_regularization * l1_loss).item()

        # KL
        if hasattr(self.model, "get_kl"):
            kl_div = self.model.get_kl()
            if kl_div is not None:
                total_loss += self.config.kl_weight * kl_div
                self.components["kl_loss"] = (self.config.kl_weight * kl_div).item()

        # MoE aux
        if hasattr(self.model, "get_aux_loss"):
            aux_loss = self.model.get_aux_loss()
            if aux_loss is not None:
                total_loss += aux_loss.detach()
                self.components["aux_loss"] = aux_loss.item()

        return total_loss


# ============================================================================
# Training History (enhanced with alpha tracking)
# ============================================================================

@dataclass
class TrainingHistory:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    task_losses: List[float] = field(default_factory=list)
    distillation_losses: List[float] = field(default_factory=list)
    model_info: List[Dict[str, Any]] = field(default_factory=list)

    # ── NEW: NAS tracking ──
    alpha_values: List[Dict[str, Any]] = field(default_factory=list)  # Per-epoch alpha snapshots

    def record_epoch(
        self,
        train_loss: float,
        val_loss: Optional[float],
        lr: float,
        loss_components: Dict[str, float],
        model_info: Optional[Dict[str, Any]] = None,
        alpha_info: Optional[Dict[str, Any]] = None,
    ):
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

        if alpha_info:
            self.alpha_values.append(alpha_info)


# ============================================================================
# NAS Helper - detects and manages HeadComposer alphas
# ============================================================================

class NASHelper:
    """
    Utility to detect HeadComposer modules and manage alpha optimization.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.composers: List[Tuple[str, nn.Module]] = []

        # Find all HeadComposer instances
        if HeadComposer is not None:
            for name, module in model.named_modules():
                if isinstance(module, HeadComposer):
                    self.composers.append((name, module))

        self.has_nas = len(self.composers) > 0

        if config.train_nas and not self.has_nas:
            logging.warning("train_nas=True but no HeadComposer found in model")

    def get_alpha_parameters(self) -> List[torch.nn.Parameter]:
        """Collect all alpha parameters from all composers."""
        params = []
        for name, composer in self.composers:
            params.extend(composer.arch_parameters())
        return params

    def get_weight_parameters(self) -> List[torch.nn.Parameter]:
        """Collect all non-alpha parameters."""
        if not self.has_nas:
            return list(self.model.parameters())

        # Collect alpha param IDs to exclude
        alpha_ids = {id(p) for _, composer in self.composers
                    for p in composer.arch_parameters()}

        # Return all other params
        weight_params = []
        for p in self.model.parameters():
            if id(p) not in alpha_ids:
                weight_params.append(p)
        return weight_params

    def collect_alpha_report(self) -> Dict[str, Any]:
        """Collect alpha values from all composers."""
        report = {}
        for name, composer in self.composers:
            report[name] = composer.alpha_report()
        return report

    def discretize_all(self, threshold: Optional[float] = None):
        """Discretize alphas in all composers."""
        thresh = threshold or self.config.nas_discretize_threshold
        for name, composer in self.composers:
            composer.discretize_(threshold=thresh)
            logging.info(f"Discretized alphas in {name} with threshold={thresh}")



# ============================================================================
# Utility: Plot alpha evolution
# ============================================================================

def plot_alpha_evolution(history: TrainingHistory, figsize: Tuple[int, int] = (15, 6)):
    """
    Plot the evolution of alpha values across training epochs.

    Args:
        history: TrainingHistory with alpha_values recorded
        figsize: Figure size
    """
    if not history.alpha_values:
        print("No alpha values recorded in history")
        return None

    # Organize data by composer and head
    alpha_series = {}  # {(composer, head, metric): [values]}

    for epoch_alphas in history.alpha_values:
        for composer_name, heads in epoch_alphas.items():
            for head_name, values in heads.items():
                for metric, value in values.items():
                    if isinstance(value, (int, float)):
                        key = (composer_name, head_name, metric)
                        if key not in alpha_series:
                            alpha_series[key] = []
                        alpha_series[key].append(value)

    if not alpha_series:
        print("No numeric alpha values found")
        return None

    # Group by composer
    composers = {}
    for (comp, head, metric), values in alpha_series.items():
        if comp not in composers:
            composers[comp] = {}
        if head not in composers[comp]:
            composers[comp][head] = {}
        composers[comp][head][metric] = values

    # Create subplots
    n_composers = len(composers)
    fig, axes = plt.subplots(n_composers, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    epochs = list(range(len(history.alpha_values)))

    for idx, (composer_name, heads) in enumerate(composers.items()):
        ax = axes[idx]

        for head_name, metrics in heads.items():
            for metric, values in metrics.items():
                if metric in ['w_head', 'p_on']:  # Plot relevant metrics
                    label = f"{head_name}.{metric}"
                    ax.plot(epochs, values, label=label, marker='o', markersize=3)

        ax.set_title(f"Alpha Evolution: {composer_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Alpha Value")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()
    return fig
