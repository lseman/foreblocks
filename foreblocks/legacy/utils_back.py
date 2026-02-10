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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from foreblocks.ui_aux.node_spec import node

# Optional: import your MoE classes and HeadComposer
try:
    from foreblocks.tf.experts.moe import FeedForwardBlock, MoEFeedForwardDMoE
    from foreblocks.tf.experts.moe_logging import MoELogger, ReportInputs, build_moe_report
except Exception:
    MoELogger = None
    ReportInputs = None
    def build_moe_report(*args, **kwargs):
        raise RuntimeError("MoE logging not available")

# Try to import HeadComposer for NAS detection
try:
    from foreblocks.heads.head_composer import HeadComposer
except Exception:
    HeadComposer = None


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class TrainingConfig:
    """Type-safe training configuration with NAS support"""
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

    # MoE logging toggles
    moe_logging: bool = False
    moe_log_latency: bool = False
    moe_condition_name: Optional[str] = None
    moe_condition_cardinality: Optional[int] = None

    # ── NEW: NAS training toggles ──
    train_nas: bool = False                    # Enable two-step NAS optimization
    nas_alpha_lr: float = 3e-4                 # Learning rate for α parameters
    nas_alpha_weight_decay: float = 1e-3       # Weight decay for α
    nas_warmup_epochs: int = 5                 # Epochs before starting α optimization
    nas_alternate_steps: int = 1               # Steps of α opt per weight opt step
    nas_use_val_for_alpha: bool = True         # Use validation loss for α updates
    nas_discretize_at_end: bool = True         # Discretize alphas after training
    nas_discretize_threshold: float = 0.5      # Threshold for discretization
    nas_log_alphas: bool = True                # Log alpha values during training

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config key '{key}' not found")


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
# Core Trainer (enhanced with NAS)
# ============================================================================

@node(
  type_id="trainer",
  name="Trainer",
  category="Training",
  inputs=["X_train","Y_train","model"],
  outputs=["trained_model"],
  config={"epochs":100},
  infer=False
)
class Trainer:
    """Trainer for time series models with optional MoE logging and NAS support"""

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
        moe_meta_builder: Optional[Callable[[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int, int], Optional[Dict[str, Any]]]] = None,
        # ── NEW: optional custom alpha optimizer ──
        alpha_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_wandb = use_wandb

        if isinstance(config, dict):
            self.config = TrainingConfig()
            self.config.update(**config)
        else:
            self.config = config or TrainingConfig()

        # ── NAS setup ──
        self.nas_helper = NASHelper(self.model, self.config)
        
        if self.config.train_nas and self.nas_helper.has_nas:
            # Separate optimizers for NAS
            weight_params = self.nas_helper.get_weight_parameters()
            alpha_params = self.nas_helper.get_alpha_parameters()
            
            self.optimizer = optimizer or torch.optim.AdamW(
                weight_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            
            self.alpha_optimizer = alpha_optimizer or torch.optim.Adam(
                alpha_params,
                lr=self.config.nas_alpha_lr,
                weight_decay=self.config.nas_alpha_weight_decay,
            )
            
            logging.info(f"NAS mode enabled: {len(alpha_params)} alpha params, {len(weight_params)} weight params")
        else:
            # Standard training
            self.optimizer = optimizer or self._create_optimizer()
            self.alpha_optimizer = None

        self.scheduler = scheduler or self._create_scheduler()
        self.loss_computer = LossComputer(self.model, self.config, criterion)
        self.scaler = GradScaler() if self.config.use_amp else None

        self.history = TrainingHistory()
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.current_epoch = 0

        # MoE logging setup (unchanged)
        self.global_step = 0
        self.moe_log: Optional[MoELogger] = None
        self.moe_meta_builder = moe_meta_builder or self._default_moe_meta_builder

        if self.config.moe_logging and MoELogger is not None:
            print("MoE logging enabled.")
            self.moe_log = MoELogger()
            self._wire_moe_logger(self.model, self.moe_log, lambda: self.global_step, self.config.moe_log_latency)
        else:
            self.moe_log = None

        if self.use_wandb:
            try:
                import wandb
                wandb.init(**(wandb_config or {}))
                wandb.watch(self.model, log="all", log_freq=100)
            except ImportError:
                logging.warning("wandb not installed, disabling W&B logging")

    @staticmethod
    def _default_moe_meta_builder(
        X: torch.Tensor,
        y: Optional[torch.Tensor],
        time_feat: Optional[torch.Tensor],
        epoch: int,
        batch_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Build a small meta dict from time_feat."""
        if time_feat is None:
            return None
        meta = {}
        if time_feat.dtype in (torch.int32, torch.int64) and time_feat.ndim >= 1:
            meta["hour"] = time_feat.view(-1).clamp_min(0).clamp_max(23)
        return meta or None

    def _wire_moe_logger(self, module: nn.Module, moe_logger: "MoELogger", step_getter: Callable[[], int], log_latency: bool):
        """Wire MoE logger into MoE modules (unchanged)."""
        for child in module.modules():
            try:
                is_moe = False
                if MoEFeedForwardDMoE is not None and isinstance(child, MoEFeedForwardDMoE):
                    is_moe = True
                if FeedForwardBlock is not None and isinstance(child, FeedForwardBlock) and getattr(child, "use_moe", False):
                    is_moe = True
                    moe_block = getattr(child, "block", None)
                    if moe_block is not None:
                        setattr(moe_block, "moe_logger", moe_logger)
                        setattr(moe_block, "step_getter", step_getter)
                        setattr(moe_block, "log_latency", bool(log_latency))
                        continue
                if is_moe and hasattr(child, "moe_logger"):
                    setattr(child, "moe_logger", moe_logger)
                    setattr(child, "step_getter", step_getter)
                    setattr(child, "log_latency", bool(log_latency))
            except Exception:
                pass

    @property
    def criterion(self):
        return self.loss_computer.criterion

    @criterion.setter
    def criterion(self, value):
        self.loss_computer.criterion = value

    def _create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> Optional[Any]:
        if self.config.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        return None

    @contextlib.contextmanager
    def _amp_context(self):
        if self.config.use_amp and self.device == "cuda":
            with autocast("cuda"):
                yield
        else:
            yield

    def _forward_pass(
        self, X: torch.Tensor, y: torch.Tensor, time_feat: Optional[torch.Tensor] = None, batch_idx: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with optional MoE meta (unchanged)."""
        aux = {}
        meta = self.moe_meta_builder(X, y, time_feat, self.current_epoch, batch_idx) if self.moe_log is not None else None

        if hasattr(self.model, "get_distillation_info"):
            distill_info = self.model.get_distillation_info()
            if distill_info.get("distillation_enabled", False):
                try:
                    result = self.model(X, y, time_feat, self.current_epoch, return_teacher_outputs=True, meta=meta)
                except TypeError:
                    result = self.model(X, y, time_feat, self.current_epoch, return_teacher_outputs=True)
                if isinstance(result, tuple) and len(result) == 2:
                    outputs, aux["teacher_outputs"] = result[0], result[1]
                else:
                    outputs = result[0] if isinstance(result, tuple) else result
                return outputs, aux

        try:
            result = self.model(X, y, time_feat, self.current_epoch, meta=meta)
        except TypeError:
            result = self.model(X, y, time_feat, self.current_epoch)

        outputs = result[0] if isinstance(result, tuple) else result
        return outputs, aux

    def _backward_step(self, loss: torch.Tensor, batch_idx: int, total_batches: int, optimizer: torch.optim.Optimizer):
        """Backward step with gradient accumulation."""
        loss = loss / self.config.gradient_accumulation_steps

        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or batch_idx + 1 == total_batches:
                if self.config.gradient_clip_val:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or batch_idx + 1 == total_batches:
                if self.config.gradient_clip_val:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                optimizer.step()
                optimizer.zero_grad()

    # ── NEW: Alpha optimization step ──
    def _alpha_step(self, dataloader: DataLoader, num_steps: int = 1) -> float:
        """
        Optimize alpha parameters using validation data.
        
        Args:
            dataloader: Validation dataloader (or train if val not available)
            num_steps: Number of batches to process
            
        Returns:
            Average loss over the steps
        """
        self.model.train()
        total_loss = 0.0
        
        dataloader_iter = iter(dataloader)
        
        for step in range(num_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    X, y, time_feat = batch
                elif len(batch) == 2:
                    X, y = batch
                    time_feat = None
                else:
                    X, y, time_feat = batch[0], batch[1], (batch[2] if len(batch) > 2 else None)
            else:
                X, y, time_feat = batch, None, None

            X = X.to(self.device)
            if y is not None:
                y = y.to(self.device)
            if time_feat is not None:
                time_feat = time_feat.to(self.device)

            # Forward pass
            with self._amp_context():
                outputs, aux = self._forward_pass(X, y, time_feat, 0)
                if y is not None and outputs.ndim == 4 and y.ndim == 3:
                    y = y.unsqueeze(-1)
                target = y if y is not None else outputs.detach() * 0
                loss = self.loss_computer.compute(outputs, target, aux)

            # Backward only for alpha params
            self._backward_step(loss, step, num_steps, self.alpha_optimizer)
            total_loss += loss.item()

        return total_loss / num_steps

    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch with optional NAS alpha optimization.
        
        When train_nas=True and past warmup:
          - Alternate between alpha steps (on val data) and weight steps (on train data)
        """
        self.model.train()
        total_loss = 0.0
        all_components: Dict[str, List[float]] = {}
        
        # Check if we should do NAS this epoch
        do_nas = (
            self.config.train_nas 
            and self.nas_helper.has_nas 
            and self.current_epoch >= self.config.nas_warmup_epochs
            and val_loader is not None
        )

        for batch_idx, batch in enumerate(train_loader):
            # ── Step A: Alpha optimization (if NAS enabled) ──
            if do_nas and self.config.nas_use_val_for_alpha:
                alpha_loss = self._alpha_step(val_loader, num_steps=self.config.nas_alternate_steps)
                # Optionally log alpha loss
                all_components.setdefault("alpha_loss", []).append(alpha_loss)

            # ── Step B: Weight optimization (standard training) ──
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    X, y, time_feat = batch
                elif len(batch) == 2:
                    X, y = batch
                    time_feat = None
                else:
                    X, y, time_feat = batch[0], batch[1], (batch[2] if len(batch) > 2 else None)
            else:
                X, y, time_feat = batch, None, None

            X = X.to(self.device)
            if y is not None:
                y = y.to(self.device)
            if time_feat is not None:
                time_feat = time_feat.to(self.device)

            with self._amp_context():
                outputs, aux = self._forward_pass(X, y, time_feat, batch_idx)
                if y is not None and outputs.ndim == 4 and y.ndim == 3:
                    y = y.unsqueeze(-1)
                target = y if y is not None else outputs.detach() * 0
                loss = self.loss_computer.compute(outputs, target, aux)

            self._backward_step(loss, batch_idx, len(train_loader), self.optimizer)
            total_loss += loss.item()
            self.global_step += 1

            for k, v in self.loss_computer.components.items():
                all_components.setdefault(k, []).append(v)

        avg_components = {k: np.mean(v) for k, v in all_components.items()}
        return total_loss / len(train_loader), avg_components

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on dataloader (unchanged)."""
        if dataloader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                X, y = batch[0], batch[1]
            else:
                X, y = batch, None
            X = X.to(self.device)
            if y is not None:
                y = y.to(self.device)

            with self._amp_context():
                try:
                    result = self.model(X)
                except TypeError:
                    result = self.model(X, y, None, self.current_epoch)
                outputs = result[0] if isinstance(result, tuple) else result

                if y is None:
                    continue
                loss = nn.MSELoss()(outputs, y)
                bs = X.size(0)
                total_loss += loss.item() * bs
                n += bs

        return (total_loss / max(n, 1)) if n > 0 else float("nan")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Any]] = None,
        epochs: Optional[int] = None,
        moe_report_outdir: Optional[str] = None,
    ) -> TrainingHistory:
        """
        Main training loop with NAS support.
        
        When train_nas=True:
          - Warmup for nas_warmup_epochs with weight optimization only
          - Then alternate between alpha and weight optimization
          - Optionally discretize alphas at the end
        """
        callbacks = callbacks or []
        num_epochs = epochs if epochs is not None else self.config.num_epochs

        # Validate NAS requirements
        if self.config.train_nas and self.nas_helper.has_nas:
            if val_loader is None and self.config.nas_use_val_for_alpha:
                logging.warning("train_nas=True but no validation loader provided. Alpha will use train data.")
            logging.info(f"NAS training enabled with {self.config.nas_warmup_epochs} warmup epochs")

        for cb in callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(self)

        with tqdm(range(num_epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                self.current_epoch = epoch

                for cb in callbacks:
                    if hasattr(cb, "on_epoch_begin"):
                        cb.on_epoch_begin(self, epoch)

                # Train epoch with potential NAS
                train_loss, loss_components = self.train_epoch(train_loader, val_loader)
                val_loss = self.evaluate(val_loader) if val_loader else None

                lr = self.optimizer.param_groups[0]["lr"]
                model_info = self.model.get_model_size() if hasattr(self.model, "get_model_size") else None
                
                # ── Collect alpha info if NAS ──
                alpha_info = None
                if self.config.train_nas and self.nas_helper.has_nas and self.config.nas_log_alphas:
                    alpha_info = self.nas_helper.collect_alpha_report()
                
                self.history.record_epoch(train_loss, val_loss, lr, loss_components, model_info, alpha_info)

                # W&B logging
                if self.use_wandb:
                    try:
                        import wandb
                        log_dict = {"epoch": epoch + 1, "train_loss": train_loss, "learning_rate": lr}
                        if val_loss is not None:
                            log_dict["val_loss"] = val_loss
                        log_dict.update(loss_components)
                        if model_info:
                            log_dict.update(model_info)
                        if alpha_info:
                            # Flatten alpha info for W&B
                            for composer_name, heads in alpha_info.items():
                                for head_name, values in heads.items():
                                    for k, v in values.items():
                                        if isinstance(v, (int, float)):
                                            log_dict[f"alpha/{composer_name}/{head_name}/{k}"] = v
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
                postfix = {"train_loss": f"{train_loss:.4f}", "lr": f"{lr:.2e}"}
                if val_loss is not None:
                    postfix["val_loss"] = f"{val_loss:.4f}"
                if alpha_info and self.config.nas_log_alphas:
                    # Show a summary of alpha values
                    postfix["alphas"] = "tracked"
                pbar.set_postfix(postfix)

                if self.scheduler:
                    self.scheduler.step(val_loss if val_loader else train_loss)

                logs = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
                for cb in callbacks:
                    if hasattr(cb, "on_epoch_end"):
                        cb.on_epoch_end(self, epoch, logs)

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        # ── NAS: Discretize alphas if requested ──
        if self.config.train_nas and self.nas_helper.has_nas and self.config.nas_discretize_at_end:
            print("\n" + "="*60)
            print("Discretizing alpha parameters...")
            print("="*60)
            pre_discretize = self.nas_helper.collect_alpha_report()
            self.nas_helper.discretize_all()
            post_discretize = self.nas_helper.collect_alpha_report()
            
            print("\nFinal Architecture Decisions:")
            for composer_name, heads in post_discretize.items():
                print(f"\n{composer_name}:")
                for head_name, values in heads.items():
                    hardened = values.get('hardened', None)
                    if hardened is not None:
                        status = "ENABLED" if hardened else "DISABLED"
                        print(f"  {head_name}: {status}")
                    else:
                        print(f"  {head_name}: (no alpha)")

        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(self)

        # MoE report export
        if moe_report_outdir and self.moe_log is not None and ReportInputs is not None:
            ri = ReportInputs(
                log=self.moe_log.state_dict(),
                E=self._infer_num_experts(self.model),
                condition_name=self.config.moe_condition_name,
                condition_cardinality=self.config.moe_condition_cardinality,
            )
            build_moe_report(ri, outdir=moe_report_outdir)
            print(f"MoE report exported to: {moe_report_outdir}")

        return self.history

    @staticmethod
    def _infer_num_experts(model: nn.Module) -> Optional[int]:
        for m in model.modules():
            if hasattr(m, "num_experts"):
                try:
                    ne = int(getattr(m, "num_experts"))
                    if ne > 0:
                        return ne
                except Exception:
                    pass
        return None

    def save(self, path: str):
        """Save model, optimizer, and training state."""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "history": {
                "train_losses": self.history.train_losses,
                "val_losses": self.history.val_losses,
                "learning_rates": self.history.learning_rates,
                "alpha_values": self.history.alpha_values,
            },
        }
        if self.alpha_optimizer is not None:
            save_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        """Load model, optimizer, and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "alpha_optimizer_state_dict" in checkpoint and self.alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        if "config" in checkpoint:
            self.config.update(**checkpoint["config"])

    # ── Visualization methods (unchanged from original) ──
    
    def plot_prediction(self, X_val: torch.Tensor, y_val: torch.Tensor, full_series: Optional[torch.Tensor] = None,
                        offset: int = 0, stride: int = 1, figsize: Tuple[int, int] = (12, 4),
                        show: bool = True, names: Optional[Union[str, list]] = None) -> plt.Figure:
        evaluator = ModelEvaluator(self)
        predictions = evaluator.predict(X_val)
        N, H = predictions.shape[0], predictions.shape[1]
        D = predictions.shape[2] if predictions.ndim >= 3 else 1

        if full_series is not None:
            series = full_series.detach().cpu().numpy()
            if series.ndim == 1:
                series = series[:, None]
            T, S_dim = series.shape
            D_plot = S_dim
            names = names or [f"Feature {i}" for i in range(D_plot)]
            seq_len = X_val.shape[1]
            starts = offset + seq_len + np.arange(N) * stride
            coverage_end = min(T, int(starts[-1] + H)) if N > 0 else 0

            fig, axes = plt.subplots(D_plot, 1, figsize=(figsize[0], figsize[1] * D_plot), sharex=True)
            axes = np.atleast_1d(axes)
            pred_np = predictions.detach().cpu().numpy()

            for j in range(D_plot):
                ax = axes[j]
                acc = np.zeros(T); cnt = np.zeros(T)
                for k in range(N):
                    s = int(starts[k])
                    if s >= T: continue
                    e = min(s + H, T)
                    if e > s:
                        pred_col = j if D > j else 0
                        acc[s:e] += pred_np[k, :e - s, pred_col]
                        cnt[s:e] += 1
                have = cnt > 0
                mean_pred = np.zeros(T)
                mean_pred[have] = acc[have] / cnt[have]
                x = np.arange(coverage_end)
                ax.plot(series[:coverage_end, j], label=f"Actual {names[j]}", alpha=0.8)
                if have[:coverage_end].any():
                    ax.plot(x[have[:coverage_end]], mean_pred[:coverage_end][have[:coverage_end]],
                            label=f"Predicted {names[j]}", linestyle='--')
                    y_true = series[:coverage_end, j][have[:coverage_end]]
                    y_pred = mean_pred[:coverage_end][have[:coverage_end]]
                    mask = np.isfinite(y_true) & np.isfinite(y_pred)
                    if mask.any():
                        ax.fill_between(x[have[:coverage_end]][mask], y_pred[mask], y_true[mask], alpha=0.2, label="Error")
                ax.axvline(offset + seq_len, color='gray', linestyle='--', alpha=0.5, label='First forecast')
                ax.set_title(f"{names[j]}: Prediction vs Actual")
                ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time Step")
            plt.tight_layout()
            if show: plt.show()
            return fig

        # Simple plot
        pred_np = predictions.detach().cpu().numpy()
        y_np = y_val.detach().cpu().numpy()
        pred_mean = pred_np.mean(axis=0); y_mean = y_np.mean(axis=0)
        if pred_mean.ndim == 1:
            pred_mean = pred_mean[:, None]; y_mean = y_mean[:, None]
        D_plot = pred_mean.shape[1]
        names = names or [f"Feature {i}" for i in range(D_plot)]

        fig, axes = plt.subplots(D_plot, 1, figsize=(figsize[0], figsize[1] * D_plot), sharex=True)
        axes = np.atleast_1d(axes)
        for j in range(D_plot):
            ax = axes[j]; horizon = np.arange(len(pred_mean))
            ax.plot(horizon, y_mean[:, j], label=f"Actual {names[j]}", marker='o', alpha=0.7)
            ax.plot(horizon, pred_mean[:, j], label=f"Predicted {names[j]}", marker='s', linestyle='--', alpha=0.7)
            ax.set_title(f"{names[j]}: Average Forecast"); ax.legend(); ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Forecast Horizon")
        plt.tight_layout()
        if show: plt.show()
        return fig

    def metrics(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        evaluator = ModelEvaluator(self)
        return evaluator.compute_metrics(X_val, y_val, batch_size)

    def cv(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_windows: int,
        horizon: int,
        step_size: Optional[int] = None,
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """Time series cross-validation."""
        evaluator = ModelEvaluator(self)
        return evaluator.cross_validation(X, y, n_windows, horizon, step_size, batch_size)


# ============================================================================
# Model Evaluator (unchanged from original)
# ============================================================================

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