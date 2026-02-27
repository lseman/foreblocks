# ─── Standard Library ─────────────────────────────────────────────
import concurrent.futures
import copy
import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ─── Third-Party Libraries ────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ─── Local Imports ─────────────────────────────────────────────────
from .architecture import *
from .architecture.finalization import (
    derive_final_architecture as derive_fixed_architecture,
)
from .darts_metrics import *
from .scoring import score_from_metrics
from .search.candidate_scoring import candidate_signature
from .search.orchestrator import (
    evaluate_search_candidate,
    make_default_search_candidate_config,
    run_parallel_candidate_collection,
    select_top_candidates,
)
from .search.stats_reporting import (
    append_whatif_estimates,
    mean_std,
    save_csv,
    save_json,
)
from .search.weight_schemes import (
    build_weight_schemes,
)
from .search.weight_schemes import (
    ranks_desc as _ranks_desc,
)
from .search.weight_schemes import (
    spearman_from_scores as _spearman_from_scores,
)
from .search.weight_schemes import (
    topk_overlap_from_scores as _topk_overlap_from_scores,
)
from .training.helpers import (
    AlphaTracker,
    ArchitectureRegularizer,
    BilevelOptimizer,
    RegularizationType,
    TemperatureScheduler,
)
from .training.helpers import (
    default_as_probability_vector as _as_probability_vector,
)

# Optional: configure a custom logger
logger = logging.getLogger("NASLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# Stable architecture-signature alias used throughout this module.
_sig_from_cfg = candidate_signature


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
        self.alpha_tracker = AlphaTracker(
            as_probability_vector_fn=_as_probability_vector
        )

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
            "PatchEmbed",
            "TCN",
            "Mamba",
            "InvertedAttention",
        ]

        # Training history
        self.search_history = []
        self.training_history = []

        print(f"🚀 DARTSTrainer initialized on {device}")
        print(f"   Input dim: {input_dim}, Forecast horizon: {forecast_horizon}")
        print(f"   Available operations: {len(self.all_ops)}")

    def _get_loss_function(self, loss_type: str):
        loss_functions = {
            "huber": lambda p, t: F.huber_loss(p, t, delta=0.1),
            "mse": F.mse_loss,
            "mae": F.l1_loss,
            "smooth_l1": F.smooth_l1_loss,
        }
        return loss_functions.get(loss_type, loss_functions["huber"])

    def _create_progress_bar(self, iterable, desc: str, leave: bool = True, **kwargs):
        if "unit" not in kwargs:
            kwargs["unit"] = "batch"
        return tqdm(iterable, desc=desc, leave=leave, **kwargs)

    def _autocast(self, enabled: bool):
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        return autocast(device_type=device_type, enabled=enabled)

    def _split_architecture_and_model_params(self, model):
        """Collect and split trainable params into architecture vs model groups."""
        arch_params: List[torch.Tensor] = []
        model_params: List[torch.Tensor] = []
        edge_arch_params: List[torch.Tensor] = []
        component_arch_params: List[torch.Tensor] = []
        arch_param_ids = set()

        for name, param in model.named_parameters():
            if any(
                arch_name in name
                for arch_name in ["alphas", "arch_", "alpha_", "norm_alpha"]
            ):
                if id(param) in arch_param_ids:
                    continue
                arch_params.append(param)
                arch_param_ids.add(id(param))
                if "cells." in name and "edges." in name:
                    edge_arch_params.append(param)
                else:
                    component_arch_params.append(param)
            else:
                model_params.append(param)

        # Include component alpha tensors even if name heuristics miss them.
        for source in self.alpha_tracker.component_alpha_sources(model):
            alpha_tensor = source["alpha"]
            if id(alpha_tensor) in arch_param_ids:
                continue
            arch_params.append(alpha_tensor)
            arch_param_ids.add(id(alpha_tensor))
            component_arch_params.append(alpha_tensor)

        return arch_params, model_params, edge_arch_params, component_arch_params

    def _build_arch_param_groups(
        self,
        edge_arch_params: List[torch.Tensor],
        component_arch_params: List[torch.Tensor],
        arch_learning_rate: float,
        arch_params: List[torch.Tensor],
    ):
        """Build optimizer groups for architecture parameters."""
        arch_param_groups = []
        if edge_arch_params:
            arch_param_groups.append(
                {"params": edge_arch_params, "lr": arch_learning_rate * 1.5}
            )
        if component_arch_params:
            arch_param_groups.append(
                {"params": component_arch_params, "lr": arch_learning_rate}
            )
        if not arch_param_groups:
            arch_param_groups = [{"params": arch_params, "lr": arch_learning_rate}]
        return arch_param_groups

    def _reset_model_parameters(self, model: nn.Module) -> int:
        """
        Reinitialize all modules exposing `reset_parameters`.
        Returns number of modules reset.
        """
        reset_count = 0
        for module in model.modules():
            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                    reset_count += 1
                except Exception:
                    continue
        return reset_count

    def _capture_progressive_state(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Capture per-cell progressive stage so best checkpoints can be restored safely."""
        if not hasattr(model, "cells"):
            return None

        cells_state = []
        for cell in getattr(model, "cells", []):
            cells_state.append(
                {"progressive_stage": getattr(cell, "progressive_stage", None)}
            )

        return {"cells": cells_state}

    def _restore_progressive_state(
        self, model: nn.Module, state: Optional[Dict[str, Any]]
    ) -> None:
        """Restore per-cell progressive stage prior to loading a checkpoint."""
        if state is None or not hasattr(model, "cells"):
            return

        cells = getattr(model, "cells", [])
        saved_cells = state.get("cells", [])
        for idx, cell in enumerate(cells):
            if idx >= len(saved_cells):
                break

            stage = saved_cells[idx].get("progressive_stage")
            if stage is None:
                continue

            if hasattr(cell, "set_progressive_stage"):
                try:
                    cell.set_progressive_stage(stage)
                except Exception:
                    continue
            else:
                cell.progressive_stage = stage

    def _run_model_training_epoch(
        self,
        *,
        model: nn.Module,
        train_model_loader,
        model_params: List[torch.Tensor],
        model_optimizer,
        model_scheduler,
        scaler: GradScaler,
        loss_fn,
        gradient_accumulation_steps: int,
        use_amp: bool,
        verbose: bool,
        epoch: int,
    ) -> float:
        """Run one model-parameter training epoch and return mean training loss."""
        model.train()
        epoch_train_loss = 0.0
        batch_pbar = (
            self._create_progress_bar(
                enumerate(train_model_loader),
                f"Epoch {epoch + 1:3d}",
                leave=False,
                total=len(train_model_loader),
            )
            if verbose
            else enumerate(train_model_loader)
        )

        model_optimizer.zero_grad()
        for batch_idx, (batch_x, batch_y, *_) in batch_pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            with self._autocast(use_amp):
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

            if verbose and hasattr(batch_pbar, "set_postfix"):
                batch_pbar.set_postfix(
                    {
                        "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                        "avg": f"{epoch_train_loss / (batch_idx + 1):.4f}",
                    }
                )

        if verbose and hasattr(batch_pbar, "close"):
            batch_pbar.close()

        return epoch_train_loss / max(len(train_model_loader), 1)

    def _run_validation_epoch(
        self,
        *,
        model: nn.Module,
        val_loader,
        loss_fn,
        use_amp: bool,
        verbose: bool,
    ) -> float:
        """Run one validation epoch and return mean validation loss."""
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = (
                self._create_progress_bar(val_loader, "Val", leave=False)
                if verbose
                else val_loader
            )

            for batch_data in val_pbar:
                batch_x, batch_y = (
                    batch_data[0].to(self.device),
                    batch_data[1].to(self.device),
                )

                with self._autocast(use_amp):
                    preds = model(batch_x)
                    val_loss += loss_fn(preds, batch_y).item()

                if verbose and hasattr(val_pbar, "set_postfix"):
                    val_pbar.set_postfix(
                        {"val_loss": f"{val_loss / max(len(val_loader), 1):.4f}"}
                    )

            if verbose and hasattr(val_pbar, "close"):
                val_pbar.close()

        return val_loss / max(len(val_loader), 1)

    def _evaluate_model(
        self, model: nn.Module, dataloader, loss_type: str = "huber"
    ) -> float:
        model.eval()
        loss_fn = self._get_loss_function(loss_type)
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, *_ in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                preds = model(batch_x)
                total_loss += loss_fn(preds, batch_y).item()

        return total_loss / len(dataloader)

    def _make_candidate_config(
        self,
        rng,
        allowed_ops: List[str],
        hidden_dim_choices: List[int],
        cell_range: Tuple[int, int],
        node_range: Tuple[int, int],
        *,
        min_ops: int = 2,
        max_ops: Optional[int] = None,
        require_identity: bool = True,
    ) -> Dict[str, Any]:
        ops_pool = [op for op in allowed_ops if op != "Identity"]
        max_ops_local = min(
            max_ops or len(allowed_ops), len(ops_pool) + (1 if require_identity else 0)
        )
        min_ops_local = min(min_ops, max_ops_local)

        n_ops = rng.randint(min_ops_local, max_ops_local)
        picked = rng.sample(ops_pool, k=max(0, n_ops - (1 if require_identity else 0)))

        selected_ops = ["Identity"] + picked if require_identity else picked
        if not selected_ops:
            selected_ops = (
                ["Identity"] if require_identity else [rng.choice(allowed_ops)]
            )

        return {
            "selected_ops": selected_ops,
            "hidden_dim": rng.choice(hidden_dim_choices),
            "num_cells": rng.randint(cell_range[0], cell_range[1]),
            "num_nodes": rng.randint(node_range[0], node_range[1]),
        }

    def _build_candidate_model(self, cfg: Dict[str, Any]) -> nn.Module:
        return TimeSeriesDARTS(
            input_dim=self.input_dim,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["hidden_dim"],
            forecast_horizon=self.forecast_horizon,
            seq_length=self.seq_length,
            num_cells=cfg["num_cells"],
            num_nodes=cfg["num_nodes"],
            selected_ops=cfg["selected_ops"],
        ).to(self.device)

    def _compute_metrics(
        self, preds: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))
        mape = np.mean(np.abs((preds - targets) / (np.abs(targets) + 1e-8))) * 100
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

    def _create_bilevel_loaders(self, train_loader, seed: int = 42):
        dataset = train_loader.dataset
        train_size = int(0.7 * len(dataset))
        arch_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(int(seed))
        train_dataset, arch_dataset = torch.utils.data.random_split(
            dataset, [train_size, arch_size], generator=generator
        )
        train_model_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
        )
        train_arch_loader = torch.utils.data.DataLoader(
            arch_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
        )
        return train_arch_loader, train_model_loader

    def _make_default_search_candidate_config(self, rng=None) -> Dict[str, Any]:
        return make_default_search_candidate_config(
            trainer=self,
            rng=rng,
        )

    def _evaluate_search_candidate(
        self,
        candidate_id: int,
        val_loader,
        max_samples: int,
        *,
        num_batches: int = 1,
        include_timing: bool = False,
        rng=None,
    ) -> Dict[str, Any]:
        return evaluate_search_candidate(
            trainer=self,
            candidate_id=candidate_id,
            val_loader=val_loader,
            max_samples=max_samples,
            num_batches=num_batches,
            include_timing=include_timing,
            rng=rng,
        )

    @staticmethod
    def _select_top_candidates(candidates: List[Dict[str, Any]], top_k: int):
        return select_top_candidates(candidates, top_k)

    def _run_parallel_candidate_collection(
        self,
        num_candidates: int,
        candidate_fn,
        *,
        max_workers: Optional[int] = None,
        on_result=None,
        error_log_fn=None,
    ) -> List[Dict[str, Any]]:
        return run_parallel_candidate_collection(
            num_candidates=num_candidates,
            candidate_fn=candidate_fn,
            max_workers=max_workers,
            on_result=on_result,
            error_log_fn=error_log_fn,
        )

    def evaluate_zero_cost_metrics_raw(
        self,
        model: nn.Module,
        dataloader,
        max_samples: int = 32,
        num_batches: int = 1,
    ) -> Dict[str, Any]:
        """Compute zero-cost *raw* metrics once (no weighting)."""

        def create_custom_config(
            max_samples: int = 32, max_outputs: int = 10
        ) -> Config:
            return Config(max_samples=max_samples, max_outputs=max_outputs)

        cfg = create_custom_config(max_samples=max_samples, max_outputs=10)
        nas_evaluator = ZeroCostNAS(config=cfg)

        print(
            f"Computing zero-cost raw metrics (max_samples={cfg.max_samples}, max_outputs={cfg.max_outputs})..."
        )
        # 🔹 RAW call (correct)
        out = nas_evaluator.evaluate_model_raw_metrics(
            model=model,
            dataloader=dataloader,
            device=self.device,
            num_batches=num_batches,
        )

        print(
            f"✅ Zero-cost raw metrics computed: {len(out.get('raw_metrics', {}))} metrics"
        )
        # print the metric names and values
        for k, v in out.get("raw_metrics", {}).items():
            print(f"   - {k}: {v:.6f}")
        # 🔹 Return what actually exists
        return {
            "raw_metrics": out.get("raw_metrics", {}),
            "success_rates": out.get("success_rates", {}),
            "errors": out.get("errors", {}),
            "base_weights": dict(cfg.weights),
        }

    def evaluate_zero_cost_metrics(
        self,
        model: nn.Module,
        dataloader,
        max_samples: int = 32,
        num_batches: int = 1,
        ablation: bool = False,
        n_random: int = 20,
        random_sigma: float = 0.25,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate model using zero-cost metrics. If ablation=True, evaluate multiple weight schemes."""

        def create_custom_config(
            max_samples: int = 32,
            max_outputs: int = 10,
            timeout_seconds: float = 30.0,
            enable_mixed_precision: bool = False,
            weights: Optional[Dict[str, float]] = None,
        ) -> Config:
            cfg = Config(max_samples=max_samples, max_outputs=max_outputs)
            cfg.timeout = float(timeout_seconds)
            cfg.enable_mixed_precision = bool(enable_mixed_precision)
            if weights is not None:
                cfg.weights = weights
            return cfg

        # First run baseline once to get raw metrics (and baseline score)
        base_config = create_custom_config(
            max_samples=max_samples,
            max_outputs=10,
            timeout_seconds=30.0,
            enable_mixed_precision=True,
            weights=None,  # use default weights from Config
        )

        # print("Evaluating zero-cost metrics with baseline weights...")

        if not ablation:
            # print(
            #     "   Ablation disabled, running single evaluation with baseline weights."
            # )
            nas_evaluator = ZeroCostNAS(config=base_config)
            # print("   Running evaluation...")
            return nas_evaluator.evaluate_model(
                model, dataloader, self.device, num_batches=num_batches
            )

        # Ablation: run multiple schemes (same metrics computed each time, but simplest is re-run evaluate_model)
        schemes = build_weight_schemes(
            base_weights=dict(base_config.weights),
            n_random=n_random,
            random_sigma=random_sigma,
            seed=seed,
        )

        per_scheme = {}
        # Optional micro-optimization: compute metrics once and re-score multiple weight sets.
        # Your current ZeroCostNAS does not expose that, so we just rerun; acceptable for reviewer ablation on a subset.
        for scheme_name, w in schemes.items():
            cfg = create_custom_config(
                max_samples=max_samples,
                max_outputs=10,
                timeout_seconds=30.0,
                enable_mixed_precision=True,
                weights=w,
            )
            nas_evaluator = ZeroCostNAS(config=cfg)
            out = nas_evaluator.evaluate_model(
                model, dataloader, self.device, num_batches=num_batches
            )
            per_scheme[scheme_name] = {
                "aggregate_score": out["aggregate_score"],
                "metrics": out["metrics"],  # raw metrics
                "success_rates": out["success_rates"],
            }

        return {
            "ablation": True,
            "base_weights": dict(base_config.weights),
            "schemes": list(per_scheme.keys()),
            "per_scheme": per_scheme,
        }

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
        hybrid_pruning_start_epoch: int = 20,
        hybrid_pruning_interval: int = 10,
        hybrid_pruning_base_threshold: float = 0.15,
        hybrid_pruning_strategy: str = "performance",
        hybrid_pruning_freeze_logit: float = -20.0,
        use_bilevel_optimization: bool = True,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        verbose: bool = True,
        regularization_types: Optional[List[str]] = None,
        regularization_weights: Optional[List[float]] = None,
        temperature_schedule: str = "cosine",
        edge_sharpening_max_weight: float = 0.03,
        edge_sharpening_start_frac: float = 0.35,
        hessian_penalty_weight: float = 0.0,
        hessian_fd_eps: float = 1e-2,
        hessian_update_freq: int = 1,
        bilevel_split_seed: int = 42,
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
        (
            arch_params,
            model_params,
            edge_arch_params,
            component_arch_params,
        ) = self._split_architecture_and_model_params(model)

        if verbose:
            print(
                f"📊 Architecture params: {len(arch_params)}, Model params: {len(model_params)}"
            )
            print(
                f"   Edge arch params: {len(edge_arch_params)}, Component arch params: {len(component_arch_params)}"
            )

        arch_param_groups = self._build_arch_param_groups(
            edge_arch_params=edge_arch_params,
            component_arch_params=component_arch_params,
            arch_learning_rate=arch_learning_rate,
            arch_params=arch_params,
        )

        # Setup optimizers with fused operations if available
        try:
            arch_optimizer = torch.optim.Adam(
                arch_param_groups,
                betas=(0.5, 0.999),
                weight_decay=arch_weight_decay,
                fused=True,
            )
            model_optimizer = torch.optim.Adam(
                model_params,
                lr=model_learning_rate,
                weight_decay=model_weight_decay,
                fused=True,
            )
        except (TypeError, RuntimeError):
            arch_optimizer = torch.optim.Adam(
                arch_param_groups,
                betas=(0.5, 0.999),
                weight_decay=arch_weight_decay,
            )
            model_optimizer = torch.optim.AdamW(
                model_params, lr=model_learning_rate, weight_decay=model_weight_decay
            )

        # Learning rate schedulers
        arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            arch_optimizer, T_max=epochs, eta_min=arch_learning_rate * 0.01
        )
        model_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model_optimizer,
            max_lr=model_learning_rate,
            epochs=epochs,
            steps_per_epoch=max(
                1, len(train_loader) // max(1, gradient_accumulation_steps)
            ),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        # Loss function and data loaders
        loss_fn = self._get_loss_function(loss_type)
        train_arch_loader = None
        if use_bilevel_optimization:
            train_arch_loader, train_model_loader = self._create_bilevel_loaders(
                train_loader, seed=bilevel_split_seed
            )
        else:
            train_model_loader = train_loader
        bilevel_optimizer = BilevelOptimizer(
            arch_optimizer=arch_optimizer,
            arch_scheduler=arch_scheduler,
            arch_params=arch_params,
            edge_arch_params=edge_arch_params,
            component_arch_params=component_arch_params,
            use_bilevel_optimization=use_bilevel_optimization,
            train_arch_loader=train_arch_loader,
            val_loader=val_loader,
            train_model_loader=train_model_loader,
        )

        # SWA setup
        swa_model, swa_start = None, None
        if use_swa:
            swa_start = max(epochs // 2, warmup_epochs + 5)
            swa_model = torch.optim.swa_utils.AveragedModel(model).to(self.device)

        # Mixed precision
        scaler = GradScaler(enabled=use_amp and self.device.startswith("cuda"))

        # Training state
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        best_progressive_state = None
        train_losses, val_losses, alpha_values = [], [], []
        diversity_scores = []
        prev_component_probs = {}
        prev_edge_probs = {}
        last_edge_entropy = float("nan")
        last_edge_sharpen_weight = 0.0
        edge_diversity_weight = 0.02

        if verbose:
            print(f"🔍 Training DARTS for {epochs} epochs")
            print(f"   Arch LR: {arch_learning_rate}, Model LR: {model_learning_rate}")
            print(
                f"   Bilevel: {use_bilevel_optimization}, SWA: {use_swa}, AMP: {use_amp}"
            )
            print("-" * 60)

        # Main training loop
        epoch_pbar = (
            self._create_progress_bar(range(epochs), "DARTS", unit="epoch")
            if verbose
            else range(epochs)
        )
        # Setup regularization
        if regularization_types is None:
            regularization_types = ["kl_divergence", "efficiency"]
        if regularization_weights is None:
            regularization_weights = [0.05, 0.01]

        reg_types = [RegularizationType(rt) for rt in regularization_types]
        regularizer = ArchitectureRegularizer(reg_types, regularization_weights)

        # Setup temperature scheduler
        temp_scheduler = TemperatureScheduler(
            initial_temp=2.0,
            final_temp=0.1,
            schedule_type=temperature_schedule,
            warmup_epochs=warmup_epochs,
        )
        for epoch in epoch_pbar:
            model.train()

            # Dynamic temperature
            if hasattr(model, "schedule_temperature"):
                current_temperature = model.schedule_temperature(
                    epoch=epoch,
                    total_epochs=epochs,
                    schedule_type=temperature_schedule,
                    final_temp=temp_scheduler.final_temp,
                    warmup_epochs=warmup_epochs,
                )
            else:
                current_temperature = temp_scheduler.get_temperature(epoch, epochs)

            if hasattr(model, "set_temperature") and not hasattr(
                model, "schedule_temperature"
            ):
                model.set_temperature(current_temperature)

            if progressive_shrinking and hasattr(model, "schedule_progressive_stage"):
                model.schedule_progressive_stage(epoch=epoch, total_epochs=epochs)

            # Track alphas every 5 epochs
            if epoch % 5 == 0:
                alpha_values.append(self.alpha_tracker.extract_alpha_values(model))

            # Architecture updates
            if epoch >= warmup_epochs and epoch % architecture_update_freq == 0:
                for _ in range(2):  # 2 architecture steps
                    arch_batch = bilevel_optimizer.next_arch_batch()

                    arch_x, arch_y = (
                        arch_batch[0].to(self.device),
                        arch_batch[1].to(self.device),
                    )
                    bilevel_optimizer.zero_arch_grads()

                    with self._autocast(use_amp):
                        arch_preds = model(arch_x)
                        arch_loss = loss_fn(arch_preds, arch_y)

                        # Simple regularization
                        reg_losses = regularizer.compute_regularization(
                            model, arch_params, epoch, epochs
                        )

                        total_arch_loss = arch_loss + reg_losses["total"]

                        hessian_penalty = torch.tensor(0.0, device=self.device)
                        if (
                            hessian_penalty_weight > 0.0
                            and hessian_fd_eps > 0.0
                            and hessian_update_freq > 0
                            and (epoch - warmup_epochs) % hessian_update_freq == 0
                            and model_params
                        ):
                            h_batch = bilevel_optimizer.next_hessian_batch()

                            h_train_x, h_train_y = (
                                h_batch[0].to(self.device),
                                h_batch[1].to(self.device),
                            )
                            hessian_penalty = self._finite_difference_hessian_penalty(
                                model=model,
                                loss_fn=loss_fn,
                                arch_loss=arch_loss,
                                arch_x=arch_x,
                                arch_y=arch_y,
                                train_x=h_train_x,
                                train_y=h_train_y,
                                model_params=model_params,
                                eps=hessian_fd_eps,
                                use_amp=use_amp,
                            )
                            total_arch_loss = (
                                total_arch_loss
                                + hessian_penalty_weight * hessian_penalty
                            )

                        # Encourage different edges in a cell to specialize on
                        # different operations instead of collapsing to one op.
                        edge_diversity_loss = torch.tensor(0.0, device=self.device)
                        edge_diversity_pairs = 0
                        for cell in getattr(model, "cells", []):
                            if not hasattr(cell, "edges"):
                                continue

                            edge_prob_vectors = []
                            for edge in cell.edges:
                                probs = None
                                if (
                                    hasattr(edge, "use_hierarchical")
                                    and edge.use_hierarchical
                                    and hasattr(edge, "_get_weights")
                                    and hasattr(edge, "ops")
                                ):
                                    try:
                                        routed = edge._get_weights(top_k=None)
                                        if routed:
                                            probs = torch.zeros(
                                                len(edge.ops),
                                                device=routed[0][1].device,
                                                dtype=routed[0][1].dtype,
                                            )
                                            for op_idx, weight in routed:
                                                probs[op_idx] = probs[op_idx] + weight
                                    except Exception:
                                        probs = None
                                elif hasattr(edge, "_alphas"):
                                    temp = max(
                                        float(
                                            getattr(
                                                edge,
                                                "op_temperature",
                                                getattr(edge, "temperature", 1.0),
                                            )
                                        ),
                                        1e-6,
                                    )
                                    probs = F.softmax(edge._alphas / temp, dim=0)

                                if probs is not None and probs.numel() > 1:
                                    edge_prob_vectors.append(
                                        probs / probs.norm(p=2).clamp_min(1e-8)
                                    )

                            for i in range(len(edge_prob_vectors)):
                                for j in range(i + 1, len(edge_prob_vectors)):
                                    cos_ij = torch.dot(
                                        edge_prob_vectors[i], edge_prob_vectors[j]
                                    )
                                    edge_diversity_loss = edge_diversity_loss + cos_ij
                                    edge_diversity_pairs += 1

                        if edge_diversity_pairs > 0:
                            edge_diversity_loss = (
                                edge_diversity_loss / edge_diversity_pairs
                            )
                            total_arch_loss = (
                                total_arch_loss
                                + edge_diversity_weight * edge_diversity_loss
                            )

                        # Late-phase sharpening: reduce edge entropy so operation
                        # choices become more decisive near the end of search.
                        edge_entropy = torch.tensor(0.0, device=self.device)
                        edge_sharpen_weight = 0.0
                        if edge_sharpening_max_weight > 0 and epoch >= warmup_epochs:
                            progress = (epoch - warmup_epochs) / max(
                                1, epochs - warmup_epochs
                            )
                            if progress >= edge_sharpening_start_frac:
                                ramp = (progress - edge_sharpening_start_frac) / max(
                                    1e-8, 1.0 - edge_sharpening_start_frac
                                )
                                edge_sharpen_weight = edge_sharpening_max_weight * min(
                                    1.0, max(0.0, ramp)
                                )

                                entropy_terms = []
                                for cell in getattr(model, "cells", []):
                                    if not hasattr(cell, "edges"):
                                        continue
                                    for edge in cell.edges:
                                        if (
                                            hasattr(edge, "use_hierarchical")
                                            and edge.use_hierarchical
                                            and hasattr(edge, "group_alphas")
                                        ):
                                            group_temp = max(
                                                float(
                                                    getattr(
                                                        edge,
                                                        "group_temperature",
                                                        getattr(
                                                            edge, "temperature", 1.0
                                                        ),
                                                    )
                                                ),
                                                1e-6,
                                            )
                                            g_probs = F.softmax(
                                                edge.group_alphas / group_temp, dim=0
                                            )
                                            g_ent = -(
                                                g_probs * torch.log(g_probs + 1e-8)
                                            ).sum() / np.log(max(g_probs.numel(), 2))
                                            entropy_terms.append(g_ent)

                                            if hasattr(edge, "op_alphas"):
                                                op_temp = max(
                                                    float(
                                                        getattr(
                                                            edge,
                                                            "op_temperature",
                                                            getattr(
                                                                edge, "temperature", 1.0
                                                            ),
                                                        )
                                                    ),
                                                    1e-6,
                                                )
                                                for alpha in edge.op_alphas.values():
                                                    o_probs = F.softmax(
                                                        alpha / op_temp, dim=0
                                                    )
                                                    o_ent = -(
                                                        o_probs
                                                        * torch.log(o_probs + 1e-8)
                                                    ).sum() / np.log(
                                                        max(o_probs.numel(), 2)
                                                    )
                                                    entropy_terms.append(o_ent)
                                        elif hasattr(edge, "_alphas"):
                                            op_temp = max(
                                                float(
                                                    getattr(
                                                        edge,
                                                        "op_temperature",
                                                        getattr(
                                                            edge, "temperature", 1.0
                                                        ),
                                                    )
                                                ),
                                                1e-6,
                                            )
                                            probs = F.softmax(
                                                edge._alphas / op_temp, dim=0
                                            )
                                            ent = -(
                                                probs * torch.log(probs + 1e-8)
                                            ).sum() / np.log(max(probs.numel(), 2))
                                            entropy_terms.append(ent)

                                if entropy_terms:
                                    edge_entropy = torch.stack(entropy_terms).mean()
                                    total_arch_loss = (
                                        total_arch_loss
                                        + edge_sharpen_weight * edge_entropy
                                    )

                        last_edge_entropy = float(edge_entropy.detach().item())
                        last_edge_sharpen_weight = float(edge_sharpen_weight)

                    bilevel_optimizer.step_architecture(
                        total_arch_loss, scaler, already_backward=False
                    )
                    scaler.update()

                    if (
                        verbose
                        and hasattr(model, "forecast_encoder")
                        and hasattr(model, "forecast_decoder")
                    ):
                        enc_alphas = getattr(model.forecast_encoder, "alphas", None)
                        dec_alphas = getattr(model.forecast_decoder, "alphas", None)
                        enc_offsets = getattr(
                            model.forecast_encoder, "layer_alpha_offsets", None
                        )
                        dec_offsets = getattr(
                            model.forecast_decoder, "layer_alpha_offsets", None
                        )
                        att_alphas = getattr(
                            model.forecast_decoder, "attention_alphas", None
                        )
                        if enc_alphas is not None and dec_alphas is not None:
                            enc_gn = (
                                enc_alphas.grad.norm().item()
                                if enc_alphas.grad is not None
                                else float("nan")
                            )
                            dec_gn = (
                                dec_alphas.grad.norm().item()
                                if dec_alphas.grad is not None
                                else float("nan")
                            )
                            cos = F.cosine_similarity(
                                enc_alphas.detach().view(-1),
                                dec_alphas.detach().view(-1),
                                dim=0,
                            ).item()
                            enc_off_gn = (
                                enc_offsets.grad.norm().item()
                                if enc_offsets is not None
                                and enc_offsets.grad is not None
                                else float("nan")
                            )
                            dec_off_gn = (
                                dec_offsets.grad.norm().item()
                                if dec_offsets is not None
                                and dec_offsets.grad is not None
                                else float("nan")
                            )
                            att_gn = (
                                att_alphas.grad.norm().item()
                                if att_alphas is not None
                                and att_alphas.grad is not None
                                else float("nan")
                            )
                            same_obj = enc_alphas is dec_alphas
                            print(
                                f"   [Arch Grad] enc={enc_gn:.6e}, dec={dec_gn:.6e}, "
                                f"enc_off={enc_off_gn:.6e}, dec_off={dec_off_gn:.6e}, "
                                f"att={att_gn:.6e}, cos={cos:.4f}, shared={same_obj}"
                            )

                bilevel_optimizer.step_scheduler()

                if verbose:
                    with torch.no_grad():
                        self.alpha_tracker.log_architecture_update_block(
                            model=model,
                            prev_component_probs=prev_component_probs,
                            prev_edge_probs=prev_edge_probs,
                            last_edge_sharpen_weight=last_edge_sharpen_weight,
                            last_edge_entropy=last_edge_entropy,
                            hessian_penalty_weight=hessian_penalty_weight,
                            hessian_penalty=hessian_penalty,
                        )

                # print("Architecture gradients:")
                # for name, param in model.named_parameters():
                #     if any(param is p for p in arch_params):  # ✅ identity check
                #         if param.grad is not None:
                #             print(f"{name} grad norm: {param.grad.norm().item():.4e}")
                #         else:
                #             print(f"{name} grad: None")

            # Model parameter updates
            avg_train_loss = self._run_model_training_epoch(
                model=model,
                train_model_loader=train_model_loader,
                model_params=model_params,
                model_optimizer=model_optimizer,
                model_scheduler=model_scheduler,
                scaler=scaler,
                loss_fn=loss_fn,
                gradient_accumulation_steps=gradient_accumulation_steps,
                use_amp=use_amp,
                verbose=verbose,
                epoch=epoch,
            )

            avg_val_loss = self._run_validation_epoch(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                use_amp=use_amp,
                verbose=verbose,
            )
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Hybrid pruning schedule: soft threshold growth + hard freezing.
            should_prune = (
                progressive_shrinking
                and epoch > int(hybrid_pruning_start_epoch)
                and int(hybrid_pruning_interval) > 0
                and epoch % int(hybrid_pruning_interval) == 0
            )
            if should_prune and hasattr(model, "prune_weak_operations"):
                threshold = float(hybrid_pruning_base_threshold) * (
                    float(epoch) / float(max(epochs, 1))
                )
                threshold = min(max(threshold, 0.0), 0.95)
                pruning_stats = model.prune_weak_operations(
                    threshold=threshold,
                    strategy=hybrid_pruning_strategy,
                )
                frozen = 0
                if hasattr(model, "freeze_pruned_operations"):
                    frozen = int(
                        model.freeze_pruned_operations(
                            pruning_stats=pruning_stats,
                            logit_value=hybrid_pruning_freeze_logit,
                        )
                    )
                if verbose:
                    print(
                        f"   [Hybrid Prune] epoch={epoch + 1}/{epochs} "
                        f"threshold={threshold:.3f} "
                        f"pruned={int(pruning_stats.get('operations_pruned', 0))} "
                        f"frozen={frozen}"
                    )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {
                    k: v.detach().clone().float() for k, v in model.state_dict().items()
                }
                best_progressive_state = self._capture_progressive_state(model)

                if use_swa and swa_model and epoch >= swa_start:
                    swa_model.update_parameters(model)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            # Progress update
            if verbose and hasattr(epoch_pbar, "set_postfix"):
                epoch_pbar.set_postfix(
                    {
                        "train": f"{avg_train_loss:.4f}",
                        "val": f"{avg_val_loss:.4f}",
                        "best": f"{best_val_loss:.4f}",
                        "patience": f"{patience_counter}/{patience}",
                    }
                )

        if verbose and hasattr(epoch_pbar, "close"):
            epoch_pbar.close()

        training_time = time.time() - start_time

        # SWA finalization
        if use_swa and swa_model and epoch >= swa_start:
            if verbose:
                print("\n🔄 Finalizing SWA...")
            try:
                torch.optim.swa_utils.update_bn(
                    train_loader, swa_model, device=self.device
                )
                swa_val_loss = self._evaluate_model(swa_model, val_loader, loss_type)
                if swa_val_loss < best_val_loss:
                    if verbose:
                        print("✓ SWA model is better")
                    best_state = {
                        k: v.detach().clone() for k, v in swa_model.state_dict().items()
                    }
                    best_progressive_state = self._capture_progressive_state(model)
                    best_val_loss = swa_val_loss
            except Exception as e:
                if verbose:
                    print(f"Warning: SWA failed ({e})")

        # Load best model
        self._restore_progressive_state(model, best_progressive_state)
        try:
            model.load_state_dict(best_state)
        except RuntimeError as e:
            # Handle missing buffers and/or shape mismatch after search-space changes.
            current_state = model.state_dict()
            filtered_state = {}
            for k, v in best_state.items():
                if k.startswith("_forecast_buffer") or k.startswith("_context_buffer"):
                    continue
                if k in current_state and current_state[k].shape == v.shape:
                    filtered_state[k] = v

            if filtered_state:
                if verbose:
                    dropped = len(best_state) - len(filtered_state)
                    print(
                        f"Warning: partial checkpoint load due to state mismatch ({dropped} tensors skipped)."
                    )
                model.load_state_dict(filtered_state, strict=False)
            else:
                raise e

        # Ensure float32
        if hasattr(model, "ensure_float32_dtype"):
            model.ensure_float32_dtype()
        else:
            model = model.float()

        # Final results
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

    def _finite_difference_hessian_penalty(
        self,
        model: nn.Module,
        loss_fn,
        arch_loss: torch.Tensor,
        arch_x: torch.Tensor,
        arch_y: torch.Tensor,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        model_params: List[torch.Tensor],
        eps: float = 1e-2,
        use_amp: bool = True,
    ) -> torch.Tensor:
        """Finite-difference curvature proxy used to penalize sharp architecture landscapes."""
        if eps <= 0 or not model_params:
            return torch.tensor(0.0, device=arch_x.device)

        with self._autocast(use_amp):
            train_preds = model(train_x)
            train_loss = loss_fn(train_preds, train_y)

        grads = torch.autograd.grad(
            train_loss,
            model_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        direction = []
        norm_sq = torch.tensor(0.0, device=arch_x.device)
        for p, g in zip(model_params, grads):
            d = torch.zeros_like(p) if g is None else g.detach()
            direction.append(d)
            norm_sq = norm_sq + d.pow(2).sum()

        norm = torch.sqrt(norm_sq).clamp_min(1e-12)
        step = float(eps)
        scale = step / norm

        with torch.no_grad():
            originals = [p.detach().clone() for p in model_params]

        try:
            with torch.no_grad():
                for p, d in zip(model_params, direction):
                    p.add_(scale * d)

            with self._autocast(use_amp):
                loss_plus = loss_fn(model(arch_x), arch_y)

            with torch.no_grad():
                for p, d in zip(model_params, direction):
                    p.add_(-2.0 * scale * d)

            with self._autocast(use_amp):
                loss_minus = loss_fn(model(arch_x), arch_y)
        finally:
            with torch.no_grad():
                for p, orig in zip(model_params, originals):
                    p.copy_(orig)

        # Baseline term is used as a value anchor to avoid coupling this penalty
        # to any previously-consumed autograd graph in the caller.
        curvature = (loss_plus + loss_minus - 2.0 * arch_loss.detach()) / (step**2)
        return F.relu(curvature)

    def _compute_final_metrics(self, model: nn.Module, val_loader) -> Dict[str, float]:
        """Compute final metrics on validation set."""
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y, *_ in self._create_progress_bar(
                val_loader, "Computing metrics", leave=False
            ):
                batch_x, batch_y = (
                    batch_x.to(self.device).float(),
                    batch_y.to(self.device),
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
        return derive_fixed_architecture(
            model=model,
            as_probability_vector_fn=_as_probability_vector,
        )

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
                f"Epoch {epoch + 1:3d} Train",
                leave=False,
                total=num_train_batches,
            )

            for batch_idx, (batch_x, batch_y, *_) in enumerate(train_pbar):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()

                with self._autocast(True):
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
                        "avg_loss": f"{epoch_train_loss / (batch_idx + 1):.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
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
                    epoch_pbar.set_description(f"Early stopping at epoch {epoch + 1}")
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
                best_state = {
                    k.replace("module.", "", 1): v for k, v in best_state.items()
                }

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
                with self._autocast(True):
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
            f"{'Training duration:':<30} {results['training_time']:.1f} seconds  ({results['training_time'] / 60:.1f} minutes)"
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

    def ablation_weight_search(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        num_candidates: int = 20,
        max_samples: int = 32,
        num_batches: int = 1,
        top_k: int = 5,
        max_workers: Optional[int] = None,
        n_random: int = 50,
        random_sigma: float = 0.25,
        seed: int = 0,
        save_dir: str = ".",
        save_prefix: str = "zc_weight_ablation",
    ) -> Dict[str, Any]:
        """
        Run a lightweight ablation search that evaluates raw zero-cost metrics ONCE per candidate,
        then re-scores candidates under many weight schemes (baseline/uniform/subsets/LOO/random).

        Produces:
        - tables (pandas) and plots (matplotlib) saved to disk
        - returns all raw data for paper tables and reproducibility
        """
        import os

        import pandas as pd

        os.makedirs(save_dir, exist_ok=True)
        rng = np.random.default_rng(seed)

        print("Weight ablation search (zero-cost weighting).")
        print(
            f"  candidates: {num_candidates} | max_samples: {max_samples} | num_batches: {num_batches}"
        )
        print(
            f"  top_k: {top_k} | n_random: {n_random} | sigma: {random_sigma} | seed: {seed}"
        )
        print("-" * 70)

        # ---------------------------------------------------------------------
        # Phase 1: generate candidates + compute raw metrics ONCE (parallel)
        # ---------------------------------------------------------------------
        def _make_candidate_config() -> Dict[str, Any]:
            return self._make_candidate_config(
                random,
                self.all_ops,
                self.hidden_dims,
                (1, 2),
                (2, 4),
                min_ops=2,
                max_ops=len(self.all_ops),
                require_identity=True,
            )

        def _eval_one(candidate_id: int) -> Dict[str, Any]:
            print(f"Evaluating candidate {candidate_id + 1}/{num_candidates}...")
            # try:
            cfg = _make_candidate_config()
            model = self._build_candidate_model(cfg)

            print("  → Model built. Evaluating zero-cost metrics...")

            out = self.evaluate_zero_cost_metrics_raw(
                model=model,
                dataloader=val_loader,
                max_samples=max_samples,
                num_batches=num_batches,
            )

            print(f"  → Metrics computed: {len(out['raw_metrics'])} samples.")

            return {
                "candidate_id": candidate_id,
                "success": True,
                **cfg,
                # ✅ correct keys
                "raw_metrics": out["raw_metrics"],
                "success_rates": out.get("success_rates", {}),
                "errors": out.get("errors", {}),
                "base_weights": out.get("base_weights", {}),
            }

            # except Exception as e:
            #     return {
            #         "candidate_id": candidate_id,
            #         "success": False,
            #         "error": str(e),
            #     }

        candidates: List[Dict[str, Any]] = []
        lock = threading.Lock()
        done = 0

        def _cb(fut):
            nonlocal done
            r = fut.result()
            with lock:
                done += 1
                if r.get("success", False):
                    print(
                        f"  [{done:>3}/{num_candidates}] ok   id={r['candidate_id']} raw_metrics={len(r['raw_metrics'])}"
                    )
                else:
                    print(
                        f"  [{done:>3}/{num_candidates}] fail id={r.get('candidate_id', -1)}"
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_eval_one, i) for i in range(num_candidates)]
            for f in futs:
                f.add_done_callback(_cb)
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                if r.get("success", False):
                    candidates.append(r)

        if not candidates:
            raise RuntimeError("All candidates failed in raw zero-cost evaluation.")

        print("-" * 70)
        print(f"Raw eval done: {len(candidates)}/{num_candidates} successful.")

        # ---------------------------------------------------------------------
        # Phase 2: build schemes and re-score WITHOUT recomputing metrics
        # ---------------------------------------------------------------------
        base_weights = dict(candidates[0].get("base_weights", {}))
        schemes = build_weight_schemes(
            base_weights=base_weights,
            n_random=n_random,
            random_sigma=random_sigma,
            seed=seed,
        )
        scheme_names = list(schemes.keys())

        # score matrix: [N, S]
        N = len(candidates)
        S = len(scheme_names)
        score_mat = np.zeros((N, S), dtype=np.float64)

        for i, c in enumerate(candidates):
            m = c["raw_metrics"]
            for j, name in enumerate(scheme_names):
                score_mat[i, j] = score_from_metrics(m, schemes[name])

        # store scheme_scores in candidates for convenience
        for i, c in enumerate(candidates):
            c["scheme_scores"] = {
                scheme_names[j]: float(score_mat[i, j]) for j in range(S)
            }

        # ---------------------------------------------------------------------
        # Phase 3: build tables (stability, top candidates, LOO importance)
        # ---------------------------------------------------------------------
        baseline_idx = (
            scheme_names.index("baseline") if "baseline" in scheme_names else 0
        )
        baseline_scores = score_mat[:, baseline_idx]

        # Stability table vs baseline
        rows = []
        for j, name in enumerate(scheme_names):
            rows.append(
                {
                    "scheme": name,
                    "spearman_vs_baseline": _spearman_from_scores(
                        baseline_scores, score_mat[:, j]
                    ),
                    "topk_overlap_vs_baseline": _topk_overlap_from_scores(
                        baseline_scores, score_mat[:, j], top_k
                    ),
                }
            )
        df_stability = pd.DataFrame(rows).sort_values(
            by=["spearman_vs_baseline", "topk_overlap_vs_baseline"], ascending=False
        )

        # Candidate summary table (top by baseline)
        baseline_rank = _ranks_desc(baseline_scores)
        order = np.argsort(baseline_rank)
        top_show = min(10, N)

        cand_rows = []
        for idx in order[:top_show]:
            c = candidates[idx]
            row = {
                "candidate_id": c["candidate_id"],
                "baseline_rank": int(baseline_rank[idx]),
                "hidden_dim": c["hidden_dim"],
                "num_cells": c["num_cells"],
                "num_nodes": c["num_nodes"],
                "num_ops": len(c["selected_ops"]),
                "selected_ops": ", ".join(c["selected_ops"]),
            }
            # add a few key scheme scores
            row["baseline_score"] = float(score_mat[idx, baseline_idx])
            if "uniform" in scheme_names:
                row["uniform_score"] = float(
                    score_mat[idx, scheme_names.index("uniform")]
                )
            cand_rows.append(row)
        df_top_candidates = pd.DataFrame(cand_rows)

        # LOO importance (only schemes starting with loo_minus_)
        # We quantify importance by how much stability drops when removing metric.
        loo_rows = []
        loo_names = [n for n in scheme_names if n.startswith("loo_minus_")]
        for n in loo_names:
            j = scheme_names.index(n)
            loo_rows.append(
                {
                    "metric_removed": n.replace("loo_minus_", ""),
                    "spearman_vs_baseline": _spearman_from_scores(
                        baseline_scores, score_mat[:, j]
                    ),
                    "topk_overlap_vs_baseline": _topk_overlap_from_scores(
                        baseline_scores, score_mat[:, j], top_k
                    ),
                }
            )
        df_loo = pd.DataFrame(loo_rows)
        if len(df_loo) > 0:
            # importance = 1 - stability
            df_loo["importance_spearman_drop"] = 1.0 - df_loo["spearman_vs_baseline"]
            df_loo["importance_topk_drop"] = 1.0 - df_loo["topk_overlap_vs_baseline"]
            df_loo = df_loo.sort_values(
                by=["importance_spearman_drop", "importance_topk_drop"], ascending=False
            )

        # Random stability: baseline winner rank distribution
        rand_names = [n for n in scheme_names if n.startswith("rand_")]
        baseline_winner_idx = int(np.argmax(baseline_scores))
        winner_ranks = []
        for rn in rand_names:
            j = scheme_names.index(rn)
            ranks = _ranks_desc(score_mat[:, j])
            winner_ranks.append(int(ranks[baseline_winner_idx]))
        winner_ranks = (
            np.array(winner_ranks, dtype=np.int64)
            if len(winner_ranks)
            else np.array([], dtype=np.int64)
        )

        # Random top-k frequency per candidate
        topk_freq = np.zeros(N, dtype=np.int64)
        for rn in rand_names:
            j = scheme_names.index(rn)
            ranks = _ranks_desc(score_mat[:, j])
            topk_ids = np.where(ranks <= top_k)[0]
            topk_freq[topk_ids] += 1

        # ---------------------------------------------------------------------
        # Phase 4: plots (matplotlib only, no seaborn, no fixed colors)
        # ---------------------------------------------------------------------
        def _savefig(path: str):
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {path}")

        # Plot 1: heatmap of ranks (top candidates x schemes)
        # Use imshow (no fixed colormap specified; matplotlib default is fine)
        ranks_mat = np.zeros_like(score_mat, dtype=np.int64)
        for j in range(S):
            ranks_mat[:, j] = _ranks_desc(score_mat[:, j])

        # show top_show candidates (baseline order), and a manageable number of schemes
        scheme_keep = [
            "baseline",
            "uniform",
            "subset_grad",
            "subset_act",
            "subset_complexity",
            "subset_no_penalties",
            "subset_pos_only",
        ]
        scheme_keep = [s for s in scheme_keep if s in scheme_names]
        # add a few random schemes to visualize
        scheme_keep += rand_names[: min(6, len(rand_names))]
        scheme_keep = list(dict.fromkeys(scheme_keep))
        keep_idx = [scheme_names.index(s) for s in scheme_keep]

        sub_ranks = ranks_mat[order[:top_show]][:, keep_idx]

        plt.figure(figsize=(max(8, 0.8 * len(keep_idx)), max(4, 0.5 * top_show)))
        plt.imshow(sub_ranks, aspect="auto")
        plt.xticks(range(len(keep_idx)), scheme_keep, rotation=45, ha="right")
        plt.yticks(
            range(top_show),
            [int(candidates[i]["candidate_id"]) for i in order[:top_show]],
        )
        plt.xlabel("Weight scheme")
        plt.ylabel("Candidate id (baseline top)")
        plt.title("Candidate ranks across weight schemes (lower is better)")
        _savefig(os.path.join(save_dir, f"{save_prefix}_rank_heatmap.png"))

        # Plot 2: histogram of baseline winner rank under random weights
        if len(winner_ranks) > 0:
            plt.figure(figsize=(8, 4))
            plt.hist(
                winner_ranks,
                bins=min(20, max(5, int(np.sqrt(len(winner_ranks))))),
                edgecolor="black",
            )
            plt.xlabel("Rank of baseline winner under random weights (1=best)")
            plt.ylabel("Count")
            plt.title(
                "Baseline-winner rank distribution under random weight perturbations"
            )
            _savefig(os.path.join(save_dir, f"{save_prefix}_winner_rank_hist.png"))

        # Plot 3: top-k frequency under random schemes (bar, top 10)
        if len(rand_names) > 0:
            freq_order = np.argsort(-topk_freq)
            show = min(10, N)
            plt.figure(figsize=(10, 4))
            plt.bar(range(show), topk_freq[freq_order[:show]])
            plt.xticks(
                range(show),
                [int(candidates[i]["candidate_id"]) for i in freq_order[:show]],
                rotation=0,
            )
            plt.xlabel("Candidate id")
            plt.ylabel(
                f"Times in top-{top_k} (across {len(rand_names)} random schemes)"
            )
            plt.title(f"Top-{top_k} frequency under random weight perturbations")
            _savefig(os.path.join(save_dir, f"{save_prefix}_topk_freq.png"))

        # Plot 4: LOO importance (bar)
        if len(df_loo) > 0:
            show = min(12, len(df_loo))
            df_plot = df_loo.head(show)
            plt.figure(figsize=(10, 4))
            plt.bar(range(show), df_plot["importance_spearman_drop"].to_numpy())
            plt.xticks(
                range(show), df_plot["metric_removed"].tolist(), rotation=45, ha="right"
            )
            plt.xlabel("Removed metric")
            plt.ylabel("Importance (1 - Spearman vs baseline)")
            plt.title("Leave-one-out importance of each zero-cost metric")
            _savefig(os.path.join(save_dir, f"{save_prefix}_loo_importance.png"))

        # ---------------------------------------------------------------------
        # Phase 5: save tables to disk
        # ---------------------------------------------------------------------
        stability_path = os.path.join(save_dir, f"{save_prefix}_stability.csv")
        topcand_path = os.path.join(save_dir, f"{save_prefix}_top_candidates.csv")
        loo_path = os.path.join(save_dir, f"{save_prefix}_loo_importance.csv")

        df_stability.to_csv(stability_path, index=False)
        df_top_candidates.to_csv(topcand_path, index=False)
        if len(df_loo) > 0:
            df_loo.to_csv(loo_path, index=False)

        print("Tables saved:")
        print(f"  {stability_path}")
        print(f"  {topcand_path}")
        if len(df_loo) > 0:
            print(f"  {loo_path}")

        # ---------------------------------------------------------------------
        # Return everything needed for paper + reproducibility
        # ---------------------------------------------------------------------
        summary = {
            "candidates": candidates,
            "scheme_names": scheme_names,
            "schemes": schemes,  # weights per scheme
            "score_matrix": score_mat,
            "rank_matrix": ranks_mat,
            "tables": {
                "stability": df_stability,
                "top_candidates": df_top_candidates,
                "loo_importance": df_loo if len(df_loo) > 0 else None,
            },
            "random_analysis": {
                "baseline_winner_candidate_id": int(
                    candidates[baseline_winner_idx]["candidate_id"]
                ),
                "baseline_winner_rank_under_random": winner_ranks.tolist(),
                "topk_frequency_under_random": topk_freq.tolist(),
            },
            "artifacts": {
                "rank_heatmap": os.path.join(
                    save_dir, f"{save_prefix}_rank_heatmap.png"
                ),
                "winner_rank_hist": os.path.join(
                    save_dir, f"{save_prefix}_winner_rank_hist.png"
                ),
                "topk_freq": os.path.join(save_dir, f"{save_prefix}_topk_freq.png"),
                "loo_importance": os.path.join(
                    save_dir, f"{save_prefix}_loo_importance.png"
                ),
                "stability_csv": stability_path,
                "top_candidates_csv": topcand_path,
                "loo_csv": loo_path if len(df_loo) > 0 else None,
            },
            "config": {
                "num_candidates": num_candidates,
                "max_samples": max_samples,
                "num_batches": num_batches,
                "top_k": top_k,
                "n_random": n_random,
                "random_sigma": random_sigma,
                "seed": seed,
            },
        }
        return summary

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
        *,
        collect_stats: bool = False,
        parallelism_levels=None,
        est_overhead_per_task: float = 0.0,
        est_fixed_overhead_phase1: float = 0.0,
        est_fixed_overhead_phase3: float = 0.0,
        benchmark_phase1_workers=None,
        benchmark_phase1_candidates: int = None,
        stats_dir: str = "search_stats",
        run_name: str = None,
        logger=None,
        retrain_final_from_scratch: bool = True,
        discrete_arch_threshold: float = 0.3,
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
        return self._run_multi_fidelity_search(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_candidates=num_candidates,
            search_epochs=search_epochs,
            final_epochs=final_epochs,
            max_samples=max_samples,
            top_k=top_k,
            max_workers=max_workers,
            parallelism_levels=parallelism_levels,
            est_overhead_per_task=est_overhead_per_task,
            est_fixed_overhead_phase1=est_fixed_overhead_phase1,
            est_fixed_overhead_phase3=est_fixed_overhead_phase3,
            benchmark_phase1_workers=benchmark_phase1_workers,
            benchmark_phase1_candidates=benchmark_phase1_candidates,
            stats_dir=stats_dir,
            run_name=run_name,
            logger=logger,
            collect_stats=collect_stats,
            retrain_final_from_scratch=retrain_final_from_scratch,
            discrete_arch_threshold=discrete_arch_threshold,
        )

    def _run_multi_fidelity_search(
        self,
        train_loader,
        val_loader,
        test_loader,
        *,
        num_candidates: int = 10,
        search_epochs: int = 10,
        final_epochs: int = 100,
        max_samples: int = 32,
        top_k: int = 5,
        max_workers: int = None,
        # statistics / what-if parallelism
        parallelism_levels=None,  # e.g., [1,2,4,8,16]
        est_overhead_per_task: float = 0.0,
        est_fixed_overhead_phase1: float = 0.0,
        est_fixed_overhead_phase3: float = 0.0,
        # optional micro-benchmark of phase 1 at multiple workers (REAL timings)
        benchmark_phase1_workers=None,  # e.g., [1,2,4,8]
        benchmark_phase1_candidates: int = None,  # if None -> min(num_candidates, 20)
        # output
        stats_dir: str = "search_stats",
        run_name: str = None,
        logger=None,
        collect_stats: bool = True,
        retrain_final_from_scratch: bool = True,
        discrete_arch_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Single-function, instrumented multi-fidelity search:
        - records wall-time for Phase 1..5
        - records per-candidate dt for Phase 1 (zero-cost) and per-candidate dt for Phase 3 (search train + derive)
        - produces what-if wall-time estimates for Phase 1 and Phase 3 under different worker counts
        - optionally benchmarks Phase 1 with different worker counts (actually runs phase 1 multiple times)
        - saves JSON + CSV stats to stats_dir/run_id/

        IMPORTANT:
        - keeps everything inside this one method (no extra helper functions/classes).
        - uses ThreadPoolExecutor like your original for Phase 1.
        """
        import datetime
        import os
        import time

        import torch

        # -------------------------
        # Setup
        # -------------------------
        if logger is None:
            logger = getattr(self, "logger", None)
        if logger is None:
            import logging

            logger = logging.getLogger("NASLogger")

        if parallelism_levels is None:
            cpu = os.cpu_count() or 8
            parallelism_levels = sorted(set([1, 2, 4, 8, cpu]))
        else:
            parallelism_levels = list(parallelism_levels)

        if benchmark_phase1_workers is not None:
            benchmark_phase1_workers = list(benchmark_phase1_workers)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = run_name or f"multifidelity_{ts}"
        out_base = os.path.join(stats_dir, run_id) if collect_stats else None
        if out_base is not None:
            os.makedirs(out_base, exist_ok=True)

        sys_info = {
            "run_id": run_id,
            "timestamp_local": datetime.datetime.now().isoformat(),
            "cpu_count_os": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads()
            if hasattr(torch, "get_num_threads")
            else None,
            "torch_num_interop_threads": torch.get_num_interop_threads()
            if hasattr(torch, "get_num_interop_threads")
            else None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "parallelism_levels": list(map(int, parallelism_levels)),
            "max_workers_used": max_workers,
            "config": {
                "num_candidates": num_candidates,
                "search_epochs": search_epochs,
                "final_epochs": final_epochs,
                "max_samples": max_samples,
                "top_k": top_k,
            },
        }

        logger.info(
            "Starting multi-fidelity DARTS search"
            + (" (instrumented)" if collect_stats else "")
        )
        logger.info(
            f"candidates={num_candidates}, search_epochs={search_epochs}, final_epochs={final_epochs}, "
            f"top_k={top_k}, max_samples={max_samples}, max_workers={max_workers or 'auto'}"
        )

        # Accumulators
        phase_summary: Dict[str, Any] = {}
        per_candidate_rows = []  # CSV rows
        whatif_rows = []  # CSV rows
        bench_rows = []  # CSV rows (phase1 benchmark)

        # -------------------------
        # Optional: Phase 1 real benchmark across worker counts
        # -------------------------
        phase1_benchmark_results = []
        if collect_stats and benchmark_phase1_workers:
            bench_n = int(benchmark_phase1_candidates or min(num_candidates, 20))
            logger.info(
                f"Phase 1 benchmark enabled: workers={benchmark_phase1_workers}, candidates_per_run={bench_n}"
            )

            for w in benchmark_phase1_workers:
                # run minimal phase1 loop to measure
                t_wall0 = time.perf_counter()
                task_times = []

                def _bench_task(cid: int):
                    result = self._evaluate_search_candidate(
                        candidate_id=cid,
                        val_loader=val_loader,
                        max_samples=max_samples,
                        num_batches=1,
                        include_timing=True,
                    )
                    return {
                        "cid": cid,
                        "dt": float(result.get("phase1_dt", 0.0)),
                        "score": float(result["score"]),
                    }

                with concurrent.futures.ThreadPoolExecutor(max_workers=w) as ex:
                    futs = [ex.submit(_bench_task, i) for i in range(bench_n)]
                    for f in concurrent.futures.as_completed(futs):
                        r = f.result()
                        task_times.append(r["dt"])

                wall = time.perf_counter() - t_wall0
                m, s = mean_std(task_times)

                phase1_benchmark_results.append(
                    {
                        "workers": int(w),
                        "ncand": int(bench_n),
                        "wall_time_sec": float(wall),
                        "task_mean_sec": m,
                        "task_std_sec": s,
                    }
                )
                bench_rows.append([run_id, w, bench_n, wall, m, s])
                logger.info(
                    f"[Phase1 bench] workers={w}: wall={wall:.3f}s task_mean={m:.3f}s task_std={s:.3f}s"
                )

        # -------------------------
        # Phase 1: Generate & zero-cost evaluate candidates (parallel)
        # -------------------------
        logger.info("Phase 1: generating + zero-cost evaluating candidates (parallel)")
        phase1_task_times = []

        def generate_and_evaluate_candidate(cid: int) -> Dict[str, Any]:
            return self._evaluate_search_candidate(
                candidate_id=cid,
                val_loader=val_loader,
                max_samples=max_samples,
                num_batches=1,
                include_timing=True,
            )

        def _on_phase1_result(r, completed):
            if r.get("success", False):
                phase1_task_times.append(float(r.get("phase1_dt", 0.0)))
                logger.info(
                    f"[Phase 1] {completed}/{num_candidates} ID={r.get('candidate_id')} "
                    f"score={r.get('score', 0.0):.4f} ops={len(r.get('selected_ops', []))} "
                    f"hidden={r.get('hidden_dim', 'N/A')} dt={r.get('phase1_dt', 0.0):.3f}s"
                )
            else:
                logger.info(f"[Phase 1] {completed}/{num_candidates} failed")

        t_p1_0 = time.perf_counter()
        candidates = self._run_parallel_candidate_collection(
            num_candidates=num_candidates,
            candidate_fn=generate_and_evaluate_candidate,
            max_workers=max_workers,
            on_result=_on_phase1_result,
            error_log_fn=lambda e: logger.warning(f"[Phase 1] future error: {e}"),
        )
        t_p1 = time.perf_counter() - t_p1_0

        if not candidates and num_candidates > 0:
            logger.warning(
                "Phase 1 produced 0 successful candidates in parallel mode. "
                "Retrying sequentially for diagnostics."
            )
            seq_t0 = time.perf_counter()
            seq_candidates = []
            seq_errors = []

            for cid in range(num_candidates):
                try:
                    result = generate_and_evaluate_candidate(cid)
                    if result.get("success", False):
                        seq_candidates.append(result)
                        phase1_task_times.append(float(result.get("phase1_dt", 0.0)))
                        logger.info(
                            f"[Phase 1 fallback] ID={result.get('candidate_id')} "
                            f"score={result.get('score', 0.0):.4f} "
                            f"ops={len(result.get('selected_ops', []))} "
                            f"hidden={result.get('hidden_dim', 'N/A')}"
                        )
                    else:
                        logger.warning(
                            f"[Phase 1 fallback] ID={cid} returned success=False"
                        )
                except Exception as e:
                    seq_errors.append(f"candidate_id={cid}: {e}")
                    logger.warning(f"[Phase 1 fallback] candidate {cid} failed: {e}")

            t_p1 += time.perf_counter() - seq_t0
            candidates = seq_candidates

            if not candidates:
                error_preview = (
                    "; ".join(seq_errors[:3])
                    if seq_errors
                    else "no exceptions captured"
                )
                raise RuntimeError(
                    "Phase 1 produced zero successful candidates in both parallel and "
                    f"sequential evaluation. First errors: {error_preview}"
                )

        p1_mean, p1_std = mean_std(phase1_task_times)

        phase_summary["phase1"] = {
            "wall_time_sec": float(t_p1),
            "num_success": int(len(candidates)),
            "num_total": int(num_candidates),
            "task_mean_sec": p1_mean,
            "task_std_sec": p1_std,
            "task_min_sec": float(min(phase1_task_times)) if phase1_task_times else 0.0,
            "task_max_sec": float(max(phase1_task_times)) if phase1_task_times else 0.0,
        }

        logger.info(
            f"Phase 1 done: {len(candidates)}/{num_candidates} successful (wall={t_p1:.3f}s)"
        )

        # What-if estimates for Phase 1
        if collect_stats:
            phase1_whatif = append_whatif_estimates(
                phase="phase1",
                run_id=run_id,
                work_times=phase1_task_times,
                parallelism_levels=parallelism_levels,
                overhead_per_task=est_overhead_per_task,
                fixed_overhead=est_fixed_overhead_phase1,
                whatif_rows=whatif_rows,
            )
            phase_summary["phase1"]["whatif_estimates"] = phase1_whatif

        # -------------------------
        # Phase 2: Select top-k
        # -------------------------
        logger.info(f"Phase 2: selecting top {top_k} candidates")
        t_p2_0 = time.perf_counter()

        top_candidates = self._select_top_candidates(candidates, top_k)
        top_k_eff = len(top_candidates)

        t_p2 = time.perf_counter() - t_p2_0
        phase_summary["phase2"] = {
            "wall_time_sec": float(t_p2),
            "top_k_eff": int(top_k_eff),
        }

        for i, c in enumerate(top_candidates):
            logger.info(
                f"[Phase 2] {i + 1}: score={c['score']:.4f} ops={len(c.get('selected_ops', []))} "
                f"hidden={c.get('hidden_dim')} arch={c.get('num_cells')}x{c.get('num_nodes')}"
            )

        if top_k_eff == 0:
            raise RuntimeError(
                "Phase 2 selected zero candidates. Phase 1 had no successful candidate "
                "evaluations."
            )

        # -------------------------
        # Phase 3: Short DARTS training for top candidates (serial, but we log per-candidate dt)
        # -------------------------
        logger.info(
            f"Phase 3: training top {top_k_eff} candidates (search_epochs={search_epochs})"
        )
        t_p3_0 = time.perf_counter()

        trained_candidates = []
        trained_non_derived_candidates = []
        phase3_task_times = []  # per-candidate total for Phase 3
        phase3_search_times = []
        phase3_derive_eval_times = []

        for i, cand in enumerate(top_candidates):
            cid = cand.get("candidate_id", -1)
            logger.info(f"[Phase 3] training candidate {i + 1}/{top_k_eff} (ID={cid})")

            t_c0 = time.perf_counter()
            # Quick DARTS training
            t_s0 = time.perf_counter()
            search_results = self.train_darts_model(
                model=cand["model"],
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=search_epochs,
                use_swa=False,
            )
            t_search = time.perf_counter() - t_s0

            trained_non_derived_candidates.append(
                {
                    "model": copy.deepcopy(search_results["model"]),
                    "val_loss": search_results["best_val_loss"],
                    "candidate": cand,
                    "search_results": search_results,
                }
            )

            # Derive + eval
            t_d0 = time.perf_counter()
            derived_model = self.derive_final_architecture(search_results["model"])
            val_loss = self._evaluate_model(derived_model, val_loader)
            t_derive = time.perf_counter() - t_d0

            t_total = time.perf_counter() - t_c0

            phase3_task_times.append(float(t_total))
            phase3_search_times.append(float(t_search))
            phase3_derive_eval_times.append(float(t_derive))

            trained_candidates.append(
                {
                    "model": derived_model,
                    "val_loss": float(val_loss),
                    "candidate": cand,
                    "search_results": search_results,
                }
            )

            logger.info(
                f"[Phase 3] ID={cid} val_loss={val_loss:.6f} "
                f"dt_total={t_total:.3f}s (train={t_search:.3f}s derive+eval={t_derive:.3f}s)"
            )

            if collect_stats:
                # CSV per-candidate row
                per_candidate_rows.append(
                    [
                        run_id,
                        "phase1",
                        cid,
                        cand.get("score", 0.0),
                        cand.get("hidden_dim", None),
                        len(cand.get("selected_ops", [])),
                        cand.get("phase1_dt", 0.0),
                        "",  # phase3_total
                        "",  # phase3_train
                        "",  # phase3_derive
                        "",  # phase3_val_loss
                    ]
                )
                per_candidate_rows.append(
                    [
                        run_id,
                        "phase3",
                        cid,
                        cand.get("score", 0.0),
                        cand.get("hidden_dim", None),
                        len(cand.get("selected_ops", [])),
                        "",  # phase1_dt
                        t_total,
                        t_search,
                        t_derive,
                        float(val_loss),
                    ]
                )

        t_p3 = time.perf_counter() - t_p3_0

        p3_mean, p3_std = mean_std(phase3_task_times)
        p3_train_mean, p3_train_std = mean_std(phase3_search_times)
        p3_der_mean, p3_der_std = mean_std(phase3_derive_eval_times)

        phase_summary["phase3"] = {
            "wall_time_sec": float(t_p3),
            "task_mean_sec": p3_mean,
            "task_std_sec": p3_std,
            "train_mean_sec": p3_train_mean,
            "train_std_sec": p3_train_std,
            "derive_eval_mean_sec": p3_der_mean,
            "derive_eval_std_sec": p3_der_std,
        }

        # What-if estimates for Phase 3 (IF you parallelize top-k training in the future)
        if collect_stats:
            phase3_whatif = append_whatif_estimates(
                phase="phase3",
                run_id=run_id,
                work_times=phase3_task_times,
                parallelism_levels=parallelism_levels,
                overhead_per_task=est_overhead_per_task,
                fixed_overhead=est_fixed_overhead_phase3,
                whatif_rows=whatif_rows,
            )
            phase_summary["phase3"]["whatif_estimates"] = phase3_whatif

        # -------------------------
        # Phase 4: Select best candidate
        # -------------------------
        logger.info("Phase 4: selecting best candidate")
        t_p4_0 = time.perf_counter()

        if not trained_candidates:
            raise RuntimeError(
                "Phase 3 produced zero trained candidates; cannot select a best model."
            )

        best_candidate = min(trained_candidates, key=lambda x: x["val_loss"])
        t_p4 = time.perf_counter() - t_p4_0

        phase_summary["phase4"] = {"wall_time_sec": float(t_p4)}
        logger.info(
            f"[Phase 4] best val_loss={best_candidate['val_loss']:.6f} "
            f"ops={best_candidate['candidate'].get('selected_ops')} "
            f"arch={best_candidate['candidate'].get('num_cells')}x{best_candidate['candidate'].get('num_nodes')} "
            f"hidden={best_candidate['candidate'].get('hidden_dim')}"
        )

        # -------------------------
        # Phase 5: Train final model
        # -------------------------
        logger.info("Phase 5: training final model")
        t_p5_0 = time.perf_counter()

        final_model = copy.deepcopy(best_candidate["model"])
        final_conf = final_model.get_config()
        final_discrete_arch = {}
        if hasattr(final_model, "derive_discrete_architecture"):
            try:
                final_discrete_arch = final_model.derive_discrete_architecture(
                    threshold=discrete_arch_threshold
                )
            except Exception as e:
                logger.warning(f"[Phase 5] discrete architecture derivation failed: {e}")

        modules_reset = 0
        if retrain_final_from_scratch:
            modules_reset = self._reset_model_parameters(final_model)
            logger.info(
                f"[Phase 5] reinitialized {modules_reset} modules before final training"
            )

        final_results = self.train_final_model(
            model=final_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=final_epochs,
            learning_rate=5e-4,
            weight_decay=1e-5,
        )

        # plot training curve (keep your behavior)
        self._plot_training_curve(
            final_results["train_losses"],
            final_results["val_losses"],
            title="Final Model Training Progress",
            save_path=(
                os.path.join(out_base, "final_model_training.pdf")
                if out_base is not None
                else "final_model_training.pdf"
            ),
        )

        t_p5 = time.perf_counter() - t_p5_0
        phase_summary["phase5"] = {"wall_time_sec": float(t_p5)}

        # -------------------------
        # Build summary & save stats
        # -------------------------
        total_wall = sum(
            phase_summary[p]["wall_time_sec"]
            for p in ["phase1", "phase2", "phase3", "phase4", "phase5"]
        )
        phase_summary["total"] = {"wall_time_sec": float(total_wall)}

        # compact leaderboard info (for JSON)
        top_table = [
            {
                "rank": i + 1,
                "candidate_id": int(c.get("candidate_id", -1)),
                "score": float(c.get("score", 0.0)),
                "hidden_dim": c.get("hidden_dim", None),
                "num_ops": int(len(c.get("selected_ops", []))),
                "arch": f"{c.get('num_cells')}x{c.get('num_nodes')}",
                "phase1_dt": float(c.get("phase1_dt", 0.0)),
            }
            for i, c in enumerate(top_candidates)
        ]

        stats_payload = {
            "system": sys_info,
            "phase_summary": phase_summary,
            "phase1_benchmark_results": phase1_benchmark_results,
            "top_candidates": top_table,
            "best_candidate": {
                "candidate_id": int(
                    best_candidate["candidate"].get("candidate_id", -1)
                ),
                "val_loss": float(best_candidate["val_loss"]),
                "score": float(best_candidate["candidate"].get("score", 0.0)),
                "hidden_dim": best_candidate["candidate"].get("hidden_dim", None),
                "selected_ops": list(
                    best_candidate["candidate"].get("selected_ops", [])
                ),
                "arch": f"{best_candidate['candidate'].get('num_cells')}x{best_candidate['candidate'].get('num_nodes')}",
            },
        }

        if collect_stats and out_base is not None:
            save_json(os.path.join(out_base, "stats.json"), stats_payload)

            # CSVs
            save_csv(
                os.path.join(out_base, "per_candidate.csv"),
                header=[
                    "run_id",
                    "phase",
                    "candidate_id",
                    "score",
                    "hidden_dim",
                    "num_ops",
                    "phase1_dt_sec",
                    "phase3_total_dt_sec",
                    "phase3_train_dt_sec",
                    "phase3_derive_eval_dt_sec",
                    "phase3_val_loss",
                ],
                rows=per_candidate_rows,
            )

            save_csv(
                os.path.join(out_base, "whatif_parallelism.csv"),
                header=["run_id", "phase", "workers", "est_wall_time_sec"],
                rows=whatif_rows,
            )

            if bench_rows:
                save_csv(
                    os.path.join(out_base, "phase1_benchmark.csv"),
                    header=[
                        "run_id",
                        "workers",
                        "ncand",
                        "wall_time_sec",
                        "task_mean_sec",
                        "task_std_sec",
                    ],
                    rows=bench_rows,
                )

            logger.info(f"Stats saved to: {out_base}")

        logger.info(
            "Phase wall-times (s): "
            + ", ".join(
                [
                    f"{p}={phase_summary[p]['wall_time_sec']:.3f}"
                    for p in ["phase1", "phase2", "phase3", "phase4", "phase5"]
                ]
            )
            + f" | total={total_wall:.3f}"
        )

        # -------------------------
        # Build the same search_summary you already return
        # -------------------------
        search_summary = {
            "final_model": final_results["model"],
            "candidates": candidates,
            "top_candidates": top_candidates,
            "trained_candidates": trained_candidates,
            "best_candidate": best_candidate,
            "final_results": final_results,
            "final_config": final_conf,
            "final_discrete_architecture": final_discrete_arch,
            "search_config": {
                "num_candidates": num_candidates,
                "search_epochs": search_epochs,
                "final_epochs": final_epochs,
                "top_k": top_k_eff,
                "max_samples": max_samples,
                "max_workers": max_workers,
                "retrain_final_from_scratch": bool(retrain_final_from_scratch),
                "discrete_arch_threshold": float(discrete_arch_threshold),
            },
            "trained_non_derived_candidates": trained_non_derived_candidates,
            "final_reset_modules": int(modules_reset),
        }

        if collect_stats:
            search_summary["stats"] = stats_payload
            search_summary["stats_dir"] = out_base

        self.search_history.append(search_summary)
        self.final_model = final_results["model"]

        logger.info(
            "Multi-fidelity search completed"
            + (" (instrumented)" if collect_stats else "")
        )
        return search_summary

    def _multi_fidelity_search_instrumented(
        self,
        train_loader,
        val_loader,
        test_loader,
        *,
        num_candidates: int = 10,
        search_epochs: int = 10,
        final_epochs: int = 100,
        max_samples: int = 32,
        top_k: int = 5,
        max_workers: int = None,
        parallelism_levels=None,
        est_overhead_per_task: float = 0.0,
        est_fixed_overhead_phase1: float = 0.0,
        est_fixed_overhead_phase3: float = 0.0,
        benchmark_phase1_workers=None,
        benchmark_phase1_candidates: int = None,
        stats_dir: str = "search_stats",
        run_name: str = None,
        logger=None,
        collect_stats: bool = True,
        retrain_final_from_scratch: bool = True,
        discrete_arch_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for `_run_multi_fidelity_search`."""
        return self._run_multi_fidelity_search(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_candidates=num_candidates,
            search_epochs=search_epochs,
            final_epochs=final_epochs,
            max_samples=max_samples,
            top_k=top_k,
            max_workers=max_workers,
            parallelism_levels=parallelism_levels,
            est_overhead_per_task=est_overhead_per_task,
            est_fixed_overhead_phase1=est_fixed_overhead_phase1,
            est_fixed_overhead_phase3=est_fixed_overhead_phase3,
            benchmark_phase1_workers=benchmark_phase1_workers,
            benchmark_phase1_candidates=benchmark_phase1_candidates,
            stats_dir=stats_dir,
            run_name=run_name,
            logger=logger,
            collect_stats=collect_stats,
            retrain_final_from_scratch=retrain_final_from_scratch,
            discrete_arch_threshold=discrete_arch_threshold,
        )

    def multi_fidelity_search_with_stats(
        self,
        train_loader,
        val_loader,
        test_loader,
        *,
        num_candidates: int = 10,
        search_epochs: int = 10,
        final_epochs: int = 100,
        max_samples: int = 32,
        top_k: int = 5,
        max_workers: int = None,
        parallelism_levels=None,
        est_overhead_per_task: float = 0.0,
        est_fixed_overhead_phase1: float = 0.0,
        est_fixed_overhead_phase3: float = 0.0,
        benchmark_phase1_workers=None,
        benchmark_phase1_candidates: int = None,
        stats_dir: str = "search_stats",
        run_name: str = None,
        logger=None,
        retrain_final_from_scratch: bool = True,
        discrete_arch_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for instrumented multi-fidelity search."""
        return self.multi_fidelity_search(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_candidates=num_candidates,
            search_epochs=search_epochs,
            final_epochs=final_epochs,
            max_samples=max_samples,
            top_k=top_k,
            max_workers=max_workers,
            collect_stats=True,
            parallelism_levels=parallelism_levels,
            est_overhead_per_task=est_overhead_per_task,
            est_fixed_overhead_phase1=est_fixed_overhead_phase1,
            est_fixed_overhead_phase3=est_fixed_overhead_phase3,
            benchmark_phase1_workers=benchmark_phase1_workers,
            benchmark_phase1_candidates=benchmark_phase1_candidates,
            stats_dir=stats_dir,
            run_name=run_name,
            logger=logger,
            retrain_final_from_scratch=retrain_final_from_scratch,
            discrete_arch_threshold=discrete_arch_threshold,
        )

    def bilevel_lr_sensitivity(
        self,
        model_factory,  # callable -> fresh TimeSeriesDARTS (or your wrapper)
        train_loader,
        val_loader,
        *,
        model_lrs=(1e-4, 3e-4, 1e-3, 3e-3),
        arch_lrs=(3e-4, 1e-3, 3e-3, 1e-2),
        seeds=(0, 1, 2),
        epochs=30,
        save_csv_path=None,
    ):
        import pandas as pd

        results = []

        for mlr in model_lrs:
            for alr in arch_lrs:
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    model = model_factory().to(self.device)

                    out = self.train_darts_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=epochs,
                        model_learning_rate=mlr,
                        arch_learning_rate=alr,
                        use_bilevel_optimization=True,
                        verbose=False,
                    )

                    # derived arch val (more meaningful than mixed model val)
                    derived = self.derive_final_architecture(out["model"])
                    derived_val = self._evaluate_model(derived, val_loader)

                    # optional: last health snapshot (if you logged diversity_scores)
                    health_last = None
                    if out.get("diversity_scores"):
                        health_last = out["diversity_scores"][-1]

                    results.append(
                        {
                            "model_lr": mlr,
                            "arch_lr": alr,
                            "seed": seed,
                            "best_val_loss_mixed": float(out["best_val_loss"]),
                            "val_loss_derived": float(derived_val),
                            "train_time_s": float(out["training_time"]),
                            "health_score": None
                            if not health_last
                            else float(health_last["health_score"]),
                            "avg_identity_dominance": None
                            if not health_last
                            else float(health_last["avg_identity_dominance"]),
                        }
                    )

        df = pd.DataFrame(results)

        if save_csv_path:
            df.to_csv(save_csv_path, index=False)

        return df

    def robust_initial_pool_over_op_pools(
        self,
        *,
        val_loader,
        # outer robustness (op-pool perturbations)
        n_pools: int = 25,
        pool_size_range: Tuple[int, int] = (
            4,
            10,
        ),  # how many ops (incl Identity optionally)
        pool_seed: int = 0,
        # inner candidate sampling (per pool)
        num_candidates: int = 30,
        top_k: int = 10,
        max_samples: int = 32,
        num_batches: int = 1,
        seed: int = 0,
        max_workers: Optional[int] = None,
        # optional: still allow weight-scheme robustness inside each pool
        use_weight_schemes: bool = False,
        n_random: int = 50,
        random_sigma: float = 0.25,
        robustness_mode: str = "topk_freq",  # across pools
        topk_ref: Optional[int] = None,
        # candidate knobs
        min_ops: int = 2,
        max_ops: Optional[int] = None,
        cell_range: tuple = (1, 2),
        node_range: tuple = (2, 4),
        hidden_dim_choices: Optional[List[int]] = None,
        require_identity: bool = True,
    ) -> Dict[str, Any]:
        """
        Robustness w.r.t. the INITIAL selected_ops POOL (allowed ops set),
        not robustness w.r.t. scoring weights.

        It samples many operator pools, runs candidate generation constrained to each pool,
        and aggregates stability of selected architectures across pools.
        """

        if hidden_dim_choices is None:
            hidden_dim_choices = list(self.hidden_dims)
        if max_ops is None:
            max_ops = len(self.all_ops)
        if topk_ref is None:
            topk_ref = top_k

        # ---------------------------
        # helpers
        # ---------------------------
        def _ranks_desc(scores: np.ndarray) -> np.ndarray:
            order = np.argsort(-scores, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(scores) + 1)
            return ranks

        # sample operator pools
        py_rng_pools = random.Random(pool_seed)
        base_ops = list(self.all_ops)

        def _sample_pool() -> List[str]:
            # build a subset of ops allowed for candidate generation
            ops_no_id = [op for op in base_ops if op != "Identity"]
            lo, hi = pool_size_range
            # size excludes Identity; we can add it back below
            k = py_rng_pools.randint(
                max(1, lo - (1 if require_identity else 0)),
                max(1, hi - (1 if require_identity else 0)),
            )
            picked = py_rng_pools.sample(ops_no_id, k=min(k, len(ops_no_id)))
            if require_identity:
                return ["Identity"] + picked
            return picked

        op_pools = []
        seen = set()
        while len(op_pools) < n_pools:
            p = tuple(_sample_pool())
            if p not in seen:
                seen.add(p)
                op_pools.append(list(p))

        # ---------------------------
        # per-pool run (constrained candidate generation)
        # ---------------------------
        all_pool_results = []
        agg = {}  # signature -> stats accumulator

        for pool_idx, allowed_ops in enumerate(op_pools):
            py_rng = random.Random(seed + pool_idx)

            def _make_candidate_config() -> Dict[str, Any]:
                return self._make_candidate_config(
                    py_rng,
                    allowed_ops,
                    hidden_dim_choices,
                    cell_range,
                    node_range,
                    min_ops=min_ops,
                    max_ops=max_ops,
                    require_identity=require_identity,
                )

            def _eval_one(candidate_id: int) -> Dict[str, Any]:
                cfg = _make_candidate_config()
                try:
                    model = self._build_candidate_model(cfg)

                    out = self.evaluate_zero_cost_metrics_raw(
                        model=model,
                        dataloader=val_loader,
                        max_samples=max_samples,
                        num_batches=num_batches,
                    )

                    raw = out.get("raw_metrics", {}) or {}
                    if not raw:
                        return {"success": False, "error": "empty_raw_metrics"}

                    return {
                        "success": True,
                        **cfg,
                        "raw_metrics": raw,
                        "base_weights": dict(out.get("base_weights", {})),
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}

            # run inner evaluation
            candidates = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_eval_one, i) for i in range(num_candidates)]
                for f in concurrent.futures.as_completed(futs):
                    r = f.result()
                    if r.get("success", False):
                        candidates.append(r)

            if not candidates:
                all_pool_results.append(
                    {
                        "pool_idx": pool_idx,
                        "allowed_ops": allowed_ops,
                        "success": False,
                        "error": "all_candidates_failed",
                    }
                )
                continue

            # scoring: either baseline only (typical for op-pool sensitivity),
            # or still do scheme perturbations inside each pool.
            base_weights = dict(candidates[0].get("base_weights", {}))

            if use_weight_schemes:
                schemes = build_weight_schemes(
                    base_weights=base_weights,
                    n_random=n_random,
                    random_sigma=random_sigma,
                    seed=seed + pool_idx,
                )
                scheme_names = list(schemes.keys())
                baseline_name = "baseline" if "baseline" in schemes else scheme_names[0]
                # collapse to a single "robust within-pool score" by taking baseline score
                # (you can replace by your within-pool robust selection if you want)
                weights_for_pool = schemes[baseline_name]
            else:
                schemes = {"baseline": base_weights}
                scheme_names = ["baseline"]
                weights_for_pool = base_weights

            scores = np.array(
                [
                    score_from_metrics(c["raw_metrics"], weights_for_pool)
                    for c in candidates
                ],
                dtype=np.float64,
            )

            # choose top_k in this pool
            order = np.argsort(-scores, kind="mergesort")
            k = min(top_k, len(candidates))
            top_idx = order[:k].tolist()

            top = []
            for local_rank, i in enumerate(top_idx, start=1):
                c = candidates[i]
                entry = {
                    "pool_idx": pool_idx,
                    "allowed_ops": allowed_ops,
                    "signature": _sig_from_cfg(c),
                    "baseline_score": float(scores[i]),
                    "rank_in_pool": int(local_rank),
                    "selected_ops": list(c["selected_ops"]),
                    "hidden_dim": int(c["hidden_dim"]),
                    "num_cells": int(c["num_cells"]),
                    "num_nodes": int(c["num_nodes"]),
                }
                top.append(entry)

                sig = entry["signature"]
                st = agg.setdefault(
                    sig,
                    {
                        "count_in_topk": 0,
                        "ranks": [],
                        "scores": [],
                        "example": entry,
                    },
                )
                st["count_in_topk"] += 1
                st["ranks"].append(entry["rank_in_pool"])
                st["scores"].append(entry["baseline_score"])

            all_pool_results.append(
                {
                    "pool_idx": pool_idx,
                    "allowed_ops": allowed_ops,
                    "success": True,
                    "topk": top,
                    "scheme_names": scheme_names,
                }
            )

        # ---------------------------
        # aggregate robustness across pools
        # ---------------------------
        table = []
        for sig, st in agg.items():
            ranks = np.array(st["ranks"], dtype=np.float64)
            scores = np.array(st["scores"], dtype=np.float64)
            row = {
                "signature": sig,
                "topk_freq": int(st["count_in_topk"]),  # across pools
                "avg_rank": float(ranks.mean()) if len(ranks) else float("inf"),
                "worst_rank": int(ranks.max()) if len(ranks) else 10**9,
                "avg_score": float(scores.mean()) if len(scores) else float("-inf"),
                "score_std": float(scores.std(ddof=0)) if len(scores) else 0.0,
                "example_selected_ops": st["example"]["selected_ops"],
                "example_hidden_dim": st["example"]["hidden_dim"],
                "example_num_cells": st["example"]["num_cells"],
                "example_num_nodes": st["example"]["num_nodes"],
            }
            table.append(row)

        # select robust top_k across pools using same style as before
        if not table:
            raise RuntimeError("No successful pools produced any top-k entries.")

        if robustness_mode == "topk_freq":
            table.sort(key=lambda r: (r["topk_freq"], r["avg_score"]), reverse=True)
        elif robustness_mode == "avg_rank":
            table.sort(key=lambda r: (r["avg_rank"], -r["avg_score"]))
        elif robustness_mode == "worst_rank":
            table.sort(key=lambda r: (r["worst_rank"], -r["avg_score"]))
        else:
            raise ValueError(f"Unknown robustness_mode='{robustness_mode}'")

        selected = table[: min(top_k, len(table))]

        return {
            "selected": selected,  # robust across pools
            "robustness_table": table,  # full aggregation
            "pool_results": all_pool_results,  # per-pool topk + metadata
            "op_pools": op_pools,
            "config": {
                "n_pools": n_pools,
                "pool_size_range": pool_size_range,
                "pool_seed": pool_seed,
                "num_candidates": num_candidates,
                "top_k": top_k,
                "max_samples": max_samples,
                "num_batches": num_batches,
                "seed": seed,
                "use_weight_schemes": use_weight_schemes,
                "n_random": n_random,
                "random_sigma": random_sigma,
                "robustness_mode": robustness_mode,
                "topk_ref": topk_ref,
                "min_ops": min_ops,
                "max_ops": max_ops,
                "cell_range": cell_range,
                "node_range": node_range,
                "hidden_dim_choices": hidden_dim_choices,
                "require_identity": require_identity,
            },
        }

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

            summary.append(f"\nSearch {i + 1}:")
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
                summary.append(f"\nSession {i + 1}:")
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
            device_type = "cuda" if str(device).startswith("cuda") else "cpu"
            for i in range(0, N, batch_size):
                batch = X_val[i : i + batch_size]
                with autocast(
                    device_type=device_type,
                    dtype=torch.float16,
                    enabled=device_type == "cuda",
                ):
                    outputs = model(batch)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    outputs = outputs.float()

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
                        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

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
                    ax.plot(
                        forecast[:, i],
                        label=f"Forecast {names[i] if names else f'Feature {i}'}",
                    )
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
