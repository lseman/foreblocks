"""
Thin DARTSTrainer orchestrator.

All instance state (dimensions, device, alpha tracker) is kept here.
Heavy logic is delegated to focused sub-modules:

- :mod:`darts.training.darts_loop`   — bilevel DARTS training
- :mod:`darts.training.final_trainer` — final-model training
- :mod:`darts.search.zero_cost`      — zero-cost NAS metrics
- :mod:`darts.search.ablation`       — weight-scheme ablation
- :mod:`darts.search.multi_fidelity` — multi-fidelity pipeline
- :mod:`darts.search.robust_pool`    — op-pool robustness
- :mod:`darts.evaluation`            — metrics & plotting
- :mod:`darts.utils`                 — loss, training utilities, I/O
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ── Architecture ──────────────────────────────────────────────────────────
from .architecture import *
from .architecture.finalization import (
    derive_final_architecture as derive_fixed_architecture,
)
from .evaluation import plotting as _plot_mod
from .search import ablation as _abl_mod
from .search import multi_fidelity as _mf_mod
from .search import robust_pool as _rp_mod

# ── Sub-module delegates ──────────────────────────────────────────────────
from .search import zero_cost as _zc_mod

# ── Metrics & Config ──────────────────────────────────────────────────────
# ── Search utilities ─────────────────────────────────────────────────────
from .search.orchestrator import (
    evaluate_search_candidate,
    make_default_search_candidate_config,
    run_parallel_candidate_collection,
    select_top_candidates,
)
from .training import darts_loop as _dl_mod
from .training import final_trainer as _ft_mod

# ── Training helpers ─────────────────────────────────────────────────────
from .training.helpers import (
    AlphaTracker,
)
from .training.helpers import (
    default_as_probability_vector as _as_probability_vector,
)

_DEFAULT_OPS = [
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
    "Mamba",
    "InvertedAttention",
]


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
        hidden_dims: List[int] = None,
        forecast_horizon: int = 6,
        seq_length: int = 12,
        device: str = "auto",
        all_ops: Optional[List[str]] = None,
    ):
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.forecast_horizon = forecast_horizon
        self.seq_length = seq_length
        self.device = device
        self.use_gumbel = True
        self.alpha_tracker = AlphaTracker(
            as_probability_vector_fn=_as_probability_vector
        )
        self.all_ops = list(all_ops) if all_ops is not None else _DEFAULT_OPS

        # Runtime accumulation
        self.search_history: List[Dict[str, Any]] = []
        self.training_history: List[Dict[str, Any]] = []
        self.final_model: Optional[nn.Module] = None

        print(f"DARTSTrainer initialised on {device}")
        print(f"  input_dim={input_dim}  forecast_horizon={forecast_horizon}")
        print(f"  operations available: {len(self.all_ops)}")

    # ─────────────────────────────────────────────────────────────────────
    # Internal utilities (kept inline — tightly coupled to instance state)
    # ─────────────────────────────────────────────────────────────────────

    def _get_loss_function(self, loss_type: str = "huber"):
        registry = {
            "huber": lambda p, t: F.huber_loss(p, t, delta=0.1),
            "mse": F.mse_loss,
            "mae": F.l1_loss,
            "smooth_l1": F.smooth_l1_loss,
        }
        return registry.get(loss_type, registry["huber"])

    def _autocast(self, enabled: bool):
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return autocast(device_type=device_type, enabled=enabled)

    def _create_progress_bar(self, iterable, desc: str, leave: bool = True, **kwargs):
        kwargs.setdefault("unit", "batch")
        return tqdm(iterable, desc=desc, leave=leave, **kwargs)

    def _split_architecture_and_model_params(
        self, model: nn.Module
    ) -> Tuple[List, List, List, List]:
        arch_params: List[torch.Tensor] = []
        model_params: List[torch.Tensor] = []
        edge_arch_params: List[torch.Tensor] = []
        component_arch_params: List[torch.Tensor] = []
        arch_param_ids: set = set()

        for name, param in model.named_parameters():
            if any(t in name for t in ["alphas", "arch_", "alpha_", "norm_alpha"]):
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
        edge_arch_params: List,
        component_arch_params: List,
        arch_params: List,
        arch_learning_rate: float,
    ) -> List[Dict]:
        groups = []
        if edge_arch_params:
            groups.append({"params": edge_arch_params, "lr": arch_learning_rate * 1.5})
        if component_arch_params:
            groups.append({"params": component_arch_params, "lr": arch_learning_rate})
        if not groups:
            groups = [{"params": arch_params, "lr": arch_learning_rate}]
        return groups

    def _reset_model_parameters(self, model: nn.Module) -> int:
        count = 0
        for module in model.modules():
            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                    count += 1
                except Exception:
                    continue
        return count

    def _capture_progressive_state(self, model: nn.Module) -> Optional[Dict]:
        if not hasattr(model, "cells"):
            return None
        return {
            "cells": [
                {"progressive_stage": getattr(cell, "progressive_stage", None)}
                for cell in getattr(model, "cells", [])
            ]
        }

    def _restore_progressive_state(
        self, model: nn.Module, state: Optional[Dict]
    ) -> None:
        if state is None or not hasattr(model, "cells"):
            return
        cells = getattr(model, "cells", [])
        for idx, cell in enumerate(cells):
            if idx >= len(state.get("cells", [])):
                break
            stage = state["cells"][idx].get("progressive_stage")
            if stage is None:
                continue
            if hasattr(cell, "set_progressive_stage"):
                try:
                    cell.set_progressive_stage(stage)
                except Exception:
                    pass
            else:
                cell.progressive_stage = stage

    def _run_model_training_epoch(
        self,
        *,
        model: nn.Module,
        train_model_loader,
        model_params: List,
        model_optimizer,
        model_scheduler,
        scaler: GradScaler,
        loss_fn,
        gradient_accumulation_steps: int,
        use_amp: bool,
        verbose: bool,
        epoch: int,
    ) -> float:
        model.train()
        total_loss = 0.0
        batch_iter = (
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
        for i, (bx, by, *_) in batch_iter:
            bx, by = bx.to(self.device), by.to(self.device)
            with self._autocast(use_amp):
                preds = model(bx)
                loss = loss_fn(preds, by) / gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(model_optimizer)
                torch.nn.utils.clip_grad_norm_(model_params, max_norm=5.0)
                scaler.step(model_optimizer)
                scaler.update()
                model_scheduler.step()
                model_optimizer.zero_grad()
            total_loss += loss.item() * gradient_accumulation_steps
        return total_loss / max(len(train_model_loader), 1)

    def _run_validation_epoch(
        self,
        *,
        model: nn.Module,
        val_loader,
        loss_fn,
        use_amp: bool,
        verbose: bool,
    ) -> float:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_iter = (
                self._create_progress_bar(val_loader, "Val", leave=False)
                if verbose
                else val_loader
            )
            for batch in val_iter:
                bx, by = batch[0].to(self.device), batch[1].to(self.device)
                with self._autocast(use_amp):
                    preds = model(bx)
                    val_loss += loss_fn(preds, by).item()
        return val_loss / max(len(val_loader), 1)

    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader,
        loss_type: str = "huber",
    ) -> float:
        model.eval()
        loss_fn = self._get_loss_function(loss_type)
        total = 0.0
        with torch.no_grad():
            for bx, by, *_ in dataloader:
                bx, by = bx.to(self.device), by.to(self.device)
                total += loss_fn(model(bx), by).item()
        return total / max(len(dataloader), 1)

    def _compute_metrics(
        self, preds: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        mse = float(np.mean((preds - targets) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds - targets)))
        mape = float(
            np.mean(np.abs((preds - targets) / (np.abs(targets) + 1e-8))) * 100
        )
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))
        return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2_score": r2}

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
        edge_to_op_target: float = 1.0,
        edge_to_op_max_ratio: float = 1.8,
    ) -> Dict[str, Any]:
        ops_no_id = [op for op in allowed_ops if op != "Identity"]
        max_ops_local = min(
            max_ops or len(allowed_ops),
            len(ops_no_id) + (1 if require_identity else 0),
        )
        min_ops_local = min(min_ops, max_ops_local)
        n_ops = rng.randint(min_ops_local, max_ops_local)
        picked = rng.sample(ops_no_id, k=max(0, n_ops - (1 if require_identity else 0)))
        selected_ops = (["Identity"] + picked) if require_identity else picked
        if not selected_ops:
            selected_ops = (
                ["Identity"] if require_identity else [rng.choice(allowed_ops)]
            )

        node_candidates = list(range(int(node_range[0]), int(node_range[1]) + 1))
        op_budget = max(len(selected_ops) - (1 if require_identity else 0), 1)
        desired_edges = max(
            1, int(round(op_budget * max(float(edge_to_op_target), 0.2)))
        )
        max_edges = max(
            3, int(math.ceil(op_budget * max(float(edge_to_op_max_ratio), 1.0)))
        )
        feasible = [
            n for n in node_candidates if (n * (n - 1) // 2) <= max_edges
        ] or node_candidates
        weights = [
            max(
                math.exp(
                    -abs((n * (n - 1) // 2) - desired_edges) / max(desired_edges, 1)
                )
                / (1.0 + 0.08 * float(n * (n - 1) // 2)),
                1e-6,
            )
            for n in feasible
        ]
        try:
            num_nodes = rng.choices(feasible, weights=weights, k=1)[0]
        except Exception:
            num_nodes = rng.choice(feasible)

        return {
            "selected_ops": selected_ops,
            "hidden_dim": rng.choice(hidden_dim_choices),
            "num_cells": rng.randint(cell_range[0], cell_range[1]),
            "num_nodes": int(num_nodes),
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

    def _create_bilevel_loaders(self, train_loader, seed: int = 42):
        dataset = train_loader.dataset
        train_size = int(0.7 * len(dataset))
        arch_size = len(dataset) - train_size
        g = torch.Generator().manual_seed(int(seed))
        train_ds, arch_ds = torch.utils.data.random_split(
            dataset, [train_size, arch_size], generator=g
        )
        train_model_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=train_loader.batch_size, shuffle=True
        )
        train_arch_loader = torch.utils.data.DataLoader(
            arch_ds, batch_size=train_loader.batch_size, shuffle=True
        )
        return train_arch_loader, train_model_loader

    def _plot_training_curve(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training Progress",
        save_path: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt

        fig = _plot_mod.plot_training_curve(
            train_losses, val_losses, title=title, save_path=save_path
        )
        plt.close(fig)

    # ── Search orchestrator delegates ─────────────────────────────────────

    def _make_default_search_candidate_config(self, rng=None) -> Dict[str, Any]:
        return make_default_search_candidate_config(trainer=self, rng=rng)

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
    def _select_top_candidates(candidates: List[Dict], top_k: int) -> List[Dict]:
        return select_top_candidates(candidates, top_k)

    def _run_parallel_candidate_collection(
        self,
        num_candidates: int,
        candidate_fn,
        *,
        max_workers: Optional[int] = None,
        on_result=None,
        error_log_fn=None,
    ) -> List[Dict]:
        return run_parallel_candidate_collection(
            num_candidates=num_candidates,
            candidate_fn=candidate_fn,
            max_workers=max_workers,
            on_result=on_result,
            error_log_fn=error_log_fn,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API — delegates to focused sub-modules
    # ─────────────────────────────────────────────────────────────────────

    # ── Zero-cost metrics ─────────────────────────────────────────────────

    def evaluate_zero_cost_metrics_raw(
        self,
        model: nn.Module,
        dataloader,
        max_samples: int = 32,
        num_batches: int = 1,
        fast_mode: bool = True,
    ) -> Dict[str, Any]:
        """Compute zero-cost raw metrics (no weighting)."""
        return _zc_mod.evaluate_zero_cost_metrics_raw(
            self,
            model,
            dataloader,
            max_samples=max_samples,
            num_batches=num_batches,
            fast_mode=fast_mode,
        )

    def evaluate_zero_cost_metrics(
        self,
        model: nn.Module,
        dataloader,
        max_samples: int = 32,
        num_batches: int = 1,
        fast_mode: bool = True,
        ablation: bool = False,
        n_random: int = 10,
        random_sigma: float = 0.1,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Compute weighted zero-cost metrics with optional weight-scheme ablation."""
        return _zc_mod.evaluate_zero_cost_metrics(
            self,
            model,
            dataloader,
            max_samples=max_samples,
            num_batches=num_batches,
            fast_mode=fast_mode,
            ablation=ablation,
            n_random=n_random,
            random_sigma=random_sigma,
            seed=seed,
        )

    # ── Architecture derivation ───────────────────────────────────────────

    def derive_final_architecture(self, model: nn.Module) -> nn.Module:
        """Create an optimised model with fixed operations based on search results."""
        return derive_fixed_architecture(
            model=model,
            as_probability_vector_fn=_as_probability_vector,
        )

    # ── Core DARTS bilevel training ───────────────────────────────────────

    def train_darts_model(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        *,
        epochs: int = 50,
        arch_learning_rate: float = 3e-3,
        model_learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        arch_weight_decay: float = 1e-3,
        temperature: float = 1.0,
        use_swa: bool = False,
        use_bilevel_optimization: bool = True,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        loss_type: str = "huber",
        patience: int = 15,
        verbose: bool = True,
        use_progressive_training: bool = False,
        use_regularization: bool = True,
        hessian_penalty_weight: float = 0.0,
        edge_diversity_weight: float = 0.0,
        edge_sharpening_start_epoch: Optional[int] = None,
        edge_sharpening_strength: float = 0.01,
        identity_dominance_cap: float = 0.7,
        pruning_enabled: bool = False,
        pruning_start_epoch: Optional[int] = None,
        pruning_threshold: float = 0.05,
        pruning_hard_epoch: Optional[int] = None,
        log_arch_gradients: bool = False,
    ) -> Dict[str, Any]:
        """
        Run DARTS bilevel architecture search training.

        Returns dict with ``model``, ``best_val_loss``, ``train_losses``,
        ``val_losses``, ``alpha_values``, ``diversity_scores``,
        ``training_time``, ``search_stats``.
        """
        return _dl_mod.train_darts_model(
            self,
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            arch_learning_rate=arch_learning_rate,
            model_learning_rate=model_learning_rate,
            weight_decay=weight_decay,
            arch_weight_decay=arch_weight_decay,
            temperature=temperature,
            use_swa=use_swa,
            use_bilevel_optimization=use_bilevel_optimization,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
            loss_type=loss_type,
            patience=patience,
            verbose=verbose,
            use_progressive_training=use_progressive_training,
            use_regularization=use_regularization,
            hessian_penalty_weight=hessian_penalty_weight,
            edge_diversity_weight=edge_diversity_weight,
            edge_sharpening_start_epoch=edge_sharpening_start_epoch,
            edge_sharpening_strength=edge_sharpening_strength,
            identity_dominance_cap=identity_dominance_cap,
            pruning_enabled=pruning_enabled,
            pruning_start_epoch=pruning_start_epoch,
            pruning_threshold=pruning_threshold,
            pruning_hard_epoch=pruning_hard_epoch,
            log_arch_gradients=log_arch_gradients,
        )

    # ── Final model training ──────────────────────────────────────────────

    def train_final_model(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        *,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 20,
        loss_type: str = "huber",
        verbose: bool = True,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        use_swa: bool = False,
        swa_start: int = 75,
        swa_lr: float = 5e-4,
        compute_final_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the final fixed-architecture model.

        Returns dict with ``model``, ``best_val_loss``, ``train_losses``,
        ``val_losses``, ``final_metrics``, ``training_info``,
        ``training_time``.
        """
        return _ft_mod.train_final_model(
            self,
            model,
            train_loader,
            val_loader,
            test_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            loss_type=loss_type,
            verbose=verbose,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_swa=use_swa,
            swa_start=swa_start,
            swa_lr=swa_lr,
            compute_final_metrics=compute_final_metrics,
        )

    # ── Multi-fidelity search ─────────────────────────────────────────────

    def multi_fidelity_search(
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
        max_workers: Optional[int] = None,
        collect_stats: bool = False,
        parallelism_levels=None,
        stats_dir: str = "search_stats",
        run_name: Optional[str] = None,
        retrain_final_from_scratch: bool = True,
        discrete_arch_threshold: float = 0.3,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the complete multi-fidelity DARTS search pipeline."""
        return _mf_mod.run_multi_fidelity_search(
            self,
            train_loader,
            val_loader,
            test_loader,
            num_candidates=num_candidates,
            search_epochs=search_epochs,
            final_epochs=final_epochs,
            max_samples=max_samples,
            top_k=top_k,
            max_workers=max_workers,
            collect_stats=collect_stats,
            parallelism_levels=parallelism_levels,
            stats_dir=stats_dir,
            run_name=run_name,
            retrain_final_from_scratch=retrain_final_from_scratch,
            discrete_arch_threshold=discrete_arch_threshold,
            **kwargs,
        )

    # Alias kept for backward compatibility
    _run_multi_fidelity_search = multi_fidelity_search
    _multi_fidelity_search_instrumented = multi_fidelity_search

    def multi_fidelity_search_with_stats(
        self,
        train_loader,
        val_loader,
        test_loader,
        **kwargs,
    ) -> Dict[str, Any]:
        """Wrapper that enables ``collect_stats=True`` by default."""
        kwargs.setdefault("collect_stats", True)
        return self.multi_fidelity_search(
            train_loader, val_loader, test_loader, **kwargs
        )

    # ── bilevel LR sensitivity ────────────────────────────────────────────

    def bilevel_lr_sensitivity(
        self,
        model_factory,
        train_loader,
        val_loader,
        *,
        model_lrs=(1e-4, 3e-4, 1e-3, 3e-3),
        arch_lrs=(3e-4, 1e-3, 3e-3, 1e-2),
        seeds=(0, 1, 2),
        epochs: int = 30,
        save_csv_path: Optional[str] = None,
    ):
        """Grid-search over (model_lr, arch_lr, seed) with bilevel optimisation."""
        return _mf_mod.bilevel_lr_sensitivity(
            self,
            model_factory,
            train_loader,
            val_loader,
            model_lrs=model_lrs,
            arch_lrs=arch_lrs,
            seeds=seeds,
            epochs=epochs,
            save_csv_path=save_csv_path,
        )

    # ── Weight-scheme ablation ────────────────────────────────────────────

    def ablation_weight_search(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        *,
        num_candidates: int = 30,
        max_samples: int = 32,
        num_batches: int = 1,
        n_random: int = 50,
        random_sigma: float = 0.25,
        top_k: int = 5,
        seed: int = 0,
        max_workers: Optional[int] = None,
        save_plots: bool = True,
        save_csv: bool = True,
        output_dir: str = "ablation_results",
        run_id: Optional[str] = None,
        cell_range: tuple = (1, 2),
        node_range: tuple = (2, 4),
        min_ops: int = 2,
        max_ops: Optional[int] = None,
        require_identity: bool = True,
    ) -> Dict[str, Any]:
        """Run weight-scheme ablation over zero-cost metrics."""
        return _abl_mod.ablation_weight_search(
            self,
            train_loader,
            val_loader,
            test_loader,
            num_candidates=num_candidates,
            max_samples=max_samples,
            num_batches=num_batches,
            n_random=n_random,
            random_sigma=random_sigma,
            top_k=top_k,
            seed=seed,
            max_workers=max_workers,
            save_plots=save_plots,
            save_csv=save_csv,
            output_dir=output_dir,
            run_id=run_id,
            cell_range=cell_range,
            node_range=node_range,
            min_ops=min_ops,
            max_ops=max_ops,
            require_identity=require_identity,
        )

    # ── Op-pool robustness ────────────────────────────────────────────────

    def robust_initial_pool_over_op_pools(
        self,
        *,
        val_loader,
        n_pools: int = 25,
        pool_size_range: tuple = (4, 10),
        pool_seed: int = 0,
        num_candidates: int = 30,
        top_k: int = 10,
        max_samples: int = 32,
        num_batches: int = 1,
        seed: int = 0,
        max_workers: Optional[int] = None,
        use_weight_schemes: bool = False,
        n_random: int = 50,
        random_sigma: float = 0.25,
        robustness_mode: str = "topk_freq",
        topk_ref: Optional[int] = None,
        min_ops: int = 2,
        max_ops: Optional[int] = None,
        cell_range: tuple = (1, 2),
        node_range: tuple = (2, 4),
        hidden_dim_choices: Optional[List[int]] = None,
        require_identity: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate robustness of architecture selection w.r.t. operator pool."""
        return _rp_mod.robust_initial_pool_over_op_pools(
            self,
            val_loader=val_loader,
            n_pools=n_pools,
            pool_size_range=pool_size_range,
            pool_seed=pool_seed,
            num_candidates=num_candidates,
            top_k=top_k,
            max_samples=max_samples,
            num_batches=num_batches,
            seed=seed,
            max_workers=max_workers,
            use_weight_schemes=use_weight_schemes,
            n_random=n_random,
            random_sigma=random_sigma,
            robustness_mode=robustness_mode,
            topk_ref=topk_ref,
            min_ops=min_ops,
            max_ops=max_ops,
            cell_range=cell_range,
            node_range=node_range,
            hidden_dim_choices=hidden_dim_choices,
            require_identity=require_identity,
        )

    # ── Summary / IO ──────────────────────────────────────────────────────

    def get_search_summary(self) -> str:
        if not self.search_history:
            return "No searches performed yet."
        lines = ["DARTS SEARCH SUMMARY", "=" * 50]
        for i, s in enumerate(self.search_history):
            fm = s["final_results"]["final_metrics"]
            cfg = s["search_config"]
            lines += [
                f"\nSearch {i + 1}:",
                f"  Candidates evaluated: {cfg['num_candidates']}",
                f"  Final test RMSE:      {fm['rmse']:.6f}",
                f"  Final R2 score:       {fm['r2_score']:.4f}",
                f"  Training time:        {s['final_results']['training_time']:.1f}s",
            ]
        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    def get_training_summary(self) -> str:
        if not self.training_history:
            return "No training sessions completed yet."
        lines = ["TRAINING SUMMARY", "=" * 40]
        for i, t in enumerate(self.training_history):
            if "final_metrics" in t:
                m = t["final_metrics"]
                lines += [
                    f"\nSession {i + 1}:",
                    f"  Best val loss: {t.get('best_val_loss', 'N/A')}",
                    f"  RMSE:          {m.get('rmse', 'N/A')}",
                    f"  R2 score:      {m.get('r2_score', 'N/A')}",
                    f"  Training time: {t.get('training_time', 'N/A')}s",
                ]
        lines.append("\n" + "=" * 40)
        return "\n".join(lines)

    def save_best_model(self, filepath: str = "best_darts_model.pth") -> None:
        if not self.search_history:
            print("No search history available to save.")
            return
        best = min(
            self.search_history,
            key=lambda x: x["final_results"]["final_metrics"]["rmse"],
        )
        torch.save(
            {
                "model_state_dict": best["final_model"].state_dict(),
                "final_metrics": best["final_results"]["final_metrics"],
                "training_info": best["final_results"]["training_info"],
                "search_config": best["search_config"],
            },
            filepath,
        )
        print(f"Best model saved to {filepath}")
        print(f"  RMSE:     {best['final_results']['final_metrics']['rmse']:.6f}")
        print(f"  R2 Score: {best['final_results']['final_metrics']['r2_score']:.4f}")

    def load_model(self, filepath: str, model_class=None):
        """Load a saved checkpoint (returns raw dict)."""
        checkpoint = torch.load(filepath, map_location=self.device)
        print(f"Loading model from {filepath}")
        print(f"  Saved RMSE:     {checkpoint['final_metrics']['rmse']:.6f}")
        print(f"  Saved R2 Score: {checkpoint['final_metrics']['r2_score']:.4f}")
        return checkpoint

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot_alpha_evolution(
        self,
        alpha_values: List,
        save_path: str = "alpha_evolution.png",
    ) -> None:
        """Plot the evolution of architecture parameters during search."""
        import matplotlib.pyplot as plt

        fig = _plot_mod.plot_alpha_evolution(alpha_values, save_path=save_path)
        if fig is not None:
            plt.close(fig)

    def _batched_forecast(
        self, X_val: torch.Tensor, batch_size: int = 256
    ) -> torch.Tensor:
        assert self.final_model is not None, "No final model set. Run search first."
        return _plot_mod.batched_forecast(
            self.final_model, X_val, batch_size=batch_size
        )

    def plot_prediction(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        *,
        full_series=None,
        offset: int = 0,
        figsize=(14, 8),
        show: bool = True,
        device: Optional[str] = None,
        names=None,
        batch_size: int = 256,
    ):
        assert self.final_model is not None, "No final model set. Run search first."
        return _plot_mod.plot_prediction(
            self.final_model,
            X_val,
            y_val,
            full_series=full_series,
            offset=offset,
            figsize=figsize,
            show=show,
            device=device or self.device,
            names=names,
            batch_size=batch_size,
        )
