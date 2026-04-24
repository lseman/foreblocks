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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .architecture.core_blocks import TimeSeriesDARTS

# ── Architecture ──────────────────────────────────────────────────────────
from .architecture.finalization import (
    derive_final_architecture as derive_fixed_architecture,
)
from .config import (
    DEFAULT_ARCH_MODES,
    DEFAULT_ATTENTION_VARIANTS,
    DEFAULT_FFN_VARIANTS,
    DEFAULT_OP_FAMILIES,
    DEFAULT_OPS as SEARCH_DEFAULT_OPS,
)
from .evaluation import plotting as _plot_mod

# ── Sub-module delegates ──────────────────────────────────────────────────
from .search import (
    ablation as _abl_mod,
    multi_fidelity as _mf_mod,
    robust_pool as _rp_mod,
    zero_cost as _zc_mod,
)

# ── Metrics & Config ──────────────────────────────────────────────────────
# ── Search utilities ─────────────────────────────────────────────────────
from .search.orchestrator import (
    evaluate_search_candidate,
    make_default_search_candidate_config,
    run_parallel_candidate_collection,
    select_top_candidates,
)
from .training import darts_loop as _dl_mod, final_trainer as _ft_mod

# ── Training helpers ─────────────────────────────────────────────────────
from .training.helpers import (
    AlphaTracker,
    default_as_probability_vector as _as_probability_vector,
)
from .utils.training import unpack_forecasting_batch


_DEFAULT_OPS = list(SEARCH_DEFAULT_OPS)


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
        hidden_dims: list[int] = None,
        forecast_horizon: int = 6,
        seq_length: int = 12,
        device: str = "auto",
        all_ops: list[str] | None = None,
        arch_modes: list[str] | None = None,
        op_families: dict[str, list[str]] | None = None,
        family_range: tuple[int, int] = (1, 3),
        attention_variants: list[str] | None = None,
        ffn_variants: list[str] | None = None,
    ):
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        _all_arch_modes = list(DEFAULT_ARCH_MODES)
        if arch_modes is None:
            arch_modes = _all_arch_modes
        else:
            invalid = [m for m in arch_modes if m not in _all_arch_modes]
            if invalid:
                raise ValueError(
                    f"Invalid arch_modes {invalid}. Valid options: {_all_arch_modes}"
                )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.forecast_horizon = forecast_horizon
        self.seq_length = seq_length
        self.device = device
        self.arch_modes = list(arch_modes)
        self.use_gumbel = True
        self.alpha_tracker = AlphaTracker(
            as_probability_vector_fn=_as_probability_vector
        )
        self.all_ops = list(all_ops) if all_ops is not None else _DEFAULT_OPS
        self.op_families = self._normalize_op_families(op_families)
        min_family = max(1, int(family_range[0]))
        max_family = max(min_family, int(family_range[1]))
        self.family_range = (min_family, max_family)
        raw_attention_variants = (
            list(attention_variants)
            if attention_variants is not None
            else list(DEFAULT_ATTENTION_VARIANTS)
        )
        self.attention_variants = [
            str(v).lower()
            for v in raw_attention_variants
            if str(v).lower()
            in {"auto", "sdp", "linear", "probsparse", "cosine", "local"}
        ] or list(DEFAULT_ATTENTION_VARIANTS)
        raw_ffn_variants = (
            list(ffn_variants)
            if ffn_variants is not None
            else list(DEFAULT_FFN_VARIANTS)
        )
        self.ffn_variants = [
            str(v).lower()
            for v in raw_ffn_variants
            if str(v).lower() in {"auto", "swiglu", "moe"}
        ] or list(DEFAULT_FFN_VARIANTS)

        # Runtime accumulation
        self.search_history: list[dict[str, Any]] = []
        self.training_history: list[dict[str, Any]] = []
        self.final_model: nn.Module | None = None

        print(f"DARTSTrainer initialised on {device}")
        print(f"  input_dim={input_dim}  forecast_horizon={forecast_horizon}")
        print(f"  operations available: {len(self.all_ops)}")
        print(f"  op families: {list(self.op_families.keys())}")
        print(f"  arch_modes: {self.arch_modes}")
        print(f"  attention_variants: {self.attention_variants}")
        print(f"  ffn_variants: {self.ffn_variants}")

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
    ) -> tuple[list, list, list, list]:
        arch_params: list[torch.Tensor] = []
        model_params: list[torch.Tensor] = []
        edge_arch_params: list[torch.Tensor] = []
        component_arch_params: list[torch.Tensor] = []
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
        edge_arch_params: list,
        component_arch_params: list,
        arch_params: list,
        arch_learning_rate: float,
    ) -> list[dict]:
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

    def _capture_progressive_state(self, model: nn.Module) -> dict | None:
        if not hasattr(model, "cells"):
            return None
        return {
            "cells": [
                {"progressive_stage": getattr(cell, "progressive_stage", None)}
                for cell in getattr(model, "cells", [])
            ]
        }

    def _restore_progressive_state(self, model: nn.Module, state: dict | None) -> None:
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
        model_params: list,
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
        for i, batch in batch_iter:
            bx, by, model_kwargs = unpack_forecasting_batch(
                batch,
                self.device,
                include_decoder_targets=True,
            )
            with self._autocast(use_amp):
                preds = model(bx, **model_kwargs)
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
                bx, by, model_kwargs = unpack_forecasting_batch(
                    batch,
                    self.device,
                    include_decoder_targets=False,
                    teacher_forcing_ratio=0.0,
                )
                with self._autocast(use_amp):
                    preds = model(bx, **model_kwargs)
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
            for batch in dataloader:
                bx, by, model_kwargs = unpack_forecasting_batch(
                    batch,
                    self.device,
                    include_decoder_targets=False,
                    teacher_forcing_ratio=0.0,
                )
                total += loss_fn(model(bx, **model_kwargs), by).item()
        return total / max(len(dataloader), 1)

    def _compute_metrics(
        self, preds: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
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

    def _normalize_op_families(
        self, op_families: dict[str, list[str]] | None
    ) -> dict[str, list[str]]:
        family_source = op_families or DEFAULT_OP_FAMILIES
        allowed_ops = set(self.all_ops)
        normalized: dict[str, list[str]] = {}
        for family_name, ops in family_source.items():
            filtered = [op for op in ops if op in allowed_ops]
            if filtered or str(family_name).lower() == "ssm":
                normalized[str(family_name).lower()] = filtered
        if not normalized:
            normalized = {"mlp": [op for op in self.all_ops if op != "Identity"]}
        return normalized

    def _resolve_candidate_families(
        self, rng, allowed_ops: list[str]
    ) -> tuple[dict[str, list[str]], list[str], list[str]]:
        allowed_set = set(allowed_ops)
        family_space: dict[str, list[str]] = {}
        for family_name, ops in self.op_families.items():
            filtered = [op for op in ops if op in allowed_set]
            if filtered:
                family_space[family_name] = filtered

        family_names = list(family_space.keys())
        if not family_names:
            fallback_ops = [op for op in allowed_ops if op != "Identity"]
            return {"mlp": fallback_ops}, ["mlp"], ["mlp"]

        min_families = min(self.family_range[0], len(family_names))
        max_families = min(self.family_range[1], len(family_names))
        if max_families < min_families:
            max_families = min_families
        num_families = rng.randint(min_families, max_families)
        selected_families = rng.sample(family_names, k=num_families)
        op_families = [name for name in selected_families if family_space.get(name)]

        if not op_families:
            non_ssm = [name for name, ops in family_space.items() if ops]
            if non_ssm:
                fallback = rng.choice(non_ssm)
                if fallback not in selected_families:
                    selected_families.append(fallback)
                op_families = [fallback]

        return family_space, selected_families, op_families

    def _select_family_operations(
        self,
        rng,
        *,
        family_space: dict[str, list[str]],
        op_families: list[str],
        require_identity: bool,
        min_ops: int,
        max_ops: int | None,
    ) -> tuple[list[str], dict[str, list[str]]]:
        guaranteed_ops: list[str] = []
        family_choices: dict[str, list[str]] = {}

        for family_name in op_families:
            family_pool = [
                op for op in family_space.get(family_name, []) if op != "Identity"
            ]
            if not family_pool:
                family_pool = list(family_space.get(family_name, []))
            if not family_pool:
                continue
            chosen_op = rng.choice(family_pool)
            guaranteed_ops.append(chosen_op)
            family_choices[family_name] = [chosen_op]

        guaranteed_ops = list(dict.fromkeys(guaranteed_ops))
        ops_no_id = list(
            dict.fromkeys(
                op
                for family_name in op_families
                for op in family_space.get(family_name, [])
                if op != "Identity"
            )
        )
        if not ops_no_id:
            ops_no_id = [op for op in self.all_ops if op != "Identity"]

        max_ops_local = min(
            max_ops or len(ops_no_id) + (1 if require_identity else 0),
            len(ops_no_id) + (1 if require_identity else 0),
        )
        min_required = len(guaranteed_ops) + (1 if require_identity else 0)
        min_ops_local = max(min_ops, min_required)
        if max_ops_local < min_ops_local:
            max_ops_local = min_ops_local

        n_ops = rng.randint(min_ops_local, max_ops_local)
        non_identity_target = max(0, n_ops - (1 if require_identity else 0))
        extra_pool = [op for op in ops_no_id if op not in guaranteed_ops]
        extra_count = min(
            len(extra_pool), max(0, non_identity_target - len(guaranteed_ops))
        )
        extra_ops = rng.sample(extra_pool, k=extra_count) if extra_count > 0 else []
        selected_non_identity = guaranteed_ops + extra_ops

        if not selected_non_identity and ops_no_id:
            selected_non_identity = [rng.choice(ops_no_id)]

        selected_ops = (
            ["Identity"] + selected_non_identity
            if require_identity
            else selected_non_identity
        )
        return selected_ops, family_choices

    def _make_candidate_config(
        self,
        rng,
        allowed_ops: list[str],
        hidden_dim_choices: list[int],
        cell_range: tuple[int, int],
        node_range: tuple[int, int],
        *,
        min_ops: int = 2,
        max_ops: int | None = None,
        require_identity: bool = True,
        edge_to_op_target: float = 1.0,
        edge_to_op_max_ratio: float = 1.8,
    ) -> dict[str, Any]:
        family_space, selected_families, op_families = self._resolve_candidate_families(
            rng, allowed_ops
        )
        selected_ops, family_choices = self._select_family_operations(
            rng,
            family_space=family_space,
            op_families=op_families,
            require_identity=require_identity,
            min_ops=min_ops,
            max_ops=max_ops,
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

        if not self.arch_modes:
            arch_mode = "encoder_decoder"
        else:
            arch_mode = rng.choice(self.arch_modes)

        transformer_self_attention_type = (
            rng.choice(self.attention_variants) if self.attention_variants else "auto"
        )
        transformer_ffn_variant = (
            rng.choice(self.ffn_variants)
            if "attention" in selected_families
            else "auto"
        )
        family_choices = dict(family_choices)
        if "attention" in selected_families:
            family_choices["attention_variant"] = [transformer_self_attention_type]
            family_choices["attention_ffn"] = [transformer_ffn_variant]

        return {
            "selected_ops": selected_ops,
            "selected_families": selected_families,
            "family_choices": family_choices,
            "hidden_dim": rng.choice(hidden_dim_choices),
            "num_cells": rng.randint(cell_range[0], cell_range[1]),
            "num_nodes": int(num_nodes),
            "arch_mode": arch_mode,
            "transformer_self_attention_type": transformer_self_attention_type,
            "transformer_ffn_variant": transformer_ffn_variant,
            "transformer_use_moe": transformer_ffn_variant == "moe",
        }

    def _build_candidate_model(self, cfg: dict[str, Any]) -> nn.Module:
        return TimeSeriesDARTS(
            input_dim=self.input_dim,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["hidden_dim"],
            forecast_horizon=self.forecast_horizon,
            seq_length=self.seq_length,
            num_cells=cfg["num_cells"],
            num_nodes=cfg["num_nodes"],
            selected_ops=cfg["selected_ops"],
            arch_mode=cfg.get("arch_mode", "encoder_decoder"),
            transformer_self_attention_type=cfg.get(
                "transformer_self_attention_type", "auto"
            ),
            transformer_ffn_variant=cfg.get("transformer_ffn_variant", "swiglu"),
        ).to(self.device)

    def _create_bilevel_loaders(self, train_loader, seed: int = 42):
        dataset = train_loader.dataset
        train_size = int(0.7 * len(dataset))
        arch_size = len(dataset) - train_size
        g = torch.Generator().manual_seed(int(seed))
        train_ds, arch_ds = torch.utils.data.random_split(
            dataset, [train_size, arch_size], generator=g
        )
        _pin = train_loader.pin_memory
        train_model_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=train_loader.batch_size,
            shuffle=True,
            pin_memory=_pin,
            num_workers=train_loader.num_workers,
            persistent_workers=bool(train_loader.num_workers > 0),
        )
        train_arch_loader = torch.utils.data.DataLoader(
            arch_ds,
            batch_size=train_loader.batch_size,
            shuffle=True,
            pin_memory=_pin,
            num_workers=train_loader.num_workers,
            persistent_workers=bool(train_loader.num_workers > 0),
        )
        return train_arch_loader, train_model_loader

    def _plot_training_curve(
        self,
        train_losses: list[float],
        val_losses: list[float],
        title: str = "Training Progress",
        save_path: str | None = None,
    ):
        import matplotlib.pyplot as plt

        fig = _plot_mod.plot_training_curve(
            train_losses, val_losses, title=title, save_path=save_path
        )
        plt.close(fig)

    # ── Search orchestrator delegates ─────────────────────────────────────

    def _make_default_search_candidate_config(self, rng=None) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
    def _select_top_candidates(candidates: list[dict], top_k: int) -> list[dict]:
        return select_top_candidates(candidates, top_k)

    def _run_parallel_candidate_collection(
        self,
        num_candidates: int,
        candidate_fn,
        *,
        max_workers: int | None = None,
        on_result=None,
        error_log_fn=None,
    ) -> list[dict]:
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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
        # ── weight-decay ────────────────────────────────────────────────
        model_weight_decay: float = 1e-4,
        arch_weight_decay: float = 1e-3,
        # legacy alias kept for backward compatibility
        weight_decay: float | None = None,
        # ── bilevel / optimization ───────────────────────────────────────
        use_swa: bool = False,
        use_bilevel_optimization: bool = True,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        loss_type: str = "huber",
        patience: int = 15,
        verbose: bool = True,
        warmup_epochs: int = 2,
        architecture_update_freq: int = 3,
        # ── regularization ───────────────────────────────────────────────
        regularization_types: list[str] | None = None,
        regularization_weights: list[float] | None = None,
        # legacy: if False, disables regularization entirely
        use_regularization: bool = True,
        # ── temperature ──────────────────────────────────────────────────
        temperature_schedule: str = "cosine",
        # ── hessian penalty ──────────────────────────────────────────────
        hessian_penalty_weight: float = 0.0,
        hessian_fd_eps: float = 1e-2,
        hessian_update_freq: int = 1,
        # ── edge diversity / identity cap ────────────────────────────────
        edge_diversity_weight: float = 0.03,
        edge_usage_balance_weight: float = 0.04,
        edge_identity_cap: float = 0.45,
        edge_identity_cap_weight: float = 0.02,
        # legacy aliases
        identity_dominance_cap: float | None = None,
        # ── edge sharpening ──────────────────────────────────────────────
        edge_sharpening_max_weight: float = 0.03,
        edge_sharpening_start_frac: float = 0.35,
        # legacy aliases
        edge_sharpening_strength: float | None = None,
        edge_sharpening_start_epoch: int | None = None,
        # ── progressive shrinking / pruning ──────────────────────────────
        progressive_shrinking: bool = True,
        hybrid_pruning_start_epoch: int = 20,
        hybrid_pruning_interval: int = 10,
        hybrid_pruning_base_threshold: float = 0.15,
        hybrid_pruning_strategy: str = "performance",
        hybrid_pruning_freeze_logit: float = -20.0,
        # legacy aliases
        use_progressive_training: bool | None = None,
        pruning_enabled: bool | None = None,
        pruning_start_epoch: int | None = None,
        pruning_threshold: float | None = None,
        # ── misc ─────────────────────────────────────────────────────────
        bilevel_split_seed: int = 42,
        state_mix_ortho_reg_weight: float = 1e-3,
        arch_grad_ema_beta: float = 0.0,
        beta_darts_weight: float = 0.0,
        moe_balance_weight: float = 5e-3,
        transformer_exploration_weight: float = 1e-2,
        # silently absorbed legacy-only kwargs
        temperature: float = 1.0,  # noqa: ARG002
        pruning_hard_epoch: int | None = None,  # noqa: ARG002
        log_arch_gradients: bool = False,  # noqa: ARG002
        use_gdas: bool = False,
    ) -> dict[str, Any]:
        """
        Run DARTS bilevel architecture search training.

        Returns dict with ``model``, ``best_val_loss``, ``train_losses``,
        ``val_losses``, ``alpha_values``, ``diversity_scores``,
        ``training_time``, ``final_metrics``.
        """
        # ── resolve legacy aliases ────────────────────────────────────────
        eff_model_wd = weight_decay if weight_decay is not None else model_weight_decay
        eff_edge_id_cap = (
            identity_dominance_cap
            if identity_dominance_cap is not None
            else edge_identity_cap
        )
        eff_sharpen_w = (
            edge_sharpening_strength
            if edge_sharpening_strength is not None
            else edge_sharpening_max_weight
        )
        eff_progressive = (
            progressive_shrinking
            if use_progressive_training is None
            else bool(use_progressive_training)
        )
        if pruning_enabled is not None and not pruning_enabled:
            eff_progressive = False
        eff_prune_start = (
            pruning_start_epoch
            if pruning_start_epoch is not None
            else hybrid_pruning_start_epoch
        )
        eff_prune_thresh = (
            pruning_threshold
            if pruning_threshold is not None
            else hybrid_pruning_base_threshold
        )
        if edge_sharpening_start_epoch is not None and epochs > 0:
            eff_sharpen_frac = float(edge_sharpening_start_epoch) / float(epochs)
        else:
            eff_sharpen_frac = edge_sharpening_start_frac

        if not use_regularization:
            regularization_types = []
            regularization_weights = []

        return _dl_mod.train_darts_model(
            self,
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            arch_learning_rate=arch_learning_rate,
            model_learning_rate=model_learning_rate,
            model_weight_decay=eff_model_wd,
            arch_weight_decay=arch_weight_decay,
            use_swa=use_swa,
            use_bilevel_optimization=use_bilevel_optimization,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
            loss_type=loss_type,
            patience=patience,
            verbose=verbose,
            warmup_epochs=warmup_epochs,
            architecture_update_freq=architecture_update_freq,
            regularization_types=regularization_types,
            regularization_weights=regularization_weights,
            temperature_schedule=temperature_schedule,
            hessian_penalty_weight=hessian_penalty_weight,
            hessian_fd_eps=hessian_fd_eps,
            hessian_update_freq=hessian_update_freq,
            edge_diversity_weight=edge_diversity_weight,
            edge_usage_balance_weight=edge_usage_balance_weight,
            edge_identity_cap=eff_edge_id_cap,
            edge_identity_cap_weight=edge_identity_cap_weight,
            edge_sharpening_max_weight=eff_sharpen_w,
            edge_sharpening_start_frac=eff_sharpen_frac,
            progressive_shrinking=eff_progressive,
            hybrid_pruning_start_epoch=eff_prune_start,
            hybrid_pruning_interval=hybrid_pruning_interval,
            hybrid_pruning_base_threshold=eff_prune_thresh,
            hybrid_pruning_strategy=hybrid_pruning_strategy,
            hybrid_pruning_freeze_logit=hybrid_pruning_freeze_logit,
            bilevel_split_seed=bilevel_split_seed,
            state_mix_ortho_reg_weight=state_mix_ortho_reg_weight,
            arch_grad_ema_beta=arch_grad_ema_beta,
            beta_darts_weight=beta_darts_weight,
            moe_balance_weight=moe_balance_weight,
            transformer_exploration_weight=transformer_exploration_weight,
            use_gdas=use_gdas,
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
        use_onecycle: bool = True,
        swa_start_ratio: float = 0.33,
        grad_clip_norm: float = 1.0,
        # legacy-only kwargs silently absorbed
        verbose: bool = True,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        use_swa: bool = False,
        swa_start: int = 75,
        swa_lr: float = 5e-4,
        compute_final_metrics: bool = True,
    ) -> dict[str, Any]:
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
            use_onecycle=use_onecycle,
            swa_start_ratio=swa_start_ratio,
            grad_clip_norm=grad_clip_norm,
            use_amp=use_amp,
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
        max_workers: int | None = None,
        collect_stats: bool = False,
        parallelism_levels=None,
        stats_dir: str = "search_stats",
        run_name: str | None = None,
        retrain_final_from_scratch: bool = True,
        discrete_arch_threshold: float = 0.3,
        use_amp: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
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
            use_amp=use_amp,
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
    ) -> dict[str, Any]:
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
        save_csv_path: str | None = None,
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
        max_workers: int | None = None,
        save_plots: bool = True,
        save_csv: bool = True,
        output_dir: str = "ablation_results",
        run_id: str | None = None,
        cell_range: tuple = (1, 2),
        node_range: tuple = (2, 4),
        min_ops: int = 2,
        max_ops: int | None = None,
        require_identity: bool = True,
    ) -> dict[str, Any]:
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
        max_workers: int | None = None,
        use_weight_schemes: bool = False,
        n_random: int = 50,
        random_sigma: float = 0.25,
        robustness_mode: str = "topk_freq",
        topk_ref: int | None = None,
        min_ops: int = 2,
        max_ops: int | None = None,
        cell_range: tuple = (1, 2),
        node_range: tuple = (2, 4),
        hidden_dim_choices: list[int] | None = None,
        require_identity: bool = True,
    ) -> dict[str, Any]:
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
        alpha_values: list,
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
        device: str | None = None,
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
