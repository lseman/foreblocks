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

from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler

from .architecture.core_blocks import TimeSeriesDARTS
from .architecture.finalization import (
    derive_final_architecture as derive_fixed_architecture,
)
from .config import (
    DEFAULT_ARCH_MODES,
    DEFAULT_ATTENTION_VARIANTS,
    DEFAULT_FFN_VARIANTS,
    DARTSTrainConfig,
)
from .config import (
    DEFAULT_OPS as SEARCH_DEFAULT_OPS,
)
from .evaluation import plotting as _plot_mod
from .evaluation.metrics import evaluate_on_loader
from .search import (
    ablation as _abl_mod,
)
from .search import (
    multi_fidelity as _mf_mod,
)
from .search import (
    robust_pool as _rp_mod,
)
from .search import (
    zero_cost as _zc_mod,
)
from .search.candidate_config import (
    make_candidate_config,
    normalize_op_families,
)
from .search.orchestrator import (
    evaluate_search_candidate,
    make_default_search_candidate_config,
    run_parallel_candidate_collection,
    select_top_candidates,
)
from .training import darts_loop as _dl_mod
from .training import final_trainer as _ft_mod
from .training.helpers import (
    AlphaTracker,
)
from .training.helpers import (
    default_as_probability_vector as _as_probability_vector,
)
from .utils.training import (
    autocast_ctx,
    create_progress_bar,
    get_loss_function,
    unpack_forecasting_batch,
)

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
        self.op_families = normalize_op_families(op_families, self.all_ops)
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
    # Internal utilities (delegated to shared helper modules)
    # ─────────────────────────────────────────────────────────────────────

    def _get_loss_function(self, loss_type: str = "huber"):
        return get_loss_function(loss_type)

    def _autocast(self, enabled: bool):
        return autocast_ctx(self.device, enabled=enabled)

    def _create_progress_bar(self, iterable, desc: str, leave: bool = True, **kwargs):
        return create_progress_bar(iterable, desc=desc, leave=leave, **kwargs)

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
        return evaluate_on_loader(
            model=model,
            dataloader=dataloader,
            loss_fn=self._get_loss_function(loss_type),
            device=self.device,
        )

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
        return make_candidate_config(
            rng=rng,
            allowed_ops=allowed_ops,
            hidden_dim_choices=hidden_dim_choices,
            cell_range=cell_range,
            node_range=node_range,
            op_families=self.op_families,
            family_range=self.family_range,
            min_ops=min_ops,
            max_ops=max_ops,
            require_identity=require_identity,
            edge_to_op_target=edge_to_op_target,
            edge_to_op_max_ratio=edge_to_op_max_ratio,
            arch_modes=self.arch_modes,
            attention_variants=self.attention_variants,
            ffn_variants=self.ffn_variants,
        )

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
        progress: bool = False,
        progress_desc: str = "Phase 1 candidates",
        candidate_timeout: float = 120.0,
    ) -> list[dict]:
        return run_parallel_candidate_collection(
            num_candidates=num_candidates,
            candidate_fn=candidate_fn,
            max_workers=max_workers,
            on_result=on_result,
            error_log_fn=error_log_fn,
            progress=progress,
            progress_desc=progress_desc,
            candidate_timeout=candidate_timeout,
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
        train_config: DARTSTrainConfig | None = None,
        # Deprecated: positional/keyword params absorbed into train_config.
        # Kept for backward compat but deprecated — pass a DARTSTrainConfig instead.
        epochs: int | None = None,
        arch_learning_rate: float | None = None,
        model_learning_rate: float | None = None,
        model_weight_decay: float | None = None,
        arch_weight_decay: float | None = None,
        use_swa: bool | None = None,
        use_bilevel_optimization: bool | None = None,
        gradient_accumulation_steps: int | None = None,
        use_amp: bool | None = None,
        loss_type: str | None = None,
        patience: int | None = None,
        verbose: bool | None = None,
        warmup_epochs: int | None = None,
        architecture_update_freq: int | None = None,
        regularization_types: list[str] | None = None,
        regularization_weights: list[float] | None = None,
        temperature_schedule: str | None = None,
        hessian_penalty_weight: float | None = None,
        hessian_fd_eps: float | None = None,
        hessian_update_freq: int | None = None,
        edge_diversity_weight: float | None = None,
        edge_usage_balance_weight: float | None = None,
        edge_identity_cap: float | None = None,
        edge_identity_cap_weight: float | None = None,
        edge_sharpening_max_weight: float | None = None,
        edge_sharpening_start_frac: float | None = None,
        progressive_shrinking: bool | None = None,
        hybrid_pruning_start_epoch: int | None = None,
        hybrid_pruning_interval: int | None = None,
        hybrid_pruning_base_threshold: float | None = None,
        hybrid_pruning_strategy: str | None = None,
        hybrid_pruning_freeze_logit: float | None = None,
        bilevel_split_seed: int | None = None,
        state_mix_ortho_reg_weight: float | None = None,
        arch_grad_ema_beta: float | None = None,
        beta_darts_weight: float | None = None,
        moe_balance_weight: float | None = None,
        transformer_exploration_weight: float | None = None,
        op_gdas: bool | None = None,
        variant_gdas: bool | None = None,
        compute_metrics: bool = True,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        # Legacy aliases (deprecated):
        weight_decay: float | None = None,  # noqa: ARG002
        identity_dominance_cap: float | None = None,  # noqa: ARG002
        edge_sharpening_strength: float | None = None,  # noqa: ARG002
        use_regularization: bool = True,  # noqa: ARG002
        use_progressive_training: bool | None = None,  # noqa: ARG002
        pruning_enabled: bool | None = None,  # noqa: ARG002
        pruning_start_epoch: int | None = None,  # noqa: ARG002
        pruning_threshold: float | None = None,  # noqa: ARG002
        temperature: float = 1.0,  # noqa: ARG002
        pruning_hard_epoch: int | None = None,  # noqa: ARG002
        log_arch_gradients: bool = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Run DARTS bilevel architecture search training.

        Prefer passing a :class:`~darts.config.DARTSTrainConfig` as
        ``train_config``.  Legacy keyword arguments are still accepted but
        deprecated — they will be removed in a future release.

        Returns dict with ``model``, ``best_val_loss``, ``train_losses``,
        ``val_losses``, ``alpha_values``, ``diversity_scores``,
        ``training_time``, ``final_metrics``.
        """
        # Merge legacy kwargs into config (deprecated path)
        if train_config is None:
            train_config = DARTSTrainConfig()
            # Apply any non-None legacy overrides
            if epochs is not None:
                train_config.epochs = epochs
            if arch_learning_rate is not None:
                train_config.arch_learning_rate = arch_learning_rate
            if model_learning_rate is not None:
                train_config.model_learning_rate = model_learning_rate
            if model_weight_decay is not None:
                train_config.model_weight_decay = model_weight_decay
            if arch_weight_decay is not None:
                train_config.arch_weight_decay = arch_weight_decay
            if use_swa is not None:
                train_config.use_swa = use_swa
            if use_bilevel_optimization is not None:
                train_config.use_bilevel_optimization = use_bilevel_optimization
            if gradient_accumulation_steps is not None:
                train_config.gradient_accumulation_steps = gradient_accumulation_steps
            if use_amp is not None:
                train_config.use_amp = use_amp
            if loss_type is not None:
                train_config.loss_type = loss_type
            if patience is not None:
                train_config.patience = patience
            if verbose is not None:
                train_config.verbose = verbose
            if warmup_epochs is not None:
                train_config.warmup_epochs = warmup_epochs
            if architecture_update_freq is not None:
                train_config.architecture_update_freq = architecture_update_freq
            if regularization_types is not None:
                train_config.regularization_types = regularization_types
            if regularization_weights is not None:
                train_config.regularization_weights = regularization_weights
            if temperature_schedule is not None:
                train_config.temperature_schedule = temperature_schedule
            if hessian_penalty_weight is not None:
                train_config.hessian_penalty_weight = hessian_penalty_weight
            if hessian_fd_eps is not None:
                train_config.hessian_fd_eps = hessian_fd_eps
            if hessian_update_freq is not None:
                train_config.hessian_update_freq = hessian_update_freq
            if edge_diversity_weight is not None:
                train_config.edge_diversity_weight = edge_diversity_weight
            if edge_usage_balance_weight is not None:
                train_config.edge_usage_balance_weight = edge_usage_balance_weight
            if edge_identity_cap is not None:
                train_config.edge_identity_cap = edge_identity_cap
            if edge_identity_cap_weight is not None:
                train_config.edge_identity_cap_weight = edge_identity_cap_weight
            if edge_sharpening_max_weight is not None:
                train_config.edge_sharpening_max_weight = edge_sharpening_max_weight
            if edge_sharpening_start_frac is not None:
                train_config.edge_sharpening_start_frac = edge_sharpening_start_frac
            if progressive_shrinking is not None:
                train_config.progressive_shrinking = progressive_shrinking
            if hybrid_pruning_start_epoch is not None:
                train_config.hybrid_pruning_start_epoch = hybrid_pruning_start_epoch
            if hybrid_pruning_interval is not None:
                train_config.hybrid_pruning_interval = hybrid_pruning_interval
            if hybrid_pruning_base_threshold is not None:
                train_config.hybrid_pruning_base_threshold = hybrid_pruning_base_threshold
            if hybrid_pruning_strategy is not None:
                train_config.hybrid_pruning_strategy = hybrid_pruning_strategy
            if hybrid_pruning_freeze_logit is not None:
                train_config.hybrid_pruning_freeze_logit = hybrid_pruning_freeze_logit
            if bilevel_split_seed is not None:
                train_config.bilevel_split_seed = bilevel_split_seed
            if state_mix_ortho_reg_weight is not None:
                train_config.state_mix_ortho_reg_weight = state_mix_ortho_reg_weight
            if arch_grad_ema_beta is not None:
                train_config.arch_grad_ema_beta = arch_grad_ema_beta
            if beta_darts_weight is not None:
                train_config.beta_darts_weight = beta_darts_weight
            if moe_balance_weight is not None:
                train_config.moe_balance_weight = moe_balance_weight
            if transformer_exploration_weight is not None:
                train_config.transformer_exploration_weight = transformer_exploration_weight
            if op_gdas is not None:
                train_config.op_gdas = op_gdas
            if variant_gdas is not None:
                train_config.variant_gdas = variant_gdas
            if max_train_batches is not None:
                train_config.max_train_batches = max_train_batches
            if max_val_batches is not None:
                train_config.max_val_batches = max_val_batches

        return _dl_mod.train_darts_model(
            self,
            model,
            train_loader,
            val_loader,
            epochs=train_config.epochs,
            arch_learning_rate=train_config.arch_learning_rate,
            model_learning_rate=train_config.model_learning_rate,
            arch_weight_decay=train_config.arch_weight_decay,
            model_weight_decay=train_config.model_weight_decay,
            patience=train_config.patience,
            loss_type=train_config.loss_type,
            use_swa=train_config.use_swa,
            warmup_epochs=train_config.warmup_epochs,
            architecture_update_freq=train_config.architecture_update_freq,
            diversity_check_freq=train_config.diversity_check_freq,
            progressive_shrinking=train_config.progressive_shrinking,
            hybrid_pruning_start_epoch=train_config.hybrid_pruning_start_epoch,
            hybrid_pruning_interval=train_config.hybrid_pruning_interval,
            hybrid_pruning_base_threshold=train_config.hybrid_pruning_base_threshold,
            hybrid_pruning_strategy=train_config.hybrid_pruning_strategy,
            hybrid_pruning_freeze_logit=train_config.hybrid_pruning_freeze_logit,
            use_bilevel_optimization=train_config.use_bilevel_optimization,
            use_amp=train_config.use_amp,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            verbose=train_config.verbose,
            regularization_types=train_config.regularization_types,
            regularization_weights=train_config.regularization_weights,
            temperature_schedule=train_config.temperature_schedule,
            edge_sharpening_max_weight=train_config.edge_sharpening_max_weight,
            edge_sharpening_start_frac=train_config.edge_sharpening_start_frac,
            hessian_penalty_weight=train_config.hessian_penalty_weight,
            hessian_fd_eps=train_config.hessian_fd_eps,
            hessian_update_freq=train_config.hessian_update_freq,
            bilevel_split_seed=train_config.bilevel_split_seed,
            state_mix_ortho_reg_weight=train_config.state_mix_ortho_reg_weight,
            edge_diversity_weight=train_config.edge_diversity_weight,
            edge_usage_balance_weight=train_config.edge_usage_balance_weight,
            edge_identity_cap=train_config.edge_identity_cap,
            edge_identity_cap_weight=train_config.edge_identity_cap_weight,
            arch_grad_ema_beta=train_config.arch_grad_ema_beta,
            beta_darts_weight=train_config.beta_darts_weight,
            moe_balance_weight=train_config.moe_balance_weight,
            transformer_exploration_weight=train_config.transformer_exploration_weight,
            op_gdas=train_config.op_gdas,
            variant_gdas=train_config.variant_gdas,
            compute_metrics=compute_metrics,
            max_train_batches=train_config.max_train_batches,
            max_val_batches=train_config.max_val_batches,
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
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        max_test_batches: int | None = None,
        compile_model: bool = False,
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
            use_swa=use_swa,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            max_test_batches=max_test_batches,
            compile_model=compile_model,
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
        phase1_progress: bool = False,
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
            phase1_progress=phase1_progress,
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
