"""
Configuration dataclasses for DARTS Neural Architecture Search.

Centralises all hyperparameters so callers can build, override, and pass a
single object instead of long keyword-argument lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Architecture / Search-space
# ---------------------------------------------------------------------------

DEFAULT_ARCH_MODES: list[str] = [
    "encoder_decoder",
    "encoder_only",
    "decoder_only",
]

DEFAULT_OPS: list[str] = [
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
    "InvertedAttention",
    "DLinear",
    "TimeMixer",
    "NBeats",
    "TimesNet",
]

DEFAULT_OP_FAMILIES: dict[str, list[str]] = {
    "conv": [
        "TimeConv",
        "TCN",
        "ConvMixer",
        "MultiScaleConv",
        "PyramidConv",
    ],
    "frequency": [
        "Wavelet",
        "Fourier",
        "DLinear",
        "TimesNet",
    ],
    "attention": [
        "PatchEmbed",
        "InvertedAttention",
    ],
    "mlp": [
        "GRN",
        "ResidualMLP",
        "TimeMixer",
        "NBeats",
    ],
}

DEFAULT_ATTENTION_VARIANTS: list[str] = ["auto"]
DEFAULT_FFN_VARIANTS: list[str] = ["auto"]


@dataclass
class DARTSSearchSpaceConfig:
    """Defines the operation/cell/node search space."""

    all_ops: list[str] = field(default_factory=lambda: list(DEFAULT_OPS))
    arch_modes: list[str] = field(default_factory=lambda: list(DEFAULT_ARCH_MODES))
    op_families: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_OP_FAMILIES.items()}
    )
    hidden_dims: list[int] = field(default_factory=lambda: [32, 64, 128])
    cell_range: tuple[int, int] = (1, 2)
    node_range: tuple[int, int] = (2, 4)
    family_range: tuple[int, int] = (1, 3)
    min_ops: int = 2
    max_ops: int | None = None
    require_identity: bool = True
    edge_to_op_target: float = 1.0
    edge_to_op_max_ratio: float = 1.8
    attention_variants: list[str] = field(
        default_factory=lambda: list(DEFAULT_ATTENTION_VARIANTS)
    )
    ffn_variants: list[str] = field(default_factory=lambda: list(DEFAULT_FFN_VARIANTS))


# ---------------------------------------------------------------------------
# DARTS Bilevel Training
# ---------------------------------------------------------------------------


@dataclass
class DARTSTrainConfig:
    """Hyperparameters for the DARTS bilevel training phase.

    This is the single source of truth for all bilevel-DARTS knobs.  Use it as
    the sole argument to ``trainer.train_darts_model()`` — positional and
    keyword overrides on the old API surface are kept for backward compatibility
    but are deprecated.

    Grouped by concern for readability.  See each field for defaults.
    """

    # ── Epochs / schedule ─────────────────────────────────────────────────
    epochs: int = 50
    warmup_epochs: int = 2
    patience: int = 10
    loss_type: str = "huber"
    verbose: bool = True
    use_amp: bool = True
    use_swa: bool = False

    # ── Learning rates ────────────────────────────────────────────────────
    arch_learning_rate: float = 1e-2
    model_learning_rate: float = 1e-3
    arch_weight_decay: float = 1e-3
    model_weight_decay: float = 1e-4

    # ── Optimisation ──────────────────────────────────────────────────────
    use_bilevel_optimization: bool = True
    bilevel_split_seed: int = 42
    gradient_accumulation_steps: int = 1
    architecture_update_freq: int = 3
    diversity_check_freq: int = 1

    # ── Temperature / sampling ────────────────────────────────────────────
    temperature_schedule: str = "cosine"

    # ── Progressive shrinking & pruning ───────────────────────────────────
    progressive_shrinking: bool = True
    hybrid_pruning_start_epoch: int = 20
    hybrid_pruning_interval: int = 10
    hybrid_pruning_base_threshold: float = 0.15
    hybrid_pruning_strategy: str = "performance"
    hybrid_pruning_freeze_logit: float = -20.0

    # ── Architecture regularisation ───────────────────────────────────────
    regularization_types: list[str] | None = None  # defaults to ["kl_divergence", "efficiency"]
    regularization_weights: list[float] | None = None  # defaults to [0.05, 0.01]
    beta_darts_weight: float = 0.0  # L2 on raw arch logits
    state_mix_ortho_reg_weight: float = 1e-3
    moe_balance_weight: float = 5e-3
    transformer_exploration_weight: float = 1e-2
    arch_grad_ema_beta: float = 0.0  # EMA on arch gradients [0.7, 0.9] recommended for noisy bilevel

    # ── Edge-level regularisation ─────────────────────────────────────────
    edge_diversity_weight: float = 0.03
    edge_usage_balance_weight: float = 0.04
    edge_identity_cap: float = 0.45
    edge_identity_cap_weight: float = 0.02
    edge_sharpening_max_weight: float = 0.03
    edge_sharpening_start_frac: float = 0.35

    # ── Hessian / second-order ────────────────────────────────────────────
    hessian_penalty_weight: float = 0.0
    hessian_fd_eps: float = 1e-2
    hessian_update_freq: int = 1

    # ── GDAS / DrNAS ──────────────────────────────────────────────────────
    op_gdas: bool = True  # single-path sampling per edge (dominant speedup)
    variant_gdas: bool | None = None  # mirrors op_gdas unless explicitly set

    # ── Batching overrides ────────────────────────────────────────────────
    max_train_batches: int | None = None
    max_val_batches: int | None = None

    def resolve_device(self) -> str:
        """Resolve to a concrete device string (used when wrapping in DARTSConfig)."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- Deprecated aliases (kept for backward compat) --
    # The following fields were renamed; the old names are absorbed in the
    # trainer method.  They remain here so code that reads config fields
    # by name does not break.
    identity_dominance_cap: float = 0.45  # noqa: ERA001  # → edge_identity_cap
    edge_sharpening_strength: float = 0.03  # noqa: ERA001  # → edge_sharpening_max_weight
    progressive_training: bool = True  # noqa: ERA001  # → progressive_shrinking
    pruning_enabled: bool = True  # noqa: ERA001  # → derived from progressive_shrinking
    pruning_start_epoch: int = 20  # noqa: ERA001  # → hybrid_pruning_start_epoch
    pruning_threshold: float = 0.15  # noqa: ERA001  # → hybrid_pruning_base_threshold
    log_arch_gradients: bool = False  # noqa: ERA001  # absorbed


# ---------------------------------------------------------------------------
# Final Model Training
# ---------------------------------------------------------------------------


@dataclass
class FinalTrainConfig:
    """Hyperparameters for the fixed-architecture final training phase."""

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 50
    loss_type: str = "huber"
    use_onecycle: bool = True
    swa_start_ratio: float = 0.33
    grad_clip_norm: float = 1.0
    use_amp: bool = True


# ---------------------------------------------------------------------------
# Multi-Fidelity Search
# ---------------------------------------------------------------------------


@dataclass
class MultiFidelitySearchConfig:
    """Hyperparameters for the multi-fidelity NAS search pipeline."""

    num_candidates: int = 10
    search_epochs: int = 10
    final_epochs: int = 100
    max_samples: int = 32
    top_k: int = 5
    max_workers: int | None = None
    collect_stats: bool = False
    parallelism_levels: list[int] | None = None
    est_overhead_per_task: float = 0.0
    est_fixed_overhead_phase1: float = 0.0
    est_fixed_overhead_phase3: float = 0.0
    benchmark_phase1_workers: list[int] | None = None
    benchmark_phase1_candidates: int | None = None
    stats_dir: str = "search_stats"
    run_name: str | None = None
    retrain_final_from_scratch: bool = True
    discrete_arch_threshold: float = 0.3
    phase1_rescore_mode: str = "pool"
    phase3_reduction_factor: int = 2
    phase3_min_epoch_budget: int = 2
    phase3_rung_epochs: list[int] | None = None
    phase3_train_max_batches: int | None = None
    phase3_val_max_batches: int | None = None


# ---------------------------------------------------------------------------
# Ablation / Zero-Cost Search
# ---------------------------------------------------------------------------


@dataclass
class AblationSearchConfig:
    """Hyperparameters for the zero-cost weight-scheme ablation search."""

    num_candidates: int = 20
    max_samples: int = 32
    num_batches: int = 1
    top_k: int = 5
    max_workers: int | None = None
    n_random: int = 50
    random_sigma: float = 0.25
    seed: int = 0
    save_dir: str = "."
    save_prefix: str = "zc_weight_ablation"


# ---------------------------------------------------------------------------
# Robust Pool Search
# ---------------------------------------------------------------------------


@dataclass
class RobustPoolSearchConfig:
    """Hyperparameters for robust initial-pool search over op-pool perturbations."""

    n_pools: int = 25
    pool_size_range: tuple[int, int] = (4, 10)
    pool_seed: int = 0
    num_candidates: int = 50
    top_k: int = 10
    max_samples: int = 32
    num_batches: int = 1
    max_workers: int | None = None
    seed: int = 0
    use_weight_schemes: bool = False
    n_random: int = 0
    random_sigma: float = 0.25
    robustness_mode: str = "spearman"
    topk_ref: int | None = None
    min_ops: int = 2
    max_ops: int | None = None
    cell_range: tuple[int, int] = (1, 2)
    node_range: tuple[int, int] = (2, 4)
    hidden_dim_choices: list[int] | None = None
    require_identity: bool = True


# ---------------------------------------------------------------------------
# Composite top-level config
# ---------------------------------------------------------------------------


@dataclass
class DARTSConfig:
    """
    Master configuration for the entire DARTS pipeline.

    Compose individual configs or override individual fields.

    Example::

        cfg = DARTSConfig(input_dim=5, forecast_horizon=12)
        cfg.train.epochs = 30
        cfg.search.num_candidates = 50
    """

    # Model / runner identity
    input_dim: int = 3
    forecast_horizon: int = 6
    seq_length: int = 12
    device: str = "auto"  # "auto" resolves to cuda if available

    # Nested sub-configs
    space: DARTSSearchSpaceConfig = field(default_factory=DARTSSearchSpaceConfig)
    train: DARTSTrainConfig = field(default_factory=DARTSTrainConfig)
    final: FinalTrainConfig = field(default_factory=FinalTrainConfig)
    search: MultiFidelitySearchConfig = field(
        default_factory=MultiFidelitySearchConfig
    )
    ablation: AblationSearchConfig = field(default_factory=AblationSearchConfig)
    robust_pool: RobustPoolSearchConfig = field(default_factory=RobustPoolSearchConfig)

    def resolve_device(self) -> str:
        """Return concrete device string, resolving 'auto'."""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# Deprecated: misspelled alias kept for backwards compatibility. Use
# ``MultiFidelitySearchConfig`` in new code.
MultiFildelitySearchConfig = MultiFidelitySearchConfig
