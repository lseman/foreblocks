from dataclasses import dataclass
from typing import List, Optional, Union

import torch


@dataclass
class FeatureConfig:
    """Configuration class for feature engineering parameters."""

    # Core settings
    task: str = "regression"
    random_state: int = 42
    corr_threshold: float = 0.95

    # Feature creation flags
    create_math_features: bool = True
    create_interactions: bool = True
    create_polynomials: bool = True
    create_categorical: bool = True
    create_binning: bool = True
    create_statistical: bool = True
    create_clustering: bool = True
    create_fourier: bool = False
    create_rff: bool = True
    use_boruta: bool = False
    use_loo: bool = False
    use_woe: bool = False
    use_autoencoder: bool = False

    # Global thresholds
    rare_threshold: float = 0.01
    min_variance_threshold: float = 1e-6
    max_rows_score: int = 50000
    epsilon: float = 1e-8
    dtype_out: str = "float32"
    math_target_aware: bool = True
    math_target_weight: float = 0.25
    math_max_transforms_per_feature: int = 3

    # Feature limits
    n_bins: int = 5
    n_clusters: int = 8
    n_fourier_terms: int = 3
    max_interactions: int = 200
    max_selected_interactions: int = 64
    max_polynomials: int = 50
    max_features: int = 500
    min_features: int = 1
    min_samples: int = 10

    # Selection parameters
    use_quantile_transform: bool = True
    mi_threshold: float = 0.001
    shap_threshold: float = 0.001
    mi_spearman_gate: float = 0.05
    mi_min_overlap: int = 50
    mi_bins: int = 16
    selector_stable_mi: bool = True
    selector_cv: int = 5
    selector_min_freq: float = 0.5
    selector_redundancy_prune: bool = True
    selector_redundancy_threshold: float = 0.98
    selector_redundancy_pool: int = 200

    # RFECV parameters
    use_rfecv: bool = False
    rfecv_step: Union[int, float] = 0.1
    rfecv_cv: int = 5
    rfecv_min_features: Optional[int] = None
    rfecv_max_features: Optional[int] = None
    rfecv_patience: int = 5
    rfecv_use_ensemble: bool = True
    rfecv_stability_selection: bool = True

    # Categorical settings
    target_encode_threshold: int = 10
    cat_top_k: Optional[int] = None
    cat_use_stratified_kfold: bool = True
    cat_target_noise_std: float = 0.0

    # Binning settings
    binning_strategies: Optional[List[str]] = None
    binning_auto_supervised: bool = True
    binning_min_bin_fraction: float = 0.01

    # Interaction settings
    scorer: str = "mi"
    n_splits: int = 5
    min_selected_per_fold: int = 20
    importance_agg: str = "median"
    max_pairs_screen: int = 200
    interaction_prescreen_topk: int = 32
    interaction_redundancy_corr: float = 0.985
    interaction_prune_redundancy: bool = False
    interaction_stability_selection: bool = True
    interaction_stability_min_freq: float = 0.5
    interaction_fast_mode: bool = True
    interaction_row_subsample: int = 5000
    interaction_n_jobs: int = -1
    interaction_prescreen_spearman: bool = False
    interaction_winsor_p: float = 0.001
    pair_corr_with_y: bool = True
    pair_max_per_feature: int = 32
    corr_avoid_redundancy: float = 0.995

    # Interaction operation toggles
    include_sum: bool = True
    include_diff: bool = True
    include_prod: bool = True
    include_ratio: bool = True
    include_norm_ratio: bool = True
    include_minmax: bool = True
    include_zdiff: bool = True
    include_logratio: bool = True
    include_rootprod: bool = True
    include_square: bool = True
    include_sqrt: bool = True
    include_cube: bool = False
    include_reciprocal: bool = False
    include_log: bool = False

    # Autoencoder settings
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 64
    autoencoder_lr: float = 1e-3
    autoencoder_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder_latent_ratio: float = 0.25

    # Random Fourier Features settings
    rff_n_components: int = 100
    rff_gamma: Union[float, str] = "auto"
    rff_kernel: str = "rbf"
    rff_max_features: int = 50
