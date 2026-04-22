from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VMDParameters:
    # Optuna
    n_trials: int = 30
    max_K: int = 6

    # VMD
    tol: float = 1e-6
    tol_omega: float | None = None
    alpha_min: float = 500
    alpha_max: float = 5000
    tau: float = 0.0
    DC: int = 0
    init: int = 1
    random_seed: int | None = None
    max_iter: int = 300
    # ADMM acceleration (over-relaxation; 1.0 disables)
    admm_over_relax: float = 1.6
    # nonstationary/chirp extension (VNCMD/NCMD IF-track solver)
    if_tracking: bool = False
    if_window_size: int = 256
    if_hop_size: int = 128
    if_center_smooth: float = 0.85
    if_update_step: float = 1.0
    vncmd_min_track_gap: float | None = None
    vncmd_envelope_ridge_scale: float = 0.0
    # optional warm-start state (streaming / rolling windows)
    warm_start_u_hat: np.ndarray | None = None
    warm_start_omega: np.ndarray | None = None
    # omega update stabilization (frequency-center damping)
    omega_momentum: float = 0.0
    omega_shrinkage: float = 0.0
    omega_max_step: float = 0.0
    # mode-specific bandwidth penalty (alpha_k) adaptation
    adaptive_alpha: bool = False
    adaptive_alpha_start_iter: int = 10
    adaptive_alpha_update_every: int = 5
    adaptive_alpha_lr: float = 0.15
    adaptive_alpha_min_scale: float = 0.3
    adaptive_alpha_max_scale: float = 6.0
    adaptive_alpha_skip_dc: bool = True

    # boundary
    boundary_method: str = "mirror"
    use_soft_junction: bool = False
    window_alpha: float | None = None  # if None -> auto
    fft_backend: str = "fftw"  # "fftw" | "torch"
    fft_device: str = "auto"  # "auto" | "cpu" | "cuda"

    # post
    apply_tapering: bool = True
    mode_energy_floor: float = 0.01
    merge_freq_tol: float = 0.15

    # extras
    use_fs_vmd: bool = False
    use_mvmd: bool = False

    # K selection
    k_selection: str = "penalized"  # "penalized" | "fbd" | "entropy" | "optuna"
    k_penalty_lambda: float = 0.02
    k_overlap_mu: float = 0.10
    # global optimizer toggle
    search_method: str = "optuna"  # "optuna" | "entropy"
    # entropy-based K+alpha estimation
    entropy_alpha_default: float = 2000.0
    entropy_embed_dim: int = 3
    entropy_delay: int = 1
    entropy_classes: int = 6
    entropy_weight_pe: float = 0.5
    entropy_weight_de: float = 0.5
    entropy_k_penalty: float = 0.02
    entropy_alpha_span: float = 2.0
    entropy_alpha_grid_points: int = 7
    entropy_overlap_weight: float = 0.25
    entropy_alpha_full_range: bool = False
    # DE objective details
    entropy_scales: int = 3
    entropy_refined_composite: bool = True
    entropy_w_mode_de: float = 0.55
    entropy_w_recon: float = 0.18
    entropy_w_residual_de: float = 0.12
    entropy_w_overlap: float = 0.08
    entropy_w_k: float = 0.07
    entropy_recon_threshold: float = 0.03
    entropy_recon_penalty: float = 6.0

    # decorrelation (exact Gram-Schmidt projection when enabled)
    # corr_* are kept for backward compatibility with older configs.
    enforce_uncorrelated: bool = True
    corr_rho: float = 0.05  # dual ascent step
    corr_update_every: int = 5  # update Gamma every N iterations
    corr_ema: float = 0.8  # EMA smoothing for corr estimates
    corr_floor: float = 1e-12
    use_anderson: bool = False
    gram_schmidt_every: int = 0


@dataclass
class HierarchicalParameters:
    max_levels: int = 3
    energy_threshold: float = 0.01
    min_samples_per_level: int = 100

    # downsampling behavior
    use_anti_aliasing_level_0: bool = True
    use_anti_aliasing_higher_levels: bool = False
    min_samples_for_fir_decimation: int = 600

    # hybrid
    use_emd_hybrid: bool = False
