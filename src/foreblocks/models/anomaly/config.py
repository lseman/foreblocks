"""Configuration for the unified anomaly detector."""

from dataclasses import dataclass, field
from typing import Literal

from foreblocks.models.anomaly.blocks import AnomalyBlockSpec


@dataclass
class AnomalyDetectorConfig:
    detection_mode: Literal[
        "auto",
        "forecasting",
        "reconstruction",
        "representation",
        "hybrid",
        "classical",
        "native",
        "statistical",
        "patch_mamba",
        "i_transformer",
    ] = "auto"
    model_type: Literal[
        "transformer_vae",
        "mlp_vae",
        "omni_anomaly",
        "anomaly_transformer",
        "dagmm",
        "tranad",
        "patch_mamba",
        "i_transformer",
        "inne",
        "loda",
        "knn",
        "gmm",
        "dif",
        "autoencoder",
        "vae",
        "deep_svdd",
        "ecod",
        "copod",
        "hbos",
        "isolation_forest",
        "lof",
        "pca_mahalanobis",
        "matrix_profile",
        "ebs",
        "cusum",
        "ewma",
        "seasonal_hybrid",
        "stl_residual",
    ] = "transformer_vae"
    scorer_kwargs: dict = field(default_factory=dict)
    hbos_bins: int = 10
    hbos_alpha: float = 0.1
    block_stack: list[str | AnomalyBlockSpec] | None = None
    decision_strategy: Literal["majority", "weighted", "all", "any"] = "majority"
    window_size: int = 32
    contamination: float = 0.01
    decision_contamination: float = 0.01
    score_align: Literal["end", "center", "all"] = "end"
    d_model: int = 128
    latent_size: int = 32
    hidden_size: int = 128
    n_heads: int | None = None
    n_layers: int = 2
    dim_feedforward: int | None = None
    layer_attention_type: str = "standard"
    projection_size: int = 64
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    beta: float = 0.05
    beta_warmup_epochs: int = 5
    patience: int = 5
    scaler_type: Literal["robust", "standard", "minmax"] = "robust"
    device: str | None = None
    num_workers: int = 0
    use_mixed_precision: bool = True
    gradient_clip: float = 1.0
    seed: int | None = 42
    contrastive_temperature: float = 0.2
    augmentation_noise_std: float = 0.05
    reconstruction_weight: float = 1.0
    forecasting_weight: float = 1.0
    representation_weight: float = 0.25
    association_weight: float = 0.1
    energy_weight: float = 0.1
    covariance_weight: float = 0.005
    gmm_components: int = 4
    decision_weights: dict[str, float] | None = None
