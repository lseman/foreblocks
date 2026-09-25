"""foreblocks.models.anomaly.

Unified anomaly detection framework: forecasting + reconstruction + representation
+ classical + statistical methods.

Provides a modular anomaly-detection pipeline that supports multiple
detection strategies (forecasting residuals, reconstruction error, learned
representations, classical isolation/density, statistical control charts, and
seasonal decomposition) with composable backends (Mamba, Transformer, iTransformer,
graph models). Includes confidence calibration (Platt scaling, temperature
scaling, isotonic regression) to map raw scores to reliable uncertainty estimates.

Core API:
- ForeblocksAnomalyDetector: unified anomaly detection pipeline
- AnomalyDetectorConfig: configuration for anomaly detection
- AnomalyResult, AnomalyDecisionResult: detection result types
- ForecastingMode, ReconstructionMode, RepresentationMode, HybridMode, PatchMambaMode,
  iTransformerMode, ClassicalMode, StatisticalMode: detection modes
- AnomalyBlock, AnomalyBlockSpec, AnomalyBlockStack: modular block composition
- isolation_forest_score, lof_score, pca_mahalanobis_score, matrix_profile_score:
  classical anomaly scoring functions
- ebs_score, cusum_score, ewma_score, seasonal_hybrid_score, stl_residual_score:
  statistical anomaly scoring functions
- TemperatureScaler, PlattScaler, EnsembleScoreCombiner, isotonic_calibrate,
  compute_confidence, fit_score_distribution, ConfidenceResult: confidence calibration
- StreamingAnomalyDetector, TENTAdapter, BNAdaptiveWrapper, EMAStatistics:
  online/streaming anomaly detection

"""

from foreblocks.models.anomaly.backbones import (
    DAGMM,
    MLPVAE,
    AnomalyTransformer,
    OmniAnomaly,
    PatchMamba,
    TransformerVAE,
    iTransformer,
)
from foreblocks.models.anomaly.blocks import (
    AnomalyBlock,
    AnomalyBlockSpec,
    AnomalyBlockStack,
    AnomalyDecisionResult,
    DecisionConfig,
    list_blocks,
    register_block,
    resolve_block,
)
from foreblocks.models.anomaly.calibration import (
    ConfidenceResult,
    EnsembleScoreCombiner,
    PlattScaler,
    TemperatureScaler,
    compute_confidence,
    fit_score_distribution,
    isotonic_calibrate,
)
from foreblocks.models.anomaly.config import AnomalyDetectorConfig
from foreblocks.models.anomaly.detector import (
    AnomalyResult,
    ForeblocksAnomalyDetector,
)
from foreblocks.models.anomaly.modes import (
    ClassicalMode,
    ForecastingMode,
    HybridMode,
    NativeMode,
    PatchMambaMode,
    ReconstructionMode,
    RepresentationMode,
    StatisticalMode,
    iTransformerMode,
    resolve_mode,
)
from foreblocks.models.anomaly.online import (
    BNAdaptiveWrapper,
    EMAStatistics,
    StreamingAnomalyDetector,
    TENTAdapter,
)
from foreblocks.models.anomaly.scorers import (
    cusum_score,
    ebs_score,
    ewma_score,
    isolation_forest_score,
    lof_score,
    matrix_profile_score,
    pca_mahalanobis_score,
    seasonal_hybrid_score,
    stl_residual_score,
)
from foreblocks.models.anomaly.scorers.empirical import (
    COPOD,
    ECOD,
    HBOS,
    copod_score,
    ecod_score,
    hbos_score,
)
from foreblocks.models.anomaly.scorers.native import (
    INNE,
    LODA,
    NATIVE_MODELS,
    AutoEncoderScorer,
    DeepIsolationForest,
    DeepSVDD,
    GaussianMixtureScorer,
    KNNScorer,
    VAEScorer,
)
from foreblocks.models.anomaly.tranad_detector import (
    TranAD,
    TranADDataset,
    TranADDetector,
    create_sequences_vectorized,
)
from foreblocks.models.anomaly.windows import (
    build_sliding_windows,
    map_window_scores,
    robust_threshold,
)

__all__ = [
    "NATIVE_MODELS",
    "INNE",
    "LODA",
    "KNNScorer",
    "GaussianMixtureScorer",
    "DeepIsolationForest",
    "AutoEncoderScorer",
    "VAEScorer",
    "DeepSVDD",
    "ECOD",
    "COPOD",
    "HBOS",
    "ecod_score",
    "copod_score",
    "hbos_score",
    "AnomalyDetectorConfig",
    "AnomalyResult",
    "AnomalyDecisionResult",
    "ForeblocksAnomalyDetector",
    "ForecastingMode",
    "ReconstructionMode",
    "RepresentationMode",
    "HybridMode",
    "PatchMambaMode",
    "iTransformerMode",
    "ClassicalMode",
    "NativeMode",
    "StatisticalMode",
    "AnomalyBlock",
    "AnomalyBlockSpec",
    "AnomalyBlockStack",
    "DecisionConfig",
    "register_block",
    "resolve_block",
    "list_blocks",
    "resolve_mode",
    "AnomalyTransformer",
    "DAGMM",
    "MLPVAE",
    "OmniAnomaly",
    "TransformerVAE",
    "PatchMamba",
    "iTransformer",
    "isolation_forest_score",
    "lof_score",
    "pca_mahalanobis_score",
    "matrix_profile_score",
    "ebs_score",
    "cusum_score",
    "ewma_score",
    "seasonal_hybrid_score",
    "stl_residual_score",
    "TranAD",
    "TranADDataset",
    "TranADDetector",
    "create_sequences_vectorized",
    "build_sliding_windows",
    "map_window_scores",
    "robust_threshold",
    "TemperatureScaler",
    "PlattScaler",
    "EnsembleScoreCombiner",
    "isotonic_calibrate",
    "compute_confidence",
    "fit_score_distribution",
    "ConfidenceResult",
    "StreamingAnomalyDetector",
    "TENTAdapter",
    "BNAdaptiveWrapper",
    "EMAStatistics",
]
