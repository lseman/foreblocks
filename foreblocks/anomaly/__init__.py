"""foreblocks.anomaly.

Unified anomaly detection framework: forecasting + reconstruction + representation.

Anomaly provides a modular anomaly-detection pipeline that supports multiple
detection strategies (forecasting residuals, reconstruction error, learned
representations) with composable backends (Mamba, Transformer, iTransformer,
graph models). Includes confidence calibration (Platt scaling, temperature
scaling, isotonic regression) to map raw scores to reliable uncertainty estimates."""

from foreblocks.anomaly.calibration import (
    ConfidenceResult,
    EnsembleScoreCombiner,
    PlattScaler,
    TemperatureScaler,
    compute_confidence,
    fit_score_distribution,
    isotonic_calibrate,
)
from foreblocks.anomaly.detector import (
    AnomalyDetectorConfig,
    AnomalyResult,
    ForeblocksAnomalyDetector,
)
from foreblocks.anomaly.modes import (
    AnomalyBlock,
    AnomalyBlockSpec,
    AnomalyBlockStack,
    AnomalyDecisionResult,
    DecisionConfig,
    ForecastingMode,
    HybridMode,
    ReconstructionMode,
    RepresentationMode,
    iTransformerMode,
    PatchMambaMode,
    register_block,
    resolve_block,
    list_blocks,
    resolve_mode,
)
from foreblocks.anomaly.models import (
    AnomalyTransformer,
    DAGMM,
    MLPVAE,
    OmniAnomaly,
    PatchMamba,
    TransformerVAE,
    iTransformer,
)
from foreblocks.anomaly.online import (
    BNAdaptiveWrapper,
    EMAStatistics,
    StreamingAnomalyDetector,
    TENTAdapter,
)
from foreblocks.anomaly.tranad import (
    TranAD,
    TranADDataset,
    TranADDetector,
    create_sequences_vectorized,
)
from foreblocks.anomaly.windows import (
    build_sliding_windows,
    map_window_scores,
    robust_threshold,
)

__all__ = [
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
