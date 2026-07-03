"""Model backbones for Foreblocks anomaly detection."""

from foreblocks.anomaly.models.anomaly_transformer import (
    AnomalyTransformer,
    AnomalyTransformerForward,
    association_discrepancy,
)
from foreblocks.anomaly.models.base import (
    ForeblocksEncoderStack,
    VAEForward,
    choose_heads,
)
from foreblocks.anomaly.models.dagmm import DAGMM, DAGMMForward
from foreblocks.anomaly.models.diffusion import (
    DiffusionAnomaly,
    DiffusionAnomalyForward,
)
from foreblocks.anomaly.models.forecasting import TransformerForecaster
from foreblocks.anomaly.models.frequency import (
    FrequencyAnomaly,
    LogFreqAnomaly,
)
from foreblocks.anomaly.models.omni_anomaly import OmniAnomaly
from foreblocks.anomaly.models.patch_tst import (
    CrossVarTransformer,
    MaskedForecaster,
    PatchTSTForecaster,
)
from foreblocks.anomaly.models.reconstruction import MLPVAE, TransformerVAE
from foreblocks.anomaly.models.representation import ContrastiveTransformerEncoder
from foreblocks.anomaly.models.state_space import (
    iTransformer,
    iTransformerForward,
    PatchMamba,
    PatchMambaForward,
    S6Block,
)
from foreblocks.anomaly.models.tranad import TranAD

__all__ = [
    "VAEForward",
    "choose_heads",
    "ForeblocksEncoderStack",
    "AnomalyTransformer",
    "AnomalyTransformerForward",
    "association_discrepancy",
    "DAGMM",
    "DAGMMForward",
    "DiffusionAnomaly",
    "DiffusionAnomalyForward",
    "FrequencyAnomaly",
    "LogFreqAnomaly",
    "OmniAnomaly",
    "MLPVAE",
    "TransformerVAE",
    "TransformerForecaster",
    "PatchTSTForecaster",
    "CrossVarTransformer",
    "MaskedForecaster",
    "ContrastiveTransformerEncoder",
    "PatchMamba",
    "PatchMambaForward",
    "iTransformer",
    "iTransformerForward",
    "S6Block",
    "TranAD",
]
