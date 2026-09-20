"""foreblocks.forecasting.anomaly.models.

Model backbones for Foreblocks anomaly detection.

Provides anomaly detection model implementations including VAE-based reconstruction
(MLPVAE, TransformerVAE), forecasting models (TransformerForecaster, TranAD),
representation learning (ContrastiveTransformerEncoder), graph autoencoders (DAGMM),
and state-space models (PatchMamba, iTransformer).

Core API:
- TransformerVAE, MLPVAE: VAE-based reconstruction models
- TransformerForecaster, TranAD: forecasting-based anomaly detection models
- ContrastiveTransformerEncoder: representation learning model
- AnomalyTransformer, OmniAnomaly, DAGMM: specialized anomaly detection architectures
- PatchMamba, iTransformer: state-space and inverted transformer models

"""

from foreblocks.forecasting.anomaly.models.anomaly_transformer import (
    AnomalyTransformer,
    AnomalyTransformerForward,
    association_discrepancy,
)
from foreblocks.forecasting.anomaly.models.base import (
    ForeblocksEncoderStack,
    VAEForward,
    choose_heads,
)
from foreblocks.forecasting.anomaly.models.dagmm import DAGMM, DAGMMForward
from foreblocks.forecasting.anomaly.models.diffusion import (
    DiffusionAnomaly,
    DiffusionAnomalyForward,
)
from foreblocks.forecasting.anomaly.models.forecasting import TransformerForecaster
from foreblocks.forecasting.anomaly.models.frequency import (
    FrequencyAnomaly,
    LogFreqAnomaly,
)
from foreblocks.forecasting.anomaly.models.omni_anomaly import OmniAnomaly
from foreblocks.forecasting.anomaly.models.patch_tst import (
    CrossVarTransformer,
    MaskedForecaster,
    PatchTSTForecaster,
)
from foreblocks.forecasting.anomaly.models.reconstruction import MLPVAE, TransformerVAE
from foreblocks.forecasting.anomaly.models.representation import ContrastiveTransformerEncoder
from foreblocks.forecasting.anomaly.models.state_space import (
    PatchMamba,
    PatchMambaForward,
    S6Block,
    iTransformer,
    iTransformerForward,
)
from foreblocks.forecasting.anomaly.models.tranad import TranAD

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
