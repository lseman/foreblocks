"""foreblocks.models.anomaly.backbones.

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

from foreblocks.models.anomaly.backbones.anomaly_transformer import (
    AnomalyTransformer,
    AnomalyTransformerForward,
    association_discrepancy,
)
from foreblocks.models.anomaly.backbones.base import (
    ForeblocksEncoderStack,
    VAEForward,
    choose_heads,
)
from foreblocks.models.anomaly.backbones.dagmm import DAGMM, DAGMMForward
from foreblocks.models.anomaly.backbones.diffusion import (
    DiffusionAnomaly,
    DiffusionAnomalyForward,
)
from foreblocks.models.anomaly.backbones.forecasting import TransformerForecaster
from foreblocks.models.anomaly.backbones.frequency import (
    FrequencyAnomaly,
    LogFreqAnomaly,
)
from foreblocks.models.anomaly.backbones.omni_anomaly import OmniAnomaly
from foreblocks.models.anomaly.backbones.patch_tst import (
    CrossVarTransformer,
    MaskedForecaster,
    PatchTSTForecaster,
)
from foreblocks.models.anomaly.backbones.reconstruction import MLPVAE, TransformerVAE
from foreblocks.models.anomaly.backbones.representation import (
    ContrastiveTransformerEncoder,
)
from foreblocks.models.anomaly.backbones.state_space import (
    PatchMamba,
    PatchMambaForward,
    S6Block,
    iTransformer,
    iTransformerForward,
)
from foreblocks.models.anomaly.backbones.tranad import TranAD

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
