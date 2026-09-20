"""foreblocks.forecasting.popular.

Widely-used time series forecasting architectures.

Implements DLinear, Informer, Autoformer, PatchTST, N-BEATS, N-HiTS,
FEDformer, ETSformer, TimesNet, CrossFormer, TimeMixer, TimeXer, TFT,
and non-stationary transformer variants. These are production-ready
implementations of state-of-the-art forecasting models.

Core API:
- DLinear: linear baseline with trend-seasonal decomposition
- Informer: ProbSparse transformer for long sequences
- PatchTST: patch-based transformer (ICLR 2023)
- Autoformer: auto-correlation based transformer
- NBEATS, NBEATSx, NBEATSInterpretable: decomposition-based forecasters
- NHiTS: hierarchical interpolation network
- TimesNet: 2D variation for time series
- CrossFormer: cross-scale transformer
- TimeMixer, TimeXer: time-domain mixing and cross-variable modeling
- TemporalFusionTransformer: multi-horizon attention model
- NonStationaryTransformer, NonStationaryWrapper: non-stationary aware models
- OryxMixerBlock, OryxTransformer: mixer architectures

"""

from foreblocks.forecasting.popular.autoformer import Autoformer
from foreblocks.forecasting.popular.crossformer import CrossFormer
from foreblocks.forecasting.popular.dlinear import DLinear
from foreblocks.forecasting.popular.etsformer import ETSformer
from foreblocks.forecasting.popular.fedformer import FEDformer
from foreblocks.forecasting.popular.informer import Informer
from foreblocks.forecasting.popular.itransformer import ITransformer
from foreblocks.forecasting.popular.nbeats import NBEATS, NBEATSInterpretable, NBEATSx
from foreblocks.forecasting.popular.nhits import NHiTS
from foreblocks.forecasting.popular.nonstationary import (
    NonStationaryTransformer,
    NonStationaryWrapper,
)
from foreblocks.forecasting.popular.oryx import OryxMixerBlock, OryxTransformer
from foreblocks.forecasting.popular.patch import PatchTST
from foreblocks.forecasting.popular.tft import TemporalFusionTransformer
from foreblocks.forecasting.popular.timemixer import TimeMixer
from foreblocks.forecasting.popular.timesnet import TimesNet
from foreblocks.forecasting.popular.timexer import TimeXer

__all__ = [
    "Autoformer",
    "DLinear",
    "FEDformer",
    "Informer",
    "ITransformer",
    "NBEATS",
    "NBEATSInterpretable",
    "NBEATSx",
    "NHiTS",
    "NonStationaryTransformer",
    "NonStationaryWrapper",
    "OryxMixerBlock",
    "OryxTransformer",
    "PatchTST",
    "TemporalFusionTransformer",
    "TimesNet",
    "CrossFormer",
    "ETSformer",
    "TimeMixer",
    "TimeXer",
]
