"""foreblocks.modules.heads.modules.

Reusable head modules for time-series preprocessing and feature transformation.

Provides sequence-preserving head modules including wavelet decomposition,
Fourier seasonal extraction, multi-kernel convolution, multiscale pyramid,
patch embedding, RevIN normalization, time2vec encoding, and per-feature
time attention. Use as preprocessing or feature-engineering layers in
forecasting transformer architectures.

Core API:
- HaarWaveletTopK, HaarWaveletTopKHead: wavelet analysis with sparse detail
- LearnableFourierSeasonal, LearnableFourierSeasonalHead: Fourier seasonal decomposition
- MultiKernelConvHead: multi-kernel depthwise conv
- MultiScaleConv, MultiScaleConvHead: multiscale pyramid with spectral filtering
- PatchEmbed, PatchEmbedHead: depthwise patch embedding
- RevIN, RevINHead: reversible instance normalization
- Time2Vec, Time2VecHead: periodic temporal encoding
- TimeAttention, TimeAttentionHead: per-feature time Transformer
- DAIN, DAINHead: deep attention interpolation
- DecompositionBlock, DecompositionHead: trend-seasonality decomposition
- Differencing, DifferencingHead: reversible differencing
- DropoutTSHead: dropout-based temporal series regularization
- FFTTopK, FFTTopKHead: FFT-based sparse decomposition
- Chronos2EmbedHead: Chronos-2 style embeddings

"""

from foreblocks.modules.heads.modules.chronos2_embed_head import Chronos2EmbedHead
from foreblocks.modules.heads.modules.dain_head import DAIN, DAINHead
from foreblocks.modules.heads.modules.decomposition_head import (
    DecompositionBlock,
    DecompositionHead,
)
from foreblocks.modules.heads.modules.differencing_head import (
    Differencing,
    DifferencingHead,
)
from foreblocks.modules.heads.modules.dropoutts_head import DropoutTSHead
from foreblocks.modules.heads.modules.fft_topk_head import FFTTopK, FFTTopKHead
from foreblocks.modules.heads.modules.haar_wavelet_topk_head import (
    HaarWaveletTopK,
    HaarWaveletTopKHead,
)
from foreblocks.modules.heads.modules.learnable_fourier_seasonal_head import (
    LearnableFourierSeasonal,
    LearnableFourierSeasonalHead,
)
from foreblocks.modules.heads.modules.multikernel_conv_head import MultiKernelConvHead
from foreblocks.modules.heads.modules.multiscale_conv_head import (
    MultiScaleConv,
    MultiScaleConvHead,
)
from foreblocks.modules.heads.modules.patch_embed_head import PatchEmbed, PatchEmbedHead
from foreblocks.modules.heads.modules.revin_head import RevIN, RevINHead
from foreblocks.modules.heads.modules.time2vec_head import Time2Vec, Time2VecHead
from foreblocks.modules.heads.modules.time_attention_head import (
    TimeAttention,
    TimeAttentionHead,
)

__all__ = [
    "DAIN",
    "Chronos2EmbedHead",
    "DAINHead",
    "DecompositionBlock",
    "DecompositionHead",
    "Differencing",
    "DifferencingHead",
    "DropoutTSHead",
    "FFTTopK",
    "FFTTopKHead",
    "HaarWaveletTopK",
    "HaarWaveletTopKHead",
    "LearnableFourierSeasonal",
    "LearnableFourierSeasonalHead",
    "MultiKernelConvHead",
    "MultiScaleConv",
    "MultiScaleConvHead",
    "PatchEmbed",
    "PatchEmbedHead",
    "RevIN",
    "RevINHead",
    "Time2Vec",
    "Time2VecHead",
    "TimeAttention",
    "TimeAttentionHead",
]
