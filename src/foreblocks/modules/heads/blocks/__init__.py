"""foreblocks.modules.heads.blocks.

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

from foreblocks.modules.heads.blocks.chronos2_embed_head import Chronos2EmbedHead
from foreblocks.modules.heads.blocks.dain_head import DAIN, DAINHead
from foreblocks.modules.heads.blocks.decomposition_head import (
    DecompositionBlock,
    DecompositionHead,
)
from foreblocks.modules.heads.blocks.differencing_head import (
    Differencing,
    DifferencingHead,
)
from foreblocks.modules.heads.blocks.dropoutts_head import DropoutTSHead
from foreblocks.modules.heads.blocks.fft_topk_head import FFTTopK, FFTTopKHead
from foreblocks.modules.heads.blocks.haar_wavelet_topk_head import (
    HaarWaveletTopK,
    HaarWaveletTopKHead,
)
from foreblocks.modules.heads.blocks.learnable_fourier_seasonal_head import (
    LearnableFourierSeasonal,
    LearnableFourierSeasonalHead,
)
from foreblocks.modules.heads.blocks.multikernel_conv_head import MultiKernelConvHead
from foreblocks.modules.heads.blocks.multiscale_conv_head import (
    MultiScaleConv,
    MultiScaleConvHead,
)
from foreblocks.modules.heads.blocks.patch_embed_head import PatchEmbed, PatchEmbedHead
from foreblocks.modules.heads.blocks.revin_head import RevIN, RevINHead
from foreblocks.modules.heads.blocks.time2vec_head import Time2Vec, Time2VecHead
from foreblocks.modules.heads.blocks.time_attention_head import (
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
