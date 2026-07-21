"""foreblocks.modules.heads.heads.

Aggregated re-exports of all foreblocks time-series heads.

Convenience import for all head implementations: DAIN, decomposition,
differencing, DropoutTS, FFT top-K, Haar wavelet top-K, learnable Fourier
seasonal, multi-scale/conv, patch embedding, RevIN, Time2Vec, and
TimeAttention. Use for quick imports without navigating subpackages.

Core API (re-exports):
- DAIN / DAINHead: deep attention interpolation head
- DecompositionBlock / DecompositionHead: trend-seasonal decomposition
- Differencing / DifferencingHead: differencing-based head
- DropoutTSHead: dropout-based time series head
- FFTTopK / FFTTopKHead: FFT top-K frequency selection head
- HaarWaveletTopK / HaarWaveletTopKHead: Haar wavelet top-K head
- LearnableFourierSeasonal / LearnableFourierSeasonalHead: learnable Fourier seasonal
- MultiScaleConv / MultiScaleConvHead: multi-scale convolution head
- MultiKernelConvHead: multi-kernel convolution head
- PatchEmbed / PatchEmbedHead: patch embedding head
- RevIN / RevINHead: reversible instance normalization
- Time2Vec / Time2VecHead: time embedding layer
- TimeAttention / TimeAttentionHead: time attention head
- Chronos2EmbedHead: Chronos 2 embedding head

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
