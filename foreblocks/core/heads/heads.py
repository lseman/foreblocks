"""Compatibility module aggregating all time-series heads.

Implementation now lives in per-head modules under this package.
"""

from .modules.chronos2_embed_head import Chronos2EmbedHead
from .modules.dain_head import DAIN
from .modules.dain_head import DAINHead
from .modules.decomposition_head import DecompositionBlock
from .modules.decomposition_head import DecompositionHead
from .modules.differencing_head import Differencing
from .modules.differencing_head import DifferencingHead
from .modules.dropoutts_head import DropoutTSHead
from .modules.fft_topk_head import FFTTopK
from .modules.fft_topk_head import FFTTopKHead
from .modules.haar_wavelet_topk_head import HaarWaveletTopK
from .modules.haar_wavelet_topk_head import HaarWaveletTopKHead
from .modules.learnable_fourier_seasonal_head import LearnableFourierSeasonal
from .modules.learnable_fourier_seasonal_head import LearnableFourierSeasonalHead
from .modules.multikernel_conv_head import MultiKernelConvHead
from .modules.multiscale_conv_head import MultiScaleConv
from .modules.multiscale_conv_head import MultiScaleConvHead
from .modules.patch_embed_head import PatchEmbed
from .modules.patch_embed_head import PatchEmbedHead
from .modules.revin_head import RevIN
from .modules.revin_head import RevINHead
from .modules.time2vec_head import Time2Vec
from .modules.time2vec_head import Time2VecHead
from .modules.time_attention_head import TimeAttention
from .modules.time_attention_head import TimeAttentionHead


__all__ = [
    "Chronos2EmbedHead",
    "DAIN",
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
    "MultiScaleConv",
    "MultiScaleConvHead",
    "MultiKernelConvHead",
    "PatchEmbed",
    "PatchEmbedHead",
    "RevIN",
    "RevINHead",
    "Time2Vec",
    "Time2VecHead",
    "TimeAttention",
    "TimeAttentionHead",
]
