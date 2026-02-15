"""Compatibility module aggregating all time-series heads.

Implementation now lives in per-head modules under this package.
"""

from .chronos2_embed_head import Chronos2EmbedHead
from .dain_head import DAIN, DAINHead
from .decomposition_head import DecompositionBlock, DecompositionHead
from .differencing_head import Differencing, DifferencingHead
from .dropoutts_head import DropoutTSHead
from .fft_topk_head import FFTTopK, FFTTopKHead
from .haar_wavelet_topk_head import HaarWaveletTopK, HaarWaveletTopKHead
from .learnable_fourier_seasonal_head import (
    LearnableFourierSeasonal,
    LearnableFourierSeasonalHead,
)
from .multiscale_conv_head import MultiScaleConv, MultiScaleConvHead
from .patch_embed_head import PatchEmbed, PatchEmbedHead
from .revin_head import RevIN, RevINHead
from .time2vec_head import Time2Vec, Time2VecHead
from .time_attention_head import TimeAttention, TimeAttentionHead

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
    "PatchEmbed",
    "PatchEmbedHead",
    "RevIN",
    "RevINHead",
    "Time2Vec",
    "Time2VecHead",
    "TimeAttention",
    "TimeAttentionHead",
]
