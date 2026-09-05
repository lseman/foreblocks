"""foreblocks.modules.heads.

Composable forecasting head pipeline with serial, parallel, and hybrid staging.

For the head engine (HeadComposer, HeadGraph), import from
``foreblocks.modules.heads.engine``.
For head implementations (RevIN, PatchEmbed, etc.), import from
``foreblocks.modules.heads.blocks``.

Top-level re-exports:
- HeadComposer: serial/parallel/hybrid head composition
- HeadGraph: graph-based head orchestration
- HeadOutput, HeadShape: tensor contracts
- HeadSpec, HeadStage: head/ stage declarations
- DAIN, RevIN, PatchEmbed, HaarWaveletTopK, etc.: head implementations
- HeadComposerConfig, StageKind, NASMode: configuration

"""

from foreblocks.modules.heads.config import (
    AlignmentMode,
    HeadComposerConfig,
    HeadNASConfig,
    NASMode,
    ParallelFusion,
    ParallelStageConfig,
    SerialMerge,
    SerialStageConfig,
    StageKind,
    StructuredOutputPolicy,
)
from foreblocks.modules.heads.core.contracts import (
    HeadOutput,
    HeadShape,
)
from foreblocks.modules.heads.core.types import HeadSpec
from foreblocks.modules.heads.engine.composer import HeadComposer
from foreblocks.modules.heads.engine.execution import (
    build_stage_composer,
    execute_stage,
)
from foreblocks.modules.heads.engine.graph import (
    HeadGraph,
    HeadGraphState,
    HeadStage,
)
from foreblocks.modules.heads.blocks import (
    Chronos2EmbedHead,
    DAIN,
    DAINHead,
    DecompositionBlock,
    DecompositionHead,
    Differencing,
    DifferencingHead,
    DropoutTSHead,
    FFTTopK,
    FFTTopKHead,
    HaarWaveletTopK,
    HaarWaveletTopKHead,
    LearnableFourierSeasonal,
    LearnableFourierSeasonalHead,
    MultiKernelConvHead,
    MultiScaleConv,
    MultiScaleConvHead,
    PatchEmbed,
    PatchEmbedHead,
    RevIN,
    RevINHead,
    Time2Vec,
    Time2VecHead,
    TimeAttention,
    TimeAttentionHead,
)

__all__ = [
    "DAIN",
    "AlignmentMode",
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
    "HeadComposer",
    "HeadComposerConfig",
    "HeadGraph",
    "HeadGraphState",
    "HeadNASConfig",
    "HeadOutput",
    "HeadShape",
    "HeadSpec",
    "HeadStage",
    "LearnableFourierSeasonal",
    "LearnableFourierSeasonalHead",
    "MultiKernelConvHead",
    "MultiScaleConv",
    "MultiScaleConvHead",
    "NASMode",
    "ParallelFusion",
    "ParallelStageConfig",
    "PatchEmbed",
    "PatchEmbedHead",
    "RevIN",
    "RevINHead",
    "SerialMerge",
    "SerialStageConfig",
    "StageKind",
    "StructuredOutputPolicy",
    "Time2Vec",
    "Time2VecHead",
    "TimeAttention",
    "TimeAttentionHead",
    "build_stage_composer",
    "execute_stage",
]
