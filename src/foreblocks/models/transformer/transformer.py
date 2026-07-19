"""foreblocks.models.transformer.transformer.

Re-exports core transformer encoder/decoder components.

Provides full encoder, decoder, and layer abstractions for building transformer-based
time series models. Layers include multi-head attention (with MHC hyper-connection
support), norm wrappers, residual blocks, and MoE variants.

Core API:
- TransformerEncoder: full transformer encoder stack
- TransformerDecoder: full transformer decoder stack
- TransformerEncoderLayer: encoder layer with attention + FFN
- TransformerDecoderLayer: decoder layer with cross-attention
- BaseTransformerLayer: base transformer layer with norm + residual support

"""

from foreblocks.models.transformer.core.base import (
    BaseTransformer,
    BaseTransformerLayer,
)
from foreblocks.models.transformer.runtime.execution import (
    MHCBlockMixin,
    NormWrapper,
    ResidualBlockMixin,
    ResidualRunCfg,
)
from foreblocks.models.transformer.core.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from foreblocks.models.transformer.core.encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from foreblocks.models.transformer.runtime.outputs import (
    TransformerDecoderOutput,
    TransformerEncoderOutput,
)

__all__ = [
    "NormWrapper",
    "ResidualRunCfg",
    "ResidualBlockMixin",
    "MHCBlockMixin",
    "BaseTransformerLayer",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "BaseTransformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerDecoderOutput",
    "TransformerEncoderOutput",
]
