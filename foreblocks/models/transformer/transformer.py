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

from foreblocks.models.transformer.tf_base import (
    BaseTransformer,
    BaseTransformerLayer,
    MHCBlockMixin,
    NormWrapper,
    ResidualBlockMixin,
    ResidualRunCfg,
)
from foreblocks.models.transformer.tf_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from foreblocks.models.transformer.tf_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
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
]
