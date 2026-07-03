"""foreblocks.layers.embeddings.

Positional, rotary, and time embedding layers for sequence models.

Provides sinusoidal and learnable positional encoding, calendar time embedding,
RoPE with Triton acceleration, ALiBi positional bias, and helper functions
for applying positional encodings to attention tensors.

Core API:
- PositionalEncoding: sinusoidal positional encoding
- LearnablePositionalEncoding: learnable absolute position embeddings
- InformerTimeEmbedding: calendar/time feature embedding
- RotaryEmbedding: rotary position embeddings
- apply_rotary_emb: apply RoPE to tensors

"""

from foreblocks.layers.embeddings.informer_time_embedding import InformerTimeEmbedding
from foreblocks.layers.embeddings.learnable_positional_encoding import (
    LearnablePositionalEncoding,
)
from foreblocks.layers.embeddings.positional_encoding import PositionalEncoding
from foreblocks.layers.embeddings.rotary import RotaryEmbedding, apply_rotary_emb

__all__ = [
    "PositionalEncoding",
    "LearnablePositionalEncoding",
    "InformerTimeEmbedding",
    "RotaryEmbedding",
    "apply_rotary_emb",
]
