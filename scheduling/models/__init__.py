"""Neural scheduler models for the ONTS NCO framework.

Layer 1: Protocol ABCs (EncoderProtocol, DecoderProtocol)
Layer 2: Registries (EncoderRegistry, DecoderRegistry)
Layer 3: Composition (NCOModel, build_model)
Layer 4: Adapters (adapt_encoder, adapt_decoder)
Layer 5: ONTS-ready encoder/decoder implementations

Usage:
    from models import build_model
    model = build_model(encoder="bipartite", decoder="bipartite", d_model=128)

    # Or use existing classes directly:
    from models import ActorCritic
    model = ActorCritic(f_static=9, f_dynamic=9, f_global=4, encoder_type="transformer")
"""

# ── Layer 1 & 2: Protocols and Registries ──────────────────────────────

from .protocols import EncoderProtocol, DecoderProtocol, TrainerProtocol
from .nco_model import (
    EncoderRegistry,
    DecoderRegistry,
    NCOModel,
    build_model,
    register_builtin_models,
)

# ── Layer 3 & 4: Shared decode and adapters ────────────────────────────

from .shared_decode import shared_decode_loop
from .adapters import (
    TransformerEncoderAdapter,
    PointerDecoderAdapter,
    BipartiteDecoderAdapter,
    adapt_encoder,
    adapt_decoder,
)

# ── Layer 5: ONTS-ready implementations ────────────────────────────────

from .pointer_net import (
    TransformerEncoder,
    PointerDecoder,
    ActorCritic,
    UnifiedTransformerEncoder,
    UnifiedBipartiteEncoder,
    _DecoderAdapter,
)
from .bipartite_gnn import (
    BipartiteGNN,
    BipartiteDecoder,
    BipartiteGNNScheduler,
    build_bipartite_gnn,
)

# Backwards compat alias
TaskEncoder = TransformerEncoder

# ── Register built-in models on import ─────────────────────────────────

register_builtin_models()

__all__ = [
    # Layer 1: Protocols
    "EncoderProtocol",
    "DecoderProtocol",
    "TrainerProtocol",
    # Layer 2: Registries
    "EncoderRegistry",
    "DecoderRegistry",
    # Layer 3: Composition
    "NCOModel",
    "build_model",
    "register_builtin_models",
    # Layer 4: Adapters
    "shared_decode_loop",
    "TransformerEncoderAdapter",
    "PointerDecoderAdapter",
    "BipartiteDecoderAdapter",
    "adapt_encoder",
    "adapt_decoder",
    # Layer 5: Existing implementations
    "TransformerEncoder",
    "TaskEncoder",
    "PointerDecoder",
    "ActorCritic",
    "UnifiedTransformerEncoder",
    "UnifiedBipartiteEncoder",
    "_DecoderAdapter",
    "BipartiteGNN",
    "BipartiteDecoder",
    "BipartiteGNNScheduler",
    "build_bipartite_gnn",
]
