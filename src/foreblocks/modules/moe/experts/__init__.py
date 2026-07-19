"""foreblocks.modules.moe.experts.

Mixture-of-Experts (MoE) expert routing, dispatching, and expert-layer implementations.

Provides a complete MoE stack: routers (noisy-top-K, linear, hash-based, continuous,
straight-through, soft-dense, auxiliary-token), dispatchers (token-choice and
expert-choice), expert blocks (SwiGLU and FFN with configurable dropout), a full
dMoE feed-forward layer with multiple fast paths, logging/reporting utilities,
and a unified feed-forward wrapper. Use to add MoE layers to transformer
architectures.

Core API:
- NoisyTopKRouter, LinearRouter, HashTopKRouter, ContinuousTopKRouter: routing strategies
- DroplessPackedDispatcher, ExpertChoiceDispatcher: token packing dispatchers
- MoE_SwiGLUExpert, MoE_FFNExpert: expert FFN implementations
- MoEFeedForwardDMoE: full dMoE feed-forward with fast paths
- FeedForwardBlock: unified FFN/MoE wrapper
- MoELogger, attach_router_hook: MoE training diagnostics

"""

from foreblocks.modules.moe.experts import dispatchers, moe, moe_logging, routers
from foreblocks.modules.moe.experts.dispatchers import (
    ConfidenceCapacityDispatcher,
    DroplessPackedDispatcher,
    ExpertChoiceDispatcher,
)
from foreblocks.modules.moe.experts.expert_blocks import (
    MoE_FFNExpert,
    MoE_SwiGLUExpert,
    MTPHead,
)
from foreblocks.modules.moe.experts.moe import (
    MoEFeedForwardDMoE,
    MoERoutingState,
    eager_topk_routing,
    maybe_compile,
    optimized_topk_routing,
)
from foreblocks.modules.moe.experts.moe_logging import MoELogger, attach_router_hook
from foreblocks.modules.moe.experts.routers import (
    AdaptiveNoisyTopKRouter,
    ContinuousTopKRouter,
    HashTopKRouter,
    LinearRouter,
    NoisyTopKRouter,
    Router,
    StraightThroughTopKRouter,
)

__all__ = [
    "AdaptiveNoisyTopKRouter",
    "ContinuousTopKRouter",
    "ConfidenceCapacityDispatcher",
    "DroplessPackedDispatcher",
    "ExpertChoiceDispatcher",
    "HashTopKRouter",
    "LinearRouter",
    "MTPHead",
    "MoEFeedForwardDMoE",
    "MoERoutingState",
    "MoELogger",
    "MoE_FFNExpert",
    "MoE_SwiGLUExpert",
    "NoisyTopKRouter",
    "Router",
    "StraightThroughTopKRouter",
    "attach_router_hook",
    "dispatchers",
    "eager_topk_routing",
    "moe",
    "moe_logging",
    "maybe_compile",
    "optimized_topk_routing",
    "routers",
]
