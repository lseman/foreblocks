from .group_norm import ChannelLastGroupNorm
from .layer_norm import AdaptiveLayerNorm, FastLayerNorm
from .revin import RevIN
from .rms_norm import AdaptiveRMSNorm, RMSNorm
from .temporal_norm import TemporalNorm

__all__ = [
    "AdaptiveLayerNorm",
    "AdaptiveRMSNorm",
    "ChannelLastGroupNorm",
    "create_norm_layer",
    "FastLayerNorm",
    "RMSNorm",
    "RevIN",
    "TemporalNorm",
]


def create_norm_layer(
    norm_type: str,
    d_model: int,
    eps: float = 1e-5,
    **kwargs,
):
    norm_type = norm_type.lower().replace("_", "").replace("-", "")
    if norm_type in ("layer", "layernorm"):
        return FastLayerNorm(d_model, eps=eps, **kwargs)
    if norm_type in ("temporal", "temporalnorm"):
        return TemporalNorm(d_model, eps=eps, **kwargs)
    if norm_type in ("revin",):
        affine = kwargs.pop("affine", True)
        return RevIN(d_model, affine=affine, eps=eps)
    if norm_type in ("rms", "rmsnorm"):
        return RMSNorm(d_model, eps=eps, **kwargs)
    if norm_type in ("adaptivelayer", "adaptivelayernorm"):
        return AdaptiveLayerNorm(d_model, eps=eps, **kwargs)
    if norm_type in ("adaptiverms", "adaptivermsnorm"):
        return AdaptiveRMSNorm(d_model, eps=eps, **kwargs)
    if norm_type in ("group", "groupnorm"):
        num_groups = kwargs.pop("num_groups", 32)
        return ChannelLastGroupNorm(num_groups, d_model, eps=eps, **kwargs)
    raise ValueError(
        f"Unsupported norm type: {norm_type}. "
        "Supported types: layer, rms, adaptive_layer, adaptive_rms, group, temporal, revin"
    )
