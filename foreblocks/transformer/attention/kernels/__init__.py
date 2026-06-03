try:
    from .fla_backend import (
        fla_chunk_delta_rule,
        fla_fused_chunk_delta_rule,
        fla_fused_recurrent_delta_rule,
        fla_path,
        fla_rms_norm_gated,
        has_fla_checkout,
        import_fla_module,
        is_fla_available,
    )
except Exception:

    def fla_path(*args, **kwargs):  # type: ignore[misc]
        return None

    def has_fla_checkout(*args, **kwargs):  # type: ignore[misc]
        return False

    def import_fla_module(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention adapter is unavailable.")

    def is_fla_available(*args, **kwargs):  # type: ignore[misc]
        return False

    def fla_chunk_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_fused_chunk_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_fused_recurrent_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_rms_norm_gated(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")


try:
    from .chunked_causal_linear_attention import (
        can_use_fused_recurrent_linear_attn,
        chunked_causal_linear_attn,
        fused_recurrent_causal_linear_attn,
    )
except Exception:

    def can_use_fused_recurrent_linear_attn(*args, **kwargs):  # type: ignore[misc]
        return False

    def chunked_causal_linear_attn(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "chunked_causal_linear_attn cannot be used in this environment."
        )

    def fused_recurrent_causal_linear_attn(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "Triton is not available; fused_recurrent_causal_linear_attn cannot be used."
        )


try:
    from .delta_rule import (
        can_use_fla_recurrent_delta_rule,
        can_use_fused_recurrent_delta_rule,
        fla_recurrent_delta_rule,
        fused_recurrent_delta_rule,
    )
except Exception:

    def can_use_fla_recurrent_delta_rule(*args, **kwargs):  # type: ignore[misc]
        return False

    def can_use_fused_recurrent_delta_rule(*args, **kwargs):  # type: ignore[misc]
        return False

    def fla_recurrent_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "flash-linear-attention fused recurrent delta rule cannot be used."
        )

    def fused_recurrent_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "Triton is not available; fused_recurrent_delta_rule cannot be used."
        )


try:
    from .fused_norm_gate import (
        can_use_fused_rmsnorm_sigmoid_gate,
        fused_rmsnorm_sigmoid_gate,
    )
except Exception:

    def can_use_fused_rmsnorm_sigmoid_gate(*args, **kwargs):  # type: ignore[misc]
        return False

    def fused_rmsnorm_sigmoid_gate(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "Triton is not available; fused_rmsnorm_sigmoid_gate cannot be used."
        )


try:
    from .gdn2_chunk import can_use_gdn2_chunk, chunk_gdn2
except Exception:

    def can_use_gdn2_chunk(*args, **kwargs):  # type: ignore[misc]
        return False

    def chunk_gdn2(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not available; chunk_gdn2 cannot be used.")


try:
    from .gdn2_triton import can_use_gdn2_triton, gdn2_chunk_fwd_triton
except Exception:

    def can_use_gdn2_triton(*args, **kwargs):  # type: ignore[misc]
        return False

    def gdn2_chunk_fwd_triton(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "Triton is not available; gdn2_chunk_fwd_triton cannot be used."
        )


try:
    from .fused_rope import triton_apply_rope
except Exception:

    def triton_apply_rope(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not available; triton_apply_rope cannot be used.")


try:
    from .paged_decode import triton_paged_decode
except Exception:

    def triton_paged_decode(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "Triton is not available; triton_paged_decode cannot be used."
        )


__all__ = [
    "can_use_fused_recurrent_linear_attn",
    "fla_chunk_delta_rule",
    "fla_fused_chunk_delta_rule",
    "fla_fused_recurrent_delta_rule",
    "fla_path",
    "fla_rms_norm_gated",
    "has_fla_checkout",
    "import_fla_module",
    "is_fla_available",
    "chunked_causal_linear_attn",
    "can_use_fla_recurrent_delta_rule",
    "fused_recurrent_causal_linear_attn",
    "can_use_fused_recurrent_delta_rule",
    "fla_recurrent_delta_rule",
    "fused_recurrent_delta_rule",
    "can_use_fused_rmsnorm_sigmoid_gate",
    "fused_rmsnorm_sigmoid_gate",
    "can_use_gdn2_chunk",
    "chunk_gdn2",
    "can_use_gdn2_triton",
    "gdn2_chunk_fwd_triton",
    "triton_paged_decode",
    "triton_apply_rope",
]
