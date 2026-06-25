try:
    from foreblocks.ops.attention.fla_backend import (
        fla_chunk_delta_rule,
        fla_chunk_gated_delta_rule,
        fla_chunk_gdn2,
        fla_chunk_gla,
        fla_chunk_kda,
        fla_chunk_linear_attn,
        fla_fused_chunk_delta_rule,
        fla_fused_recurrent_delta_rule,
        fla_fused_recurrent_gated_delta_rule,
        fla_fused_recurrent_gdn2,
        fla_fused_recurrent_gla,
        fla_fused_recurrent_kda,
        fla_fused_recurrent_linear_attn,
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

    def fla_chunk_gla(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_fused_recurrent_gla(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_chunk_linear_attn(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_fused_recurrent_linear_attn(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_chunk_gated_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_fused_recurrent_gated_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_chunk_gdn2(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_fused_recurrent_gdn2(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_chunk_kda(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_fused_recurrent_kda(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_fused_recurrent_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")

    def fla_rms_norm_gated(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention is unavailable.")


try:
    from foreblocks.ops.attention.chunked_causal_linear_attention import (
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
    from foreblocks.ops.attention.fla_delta_rule import (
        can_use_fla_delta_rule,
        can_use_fla_recurrent_delta_rule,
        fla_delta_rule_forward,
        fla_recurrent_delta_rule,
    )
except Exception:

    def can_use_fla_delta_rule(*args, **kwargs):  # type: ignore[misc]
        return False

    def can_use_fla_recurrent_delta_rule(*args, **kwargs):  # type: ignore[misc]
        return False

    def fla_delta_rule_forward(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention delta rule cannot be used.")

    def fla_recurrent_delta_rule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "flash-linear-attention fused recurrent delta rule cannot be used."
        )


try:
    from foreblocks.ops.attention.fused_norm_gate import (
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
    from foreblocks.ops.attention.fla_gated_delta_rule import (
        can_use_fla_gated_delta_rule,
        fla_gated_delta_rule_forward,
    )
except Exception:

    def can_use_fla_gated_delta_rule(*args, **kwargs):  # type: ignore[misc]
        return False

    def fla_gated_delta_rule_forward(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention gated_delta_rule cannot be used.")


try:
    from foreblocks.ops.attention.fla_linear_attention import (
        can_use_fla_linear_attn,
        fla_recurrent_linear_attn_forward,
    )
except Exception:

    def can_use_fla_linear_attn(*args, **kwargs):  # type: ignore[misc]
        return False

    def fla_recurrent_linear_attn_forward(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention linear_attn cannot be used.")


try:
    from foreblocks.ops.attention.fla_gla import can_use_fla_gla, fla_gla_forward
except Exception:

    def can_use_fla_gla(*args, **kwargs):  # type: ignore[misc]
        return False

    def fla_gla_forward(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention GLA cannot be used.")


try:
    from foreblocks.ops.attention.fla_gdn2 import (
        can_use_fla_gdn2,
        can_use_fla_gdn2_chunk,
        fla_gdn2_chunk_forward,
        fla_gdn2_forward,
    )
except Exception:

    def can_use_fla_gdn2(*args, **kwargs):  # type: ignore[misc]
        return False

    def can_use_fla_gdn2_chunk(*args, **kwargs):  # type: ignore[misc]
        return False

    def fla_gdn2_forward(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention GDN-2 cannot be used.")

    def fla_gdn2_chunk_forward(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention GDN-2 cannot be used.")


try:
    from foreblocks.ops.attention.fla_kda import can_use_fla_kda, fla_kda_forward
except Exception:

    def can_use_fla_kda(*args, **kwargs):  # type: ignore[misc]
        return False

    def fla_kda_forward(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("flash-linear-attention KDA cannot be used.")


try:
    from foreblocks.ops.attention.fused_rope import triton_apply_rope
except Exception:

    def triton_apply_rope(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not available; triton_apply_rope cannot be used.")


try:
    from foreblocks.ops.attention.paged_decode import triton_paged_decode
except Exception:

    def triton_paged_decode(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "Triton is not available; triton_paged_decode cannot be used."
        )


__all__ = [
    "can_use_fused_recurrent_linear_attn",
    "fla_chunk_delta_rule",
    "fla_chunk_gated_delta_rule",
    "fla_chunk_gdn2",
    "fla_chunk_gla",
    "fla_chunk_kda",
    "fla_chunk_linear_attn",
    "fla_fused_chunk_delta_rule",
    "fla_fused_recurrent_gated_delta_rule",
    "fla_fused_recurrent_gdn2",
    "fla_fused_recurrent_delta_rule",
    "fla_fused_recurrent_gla",
    "fla_fused_recurrent_kda",
    "fla_fused_recurrent_linear_attn",
    "fla_path",
    "fla_rms_norm_gated",
    "has_fla_checkout",
    "import_fla_module",
    "is_fla_available",
    "chunked_causal_linear_attn",
    "can_use_fla_delta_rule",
    "can_use_fla_recurrent_delta_rule",
    "fla_delta_rule_forward",
    "fused_recurrent_causal_linear_attn",
    "fla_recurrent_delta_rule",
    "can_use_fused_rmsnorm_sigmoid_gate",
    "fused_rmsnorm_sigmoid_gate",
    "can_use_fla_gla",
    "fla_gla_forward",
    "can_use_fla_gated_delta_rule",
    "fla_gated_delta_rule_forward",
    "can_use_fla_linear_attn",
    "fla_recurrent_linear_attn_forward",
    "can_use_fla_gdn2_chunk",
    "can_use_fla_gdn2",
    "fla_gdn2_forward",
    "fla_gdn2_chunk_forward",
    "can_use_fla_kda",
    "fla_kda_forward",
    "triton_paged_decode",
    "triton_apply_rope",
]
