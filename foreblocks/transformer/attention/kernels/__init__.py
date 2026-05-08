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


__all__ = ["triton_paged_decode", "triton_apply_rope"]
