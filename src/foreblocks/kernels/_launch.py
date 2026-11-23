"""Launch parameters shared by row-wise Triton kernels."""


def num_warps_for_block(block_size: int) -> int:
    """Choose the warp count for a power-of-two row block."""
    if block_size <= 64:
        return 2
    if block_size <= 512:
        return 4
    if block_size <= 2048:
        return 8
    return 16
