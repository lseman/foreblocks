"""GPU correctness tests for the Triton attention kernels.

Each kernel is checked against an independent eager reference:
  - fused_rope (triton_apply_rope / triton_apply_rope_bthd) vs a rotate-half
    RoPE reference using the GPT-NeoX convention (cos/sin = cat([freqs, freqs])).
    This matches embeddings/rotary.py, which only dispatches to the Triton path
    for the non-interleaved case.
  - paged_decode (triton_paged_decode) vs the eager online-softmax decode
    (paged_stream_decode_standard), which is itself pinned to dense attention
    in tests/test_kv_cache.py.

These require CUDA + a working Triton install and are skipped otherwise.
"""
import pytest
import torch

from foreblocks.layers.embeddings.rotary import apply_rotary_emb
from foreblocks.modules.attention.cache.decode_stream import (
    paged_stream_decode_standard,
)
from foreblocks.modules.attention.cache.paged import PagedKVCache
from foreblocks.ops.attention.fused_rope import (
    triton_apply_rope,
    triton_apply_rope_bthd,
)
from foreblocks.ops.attention.paged_decode import triton_paged_decode


triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)


DEV = "cuda"
ATOL = 1e-3


def _build_cossin(T, D, device):
    """GPT-NeoX RoPE tables: emb = cat([freqs, freqs]) -> cos/sin of shape [T, D]."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=device).float() / D))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv_freq)  # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
    return emb.cos(), emb.sin()


def _rope_ref(x, cos, sin):
    """Non-interleaved rotate-half RoPE. x:[B,H,T,D]; cos/sin:[T,D]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos.view(1, 1, *cos.shape) + rot * sin.view(1, 1, *sin.shape)


def _rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _rope_ref_bthd(x, cos, sin, interleaved=False):
    """RoPE reference for x:[B,T,H,D], cos/sin:[B,T,R] or [T,R]."""
    ro_dim = cos.shape[-1] * 2
    x_rot = x[..., :ro_dim]
    if not interleaved:
        cos_full = torch.cat([cos, cos], dim=-1)
        sin_full = torch.cat([sin, sin], dim=-1)
    else:
        cos_full = torch.stack([cos, cos], dim=-1).flatten(-2)
        sin_full = torch.stack([sin, sin], dim=-1).flatten(-2)
    while cos_full.dim() < x_rot.dim():
        cos_full = cos_full.unsqueeze(-2)
        sin_full = sin_full.unsqueeze(-2)
    rotated = x_rot * cos_full + _rotate_half(x_rot, interleaved) * sin_full
    if ro_dim == x.shape[-1]:
        return rotated
    return torch.cat([rotated, x[..., ro_dim:]], dim=-1)


def test_triton_apply_rope_bhtd_matches_reference():
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 7, 16
    q = torch.randn(B, H, T, D, device=DEV)
    k = torch.randn(B, H, T, D, device=DEV)
    cos, sin = _build_cossin(T, D, DEV)  # [T, D]

    oq, ok = triton_apply_rope(q, k, cos[:, : D // 2], sin[:, : D // 2], block_t=16)
    torch.testing.assert_close(oq, _rope_ref(q, cos, sin), atol=ATOL, rtol=0)
    torch.testing.assert_close(ok, _rope_ref(k, cos, sin), atol=ATOL, rtol=0)


def test_triton_apply_rope_seqlen_offset():
    torch.manual_seed(1)
    B, H, T, D, off = 2, 4, 4, 16, 3
    x = torch.randn(B, H, T, D, device=DEV)
    cos, sin = _build_cossin(off + T, D, DEV)
    oq, _ = triton_apply_rope(
        x, x.clone(), cos[:, : D // 2], sin[:, : D // 2], seqlen_offset=off, block_t=16
    )
    ref = _rope_ref(x, cos[off : off + T], sin[off : off + T])
    torch.testing.assert_close(oq, ref, atol=ATOL, rtol=0)


def test_triton_apply_rope_bthd_matches_reference():
    torch.manual_seed(2)
    B, T, H, D = 2, 7, 4, 16
    x = torch.randn(B, T, H, D, device=DEV)
    cos, sin = _build_cossin(T, D, DEV)
    out = triton_apply_rope_bthd(x, cos[:, : D // 2], sin[:, : D // 2], block_t=16)
    ref = _rope_ref(x.transpose(1, 2), cos, sin).transpose(1, 2)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=0)


def test_triton_apply_rope_bthd_partial_interleaved_tensor_offsets():
    torch.manual_seed(4)
    B, T, H, D, R = 3, 5, 2, 16, 4
    x = torch.randn(B, T, H, D, device=DEV)
    offsets = torch.tensor([0, 2, 4], device=DEV, dtype=torch.int32)
    angles = torch.randn(T + int(offsets.max().item()), R, device=DEV)
    cos, sin = angles.cos(), angles.sin()

    out = triton_apply_rope_bthd(
        x,
        cos,
        sin,
        seqlen_offset=offsets,
        interleaved=True,
        block_t=16,
    )
    pos = offsets[:, None] + torch.arange(T, device=DEV)
    ref = _rope_ref_bthd(x, cos[pos], sin[pos], interleaved=True)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=0)


def test_apply_rotary_emb_backward_with_triton_general_path():
    torch.manual_seed(5)
    B, T, H, D, R = 2, 6, 3, 16, 4
    offsets = torch.tensor([1, 3], device=DEV, dtype=torch.int64)
    angles = torch.randn(T + int(offsets.max().item()), R, device=DEV)
    cos, sin = angles.cos(), angles.sin()
    grad = torch.randn(B, T, H, D, device=DEV)

    x = torch.randn(B, T, H, D, device=DEV, requires_grad=True)
    y = apply_rotary_emb(x, cos, sin, interleaved=True, seqlen_offsets=offsets)
    y.backward(grad)

    x_ref = x.detach().clone().requires_grad_(True)
    pos = offsets[:, None] + torch.arange(T, device=DEV)
    y_ref = _rope_ref_bthd(x_ref, cos[pos], sin[pos], interleaved=True)
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref, atol=ATOL, rtol=0)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    "B,Hkv,D,bs,Tk,Tq,n_rep",
    [
        (2, 2, 16, 4, 1, 1, 2),   # decode step, GQA
        (1, 2, 16, 4, 9, 1, 1),   # multi-block, MHA
        (2, 4, 16, 8, 17, 3, 2),  # multi-block, GQA, multi-query
        (1, 1, 32, 4, 5, 2, 1),   # single head
    ],
)
def test_triton_paged_decode_matches_eager(B, Hkv, D, bs, Tk, Tq, n_rep):
    torch.manual_seed(3)
    Hq = Hkv * n_rep
    scale = D ** -0.5
    c = PagedKVCache(B, Hkv, D, None, bs, 128, torch.device(DEV), torch.float32)
    c.append(
        torch.randn(B, Hkv, Tk, D, device=DEV),
        torch.randn(B, Hkv, Tk, D, device=DEV),
    )

    q = torch.randn(B, Hq, Tq, D, device=DEV)
    q_start = (c.logical_seq_len - Tq).clamp_min(0).to(torch.int32)

    tri = triton_paged_decode(q, c, n_rep, scale, q_start_pos=q_start)
    eager = paged_stream_decode_standard(
        q, c, kv_repeat=n_rep, scale=scale, dropout_p=0.0,
        training=False, is_causal=True, q_start_pos=q_start,
    )
    torch.testing.assert_close(tri, eager, atol=ATOL, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
