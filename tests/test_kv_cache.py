"""Correctness tests for the paged KV cache package.

The central property: the block-paged store + streaming (online-softmax) decode
must agree with a plain dense attention reference for the same Q/K/V. These tests
exercise PagedKVCache (write/gather, block boundaries, growth, reset, MLA latent),
the KV providers, and paged_stream_decode_standard.
"""
import pytest
import torch

from foreblocks.modules.attention.cache import (
    DenseKVProvider,
    PagedKVCache,
    PagedKVProvider,
    paged_stream_decode_standard,
)


CPU = torch.device("cpu")
torch.manual_seed(0)


def _cache(B=2, Hkv=2, D=8, block_size=4, max_blocks=64, latent_dim=None):
    return PagedKVCache(
        batch_size=B,
        n_kv_heads=Hkv,
        head_dim=D,
        latent_dim=latent_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        device=CPU,
        dtype=torch.float32,
    )


def _dense_attention(q, k, v, scale, is_causal, q_start_pos):
    """Reference dense attention. q:[B,Hq,Tq,D], k/v:[B,Hq,Tk,D] (already repeated)."""
    B, Hq, Tq, D = q.shape
    Tk = k.shape[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,Hq,Tq,Tk]
    if is_causal:
        q_pos = q_start_pos.view(B, 1, 1) + torch.arange(Tq).view(1, Tq, 1)
        k_pos = torch.arange(Tk).view(1, 1, Tk)
        mask = k_pos > q_pos  # [B,Tq,Tk]
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


# --------------------------------------------------------------- PagedKVCache I/O
def test_gather_roundtrip_within_block():
    c = _cache(B=1, block_size=4)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    c.append(k, v)
    gk, gv = c.gather_kv_batched()
    assert gk.shape == (1, 2, 3, 8)
    torch.testing.assert_close(gk, k)
    torch.testing.assert_close(gv, v)


def test_gather_exactly_full_block():
    # Boundary: seq length is an exact multiple of block_size. The last block is
    # full but write offset wraps to a fresh block at offset 0.
    c = _cache(B=1, block_size=4)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    c.append(k, v)
    assert c.get_seq_len(0) == 4
    gk, gv = c.gather_kv_batched()
    torch.testing.assert_close(gk, k)
    torch.testing.assert_close(gv, v)


def test_gather_spans_multiple_blocks_incremental():
    c = _cache(B=1, block_size=4)
    parts_k, parts_v = [], []
    for t in (3, 5, 4, 1):  # crosses several block boundaries
        k = torch.randn(1, 2, t, 8)
        v = torch.randn(1, 2, t, 8)
        c.append(k, v)
        parts_k.append(k)
        parts_v.append(v)
    gk, gv = c.gather_kv_batched()
    torch.testing.assert_close(gk, torch.cat(parts_k, dim=2))
    torch.testing.assert_close(gv, torch.cat(parts_v, dim=2))
    assert c.get_seq_len(0) == 13


def test_gather_for_seq_matches_batched():
    c = _cache(B=2, block_size=4)
    k = torch.randn(2, 2, 7, 8)
    v = torch.randn(2, 2, 7, 8)
    c.append(k, v)
    gk, _ = c.gather_kv_batched()
    s0k, _ = c.gather_kv_for_seq(0)
    torch.testing.assert_close(s0k, gk[0])


def test_positions_are_tracked():
    c = _cache(B=1, block_size=4)
    c.append(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    c.append(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
    pos = c.gather_positions_for_seq(0)
    torch.testing.assert_close(pos, torch.arange(7))


def test_reset_seq_reclaims_blocks():
    c = _cache(B=2, block_size=4, max_blocks=8)
    used0 = c.num_used_blocks()
    c.append(torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8))
    assert c.num_used_blocks() > used0
    c.reset_all()
    assert c.num_used_blocks() == 0
    assert c.get_seq_len(0) == 0
    assert c.get_logical_seq_len(0) == 0


def test_out_of_blocks_raises():
    c = _cache(B=1, block_size=4, max_blocks=1)
    with pytest.raises(RuntimeError):
        c.append(torch.randn(1, 2, 100, 8), torch.randn(1, 2, 100, 8))


# ----------------------------------------------- paged decode vs dense reference
@pytest.mark.parametrize("Tk", [1, 4, 5, 9, 16])
@pytest.mark.parametrize("Tq", [1, 3])
def test_paged_decode_matches_dense(Tk, Tq):
    B, Hkv, D, bs = 2, 2, 8, 4
    n_rep = 2
    Hq = Hkv * n_rep
    scale = D ** -0.5

    c = _cache(B=B, Hkv=Hkv, D=D, block_size=bs, max_blocks=64)
    k = torch.randn(B, Hkv, Tk, D)
    v = torch.randn(B, Hkv, Tk, D)
    c.append(k, v)

    q = torch.randn(B, Hq, Tq, D)
    # Queries are the last Tq cached positions.
    q_start = (c.logical_seq_len - Tq).clamp_min(0)

    out = paged_stream_decode_standard(
        q, c, kv_repeat=n_rep, scale=scale,
        dropout_p=0.0, training=False, is_causal=True,
        q_start_pos=q_start,
    )

    k_rep = k.repeat_interleave(n_rep, dim=1)
    v_rep = v.repeat_interleave(n_rep, dim=1)
    ref = _dense_attention(q, k_rep, v_rep, scale, is_causal=True, q_start_pos=q_start)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_paged_decode_prefill_then_decode_step():
    # Prefill T tokens, then decode 1 new token whose query must attend to all
    # T+1 keys. Exercises causal masking via stored positions across a real
    # two-call decode sequence (the actual serving pattern).
    B, Hkv, D, bs = 1, 2, 8, 4
    scale = D ** -0.5
    c = _cache(B=B, Hkv=Hkv, D=D, block_size=bs, max_blocks=64)

    k_pre, v_pre = torch.randn(B, Hkv, 6, D), torch.randn(B, Hkv, 6, D)
    c.append(k_pre, v_pre)
    k_new, v_new = torch.randn(B, Hkv, 1, D), torch.randn(B, Hkv, 1, D)
    c.append(k_new, v_new)

    q = torch.randn(B, Hkv, 1, D)  # decode the single new position (index 6)
    q_start = torch.tensor([6])
    out = paged_stream_decode_standard(
        q, c, kv_repeat=1, scale=scale, dropout_p=0.0,
        training=False, is_causal=True, q_start_pos=q_start,
    )
    k_all = torch.cat([k_pre, k_new], dim=2)
    v_all = torch.cat([v_pre, v_new], dim=2)
    ref = _dense_attention(q, k_all, v_all, scale, is_causal=True, q_start_pos=q_start)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_paged_decode_noncausal_matches_dense():
    B, Hkv, D, bs = 1, 2, 8, 4
    c = _cache(B=B, Hkv=Hkv, D=D, block_size=bs)
    k = torch.randn(B, Hkv, 10, D)
    v = torch.randn(B, Hkv, 10, D)
    c.append(k, v)
    q = torch.randn(B, Hkv, 2, D)
    scale = D ** -0.5
    out = paged_stream_decode_standard(
        q, c, kv_repeat=1, scale=scale, dropout_p=0.0,
        training=False, is_causal=False, q_start_pos=None,
    )
    ref = _dense_attention(q, k, v, scale, is_causal=False, q_start_pos=torch.zeros(B))
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------- providers
def test_dense_provider_growing_concat():
    p = DenseKVProvider(layer_state={}, cross_attention=False)
    k1, v1 = torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)
    k2, v2 = torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8)
    fk, _ = p.get_kv(k1, v1)
    assert fk.shape[2] == 3 and p.get_current_length(0) == 3
    fk, _ = p.get_kv(k2, v2)
    assert fk.shape[2] == 5 and p.get_current_length(0) == 5
    torch.testing.assert_close(fk, torch.cat([k1, k2], dim=2))


def test_dense_provider_cross_attention_passthrough():
    p = DenseKVProvider(layer_state={}, cross_attention=True)
    k, v = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
    fk, fv = p.get_kv(k, v)
    torch.testing.assert_close(fk, k)
    assert p.get_current_length(0) == 0


def test_paged_provider_matches_dense_provider():
    c = _cache(B=1, Hkv=2, D=8, block_size=4)
    pp = PagedKVProvider(c)
    dp = DenseKVProvider(layer_state={}, cross_attention=False)
    for t in (3, 4, 2):
        k, v = torch.randn(1, 2, t, 8), torch.randn(1, 2, t, 8)
        pk, _ = pp.get_kv(k.clone(), v.clone())
        dk, _ = dp.get_kv(k.clone(), v.clone())
        torch.testing.assert_close(pk, dk)
        assert pp.get_current_length(0) == dp.get_current_length(0)


# ------------------------------------------------------------------- MLA latent
def test_latent_mode_roundtrip():
    L = 6
    c = _cache(B=2, Hkv=2, D=8, block_size=4, latent_dim=L)
    assert c.use_latent_cache
    lat = torch.randn(2, 5, L)
    c.append_latent(lat)
    out = c.gather_latent_batched()
    assert out.shape == (2, 5, L)
    torch.testing.assert_close(out, lat)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
