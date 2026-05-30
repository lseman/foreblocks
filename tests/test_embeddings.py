"""Correctness tests for foreblocks.transformer.embeddings.

Covers the sinusoidal, learnable, and Informer time embeddings. RoPE
(rotary.py) is already exercised on GPU in tests/test_triton_kernels.py and
via its usage in the attention paths.
"""
import torch

from foreblocks.transformer.embeddings.positional_encoding import PositionalEncoding
from foreblocks.transformer.embeddings.learnable_positional_encoding import (
    LearnablePositionalEncoding,
)
from foreblocks.transformer.embeddings.informer_time_embedding import (
    InformerTimeEmbedding,
)


# ------------------------------------------------------------ PositionalEncoding
def test_pe_adds_sinusoid_and_broadcasts():
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_len=32)
    out = pe(torch.zeros(2, 5, 16))  # adding to zeros -> the table itself
    assert out.shape == (2, 5, 16)
    assert torch.allclose(out[0], out[1])  # same across batch
    # col 0 is sin(pos * 1) = sin(pos)
    assert torch.allclose(out[0, :, 0], torch.sin(torch.arange(5.0)), atol=1e-6)


def test_pe_explicit_positions_match_default():
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_len=32)
    x = torch.zeros(2, 5, 16)
    default = pe(x)
    explicit = pe(x, pos=torch.arange(5))
    torch.testing.assert_close(default, explicit, atol=1e-6, rtol=0)


def test_pe_odd_d_model_last_col_zero_and_finite():
    pe = PositionalEncoding(d_model=15, dropout=0.0, max_len=8)
    out = pe(torch.zeros(1, 4, 15))
    assert out.shape == (1, 4, 15)
    assert torch.isfinite(out).all()
    # odd-D branch zeroes the final column
    torch.testing.assert_close(out[..., -1], torch.zeros_like(out[..., -1]))


def test_pe_position_beyond_max_len():
    pe = PositionalEncoding(d_model=8, dropout=0.0, max_len=4)  # table covers 0..3
    out = pe(torch.zeros(1, 1, 8), pos=torch.tensor([[10]]))
    ref = PositionalEncoding._build_table(8, 11).squeeze(0)[10]
    torch.testing.assert_close(out[0, 0], ref, atol=1e-6, rtol=0)


def test_pe_cache_eviction_is_stable():
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_len=8, cache_limit=2)
    first = {D: pe(torch.zeros(1, 4, D)) for D in (10, 12, 14, 16)}
    for D in (10, 12, 14, 16):  # re-query after evictions
        torch.testing.assert_close(pe(torch.zeros(1, 4, D)), first[D], atol=1e-6, rtol=0)


# --------------------------------------------------- LearnablePositionalEncoding
def test_lpe_explicit_positions_match_default():
    lpe = LearnablePositionalEncoding(
        d_model=16, max_len=32, dropout=0.0, use_layer_norm=False
    )
    x = torch.zeros(2, 5, 16)
    default = lpe(x)
    explicit = lpe(x, positions=torch.arange(5).unsqueeze(0).expand(2, -1))
    torch.testing.assert_close(default, explicit, atol=1e-5, rtol=0)


def test_lpe_low_rank_matches_full_form():
    lpe = LearnablePositionalEncoding(
        d_model=16, max_len=32, dropout=0.0, low_rank_dim=4, use_layer_norm=False
    )
    out = lpe(torch.zeros(2, 5, 16))
    out_pos = lpe(torch.zeros(2, 5, 16), positions=torch.arange(5).unsqueeze(0).expand(2, -1))
    assert out.shape == (2, 5, 16)
    torch.testing.assert_close(out, out_pos, atol=1e-5, rtol=0)


# ------------------------------------------------------- InformerTimeEmbedding
def test_ite_output_dim_with_and_without_projection():
    a = InformerTimeEmbedding(64)   # embed_dim*4 == d_model -> no projection
    assert a.projection is None
    assert a(torch.zeros(2, 5, 4)).shape == (2, 5, 64)

    c = InformerTimeEmbedding(512)  # embed_dim*4 != d_model -> projection
    assert c.projection is not None
    assert c(torch.zeros(2, 5, 4)).shape == (2, 5, 512)


def test_ite_clamps_out_of_range_time_features():
    ite = InformerTimeEmbedding(64)
    out = ite(torch.full((1, 3, 4), 999.0))  # OOB indices must be clamped
    assert torch.isfinite(out).all()
    assert out.shape == (1, 3, 64)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
