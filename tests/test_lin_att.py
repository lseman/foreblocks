import unittest

import torch
import torch.nn.functional as F

from foreblocks.transformer.attention.modules.lin_att import LinearAttention


def _manual_causal(m, x):
    """Reference causal linear attention with the same feature map + scale."""
    B, L, _ = x.shape
    q = m.q_proj(x).view(B, L, m.n_heads, m.d_head).transpose(1, 2) * m.scale
    k = m.k_proj(x).view(B, L, m.n_heads, m.d_head).transpose(1, 2) * m.scale
    v = m.v_proj(x).view(B, L, m.n_heads, m.d_head).transpose(1, 2)
    qp, kp = F.elu(q) + 1, F.elu(k) + 1
    out = torch.zeros_like(v)
    for t in range(L):
        kv = torch.einsum("bhjf,bhjd->bhfd", kp[:, :, : t + 1], v[:, :, : t + 1])
        num = torch.einsum("bhf,bhfd->bhd", qp[:, :, t], kv)
        den = torch.einsum("bhf,bhf->bh", qp[:, :, t], kp[:, :, : t + 1].sum(2))
        out[:, :, t] = num / (den.unsqueeze(-1) + 1e-6)
    return m.out_proj(out.transpose(1, 2).reshape(B, L, m.d_model))


class TestLinearAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = LinearAttention(d_model=16, n_heads=2, dropout=0.0).double().eval()
        self.x = torch.randn(2, 7, 16, dtype=torch.float64)

    def test_causal_fallback_runs_and_matches_reference(self):
        # Previously crashed (matmul dim mismatch); must run and be correct.
        with torch.no_grad():
            out, _, _ = self.m(self.x, self.x, self.x, is_causal=True)
            ref = _manual_causal(self.m, self.x)
        self.assertEqual(out.shape, self.x.shape)
        self.assertLess((out - ref).abs().max().item(), 1e-10)

    def test_causal_does_not_leak_future(self):
        x2 = self.x.clone()
        x2[:, -1] += 10.0  # perturb only the last token
        with torch.no_grad():
            o1, _, _ = self.m(self.x, self.x, self.x, is_causal=True)
            o2, _, _ = self.m(x2, x2, x2, is_causal=True)
        # earlier outputs must be unchanged
        self.assertLess((o1[:, :-1] - o2[:, :-1]).abs().max().item(), 1e-12)

    def test_causal_trains_with_grad(self):
        m = LinearAttention(d_model=16, n_heads=2, dropout=0.0)
        x = torch.randn(2, 7, 16, requires_grad=True)
        out, _, _ = m(x, x, x, is_causal=True)  # grad enabled -> fallback path
        out.pow(2).mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_incremental_matches_full_causal(self):
        with torch.no_grad():
            full, _, _ = self.m(self.x, self.x, self.x, is_causal=True)
            # decode one token at a time, carrying state
            ls: dict = {}
            outs = []
            for t in range(self.x.size(1)):
                xt = self.x[:, t : t + 1]
                o, _, ls = self.m(xt, xt, xt, is_causal=True, layer_state=ls)
                outs.append(o)
            inc = torch.cat(outs, dim=1)
        self.assertLess((inc - full).abs().max().item(), 1e-10)

    def test_incremental_chunked_prefill_then_step(self):
        # prefill 4 tokens, then 3 single steps -> matches full forward
        with torch.no_grad():
            full, _, _ = self.m(self.x, self.x, self.x, is_causal=True)
            ls: dict = {}
            pre = self.x[:, :4]
            o_pre, _, ls = self.m(pre, pre, pre, is_causal=True, layer_state=ls)
            outs = [o_pre]
            for t in range(4, self.x.size(1)):
                xt = self.x[:, t : t + 1]
                o, _, ls = self.m(xt, xt, xt, is_causal=True, layer_state=ls)
                outs.append(o)
            inc = torch.cat(outs, dim=1)
        self.assertLess((inc - full).abs().max().item(), 1e-10)

    def test_noncausal_still_works(self):
        with torch.no_grad():
            out, _, _ = self.m(self.x, self.x, self.x, is_causal=False)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
