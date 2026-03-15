import pytest
import torch

pytest.importorskip("triton", reason="Triton not available")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_grouped_inputs(E, K, N, Ms, device="cuda", dtype=torch.float32):
    S = sum(Ms)
    A = torch.randn(S, K, device=device, dtype=dtype, requires_grad=True)
    B_list = [
        torch.randn(K, N, device=device, dtype=dtype, requires_grad=True) for _ in range(E)
    ]
    offsets = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(Ms, dtype=torch.int64), 0).tolist()),
        device=device, dtype=torch.int64,
    )
    return A, B_list, offsets


def _pytorch_forward(A_packed, offsets, B_list):
    parts = []
    for e in range(len(B_list)):
        s, t = int(offsets[e].item()), int(offsets[e + 1].item())
        parts.append(A_packed[s:t] @ B_list[e])
    return torch.cat(parts, dim=0)


def test_grad_A_matches_reference():
    from foreblocks.tf.compute.kernels import grouped_mm_varM
    torch.manual_seed(3)
    E, K, N, Ms = 4, 16, 16, [3, 5, 2, 4]
    A_ref, B_ref, offsets = _make_grouped_inputs(E, K, N, Ms)
    A_tri, B_tri, _ = _make_grouped_inputs(E, K, N, Ms)

    _pytorch_forward(A_ref, offsets, B_ref).sum().backward()
    grouped_mm_varM(A_tri, offsets, B_tri).sum().backward()

    torch.testing.assert_close(A_tri.grad, A_ref.grad, rtol=1e-3, atol=1e-3)


def test_grad_B_matches_reference():
    from foreblocks.tf.compute.kernels import grouped_mm_varM
    torch.manual_seed(4)
    E, K, N, Ms = 4, 16, 16, [3, 5, 2, 4]
    A_ref, B_ref, offsets = _make_grouped_inputs(E, K, N, Ms)
    A_tri, B_tri, _ = _make_grouped_inputs(E, K, N, Ms)

    _pytorch_forward(A_ref, offsets, B_ref).sum().backward()
    grouped_mm_varM(A_tri, offsets, B_tri).sum().backward()

    for e in range(E):
        torch.testing.assert_close(B_tri[e].grad, B_ref[e].grad, rtol=1e-3, atol=1e-3)


def test_gradcheck_float32():
    from foreblocks.tf.compute.kernels import _GroupedMMVarMFunction
    torch.manual_seed(5)
    E, K, N, Ms = 2, 8, 8, [2, 3]
    S = sum(Ms)
    A = torch.randn(S, K, device="cuda", dtype=torch.float32, requires_grad=True)
    offsets = torch.tensor([0, Ms[0], S], device="cuda", dtype=torch.int64)
    B_cat = torch.randn(E, K, N, device="cuda", dtype=torch.float32, requires_grad=True)
    assert torch.autograd.gradcheck(
        _GroupedMMVarMFunction.apply,
        (A, offsets, B_cat, False),
        eps=1e-3,
        atol=1e-2,
        fast_mode=True,
    )


def test_empty_expert_segment():
    from foreblocks.tf.compute.kernels import grouped_mm_varM
    E, K, N = 3, 8, 8
    Ms = [4, 0, 3]
    A, B_list, offsets = _make_grouped_inputs(E, K, N, Ms)
    grouped_mm_varM(A, offsets, B_list).sum().backward()
    assert A.grad is not None
    assert B_list[0].grad is not None
    # Expert 1 has no tokens — its grad should be zero
    assert B_list[1].grad is None or B_list[1].grad.abs().max() == 0
