import pytest
import torch

pytest.importorskip("triton", reason="Triton not available")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _ref_swiglu(a, b):
    return torch.nn.functional.silu(a) * b


def test_forward_unchanged():
    from foreblocks.tf.compute.triton_helpers import swiglu_gate
    torch.manual_seed(1)
    a = torch.randn(2, 8, 64, device="cuda")
    b = torch.randn(2, 8, 64, device="cuda")
    out = swiglu_gate(a, b)
    ref = _ref_swiglu(a, b)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_backward_matches_pytorch():
    from foreblocks.tf.compute.triton_helpers import TritonSwiGLUGate
    torch.manual_seed(2)
    B, T, D = 2, 8, 64
    a_ref = torch.randn(B, T, D, device="cuda", requires_grad=True)
    b_ref = torch.randn(B, T, D, device="cuda", requires_grad=True)
    a_tri = a_ref.detach().clone().requires_grad_(True)
    b_tri = b_ref.detach().clone().requires_grad_(True)

    _ref_swiglu(a_ref, b_ref).sum().backward()
    TritonSwiGLUGate.apply(a_tri, b_tri).sum().backward()

    torch.testing.assert_close(a_tri.grad, a_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b_tri.grad, b_ref.grad, rtol=1e-3, atol=1e-3)


def test_gradcheck_float32():
    from foreblocks.tf.compute.triton_helpers import TritonSwiGLUGate
    B, T, D = 1, 2, 16
    a = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    assert torch.autograd.gradcheck(
        TritonSwiGLUGate.apply, (a, b), eps=1e-3, atol=1e-2, fast_mode=True
    )


def test_only_a_and_b_saved():
    from foreblocks.tf.compute.triton_helpers import TritonSwiGLUGate

    saved = []

    def pack_hook(t):
        saved.append(t)
        return t

    def unpack_hook(t):
        return t

    a = torch.randn(1, 2, 8, device="cuda", requires_grad=True)
    b = torch.randn(1, 2, 8, device="cuda", requires_grad=True)

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        TritonSwiGLUGate.apply(a, b)

    assert len(saved) == 2, f"Expected 2 saved tensors (a, b), got {len(saved)}"
