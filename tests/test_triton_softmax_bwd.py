import pytest
import torch

pytest.importorskip("triton", reason="Triton not available")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_triton_softmax_forward_and_backward_match_torch(dtype, dim):
    from foreblocks.ops.kernels.softmax import SoftmaxTritonFunction

    shape = (4, 5, 257)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(x)

    actual = SoftmaxTritonFunction.apply(x, dim)
    expected = torch.softmax(x_ref, dim=dim)
    actual.backward(grad_output)
    expected.backward(grad_output)

    atol = 2e-3 if dtype == torch.float16 else 2e-5
    rtol = 2e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)


def test_triton_softmax_accepts_noncontiguous_input_and_gradient():
    from foreblocks.ops.kernels.softmax import SoftmaxTritonFunction

    x = torch.randn(5, 4, 257, device="cuda").transpose(0, 1).requires_grad_()
    x_ref = x.detach().clone().requires_grad_(True)
    grad_output = torch.randn(5, 4, 257, device="cuda").transpose(0, 1)

    actual = SoftmaxTritonFunction.apply(x, 1)
    expected = torch.softmax(x_ref, dim=1)
    actual.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-5, rtol=2e-5)
