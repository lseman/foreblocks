import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(8192,), (3, 5, 257)])
def test_triton_gelu_backward_matches_torch(dtype, shape):
    from foreblocks.ops.kernels.gelu import GeluTritonFunction

    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(x)

    GeluTritonFunction.apply(x).backward(grad_output)
    F.gelu(x_ref, approximate="none").backward(grad_output)

    atol = 2e-3 if dtype == torch.float16 else 2e-5
    rtol = 2e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)
