import pytest
import torch

pytest.importorskip("triton", reason="Triton not available")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)


def _make_inputs(B, T, D, dtype=torch.float32, device="cuda", requires_grad=True):
    residual = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    update = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    weight = torch.ones(D, device=device, dtype=dtype, requires_grad=True)
    return residual, update, weight


def test_output_shape():
    from foreblocks.tf.norms.triton_backend import fused_add_rmsnorm
    residual, update, weight = _make_inputs(2, 8, 64)
    out = fused_add_rmsnorm(residual, update, weight, eps=1e-5)
    assert out.shape == residual.shape


def test_output_matches_pytorch_reference():
    from foreblocks.tf.norms.triton_backend import fused_add_rmsnorm
    torch.manual_seed(0)
    residual, update, weight = _make_inputs(2, 8, 64)
    out_fused = fused_add_rmsnorm(
        residual.detach().clone(), update.detach().clone(), weight.detach().clone(), eps=1e-5
    )
    x = residual.detach() + update.detach()
    rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-5).rsqrt()
    out_ref = (x.float() * rms * weight.detach().float()).to(x.dtype)
    torch.testing.assert_close(out_fused, out_ref, rtol=1e-3, atol=1e-3)


def test_gradcheck_float32():
    from foreblocks.tf.norms.triton_backend import FusedAddRMSNormFunction
    B, T, D = 2, 4, 32
    residual = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    update = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.ones(D, device="cuda", dtype=torch.float32, requires_grad=True)
    assert torch.autograd.gradcheck(
        FusedAddRMSNormFunction.apply,
        (residual, update, weight, 1e-5),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
        fast_mode=True,
    )


def test_grad_residual_equals_grad_update():
    from foreblocks.tf.norms.triton_backend import FusedAddRMSNormFunction
    B, T, D = 2, 4, 64
    residual = torch.randn(B, T, D, device="cuda", requires_grad=True)
    update = torch.randn(B, T, D, device="cuda", requires_grad=True)
    weight = torch.ones(D, device="cuda", requires_grad=False)
    out = FusedAddRMSNormFunction.apply(residual, update, weight, 1e-5)
    out.sum().backward()
    torch.testing.assert_close(residual.grad, update.grad)


def test_fallback_for_large_d():
    from foreblocks.tf.fusions import fused_dropout_add_norm
    from foreblocks.tf.norms.rms_norm import RMSNorm
    D = 4096
    residual = torch.randn(1, 4, D, device="cuda")
    update = torch.randn(1, 4, D, device="cuda")
    norm = RMSNorm(D).cuda()
    out = fused_dropout_add_norm(residual, update, norm_layer=norm, p=0.0, training=False)
    assert out.shape == residual.shape
