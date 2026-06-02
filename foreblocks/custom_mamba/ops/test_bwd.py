"""Debug script for Triton backward kernel issues."""

import torch
import math


def test_dt_prep_bwd():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.float32
    B, T, D = 2, 8, 16

    from foreblocks.custom_mamba.ops import dt_prep_triton, dt_prep_fallback, dt_prep_bwd_triton

    dt_raw = torch.randn(B, T, D, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype)

    y_triton = dt_prep_triton(dt_raw, bias, dt_min=1e-4, dt_max=1.0)

    # Check specific values
    for bi, ti, di in [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]:
        v = dt_raw[bi, ti, di].item() + bias[di].item()
        y = y_triton[bi, ti, di].item()
        dy = 2.0 * y
        sigmoid_v = 1.0 / (1.0 + math.exp(-v))
        v_softplus = math.log(1 + math.exp(v))
        clamped = max(min(v_softplus, 1.0), 1e-4)
        pass_through = 1.0 if (1e-4 < clamped < 1.0) else 0.0
        expected_dv = dy * pass_through * sigmoid_v
        print(f"[{bi},{ti},{di}] v={v:.4f} y={y:.4f} sp={v_softplus:.4f} clamp={clamped:.4f} "
              f"pt={pass_through} sig={sigmoid_v:.4f} exp_dv={expected_dv:.4f}")

    # Get kernel output
    grad_out = 2.0 * y_triton
    d_dt_raw, d_bias = dt_prep_bwd_triton(grad_out, dt_raw, bias, dt_min=1e-4, dt_max=1.0)

    for bi, ti, di in [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]:
        print(f"  kernel dv[{bi},{ti},{di}] = {d_dt_raw[bi, ti, di].item():.4f}")


def test_fused_out_bwd():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.float32
    B, T, D = 2, 4, 8

    from foreblocks.custom_mamba.ops import fused_out_triton, fused_out_fallback, fused_out_bwd_triton

    y = torch.randn(B, T, D, device=device, dtype=dtype)
    z = torch.randn(B, T, D, device=device, dtype=dtype)
    residual = torch.randn(B, T, D, device=device, dtype=dtype)
    norm_weight = torch.ones(D, device=device, dtype=dtype)

    out_triton = fused_out_triton(y, z, residual, norm_weight)
    out_fallback = fused_out_fallback(y, z, residual, norm_weight)

    print('Forward match:', (out_triton - out_fallback).abs().max().item())

    # Compute backward
    grad_out = 2.0 * out_triton
    dy, dz, dr, dw = fused_out_bwd_triton(grad_out, y, z, residual, norm_weight, eps=1e-6)

    # Fallback backward via autograd
    y_g = y.clone().detach().requires_grad_(True)
    z_g = z.clone().detach().requires_grad_(True)
    r_g = residual.clone().detach().requires_grad_(True)
    w_g = norm_weight.clone().detach().requires_grad_(True)
    out_f = fused_out_fallback(y_g, z_g, r_g, w_g)
    loss = out_f.square().sum()
    loss.backward()

    print()
    print('fused_out backward:')
    print(f'  dy max diff: {(dy - 2.0 * y_g.grad).abs().max().item():.6e}')
    print(f'  dz max diff: {(dz - z_g.grad).abs().max().item():.6e}')
    print(f'  dr max diff: {(dr - r_g.grad).abs().max().item():.6e}')
    print(f'  dw max diff: {(dw - w_g.grad).abs().max().item():.6e}')


if __name__ == '__main__':
    print('=== dt_prep bwd ===')
    test_dt_prep_bwd()
    print()
    print('=== fused_out bwd ===')
    test_fused_out_bwd()
