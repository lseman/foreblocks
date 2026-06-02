"""Test atomic_add with floats in Triton 3.7."""
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_test_kernel(
    in_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    idx = offs % 4
    tl.atomic_add(out_ptr + idx, val, mask=mask)

# Test
n = 128
values = torch.randn(n, device='cuda', dtype=torch.float32)
out = torch.zeros(4, device='cuda', dtype=torch.float32)

grid = (triton.cdiv(n, 64),)
atomic_test_kernel[grid](values, out, n, BLOCK=64)

print('values[:8]:', values[:8].tolist())
print()
print('out (kernel):', out.tolist())
print('out[0] expected (sum values[::4]):', values[::4].sum().item())
print('out[1] expected (sum values[1::4]):', values[1::4].sum().item())
print('out[2] expected (sum values[2::4]):', values[2::4].sum().item())
print('out[3] expected (sum values[3::4]):', values[3::4].sum().item())
