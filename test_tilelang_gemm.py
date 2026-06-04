"""Test TileLang gemm accumulate=True behavior."""
import torch
import tilelang
import tilelang.language as T


def build_test():
    @tilelang.jit(out_idx=None)
    def test_kernel(A, B, C_out):
        @T.prim_func
        def inner(
            A: T.Tensor([32, 32], T.float16),
            B: T.Tensor([32, 32], T.float16),
            C_out: T.Tensor([32, 32], T.float32),
        ):
            A_s = T.alloc_shared([32, 32], T.float16)
            B_s = T.alloc_shared([32, 32], T.float16)
            C_f = T.alloc_fragment([32, 32], T.float32)

            T.copy(A, A_s)
            T.copy(B, B_s)
            T.fill(C_f, 0.0)
            T.gemm(A_s, B_s, C_f, clear_accum=True)
            T.gemm(A_s, B_s, C_f, accumulate=True)
            T.copy(C_f, C_out)
        return inner
    return test_kernel


def main():
    torch.manual_seed(0)
    kern = build_test()
    A = torch.randn(32, 32, device='cuda', dtype=torch.float16)
    B = torch.randn(32, 32, device='cuda', dtype=torch.float16)
    C = torch.empty(32, 32, device='cuda', dtype=torch.float32)
    kern(A, B, C)
    print(f'C shape: {C.shape}, dtype: {C.dtype}')
    # Should be A@B^T + A@B^T = 2 * (A@B^T)
    ref = 2.0 * (A @ B.t()).float()
    max_abs = (C.float() - ref).abs().max().item()
    print(f'Max abs error vs ref: {max_abs:.2e}')
    print(f'Close: {torch.allclose(C.float(), ref, atol=1e-3, rtol=1e-3)}')


if __name__ == "__main__":
    main()
