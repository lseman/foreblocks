import torch
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _swiglu_gate_forward(
        A,
        B,
        C,
        D,  # A=W1x + b1, B=W3x + b3, C=output buffer, D=optional residual in/out
        stride_ba,
        stride_ta,
        stride_da,
        stride_bb,
        stride_tb,
        stride_db,
        stride_c,
        Dhidden: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1)
        offs = tl.arange(0, BLOCK)
        pa = A + b * stride_ba + t * stride_ta + offs * stride_da
        pb = B + b * stride_bb + t * stride_tb + offs * stride_db
        pc = C + b * stride_c + t * Dhidden + offs
        a = tl.load(pa, mask=offs < Dhidden, other=0.0)
        g = tl.load(pb, mask=offs < Dhidden, other=0.0)
        # SwiGLU: swish(a) * g  where swish(a)=a*sigmoid(a)
        sig = 1.0 / (1.0 + tl.exp(-a))
        swish = a * sig
        y = swish * g
        tl.store(pc, y, mask=offs < Dhidden)

    @triton.jit
    def _swiglu_gate_backward(
        DA,
        DB,
        A,
        B,
        DY,
        stride_b,
        stride_t,
        stride_d,
        Dhidden: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        time = tl.program_id(1)
        offs = tl.arange(0, BLOCK)
        base = batch * stride_b + time * stride_t
        mask = offs < Dhidden

        a = tl.load(A + base + offs * stride_d, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + base + offs * stride_d, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + base + offs * stride_d, mask=mask, other=0.0).to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-a))
        swish = a * sig
        da = dy * b * (sig + a * sig * (1.0 - sig))
        db = dy * swish

        tl.store(DA + base + offs * stride_d, da, mask=mask)
        tl.store(DB + base + offs * stride_d, db, mask=mask)

    class TritonSwiGLUGate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a: torch.Tensor, b: torch.Tensor):
            assert a.is_cuda and b.is_cuda and a.shape == b.shape
            a = a.contiguous()
            b = b.contiguous()
            B, T, Dh = a.shape
            y = torch.empty_like(a)
            BLOCK = triton.next_power_of_2(Dh)
            _swiglu_gate_forward[(B, T)](
                a, b, y, None,
                a.stride(0), a.stride(1), a.stride(2),
                b.stride(0), b.stride(1), b.stride(2),
                y.stride(1),
                Dh, BLOCK=BLOCK,
            )
            ctx.save_for_backward(a, b)
            return y

        @staticmethod
        def backward(ctx, dy):
            a, b = ctx.saved_tensors
            if not (HAS_TRITON and a.is_cuda
                    and a.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and not torch.jit.is_scripting()):
                with torch.amp.autocast("cuda", enabled=False):
                    a32, b32, dy32 = a.float(), b.float(), dy.float()
                    sig = torch.sigmoid(a32)
                    swish = a32 * sig
                    da = dy32 * b32 * (sig + a32 * sig * (1 - sig))
                    db = dy32 * swish
                return da.to(a.dtype), db.to(b.dtype)

            B, T, Dh = a.shape
            da = torch.empty_like(a)
            db = torch.empty_like(b)
            BLOCK = triton.next_power_of_2(Dh)
            _swiglu_gate_backward[(B, T)](
                da, db, a, b, dy.contiguous(),
                a.stride(0), a.stride(1), a.stride(2),
                Dh, BLOCK=BLOCK,
            )
            return da, db


def swiglu_gate(a, b):
    if HAS_TRITON and a.is_cuda and b.is_cuda:
        return TritonSwiGLUGate.apply(a, b)
    # Fallback
    return F.silu(a) * b
