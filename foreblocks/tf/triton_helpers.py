import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

@triton.jit
def _swiglu_gate_forward(A, B, C, D,  # A=W1x + b1, B=W3x + b3, C=output buffer, D=optional residual in/out
                         stride_ba, stride_ta, stride_da,
                         stride_bb, stride_tb, stride_db,
                         stride_c, Dhidden: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    t = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    pa = A + b*stride_ba + t*stride_ta + offs*stride_da
    pb = B + b*stride_bb + t*stride_tb + offs*stride_db
    pc = C + b*stride_c + t*Dhidden + offs
    a = tl.load(pa, mask=offs < Dhidden, other=0.0)
    g = tl.load(pb, mask=offs < Dhidden, other=0.0)
    # SwiGLU: swish(a) * g  where swish(a)=a*sigmoid(a)
    sig = 1.0 / (1.0 + tl.exp(-a))
    swish = a * sig
    y = swish * g
    tl.store(pc, y, mask=offs < Dhidden)

class TritonSwiGLUGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        # a, b: [B, T, Dhid]
        assert a.is_cuda and b.is_cuda and a.shape == b.shape
        B, T, Dh = a.shape
        y = torch.empty_like(a)
        BLOCK = triton.next_power_of_2(Dh)
        _swiglu_gate_forward[(B, T)](
            a, b, y, None,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            y.stride(1), Dh, BLOCK=BLOCK,
        )
        ctx.save_for_backward(a, b, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        # For maximum speed, write a matching Triton backward. For simplicity (and still fast),
        # compute grads in PyTorch using saved tensors.
        a, b, y = ctx.saved_tensors
        with torch.cuda.amp.autocast(enabled=False):
            a32 = a.float()
            b32 = b.float()
            dy32 = dy.float()
            sig = torch.sigmoid(a32)
            swish = a32 * sig
            # y = swish * b
            da = dy32 * b32 * (sig + a32 * sig * (1 - sig))  # d(swish)/da
            db = dy32 * swish
            return da.to(a.dtype), db.to(b.dtype)

def swiglu_gate(a, b):
    if HAS_TRITON and a.is_cuda and b.is_cuda:
        return TritonSwiGLUGate.apply(a, b)
    # Fallback
    return F.silu(a) * b
