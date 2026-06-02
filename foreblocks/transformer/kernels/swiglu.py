# swiglu.py
# -----------------------------------------------------------------------------
# SwiGLU activation kernels (Triton) and grouped MLP SwiGLU for packed MoE.
# -----------------------------------------------------------------------------

from collections.abc import Sequence

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from .grouped_gemm import grouped_mm_varM

# =============================== Triton kernels ===============================

if HAS_TRITON:

    @triton.jit
    def _swiglu_gate_forward(
        A,
        B,
        C,
        D,
        stride_ba,
        stride_ta,
        stride_da,
        stride_bb,
        stride_tb,
        stride_db,
        stride_bc,
        stride_tc,
        stride_dc,
        Dhidden: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1)
        offs = tl.arange(0, BLOCK)
        pa = A + b * stride_ba + t * stride_ta + offs * stride_da
        pb = B + b * stride_bb + t * stride_tb + offs * stride_db
        pc = C + b * stride_bc + t * stride_tc + offs * stride_dc
        a = tl.load(pa, mask=offs < Dhidden, other=0.0)
        g = tl.load(pb, mask=offs < Dhidden, other=0.0)
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
                a,
                b,
                y,
                None,
                a.stride(0),
                a.stride(1),
                a.stride(2),
                b.stride(0),
                b.stride(1),
                b.stride(2),
                y.stride(0),
                y.stride(1),
                y.stride(2),
                Dh,
                BLOCK=BLOCK,
            )
            ctx.save_for_backward(a, b)
            return y

        @staticmethod
        def backward(ctx, dy):
            a, b = ctx.saved_tensors
            if not (
                HAS_TRITON
                and a.is_cuda
                and a.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and not torch.jit.is_scripting()
            ):
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
                da,
                db,
                a,
                b,
                dy.contiguous(),
                a.stride(0),
                a.stride(1),
                a.stride(2),
                Dh,
                BLOCK=BLOCK,
            )
            return da, db


def swiglu_gate(a, b):
    if HAS_TRITON and a.is_cuda and b.is_cuda:
        return TritonSwiGLUGate.apply(a, b)
    return F.silu(a) * b


# ============================== Grouped MLP SwiGLU ============================


def _weights_from_swiglu_experts(
    experts: Sequence[torch.nn.Module],
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    """Extract per-expert weights for SwiGLU experts. Returns ([w12_e], [w3_e], H)."""
    w12_list: list[torch.Tensor] = []
    w3_list: list[torch.Tensor] = []
    H: int | None = None

    for e in experts:
        if hasattr(e, "w12") and hasattr(e, "w3"):
            w12, w3 = e.w12.weight.t().contiguous(), e.w3.weight.t().contiguous()
        elif hasattr(e, "gate_up_proj") and hasattr(e, "down_proj"):
            w12, w3 = (
                e.gate_up_proj.weight.t().contiguous(),
                e.down_proj.weight.t().contiguous(),
            )
        else:
            raise ValueError("Unsupported expert layout for SwiGLU extraction.")

        if H is None:
            H = w3.shape[0]
        w12_list.append(w12)
        w3_list.append(w3)

    assert H is not None
    return w12_list, w3_list, H


def grouped_mlp_swiglu(
    packed_x: torch.Tensor,
    offsets: torch.Tensor,
    experts: Sequence[torch.nn.Module],
    dropout_p: float = 0.0,
    training: bool = False,
    out_dtype: torch.dtype | None = None,
    use_fp16_acc: bool = False,
    use_shared_b: bool = False,
    allow_triton_training: bool = True,
    B12_cat_prepacked: torch.Tensor | None = None,
    B3_cat_prepacked: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    SwiGLU MLP in grouped mode over packed slices:

        GU = grouped_mm_varM(packed_x, W12[e])  # D -> 2H
        H  = SiLU(G) * U
        Y  = grouped_mm_varM(H, W3[e])         # H -> D

    Returns Y packed in the same order ([S, D]).
    """
    if offsets.numel() <= 1:
        return packed_x.new_zeros((0, packed_x.shape[1]))

    D = packed_x.shape[1]
    if (B12_cat_prepacked is not None) and (B3_cat_prepacked is not None):
        B12 = B12_cat_prepacked
        B3 = B3_cat_prepacked
        if B12.ndim != 3 or B3.ndim != 3:
            raise ValueError(
                "Prepacked tensors must be rank-3: [E, D, 2H] and [E, H, D]."
            )
        H = B3.shape[1]
        w12_list = list(B12.unbind(0))
        w3_list = list(B3.unbind(0))
    else:
        w12_list, w3_list, H = _weights_from_swiglu_experts(experts)

    assert all(w.shape == (D, 2 * H) for w in w12_list), "All w12 must be [D, 2H]"
    assert all(w.shape == (H, D) for w in w3_list), "All w3 must be [H, D]"

    GU = grouped_mm_varM(
        A_packed=packed_x,
        offsets=offsets,
        B_per_expert=w12_list,
        out_dtype=packed_x.dtype,
        use_fp16_acc=use_fp16_acc,
        use_shared_b=use_shared_b,
        allow_triton_training=allow_triton_training,
        block_m=128,
        block_n=128,
        block_k=64,
    )

    if (
        HAS_TRITON
        and GU.is_cuda
        and not torch.jit.is_scripting()
    ):
        G_3d = GU[:, :H].unsqueeze(0).contiguous()
        U_3d = GU[:, H:].unsqueeze(0).contiguous()
        H_act = TritonSwiGLUGate.apply(G_3d, U_3d).squeeze(0)
    else:
        G, U = GU.split(H, dim=-1)
        H_act = F.silu(G) * U

    if training and dropout_p > 0:
        H_act = F.dropout(H_act, p=dropout_p, training=True)

    Y = grouped_mm_varM(
        A_packed=H_act,
        offsets=offsets,
        B_per_expert=w3_list,
        out_dtype=out_dtype or packed_x.dtype,
        use_fp16_acc=use_fp16_acc,
        use_shared_b=use_shared_b,
        allow_triton_training=allow_triton_training,
        block_m=128,
        block_n=128,
        block_k=64,
    )
    return Y


__all__ = [
    "HAS_TRITON",
    "TritonSwiGLUGate",
    "swiglu_gate",
    "_weights_from_swiglu_experts",
    "grouped_mlp_swiglu",
]
