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
                y.stride(1),
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
    # Fallback
    return F.silu(a) * b


# ─────────────────────────────────────────────────────────────────────────────
# Causal linear attention sequential scan (inference-only)
# ─────────────────────────────────────────────────────────────────────────────
# Avoids materialising the O(B·H·T·F·Dh) intermediate tensor that the naive
# PyTorch implementation produces via unsqueeze + cumsum.
# Each GPU block handles one (batch, head) pair and scans all T positions
# sequentially, maintaining running KV[F, Dh] and k_sum[F] state in registers.
# ─────────────────────────────────────────────────────────────────────────────
if HAS_TRITON:

    @triton.jit
    def _causal_lin_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        OUT_ptr,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qf,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kf,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_ot,
        stride_od,
        T,
        H,
        F,
        Dh,
        EPS: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BLOCK_Dh: tl.constexpr,
    ):
        """
        Causal linear attention scan for one (b, h) pair.
        Inclusive cumsum semantics: output[t] uses k[0..t] and v[0..t].
        """
        pid_bh = tl.program_id(0)
        pid_b = pid_bh // H
        pid_h = pid_bh % H

        f_offs = tl.arange(0, BLOCK_F)
        d_offs = tl.arange(0, BLOCK_Dh)
        f_mask = f_offs < F
        d_mask = d_offs < Dh

        # Running state (in registers / L1 SRAM)
        k_sum = tl.zeros([BLOCK_F], dtype=tl.float32)
        KV = tl.zeros([BLOCK_F, BLOCK_Dh], dtype=tl.float32)

        q_base0 = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base0 = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base0 = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        o_base0 = OUT_ptr + pid_b * stride_ob + pid_h * stride_oh

        for t in range(T):
            q = tl.load(
                q_base0 + t * stride_qt + f_offs * stride_qf, mask=f_mask, other=0.0
            ).to(tl.float32)
            k = tl.load(
                k_base0 + t * stride_kt + f_offs * stride_kf, mask=f_mask, other=0.0
            ).to(tl.float32)
            v = tl.load(
                v_base0 + t * stride_vt + d_offs * stride_vd, mask=d_mask, other=0.0
            ).to(tl.float32)

            # Inclusive update: include current (k, v) before computing output
            k_sum = k_sum + k
            KV = KV + k[:, None] * v[None, :]  # outer product [F, Dh]

            # Compute output: q @ KV / (q · k_sum)
            numer = tl.sum(KV * q[:, None], axis=0)  # [Dh]: matmul q[F] @ KV[F,Dh]
            denom = tl.sum(q * k_sum) + EPS

            tl.store(
                o_base0 + t * stride_ot + d_offs * stride_od,
                numer / denom,
                mask=d_mask,
            )


def triton_causal_linear_attn(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    eps: float = 1e-6,
) -> "torch.Tensor":
    """
    Inference-only causal linear attention via sequential scan.

    Args:
        q, k: [B, H, T, F]  (feature-mapped queries/keys, e.g. ELU+1)
        v:    [B, H, T, Dh]
        eps:  denominator stabiliser

    Returns:
        out:  [B, H, T, Dh]

    Memory: O(B·H·(T·F + T·Dh)) instead of O(B·H·T·F·Dh) for the naive path.
    Only called in non-grad (inference) mode; training uses the PyTorch fallback.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton_causal_linear_attn requires Triton.")
    B, H, T, F = q.shape
    Dh = v.shape[-1]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty(B, H, T, Dh, dtype=q.dtype, device=q.device)

    BLOCK_F = triton.next_power_of_2(F)
    BLOCK_Dh = triton.next_power_of_2(Dh)

    _causal_lin_attn_fwd_kernel[(B * H,)](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        T,
        H,
        F,
        Dh,
        EPS=eps,
        BLOCK_F=BLOCK_F,
        BLOCK_Dh=BLOCK_Dh,
        num_warps=4,
    )
    return out
