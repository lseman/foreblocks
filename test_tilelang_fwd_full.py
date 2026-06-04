"""Full integration test: TileLang fwd+bwd vs reference."""
import torch
import math
import torch.nn.functional as F
from custom_att.src.tilelang_fwd import tilelang_flash_fwd, can_use_tilelang_fwd
from custom_att.src.tilelang_bwd import tilelang_flash_bwd, can_use_tilelang_bwd
from custom_att import flash_attn_func


def ref(q, k, v, causal=False, softmax_scale=None):
    """Reference PyTorch implementation."""
    scale = softmax_scale or (1.0 / math.sqrt(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    out = torch.matmul(probs, v)
    lse = torch.logsumexp(scores.float(), dim=-1)
    return out, lse


def ref_grad(q, k, v, grad_out, causal=False, softmax_scale=None):
    """Reference gradients via autograd."""
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out, _ = ref(q_, k_, v_, causal, softmax_scale)
        dq, dk, dv = torch.autograd.grad(
            out, (q_, k_, v_), grad_out, retain_graph=False)
    return dq, dk, dv


def test_config(B, H, N, D, causal):
    """Test one config."""
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)

    cu_fwd = can_use_tilelang_fwd(q)
    cu_bwd = can_use_tilelang_bwd(q)
    if not cu_fwd or not cu_bwd:
        return f"B={B} H={H} N={N} D={D} causal={int(causal)} skip (cu_fwd={cu_fwd}, cu_bwd={cu_bwd})"

    # TileLang forward + backward
    out_tl, lse_tl = tilelang_flash_fwd(q, k, v, causal=causal)
    dq_tl, dk_tl, dv_tl = tilelang_flash_bwd(
        out_tl, q, k, v, out_tl, lse_tl, causal=causal)

    # Reference
    out_ref, lse_ref = ref(q, k, v, causal=causal)
    dq_ref, dk_ref, dv_ref = ref_grad(q, k, v, out_tl, causal=causal)

    out_err = (out_tl - out_ref).abs().max().item()
    lse_err = (lse_tl - lse_ref).abs().max().item()
    dq_err = (dq_tl - dq_ref).abs().max().item()
    dk_err = (dk_tl - dk_ref).abs().max().item()
    dv_err = (dv_tl - dv_ref).abs().max().item()

    # Rel error: abs_err / (max(abs_ref) + 1e-6)
    out_rel = out_err / (out_ref.abs().max().item() + 1e-6)
    dq_rel = dq_err / (dq_ref.abs().max().item() + 1e-6)
    dk_rel = dk_err / (dk_ref.abs().max().item() + 1e-6)
    dv_rel = dv_err / (dv_ref.abs().max().item() + 1e-6)

    ok = out_rel < 0.01 and dq_rel < 0.05 and dk_rel < 0.05 and dv_rel < 0.05
    status = "PASS" if ok else "FAIL"

    return (f"B={B} H={H} N={N} D={D} causal={int(causal)} "
            f"out_err={out_err:.4e}/{out_rel:.4e} "
            f"dq_err={dq_err:.4e}/{dq_rel:.4e} "
            f"dk_err={dk_err:.4e}/{dk_rel:.4e} "
            f"dv_err={dv_err:.4e}/{dv_rel:.4e} {status}")


def main():
    configs = [
        (2, 4, 128, 64, False),
        (2, 4, 128, 64, True),
        (2, 4, 256, 64, False),
        (2, 4, 256, 128, True),
        (2, 4, 512, 64, False),
        (1, 8, 64, 128, True),
    ]
    for cfg in configs:
        print(test_config(*cfg))


if __name__ == "__main__":
    main()
