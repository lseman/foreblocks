"""Test TileLang forward kernel compilation and correctness."""
import torch
import math
import torch.nn.functional as F
from custom_att.src.tilelang_fwd import tilelang_flash_fwd, can_use_tilelang_fwd


def ref(q, k, v, causal=False, softmax_scale=None):
    scale = softmax_scale or (1.0 / math.sqrt(q.shape[-1]))
    scores = q @ k.transpose(-1, -2) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
    out = probs @ v
    lse = torch.logsumexp(scores.float(), dim=-1)
    return out, lse


def main():
    torch.manual_seed(0)

    for B, H, N, D, causal in [
        (2, 4, 128, 64, False),
        (2, 4, 128, 64, True),
        (2, 4, 256, 64, False),
        (2, 4, 256, 128, True),
        (2, 4, 512, 64, False),
    ]:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        cu = can_use_tilelang_fwd(q)
        print(f"B={B} H={H} N={N} D={D} causal={int(causal)} can_use={cu}", end="  ")

        if not cu:
            print("skipped")
            continue

        out_tl, lse_tl = tilelang_flash_fwd(q, k, v, causal=causal)
        out_ref, lse_ref = ref(q, k, v, causal=causal)

        out_err = (out_tl - out_ref).abs().max().item()
        lse_err = (lse_tl - lse_ref).abs().max().item()

        status = "PASS" if out_err < 0.1 and lse_err < 0.1 else "FAIL"
        print(f"out_err={out_err:.4e} lse_err={lse_err:.4e} {status}")


if __name__ == "__main__":
    main()
