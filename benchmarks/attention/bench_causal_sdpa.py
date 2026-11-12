#!/usr/bin/env python3
"""Compare eager and SDPA causal full forwards with identical weights.

Run from the repo root with PYTHONPATH=src. Includes QKV/output projections
and positional encoding. Reports inference latency and temporary CUDA memory;
this is not an end-to-end transformer training benchmark.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import torch
from torch.utils.benchmark import Timer

from foreblocks.nn.attention.config import (
    AttentionCacheConfig,
    AttentionConfig,
    AttentionShapeConfig,
    AttentionVariantConfig,
)
from foreblocks.nn.attention.multihead import MultiAttention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 1024])
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(
        json.dumps(
            {"torch": torch.__version__, "device": str(device), "dtype": str(dtype)}
        )
    )
    torch.manual_seed(17)
    config = AttentionConfig(
        shape=AttentionShapeConfig(d_model=512, n_heads=8, dropout=0),
        cache=AttentionCacheConfig(use_mla=False),
        variant=AttentionVariantConfig(backend="eager"),
    )
    eager = MultiAttention(config).to(device=device, dtype=dtype).eval()
    sdpa = MultiAttention(
        replace(config, variant=replace(config.variant, backend="sdpa"))
    )
    sdpa = sdpa.to(device=device, dtype=dtype).eval()
    sdpa.load_state_dict(eager.state_dict(), strict=True)
    with torch.inference_mode():
        for length in args.lengths:
            value = torch.randn(4, length, 512, device=device, dtype=dtype)
            expected = eager(value, is_causal=True)[0]
            actual = sdpa(value, is_causal=True)[0]
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            del expected, actual
            measurements = {}
            # Alternate order to reduce bias from GPU clock/temperature changes.
            samples = {"eager": [], "sdpa": []}
            for round_index in range(4):
                order = [("eager", eager), ("sdpa", sdpa)]
                if round_index % 2:
                    order.reverse()
                for name, model in order:
                    measurement = Timer(
                        stmt="model(value, is_causal=True)",
                        globals={"model": model, "value": value},
                    ).blocked_autorange(min_run_time=0.2)
                    samples[name].append(measurement.median * 1000)
            for name, model in [("eager", eager), ("sdpa", sdpa)]:
                result = {"latency_ms": samples[name]}
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    allocated = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    model(value, is_causal=True)
                    torch.cuda.synchronize(device)
                    result["temporary_mib"] = (
                        torch.cuda.max_memory_allocated(device) - allocated
                    ) / 2**20
                measurements[name] = result
            print(json.dumps({"length": length, "measurements": measurements}))


if __name__ == "__main__":
    main()
