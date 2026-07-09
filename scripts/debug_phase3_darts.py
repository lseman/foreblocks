#!/usr/bin/env python3
"""Run a minimal Phase-3-style DARTS training repro on synthetic signals.

This script skips candidate generation and zero-cost scoring. It builds one
fixed candidate, creates random-but-structured time-series windows, and calls
``DARTSTrainer.train_darts_model`` exactly like Phase 3 does.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from darts.config import DEFAULT_OPS  # noqa: E402
from darts.search.phase_utils import _resolve_phase3_rung_epochs  # noqa: E402
from darts.trainer import DARTSTrainer  # noqa: E402


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _make_signal_windows(
    *,
    samples: int,
    seq_len: int,
    horizon: int,
    channels: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    total_len = samples + seq_len + horizon + 8
    t = np.arange(total_len, dtype=np.float32)
    series = np.zeros((total_len, channels), dtype=np.float32)

    for c in range(channels):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        slow = np.sin(2.0 * math.pi * t / (24.0 + 3.0 * c) + phase)
        fast = 0.35 * np.sin(2.0 * math.pi * t / (7.0 + c) + 0.5 * phase)
        trend = 0.002 * (c + 1) * t
        noise = 0.05 * rng.standard_normal(total_len).astype(np.float32)
        series[:, c] = slow + fast + trend + noise

    x = np.stack([series[i : i + seq_len] for i in range(samples)], axis=0)
    y = np.stack(
        [series[i + seq_len : i + seq_len + horizon] for i in range(samples)],
        axis=0,
    )
    return torch.from_numpy(x), torch.from_numpy(y)


def _build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    x, y = _make_signal_windows(
        samples=args.samples,
        seq_len=args.seq_len,
        horizon=args.horizon,
        channels=args.channels,
        seed=args.seed,
    )
    split = max(2, int(round(args.samples * 0.8)))
    split = min(split, args.samples - 1)
    train_ds = TensorDataset(x[:split], y[:split])
    val_ds = TensorDataset(x[split:], y[split:])
    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(args.num_workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(args.num_workers > 0),
        drop_last=False,
    )
    return train_loader, val_loader


def _candidate(args: argparse.Namespace) -> dict[str, Any]:
    ops = _csv(args.ops) or ["Identity", "ResidualMLP", "TimeConv"]
    families = _csv(args.families) or ["mlp", "conv"]
    return {
        "candidate_id": int(args.candidate_id),
        "selected_ops": ops,
        "selected_families": families,
        "family_choices": {},
        "hidden_dim": int(args.hidden_dim),
        "num_cells": int(args.num_cells),
        "num_nodes": int(args.num_nodes),
        "arch_mode": str(args.arch_mode),
        "transformer_self_attention_type": str(args.attn),
        "transformer_ffn_variant": str(args.ffn),
        "transformer_use_moe": str(args.ffn).lower() == "moe",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct Phase-3 DARTS repro with synthetic time-series data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-cells", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--candidate-id", type=int, default=1)
    parser.add_argument("--arch-mode", default="encoder_decoder")
    parser.add_argument("--ops", default="Identity,ResidualMLP,TimeConv")
    parser.add_argument("--families", default="mlp,conv")
    parser.add_argument("--attn", default="auto")
    parser.add_argument("--ffn", default="auto")
    parser.add_argument("--search-epochs", type=int, default=32)
    parser.add_argument("--rungs", default="2,4,8,32")
    parser.add_argument("--phase3-min-epoch-budget", type=int, default=2)
    parser.add_argument("--phase3-reduction-factor", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument("--no-gdas", action="store_true", help="Disable op GDAS")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--debug-crash",
        action="store_true",
        help="Set FORE_DARTS_DEBUG_CRASH=1 for stage markers.",
    )
    parser.add_argument(
        "--disable-fused-optimizers",
        action="store_true",
        help="Disable fused Adam/AdamW kernels on CUDA.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.device = _device(args.device)
    if args.debug_crash:
        os.environ["FORE_DARTS_DEBUG_CRASH"] = "1"
    if args.disable_fused_optimizers:
        os.environ["FORE_DARTS_DISABLE_FUSED_OPTIMIZERS"] = "1"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader = _build_loaders(args)
    cand = _candidate(args)
    all_ops = list(dict.fromkeys(list(DEFAULT_OPS) + cand["selected_ops"]))
    trainer = DARTSTrainer(
        input_dim=args.channels,
        hidden_dims=[args.hidden_dim],
        forecast_horizon=args.horizon,
        seq_length=args.seq_len,
        device=args.device,
        all_ops=all_ops,
        arch_modes=[args.arch_mode],
        attention_variants=[args.attn],
        ffn_variants=[args.ffn],
    )
    model = trainer._build_candidate_model(cand)

    explicit_rungs = [int(x) for x in (_csv(args.rungs) or [])]
    rung_epochs = _resolve_phase3_rung_epochs(
        search_epochs=args.search_epochs,
        min_epoch_budget=args.phase3_min_epoch_budget,
        reduction_factor=args.phase3_reduction_factor,
        explicit=explicit_rungs or None,
    )

    print(
        f"\n=== Phase 3 debug: training 1 candidate across rungs {rung_epochs} ===",
        flush=True,
    )
    print(
        f"[debug] device={args.device} torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        print(f"[debug] gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[debug] candidate={cand}", flush=True)

    epochs_trained = 0
    best_val = float("inf")
    for rung_idx, rung_epoch in enumerate(rung_epochs):
        delta_epochs = int(rung_epoch) - int(epochs_trained)
        if delta_epochs <= 0:
            continue
        print(
            f"[P3][rung {rung_idx + 1}/{len(rung_epochs)}] "
            f"budget={rung_epoch} active=1",
            flush=True,
        )
        print(
            f"[P3] training rung {rung_idx + 1} candidate 1/1 "
            f"(id={cand['candidate_id']}) for +{delta_epochs} epochs "
            f"| arch_mode={cand['arch_mode']} "
            f"| families={cand['selected_families']} "
            f"| attn={cand['transformer_self_attention_type']} "
            f"| ffn={cand['transformer_ffn_variant']}",
            flush=True,
        )
        results = trainer.train_darts_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=delta_epochs,
            use_swa=False,
            use_amp=bool(args.amp and args.device.startswith("cuda")),
            verbose=bool(args.verbose),
            compute_metrics=False,
            op_gdas=not args.no_gdas,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )
        model = results["model"]
        epochs_trained = int(rung_epoch)
        best_val = float(results["best_val_loss"])
        print(
            f"[P3] completed rung {rung_idx + 1} candidate id={cand['candidate_id']} "
            f"| mixed_val_loss={best_val:.6f}",
            flush=True,
        )

    print(f"[debug] done best_val_loss={best_val:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
