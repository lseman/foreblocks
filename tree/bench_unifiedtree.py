#!/usr/bin/env python3
"""Small benchmark harness for UnifiedTree.

Requires:
- numpy
- foretree module built and importable

Optional:
- scikit-learn (for baseline comparison)
"""

from __future__ import annotations

import argparse
import time
import sys


def _need(mod: str):
    try:
        return __import__(mod)
    except Exception as e:
        print(f"missing dependency: {mod} ({e})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    np = _need("numpy")
    if np is None:
        return 2

    try:
        import foretree
    except Exception as e:
        print(f"failed to import foretree: {e}")
        return 2

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, args.p)).astype(np.float64)
    y = (X[:, 0] * 1.2 + X[:, 1] * X[:, 2] - 0.8 * X[:, 3] > 0.0).astype(np.float64)

    g = (0.5 - y).astype(np.float64)
    h = np.full(args.n, 1.0, dtype=np.float64)

    hcfg = foretree.HistogramConfig()
    hcfg.method = "adaptive"
    hcfg.max_bins = 256

    ghs = foretree.GradientHistogramSystem(hcfg)
    ghs.fit_bins(X, g, h)
    Xb, _ = ghs.prebin_dataset(X)

    runs = []
    for name, kway, oblique in [
        ("axis", False, False),
        ("kway", True, False),
        ("kway+oblique", True, True),
    ]:
        cfg = foretree.TreeConfig()
        cfg.max_depth = 6
        cfg.max_leaves = 31
        cfg.enable_kway_splits = kway
        cfg.kway_max_groups = 8
        cfg.enable_oblique_splits = oblique
        cfg.oblique_k_features = 4
        cfg.oblique_ridge = 1e-3
        cfg.axis_vs_oblique_guard = 1.02

        t0 = time.perf_counter()
        t = foretree.UnifiedTree(cfg, ghs)
        t.fit_binned(Xb, g, h)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = np.asarray(t.predict_binned(Xb), dtype=np.float64)
        pred_time = time.perf_counter() - t1

        score = float(np.mean((pred - y) ** 2))
        runs.append((name, train_time, pred_time, score, int(t.n_nodes), int(t.n_leaves)))

    print("name\ttrain_s\tpredict_s\tmse\tnodes\tleaves")
    for r in runs:
        print(f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.6f}\t{r[4]}\t{r[5]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
