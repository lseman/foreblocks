"""Time CPU multiclass fits. Use --compare to include XGBoost, LightGBM and CatBoost."""

import argparse
import statistics
import time

import numpy as np
import foreforest as ff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--save-predictions")
    args = parser.parse_args()
    classes, rounds, threads = 8, 12, 4
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(25000, 12))
    scores = x @ rng.normal(size=(12, classes)) + rng.normal(size=(len(x), classes))
    labels = scores.argmax(axis=1)
    train, test = x[:20000], x[20000:]
    y, target = labels[:20000], labels[20000:]

    cfg = ff.ForeForestConfig()
    cfg.mode = ff.Mode.GBDT
    cfg.device = ff.Device.CPU
    cfg.num_classes = classes
    cfg.n_estimators = rounds * (classes - 1)
    cfg.threads = threads
    cfg.learning_rate = 0.1
    cfg.tree_cfg.max_depth = 4
    cfg.tree_cfg.max_leaves = 16
    models = [("ForeTree", lambda: ff.ForeForest(cfg))]
    if args.compare:
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        common = dict(n_estimators=rounds, learning_rate=0.1, max_depth=4, n_jobs=threads)
        models.extend([
            ("LightGBM", lambda: LGBMClassifier(**common, num_leaves=16, verbosity=-1)),
            ("XGBoost", lambda: XGBClassifier(**common, tree_method="hist")),
            ("CatBoost", lambda: CatBoostClassifier(iterations=rounds, depth=4, learning_rate=0.1,
                                                    thread_count=threads, verbose=False, allow_writing_files=False)),
        ])

    for name, create in models:
        times = []
        fit_labels = y.astype(float) if name == "ForeTree" else y
        for run in range(args.repeats + 1):
            model = create()
            fit = model.fit_complete if name == "ForeTree" else model.fit
            start = time.perf_counter()
            fit(train, fit_labels)
            elapsed = time.perf_counter() - start
            if run:
                times.append(elapsed)
        probs = model.predict(test) if name == "ForeTree" else model.predict_proba(test)
        loss = -np.log(probs[np.arange(len(target)), target]).mean()
        print(f"{name} median {statistics.median(times):.3f}s, range {min(times):.3f}–{max(times):.3f}s, "
              f"held-out logloss {loss:.6f}", flush=True)
        if args.save_predictions and name == "ForeTree":
            np.save(args.save_predictions, probs)


if __name__ == "__main__":
    main()
