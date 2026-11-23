"""CPU quantile comparison. Run with the built foreforest module on PYTHONPATH.

python tests/bench_quantile.py --seed 83 --compare
The optional comparison requires lightgbm and xgboost.
"""

import argparse

import numpy as np

import foreforest as ff


def sample(rng, n):
    x = rng.uniform(-1, 1, (n, 3))
    y = 25 * x[:, 0] + rng.normal(size=n) * (3 + 5 * (x[:, 1] > 0))
    return x, y


def report(model, tau, y, predictions):
    residual = y - predictions
    loss = np.maximum(tau * residual, (tau - 1) * residual).mean()
    print(f"{model:16} {tau:4.2f} {(residual <= 0).mean():9.4f} {loss:9.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    train, labels = sample(rng, 5000)
    test, targets = sample(rng, 5000)
    print("model             tau  coverage   pinball")
    for tau in (0.1, 0.9):
        cfg = ff.ForeForestConfig()
        cfg.mode = ff.Mode.GBDT
        cfg.device = ff.Device.CPU
        cfg.objective = ff.Objective.QuantileError
        cfg.quantile_tau = tau
        cfg.threads = 2
        cfg.n_estimators = 60
        cfg.learning_rate = 0.1
        cfg.tree_cfg.max_depth = 6
        cfg.tree_cfg.max_leaves = 31
        model = ff.ForeForest(cfg)
        model.fit_complete(train, labels)
        report("ForeTree", tau, targets, model.predict(test))

        if args.compare:
            import lightgbm as lgb
            import xgboost as xgb

            light = lgb.train(
                dict(objective="quantile", alpha=tau, max_depth=6, num_leaves=31,
                     learning_rate=0.1, num_threads=2, verbosity=-1),
                lgb.Dataset(train, label=labels), num_boost_round=60,
            )
            report(f"LightGBM {lgb.__version__}", tau, targets, light.predict(test, num_threads=2))
            boost = xgb.train(
                dict(objective="reg:quantileerror", quantile_alpha=tau, tree_method="hist",
                     grow_policy="lossguide", max_depth=6, max_leaves=31, eta=0.1, nthread=2),
                xgb.QuantileDMatrix(train, label=labels, nthread=2), num_boost_round=60,
            )
            report(f"XGBoost {xgb.__version__}", tau, targets, boost.inplace_predict(test))


if __name__ == "__main__":
    main()
