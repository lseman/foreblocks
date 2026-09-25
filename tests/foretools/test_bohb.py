import math

import numpy as np
import pytest

from foretools.bohb import BOHB
from foretools.bohb.mftpe import MultiFidelityTPE
from foretools.bohb.trial import Trial

SPACE = {
    "lr": ("float", (1e-4, 1e-1, "log")),
    "x": ("float", (-2.0, 2.0)),
    "n": ("int", (1, 1000)),
    "k": ("int", (0, 4)),
    "act": ("choice", ["relu", "tanh", "gelu"]),
}


def _objective(cfg, budget):
    v = (math.log10(cfg["lr"]) + 2) ** 2 + cfg["x"] ** 2
    v += (math.log10(cfg["n"]) - 2) ** 2 + 0.1 * (cfg["k"] - 2) ** 2
    v += {"relu": 0.2, "tanh": 0.5, "gelu": 0.0}[cfg["act"]]
    return v + 1.0 / budget


def _in_space(cfg):
    assert 1e-4 <= cfg["lr"] <= 1e-1
    assert -2.0 <= cfg["x"] <= 2.0
    assert isinstance(cfg["n"], int) and 1 <= cfg["n"] <= 1000
    assert isinstance(cfg["k"], int) and 0 <= cfg["k"] <= 4
    assert cfg["act"] in {"relu", "tanh", "gelu"}


def test_mftpe_suggestions_respect_space_and_are_unique():
    tpe = MultiFidelityTPE(SPACE, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        cfg = tpe.suggest(1)[0]
        _in_space(cfg)
        tpe.observe(cfg, _objective(cfg, 9.0) + rng.normal(0, 0.01), budget=9.0)
    cfgs, scores = tpe.suggest(20, return_scores=True)
    assert len(cfgs) == len(scores) == 20
    for cfg in cfgs:
        _in_space(cfg)
    keys = {tuple(sorted(c.items())) for c in cfgs}
    assert len(keys) == 20
    assert tpe.diagnostics()["model_budget"] == 9.0


def test_mftpe_models_largest_populated_budget():
    tpe = MultiFidelityTPE(SPACE, seed=0, min_points_in_model=5)
    for i in range(10):
        tpe.observe(tpe.suggest(1)[0], float(i), budget=1.0)
    for i in range(3):
        tpe.observe(tpe.suggest(1)[0], float(i), budget=9.0)
    tpe.suggest(2)
    assert tpe.diagnostics()["model_budget"] == 1.0
    for i in range(3):
        tpe.observe(tpe.suggest(1)[0], float(i), budget=9.0)
    tpe.suggest(2)
    assert tpe.diagnostics()["model_budget"] == 9.0


def test_mftpe_conditional_params():
    space = {
        "opt": ("choice", ["sgd", "adam"]),
        "momentum": (
            "float",
            (0.0, 0.99),
            {"condition": {"parent": "opt", "values": ["sgd"]}},
        ),
        "beta1": (
            "float",
            (0.8, 0.999),
            {"condition": {"parent": "opt", "values": ["adam"]}},
        ),
    }
    tpe = MultiFidelityTPE(space, seed=1)
    for _ in range(30):
        cfg = tpe.suggest(1)[0]
        assert ("momentum" in cfg) == (cfg["opt"] == "sgd")
        assert ("beta1" in cfg) == (cfg["opt"] == "adam")
        loss = cfg.get("momentum", 0.0) if cfg["opt"] == "sgd" else 1.0
        tpe.observe(cfg, loss, budget=1.0)
    for cfg in tpe.suggest(10):
        assert ("momentum" in cfg) == (cfg["opt"] == "sgd")


def test_mftpe_hard_constraints():
    tpe = MultiFidelityTPE(SPACE, seed=0, hard_constraints=[lambda c: c["x"]])
    for _ in range(30):
        cfg = tpe.suggest(1)[0]
        assert cfg["x"] <= 0
        tpe.observe(cfg, _objective(cfg, 1.0), budget=1.0)
    assert all(c["x"] <= 0 for c in tpe.suggest(10))


def test_mftpe_beats_random_on_quadratic():
    space = {f"x{i}": ("float", (-1.0, 1.0)) for i in range(4)}

    def f(c):
        return sum(c[f"x{i}"] ** 2 for i in range(4))

    tpe_best, rnd_best = [], []
    for seed in range(5):
        tpe = MultiFidelityTPE(space, seed=seed)
        for _ in range(60):
            cfg = tpe.suggest(1)[0]
            tpe.observe(cfg, f(cfg), budget=1.0)
        tpe_best.append(min(loss for _, loss, _ in tpe.observations))
        rng = np.random.default_rng(seed)
        rnd_best.append(min(np.sum(rng.uniform(-1, 1, (60, 4)) ** 2, axis=1)))
    assert np.mean(tpe_best) < 0.5 * np.mean(rnd_best)


@pytest.mark.parametrize("parallel_jobs", [1, 3])
def test_bohb_runs_and_follows_successive_halving(parallel_jobs):
    bohb = BOHB(
        SPACE,
        _objective,
        min_budget=1,
        max_budget=9,
        eta=3,
        n_iterations=2,
        verbose=False,
        seed=0,
        parallel_jobs=parallel_jobs,
    )
    cfg, loss = bohb.run()
    _in_space(cfg)
    assert math.isfinite(loss)
    # Bracket s=2 of the first iteration: 9 -> 3 -> 1 configs.
    first = [h for h in bohb.history if h["iteration"] == 0 and h["bracket"] == 2]
    counts = [sum(h["round"] == r for h in first) for r in range(3)]
    if parallel_jobs == 1:
        assert counts == [9, 3, 1]
    else:
        assert counts[0] == 9 and counts[1] >= 3 and counts[2] >= 1


def test_bohb_detects_trial_argument():
    def two_args(config, budget, rng=None):
        return 0.0

    def three_args(config, budget, trial):
        return 0.0

    def named_trial(config, budget, trial=None):
        return 0.0

    assert not BOHB._accepts_trial_argument(two_args)
    assert BOHB._accepts_trial_argument(three_args)
    assert BOHB._accepts_trial_argument(named_trial)


def test_bohb_step_pruning_feeds_sampler():
    def objective(config, budget, trial: Trial):
        loss = _objective(config, budget)
        for step in range(int(budget)):
            trial.report(step, loss + 1.0 / (step + 1))
        return loss

    bohb = BOHB(
        SPACE,
        objective,
        min_budget=1,
        max_budget=9,
        eta=3,
        n_iterations=3,
        verbose=False,
        seed=0,
        pruning_mode="aggressive",
        pruning_overrides={"step_min_history": 3},
    )
    bohb.run()
    assert len(bohb.tpe.observations) >= len(bohb.history)


def test_bohb_final_prune_keeps_observation():
    bohb = BOHB(
        SPACE,
        _objective,
        min_budget=1,
        max_budget=9,
        eta=3,
        n_iterations=2,
        verbose=False,
        seed=0,
        pruning_mode="aggressive",
    )
    bohb.run()
    # Every finished evaluation reaches both the history and the sampler.
    assert len(bohb.tpe.observations) == len(bohb.history)


def test_bohb_warm_start_roundtrip(tmp_path):
    path = tmp_path / "hist.jsonl"
    a = BOHB(
        SPACE,
        _objective,
        min_budget=1,
        max_budget=9,
        n_iterations=1,
        verbose=False,
        seed=0,
        history_export_jsonl=str(path),
    )
    a.run()
    b = BOHB(
        SPACE,
        _objective,
        min_budget=1,
        max_budget=9,
        n_iterations=1,
        verbose=False,
        seed=1,
        prior_trials_jsonl=str(path),
    )
    assert len(b.tpe.observations) == len(a.history)
    _, loss_b = b.run()
    assert loss_b <= a.best_loss + 1e-12


def test_bohb_legacy_sampler_still_works():
    bohb = BOHB(
        SPACE,
        _objective,
        min_budget=1,
        max_budget=9,
        n_iterations=1,
        verbose=False,
        seed=0,
        sampler="legacy",
    )
    cfg, loss = bohb.run()
    _in_space(cfg)
    assert math.isfinite(loss)
