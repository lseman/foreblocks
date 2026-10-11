"""Numba kernels return the same rows, values and violations as the numpy references."""

import math

import numpy as np
import pytest

from projects.scheduling.constellation import (
    _best_profile_multi_separated,
    antunes_profile_multi,
    best_profile_multi,
    generate_constellation,
    solve_constellation,
    validate_constellation,
)
from projects.scheduling.job_profile import JobRules, best_profile


def random_rules(rng, T):
    smin, rmin, gmin = int(rng.integers(0, 4)), int(rng.integers(1, 5)), int(rng.integers(1, 6))
    return JobRules(
        smin,
        smin + int(rng.integers(0, 6)),
        rmin,
        rmin + int(rng.integers(0, 8)),
        gmin,
        gmin + int(rng.integers(0, T)) if rng.random() < 0.6 else T + 1,
        int(rng.integers(0, 4)),
        T - int(rng.integers(0, 4)),
    )


def same(a, b):
    if a[0] is None or b[0] is None:
        return a[0] is None and b[0] is None and a[-1] == b[-1] == -math.inf
    return all(np.array_equal(x, y) for x, y in zip(a[:-1], b[:-1])) and a[-1] == b[-1]


@pytest.mark.parametrize("truncated", [True, False])
def test_best_profile_kernel_matches_numpy(truncated):
    rng = np.random.default_rng(11)
    for _ in range(300):
        T = int(rng.integers(1, 60))
        rules = random_rules(rng, T)
        reward = rng.normal(size=T)
        reward[rng.random(T) < 0.15] = -math.inf
        if rng.random() < 0.3:
            reward = np.round(reward)  # ties
        a = best_profile(reward, rules, truncated_final_run=truncated, compiled=False)
        b = best_profile(reward, rules, truncated_final_run=truncated, compiled=True)
        assert same(a, b)


@pytest.mark.parametrize(
    "dp",
    [
        lambda r, ru, c: best_profile_multi(r, ru, compiled=c),
        lambda r, ru, c: _best_profile_multi_separated(r, ru, compiled=c),
        lambda r, ru, c: antunes_profile_multi(r, ru, compiled=c),
    ],
    ids=["switch", "separated", "state"],
)
def test_multi_kernels_match_numpy(dp):
    rng = np.random.default_rng(12)
    for _ in range(150):
        S, T = int(rng.integers(1, 5)), int(rng.integers(1, 40))
        rules = random_rules(rng, T)
        rewards = rng.normal(size=(S, T))
        rewards[rng.random((S, T)) < 0.15] = -math.inf
        if rng.random() < 0.3:
            rewards = np.round(rewards)
        assert same(dp(rewards, rules, False), dp(rewards, rules, True))


def test_validator_kernel_matches_numpy():
    rng = np.random.default_rng(13)
    for seed in range(4):
        inst = generate_constellation(S=3, Js=3, Jc=2, T=48, seed=seed)
        X = solve_constellation(inst, time_limit=1.5, patience=3, seed=seed)["schedule"]
        for trial in range(40):
            Y = X.copy() if X is not None else np.zeros((inst.S, inst.R, inst.T), bool)
            Y.ravel()[rng.integers(0, Y.size, size=int(rng.integers(0, 12)))] ^= True
            assert validate_constellation(inst, Y, compiled=False) == validate_constellation(
                inst, Y, compiled=True
            )
