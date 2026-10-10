"""Scalar reference implementation of the per-job DP (pre-vectorization).

Kept to test that the vectorized best_profile returns identical rows and values.
"""

import math

import numpy as np

from projects.scheduling.job_profile import JobRules


def reference_best_profile(
    reward: np.ndarray, rules: JobRules, *, truncated_final_run: bool = True
) -> tuple[np.ndarray | None, float]:
    """Maximize sum(reward * row) over rows satisfying one job's constraints.

    reward[t] = -inf forbids slot t. Returns (None, -inf) if no profile exists.
    Maximum period P <= T means: first start <= P-1, consecutive starts at most
    P apart and last start >= T-P, which is equivalent to every full window of
    length P containing a start. Consecutive runs need an idle slot between them.
    """
    reward = np.asarray(reward, dtype=float)
    T = len(reward)
    allowed = np.isfinite(reward)
    allowed[: max(0, rules.win_min)] = False
    allowed[min(T, rules.win_max) :] = False
    # nxt[t]: first forbidden slot at or after t.
    nxt = np.full(T + 1, T)
    for t in range(T - 1, -1, -1):
        nxt[t] = t if not allowed[t] else nxt[t + 1]
    prefix = np.concatenate([[0.0], np.cumsum(np.where(allowed, reward, 0.0))])
    periodic = rules.max_gap <= T
    # No profile can use more starts than fit with idle separators.
    K = min(rules.max_starts, (T + 1) // 2)
    if rules.min_starts > K:
        return None, -math.inf
    value = np.full((K + 2, T), -math.inf)
    choice: dict[tuple[int, int], tuple[int, int]] = {}
    for k in range(K, 0, -1):
        following = value[k + 1] if k < K else None
        if following is not None:
            # Running maxima of following[a:] for every a, with argmax positions.
            best_after = [np.maximum.accumulate(following[a:]) for a in range(T)]
        terminal_ok = k >= rules.min_starts
        for s in range(T):
            if not allowed[s]:
                continue
            shortest = rules.min_run
            if truncated_final_run:
                shortest = min(shortest, T - s)
            elif s + shortest > T:
                continue
            longest = min(rules.max_run, T - s, nxt[s] - s)
            for length in range(shortest, longest + 1):
                end = s + length
                run = prefix[end] - prefix[s]
                cont, nxt_start = -math.inf, -1
                if terminal_ok and (not periodic or s >= T - rules.max_gap):
                    cont = 0.0
                if following is not None:
                    lo = max(end + 1, s + rules.min_gap)
                    hi = min(s + rules.max_gap, T - 1) if periodic else T - 1
                    if lo <= hi:
                        candidate = best_after[lo][hi - lo]
                        if candidate > cont:
                            cont = candidate
                            nxt_start = lo + int(np.argmax(following[lo : hi + 1]))
                if run + cont > value[k, s]:
                    value[k, s] = run + cont
                    choice[k, s] = (length, nxt_start)
    best, first = -math.inf, -1
    if rules.min_starts == 0 and not periodic:
        best = 0.0
    last_first = min(T - 1, rules.max_gap - 1) if periodic else T - 1
    for s in range(last_first + 1):
        if value[1, s] > best:
            best, first = value[1, s], s
    if not math.isfinite(best):
        return None, -math.inf
    row = np.zeros(T, dtype=bool)
    k, s = 1, first
    while s >= 0:
        length, following_start = choice[k, s]
        row[s : s + length] = True
        k, s = k + 1, following_start
    return row, float(best)
