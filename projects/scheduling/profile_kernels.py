"""Numba kernels for the profile DPs and the constellation row checks.

Each kernel follows its numpy reference (job_profile.best_profile,
constellation.best_profile_multi, _best_profile_multi_separated,
antunes_profile_multi, validate_constellation) in evaluation order and
tie-breaking, so both return the same rows and values; the numpy versions run
with compiled=False. ONTS_NUMBA=0 makes numpy the default.
"""

from __future__ import annotations

import os

import numpy as np
from numba import njit

ENABLED = os.environ.get("ONTS_NUMBA", "1") != "0"


def use(compiled):
    return ENABLED if compiled is None else compiled


@njit(cache=True)
def _log2_table(n):
    out = np.zeros(n + 2, np.int64)
    for i in range(2, n + 2):
        out[i] = out[i // 2] + 1
    return out


@njit(cache=True)
def _sparse(values):
    """Sparse table of range maxima, as job_profile._range_max."""
    T = values.shape[0]
    levels, width = 1, 1
    while 2 * width <= T:
        levels += 1
        width *= 2
    table = np.full((levels, T), -np.inf)
    table[0] = values
    width = 1
    for level in range(1, levels):
        for i in range(T - 2 * width + 1):
            a, b = table[level - 1, i], table[level - 1, i + width]
            table[level, i] = a if a > b else b
        width *= 2
    return table


@njit(cache=True)
def _query(table, log2, a, b):
    level = log2[b - a + 1]
    x, y = table[level, a], table[level, b - (1 << level) + 1]
    return x if x > y else y


@njit(cache=True)
def _prefix_next(allowed, reward):
    """Prefix sums of allowed rewards per row and first forbidden slot at or after t."""
    S, T = allowed.shape
    prefix = np.zeros((S, T + 1))
    nxt = np.full((S, T + 1), T, np.int64)
    for s in range(S):
        for t in range(T):
            prefix[s, t + 1] = prefix[s, t] + (reward[s, t] if allowed[s, t] else 0.0)
        for t in range(T - 1, -1, -1):
            nxt[s, t] = t if not allowed[s, t] else nxt[s, t + 1]
    return prefix, nxt


@njit(cache=True)
def best_profile(reward, allowed, min_starts, max_starts, min_run, max_run, min_gap,
                 max_gap, truncated):
    """Kernel of job_profile.best_profile. Returns (found, row, value)."""
    T = reward.shape[0]
    row = np.zeros(T, np.bool_)
    prefix2, nxt2 = _prefix_next(allowed.reshape(1, T), reward.reshape(1, T))
    prefix, nxt = prefix2[0], nxt2[0]
    periodic = max_gap <= T
    spacing = max(1, min_gap, min_run + 1)
    K = min(max_starts, 1 + (T - 1) // spacing)
    if min_starts > K:
        return False, row, -np.inf
    log2 = _log2_table(T)
    value = np.full((K + 2, T), -np.inf)
    run_length = np.zeros((K + 2, T), np.int64)
    max_length = min(max_run, T)
    table = np.zeros((1, 1))
    for k in range(K, 0, -1):
        if k < K:
            table = _sparse(value[k + 1].copy())
        for u in range(T):
            if not allowed[u]:
                continue
            terminal = 0.0 if ((not periodic or u >= T - max_gap) and k >= min_starts) else -np.inf
            shortest = min(min_run, T - u) if truncated else min_run
            longest = min(max_run, T - u, nxt[u] - u)
            hi = min(u + max_gap, T - 1) if periodic else T - 1
            for length in range(max(1, shortest), min(max_length, longest) + 1):
                end = min(u + length, T)
                total = prefix[end] - prefix[u]
                cont = terminal
                if k < K:
                    lo = max(end + 1, u + min_gap)
                    if lo <= hi:
                        nextv = _query(table, log2, lo, hi)
                        if nextv > cont:
                            cont = nextv
                total = total + cont
                if total > value[k, u]:
                    value[k, u] = total
                    run_length[k, u] = length
    best, first = -np.inf, -1
    if min_starts == 0 and not periodic:
        best = 0.0
    last_first = min(T - 1, max_gap - 1) if periodic else T - 1
    for u in range(last_first + 1):
        if value[1, u] > best:
            best, first = value[1, u], u
    if not np.isfinite(best):
        return False, row, -np.inf
    k, s = 1, first
    while s >= 0:
        length = run_length[k, s]
        row[s : s + length] = True
        end = s + length
        stop = 0.0 if ((not periodic or s >= T - max_gap) and k >= min_starts) else -np.inf
        following = -1
        if k < K:
            lo = max(end + 1, s + min_gap)
            hi = min(s + max_gap, T - 1) if periodic else T - 1
            if lo <= hi:
                top, arg = value[k + 1, lo], lo
                for v in range(lo + 1, hi + 1):
                    if value[k + 1, v] > top:
                        top, arg = value[k + 1, v], v
                if top > stop:
                    following = arg
        k, s = k + 1, following
    return True, row, best


@njit(cache=True)
def best_profile_multi(rewards, allowed, min_starts, max_starts, min_run, max_run,
                       min_gap, max_gap, truncated):
    """Kernel of constellation.best_profile_multi (adjacent switches).

    Returns (found, row, owner, value)."""
    S, T = rewards.shape
    row = np.zeros(T, np.bool_)
    owner = np.full(T, -1, np.int64)
    prefix, nxt = _prefix_next(allowed, rewards)
    periodic = max_gap <= T
    spacing = max(1, min_gap, min_run)
    K = min(max_starts, 1 + (T - 1) // spacing)
    if min_starts > K:
        return False, row, owner, -np.inf
    log2 = _log2_table(T)
    value = np.full((K + 2, S, T), -np.inf)
    run_length = np.zeros((K + 2, S, T), np.int64)
    max_length = min(max_run, T)
    best1 = np.full(T, -np.inf)
    best2 = np.full(T, -np.inf)
    arg1 = np.zeros(T, np.int64)
    table = np.zeros((1, 1))
    for k in range(K, 0, -1):
        if k < K:
            for v in range(T):
                b1, a1 = value[k + 1, 0, v], 0
                for s in range(1, S):
                    if value[k + 1, s, v] > b1:
                        b1, a1 = value[k + 1, s, v], s
                b2 = -np.inf
                for s in range(S):
                    if s != a1 and value[k + 1, s, v] > b2:
                        b2 = value[k + 1, s, v]
                best1[v], arg1[v], best2[v] = b1, a1, b2
            table = _sparse(best1.copy())
        for u in range(T):
            terminal = 0.0 if ((not periodic or u >= T - max_gap) and k >= min_starts) else -np.inf
            shortest = min(min_run, T - u) if truncated else min_run
            hi = min(u + max_gap, T - 1) if periodic else T - 1
            for s in range(S):
                if not allowed[s, u]:
                    continue
                longest = min(max_run, T - u, nxt[s, u] - u)
                for length in range(max(1, shortest), min(max_length, longest) + 1):
                    end = min(u + length, T)
                    run = prefix[s, end] - prefix[s, u]
                    cont = terminal
                    if k < K:
                        lo = max(end + 1, u + min_gap)
                        if lo <= hi:
                            idle = _query(table, log2, lo, hi)
                            if idle > cont:
                                cont = idle
                        if length >= min_gap and (not periodic or length <= max_gap):
                            nu = u + length
                            if nu <= T - 1:
                                other = best2[nu] if s == arg1[nu] else best1[nu]
                                if other > cont:
                                    cont = other
                    total = run + cont
                    if total > value[k, s, u]:
                        value[k, s, u] = total
                        run_length[k, s, u] = length
    best, first, first_sat = -np.inf, -1, -1
    if min_starts == 0 and not periodic:
        best = 0.0
    last_first = min(T - 1, max_gap - 1) if periodic else T - 1
    for u in range(last_first + 1):
        top, arg = value[1, 0, u], 0
        for s in range(1, S):
            if value[1, s, u] > top:
                top, arg = value[1, s, u], s
        if top > best:
            best, first, first_sat = top, u, arg
    if not np.isfinite(best):
        return False, row, owner, -np.inf
    k, s, u = 1, first_sat, first
    while u >= 0:
        length = run_length[k, s, u]
        row[u : u + length] = True
        owner[u : u + length] = s
        end = u + length
        stop = 0.0 if ((not periodic or u >= T - max_gap) and k >= min_starts) else -np.inf
        nxt_u, nxt_s = -1, -1
        if k < K:
            switch, switch_s = -np.inf, -1
            if end <= T - 1 and length >= min_gap and (not periodic or length <= max_gap):
                for o in range(S):
                    if o != s and (switch_s < 0 or value[k + 1, o, end] > switch):
                        switch, switch_s = value[k + 1, o, end], o
            lo = max(end + 1, u + min_gap)
            hi = min(u + max_gap, T - 1) if periodic else T - 1
            idle, idle_u = -np.inf, -1
            for v in range(lo, hi + 1):
                later = value[k + 1, 0, v]
                for o in range(1, S):
                    if value[k + 1, o, v] > later:
                        later = value[k + 1, o, v]
                if idle_u < 0 or later > idle:
                    idle, idle_u = later, v
            if max(switch, idle) > stop:
                if switch >= idle:
                    nxt_u, nxt_s = end, switch_s
                else:
                    nxt_u = idle_u
                    top, nxt_s = value[k + 1, 0, nxt_u], 0
                    for o in range(1, S):
                        if value[k + 1, o, nxt_u] > top:
                            top, nxt_s = value[k + 1, o, nxt_u], o
        k, s, u = k + 1, nxt_s, nxt_u
    return True, row, owner, best


@njit(cache=True)
def best_profile_multi_separated(rewards, allowed, min_starts, max_starts, min_run,
                                 max_run, min_gap, max_gap, truncated):
    """Kernel of constellation._best_profile_multi_separated.

    Returns (found, row, owner, value)."""
    S, T = rewards.shape
    row = np.zeros(T, np.bool_)
    owner = np.full(T, -1, np.int64)
    prefix, nxt = _prefix_next(allowed, rewards)
    periodic = max_gap <= T
    spacing = max(1, min_gap, min_run + 1)
    K = min(max_starts, 1 + (T - 1) // spacing)
    if min_starts > K:
        return False, row, owner, -np.inf
    log2 = _log2_table(T)
    value = np.full((K + 2, T), -np.inf)
    run_length = np.zeros((K + 2, T), np.int64)
    run_sat = np.zeros((K + 2, T), np.int64)
    max_length = min(max_run, T)
    table = np.zeros((1, 1))
    for k in range(K, 0, -1):
        if k < K:
            table = _sparse(value[k + 1].copy())
        for u in range(T):
            terminal = 0.0 if ((not periodic or u >= T - max_gap) and k >= min_starts) else -np.inf
            shortest = min(min_run, T - u) if truncated else min_run
            hi = min(u + max_gap, T - 1) if periodic else T - 1
            for length in range(max(1, shortest), max_length + 1):
                end = min(u + length, T)
                run, sat = -np.inf, 0
                for s in range(S):
                    if (allowed[s, u] and length <= min(max_run, T - u)
                            and length <= nxt[s, u] - u):
                        r = prefix[s, end] - prefix[s, u]
                        if r > run:
                            run, sat = r, s
                if not np.isfinite(run):
                    continue
                cont = terminal
                if k < K:
                    lo = max(end + 1, u + min_gap)
                    if lo <= hi:
                        nextv = _query(table, log2, lo, hi)
                        if nextv > cont:
                            cont = nextv
                total = run + cont
                if total > value[k, u]:
                    value[k, u] = total
                    run_length[k, u] = length
                    run_sat[k, u] = sat
    best, first = -np.inf, -1
    if min_starts == 0 and not periodic:
        best = 0.0
    last_first = min(T - 1, max_gap - 1) if periodic else T - 1
    for u in range(last_first + 1):
        if value[1, u] > best:
            best, first = value[1, u], u
    if not np.isfinite(best):
        return False, row, owner, -np.inf
    k, s = 1, first
    while s >= 0:
        length = run_length[k, s]
        row[s : s + length] = True
        owner[s : s + length] = run_sat[k, s]
        end = s + length
        stop = 0.0 if ((not periodic or s >= T - max_gap) and k >= min_starts) else -np.inf
        following = -1
        if k < K:
            lo = max(end + 1, s + min_gap)
            hi = min(s + max_gap, T - 1) if periodic else T - 1
            if lo <= hi:
                top, arg = value[k + 1, lo], lo
                for v in range(lo + 1, hi + 1):
                    if value[k + 1, v] > top:
                        top, arg = value[k + 1, v], v
                if top > stop:
                    following = arg
        k, s = k + 1, following
    return True, row, owner, best


@njit(cache=True)
def antunes_profile_multi(gain, Z, W, P, min_starts, min_run, min_gap, periodic, truncated):
    """Kernel of constellation.antunes_profile_multi. Returns (found, row, owner, value)."""
    S, T = gain.shape
    row = np.zeros(T, np.bool_)
    owner = np.full(T, -1, np.int64)
    passed = np.empty(P + 1, np.int64)
    for p in range(P + 1):
        passed[p] = (p + 1 if p + 1 <= P else -1) if periodic else min(p + 1, P)
    V = np.full((T + 1, Z + 1, W + 1, S, P + 1), -np.inf)
    for z in range(Z + 1):
        if truncated or z == 0 or z >= min_run:
            for w in range(min_starts, W + 1):
                V[T, z, w, :, 1:] = 0.0
    fresh = np.full((W + 1, S), -np.inf)
    best1 = np.full(W + 1, -np.inf)
    best2 = np.full(W + 1, -np.inf)
    arg1 = np.zeros(W + 1, np.int64)
    for t in range(T - 1, -1, -1):
        _fresh(V[t + 1], gain[:, t], fresh, best1, best2, arg1)
        for z in range(Z + 1):
            for w in range(W + 1):
                for s in range(S):
                    for p in range(P + 1):
                        idle, cont, start = _actions(V[t + 1], gain[:, t], passed, fresh,
                                                     best1, best2, arg1, z, w, s, p, Z,
                                                     min_run, min_gap)
                        top = idle if idle > cont else cont
                        V[t, z, w, s, p] = top if top > start else start
    best = V[0, 0, 0, 0, 1]
    if not np.isfinite(best):
        return False, row, owner, -np.inf
    z, w, s, p = 0, 0, 0, 1
    for t in range(T):
        _fresh(V[t + 1], gain[:, t], fresh, best1, best2, arg1)
        idle, cont, start = _actions(V[t + 1], gain[:, t], passed, fresh, best1, best2,
                                     arg1, z, w, s, p, Z, min_run, min_gap)
        if idle >= max(cont, start):
            z, p = 0, passed[p]
        elif cont >= start:
            row[t], owner[t] = True, s
            z, p = z + 1, passed[p]
        else:
            top, arg = -np.inf, -1
            for o in range(S):
                if (z == 0 or o != s) and (arg < 0 or fresh[w, o] > top):
                    top, arg = fresh[w, o], o
            s = arg
            row[t], owner[t] = True, s
            z, w, p = 1, w + 1, 1
    return True, row, owner, best


@njit(cache=True)
def _fresh(nxt, gain, fresh, best1, best2, arg1):
    """Value of starting at the slot from w starts, per satellite; best and runner-up."""
    W1, S = fresh.shape
    for w in range(W1):
        for s in range(S):
            fresh[w, s] = gain[s] + nxt[1, w + 1, s, 1] if w + 1 < W1 else -np.inf
        b1, a1 = fresh[w, 0], 0
        for s in range(1, S):
            if fresh[w, s] > b1:
                b1, a1 = fresh[w, s], s
        b2 = -np.inf
        for s in range(S):
            if s != a1 and fresh[w, s] > b2:
                b2 = fresh[w, s]
        best1[w], best2[w], arg1[w] = b1, b2, a1


@njit(cache=True)
def _actions(nxt, gain, passed, fresh, best1, best2, arg1, z, w, s, p, Z, min_run, min_gap):
    idle = cont = start = -np.inf
    q = passed[p]
    if q >= 0:
        if z == 0 or z >= min_run:
            idle = nxt[0, w, s, q]
        if 1 <= z <= Z - 1:
            cont = nxt[z + 1, w, s, q] + gain[s]
    if w == 0 or p >= min_gap:
        if z == 0:
            start = best1[w]
        elif z >= min_run:
            start = best2[w] if s == arg1[w] else best1[w]
    return idle, cont, start


@njit(cache=True)
def _emit(out, n_out, kind, a, b, t):
    out[n_out, 0], out[n_out, 1], out[n_out, 2], out[n_out, 3] = kind, a, b, t
    return n_out + 1


@njit(cache=True)
def _period(phi, lo, hi, pmin, pmax, out, n_out, code, a, b):
    """Count and period checks of constellation._count_period, appended as codes."""
    T = phi.shape[0]
    cum = np.zeros(T + 1, np.int64)
    for t in range(T):
        cum[t + 1] = cum[t] + phi[t]
    n = cum[T]
    if not (lo <= n <= hi):
        n_out = _emit(out, n_out, code, a, b, 0)
    bad = n > 1 if pmin > T else False
    if pmin <= T:
        for i in range(T + 1 - pmin):
            if cum[i + pmin] - cum[i] > 1:
                bad = True
                break
    if bad:
        n_out = _emit(out, n_out, code + 1, a, b, 0)
    if pmax <= T:
        for i in range(T + 1 - pmax):
            if cum[i + pmax] - cum[i] < 1:
                n_out = _emit(out, n_out, code + 2, a, b, 0)
                break
    return n_out


@njit(cache=True)
def row_violations(X, Js, w_min, w_max, f_min, f_max, sat_period, con_period):
    """Window, run, count, period and exclusivity checks of validate_constellation.

    X [S, R, T]; w_min, w_max, f_min, f_max [S, R]; sat_period [S, Js, 4] and
    con_period [Jc, 4] hold (count min, count max, p_min, p_max). Returns codes
    [n, 4] = (kind, s or c, r, t): 0 window, 1 run, 2-4 satellite count/min/max
    period, 5 exclusive, 6-8 constellation count/min/max period.
    """
    S, R, T = X.shape
    out = np.zeros((S * R * (T + 4) + (R - Js) * 4, 4), np.int64)
    n_out = 0
    phi = np.zeros((S, R, T), np.int64)
    for s in range(S):
        for r in range(R):
            for t in range(T):
                phi[s, r, t] = 1 if X[s, r, t] and (t == 0 or not X[s, r, t - 1]) else 0
            lo, hi = w_min[s, r], max(w_min[s, r], w_max[s, r] - 1)
            for t in range(T):
                if X[s, r, t] and not (lo <= t < hi):
                    n_out = _emit(out, n_out, 0, s, r, 0)
                    break
            t = 0
            while t < T:
                if X[s, r, t]:
                    a = t
                    while t < T and X[s, r, t]:
                        t += 1
                    if t - a < min(f_min[s, r], T - a) or t - a > f_max[s, r]:
                        n_out = _emit(out, n_out, 1, s, r, a)
                else:
                    t += 1
            if r < Js:
                q = sat_period[s, r]
                n_out = _period(phi[s, r], q[0], q[1], q[2], q[3], out, n_out, 2, s, r)
    for c in range(R - Js):
        r = Js + c
        union = np.zeros(T, np.int64)
        exclusive = True
        for t in range(T):
            busy = 0
            for s in range(S):
                union[t] += phi[s, r, t]
                busy += 1 if X[s, r, t] else 0
            exclusive &= busy <= 1
        if not exclusive:
            n_out = _emit(out, n_out, 5, c, r, 0)
        q = con_period[c]
        n_out = _period(union, q[0], q[1], q[2], q[3], out, n_out, 6, c, r)
    return out[:n_out]


def warm_up():
    """Compile (or load from cache) every kernel on a tiny input, outside any budget."""
    if not ENABLED:
        return
    reward = np.ones(6)
    allowed = np.ones(6, np.bool_)
    rewards = np.ones((2, 6))
    allowed2 = np.ones((2, 6), np.bool_)
    best_profile(reward, allowed, 1, 2, 1, 2, 1, 7, True)
    best_profile_multi(rewards, allowed2, 1, 2, 1, 2, 1, 7, True)
    best_profile_multi_separated(rewards, allowed2, 1, 2, 1, 2, 1, 7, True)
    antunes_profile_multi(rewards, 2, 2, 1, 1, 1, 1, False, True)
    X = np.zeros((2, 2, 6), np.bool_)
    ones = np.ones((2, 2), np.int64)
    row_violations(X, 1, 0 * ones, 7 * ones, ones, 2 * ones,
                   np.tile(np.array([0, 2, 1, 7], np.int64), (2, 1, 1)),
                   np.array([[0, 2, 1, 7]], np.int64))
