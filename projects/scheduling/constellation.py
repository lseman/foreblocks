"""Constellation ONTS: instances, validation, MILP and profile-portfolio search.

Model of Antunes et al. (C&OR 185, 2026), with one deliberate difference: the
battery allows curtailment (i_t <= b_t / V, as in the reference single-satellite
model) instead of i_t = b_t / V. Satellite tasks run on their own satellite with
per-satellite startup counts and periods; constellation tasks may run on any
satellite, with startup counts and periods summed over satellites and at most
one satellite executing a task at a time. Windows and run lengths apply per
satellite to every task.

Schedules are boolean arrays X[s, r, t] with rows r < Js satellite tasks of
satellite s and rows r = Js + c constellation task c (on satellite s).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .job_profile import JobRules, _range_max, best_profile


@dataclass
class ConstellationInstance:
    solar: np.ndarray  # [S, T] watts
    soc0: np.ndarray  # [S]
    # Satellite tasks, arrays [S, Js]; constellation tasks, arrays [Jc].
    sat: dict[str, np.ndarray]
    con: dict[str, np.ndarray]
    ef: float = 0.9
    v_bat: float = 3.6
    q: float = 5.0
    bat_usage: float = 5.0
    soc_min: float = 0.0

    @property
    def S(self):
        return self.solar.shape[0]

    @property
    def T(self):
        return self.solar.shape[1]

    @property
    def Js(self):
        return self.sat["u"].shape[1]

    @property
    def Jc(self):
        return len(self.con["u"])

    @property
    def R(self):
        return self.Js + self.Jc

    @property
    def k(self):
        return self.ef / (self.q * self.v_bat * 60)

    def row(self, s, r, field):
        return self.sat[field][s, r] if r < self.Js else self.con[field][r - self.Js]

    def utility(self):
        """Per-row utility matrix [S, R]."""
        u = np.empty((self.S, self.R))
        u[:, : self.Js] = self.sat["u"]
        u[:, self.Js :] = self.con["u"][None, :]
        return u

    def draw(self):
        d = np.empty((self.S, self.R))
        d[:, : self.Js] = self.sat["q"]
        d[:, self.Js :] = self.con["q"][None, :]
        return d

    def rules(self, s, r):
        """Per-row DP rules; constellation rules use cross-satellite counts."""
        lo, hi = ("y_min", "y_max") if r < self.Js else ("g_min", "g_max")
        get = lambda f: int(self.row(s, r, f))  # noqa: E731
        return JobRules(
            get(lo),
            get(hi),
            get("f_min"),
            get("f_max"),
            get("p_min"),
            get("p_max"),
            get("w_min"),
            get("w_max") - 1,
        )


def _solar_profiles():
    from .research.realistic import _pools

    return [p for pool in _pools().values() for p in pool["solar"]]


def generate_constellation(S, Js, Jc, T, seed):
    """Algorithm 3 of Antunes et al. with bundled orbit solar profiles.

    Tasks whose own constraints admit no profile are redrawn (count stored in
    ``inst.redraws``); energy feasibility of the whole instance is not filtered.

    Solar power: one bundled (FloripaSat-derived) orbit profile resampled to T
    slots and cyclically shifted per satellite (different orbital phase).
    """
    rng = np.random.default_rng(seed)
    J = Js + Jc
    du = lambda lo, hi, size: rng.integers(lo, np.maximum(lo, hi) + 1, size=size)  # noqa: E731

    def tasks(size, kind):
        out = _draw_tasks(size, kind)
        # Redraw tasks whose own constraints admit no profile, decided exactly by
        # the per-task DP (a necessary condition independent of any search).
        flat = int(np.prod(size))
        for i in range(flat):
            for attempt in range(1000):
                idx = np.unravel_index(i, size)
                rules = JobRules(
                    int(out["y_min" if kind == "sat" else "g_min"][idx]),
                    int(out["y_max" if kind == "sat" else "g_max"][idx]),
                    int(out["f_min"][idx]),
                    int(out["f_max"][idx]),
                    int(out["p_min"][idx]),
                    int(out["p_max"][idx]),
                    int(out["w_min"][idx]),
                    int(out["w_max"][idx]) - 1,
                )
                if best_profile(np.ones(T), rules)[0] is not None:
                    break
                fresh = _draw_tasks((1,), kind)
                for key in out:
                    out[key][idx] = fresh[key][0]
                redraws[0] += 1
        return out

    redraws = [0]

    def _draw_tasks(size, kind):
        f_min = du(1, math.ceil(T / 10), size)
        out = dict(
            u=rng.uniform(1, J, size=size),
            q=rng.uniform(0.5, 2.5, size=size),
            f_min=f_min,
            f_max=du(f_min, math.ceil(T / 4), size),
            p_min=du(f_min, math.ceil(T / 4), size),
            w_min=du(0, math.ceil(T / 5), size),
            w_max=du(math.floor(T - T / 5), T, size),
        )
        out["p_max"] = du(out["p_min"], T, size)
        if kind == "sat":
            out["y_min"] = du(1, math.ceil(T / 50), size)
            out["y_max"] = du(out["y_min"], math.ceil(T / 7), size)
        else:
            out["g_min"] = du(1, math.ceil(T / 45), size)
            out["g_max"] = du(out["g_min"], math.ceil(T / 7), size)
        return out

    profiles = _solar_profiles()
    base = profiles[int(rng.integers(len(profiles)))]
    grid = np.linspace(0, len(base) - 1, T)
    resampled = np.interp(grid, np.arange(len(base)), base)
    solar = np.stack([np.roll(resampled, int(rng.integers(T))) for _ in range(S)])
    sat, con = tasks((S, Js), "sat"), tasks((Jc,), "con")
    inst = ConstellationInstance(
        solar=solar, soc0=rng.uniform(0.4, 0.9, size=S), sat=sat, con=con
    )
    inst.redraws = redraws[0]
    return inst


def battery(inst, usage):
    """Maximal SoC per satellite with curtailment; usage [S, T]."""
    cumulative = np.cumsum(inst.k * (inst.solar - usage), axis=1)
    return cumulative + np.minimum(
        inst.soc0[:, None], 1 - np.maximum.accumulate(cumulative, axis=1)
    )


def _starts(row):
    prev = np.concatenate([[False], row[:-1]])
    return row & ~prev


def validate_constellation(inst, X, *, tol=1e-6):
    """Independent check of constraints (2)-(13), curtailing battery."""
    X = np.asarray(X, dtype=bool)
    S, R, T = inst.S, inst.R, inst.T
    if X.shape != (S, R, T):
        raise ValueError(f"Expected schedule shape {(S, R, T)}")
    bad = []
    phi = np.zeros_like(X)
    for s in range(S):
        for r in range(R):
            row = X[s, r]
            phi[s, r] = _starts(row)
            wmin, wmax = int(inst.row(s, r, "w_min")), int(inst.row(s, r, "w_max"))
            allowed = np.zeros(T, bool)
            allowed[wmin : max(wmin, wmax - 1)] = True
            if (row & ~allowed).any():
                bad.append(f"window s={s} r={r}")
            fmin, fmax = int(inst.row(s, r, "f_min")), int(inst.row(s, r, "f_max"))
            edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a < min(fmin, T - a) or b - a > fmax:
                    bad.append(f"run s={s} r={r} t={a}")
            if r < inst.Js:
                bad += _count_period(
                    phi[s, r],
                    int(inst.sat["y_min"][s, r]),
                    int(inst.sat["y_max"][s, r]),
                    int(inst.sat["p_min"][s, r]),
                    int(inst.sat["p_max"][s, r]),
                    f"s={s} r={r}",
                )
    for c in range(inst.Jc):
        r = inst.Js + c
        union = phi[:, r].sum(axis=0)
        if (X[:, r].sum(axis=0) > 1).any():
            bad.append(f"exclusive c={c}")
        bad += _count_period(
            union,
            int(inst.con["g_min"][c]),
            int(inst.con["g_max"][c]),
            int(inst.con["p_min"][c]),
            int(inst.con["p_max"][c]),
            f"c={c}",
        )
    draw = inst.draw()
    usage = np.einsum("sr,srt->st", draw, X.astype(float))
    if (usage > inst.solar + inst.bat_usage * inst.v_bat + tol).any():
        bad.append("power")
    if (battery(inst, usage) < inst.soc_min - tol).any():
        bad.append("battery")
    objective = float((inst.utility()[:, :, None] * X).sum())
    return objective, bad


def _count_period(phi, lo, hi, pmin, pmax, label):
    out = []
    n = int(phi.sum())
    T = len(phi)
    if not lo <= n <= hi:
        out.append(f"count {label}")
    cum = np.concatenate([[0], np.cumsum(phi)])
    if pmin <= T and (cum[pmin:] - cum[:-pmin] > 1).any():
        out.append(f"min period {label}")
    if pmin > T and n > 1:
        out.append(f"min period {label}")
    if pmax <= T and (cum[pmax:] - cum[:-pmax] < 1).any():
        out.append(f"max period {label}")
    return out


class ConstellationMilp:
    """Sparse MILP over a subset of satellites; other satellites are fixed.

    With all satellites included this is the full model. With one satellite and
    a fixed schedule elsewhere, cross-satellite rows of constellation tasks become
    right-hand-side adjustments, giving the per-satellite repair submodel.
    """

    def __init__(self, inst, sats=None, fixed=None):
        self.inst = inst
        self.sats = list(range(inst.S)) if sats is None else list(sats)
        S, R, T = len(self.sats), inst.R, inst.T
        self.shape = (S, R, T)
        n_x = S * R * T
        self.n_x = n_x
        self.n = 2 * n_x + S * T
        others = [s for s in range(inst.S) if s not in self.sats]
        fixed = np.zeros((inst.S, R, T), bool) if fixed is None else fixed
        rows, cols, vals, lo, hi = [], [], [], [], []
        count = [0]

        def X(si, r, t):
            return (si * R + r) * T + t

        def add(c_idx, c_val, lower, upper):
            """c_idx [m, w] column indices per row, c_val broadcastable."""
            c_idx = np.atleast_2d(c_idx)
            m = c_idx.shape[0]
            c_val = np.broadcast_to(c_val, c_idx.shape)
            rows.append(np.repeat(np.arange(count[0], count[0] + m), c_idx.shape[1]))
            cols.append(c_idx.ravel())
            vals.append(np.asarray(c_val, float).ravel())
            lo.append(np.broadcast_to(np.asarray(lower, float), m))
            hi.append(np.broadcast_to(np.asarray(upper, float), m))
            count[0] += m

        self.lower = np.zeros(self.n)
        self.upper = np.ones(self.n)
        self.lower[2 * n_x :] = inst.soc_min
        self.integrality = np.r_[np.ones(2 * n_x), np.zeros(S * T)]
        u = inst.utility()
        self.c = np.zeros(self.n)
        t_idx = np.arange(T)
        for si, s in enumerate(self.sats):
            self.c[X(si, 0, 0) : X(si, R - 1, T - 1) + 1] = -np.repeat(u[s], T)
            for r in range(R):
                base = X(si, r, 0)
                xs = base + t_idx
                ph = n_x + xs
                wmin = int(inst.row(s, r, "w_min"))
                wmax = int(inst.row(s, r, "w_max"))
                allowed = (t_idx >= wmin) & (t_idx < wmax - 1)
                self.upper[xs[~allowed]] = 0
                # Startup detection (2a-2d).
                add(np.c_[ph[:1], xs[:1]], [1, -1], 0, 0)
                add(np.c_[ph[1:], xs[1:], xs[:-1]], [1, -1, 1], 0, np.inf)
                add(np.c_[ph[1:], xs[1:]], [1, -1], -np.inf, 0)
                add(np.c_[ph[1:], xs[:-1]], [1, 1], -np.inf, 1)
                # Minimum run (5a, 5c) with truncation at the horizon.
                fmin, fmax = int(inst.row(s, r, "f_min")), int(inst.row(s, r, "f_max"))
                full = np.arange(T - fmin + 1) if fmin <= T else np.arange(0)
                if len(full):
                    idx = np.c_[base + full[:, None] + np.arange(fmin), ph[full]]
                    add(idx, np.r_[np.ones(fmin), -fmin][None, :], 0, np.inf)
                for t in range(max(0, T - fmin + 1), T):
                    q = T - t
                    idx = np.r_[xs[t : t + q], ph[t]]
                    add(idx[None, :], np.r_[np.ones(q), -q], 0, np.inf)
                # Maximum run (5b).
                if fmax < T:
                    starts = np.arange(T - fmax)
                    add(base + starts[:, None] + np.arange(fmax + 1), 1, -np.inf, fmax)
                if r < inst.Js:
                    add(
                        ph[None, :], 1, inst.sat["y_min"][s, r], inst.sat["y_max"][s, r]
                    )
                    self._periods(
                        add,
                        ph[None, :],
                        inst.sat["p_min"][s, r],
                        inst.sat["p_max"][s, r],
                        np.zeros(T),
                    )
        # Constellation cross-satellite rows (11-13), fixed others as constants.
        for c in range(inst.Jc):
            r = inst.Js + c
            ph_rows = np.stack([n_x + X(si, r, 0) + t_idx for si in range(S)])
            x_rows = np.stack([X(si, r, 0) + t_idx for si in range(S)])
            fixed_phi = (
                sum(_starts(fixed[s, r]).astype(float) for s in others)
                if others
                else np.zeros(T)
            )
            fixed_x = fixed[others, r].sum(axis=0) if others else np.zeros(T)
            add(
                ph_rows.T.reshape(1, -1),
                1,
                inst.con["g_min"][c] - fixed_phi.sum(),
                inst.con["g_max"][c] - fixed_phi.sum(),
            )
            self._periods(
                add, ph_rows, inst.con["p_min"][c], inst.con["p_max"][c], fixed_phi
            )
            add(x_rows.T, 1, -np.inf, 1 - fixed_x)
        # Power (10) and battery with curtailment, per included satellite.
        draw = inst.draw()
        k = inst.k
        for si, s in enumerate(self.sats):
            cols_t = np.stack([X(si, r, 0) + t_idx for r in range(R)], axis=1)  # [T, R]
            add(
                cols_t,
                draw[s][None, :],
                -np.inf,
                inst.solar[s] + inst.bat_usage * inst.v_bat,
            )
            soc = 2 * n_x + si * T + t_idx
            rhs = k * inst.solar[s]
            rhs = rhs.copy()
            rhs[0] += inst.soc0[s]
            first = np.c_[cols_t[:1], soc[:1]]
            add(first, np.r_[k * draw[s], 1][None, :], -np.inf, rhs[:1])
            rest = np.c_[cols_t[1:], soc[1:], soc[:-1]]
            add(rest, np.r_[k * draw[s], 1, -1][None, :], -np.inf, rhs[1:])
        rr, cc, vv = map(np.concatenate, (rows, cols, vals))
        self.A = coo_matrix((vv, (rr, cc)), shape=(count[0], self.n)).tocsr()
        self.row_lo, self.row_hi = np.concatenate(lo), np.concatenate(hi)

    @staticmethod
    def _periods(add, ph_rows, pmin, pmax, fixed_phi):
        """Window rows over the union of start indicators in ph_rows [m, T]."""
        T = ph_rows.shape[1]
        cum = np.concatenate([[0.0], np.cumsum(fixed_phi)])
        pmin, pmax = int(pmin), int(pmax)

        def windows(p):
            t0 = np.arange(T - p + 1)
            idx = ph_rows[:, t0[:, None] + np.arange(p)]  # [m, nW, p]
            return idx.transpose(1, 0, 2).reshape(len(t0), -1), cum[t0 + p] - cum[t0]

        if pmin > T:
            add(ph_rows.reshape(1, -1), 1, -np.inf, 1 - fixed_phi.sum())
        else:
            idx, used = windows(pmin)
            add(idx, 1, -np.inf, 1 - used)
        if pmax <= T:
            idx, used = windows(pmax)
            add(idx, 1, 1 - used, np.inf)

    def solve(self, *, time_limit, incumbent=None, free=None, on_incumbent=None):
        """Maximize utility; with incumbent+free, fix x outside free.

        Returns (schedule [S_sub, R, T] or None, status string).
        """
        solve_deadline = perf_counter() + time_limit
        if time_limit <= 0:
            return None, "limit", None
        lo, hi = self.lower.copy(), self.upper.copy()
        if incumbent is not None and free is not None:
            fix = ~free.ravel()
            vals = incumbent.ravel().astype(float)
            lo[: self.n_x][fix] = vals[fix]
            hi[: self.n_x][fix] = vals[fix]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kwargs = dict(
                c=self.c,
                integrality=self.integrality,
                bounds=Bounds(lo, hi),
                constraints=LinearConstraint(self.A, self.row_lo, self.row_hi),
                options={
                    "time_limit": max(1e-3, time_limit),
                    "threads": 1,
                    "mip_rel_gap": 0.0,
                    "random_seed": 0,
                },
            )
            if on_incumbent is None:
                res = milp(**kwargs)
            else:
                from .observed_highs import milp_with_incumbents

                def observe(raw):
                    values = raw[:self.n_x]
                    if (len(values) != self.n_x or not np.isfinite(values).all()
                            or np.max(np.abs(values - values.round())) > 1e-5):
                        raise RuntimeError("Invalid constellation solver incumbent")
                    on_incumbent(values.round().astype(bool).reshape(self.shape))

                res = milp_with_incumbents(**kwargs, on_incumbent=observe,
                                          deadline=solve_deadline)
        status = {0: "optimal", 1: "limit", 2: "infeasible"}.get(res.status, "error")
        bound = getattr(res, "mip_dual_bound", None)
        if bound is not None and not math.isfinite(bound):
            bound = None
        if res.x is None or not np.isfinite(res.x).all():
            return None, status, bound
        raw = res.x[: self.n_x]
        if np.max(np.abs(raw - raw.round())) > 1e-5:
            return None, status, bound
        return raw.round().astype(bool).reshape(self.shape), status, bound


def best_profile_multi(rewards, rules, *, truncated_final_run=True):
    """Per-job DP where each run chooses its satellite (constellation tasks).

    rewards [S, T]. Same recursion as best_profile with run value
    max_s R_s(u, u+l); constraints of a constellation task are satellite
    independent, so the choice is exact over profiles whose consecutive runs are
    separated by an idle slot. Returns (row [T] bool, satellite [T] int, value).
    """
    rewards = np.asarray(rewards, float)
    S, T = rewards.shape
    allowed_s = np.isfinite(rewards)
    window = np.zeros(T, bool)
    window[max(0, rules.win_min) : min(T, rules.win_max)] = True
    allowed_s &= window[None, :]
    prefix = np.concatenate(
        [np.zeros((S, 1)), np.cumsum(np.where(allowed_s, rewards, 0.0), axis=1)], axis=1
    )
    # nxt[s, t]: first forbidden slot at or after t on satellite s.
    nxt = np.full((S, T + 1), T)
    for s in range(S):
        blocked = np.flatnonzero(~allowed_s[s])
        pos = np.searchsorted(blocked, np.arange(T + 1))
        has = pos < len(blocked)
        nxt[s, has] = blocked[pos[has]]
    periodic = rules.max_gap <= T
    spacing = max(1, rules.min_gap, rules.min_run + 1)
    K = min(rules.max_starts, 1 + (T - 1) // spacing)
    if rules.min_starts > K:
        return None, None, -math.inf
    starts = np.arange(T)
    shortest = (
        np.minimum(rules.min_run, T - starts)
        if truncated_final_run
        else np.full(T, rules.min_run)
    )
    log2 = np.zeros(T + 2, dtype=int)
    log2[2:] = np.floor(np.log2(np.arange(2, T + 2))).astype(int)
    last_ok = (starts >= T - rules.max_gap) if periodic else np.ones(T, bool)
    value = np.full((K + 2, T), -math.inf)
    run_length = np.zeros((K + 2, T), dtype=int)
    run_sat = np.zeros((K + 2, T), dtype=int)
    max_length = int(min(rules.max_run, T))
    for k in range(K, 0, -1):
        table = _range_max(value[k + 1]) if k < K else None
        terminal = np.where(last_ok & (k >= rules.min_starts), 0.0, -math.inf)
        for length in range(1, max_length + 1):
            end = np.minimum(starts + length, T)
            fits = (
                (length <= np.minimum(rules.max_run, T - starts)[None, :])
                & (length <= nxt[:, :T] - starts[None, :])
                & allowed_s
            )
            ok_s = fits & (length >= shortest)[None, :]
            if not ok_s.any():
                continue
            run_s = np.where(ok_s, prefix[:, end] - prefix[:, starts], -math.inf)
            sat = np.argmax(run_s, axis=0)
            run = run_s[sat, starts]
            ok = np.isfinite(run)
            cont = terminal.copy()
            if table is not None:
                lo = np.maximum(end + 1, starts + rules.min_gap)
                hi = (
                    np.minimum(starts + rules.max_gap, T - 1)
                    if periodic
                    else np.full(T, T - 1)
                )
                valid = ok & (lo <= hi)
                if valid.any():
                    a, b = lo[valid], hi[valid]
                    level = log2[b - a + 1]
                    nextv = np.maximum(
                        table[level, a], table[level, b - (1 << level) + 1]
                    )
                    cont[valid] = np.maximum(cont[valid], nextv)
            total = np.where(ok, run + cont, -math.inf)
            better = total > value[k]
            value[k, better] = total[better]
            run_length[k, better] = length
            run_sat[k, better] = sat[better]
    best, first = -math.inf, -1
    if rules.min_starts == 0 and not periodic:
        best = 0.0
    last_first = min(T - 1, rules.max_gap - 1) if periodic else T - 1
    cand = value[1, : last_first + 1]
    if len(cand) and cand.max() > best:
        first = int(np.argmax(cand))
        best = float(cand[first])
    if not math.isfinite(best):
        return None, None, -math.inf
    row = np.zeros(T, bool)
    owner = np.full(T, -1)
    k, s = 1, first
    while s >= 0:
        length = int(run_length[k, s])
        row[s : s + length] = True
        owner[s : s + length] = run_sat[k, s]
        end = s + length
        stop = 0.0 if (last_ok[s] and k >= rules.min_starts) else -math.inf
        following = -1
        if k < K:
            lo = max(end + 1, s + rules.min_gap)
            hi = min(s + rules.max_gap, T - 1) if periodic else T - 1
            if lo <= hi:
                seg = value[k + 1, lo : hi + 1]
                if seg.max() > stop:
                    following = lo + int(np.argmax(seg))
        k, s = k + 1, following
    return row, owner, best


def _windows(inst, soc):
    """Deficit windows per satellite: slots since the battery was last full."""
    out = []
    for s in range(inst.S):
        start = 0
        for t, level in enumerate(soc[s]):
            if level >= 1 - 1e-12:
                start = t + 1
            elif level < inst.soc_min - 1e-9:
                out.append((s, start, t + 1))
    return out


def reprofile_constellation(
    inst, X, items, rng, *, iterations=8, deadline=None, surplus_weight=0.05
):
    """Remove items (('sat', s, r) or ('con', c)) and reinsert by priced DP.

    Returns (schedule, objective, gain). The input schedule is returned when no
    validated improvement is found; X may be infeasible (construction), in which
    case any validated schedule is an improvement.
    """
    u, draw = inst.utility(), inst.draw()
    cap = inst.solar + inst.bat_usage * inst.v_bat
    rho = float(np.median(u / np.maximum(draw, 1e-6)))
    obj0, bad0 = validate_constellation(inst, X)
    best, best_obj = X.copy(), (obj0 if not bad0 else -math.inf)
    prices = np.zeros((inst.S, inst.T))
    infeasible, feasible = None, None
    max_draw = float(draw.max())
    for step in range(iterations):
        if deadline is not None and perf_counter() > deadline:
            break
        Y = X.copy()
        for item in items:
            if item[0] == "sat":
                Y[item[1], item[2]] = False
            else:
                Y[:, inst.Js + item[1]] = False
        order = list(items)
        if step:
            rng.shuffle(order)
        complete = True
        for item in order:
            usage = np.einsum("sr,srt->st", draw, Y.astype(float))
            unsat = battery(inst, usage) < 1 - 1e-9
            if item[0] == "sat":
                _, s, r = item
                rew = (
                    u[s, r]
                    - prices[s] * draw[s, r]
                    - unsat[s] * (surplus_weight * u[s, r] * draw[s, r] / max_draw)
                )
                rew[draw[s, r] > cap[s] - usage[s] + 1e-9] = -math.inf
                row, _ = best_profile(rew, inst.rules(s, r))
                if row is None:
                    complete = False
                    break
                Y[s, r] = row
            else:
                c = item[1]
                r = inst.Js + c
                rew = (
                    u[:, r][:, None]
                    - prices * draw[:, r][:, None]
                    - unsat
                    * (
                        surplus_weight
                        * u[:, r][:, None]
                        * draw[:, r][:, None]
                        / max_draw
                    )
                )
                rew[draw[:, r][:, None] > cap - usage + 1e-9] = -math.inf
                row, owner, _ = best_profile_multi(rew, inst.rules(0, r))
                if row is None:
                    complete = False
                    break
                for s in range(inst.S):
                    Y[s, r] = row & (owner == s)
        if not complete:
            break
        usage = np.einsum("sr,srt->st", draw, Y.astype(float))
        soc = battery(inst, usage)
        if (soc >= inst.soc_min - 1e-6).all():
            feasible = prices.copy()
            obj = float((u[:, :, None] * Y).sum())
            if obj > best_obj + 1e-9:
                o, bad = validate_constellation(inst, Y)
                if bad:
                    raise RuntimeError(f"Constellation profile move invalid: {bad[:3]}")
                best, best_obj = Y, o
            if infeasible is None:
                break
            prices = (infeasible + feasible) / 2
        else:
            infeasible = prices.copy()
            if feasible is not None:
                prices = (infeasible + feasible) / 2
            else:
                for s, a, b in _windows(inst, soc):
                    prices[s, a:b] += rho * 0.5 * (1.6**step)
    return best, best_obj


def solve_constellation(
    inst,
    *,
    time_limit=60.0,
    repair_time=2.0,
    patience=10,
    seed=0,
    mode="portfolio",
    fraction=0.2,
    observe_incumbents=False,
    on_incumbent=None,
):
    """Profile portfolio for constellations, or the exact MILP (mode="exact").

    Portfolio: construct by priced DP insertion of all tasks, then profile moves
    on 1-3 items; after `patience` failures, one per-satellite MILP repair of a
    random fraction of that satellite's satellite-task cells (constellation rows
    and other satellites fixed). Returns dict with schedule, objective, history.
    """
    if mode not in ("portfolio", "exact") or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Require portfolio/exact mode and a positive finite allowance")
    started = perf_counter()
    deadline = started + time_limit - min(0.25, 0.05 * time_limit)
    history = []

    def record(X, value, source):
        if history and value <= history[-1][1] + 1e-9:
            return
        if on_incumbent is not None:
            checked, bad = validate_constellation(inst, X)
            if bad or abs(checked - value) > 1e-6:
                raise RuntimeError("Invalid observed constellation incumbent")
            snapshot = X.copy()
        now = perf_counter() - started
        history.append((now, value))
        if on_incumbent is not None:
            on_incumbent(snapshot, value, now, source)

    if mode == "exact":
        model = ConstellationMilp(inst)
        built = perf_counter() - started
        def observe(X):
            value, bad = validate_constellation(inst, X)
            if bad:
                raise RuntimeError(f"Invalid HiGHS constellation incumbent: {bad[:3]}")
            record(X, value, "highs_callback")

        X, status, bound = model.solve(time_limit=deadline - perf_counter(),
                                       on_incumbent=observe if observe_incumbents else None)
        obj = None
        if X is not None:
            obj, bad = validate_constellation(inst, X)
            if bad:
                X, obj = None, None
        if X is not None:
            record(X, obj, "solver_return")
        return dict(
            schedule=X,
            objective=obj,
            status=status,
            history=history,
            bound=None if bound is None else -bound,
            build_time=built,
            runtime=perf_counter() - started,
        )
    rng = np.random.default_rng(seed)
    items_all = [("sat", s, r) for s in range(inst.S) for r in range(inst.Js)] + [
        ("con", c) for c in range(inst.Jc)
    ]
    X = np.zeros((inst.S, inst.R, inst.T), bool)
    # Construction: constellation tasks first (they choose satellites), then
    # satellite tasks by decreasing utility per watt.
    u, draw = inst.utility(), inst.draw()
    order = [("con", c) for c in np.argsort(-inst.con["u"] / inst.con["q"])] + sorted(
        [("sat", s, r) for s in range(inst.S) for r in range(inst.Js)],
        key=lambda it: -u[it[1], it[2]] / draw[it[1], it[2]],
    )
    X, obj = reprofile_constellation(
        inst, X, order, rng, iterations=12, deadline=deadline
    )
    if not math.isfinite(obj):
        history.append(("construction_failed", perf_counter() - started))
        return dict(
            schedule=None,
            objective=None,
            status="unknown",
            history=[],
            runtime=perf_counter() - started,
        )
    first = perf_counter() - started
    record(X, obj, "profile_construction")
    failures, iterations, repairs = 0, 0, 0
    while perf_counter() < deadline:
        if failures < patience:
            count = 1 + min(2, failures // 3)
            idx = rng.choice(
                len(items_all), size=min(count, len(items_all)), replace=False
            )
            Y, value = reprofile_constellation(
                inst, X, [items_all[i] for i in idx], rng, deadline=deadline
            )
        else:
            s = int(rng.integers(inst.S))
            free = np.zeros((1, inst.R, inst.T), bool)
            legal = np.zeros(inst.Js * inst.T, bool)
            n_free = math.ceil(fraction * inst.Js * inst.T)
            legal[rng.choice(inst.Js * inst.T, n_free, replace=False)] = True
            free[0, : inst.Js] = legal.reshape(inst.Js, inst.T)
            model = ConstellationMilp(inst, sats=[s], fixed=X)
            sub, _, _ = model.solve(
                time_limit=min(repair_time, deadline - perf_counter()),
                incumbent=X[[s]],
                free=free,
            )
            Y, value = X, obj
            if sub is not None:
                cand = X.copy()
                cand[s] = sub[0]
                o, bad = validate_constellation(inst, cand)
                if not bad and o > obj + 1e-9:
                    Y, value = cand, o
            repairs += 1
        if value > obj + 1e-9:
            X, obj, failures = Y, value, 0
            record(X, obj, "portfolio_improvement")
        else:
            failures = patience // 2 if failures >= patience else failures + 1
        iterations += 1
    o, bad = validate_constellation(inst, X)
    if bad:
        raise RuntimeError("Constellation search returned an infeasible schedule")
    return dict(
        schedule=X,
        objective=obj,
        status="feasible",
        history=history,
        first=first,
        iterations=iterations,
        repairs=repairs,
        runtime=perf_counter() - started,
    )
