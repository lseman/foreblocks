"""Loop-based replay independent of constellation DP/MILP and battery helpers."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np

from ..constellation import ConstellationInstance


def instance_data(instance):
    data = asdict(instance)
    for name in ("solar", "soc0"):
        data[name] = data[name].tolist()
    for name in ("sat", "con"):
        data[name] = {key: value.tolist() for key, value in data[name].items()}
    data["redraws"] = getattr(instance, "redraws", 0)
    return data


def data_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def from_data(data):
    values = dict(data)
    redraws = values.pop("redraws", 0)
    for name in ("solar", "soc0"):
        values[name] = np.asarray(values[name])
    for name in ("sat", "con"):
        values[name] = {key: np.asarray(value) for key, value in values[name].items()}
    instance = ConstellationInstance(**values)
    instance.redraws = redraws
    return instance


def replay(data, schedule, *, tol=1e-6):
    """Check all task, cross-satellite, bus and curtailing-battery constraints."""
    solar = np.asarray(data["solar"], dtype=float)
    S, T = solar.shape
    Js, Jc = len(data["sat"]["u"][0]), len(data["con"]["u"])
    raw = np.asarray(schedule)
    if raw.shape != (S, Js + Jc, T) or not np.isin(raw, [0, 1]).all():
        return None, ["schedule shape or nonbinary values"]
    X = raw.astype(bool)
    bad = []
    starts = np.zeros(X.shape, dtype=int)
    usage = np.zeros((S, T))
    objective = 0.

    def period(phi, lower, upper, pmin, pmax, label):
        total = sum(phi)
        if not lower <= total <= upper:
            bad.append(f"count {label}")
        if pmin > T:
            if total > 1:
                bad.append(f"minimum period {label}")
        elif any(sum(phi[t:t + pmin]) > 1 for t in range(T - pmin + 1)):
            bad.append(f"minimum period {label}")
        if pmax <= T and any(sum(phi[t:t + pmax]) < 1 for t in range(T - pmax + 1)):
            bad.append(f"maximum period {label}")

    for s in range(S):
        for r in range(Js + Jc):
            source, index = (data["sat"], (s, r)) if r < Js else (data["con"], (r - Js,))
            def value(key):
                item = source[key]
                for i in index:
                    item = item[i]
                return item
            row = X[s, r]
            previous, run_start = False, None
            for t in range(T):
                if row[t] and not previous:
                    starts[s, r, t] = 1
                    run_start = t
                if row[t]:
                    if not int(value("w_min")) <= t < int(value("w_max")) - 1:
                        bad.append(f"window {s} {r} {t}")
                    usage[s, t] += value("q")
                    objective += value("u")
                if previous and not row[t]:
                    length = t - run_start
                    if not min(int(value("f_min")), T - run_start) <= length <= int(value("f_max")):
                        bad.append(f"run {s} {r} {run_start}")
                previous = bool(row[t])
            if previous:
                length = T - run_start
                if not min(int(value("f_min")), length) <= length <= int(value("f_max")):
                    bad.append(f"terminal run {s} {r}")
            if r < Js:
                period(starts[s, r].tolist(), int(value("y_min")), int(value("y_max")),
                       int(value("p_min")), int(value("p_max")), f"{s} {r}")
    for c in range(Jc):
        r = Js + c
        if any(sum(int(X[s, r, t]) for s in range(S)) > 1 for t in range(T)):
            bad.append(f"exclusive {c}")
        source = data["con"]
        phi = [sum(int(starts[s, r, t]) for s in range(S)) for t in range(T)]
        period(phi, int(source["g_min"][c]), int(source["g_max"][c]),
               int(source["p_min"][c]), int(source["p_max"][c]), f"constellation {c}")
    factor = data["ef"] / (data["q"] * data["v_bat"] * 60)
    for s in range(S):
        charge = data["soc0"][s]
        for t in range(T):
            if usage[s, t] > solar[s, t] + data["bat_usage"] * data["v_bat"] + tol:
                bad.append(f"power {s} {t}")
            charge = min(1., charge + factor * (solar[s, t] - usage[s, t]))
            if charge < data["soc_min"] - tol:
                bad.append(f"battery {s} {t}")
    return float(objective), bad
