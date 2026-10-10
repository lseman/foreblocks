"""Immutable, disjoint ONTS splits generated from feasible witness schedules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..environments.onts_env import ONTSInstance, onts_instance_to_json_data
from ..milp import ONTSConfig


def instance_hash(instance: ONTSInstance) -> str:
    data = onts_instance_to_json_data(instance)
    data.pop("name", None)
    # Match the public JSON loader's float32 representation for identity.
    for name in ("priority", "uso_p", "recurso_p"):
        data[name] = torch.as_tensor(data[name], dtype=torch.float32).tolist()
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = [
        *root.glob("*.py"),
        *root.joinpath("research").glob("*.py"),
        root / "environments" / "onts_env.py",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def witness_instance(*, seed: int, regime: str, large: bool = False):
    """Construct feasibility without filtering according to a solver's success.

    The planted schedule has two isolated length-two runs per task. Parameter
    constraints contain that schedule; their relaxation permits competing
    schedules. This is a synthetic distribution with disclosed planted bias.
    """
    if regime not in ("balanced", "energy", "periodic"):
        raise ValueError("Unknown regime")
    rng = np.random.default_rng(seed)
    J, T = (12, 96) if large else ((6, 48) if regime == "balanced" else (9, 72))
    half = T // 2
    order = rng.permutation(J)
    witness = np.zeros((J, T), dtype=np.int64)
    first = []
    for j in range(J):
        offset = int(np.where(order == j)[0][0]) * 3 + int(rng.integers(0, 2))
        first.append(offset)
        witness[j, offset : offset + 2] = 1
        witness[j, offset + half : offset + half + 2] = 1
    phase = np.linspace(0, 2 * np.pi, T, endpoint=False) + rng.uniform(0, 2 * np.pi)
    if regime == "energy":
        solar = 1.0 + 3 * np.sin(phase) ** 2
        solar[T // 3 : 2 * T // 3] = 0
        draw = rng.uniform(4, 10, size=J)
    else:
        solar = 6.0 + 4 * np.sin(phase).clip(min=0)
        draw = rng.uniform(2, 8, size=J)
    if large:
        solar = solar * 0.7
    minimum = rng.integers(1, 3, size=J)
    maximum = minimum + rng.integers(3, 9, size=J)
    counts = np.full(J, 2 if regime == "periodic" else 1)
    max_counts = counts + rng.integers(2, 5, size=J)
    gap_min = rng.integers(4, 10, size=J)
    gap_max = (
        np.full(J, half + rng.integers(0, 4))
        if regime == "periodic"
        else np.full(J, T + 1)
    )
    wmin = np.maximum(0, np.array(first) - rng.integers(0, 5, size=J))
    wmax = np.minimum(T, np.array(first) + half + 2 + rng.integers(2, 12, size=J))
    instance = ONTSInstance(
        priority=torch.tensor(rng.uniform(1, 8, size=J), dtype=torch.float64),
        uso_p=torch.tensor(draw, dtype=torch.float64),
        min_statup=torch.tensor(counts),
        max_statup=torch.tensor(max_counts),
        min_cpu_time=torch.tensor(minimum),
        max_cpu_time=torch.tensor(maximum),
        min_periodo_job=torch.tensor(gap_min),
        max_periodo_job=torch.tensor(gap_max),
        win_min=torch.tensor(wmin),
        win_max=torch.tensor(wmax),
        recurso_p=torch.tensor(solar, dtype=torch.float64),
    )
    report = ONTSConfig().validate(instance, torch.from_numpy(witness))
    if not report.feasible:
        raise RuntimeError(f"Witness generation failed: {report.violations}")
    return instance, torch.from_numpy(witness)


def build_suite(
    output: Path,
    *,
    train_per_regime: int = 4,
    validation_per_regime: int = 2,
    test_per_regime: int = 4,
    ood_per_regime: int = 2,
    seed: int = 20261005,
    bundled_dir: Path | None = None,
):
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for split, count, large in [
        ("train", train_per_regime, False),
        ("validation", validation_per_regime, False),
        ("test", test_per_regime, False),
        ("ood", ood_per_regime, True),
    ]:
        for regime_index, regime in enumerate(("balanced", "energy", "periodic")):
            offset = {"train": 0, "validation": 10000, "test": 20000, "ood": 30000}[
                split
            ]
            for i in range(count):
                draw_seed = seed + offset + 1000 * regime_index + i
                instance, witness = witness_instance(
                    seed=draw_seed, regime=regime, large=large
                )
                data = onts_instance_to_json_data(
                    instance, name=f"{split}_{regime}_{draw_seed}"
                )
                data["generator"] = {
                    "name": "onts-planted-v1",
                    "regime": regime,
                    "seed": draw_seed,
                    "split": split,
                }
                path = output / split / f"{regime}_{draw_seed}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, indent=2) + "\n")
                entries.append(
                    {
                        "id": f"{split}:{regime}:{draw_seed}",
                        "path": str(path.resolve()),
                        "split": split,
                        "regime": regime,
                        "hash": instance_hash(instance),
                        "jobs": instance.J,
                        "horizon": instance.T,
                        "witness": witness.tolist(),
                        "witness_objective": ONTSConfig()
                        .validate(instance, witness)
                        .objective,
                    }
                )
    if bundled_dir is not None:
        from ..environments.onts_env import load_onts_instance

        for path in sorted(bundled_dir.glob("*.json")):
            instance = load_onts_instance(path)
            entries.append(
                {
                    "id": f"bundled:{path.stem}",
                    "path": str(path.resolve()),
                    "split": "bundled",
                    "regime": "bundled",
                    "hash": instance_hash(instance),
                    "jobs": instance.J,
                    "horizon": instance.T,
                    "witness": None,
                }
            )
    hashes = [entry["hash"] for entry in entries]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate instance content across suite entries")
    manifest = {
        "format": "onts-study-manifest-v1",
        "seed": seed,
        "config": as_config(),
        "generator_bias": "Feasible planted witnesses; no solver-based acceptance filter.",
        "entries": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def as_config():
    from dataclasses import asdict

    return asdict(ONTSConfig())
