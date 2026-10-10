"""Fixed-size, paired confirmation of the frozen semantic ranker.

No architecture tuning, optional stopping, or seed-level pseudoreplication.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest, norm, t

from ..environments.onts_env import load_onts_instance, onts_instance_to_json_data
from ..hybrid import solve_hybrid
from ..learning import train_neighborhood_policy
from ..milp import ONTSConfig
from ..neighborhood import NeighborhoodPolicy
from ..ranked_search import solve_ranked
from ..ranking import NeighborhoodRanker
from ..ranking_learning import train_neighborhood_ranker
from .datasets import build_suite, instance_hash, source_hashes
from .oracle import oracle_check
from .protocol import append_record, objectives_at_budget, read_jsonl
from .statistics import holm_adjust

METHODS = ("ranked", "imitation", "heuristic", "uniform")
SEEDS = (101, 211, 307)
BASELINES = METHODS[1:]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def infer(values, *, family=3):
    """Instance-level mean inference; simultaneous Bonferroni intervals."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(
            "At least two finite independent instance observations required"
        )
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(len(x))
    if se:
        p = float(2 * t.sf(abs(mean / se), len(x) - 1))
    else:
        p = 1.0 if mean == 0 else 0.0
    radius = float(t.ppf(1 - 0.05 / (2 * family), len(x) - 1)) * se
    wins, losses = int((x > 1e-8).sum()), int((x < -1e-8).sum())
    return dict(
        instances=len(x),
        mean_percent=mean,
        sd_percent=sd,
        simultaneous_ci95=[mean - radius, mean + radius],
        two_sided_p=p,
        wins=wins,
        losses=losses,
        ties=len(x) - wins - losses,
        sensitivity_sign_p=float(binomtest(wins, wins + losses).pvalue)
        if wins + losses
        else 1.0,
    )


def power_requirements(sd, effects=(1.0, 2.0, 3.0)):
    """Normal-approximation planning only, not a achieved-power claim."""
    z = norm.ppf(1 - 0.05 / 6) + norm.ppf(0.8)
    return {
        str(effect): max(2, math.ceil((z * sd / effect) ** 2)) for effect in effects
    }


def analyze(rows, entries, seeds=SEEDS):
    expected = {(e["id"], s, m) for e in entries for s in seeds for m in METHODS}
    indexed = {(r["instance_id"], r["search_seed"], r["method"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Complete unique paired records required before inference")
    comparisons = []
    for baseline in BASELINES:
        deltas, detail = [], []
        for e in entries:
            # Average paired seeds first: instances, not runs, are independent units.
            a = np.mean([indexed[e["id"], s, "ranked"]["score"] for s in seeds])
            b = np.mean([indexed[e["id"], s, baseline]["score"] for s in seeds])
            delta = float(100 * (a - b) / max(abs(b), 1e-9))
            deltas.append(delta)
            detail.append(
                dict(instance_id=e["id"], split=e["split"], delta_percent=delta)
            )
        result = dict(baseline=baseline, **infer(deltas), instance_deltas=detail)
        result["split_descriptive"] = {
            split: dict(
                mean_percent=float(
                    np.mean([d["delta_percent"] for d in detail if d["split"] == split])
                ),
                instances=sum(d["split"] == split for d in detail),
            )
            for split in ("test", "ood")
        }
        result["planning_n_for_80_percent_power"] = power_requirements(
            result["sd_percent"]
        )
        comparisons.append(result)
    for c, p in zip(comparisons, holm_adjust([c["two_sided_p"] for c in comparisons])):
        c["holm_p"] = p
        c["statistical_superiority"] = p < 0.05 and c["simultaneous_ci95"][0] > 0
        c["practical_superiority_1_percent"] = (
            c["statistical_superiority"] and c["simultaneous_ci95"][0] > 1
        )
    all_valid = all(
        r.get("validation_passed") and r.get("feasible") and not r.get("error")
        for r in rows
    )
    return dict(
        completed_runs=len(rows),
        comparisons=comparisons,
        all_outputs_valid_by_deadline=all_valid,
        claim_supported=all_valid
        and all(c["practical_superiority_1_percent"] for c in comparisons),
        unit="instance means over three fixed paired training/search seeds",
        scope="Equal IID/OOD mixture of planted synthetic regimes, common incumbents, 3-second CPU budget; conditional on these training datasets and seeds",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = ONTSConfig()
    if (out / "plan.json").exists():
        plan = json.loads((out / "plan.json").read_text())
        if source_hashes() != plan["source_hashes"]:
            raise RuntimeError("Frozen source changed; do not resume this study")
        manifest = json.loads((out / "data/manifest.json").read_text())
    else:
        manifest = build_suite(
            out / "data",
            seed=7261943,
            train_per_regime=8,
            validation_per_regime=3,
            test_per_regime=10,
            ood_per_regime=10,
        )
        old_hashes = set()
        for path in (out.parent).glob("*/data/manifest.json"):
            if path != out / "data/manifest.json":
                old_hashes.update(
                    e["hash"] for e in json.loads(path.read_text())["entries"]
                )
        if old_hashes & {e["hash"] for e in manifest["entries"]}:
            raise RuntimeError("New study overlaps an existing study")
        plan = dict(
            format="onts-confirmatory-v1",
            source_hashes=source_hashes(),
            config=asdict(config),
            seeds=list(SEEDS),
            methods=list(METHODS),
            budget=3.0,
            repair_time=0.3,
            fraction=0.2,
            candidate_count=8,
            expected_runs=720,
            training=dict(
                hidden=32,
                layers=2,
                rounds=4,
                epochs=12,
                repair_time=0.15,
                fraction=0.35,
            ),
            imitation_rounds=32,
            hypothesis="Ranker has >1% mean relative objective improvement versus EACH of imitation, heuristic, uniform",
            primary_population="Equal mixture: 30 IID and 30 OOD, equal balanced/energy/periodic regimes",
            tests="Two-sided paired t on instance seed-means; Holm family of 3; Bonferroni simultaneous 95% CIs",
            decision="All three simultaneous lower limits >1%, all Holm p<.05, all outputs independently valid by deadline",
            stopping="Fixed 720 runs; no significance peeking or optional stopping; errors retained with score zero; incomplete study has no inferential claim",
            limits="Planted witnesses, three seeds, fixed synthetic distribution; no universal SOTA claim",
            matched_training="Same training instances, hidden size, epochs and 768 repair calls per learner/seed; wall time saved, not matched",
        )
        save(out / "plan.json", plan)
        with tarfile.open(out / "frozen-source.tar.gz", "w:gz") as archive:
            root = Path(__file__).resolve().parents[1]
            for relative in plan["source_hashes"]:
                archive.add(
                    root / relative, arcname=str(Path("projects/scheduling") / relative)
                )
    for e in manifest["entries"]:
        if instance_hash(load_onts_instance(e["path"])) != e["hash"]:
            raise RuntimeError("Frozen data changed")
    train = [e for e in manifest["entries"] if e["split"] == "train"]
    heldout = [e for e in manifest["entries"] if e["split"] in ("test", "ood")]
    instances = [load_onts_instance(e["path"]) for e in train]
    initial = [torch.tensor(e["witness"]) for e in train]
    policies = {}
    for seed in SEEDS:
        rp, ip = out / f"ranker-{seed}.pt", out / f"imitation-{seed}.pt"
        for name, checkpoint in [("ranker", rp), ("imitation", ip)]:
            if checkpoint.exists():
                continue
            opts = dict(
                plan["training"], seed=seed, config=config, initial_schedules=initial
            )
            if name == "ranker":
                model, metadata = train_neighborhood_ranker(instances, **opts)
            else:
                opts["rounds"] = plan["imitation_rounds"]
                model, metadata = train_neighborhood_policy(instances, **opts)
            metadata["config"] = asdict(config)
            metadata["training_hashes"] = [e["hash"] for e in train]
            metadata["teacher_calls"] = 768
            save(out / f"{name}-training-{seed}.json", metadata)
            if not metadata["trained"]:
                raise RuntimeError("No informative training outcomes")
            model.save(checkpoint, metadata=metadata)
            runtime = metadata.get(
                "runtime", metadata.get("label_and_training_runtime")
            )
            print(f"trained {name} seed {seed}: {runtime:.1f}s", flush=True)
        policies[seed] = (NeighborhoodRanker.load(rp), NeighborhoodPolicy.load(ip))
    done = {r["run_id"] for r in read_jsonl(out / "runs.jsonl")}
    tasks = [(e, s, m) for e in heldout for s in SEEDS for m in METHODS]
    rng = np.random.default_rng(8231)
    for index in rng.permutation(len(tasks)):
        e, seed, method = tasks[int(index)]
        run_id = f"{e['id']}|{seed}|{method}"
        if run_id in done:
            continue
        row = dict(
            run_id=run_id,
            instance_id=e["id"],
            instance_hash=e["hash"],
            split=e["split"],
            method=method,
            search_seed=seed,
            budget=plan["budget"],
        )
        try:
            instance = load_onts_instance(e["path"])
            common = dict(
                initial_schedule=torch.tensor(e["witness"]),
                time_limit=plan["budget"],
                repair_time=plan["repair_time"],
                seed=seed,
            )
            ranker, imitation = policies[seed]
            if method in ("ranked", "uniform"):
                result = solve_ranked(
                    instance,
                    mode=method,
                    policy=ranker,
                    fraction=plan["fraction"],
                    candidate_count=plan["candidate_count"],
                    **common,
                )
            else:
                result = solve_hybrid(
                    instance,
                    selector="neural" if method == "imitation" else method,
                    policy=imitation,
                    initial_exact=False,
                    neighborhood_fraction=0.2 if method == "imitation" else 0.5,
                    **common,
                )
            payload = result.to_dict()
            valid, objective, _ = oracle_check(
                onts_instance_to_json_data(instance), payload["schedule"], config
            )
            if not valid or not np.isclose(objective, payload["objective"], atol=1e-4):
                raise RuntimeError("Independent output validation failed")
            credited = objectives_at_budget(payload, plan["budget"])
            row.update(
                payload,
                returned_objective=payload["objective"],
                objective=credited,
                feasible=credited is not None,
                score=credited if credited is not None else 0.0,
                validation_passed=True,
                overrun=max(0, result.runtime - plan["budget"]),
            )
        except Exception as exc:
            row.update(
                error=f"{type(exc).__name__}: {exc}",
                score=0.0,
                feasible=False,
                validation_passed=False,
            )
        append_record(out / "runs.jsonl", row)
        done.add(run_id)
        print(f"run {len(done)}/{len(tasks)} {run_id}", flush=True)
    analysis = analyze(read_jsonl(out / "runs.jsonl"), heldout)
    save(out / "analysis.json", analysis)
    report = [
        "# Fixed-size confirmatory study",
        "",
        f"Completed {analysis['completed_runs']}/720 runs.",
        f"Predefined practical-superiority claim supported: **{analysis['claim_supported']}**.",
        "",
        "| Comparator | Mean change (%) | Simultaneous 95% CI | Holm p | >1% supported |",
        "|---|---:|---|---:|---|",
    ]
    for c in analysis["comparisons"]:
        report.append(
            f"| {c['baseline']} | {c['mean_percent']:.3f} | [{c['simultaneous_ci95'][0]:.3f}, {c['simultaneous_ci95'][1]:.3f}] | {c['holm_p']:.5f} | {c['practical_superiority_1_percent']} |"
        )
    report += [
        "",
        analysis["unit"] + ".",
        analysis["scope"] + ".",
        "Three comparisons form one family; split results and sign tests are descriptive sensitivity checks.",
        "Training repair-call counts are matched; training wall times are not. No tuning or significance stopping.",
        "A negative result is evidence against superiority for this design and budget, not a reason to keep sampling until significant.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
