"""Regenerate tables, paired tests, figures and data-bounded claims from records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .protocol import read_jsonl
from .statistics import bootstrap_mean, holm_adjust, normalized_auc, paired_comparison


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(output: Path):
    plan = json.loads((output / "plan.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    manifest = json.loads((output / "data" / "manifest.json").read_text())
    rows = read_jsonl(output / "runs.jsonl")
    references = {
        row["instance_id"]: row for row in read_jsonl(output / "references.jsonl")
    }
    # Unknown references have no normalization denominator. Keep zero until a
    # validated benchmark incumbent supplies a positive reference.
    best = {key: ref["reference_objective"] or 0.0 for key, ref in references.items()}
    for row in rows:
        if row["feasible"]:
            previous = best.get(row["instance_id"])
            best[row["instance_id"]] = max(previous or 0, row["objective"])
    summaries = []
    keys = sorted({(r["split"], r["track"], r["budget"], r["method"]) for r in rows})
    for split, track, budget, method in keys:
        group = [
            r
            for r in rows
            if (r["split"], r["track"], r["budget"], r["method"])
            == (split, track, budget, method)
        ]
        feasible = [r for r in group if r["feasible"]]
        available = [
            r["objective"] / best[r["instance_id"]]
            for r in feasible
            if best.get(r["instance_id"], 0) > 0
        ]
        aucs = [
            normalized_auc(
                r["history"], budget=budget, reference=best[r["instance_id"]]
            )
            for r in group
            if not r.get("error") and best.get(r["instance_id"], 0) > 0
        ]
        by_instance = {}
        for row in group:
            reference = best.get(row["instance_id"])
            if reference and reference > 0 and not row.get("error"):
                value = normalized_auc(
                    row["history"], budget=budget, reference=reference
                )
                by_instance.setdefault(row["instance_id"], []).append(value)
        auc_ci = (
            bootstrap_mean([np.mean(values) for values in by_instance.values()])
            if by_instance
            else None
        )
        summaries.append(
            {
                "split": split,
                "track": track,
                "budget": budget,
                "method": method,
                "runs": len(group),
                "feasible": len(feasible),
                "feasible_rate": len(feasible) / len(group),
                "errors": sum(bool(r.get("error")) for r in group),
                "mean_quality_conditional_feasible": float(np.mean(available))
                if available
                else None,
                "mean_auc_to_best_known": float(np.mean(aucs)) if aucs else None,
                "auc_ci_low": auc_ci["ci95"][0] if auc_ci else None,
                "auc_ci_high": auc_ci["ci95"][1] if auc_ci else None,
                "auc_instances": len(by_instance),
                "mean_runtime": float(
                    np.mean([r["runtime"] for r in group if "runtime" in r])
                ),
                "mean_overrun": float(np.mean([r.get("overrun", 0) for r in group])),
                "repair_iterations": sum(r.get("iterations", 0) for r in group),
                "strictly_improved_runs": sum(
                    r["feasible"]
                    and references[r["instance_id"]]["initial_objective"] is not None
                    and r["objective"]
                    > references[r["instance_id"]]["initial_objective"] + 1e-6
                    for r in group
                )
                if track == "common"
                else None,
            }
        )
    comparisons = []
    for split, track, budget in sorted(
        {(r["split"], r["track"], r["budget"]) for r in rows}
    ):
        methods = {
            r["method"]
            for r in rows
            if (r["split"], r["track"], r["budget"]) == (split, track, budget)
        }
        for baseline in sorted(methods - {"neural"}):
            comparison = paired_comparison(
                rows,
                method="neural",
                baseline=baseline,
                track=track,
                budget=budget,
                split=split,
            )
            comparisons.append(comparison)
    adjusted = holm_adjust([c["sign_test_p"] for c in comparisons])
    for comparison, p in zip(comparisons, adjusted):
        comparison["holm_adjusted_p"] = p
    training = [
        json.loads(path.read_text())
        for path in sorted((output / "models").glob("*.json"))
    ]
    completion = (
        json.loads((output / "completion.json").read_text())
        if (output / "completion.json").exists()
        else None
    )
    result = {
        "format": "onts-study-report-v1",
        "audit": {key: value for key, value in audit.items() if key != "results"},
        "complete": bool(completion and completion["expected_runs"] == len(rows)),
        "completion": completion,
        "runs": len(rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "summaries": summaries,
        "comparisons": comparisons,
        "training_cost": sum(t["label_and_training_runtime"] for t in training),
        "reference_setup_cost": sum(r["setup_runtime"] for r in references.values()),
        "best_known": best,
        "training_seeds": [t["seed"] for t in training],
        "instance_counts": {
            split: sum(e["split"] == split for e in manifest["entries"])
            for split in ("train", "validation", "test", "ood", "bundled")
        },
    }
    (output / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    write_csv(output / "summary.csv", summaries)
    flat = []
    for c in comparisons:
        effect = c["relative_objective_change_percent"]
        flat.append(
            {
                key: c[key]
                for key in (
                    "split",
                    "track",
                    "budget",
                    "method",
                    "baseline",
                    "wins",
                    "ties",
                    "losses",
                    "sign_test_p",
                    "holm_adjusted_p",
                )
            }
            | {
                "mean_percent": effect["mean"] if effect else None,
                "ci_low": effect["ci95"][0] if effect else None,
                "ci_high": effect["ci95"][1] if effect else None,
                "instances": effect["instances"] if effect else 0,
            }
        )
    write_csv(output / "comparisons.csv", flat)
    report = [
        "# ONTS empirical evaluation",
        "",
        f"Completed runs: {len(rows)}. Study complete: {result['complete']}. Runtime errors: {result['errors']}.",
        "",
        "## Correctness",
        "",
        f"The independent oracle, validator and sparse MILP agree on all {audit['schedules']:,} schedules across {audit['cases']} tiny instances. "
        f"Disagreements: {audit['disagreement_count']}; optimum mismatches: {audit['optimum_mismatches']}.",
        "",
        "This establishes exhaustive agreement on these finite cases. It is not a proof for arbitrary sizes or a proof of performance superiority.",
        "",
        "## Protocol",
        "",
        f"Instance counts: {result['instance_counts']}. Training seeds: {result['training_seeds']}.",
        "Splits are content-disjoint. Synthetic instances have disclosed planted feasible schedules, without solver-based rejection filtering.",
        "Cold runs include construction, initialization and search. Common-incumbent runs retain the same validated initial schedule; preprocessing cost is separate.",
        "The SciPy solver receives no native MIP start. A supplied incumbent is retained as a fallback, rather than injected into branch-and-bound.",
        "Only schedules observable by the requested deadline are credited. Later outputs remain in raw records. References are best-known feasible objectives, not presumed optima.",
        "Seeds are averaged within instance. Paired bootstrap intervals and sign tests use instances as sampling units. Objective comparisons condition on both methods being feasible; feasibility failures are reported separately.",
        "Holm adjustment covers all reported comparisons. Confidence intervals are unadjusted marginal 95% intervals.",
        "Ablations mask features or remove edges at inference without retraining; they measure sensitivity to those interventions.",
        "Solver version, CPU/thread counts, source hashes, plan, instance manifest, checkpoints and raw records accompany this report.",
        "",
        "## Paired objective comparisons",
        "",
        "| Split | Track | Budget (s) | Baseline | Instances | Neural change (%) | 95% CI | W/T/L | Adjusted p |",
        "|---|---|---:|---|---:|---:|---|---|---:|",
    ]
    for c in comparisons:
        effect = c["relative_objective_change_percent"]
        estimate = f"{effect['mean']:.2f}" if effect else "NA"
        interval = (
            f"[{effect['ci95'][0]:.2f}, {effect['ci95'][1]:.2f}]" if effect else "NA"
        )
        report.append(
            f"| {c['split']} | {c['track']} | {c['budget']} | {c['baseline']} | {effect['instances'] if effect else 0} | {estimate} | {interval} | {c['wins']}/{c['ties']}/{c['losses']} | {c['holm_adjusted_p']:.4f} |"
        )
    report += [
        "",
        "## Feasibility and anytime quality",
        "",
        "| Split | Track | Budget (s) | Method | Feasible / runs | Mean AUC / best known | Mean overrun (s) |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for summary in summaries:
        auc = (
            f"{summary['mean_auc_to_best_known']:.3f}"
            if summary["mean_auc_to_best_known"] is not None
            else "NA"
        )
        report.append(
            f"| {summary['split']} | {summary['track']} | {summary['budget']} | {summary['method']} | {summary['feasible']}/{summary['runs']} | {auc} | {summary['mean_overrun']:.3f} |"
        )
    primary = [
        c
        for c in comparisons
        if c["split"] == "test"
        and c["track"] == "cold"
        and c["budget"] == max(plan["budgets"])
        and c["baseline"] == "exact"
    ]
    supported = bool(
        primary
        and primary[0]["relative_objective_change_percent"]
        and primary[0]["relative_objective_change_percent"]["ci95"][0] > 0
        and primary[0]["holm_adjusted_p"] < 0.05
    )
    report += [
        "",
        "## Claim status",
        "",
        (
            "The predefined objective comparison supports a positive neural advantage over exact solving on the tested synthetic distribution."
            if supported
            else "The predefined comparison does not establish a neural advantage over exact solving. Do not claim superiority from this study."
        ),
        "Feasibility evidence, conditional objective effects, and inference interventions should not be combined into an unrestricted claim.",
        f"Recorded label/training cost: {result['training_cost']:.2f} s. Reference and common-initialization setup: {result['reference_setup_cost']:.2f} s.",
        "",
        "## Limits and next experiments",
        "",
        "Finite synthetic samples and short budgets do not establish performance at spacecraft deployment scale.",
        "Bundled examples may be related; instance-level inference does not establish independence across unknown generation families.",
        "Three paired training/search seeds do not separately identify training variance from search variance.",
        "The exact baseline is SciPy/HiGHS, not a tuned Gurobi/SCIP baseline. No comparison with a published ONTS neural implementation has been run.",
        "AUC uses recorded internal incumbent timestamps. LNS exposes each repair to this logger; SciPy exposes the unrestricted solver solution only when the call returns. This instrumentation asymmetry prevents interpreting AUC differences as a fair comparison of native solver incumbent trajectories, or as caller streaming behavior.",
        "Test-time ablations are not retrained architectural comparisons. Longer budgets, independent generation families and a frozen confirmatory replication remain necessary.",
        "",
        "## Evaluation references",
        "",
        "[Sonnerat et al.](https://arxiv.org/abs/2107.10201) motivates learned neighborhood selection with solver repair.",
        "[Huang et al.](https://proceedings.mlr.press/v202/huang23g.html) evaluates anytime ILP methods with primal-gap and primal-integral metrics.",
        "The AUC here integrates validated objective divided by a common best-known feasible objective. It is not an optimality-based primal integral.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    make_figures(output, summaries, flat)
    return result


def make_figures(output, summaries, comparisons):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    palette = {
        "exact": "#0072B2",
        "random": "#999999",
        "heuristic": "#E69F00",
        "neural": "#009E73",
        "untrained": "#CC79A7",
        "no_future": "#D55E00",
        "no_edges": "#56B4E9",
    }
    chosen = [s for s in summaries if s["split"] == "test" and s["track"] == "cold"]
    if chosen:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ("exact", "random", "heuristic", "neural"):
            group = sorted(
                [s for s in chosen if s["method"] == method], key=lambda s: s["budget"]
            )
            ax.plot(
                [s["budget"] for s in group],
                [s["mean_auc_to_best_known"] or 0 for s in group],
                marker="o",
                label=method,
                color=palette[method],
            )
            ax.fill_between(
                [s["budget"] for s in group],
                [s["auc_ci_low"] or 0 for s in group],
                [s["auc_ci_high"] or 0 for s in group],
                color=palette[method],
                alpha=0.12,
            )
        ax.set_xlabel("Wall-clock budget (s)")
        ax.set_ylabel("Mean observed objective AUC / best known")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / "anytime.png", dpi=180)
        fig.savefig(output / "anytime.pdf")
        plt.close(fig)
    effects = [
        c
        for c in comparisons
        if c["track"] == "common"
        and c["instances"] > 0
        and c["baseline"] in ("exact", "random", "heuristic")
    ]
    if effects:
        splits = sorted({row["split"] for row in effects})
        fig, axes = plt.subplots(
            len(splits), 2, figsize=(10, 2.6 * len(splits)), squeeze=False
        )
        for split_index, split in enumerate(splits):
            for column in range(2):
                ax = axes[split_index, column]
                group = [
                    row
                    for row in effects
                    if row["split"] == split
                    and (row["baseline"] == "exact") == (column == 0)
                ]
                for index, row in enumerate(group):
                    ax.errorbar(
                        row["mean_percent"],
                        index,
                        xerr=[
                            [max(0, row["mean_percent"] - row["ci_low"])],
                            [max(0, row["ci_high"] - row["mean_percent"])],
                        ],
                        fmt="o",
                        color="#009E73",
                        capsize=3,
                    )
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(
                    range(len(group)),
                    [f"vs {r['baseline']} ({r['budget']:g} s)" for r in group],
                )
                ax.set_ylim(-0.7, max(len(group) - 0.3, 0.7))
                ax.set_title(f"{split}: {'MILP' if column == 0 else 'LNS controls'}")
                ax.set_xlabel("Neural objective change (%)")
        fig.suptitle("Common incumbent; marginal 95% bootstrap intervals")
        fig.tight_layout()
        fig.savefig(output / "paired-effects.png", dpi=180)
        fig.savefig(output / "paired-effects.pdf")
        plt.close(fig)
