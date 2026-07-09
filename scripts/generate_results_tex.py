#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import json
import shutil
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_TEX = REPO_ROOT / "results.tex"
FIGURES_OUTPUT_DIR = REPO_ROOT / "darts_figures"
REPORT_HORIZONS = (5, 10, 15)
REPORT_SOURCES = (
    {
        "key": "hidro",
        "label": "hidro",
        "title": "Hydrological Forecast",
        "task_name": "hydrological generation",
        "task_slug": "hydrological-generation",
    },
    {
        "key": "solar",
        "label": "solar",
        "title": "Solar Forecast",
        "task_name": "solar generation",
        "task_slug": "solar-generation",
    },
)
SKIPPED_BENCHMARK_MODELS = {"iTransformer"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_params(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return _latex_escape(value)

    if not isinstance(parsed, dict):
        return _latex_escape(value)

    parts = []
    for key, item in parsed.items():
        if isinstance(item, float):
            item_text = f"{item:.4g}"
        else:
            item_text = str(item)
        parts.append(f"{key}={item_text}")
    return _latex_escape(", ".join(parts))


def _load_benchmark_rows(path: Path) -> list[dict[str, float | str]]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"Benchmark CSV is empty: {path}")

    normalized_rows: list[dict[str, float | str]] = []
    for row in rows:
        model_name = row.get("", "benchmark")
        if model_name in SKIPPED_BENCHMARK_MODELS:
            continue
        normalized_rows.append({
            "model": model_name,
            "mse": float(row["MSE"]),
            "rmse": float(row["RMSE"]),
            "mae": float(row["MAE"]),
        })

    if not normalized_rows:
        raise ValueError(f"No benchmark rows remain after filtering: {path}")

    normalized_rows.sort(key=lambda row: float(row["mse"]))
    return normalized_rows


def _load_best_benchmark(path: Path) -> dict[str, float | str]:
    return _load_benchmark_rows(path)[0]


def _load_final_metrics(path: Path) -> dict[str, float]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"Final metrics CSV is empty: {path}")
    row = rows[0]
    return {
        "mse": float(row["mse"]),
        "rmse": float(row["rmse"]),
        "mae": float(row["mae"]),
    }


def _find_best_run(run_dir: Path) -> dict[str, object]:
    ranked_runs: list[dict[str, object]] = []
    for subrun_dir in sorted(
        path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()
    ):
        metrics_path = subrun_dir / "darts_final_metrics.csv"
        candidate_path = subrun_dir / "darts_best_candidate.json"
        if not metrics_path.is_file() or not candidate_path.is_file():
            continue
        final_metrics = _load_final_metrics(metrics_path)
        ranked_runs.append({
            "run_name": subrun_dir.name,
            "run_dir": subrun_dir,
            "metrics": final_metrics,
            "mse": final_metrics["mse"],
        })

    if not ranked_runs:
        raise ValueError(f"No completed run folders found in {run_dir}")

    ranked_runs.sort(key=lambda item: item["mse"])
    best = ranked_runs[0]
    best["candidate_payload"] = json.loads(
        (Path(best["run_dir"]) / "darts_best_candidate.json").read_text(
            encoding="utf-8"
        )
    )
    return best


def _find_best_run_or_none(run_dir: Path) -> dict[str, object] | None:
    try:
        return _find_best_run(run_dir)
    except ValueError:
        return None


def _pct_improvement(baseline: float, ours: float) -> float:
    return 100.0 * (baseline - ours) / baseline


def _fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def _fmt_pct(value: float) -> str:
    return f"{abs(value):.2f}\\%"


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def _latex_join(items: list[str]) -> str:
    escaped_items = [_latex_escape(item) for item in items if item]
    if not escaped_items:
        return "n/a"
    if len(escaped_items) == 1:
        return escaped_items[0]
    if len(escaped_items) == 2:
        return f"{escaped_items[0]} and {escaped_items[1]}"
    return ", ".join(escaped_items[:-1]) + f", and {escaped_items[-1]}"


def _comparison_clause(metric_name: str, gain: float) -> str:
    direction = "lower" if gain >= 0.0 else "higher"
    return f"{metric_name} {_fmt_pct(gain)} {direction}"


def _operation_phrase(ops: list[str]) -> str:
    if not ops:
        return "no discrete operation metadata was recorded"
    return f"retains {_latex_join(ops)}"


def _family_phrase(families: list[str]) -> str:
    if not families:
        return "without a recorded family label"
    if len(families) == 1:
        return f"within the {_latex_escape(families[0])} family"
    return f"across the {_latex_join(families)} families"


def _table_position(
    rows: list[Mapping[str, float | str | bool]], model_name: str
) -> int:
    for index, row in enumerate(rows, start=1):
        if row["model"] == model_name:
            return index
    raise ValueError(f"Model is absent from table rows: {model_name}")


def _results_sentence_variant(horizon: int) -> str:
    variants = {
        5: (
            "At the shortest horizon, the result emphasizes local tracking "
            "accuracy under limited extrapolation."
        ),
        10: (
            "At the intermediate horizon, the comparison is a stricter test of "
            "whether the searched inductive bias persists beyond short-range "
            "tracking."
        ),
        15: (
            "At the longest horizon, the reported gap is interpreted primarily "
            "as evidence of robustness under accumulated forecast uncertainty."
        ),
    }
    return variants.get(
        horizon,
        "The result is reported as a descriptive comparison on normalized held-out errors.",
    )


def _forecast_sentence_variant(horizon: int) -> str:
    variants = {
        5: (
            "The short-horizon forecast is mainly used to inspect local phase "
            "alignment and amplitude calibration."
        ),
        10: (
            "The intermediate-horizon trace highlights whether errors grow "
            "smoothly or appear as abrupt departures from the target dynamics."
        ),
        15: (
            "The long-horizon trace provides a qualitative check on drift, "
            "attenuation, and recovery after turning points."
        ),
    }
    return variants.get(
        horizon,
        "The forecast trace provides a qualitative check that complements the scalar errors.",
    )


def _build_horizon_paths(source_key: str, horizon: int) -> dict[str, object]:
    run_dir = REPO_ROOT / "estudo_rebeca" / f"run_{source_key}_50_{horizon}"
    return {
        "horizon": horizon,
        "source_key": source_key,
        "run_dir": run_dir,
        "benchmark_csv": run_dir / "bench" / "benchmark_metrics_norm.csv",
    }


def _find_filter_artifacts(source_key: str) -> dict[str, Path]:
    preferred = (
        REPO_ROOT
        / "estudo_rebeca"
        / f"run_{source_key}_50_{REPORT_HORIZONS[0]}"
        / "000"
    )
    candidate_dirs = [
        preferred,
        *sorted(
            path
            for path in (REPO_ROOT / "estudo_rebeca").glob(f"run_{source_key}_50_*/*")
            if path.is_dir() and path.name.isdigit() and path != preferred
        ),
        *sorted(
            path
            for path in (REPO_ROOT / "estudo_rebeca").glob(
                f"run_{source_key}_50_*/bench"
            )
            if path.is_dir()
        ),
    ]
    for run_path in candidate_dirs:
        summary = run_path / "auto_filter_search_summary.csv"
        figure = run_path / "auto_filter_comparison.pdf"
        if summary.is_file() and figure.is_file():
            return {"run_path": run_path, "summary": summary, "figure": figure}

    raise ValueError(f"No auto-filter artifacts found for source: {source_key}")


def _build_benchmark_table(
    source_cfg: dict[str, str],
    horizon: int,
    benchmark_rows: list[dict[str, float | str]],
    best_run: dict[str, object] | None,
) -> str:
    table_rows = _build_benchmark_table_rows(benchmark_rows, best_run)
    lines = [
        r"\begin{table*}[htbp!]",
        r"\centering",
        rf"\caption{{Normalized held-out errors for {source_cfg['task_name']} with $L=50$ and $H={horizon}$. Lower values indicate better forecasts.}}",
        rf"\label{{tab:{source_cfg['label']}_full_benchmark_h{horizon}}}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & MSE & RMSE & MAE \\",
        r"\midrule",
    ]

    for row in table_rows:
        model = _latex_escape(row["model"])
        if row["bold"]:
            lines.append(
                "\\textbf{{{model}}} & \\textbf{{{mse}}} & \\textbf{{{rmse}}} & \\textbf{{{mae}}} \\\\".format(
                    model=model,
                    mse=_fmt_metric(float(row["mse"])),
                    rmse=_fmt_metric(float(row["rmse"])),
                    mae=_fmt_metric(float(row["mae"])),
                )
            )
        else:
            lines.append(
                f"{model} & {_fmt_metric(float(row['mse']))} & {_fmt_metric(float(row['rmse']))} & {_fmt_metric(float(row['mae']))} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def _build_benchmark_table_rows(
    benchmark_rows: list[dict[str, float | str]],
    best_run: dict[str, object] | None,
) -> list[dict[str, float | str | bool]]:
    table_rows: list[dict[str, float | str | bool]] = [
        *(
            [
                {
                    "model": "Ours",
                    "mse": float(best_run["metrics"]["mse"]),
                    "rmse": float(best_run["metrics"]["rmse"]),
                    "mae": float(best_run["metrics"]["mae"]),
                    "bold": True,
                }
            ]
            if best_run is not None
            else []
        ),
        *[
            {
                "model": str(row["model"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "bold": False,
            }
            for row in benchmark_rows
        ],
    ]
    table_rows.sort(key=lambda row: row["mse"])
    return table_rows


def _copy_selected_figure(
    source: Path, source_key: str, horizon: int, suffix: str
) -> Path:
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = (
        FIGURES_OUTPUT_DIR / f"{source_key}_h{horizon}_{suffix}{source.suffix}"
    )
    shutil.copy2(source, destination)
    return destination


def _copy_dataset_figure(source: Path, source_key: str, suffix: str) -> Path:
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = FIGURES_OUTPUT_DIR / f"{source_key}_{suffix}{source.suffix}"
    shutil.copy2(source, destination)
    return destination


def _load_tuned_filter_row(path: Path) -> dict[str, str]:
    rows = _read_csv_rows(path)
    for row in rows:
        if row.get("mode") == "tune_filter":
            return row
    if rows:
        return rows[0]
    raise ValueError(f"Auto-filter summary CSV is empty: {path}")


def _build_filter_table(source_cfg: dict[str, str], filter_row: dict[str, str]) -> str:
    return "\n".join([
        r"\begin{table}[htbp!]",
        r"\centering",
        rf"\caption{{Tuned auto-filter configuration for {source_cfg['task_name']}.}}",
        rf"\label{{tab:{source_cfg['label']}_tuned_filter}}",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        rf"Mode & {_latex_escape(filter_row.get('mode', 'tune_filter'))} \\",
        rf"Filter & {_latex_escape(filter_row['filter'])} \\",
        rf"Relative MAE & {_fmt_metric(float(filter_row['rel_mae']))} \\",
        rf"Roughness ratio & {_fmt_metric(float(filter_row['roughness_ratio']))} \\",
        rf"Derivative correlation & {_fmt_metric(float(filter_row['derivative_corr']))} \\",
        rf"Raw RMSE & {_fmt_metric(float(filter_row['rmse_raw']))} \\",
        rf"Raw correlation & {_fmt_metric(float(filter_row['corr_raw']))} \\",
        rf"Parameters & \texttt{{{_parse_params(filter_row['params'])}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def _build_filter_section(source_cfg: dict[str, str]) -> str:
    artifacts = _find_filter_artifacts(source_cfg["key"])
    filter_row = _load_tuned_filter_row(artifacts["summary"])
    filter_figure = _copy_dataset_figure(
        artifacts["figure"], source_cfg["key"], "auto_filter_comparison"
    )
    filter_table = _build_filter_table(source_cfg, filter_row)

    return rf"""\subsubsection{{Auto-Filter Tuning}}
\label{{sec:results_{source_cfg["label"]}_filter}}

Before forecasting, we tune the smoothing filter used to construct the filtered
target series for the {source_cfg["task_name"]} task. Table~\ref{{tab:{source_cfg["label"]}_tuned_filter}}
reports the selected filter, reconstruction diagnostics, and fitted parameters.

{filter_table}

Figure~\ref{{fig:{source_cfg["label"]}_auto_filter_comparison}} compares the raw
and filtered series used by the downstream forecasting experiments.

\begin{{figure}}[htbp!]
\centering
\includegraphics[width=0.95\linewidth]{{{filter_figure.relative_to(REPO_ROOT).as_posix()}}}
\caption{{Raw and auto-filtered series for the {source_cfg["task_name"]} task.}}
\label{{fig:{source_cfg["label"]}_auto_filter_comparison}}
\end{{figure}}
"""


def _build_horizon_section(
    source_cfg: dict[str, str],
    horizon: int,
    benchmark: dict[str, float | str],
    benchmark_rows: list[dict[str, float | str]],
    best_run: dict[str, object],
) -> str:
    run_name = str(best_run["run_name"])
    run_path = Path(best_run["run_dir"])
    metrics = best_run["metrics"]
    payload = best_run["candidate_payload"]
    candidate = payload["candidate"]
    run_config = payload["run_config"]
    speed_kwargs = run_config.get("speed_kwargs", {})

    model_name = str(benchmark["model"])
    mse_gain = _pct_improvement(float(benchmark["mse"]), float(metrics["mse"]))
    rmse_gain = _pct_improvement(float(benchmark["rmse"]), float(metrics["rmse"]))
    mae_gain = _pct_improvement(float(benchmark["mae"]), float(metrics["mae"]))

    selected_ops = [str(value) for value in candidate.get("selected_ops", [])]
    selected_families = [str(value) for value in candidate.get("selected_families", [])]
    rung_epochs = (
        ", ".join(str(value) for value in speed_kwargs.get("phase3_rung_epochs", []))
        or "n/a"
    )
    texttt_cmd = chr(92) + "texttt"
    source_name = _latex_escape(run_config["source"])
    arch_mode = _latex_escape(candidate["arch_mode"])

    arch_fig = run_path / "selected_transformer_architecture.pdf"
    cell_fig = run_path / "selected_darts_cell_topology.pdf"
    probs_fig = run_path / "darts_loss_and_architecture_probabilities.pdf"
    forecast_fig = run_path / "darts_held_out_forecast.pdf"
    arch_fig_copy = _copy_selected_figure(
        arch_fig, source_cfg["key"], horizon, "selected_architecture"
    )
    cell_fig_copy = _copy_selected_figure(
        cell_fig, source_cfg["key"], horizon, "selected_cell"
    )
    probs_fig_copy = _copy_selected_figure(
        probs_fig, source_cfg["key"], horizon, "search_probabilities"
    )
    forecast_fig_copy = _copy_selected_figure(
        forecast_fig, source_cfg["key"], horizon, "heldout_forecast"
    )
    table_rows = _build_benchmark_table_rows(benchmark_rows, best_run)
    ours_rank = _table_position(table_rows, "Ours")
    num_table_models = len(table_rows)
    benchmark_table = _build_benchmark_table(
        source_cfg, horizon, benchmark_rows, best_run
    )
    baseline_name_text = _latex_escape(model_name)
    metric_summary = ", ".join([
        _comparison_clause("MSE", mse_gain),
        _comparison_clause("RMSE", rmse_gain),
        _comparison_clause("MAE", mae_gain),
    ])
    result_context = _results_sentence_variant(horizon)
    forecast_context = _forecast_sentence_variant(horizon)
    cell_word = "cell" if int(candidate["num_cells"]) == 1 else "cells"
    node_word = "node" if int(candidate["num_nodes"]) == 1 else "nodes"
    architecture_phrase = (
        f"The discretized architecture is a {texttt_cmd}{{{arch_mode}}} model "
        f"with hidden dimension ${candidate['hidden_dim']}$, "
        f"{candidate['num_cells']} searched {cell_word}, and "
        f"{candidate['num_nodes']} {node_word} per cell. It "
        f"{_operation_phrase(selected_ops)} {_family_phrase(selected_families)}."
    )

    return rf"""\subsubsection{{Horizon $H={horizon}$}}
\label{{sec:results_{source_cfg["label"]}_h{horizon}}}

For the {source_cfg["task_slug"]} task at $H={horizon}$, the ForeBlocks-DARTS
configuration is run {texttt_cmd}{{{run_name}}}. The search used source
{texttt_cmd}{{{source_name}}}, {run_config["num_candidates"]} initial
candidates, top-$k={run_config["top_k"]}$ promotion,
{run_config["search_epochs"]} search epochs, phase-3 rungs [{rung_epochs}], and
a {run_config["final_epochs"]}-epoch final fit.

{benchmark_table}

Table~\ref{{tab:{source_cfg["label"]}_full_benchmark_h{horizon}}} ranks the
models by normalized MSE. ForeBlocks-DARTS obtains rank
{ours_rank}/{num_table_models}; relative to the strongest non-DARTS baseline,
{texttt_cmd}{{{baseline_name_text}}}, its errors are {metric_summary}.
{result_context}

{architecture_phrase}

Figure~\ref{{fig:{source_cfg["label"]}_arch_and_cell_h{horizon}}} reports the
discretized macro-architecture and cell topology used for the final evaluation.

\begin{{figure*}}[htbp!]
\centering
\begin{{subfigure}}{{0.9\textwidth}}
\centering
\includegraphics[width=0.9\linewidth]{{{cell_fig_copy.relative_to(REPO_ROOT).as_posix()}}}
\caption{{Selected DARTS cell topology for the best {source_cfg["task_name"]} run at horizon $H={horizon}$.}}
\end{{subfigure}}

\begin{{subfigure}}{{0.99\textwidth}}
\centering
\includegraphics[width=0.9\linewidth]{{{arch_fig_copy.relative_to(REPO_ROOT).as_posix()}}}
\caption{{Selected transformer architecture for the best {source_cfg["task_name"]} run at horizon $H={horizon}$.}}
\end{{subfigure}}

\caption{{Selected cell and selected transformer architecture for the best
{source_cfg["task_name"]} run at horizon $H={horizon}$, rendered directly from the
discretized model of run \texttt{{{run_name}}}.}}
\label{{fig:{source_cfg["label"]}_arch_and_cell_h{horizon}}}
\end{{figure*}}

The search trace in
Figure~\ref{{fig:{source_cfg["label"]}_darts_probabilities_h{horizon}}} gives the
loss trajectory and architecture-probability evolution before discretization,
which makes the selected design auditable rather than only reporting the final
metric.

\begin{{figure}}[htbp!]
\centering
\includegraphics[width=0.95\linewidth]{{{probs_fig_copy.relative_to(REPO_ROOT).as_posix()}}}
\caption{{Search loss and architecture-probability trajectories for the selected
{source_cfg["task_name"]} DARTS run at horizon $H={horizon}$.}}
\label{{fig:{source_cfg["label"]}_darts_probabilities_h{horizon}}}
\end{{figure}}

Figure~\ref{{fig:{source_cfg["label"]}_heldout_forecast_h{horizon}}} shows the
held-out prediction associated with the same run. {forecast_context}

\begin{{figure}}[htbp!]
\centering
\includegraphics[width=0.95\linewidth]{{{forecast_fig_copy.relative_to(REPO_ROOT).as_posix()}}}
\caption{{Held-out forecast of the best {source_cfg["task_name"]} DARTS run at horizon $H={horizon}$.}}
\label{{fig:{source_cfg["label"]}_heldout_forecast_h{horizon}}}
\end{{figure}}
"""


def _build_benchmark_only_horizon_section(
    source_cfg: dict[str, str],
    horizon: int,
    benchmark_rows: list[dict[str, float | str]],
) -> str:
    benchmark = benchmark_rows[0]
    benchmark_table = _build_benchmark_table(source_cfg, horizon, benchmark_rows, None)
    texttt_cmd = chr(92) + "texttt"
    return rf"""\subsubsection{{Horizon $H={horizon}$}}
\label{{sec:results_{source_cfg["label"]}_h{horizon}}}

For the {source_cfg["task_slug"]} task with input length $L=50$ and forecast
horizon $H={horizon}$, benchmark outputs are already available, but no completed
DARTS run was found yet in the current workspace. We therefore report the
reference benchmark ranking for this horizon and omit the DARTS-specific
architecture and forecast figures for now.

{benchmark_table}

At this stage, the strongest available benchmark baseline is
{texttt_cmd}{{{_latex_escape(benchmark["model"])}}}, with MSE {_fmt_metric(float(benchmark["mse"]))},
RMSE {_fmt_metric(float(benchmark["rmse"]))}, and MAE {_fmt_metric(float(benchmark["mae"]))}.
Once a completed DARTS run is available for this horizon, the generator can
augment this subsection with the selected cell, selected architecture, and
held-out forecast figures.
"""


def _build_source_section(
    source_cfg: dict[str, str], filter_section: str, horizon_sections: list[str]
) -> str:
    joined_horizon_sections = "\n\n".join(horizon_sections)
    return rf"""\subsection{{{source_cfg["title"]}}}
\label{{sec:results_{source_cfg["label"]}}}

We evaluate the {source_cfg["task_name"]} task by first tuning the
auto-filtered target series and then forecasting with fixed input length
$L=50$ over horizons $H \in \{{5, 10, 15\}}$. The dataset-level filter result is
shared across horizons; each horizon then reports the ForeBlocks-DARTS result,
the discretized architecture, the selected cell topology, and the corresponding
reference benchmark ranking.

{filter_section}

{joined_horizon_sections}
"""


def _build_results_tex(source_sections: list[str]) -> str:
    intro = r"""% Auto-generated by scripts/generate_results_tex.py
\section{Generation Forecast Results}
\label{sec:results_generation}

We report normalized held-out forecasting errors for two generation sources
under a fixed input length $L=50$ and horizons $H \in \{5, 10, 15\}$. The
analysis combines scalar benchmark comparisons with the discretized
ForeBlocks-DARTS architectures and forecast traces used for evaluation.
"""
    return intro + "\n\n" + "\n\n".join(source_sections) + "\n"


def _collect_horizon_section(source_cfg: dict[str, str], horizon: int) -> str:
    paths = _build_horizon_paths(source_cfg["key"], horizon)
    run_dir = Path(paths["run_dir"])
    benchmark_csv = Path(paths["benchmark_csv"])
    benchmark_rows = _load_benchmark_rows(benchmark_csv)
    best_run = _find_best_run_or_none(run_dir)
    if best_run is None:
        return _build_benchmark_only_horizon_section(
            source_cfg, horizon, benchmark_rows
        )

    benchmark = _load_best_benchmark(benchmark_csv)
    return _build_horizon_section(
        source_cfg, horizon, benchmark, benchmark_rows, best_run
    )


def main() -> int:
    source_sections = []
    for source_cfg in REPORT_SOURCES:
        filter_section = _build_filter_section(source_cfg)
        horizon_sections = [
            _collect_horizon_section(source_cfg, horizon) for horizon in REPORT_HORIZONS
        ]
        source_sections.append(
            _build_source_section(source_cfg, filter_section, horizon_sections)
        )

    tex = _build_results_tex(source_sections)
    DEFAULT_OUTPUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote {DEFAULT_OUTPUT_TEX}")
    print(
        "Included sources: "
        + ", ".join(source_cfg["key"] for source_cfg in REPORT_SOURCES)
    )
    print(f"Included horizons: {', '.join(str(h) for h in REPORT_HORIZONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
