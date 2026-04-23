from __future__ import annotations

import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from foreblocks.mltracker import MLTracker


DEFAULT_TRACKING_URI = str(Path(__file__).resolve().parent / "mltracker_data")


def _sparkline(values: list[float], width: int = 80) -> str:
    if not values:
        return ""
    ticks = "▁▂▃▄▅▆▇█"
    n = len(values)
    width = max(8, int(width))
    if n > width:
        step = n / width
        sampled: list[float] = []
        i = 0.0
        while int(i) < n and len(sampled) < width:
            sampled.append(values[int(i)])
            i += step
        values_use = sampled
    else:
        values_use = values

    vmin = min(values_use)
    vmax = max(values_use)
    if abs(vmax - vmin) < 1e-12:
        return ticks[0] * len(values_use)

    out = []
    levels = len(ticks) - 1
    for v in values_use:
        idx = int((v - vmin) / (vmax - vmin) * levels)
        idx = max(0, min(levels, idx))
        out.append(ticks[idx])
    return "".join(out)


def _fetch_runs(
    tracker: MLTracker, experiment_name: str | None = None
) -> list[dict]:
    return tracker.search_runs(experiment_name=experiment_name)


def _fetch_artifacts(tracker: MLTracker, run_id: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    conn = sqlite3.connect(tracker.db_path)
    conn.row_factory = sqlite3.Row
    try:
        q = "SELECT path, artifact_type FROM artifacts WHERE run_id = ? ORDER BY path"
        for row in conn.execute(q, (run_id,)):
            rows.append((str(row["path"]), str(row["artifact_type"])))
    finally:
        conn.close()
    return rows


def _fetch_metric_history(
    tracker: MLTracker, run_id: str
) -> list[tuple[str, int, float, str]]:
    rows: list[tuple[str, int, float, str]] = []
    conn = sqlite3.connect(tracker.db_path)
    conn.row_factory = sqlite3.Row
    try:
        q = """
        SELECT key, step, value, timestamp
        FROM metrics
        WHERE run_id = ?
        ORDER BY key, step ASC, timestamp ASC
        """
        for row in conn.execute(q, (run_id,)):
            rows.append(
                (
                    str(row["key"]),
                    int(row["step"]),
                    float(row["value"]),
                    str(row["timestamp"]),
                )
            )
    finally:
        conn.close()
    return rows


def create_app(tracking_uri: str = DEFAULT_TRACKING_URI):
    try:
        from textual.app import App
        from textual.app import ComposeResult
        from textual.containers import Horizontal
        from textual.containers import Vertical
        from textual.widgets import DataTable
        from textual.widgets import Footer
        from textual.widgets import Header
        from textual.widgets import Input
        from textual.widgets import Static
        from textual.widgets import TabbedContent
        from textual.widgets import TabPane
    except Exception as exc:
        raise RuntimeError(
            "Textual is required for the MLTracker TUI. Install with: pip install textual"
        ) from exc

    class MLTrackerTUI(App):
        CSS = """
        Screen {
            layout: vertical;
        }

        #top {
            height: 3;
            padding: 0 1;
        }

        #main {
            height: 1fr;
        }

        #left {
            width: 42%;
            border: round $accent;
            padding: 1;
        }

        #right {
            width: 58%;
            border: round $accent;
            padding: 1;
        }

        #metric_plot {
            height: 6;
            border: round $secondary;
            padding: 0 1;
            margin-bottom: 1;
        }

        #metric_hist_table {
            height: 1fr;
        }

        DataTable {
            height: 1fr;
        }
        """

        BINDINGS = [
            ("r", "refresh", "Refresh"),
            ("o", "open_artifact", "Open Artifact"),
            ("l", "toggle_log_scale", "Toggle Log Plot"),
            ("p", "toggle_metric_plot", "Toggle Plot/Table"),
            ("q", "quit", "Quit"),
        ]

        def __init__(self, tracking_uri: str):
            super().__init__()
            self.tracker = MLTracker(tracking_uri=tracking_uri)
            self._runs: list[dict] = []
            self._selected_run_id: str | None = None
            self._history_by_metric: dict[str, list[tuple[int, float, str]]] = {}
            self._metric_log_scale: bool = False
            self._metric_plot_enabled: bool = True
            self._current_metric_key: str | None = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="top"):
                yield Static("Experiment filter:")
                yield Input(placeholder="all experiments", id="exp_filter")
            with Horizontal(id="main"):
                with Vertical(id="left"):
                    yield Static("Runs")
                    yield DataTable(id="runs_table")
                with Vertical(id="right"):
                    with TabbedContent():
                        with TabPane("Overview", id="tab_overview"):
                            yield DataTable(id="overview_table")
                        with TabPane("Params", id="tab_params"):
                            yield DataTable(id="params_table")
                        with TabPane("Metrics", id="tab_metrics"):
                            yield DataTable(id="metrics_table")
                        with TabPane("Tags", id="tab_tags"):
                            yield DataTable(id="tags_table")
                        with TabPane("Metric History", id="tab_metric_hist"):
                            yield Static("Select a metric to plot.", id="metric_plot")
                            yield DataTable(id="metric_hist_table")
                        with TabPane("Artifacts", id="tab_artifacts"):
                            yield DataTable(id="artifacts_table")
            yield Footer()

        def on_mount(self) -> None:
            runs_table = self.query_one("#runs_table", DataTable)
            runs_table.add_columns("run_id", "name", "status", "start_time", "end_time")
            runs_table.cursor_type = "row"

            for table_id in [
                "overview_table",
                "params_table",
                "metrics_table",
                "tags_table",
                "metric_hist_table",
                "artifacts_table",
            ]:
                table = self.query_one(f"#{table_id}", DataTable)
                table.cursor_type = "row"

            self._setup_detail_tables()
            self._reload_runs()

        def _setup_detail_tables(self) -> None:
            self.query_one("#overview_table", DataTable).add_columns("field", "value")
            self.query_one("#params_table", DataTable).add_columns("param", "value")
            self.query_one("#metrics_table", DataTable).add_columns("metric", "latest")
            self.query_one("#tags_table", DataTable).add_columns("tag", "value")
            self.query_one("#metric_hist_table", DataTable).add_columns(
                "metric", "step", "value", "timestamp"
            )
            self.query_one("#artifacts_table", DataTable).add_columns("path", "type")

        def action_refresh(self) -> None:
            self._reload_runs()

        def action_toggle_log_scale(self) -> None:
            self._metric_log_scale = not self._metric_log_scale
            mode = "log" if self._metric_log_scale else "linear"
            self._notify(f"Metric plot scale: {mode}")
            self._update_metric_plot(self._current_metric_key)

        def action_toggle_metric_plot(self) -> None:
            self._metric_plot_enabled = not self._metric_plot_enabled
            mode = "enabled" if self._metric_plot_enabled else "table-only"
            self._notify(f"Metric plot {mode}")
            self._update_metric_plot(self._current_metric_key)

        def _notify(self, message: str, *, severity: str = "information") -> None:
            notifier = getattr(self, "notify", None)
            if callable(notifier):
                try:
                    notifier(message, severity=severity)
                    return
                except Exception:
                    pass

        def _artifact_abs_path(self, rel_path: str) -> Path | None:
            if not self._selected_run_id:
                return None
            return (
                Path(self.tracker.artifacts_path)
                / str(self._selected_run_id)
                / str(rel_path)
            )

        def _open_with_os(self, path: Path) -> bool:
            try:
                if sys.platform.startswith("linux"):
                    subprocess.Popen(["xdg-open", str(path)])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(path)])
                elif os.name == "nt":
                    os.startfile(str(path))  # type: ignore[attr-defined]
                else:
                    return False
                return True
            except Exception:
                return False

        def _open_selected_artifact(self) -> None:
            table = self.query_one("#artifacts_table", DataTable)
            if table.row_count == 0 or table.cursor_row is None:
                self._notify("No artifact selected.", severity="warning")
                return

            rel_path = str(table.get_row_at(table.cursor_row)[0])
            abs_path = self._artifact_abs_path(rel_path)
            if abs_path is None:
                self._notify("No active run selected.", severity="warning")
                return
            if not abs_path.exists():
                self._notify(f"Artifact not found: {abs_path}", severity="error")
                return

            if self._open_with_os(abs_path):
                self._notify(f"Opened: {abs_path}")
            else:
                self._notify(
                    f"Could not open artifact with OS handler: {abs_path}",
                    severity="error",
                )

        def action_open_artifact(self) -> None:
            self._open_selected_artifact()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "exp_filter":
                self._reload_runs()

        def _reload_runs(self) -> None:
            filter_input = self.query_one("#exp_filter", Input)
            exp_name = (filter_input.value or "").strip() or None
            self._runs = _fetch_runs(self.tracker, experiment_name=exp_name)

            table = self.query_one("#runs_table", DataTable)
            table.clear()
            for run in self._runs:
                table.add_row(
                    str(run.get("run_id", "")),
                    str(run.get("name") or ""),
                    str(run.get("status") or ""),
                    str(run.get("start_time") or ""),
                    str(run.get("end_time") or ""),
                )

            if self._runs:
                self._selected_run_id = str(self._runs[0]["run_id"])
                self._load_run_details(self._selected_run_id)
            else:
                self._selected_run_id = None
                self._clear_details()

        def _clear_details(self) -> None:
            for table_id in [
                "overview_table",
                "params_table",
                "metrics_table",
                "tags_table",
                "metric_hist_table",
                "artifacts_table",
            ]:
                self.query_one(f"#{table_id}", DataTable).clear()
            self._history_by_metric = {}
            self._current_metric_key = None
            self.query_one("#metric_plot", Static).update("Select a metric to plot.")

        def _update_metric_plot(self, metric_key: str | None) -> None:
            plot = self.query_one("#metric_plot", Static)
            if not self._metric_plot_enabled:
                plot.update("Plot hidden (table-only mode). Press 'p' to show plot.")
                return
            if not metric_key or metric_key not in self._history_by_metric:
                plot.update("Select a metric to plot.")
                return

            points = self._history_by_metric[metric_key]
            vals = [float(v) for _, v, _ in points]
            steps = [int(s) for s, _, _ in points]
            vals_plot = vals
            if self._metric_log_scale:
                vals_plot = [
                    (1.0 if v >= 0 else -1.0)
                    * (0.0 if abs(v) < 1e-12 else float(math.log1p(abs(v))))
                    for v in vals
                ]
            line = _sparkline(vals_plot, width=88)
            vmin = min(vals)
            vmax = max(vals)
            vlast = vals[-1]
            s0 = steps[0]
            s1 = steps[-1]
            scale = "log" if self._metric_log_scale else "linear"
            plot.update(
                "\n".join(
                    [
                        f"{metric_key}  (n={len(vals)}, step {s0}→{s1}, {scale})",
                        line,
                        f"min={vmin:.6g}   max={vmax:.6g}   last={vlast:.6g}",
                    ]
                )
            )

        def _load_run_details(self, run_id: str) -> None:
            run = self.tracker.get_run(run_id)
            artifacts = _fetch_artifacts(self.tracker, run_id)
            history = _fetch_metric_history(self.tracker, run_id)

            overview = self.query_one("#overview_table", DataTable)
            params = self.query_one("#params_table", DataTable)
            metrics = self.query_one("#metrics_table", DataTable)
            tags = self.query_one("#tags_table", DataTable)
            hist = self.query_one("#metric_hist_table", DataTable)
            arts = self.query_one("#artifacts_table", DataTable)

            overview.clear()
            params.clear()
            metrics.clear()
            tags.clear()
            hist.clear()
            arts.clear()

            for key in [
                "run_id",
                "experiment_id",
                "name",
                "status",
                "start_time",
                "end_time",
            ]:
                overview.add_row(key, str(run.get(key, "")))

            for k, v in sorted((run.get("params") or {}).items()):
                params.add_row(str(k), str(v))

            for k, v in sorted((run.get("metrics") or {}).items()):
                metrics.add_row(str(k), f"{float(v):.6g}")

            for k, v in sorted((run.get("tags") or {}).items()):
                tags.add_row(str(k), str(v))

            for key, step, value, ts in history:
                hist.add_row(str(key), str(step), f"{value:.6g}", str(ts))

            self._history_by_metric = {}
            for key, step, value, ts in history:
                self._history_by_metric.setdefault(str(key), []).append(
                    (int(step), float(value), str(ts))
                )
            if self._history_by_metric:
                first_key = sorted(self._history_by_metric.keys())[0]
                self._current_metric_key = first_key
                self._update_metric_plot(first_key)
            else:
                self._current_metric_key = None
                self._update_metric_plot(None)

            for path, typ in artifacts:
                arts.add_row(path, typ)

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            if event.data_table.id == "artifacts_table":
                self._open_selected_artifact()
                return
            if event.data_table.id == "metric_hist_table":
                metric_key = str(event.data_table.get_row_at(event.cursor_row)[0])
                self._current_metric_key = metric_key
                self._update_metric_plot(metric_key)
                return
            if event.data_table.id != "runs_table":
                return
            rid = str(event.data_table.get_row_at(event.cursor_row)[0])
            self._selected_run_id = rid
            self._load_run_details(rid)

    return MLTrackerTUI(tracking_uri=tracking_uri)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MLTracker Textual TUI")
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="Path to MLTracker directory",
    )
    args = parser.parse_args()

    app = create_app(tracking_uri=args.tracking_uri)
    app.run()


if __name__ == "__main__":
    main()
