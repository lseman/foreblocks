"""
MLTracker FastAPI Server (hardened + smarter)
- Returns full runs by default for dashboard cards (params/metrics/tags)
- Pagination, filters & sorting for /api/runs
- Safe artifact upload/download (no traversal)
- Background temp cleanup, better health
- Fixes BackgroundTasks usage
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---- import your tracker (use the improved MLTracker you implemented) ----
from foreblocks.mltracker import MLTracker


# -----------------------------------------------------------------------------
# App (use lifespan to silence on_event deprecation)
# -----------------------------------------------------------------------------
def _tracker_root() -> str:
    env_path = os.getenv("MLTRACKER_DIR")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    # Default to the package-local tracker storage so dashboard_v2 always
    # reads from foreblocks/mltracker/mltracker_data regardless of cwd.
    return str((Path(__file__).resolve().parent / "mltracker_data").resolve())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tracker = MLTracker(_tracker_root())
    yield
    # optional shutdown cleanup here


app = FastAPI(
    title="MLTracker API",
    description="REST API for ML experiment tracking",
    version="1.2.0",
    lifespan=lifespan,
)

# CORS (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://yourdomain"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mount_dashboard_static() -> None:
    base_dir = Path(__file__).parent
    v2_dist_dir = base_dir / "dashboard_v2" / "dist"

    if v2_dist_dir.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=v2_dist_dir, html=True),
            name="dashboard",
        )


_mount_dashboard_static()


@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")


# -----------------------------------------------------------------------------
# Dependencies & helpers
# -----------------------------------------------------------------------------
def get_tracker() -> MLTracker:
    t: MLTracker = getattr(app.state, "tracker", None)
    if t is None:
        raise HTTPException(status_code=500, detail="Tracker not initialized")
    return t


@contextmanager
def _as_active_run(tracker: MLTracker, run_id: str):
    """Temporarily impersonate a run safely."""
    original = tracker._active_run
    tracker._active_run = run_id
    try:
        yield
    finally:
        tracker._active_run = original


def _safe_under(base: Path, target: Path) -> Path:
    """Ensure target resolves under base, else 400."""
    base = base.resolve()
    tgt = (base / target).resolve()
    if not str(tgt).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return tgt


def _delete_file(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _dur_seconds(start_iso: str | None, end_iso: str | None) -> int | None:
    if not start_iso or not end_iso:
        return None
    try:
        s = datetime.fromisoformat(start_iso.replace("Z", ""))
        e = datetime.fromisoformat(end_iso.replace("Z", ""))
        return int((e - s).total_seconds())
    except Exception:
        return None


def _run_duration_sql() -> str:
    return (
        "strftime('%s', COALESCE(end_time, datetime('now'))) - "
        "strftime('%s', start_time)"
    )


def _coerce_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _metric_lower_is_better(metric_key: str) -> bool:
    key = metric_key.lower()
    return any(
        token in key
        for token in (
            "loss",
            "error",
            "rmse",
            "mae",
            "mse",
            "mape",
            "nll",
            "perplexity",
            "wer",
            "cer",
        )
    )


def _pick_primary_metric(metric_rows: list[dict[str, Any]]) -> str | None:
    if not metric_rows:
        return None
    preference = [
        "val_loss",
        "valid_loss",
        "loss",
        "val_rmse",
        "rmse",
        "val_mae",
        "mae",
        "mse",
        "accuracy",
        "acc",
        "f1",
        "auc",
    ]

    def rank(row: dict[str, Any]) -> tuple[int, int, str]:
        key = str(row["key"])
        try:
            pref = preference.index(key.lower())
        except ValueError:
            pref = 999
        return (-int(row.get("run_count") or 0), pref, key)

    return sorted(metric_rows, key=rank)[0]["key"]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ExperimentCreate(BaseModel):
    name: str


class RunCreate(BaseModel):
    experiment_name: str = "default"
    run_name: str | None = None


class RunUpdate(BaseModel):
    status: str = Field("FINISHED", pattern="^(FINISHED|FAILED|RUNNING|CANCELED)$")


class ParamLog(BaseModel):
    key: str
    value: Any


class ParamsLog(BaseModel):
    params: dict[str, Any]


class MetricLog(BaseModel):
    key: str
    value: float
    step: int = 0


class MetricsLog(BaseModel):
    metrics: dict[str, float]
    step: int = 0


class TagSet(BaseModel):
    key: str
    value: str


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(..., min_items=1, max_items=50)


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def health_check(tracker: MLTracker = Depends(get_tracker)):
    try:
        with tracker._get_db() as conn:
            conn.execute("SELECT 1")
        ok = tracker.artifacts_path.exists() and tracker.artifacts_path.is_dir()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health failed: {e}")
    return {
        "status": "healthy" if ok else "degraded",
        "tracker_path": str(tracker.tracking_uri),
        "sqlite_path": str(tracker.db_path),
        "artifacts_path": str(tracker.artifacts_path),
    }


@app.get("/api/overview")
async def get_overview(
    experiment_name: str | None = None,
    metric_key: str | None = None,
    objective: str = Query("auto", pattern="^(auto|min|max)$"),
    tracker: MLTracker = Depends(get_tracker),
):
    """
    Return a W&B-style project overview directly from the SQLite store.

    This endpoint is intentionally aggregate-heavy and full-run-light: the UI
    can render metric catalogs, health summaries, recent activity, and best-run
    cards without fetching every run and recomputing the same summaries.
    """
    where = ["1=1"]
    params: list[Any] = []
    experiment: dict[str, Any] | None = None

    if experiment_name:
        exp_id = tracker.get_experiment(experiment_name)
        if exp_id is None:
            return {
                "experiment": None,
                "totals": {
                    "runs": 0,
                    "finished": 0,
                    "running": 0,
                    "failed": 0,
                    "canceled": 0,
                    "success_rate": 0.0,
                    "avg_duration": None,
                    "recent_24h": 0,
                },
                "status_counts": {},
                "metric_catalog": [],
                "param_catalog": [],
                "primary_metric": None,
                "best_run": None,
                "recent_runs": [],
            }
        where.append("r.experiment_id = ?")
        params.append(exp_id)
        experiment = {"experiment_id": exp_id, "name": experiment_name}

    where_sql = " AND ".join(where)
    duration_sql = _run_duration_sql()

    with tracker._get_db() as conn:
        totals = conn.execute(
            f"""
            SELECT
                COUNT(*) AS runs,
                SUM(CASE WHEN status = 'FINISHED' THEN 1 ELSE 0 END) AS finished,
                SUM(CASE WHEN status = 'RUNNING' THEN 1 ELSE 0 END) AS running,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status = 'CANCELED' THEN 1 ELSE 0 END) AS canceled,
                AVG(CASE WHEN end_time IS NOT NULL THEN {duration_sql} END) AS avg_duration,
                SUM(CASE WHEN start_time >= datetime('now', '-1 day') THEN 1 ELSE 0 END) AS recent_24h
            FROM runs r
            WHERE {where_sql}
            """,
            params,
        ).fetchone()

        status_rows = conn.execute(
            f"""
            SELECT status, COUNT(*) AS count
            FROM runs r
            WHERE {where_sql}
            GROUP BY status
            ORDER BY count DESC
            """,
            params,
        ).fetchall()

        metric_rows_raw = conn.execute(
            f"""
            SELECT
                m.key,
                COUNT(*) AS point_count,
                COUNT(DISTINCT m.run_id) AS run_count,
                MIN(m.value) AS min,
                MAX(m.value) AS max,
                AVG(m.value) AS mean,
                MAX(m.step) AS max_step
            FROM metrics m
            JOIN runs r ON r.run_id = m.run_id
            WHERE {where_sql}
            GROUP BY m.key
            ORDER BY run_count DESC, point_count DESC, m.key ASC
            """,
            params,
        ).fetchall()

        param_rows = conn.execute(
            f"""
            SELECT p.key, p.value
            FROM params p
            JOIN runs r ON r.run_id = p.run_id
            WHERE {where_sql}
            """,
            params,
        ).fetchall()

        recent_rows = conn.execute(
            f"""
            SELECT r.run_id, r.name, r.status, r.start_time, r.end_time,
                   e.name AS experiment_name,
                   {duration_sql} AS duration
            FROM runs r
            JOIN experiments e ON e.experiment_id = r.experiment_id
            WHERE {where_sql}
            ORDER BY r.start_time DESC
            LIMIT 12
            """,
            params,
        ).fetchall()

        metric_catalog = [
            {
                "key": row["key"],
                "point_count": int(row["point_count"] or 0),
                "run_count": int(row["run_count"] or 0),
                "min": row["min"],
                "max": row["max"],
                "mean": row["mean"],
                "max_step": int(row["max_step"] or 0),
            }
            for row in metric_rows_raw
        ]

        primary_metric = metric_key or _pick_primary_metric(metric_catalog)
        best_run = None
        if primary_metric:
            direction = (
                "min"
                if objective == "auto" and _metric_lower_is_better(primary_metric)
                else "max"
                if objective == "auto"
                else objective
            )
            order = "ASC" if direction == "min" else "DESC"
            best_row = conn.execute(
                f"""
                WITH latest AS (
                    SELECT m.run_id, m.key, m.value, m.step,
                           ROW_NUMBER() OVER (
                               PARTITION BY m.run_id, m.key
                               ORDER BY m.step DESC, m.timestamp DESC
                           ) AS rn
                    FROM metrics m
                    JOIN runs r ON r.run_id = m.run_id
                    WHERE {where_sql} AND m.key = ?
                )
                SELECT r.run_id, r.name, r.status, r.start_time, r.end_time,
                       latest.value, latest.step
                FROM latest
                JOIN runs r ON r.run_id = latest.run_id
                WHERE latest.rn = 1
                ORDER BY latest.value {order}
                LIMIT 1
                """,
                params + [primary_metric],
            ).fetchone()
            if best_row is not None:
                best_run = {
                    "run_id": best_row["run_id"],
                    "name": best_row["name"],
                    "status": best_row["status"],
                    "start_time": best_row["start_time"],
                    "end_time": best_row["end_time"],
                    "metric_key": primary_metric,
                    "metric_value": best_row["value"],
                    "metric_step": best_row["step"],
                    "objective": direction,
                    "duration": _dur_seconds(best_row["start_time"], best_row["end_time"]),
                }

    param_stats: dict[str, dict[str, Any]] = {}
    for row in param_rows:
        key = row["key"]
        stat = param_stats.setdefault(
            key,
            {
                "key": key,
                "run_count": 0,
                "numeric_count": 0,
                "distinct_count": 0,
                "examples": [],
                "_values": set(),
                "_numeric": [],
            },
        )
        value = row["value"]
        stat["run_count"] += 1
        stat["_values"].add(value)
        if len(stat["examples"]) < 4 and value not in stat["examples"]:
            stat["examples"].append(value)
        numeric = _coerce_float(value)
        if numeric is not None:
            stat["numeric_count"] += 1
            stat["_numeric"].append(numeric)

    param_catalog = []
    for stat in param_stats.values():
        numeric_values = stat.pop("_numeric")
        values = stat.pop("_values")
        stat["distinct_count"] = len(values)
        if numeric_values:
            stat["min"] = min(numeric_values)
            stat["max"] = max(numeric_values)
            stat["mean"] = sum(numeric_values) / len(numeric_values)
        else:
            stat["min"] = stat["max"] = stat["mean"] = None
        param_catalog.append(stat)
    param_catalog.sort(key=lambda x: (-int(x["run_count"]), x["key"]))

    run_count = int(totals["runs"] or 0)
    finished = int(totals["finished"] or 0)
    return {
        "experiment": experiment,
        "totals": {
            "runs": run_count,
            "finished": finished,
            "running": int(totals["running"] or 0),
            "failed": int(totals["failed"] or 0),
            "canceled": int(totals["canceled"] or 0),
            "success_rate": (finished / run_count * 100.0) if run_count else 0.0,
            "avg_duration": int(totals["avg_duration"]) if totals["avg_duration"] else None,
            "recent_24h": int(totals["recent_24h"] or 0),
        },
        "status_counts": {row["status"]: int(row["count"] or 0) for row in status_rows},
        "metric_catalog": metric_catalog,
        "param_catalog": param_catalog[:200],
        "primary_metric": primary_metric,
        "best_run": best_run,
        "recent_runs": [
            {
                "run_id": row["run_id"],
                "name": row["name"],
                "status": row["status"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "experiment_name": row["experiment_name"],
                "duration": int(row["duration"]) if row["duration"] is not None else None,
            }
            for row in recent_rows
        ],
    }


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
@app.post("/api/experiments", status_code=201)
async def create_experiment(
    experiment: ExperimentCreate, tracker: MLTracker = Depends(get_tracker)
):
    try:
        exp_id = tracker.create_experiment(experiment.name)
        return {"experiment_id": exp_id, "name": experiment.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/experiments")
async def list_experiments(tracker: MLTracker = Depends(get_tracker)):
    with tracker._get_db() as conn:
        rows = conn.execute(
            """
            SELECT e.experiment_id, e.name, e.created_at,
                   COUNT(r.run_id) AS run_count
            FROM experiments e
            LEFT JOIN runs r ON r.experiment_id = e.experiment_id
            GROUP BY e.experiment_id
            ORDER BY e.created_at DESC
            """
        ).fetchall()
    return [
        {
            "experiment_id": r["experiment_id"],
            "name": r["name"],
            "created_at": r["created_at"],
            "run_count": r["run_count"],
        }
        for r in rows
    ]


@app.get("/api/experiments/{experiment_name}")
async def get_experiment(
    experiment_name: str, tracker: MLTracker = Depends(get_tracker)
):
    exp_id = tracker.get_experiment(experiment_name)
    if exp_id is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    with tracker._get_db() as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (exp_id,)
        ).fetchone()
    return {
        "experiment_id": row["experiment_id"],
        "name": row["name"],
        "created_at": row["created_at"],
    }


# -----------------------------------------------------------------------------
# Runs
# -----------------------------------------------------------------------------
@app.post("/api/runs", status_code=201)
async def create_run(run: RunCreate, tracker: MLTracker = Depends(get_tracker)):
    try:
        run_id = tracker.start_run(run.experiment_name, run.run_name)
        return {"run_id": run_id, "status": "RUNNING"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/runs/{run_id}/end")
async def end_run(
    run_id: str, update: RunUpdate, tracker: MLTracker = Depends(get_tracker)
):
    try:
        with _as_active_run(tracker, run_id):
            tracker.end_run(update.status)
        return {"run_id": run_id, "status": update.status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/runs/{run_id}")
async def delete_run_with_artifacts(
    run_id: str, tracker: MLTracker = Depends(get_tracker)
):
    try:
        tracker.delete_run(run_id)
        return {"deleted": run_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str, tracker: MLTracker = Depends(get_tracker)):
    try:
        d = tracker.get_run(run_id)
        d["duration"] = _dur_seconds(d.get("start_time"), d.get("end_time"))
        return d
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs")
async def search_runs(
    experiment_name: str | None = None,
    status: str | None = Query(None, pattern="^(FINISHED|FAILED|RUNNING|CANCELED)$"),
    q: str | None = Query(None, description="search in name or run_id"),
    from_time: str | None = Query(None, description="ISO start lower bound"),
    to_time: str | None = Query(None, description="ISO start upper bound"),
    sort: str = Query(
        "-start_time", description="start_time|-start_time|duration|-duration"
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include: str = Query(
        "full", pattern="^(summary|full)$"
    ),  # default FULL for your UI
    tracker: MLTracker = Depends(get_tracker),
):
    where = ["1=1"]
    params: list[Any] = []

    if experiment_name:
        exp_id = tracker.get_experiment(experiment_name)
        if exp_id is None:
            return {"runs": [], "count": 0}
        where.append("experiment_id = ?")
        params.append(exp_id)

    if status:
        where.append("status = ?")
        params.append(status)

    if from_time:
        where.append("start_time >= ?")
        params.append(from_time)
    if to_time:
        where.append("start_time <= ?")
        params.append(to_time)

    if q:
        where.append("(name LIKE ? OR run_id LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like])

    order_sql = {
        "start_time": " ORDER BY start_time ASC",
        "-start_time": " ORDER BY start_time DESC",
        "duration": " ORDER BY (strftime('%s', COALESCE(end_time, start_time)) - strftime('%s', start_time)) ASC",
        "-duration": " ORDER BY (strftime('%s', COALESCE(end_time, start_time)) - strftime('%s', start_time)) DESC",
    }.get(sort, " ORDER BY start_time DESC")

    sql_base = f"FROM runs WHERE {' AND '.join(where)}"
    sql_rows = f"SELECT run_id, experiment_id, name, status, start_time, end_time {sql_base}{order_sql} LIMIT ? OFFSET ?"
    sql_cnt = f"SELECT COUNT(*) AS c {sql_base}"

    with tracker._get_db() as conn:
        rows = conn.execute(sql_rows, params + [limit, offset]).fetchall()
        total = conn.execute(sql_cnt, params).fetchone()["c"]

    if include == "summary":
        runs = [
            {
                "run_id": r["run_id"],
                "experiment_id": r["experiment_id"],
                "name": r["name"],
                "status": r["status"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "duration": _dur_seconds(r["start_time"], r["end_time"]),
            }
            for r in rows
        ]
    else:
        runs = []
        for r in rows:
            full = tracker.get_run(r["run_id"])
            full["duration"] = _dur_seconds(
                full.get("start_time"), full.get("end_time")
            )
            runs.append(full)

    return {"runs": runs, "count": total}


# -----------------------------------------------------------------------------
# Logging endpoints (safe impersonation)
# -----------------------------------------------------------------------------
@app.post("/api/runs/{run_id}/params")
async def log_param(
    run_id: str, param: ParamLog, tracker: MLTracker = Depends(get_tracker)
):
    try:
        with _as_active_run(tracker, run_id):
            tracker.log_param(param.key, param.value)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/runs/{run_id}/params/batch")
async def log_params(
    run_id: str, payload: ParamsLog, tracker: MLTracker = Depends(get_tracker)
):
    try:
        with _as_active_run(tracker, run_id):
            tracker.log_params(payload.params)
        return {"success": True, "count": len(payload.params)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/runs/{run_id}/metrics")
async def log_metric(
    run_id: str, metric: MetricLog, tracker: MLTracker = Depends(get_tracker)
):
    try:
        with _as_active_run(tracker, run_id):
            tracker.log_metric(metric.key, metric.value, metric.step)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/runs/{run_id}/metrics/batch")
async def log_metrics(
    run_id: str, payload: MetricsLog, tracker: MLTracker = Depends(get_tracker)
):
    try:
        with _as_active_run(tracker, run_id):
            tracker.log_metrics(payload.metrics, payload.step)
        return {"success": True, "count": len(payload.metrics)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/runs/{run_id}/tags")
async def set_tag(run_id: str, tag: TagSet, tracker: MLTracker = Depends(get_tracker)):
    try:
        with _as_active_run(tracker, run_id):
            tracker.set_tag(tag.key, tag.value)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# Artifacts (safe upload & download)
# -----------------------------------------------------------------------------
@app.post("/api/runs/{run_id}/artifacts")
async def upload_artifact(
    run_id: str,
    background: BackgroundTasks,  # <-- plain param, no Depends()
    file: UploadFile = File(...),
    artifact_path: str = "",
    tracker: MLTracker = Depends(get_tracker),
):
    try:
        # write to secure temp file
        suffix = Path(file.filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        with _as_active_run(tracker, run_id):
            tracker.log_artifact(str(tmp_path), artifact_path)

        background.add_task(_delete_file, tmp_path)
        return {"success": True, "filename": Path(file.filename).name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/runs/{run_id}/artifacts")
async def list_artifacts(run_id: str, tracker: MLTracker = Depends(get_tracker)):
    try:
        with tracker._get_db() as conn:
            rows = conn.execute(
                "SELECT path, artifact_type FROM artifacts WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return {
            "artifacts": [{"path": r["path"], "type": r["artifact_type"]} for r in rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
async def download_artifact(
    run_id: str, artifact_path: str, tracker: MLTracker = Depends(get_tracker)
):
    try:
        safe_path = _safe_under(tracker.artifacts_path / run_id, Path(artifact_path))
        if not safe_path.exists() or not safe_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(
            path=safe_path,
            filename=safe_path.name,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Metrics history & comparison
# -----------------------------------------------------------------------------
@app.get("/api/runs/{run_id}/metrics/history")
async def get_metric_history(
    run_id: str,
    metric_key: str | None = None,
    tracker: MLTracker = Depends(get_tracker),
):
    try:
        with tracker._get_db() as conn:
            if metric_key:
                rows = conn.execute(
                    """SELECT key, value, step, timestamp FROM metrics
                       WHERE run_id = ? AND key = ? ORDER BY step""",
                    (run_id, metric_key),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT key, value, step, timestamp FROM metrics
                       WHERE run_id = ? ORDER BY key, step""",
                    (run_id,),
                ).fetchall()
        hist: dict[str, list[dict[str, Any]]] = {}
        for m in rows:
            hist.setdefault(m["key"], []).append(
                {"value": m["value"], "step": m["step"], "timestamp": m["timestamp"]}
            )
        return {"metrics": hist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/runs/compare")
async def compare_runs(
    payload: CompareRequest, tracker: MLTracker = Depends(get_tracker)
):
    try:
        if len(payload.run_ids) > 50:
            raise HTTPException(status_code=400, detail="Too many run_ids (max 50)")
        runs = [tracker.get_run(rid) for rid in payload.run_ids]
        all_metrics, all_params = set(), set()
        for r in runs:
            all_metrics.update(r["metrics"].keys())
            all_params.update(r["params"].keys())
        return {
            "runs": runs,
            "metric_keys": sorted(all_metrics),
            "param_keys": sorted(all_params),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
