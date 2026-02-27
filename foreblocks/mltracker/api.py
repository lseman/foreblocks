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
from typing import Any, Dict, List, Optional

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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---- import your tracker (use the improved MLTracker you implemented) ----
from foreblocks.mltracker import MLTracker


# -----------------------------------------------------------------------------
# App (use lifespan to silence on_event deprecation)
# -----------------------------------------------------------------------------
def _tracker_root() -> str:
    return os.getenv("MLTRACKER_DIR", "./mltracker_data")


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

# Dashboard
dashboard_dir = Path(__file__).parent / "dashboard"
if dashboard_dir.exists():
    app.mount(
        "/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard"
    )


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


def _dur_seconds(start_iso: Optional[str], end_iso: Optional[str]) -> Optional[int]:
    if not start_iso or not end_iso:
        return None
    try:
        s = datetime.fromisoformat(start_iso.replace("Z", ""))
        e = datetime.fromisoformat(end_iso.replace("Z", ""))
        return int((e - s).total_seconds())
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ExperimentCreate(BaseModel):
    name: str


class RunCreate(BaseModel):
    experiment_name: str = "default"
    run_name: Optional[str] = None


class RunUpdate(BaseModel):
    status: str = Field("FINISHED", pattern="^(FINISHED|FAILED|RUNNING|CANCELED)$")


class ParamLog(BaseModel):
    key: str
    value: Any


class ParamsLog(BaseModel):
    params: Dict[str, Any]


class MetricLog(BaseModel):
    key: str
    value: float
    step: int = 0


class MetricsLog(BaseModel):
    metrics: Dict[str, float]
    step: int = 0


class TagSet(BaseModel):
    key: str
    value: str


class CompareRequest(BaseModel):
    run_ids: List[str] = Field(..., min_items=1, max_items=50)


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
        "artifacts_path": str(tracker.artifacts_path),
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
async def delete_run(run_id: str, tracker: MLTracker = Depends(get_tracker)):
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
    experiment_name: Optional[str] = None,
    status: Optional[str] = Query(None, pattern="^(FINISHED|FAILED|RUNNING|CANCELED)$"),
    q: Optional[str] = Query(None, description="search in name or run_id"),
    from_time: Optional[str] = Query(None, description="ISO start lower bound"),
    to_time: Optional[str] = Query(None, description="ISO start upper bound"),
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
    params: List[Any] = []

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
    metric_key: Optional[str] = None,
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
        hist: Dict[str, List[Dict[str, Any]]] = {}
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


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str, tracker: MLTracker = Depends(get_tracker)):
    """
    Permanently deletes a run:
      - rows in metrics/params/tags/artifacts/runs
      - artifacts directory on disk
    """
    try:
        # remove artifacts on disk
        art_dir = tracker.artifacts_path / run_id
        if art_dir.exists():
            shutil.rmtree(art_dir, ignore_errors=True)

        # delete rows
        with tracker._get_db() as conn:
            conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM params WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM tags WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            deleted = conn.execute(
                "DELETE FROM runs WHERE run_id = ?", (run_id,)
            ).rowcount

        if not deleted:
            raise HTTPException(status_code=404, detail="Run not found")
        return JSONResponse({"success": True, "run_id": run_id})
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
