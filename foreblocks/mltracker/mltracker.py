"""
MLTracker - Lightweight ML Experiment Tracking System (smart version)
- Autolog decorator with arg capture, helper injection, env/git tags
- Safer serialization & artifact helpers
- WAL mode, indices, and small quality-of-life upgrades
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import io
import json
import os
import pickle
import platform
import shutil
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Union

try:
    import getpass
except Exception:
    getpass = None

# -------------------------- utilities --------------------------

_JSON_LIMIT = 2000  # max chars per serialized value
_BIN_LIMIT = 10 * 1024 * 1024  # 10MB safety limit for artifacts logged via decorator


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_repr(x: Any, limit: int = _JSON_LIMIT) -> str:
    try:
        if dataclasses.is_dataclass(x):
            x = dataclasses.asdict(x)
        if isinstance(x, (dict, list, tuple, str, int, float, bool)) or x is None:
            s = json.dumps(x, default=str)
        else:
            s = repr(x)
    except Exception:
        s = f"<unrepr {type(x).__name__}>"
    if len(s) > limit:
        s = s[:limit] + f"... <{len(s) - limit} more>"
    return s


def _try_jsonable(x: Any) -> Any:
    # best-effort to turn x into jsonable
    try:
        if dataclasses.is_dataclass(x):
            return dataclasses.asdict(x)
        if isinstance(x, (dict, list, tuple, str, int, float, bool)) or x is None:
            return x
        # numpy / torch light support
        # note: not importing heavy deps—just duck-typing
        if hasattr(x, "tolist"):
            return x.tolist()
        if hasattr(x, "item") and callable(getattr(x, "item")):
            try:
                return x.item()
            except Exception:
                pass
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return _safe_repr(x)


def _hashdict(d: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            {k: _try_jsonable(v) for k, v in sorted(d.items())}, sort_keys=True
        )
    except Exception:
        payload = repr(sorted(d.items()))
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _maybe_git_info() -> Dict[str, str]:
    info = {}
    try:
        import subprocess

        def run(cmd):
            return (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            )

        root = run(["git", "rev-parse", "--show-toplevel"])
        info["git_root"] = root
        info["git_commit"] = run(["git", "rev-parse", "HEAD"])
        info["git_branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["git_dirty"] = "1" if run(["git", "status", "--porcelain"]) else "0"
    except Exception:
        pass
    return info


def _sys_info() -> Dict[str, str]:
    d = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "user": (getpass.getuser() if getpass else None) or "",
        "pid": str(os.getpid()),
    }
    # Heuristic CUDA/Torch availability without importing heavy libs
    try:
        import torch

        d["torch"] = torch.__version__
        d["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            d["cuda_device_count"] = str(torch.cuda.device_count())
    except Exception:
        pass
    return {k: v for k, v in d.items() if v is not None}


def _module_tree(module: Any, name: str = "model") -> Dict[str, Any]:
    children = []
    if hasattr(module, "named_children"):
        try:
            children = list(module.named_children())
        except Exception:
            children = []

    def _count_params(recurse: bool, trainable_only: bool = False) -> int:
        if not hasattr(module, "parameters"):
            return 0
        try:
            params = module.parameters(recurse=recurse)
            if trainable_only:
                return int(
                    sum(p.numel() for p in params if getattr(p, "requires_grad", False))
                )
            return int(sum(p.numel() for p in params))
        except Exception:
            return 0

    node = {
        "name": str(name),
        "type": module.__class__.__name__,
        "num_params": _count_params(recurse=False),
        "trainable_params": _count_params(recurse=False, trainable_only=True),
        "children": [],
    }

    for child_name, child in children:
        node["children"].append(_module_tree(child, child_name))
    return node


def _model_summary(model: Any) -> Dict[str, int]:
    if not hasattr(model, "parameters"):
        return {"total": 0, "trainable": 0, "non_trainable": 0}
    try:
        total = int(sum(p.numel() for p in model.parameters()))
        trainable = int(
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "requires_grad", False)
            )
        )
    except Exception:
        total = 0
        trainable = 0
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": max(0, total - trainable),
    }


# -------------------------- tracker --------------------------


class MLTracker:
    """Main tracking class for ML experiments"""

    def __init__(self, tracking_uri: str = "./mltracker"):
        self.tracking_uri = Path(tracking_uri)
        self.tracking_uri.mkdir(exist_ok=True)

        self.db_path = self.tracking_uri / "mltracker.db"
        self.artifacts_path = self.tracking_uri / "artifacts"
        self.artifacts_path.mkdir(exist_ok=True)

        self._init_db()
        self._active_run: Optional[str] = None

    # ---------- DB ----------
    @contextmanager
    def _get_db(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._get_db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id INTEGER NOT NULL,
                    name TEXT,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS params (
                    run_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (run_id, key),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    step INTEGER DEFAULT 0,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    run_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (run_id, key),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    PRIMARY KEY (run_id, path),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            # Helpful indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_exp_time ON runs(experiment_id, start_time DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_key_step ON metrics(run_id, key, step)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_params_key ON params(run_id, key)"
            )

    # ---------- Experiments ----------
    def create_experiment(self, name: str) -> int:
        with self._get_db() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO experiments (name, created_at) VALUES (?, ?)",
                (name, _now_iso()),
            )
            if cursor.lastrowid:  # created now
                return cursor.lastrowid
            row = conn.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?", (name,)
            ).fetchone()
            return int(row[0])

    def get_experiment(self, name: str) -> Optional[int]:
        with self._get_db() as conn:
            row = conn.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?", (name,)
            ).fetchone()
            return int(row[0]) if row else None

    # ---------- Runs ----------
    def start_run(
        self, experiment_name: str = "default", run_name: Optional[str] = None
    ) -> str:
        exp_id = self.get_experiment(experiment_name) or self.create_experiment(
            experiment_name
        )
        import uuid

        run_id = uuid.uuid4().hex[:16]

        with self._get_db() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, experiment_id, name, status, start_time) VALUES (?, ?, ?, ?, ?)",
                (run_id, exp_id, run_name, "RUNNING", _now_iso()),
            )

        self._active_run = run_id
        return run_id

    def end_run(self, status: str = "FINISHED"):
        if self._active_run:
            with self._get_db() as conn:
                conn.execute(
                    "UPDATE runs SET status = ?, end_time = ? WHERE run_id = ?",
                    (status, _now_iso(), self._active_run),
                )
            self._active_run = None

    # ---------- Logging ----------
    def log_param(self, key: str, value: Any):
        if not self._active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        with self._get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                (self._active_run, key, _safe_repr(value)),
            )

    def log_params(self, params: Mapping[str, Any]):
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: int = 0):
        if not self._active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        with self._get_db() as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, key, value, timestamp, step) VALUES (?, ?, ?, ?, ?)",
                (self._active_run, key, float(value), _now_iso(), int(step)),
            )

    def log_metrics(self, metrics: Mapping[str, float], step: int = 0):
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def set_tag(self, key: str, value: str):
        if not self._active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        with self._get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tags (run_id, key, value) VALUES (?, ?, ?)",
                (self._active_run, key, str(value)),
            )

    def set_tags(self, tags: Mapping[str, str]):
        for k, v in tags.items():
            self.set_tag(k, v)

    def log_artifact(self, local_path: Union[str, Path], artifact_path: str = ""):
        if not self._active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Artifact file not found: {local_path}")

        run_artifact_dir = self.artifacts_path / self._active_run / artifact_path
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        dest = run_artifact_dir / local_file.name
        shutil.copy2(local_file, dest)

        rel_path = str(Path(artifact_path) / local_file.name)
        with self._get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, path, artifact_type) VALUES (?, ?, ?)",
                (self._active_run, rel_path, (local_file.suffix or "file").lstrip(".")),
            )

    def log_bytes(self, data: bytes, filename: str, artifact_path: str = ""):
        if not self._active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        if len(data) > _BIN_LIMIT:
            raise ValueError(
                f"Refusing to store binary > {_BIN_LIMIT} bytes via decorator helper"
            )
        run_artifact_dir = self.artifacts_path / self._active_run / artifact_path
        run_artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = run_artifact_dir / filename
        dest.write_bytes(data)
        with self._get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, path, artifact_type) VALUES (?, ?, ?)",
                (
                    self._active_run,
                    str(Path(artifact_path) / filename),
                    Path(filename).suffix.lstrip("."),
                ),
            )

    def log_figure(
        self,
        fig,
        filename: str = "figure.png",
        artifact_path: str = "figures",
        dpi: int = 120,
    ):
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format=Path(filename).suffix.lstrip(".") or "png",
            dpi=dpi,
            bbox_inches="tight",
        )
        self.log_bytes(buf.getvalue(), filename=filename, artifact_path=artifact_path)

    def log_model(self, model: Any, model_name: str = "model"):
        if not self._active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        model_dir = self.artifacts_path / self._active_run / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{model_name}.pkl"
        model_pickle_ok = True
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception:
            model_pickle_ok = False

        arch_txt_path = model_dir / f"{model_name}_architecture.txt"
        arch_json_path = model_dir / f"{model_name}_architecture.json"

        summary = _model_summary(model)
        try:
            model_text = str(model)
        except Exception:
            model_text = f"<{type(model).__name__}>"

        with open(arch_txt_path, "w", encoding="utf-8") as f:
            f.write(model_text)
            f.write("\n\n")
            f.write(f"Total params: {summary['total']:,}\n")
            f.write(f"Trainable params: {summary['trainable']:,}\n")
            f.write(f"Non-trainable params: {summary['non_trainable']:,}\n")

        try:
            arch_payload = {
                "summary": summary,
                "tree": _module_tree(model, name=model.__class__.__name__),
            }
        except Exception:
            arch_payload = {
                "summary": summary,
                "tree": {
                    "name": model.__class__.__name__,
                    "type": model.__class__.__name__,
                    "num_params": summary["total"],
                    "trainable_params": summary["trainable"],
                    "children": [],
                },
            }

        with open(arch_json_path, "w", encoding="utf-8") as f:
            json.dump(arch_payload, f, indent=2)

        with self._get_db() as conn:
            if model_pickle_ok:
                conn.execute(
                    "INSERT OR REPLACE INTO artifacts (run_id, path, artifact_type) VALUES (?, ?, ?)",
                    (self._active_run, f"models/{model_name}.pkl", "model"),
                )
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, path, artifact_type) VALUES (?, ?, ?)",
                (
                    self._active_run,
                    f"models/{model_name}_architecture.txt",
                    "architecture",
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, path, artifact_type) VALUES (?, ?, ?)",
                (
                    self._active_run,
                    f"models/{model_name}_architecture.json",
                    "architecture-json",
                ),
            )

    def load_model(self, run_id: str, model_name: str = "model"):
        model_path = self.artifacts_path / run_id / "models" / f"{model_name}.pkl"
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # ---------- Queries ----------
    def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._get_db() as conn:
            run = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not run:
                raise ValueError(f"Run {run_id} not found")

            params = {
                row["key"]: row["value"]
                for row in conn.execute(
                    "SELECT key, value FROM params WHERE run_id = ?", (run_id,)
                )
            }
            # latest value per metric
            metrics: Dict[str, float] = {}
            for row in conn.execute(
                "SELECT key, value, step FROM metrics WHERE run_id = ? ORDER BY key, step DESC",
                (run_id,),
            ):
                if row["key"] not in metrics:
                    metrics[row["key"]] = float(row["value"])

            tags = {
                row["key"]: row["value"]
                for row in conn.execute(
                    "SELECT key, value FROM tags WHERE run_id = ?", (run_id,)
                )
            }

            return {
                "run_id": run["run_id"],
                "experiment_id": run["experiment_id"],
                "name": run["name"],
                "status": run["status"],
                "start_time": run["start_time"],
                "end_time": run["end_time"],
                "params": params,
                "metrics": metrics,
                "tags": tags,
            }

    def delete_run(self, run_id: str) -> None:
        """Permanently remove a run and all its associated data from the DB."""
        with self._get_db() as conn:
            row = conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Run {run_id} not found")
            for table in ("params", "metrics", "tags", "artifacts"):
                conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def search_runs(self, experiment_name: Optional[str] = None) -> List[Dict]:
        with self._get_db() as conn:
            if experiment_name:
                exp_id = self.get_experiment(experiment_name)
                if exp_id is None:
                    return []
                rows = conn.execute(
                    "SELECT run_id FROM runs WHERE experiment_id = ? ORDER BY start_time DESC",
                    (exp_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id FROM runs ORDER BY start_time DESC"
                ).fetchall()
            return [self.get_run(r["run_id"]) for r in rows]

    # ---------- Context manager ----------
    @contextmanager
    def run(self, experiment_name: str = "default", run_name: Optional[str] = None):
        run_id = self.start_run(experiment_name, run_name)
        try:
            yield run_id
            self.end_run("FINISHED")
        except Exception:
            self.end_run("FAILED")
            raise

    # ---------- Autolog decorator ----------
    def autolog(
        self,
        experiment: str = "default",
        run_name: Optional[str] = "{func}__{timestamp}",
        log_args: bool = True,
        ignore: Iterable[str] = (
            "self",
            "cls",
            "data",
            "dataset",
            "loader",
            "X",
            "Y",
        ),  # skip heavy objects
        capture_system: bool = True,
        capture_git: bool = True,
        log_return: Union[None, str, Literal["metrics", "artifact"]] = None,
        return_artifact_name: str = "return.pkl",
        inject_param: str = "_mlt",
    ):
        """
        Decorate a training function to auto-track. Example:

        @tracker.autolog(experiment="my_exp", run_name="{func}_{timestamp}", log_return="metrics")
        def train(config, _mlt=None):
            _mlt.metric("train_loss", 0.123, step=0)
            return {"val_loss": 0.11, "acc": 0.9}

        - run_name can use {func}, {timestamp}, {hash} (hash of jsonified args)
        - if parameter named `inject_param` exists, we inject a helper with .metric/.params/.artifact/.figure/.model
        - log_return:
            * "metrics": if function returns Mapping[str, float], log as metrics
            * "artifact": pickle the return into artifacts/return.pkl
            * None: do nothing special with return
        """

        def decorator(fn):
            sig = inspect.signature(fn)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # Build param snapshot BEFORE run (for naming/hashing)
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()

                def filtered_items():
                    for k, v in bound.arguments.items():
                        if k in ignore:
                            continue
                        yield k, _try_jsonable(v)

                params_snapshot = dict(filtered_items()) if log_args else {}
                name_hash = _hashdict(params_snapshot) if params_snapshot else "nohash"
                resolved_name = None
                if run_name:
                    resolved_name = str(run_name).format(
                        func=fn.__name__,
                        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
                        hash=name_hash,
                    )

                run_id = self.start_run(experiment, resolved_name)

                # baseline tags
                if capture_system:
                    self.set_tags({f"sys:{k}": v for k, v in _sys_info().items()})
                if capture_git:
                    self.set_tags({f"git:{k}": v for k, v in _maybe_git_info().items()})

                # log function params snapshot
                if log_args and params_snapshot:
                    # log both as params (stringified) and a single artifact with full JSON
                    self.log_params(
                        {f"arg:{k}": _safe_repr(v) for k, v in params_snapshot.items()}
                    )
                    try:
                        byts = json.dumps(
                            params_snapshot, default=str, indent=2
                        ).encode()
                        if len(byts) <= _BIN_LIMIT:
                            self.log_bytes(
                                byts, filename="call_args.json", artifact_path="inputs"
                            )
                    except Exception:
                        pass

                # small helper injected into function (if user accepts it)
                class _Helper:
                    def metric(_, key: str, value: float, step: int = 0):
                        self.log_metric(key, float(value), step)

                    def metrics(_, mdict: Mapping[str, float], step: int = 0):
                        self.log_metrics(mdict, step)

                    def params(_, pdict: Mapping[str, Any]):
                        self.log_params(pdict)

                    def tag(_, key: str, value: str):
                        self.set_tag(key, value)

                    def tags(_, tdict: Mapping[str, str]):
                        self.set_tags(tdict)

                    def artifact(_, path: Union[str, Path], artifact_path: str = ""):
                        self.log_artifact(path, artifact_path)

                    def bytes(_, data: bytes, filename: str, artifact_path: str = ""):
                        self.log_bytes(data, filename, artifact_path)

                    def figure(
                        _,
                        fig,
                        filename: str = "figure.png",
                        artifact_path: str = "figures",
                        dpi: int = 120,
                    ):
                        self.log_figure(fig, filename, artifact_path, dpi)

                    def model(_, model: Any, model_name: str = "model"):
                        self.log_model(model, model_name)

                    @property
                    def run_id(_):
                        return self._active_run

                injected = {}
                if inject_param in sig.parameters:
                    injected[inject_param] = _Helper()

                t0 = time.time()
                try:
                    result = fn(*args, **{**kwargs, **injected})
                    duration = time.time() - t0
                    self.log_metric("duration_seconds", duration, step=0)

                    # post-process return
                    if log_return == "metrics" and isinstance(result, Mapping):
                        # only floats/ints become metrics
                        numeric = {
                            k: float(v)
                            for k, v in result.items()
                            if isinstance(v, (int, float))
                        }
                        if numeric:
                            self.log_metrics(numeric, step=0)
                    elif log_return == "artifact":
                        try:
                            blob = pickle.dumps(result)
                            if len(blob) <= _BIN_LIMIT:
                                self.log_bytes(
                                    blob,
                                    filename=return_artifact_name,
                                    artifact_path="return",
                                )
                        except Exception:
                            pass

                    self.end_run("FINISHED")
                    return result
                except Exception:
                    self.end_run("FAILED")
                    raise

            return wrapper

        return decorator


# -------------------------- Example usage --------------------------

if __name__ == "__main__":
    tracker = MLTracker("./my_experiments")

    # Traditional context manager still works
    with tracker.run("classification_experiment", "random_forest_v1"):
        tracker.log_params(
            {
                "model_type": "random_forest",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            }
        )
        for epoch in range(5):
            tracker.log_metrics(
                {
                    "train_loss": 0.5 - epoch * 0.08,
                    "val_loss": 0.6 - epoch * 0.07,
                    "accuracy": 0.7 + epoch * 0.05,
                },
                step=epoch,
            )
        tracker.set_tags({"team": "data-science", "version": "v1.0"})
        tracker.log_model({"type": "random_forest", "params": {}}, "rf_model")

    # Autolog example: we inject `_mlt` and return metrics to auto-log
    @tracker.autolog(
        experiment="autolog_demo",
        run_name="{func}__{hash}__{timestamp}",
        log_args=True,
        log_return="metrics",
    )
    def train_epoch(config: Dict[str, Any], data=None, _mlt=None):
        # log inside the function
        _mlt.params({"optimizer": config.get("optimizer", "adam")})
        for step in range(3):
            _mlt.metric("loss", 0.9 - 0.2 * step, step=step)
        # metrics to be auto-logged from return value:
        return {
            "val_loss": 0.3,
            "acc": 0.88,
            "note": "only numeric keys are logged as metrics",
        }

    train_epoch({"lr": 3e-4, "optimizer": "adamW", "depth": 12})
    print("Runs in autolog_demo:", len(tracker.search_runs("autolog_demo")))
