# mltracker_client.py
from __future__ import annotations

import io
import json
import math
import os
import pickle
import tempfile
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import requests


class MLTrackerAPI:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.timeout = timeout

    # ---- runs ----
    def start_run(self, experiment_name: str = "default", run_name: Optional[str] = None) -> str:
        r = self.s.post(f"{self.base}/api/runs",
                        json={"experiment_name": experiment_name, "run_name": run_name},
                        timeout=self.timeout)
        r.raise_for_status()
        return r.json()["run_id"]

    def end_run(self, run_id: str, status: str = "FINISHED"):
        r = self.s.put(f"{self.base}/api/runs/{run_id}/end", json={"status": status}, timeout=self.timeout)
        r.raise_for_status()

    # ---- logging ----
    @staticmethod
    def _is_422(resp: requests.Response) -> bool:
        return resp.status_code == 422

    def log_param(self, run_id: str, key: str, value: Any):
        r = self.s.post(f"{self.base}/api/runs/{run_id}/params", json={"key": key, "value": value}, timeout=self.timeout)
        r.raise_for_status()

    def log_params(self, run_id: str, params: Mapping[str, Any]):
        payload_new = {"params": dict(params)}
        url = f"{self.base}/api/runs/{run_id}/params/batch"
        r = self.s.post(url, json=payload_new, timeout=self.timeout)
        if self._is_422(r):
            # old server expected raw dict body
            r = self.s.post(url, json=dict(params), timeout=self.timeout)
        r.raise_for_status()

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0):
        r = self.s.post(f"{self.base}/api/runs/{run_id}/metrics",
                        json={"key": key, "value": float(value), "step": int(step)},
                        timeout=self.timeout)
        r.raise_for_status()

    def log_metrics(self, run_id: str, metrics: Mapping[str, float], step: int = 0):
        url = f"{self.base}/api/runs/{run_id}/metrics/batch"
        payload_new = {"metrics": {k: float(v) for k, v in metrics.items()}, "step": int(step)}
        r = self.s.post(url, json=payload_new, timeout=self.timeout)
        if self._is_422(r):
            # old server expected raw dict + step query param
            r = self.s.post(f"{url}?step={step}",
                            json={k: float(v) for k, v in metrics.items()},
                            timeout=self.timeout)
        r.raise_for_status()

    def set_tag(self, run_id: str, key: str, value: str):
        r = self.s.post(f"{self.base}/api/runs/{run_id}/tags",
                        json={"key": key, "value": value}, timeout=self.timeout)
        r.raise_for_status()

    # ---- artifacts ----
    def upload_artifact(self, run_id: str, local_path: str | Path, artifact_path: str = ""):
        local_path = str(local_path)
        with open(local_path, "rb") as f:
            files = {"file": (Path(local_path).name, f, "application/octet-stream")}
            r = self.s.post(f"{self.base}/api/runs/{run_id}/artifacts",
                            params={"artifact_path": artifact_path},
                            files=files, timeout=self.timeout)
        r.raise_for_status()

    def upload_bytes(self, run_id: str, data: bytes, filename: str, artifact_path: str = ""):
        files = {
            "file": (filename, io.BytesIO(data), "application/octet-stream")
        }
        r = self.s.post(
            f"{self.base}/api/runs/{run_id}/artifacts",
            params={"artifact_path": artifact_path},
            files=files,
            timeout=self.timeout,
        )
        r.raise_for_status()

# -------- autolog decorator that injects _mlt helper --------

def autolog_api(api: MLTrackerAPI, *, experiment: str = "default",
                run_name: str | None = "{func}__{timestamp}",
                log_config: Mapping[str, Any] | None = None,
                log_return_as_metrics: bool = True):
    """
    Decorate a function so it automatically creates a run, logs config params,
    injects `_mlt` helper, and ends the run (FAILED on exception).
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            rn = None
            if run_name:
                rn = str(run_name).format(
                    func=fn.__name__,
                    timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
                )
            run_id = api.start_run(experiment, rn)

            class _Helper:
                def __init__(self, api: MLTrackerAPI, run_id: str):
                    self.api, self.run_id = api, run_id
                def metric(self, key, value, step: int = 0): self.api.log_metric(self.run_id, key, float(value), step)
                def metrics(self, d: Mapping[str, float], step: int = 0): self.api.log_metrics(self.run_id, d, step)
                def params(self, d: Mapping[str, Any]): self.api.log_params(self.run_id, d)
                def param(self, k: str, v: Any): self.api.log_param(self.run_id, k, v)
                def tag(self, k: str, v: str): self.api.set_tag(self.run_id, k, v)
                def artifact(self, path: str | Path, artifact_path: str = ""): self.api.upload_artifact(self.run_id, path, artifact_path)
                def model(self, obj: Any, name: str = "model.pkl"):
                    blob = pickle.dumps(obj)
                    self.api.upload_bytes(self.run_id, blob, name, artifact_path="models")

            _mlt = _Helper(api, run_id)

            if log_config:
                _mlt.params(log_config)

            try:
                ret = fn(*args, **{**kwargs, "_mlt": _mlt})
                if log_return_as_metrics and isinstance(ret, dict):
                    numeric = {k: float(v) for k, v in ret.items() if isinstance(v, (int, float))}
                    if numeric:
                        _mlt.metrics(numeric, step=0)
                api.end_run(run_id, "FINISHED")
                return ret
            except Exception:
                api.end_run(run_id, "FAILED")
                raise
        return wrapper
    return deco


# Optional: Trainer callback example
class APILogCallback:
    def __init__(self, _mlt, prefix: str = ""):
        self._mlt = _mlt
        self.prefix = prefix

    def on_epoch_end(self, trainer, epoch: int, logs: dict):
        # keep only finite numerics
        finite = {}
        for k, v in (logs or {}).items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                finite[f"{self.prefix}{k}"] = float(v)
        if finite:
            self._mlt.metrics(finite, step=epoch)
