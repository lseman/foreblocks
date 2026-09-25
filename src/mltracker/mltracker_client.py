"""mltracker.mltracker_client.

HTTP client and API integration for the MLTracker experiment tracking system.

Provides MLTrackerAPI for server-side REST communication and the autolog_api
decorator for remote experiment tracking. Designed as a client-side complement
to the file-based MLTracker, enabling cloud-synced experiment management.

Core API:
- MLTrackerAPI: REST API client for MLTracker server
- autolog_api: decorator for remote experiment autologging


# mltracker_client.py

"""

from __future__ import annotations

import io
import math
import pickle
from collections.abc import Mapping
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import requests


class MLTrackerAPI:
    """REST API client for the MLTracker server.

    Handles HTTP communication with a remote MLTracker server. Includes
    backward-compatibility handling for old and new server payload formats.

    Parameters
    ----------
    base_url : str
        Server base URL (e.g., ``"http://localhost:8000"``).
    timeout : float
        Request timeout in seconds. Default 10.0.
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.timeout = timeout
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.timeout = timeout

    # ---- runs ----
    def start_run(
        self,
        experiment_name: str = "default",
        run_name: str | None = None,
    ) -> str:
        """Start a new tracking run on the server.

        Parameters
        ----------
        experiment_name : str
            Target experiment name. Default ``"default"``.
        run_name : str | None
            Optional run name.

        Returns
        -------
        str
            The run ID.
        """
        r = self.s.post(
            f"{self.base}/api/runs",
            json={"experiment_name": experiment_name, "run_name": run_name},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["run_id"]

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """End a tracking run on the server.

        Parameters
        ----------
        run_id : str
            Run ID to end.
        status : str
            Final status. Default ``"FINISHED"``.
        """
        r = self.s.put(
            f"{self.base}/api/runs/{run_id}/end",
            json={"status": status},
            timeout=self.timeout,
        )
        r.raise_for_status()

    # ---- logging ----
    @staticmethod
    def _is_422(resp: requests.Response) -> bool:
        """Check if response is 422 (payload format mismatch)."""
        return resp.status_code == 422

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        """Log a single parameter on the server.

        Parameters
        ----------
        run_id : str
            Run ID.
        key : str
            Parameter name.
        value : Any
            Parameter value.
        """
        r = self.s.post(
            f"{self.base}/api/runs/{run_id}/params",
            json={"key": key, "value": value},
            timeout=self.timeout,
        )
        r.raise_for_status()

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        """Log multiple parameters with backward-compatible payload format.

        Tries the new payload format first (``{"params": {...}}``), falls back
        to raw dict if the server returns 422.

        Parameters
        ----------
        run_id : str
            Run ID.
        params : dict[str, Any]
            Parameter mapping.
        """
        payload_new = {"params": dict(params)}
        url = f"{self.base}/api/runs/{run_id}/params/batch"
        r = self.s.post(url, json=payload_new, timeout=self.timeout)
        if self._is_422(r):
            # old server expected raw dict body
            r = self.s.post(url, json=dict(params), timeout=self.timeout)
        r.raise_for_status()

    def log_metric(
        self, run_id: str, key: str, value: float, step: int = 0
    ) -> None:
        """Log a single metric on the server.

        Parameters
        ----------
        run_id : str
            Run ID.
        key : str
            Metric name.
        value : float
            Metric value.
        step : int
            Training step. Default 0.
        """
        r = self.s.post(
            f"{self.base}/api/runs/{run_id}/metrics",
            json={"key": key, "value": float(value), "step": int(step)},
            timeout=self.timeout,
        )
        r.raise_for_status()

    def log_metrics(
        self, run_id: str, metrics: Mapping[str, float], step: int = 0
    ) -> None:
        """Log multiple metrics with backward-compatible payload format.

        Tries the new payload format first (``{"metrics": {...}, "step": N}``),
        falls back to raw dict + query param if server returns 422.

        Parameters
        ----------
        run_id : str
            Run ID.
        metrics : dict[str, float]
            Metric mapping.
        step : int
            Training step. Default 0.
        """
        url = f"{self.base}/api/runs/{run_id}/metrics/batch"
        payload_new = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "step": int(step),
        }
        r = self.s.post(url, json=payload_new, timeout=self.timeout)
        if self._is_422(r):
            # old server expected raw dict + step query param
            r = self.s.post(
                f"{url}?step={step}",
                json={k: float(v) for k, v in metrics.items()},
                timeout=self.timeout,
            )
        r.raise_for_status()

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a tag on the server.

        Parameters
        ----------
        run_id : str
            Run ID.
        key : str
            Tag name.
        value : str
            Tag value.
        """
        r = self.s.post(
            f"{self.base}/api/runs/{run_id}/tags",
            json={"key": key, "value": value},
            timeout=self.timeout,
        )
        r.raise_for_status()

    # ---- artifacts ----
    def upload_artifact(
        self, run_id: str, local_path: str | Path, artifact_path: str = ""
    ) -> None:
        """Upload a file as an artifact.

        Parameters
        ----------
        run_id : str
            Run ID.
        local_path : str or Path
            Local file path.
        artifact_path : str
            Server-side path prefix.
        """
        local_path = str(local_path)
        with open(local_path, "rb") as f:
            files = {"file": (Path(local_path).name, f, "application/octet-stream")}
            r = self.s.post(
                f"{self.base}/api/runs/{run_id}/artifacts",
                params={"artifact_path": artifact_path},
                files=files,
                timeout=self.timeout,
            )
        r.raise_for_status()

    def upload_bytes(
        self, run_id: str, data: bytes, filename: str, artifact_path: str = ""
    ) -> None:
        """Upload raw bytes as an artifact.

        Parameters
        ----------
        run_id : str
            Run ID.
        data : bytes
            Raw byte data.
        filename : str
            Name for the stored file.
        artifact_path : str
            Server-side path prefix.
        """
        files = {"file": (filename, io.BytesIO(data), "application/octet-stream")}
        r = self.s.post(
            f"{self.base}/api/runs/{run_id}/artifacts",
            params={"artifact_path": artifact_path},
            files=files,
            timeout=self.timeout,
        )
        r.raise_for_status()


# -------- autolog decorator that injects _mlt helper --------


def autolog_api(
    api: MLTrackerAPI,
    *,
    experiment: str = "default",
    run_name: str | None = "{func}__{timestamp}",
    log_config: Mapping[str, Any] | None = None,
    log_return_as_metrics: bool = True,
):
    """Decorate a function for remote autologging.

    Automatically creates a run, logs config params, injects ``_mlt`` helper
    with metric/param/tag/artifact/model methods, and ends the run
    (``"FAILED"`` on exception).

    Parameters
    ----------
    api : MLTrackerAPI
        The remote API client instance.
    experiment : str
        Target experiment name. Default ``"default"``.
    run_name : str | None
        Run name template. Supports ``{func}`` and ``{timestamp}``.
            Default ``"{func}__{timestamp}"``. None means auto-generated.
    log_config : dict | None
        Extra parameters to log as config before the function runs.
    log_return_as_metrics : bool
        If True, numeric values in the return dict are logged as metrics.
            Default True.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> api = MLTrackerAPI("http://localhost:8000")
    >>> @autolog_api(api, experiment="my_exp", log_return_as_metrics=True)
    ... def train(config, _mlt=None):
    ...     _mlt.metric("loss", 0.5, step=0)
    ...     return {"val_loss": 0.4, "accuracy": 0.9}
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

                def metric(self, key, value, step: int = 0):
                    self.api.log_metric(self.run_id, key, float(value), step)

                def metrics(self, d: Mapping[str, float], step: int = 0):
                    self.api.log_metrics(self.run_id, d, step)

                def params(self, d: Mapping[str, Any]):
                    self.api.log_params(self.run_id, d)

                def param(self, k: str, v: Any):
                    self.api.log_param(self.run_id, k, v)

                def tag(self, k: str, v: str):
                    self.api.set_tag(self.run_id, k, v)

                def artifact(self, path: str | Path, artifact_path: str = ""):
                    self.api.upload_artifact(self.run_id, path, artifact_path)

                def model(self, obj: Any, name: str = "model.pkl"):
                    blob = pickle.dumps(obj)
                    self.api.upload_bytes(
                        self.run_id, blob, name, artifact_path="models"
                    )

            _mlt = _Helper(api, run_id)

            if log_config:
                _mlt.params(log_config)

            try:
                ret = fn(*args, **{**kwargs, "_mlt": _mlt})
                if log_return_as_metrics and isinstance(ret, dict):
                    numeric = {
                        k: float(v)
                        for k, v in ret.items()
                        if isinstance(v, (int, float))
                    }
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
    """Callback for logging training metrics to MLTracker during training loops.

    Filters logs to only finite numeric values and prefixes metric names.

    Parameters
    ----------
    _mlt : object
        The injected ``_mlt`` helper with a ``metrics()`` method.
    prefix : str
            Prefix for all logged metric names. Default empty string.
    """

    def __init__(self, _mlt: Any, prefix: str = ""):
        self._mlt = _mlt
        self.prefix = prefix

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        """Callback invoked at the end of each training epoch.

        Parameters
        ----------
        trainer : Any
            The training object (unused).
        epoch : int
            Current epoch number.
        logs : dict[str, Any]
            Metric dictionary from the training loop.
        """
        # keep only finite numerics
        finite = {}
        for k, v in (logs or {}).items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                finite[f"{self.prefix}{k}"] = float(v)
        if finite:
            self._mlt.metrics(finite, step=epoch)
