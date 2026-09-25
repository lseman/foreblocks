"""mltracker.client — HTTP client for the MLTracker experiment tracking API.

Provides a lightweight client for creating experiments, managing runs, logging
metrics and parameters, and storing artifacts. Designed for integration with
the MLTracker server and the MLTracker autologging decorator.

Core API:
- MLTrackerClient: HTTP client for MLTracker REST API
- start_run: context manager for safe run lifecycle management

Usage
-----
>>> from mltracker.client import MLTrackerClient, start_run
>>> client = MLTrackerClient("http://localhost:8000")
>>> with start_run(client, "my_experiment", "run_1"):
...     client.log_params({"lr": 0.001})
...     client.log_metric("accuracy", 0.95)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests


class MLTrackerClient:
    """HTTP client for the MLTracker experiment tracking API.

    Manages experiments, runs, parameters, metrics, tags, and artifacts
    through the MLTracker REST API. Uses a persistent :class:`requests.Session`
    for connection pooling and cookie handling.

    Parameters
    ----------
    tracking_uri : str
        Base URL of the MLTracker server. Default ``"http://localhost:8000"``.
    timeout : float
        Request timeout in seconds. Default 30.0.

    Examples
    --------
    >>> client = MLTrackerClient("http://localhost:8000", timeout=15.0)
    >>> runs = client.list_experiments()
    """

    def __init__(self, tracking_uri: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = tracking_uri.rstrip("/")
        self.timeout = timeout
        self._active_run_id: str | None = None
        self._session = requests.Session()

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make an HTTP request to the MLTracker API.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE).
        endpoint : str
            API endpoint path (e.g., ``"/api/runs"``).
        json : dict | None
            JSON request body.
        params : dict | None
            URL query parameters.
        files : dict | None
            Multipart form files for artifact uploads.

        Returns
        -------
        dict | list | None
            Parsed JSON response, or None if no content.

        Raises
        ------
        MLTrackerAPIError
            If the server returns a 4xx or 5xx status code.
        requests.exceptions.RequestException
            For network-level errors (connection refused, timeout, etc.).
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.request(
                method, url, json=json, params=params, files=files, timeout=self.timeout
            )
        except requests.exceptions.Timeout as e:
            raise MLTrackerTimeoutError(
                f"Request to {url} timed out after {self.timeout}s"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise MLTrackerConnectionError(
                f"Cannot connect to {self.base_url}. Is the server running?"
            ) from e

        if response.status_code >= 400:
            error_body = ""
            try:
                error_body = response.text[:500]
            except Exception:
                pass
            raise MLTrackerAPIError(
                f"API Error {response.status_code}: {error_body}"
            )

        return response.json() if response.content else None

    # ---- Experiments ----

    def create_experiment(self, name: str) -> dict[str, Any]:
        """Create a new experiment.

        Parameters
        ----------
        name : str
            Unique experiment name.

        Returns
        -------
        dict
            Experiment metadata including ``experiment_id`` and ``created_at``.

        Raises
        ------
        MLTrackerAPIError
            If an experiment with this name already exists.
        """
        return self._request("POST", "/api/experiments", json={"name": name})

    def get_experiment(self, name: str) -> dict[str, Any]:
        """Get experiment metadata by name.

        Parameters
        ----------
        name : str
            Experiment name.

        Returns
        -------
        dict
            Experiment metadata.

        Raises
        ------
        MLTrackerAPIError
            If the experiment does not exist.
        """
        return self._request("GET", f"/api/experiments/{name}")

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments.

        Returns
        -------
        list[dict]
            List of experiment metadata dicts.
        """
        return self._request("GET", "/api/experiments") or []

    # ---- Runs ----

    def start_run(
        self,
        experiment_name: str = "default",
        run_name: str | None = None,
    ) -> str:
        """Start a new tracking run.

        Parameters
        ----------
        experiment_name : str
            Target experiment name. Default ``"default"``.
        run_name : str | None
            Optional human-readable run name.

        Returns
        -------
        str
            The run ID (``run_id``).
        """
        response = self._request(
            "POST",
            "/api/runs",
            json={"experiment_name": experiment_name, "run_name": run_name},
        )
        self._active_run_id = response["run_id"]  # type: ignore[index]
        return str(self._active_run_id)

    def end_run(
        self,
        run_id: str | None = None,
        status: str = "FINISHED",
    ) -> None:
        """End a tracking run.

        Parameters
        ----------
        run_id : str | None
            Run ID to end. Defaults to the active run.
        status : str
            Final status: ``"FINISHED"``, ``"FAILED"``, or ``"SKIPPED"``.
        """
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run to end. Call start_run() first.")

        self._request("PUT", f"/api/runs/{rid}/end", json={"status": status})

        if rid == self._active_run_id:
            self._active_run_id = None

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Get run details.

        Parameters
        ----------
        run_id : str
            Run ID.

        Returns
        -------
        dict
            Full run metadata including params, metrics, and tags.
        """
        return self._request("GET", f"/api/runs/{run_id}")

    def search_runs(
        self,
        experiment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for runs, optionally filtered by experiment.

        Parameters
        ----------
        experiment_name : str | None
            Filter by experiment name. None returns all runs.

        Returns
        -------
        list[dict]
            List of run metadata dicts.
        """
        params = {"experiment_name": experiment_name} if experiment_name else {}
        result = self._request("GET", "/api/runs", params=params) or {}
        return result.get("runs", [])

    # ---- Logging ----

    def log_param(
        self,
        key: str,
        value: Any,
        run_id: str | None = None,
    ) -> None:
        """Log a single parameter.

        Parameters
        ----------
        key : str
            Parameter name.
        value : Any
            Parameter value (must be JSON-serializable).
        run_id : str | None
            Target run ID. Defaults to active run.

        Raises
        ------
        ValueError
            If there is no active run and no run_id provided.
        """
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")

        self._request(
            "POST", f"/api/runs/{rid}/params", json={"key": key, "value": value}
        )

    def log_params(
        self,
        params: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        """Log multiple parameters at once.

        Parameters
        ----------
        params : dict[str, Any]
            Mapping of parameter names to values.
        run_id : str | None
            Target run ID. Defaults to active run.
        """
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")

        self._request("POST", f"/api/runs/{rid}/params/batch", json=params)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int = 0,
        run_id: str | None = None,
    ) -> None:
        """Log a single metric.

        Parameters
        ----------
        key : str
            Metric name.
        value : float
            Metric value.
        step : int
            Training step or epoch. Default 0.
        run_id : str | None
            Target run ID. Defaults to active run.
        """
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")

        self._request(
            "POST",
            f"/api/runs/{rid}/metrics",
            json={"key": key, "value": float(value), "step": int(step)},
        )

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int = 0,
        run_id: str | None = None,
    ) -> None:
        """Log multiple metrics at once.

        Parameters
        ----------
        metrics : dict[str, float]
            Mapping of metric names to values.
        step : int
            Training step or epoch. Default 0.
        run_id : str | None
            Target run ID. Defaults to active run.
        """
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")

        self._request(
            "POST",
            f"/api/runs/{rid}/metrics/batch",
            json=metrics,
            params={"step": step},
        )

    def set_tag(
        self,
        key: str,
        value: str,
        run_id: str | None = None,
    ) -> None:
        """Set a tag on the current run.

        Parameters
        ----------
        key : str
            Tag name.
        value : str
            Tag value.
        run_id : str | None
            Target run ID. Defaults to active run.
        """
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")

        self._request(
            "POST", f"/api/runs/{rid}/tags", json={"key": key, "value": value}
        )

    # ---- Artifacts ----

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str = "",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload a file as an artifact.

        Parameters
        ----------
        local_path : str or Path
            Path to the local file to upload.
        artifact_path : str
            Server-side path prefix for the artifact.
        run_id : str | None
            Target run ID. Defaults to active run.

        Returns
        -------
        dict
            Upload confirmation metadata.

        Raises
        ------
        FileNotFoundError
            If the local file does not exist.
        """
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")

        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Artifact file not found: {local_path}")

        with open(local_file, "rb") as f:
            files = {"file": (local_file.name, f)}
            params = {"artifact_path": artifact_path} if artifact_path else {}

            response = self._session.post(
                f"{self.base_url}/api/runs/{rid}/artifacts",
                files=files,
                params=params,
                timeout=self.timeout,
            )

        if response.status_code >= 400:
            raise MLTrackerAPIError(f"Upload failed: {response.text[:500]}")

        return response.json()

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        """List artifacts for a run.

        Parameters
        ----------
        run_id : str
            Run ID.

        Returns
        -------
        list[dict]
            List of artifact metadata dicts.
        """
        result = self._request("GET", f"/api/runs/{run_id}/artifacts") or {}
        return result.get("artifacts", [])

    def download_artifact(
        self,
        run_id: str,
        artifact_path: str,
        local_path: str | Path,
    ) -> Path:
        """Download an artifact to a local file.

        Parameters
        ----------
        run_id : str
            Run ID containing the artifact.
        artifact_path : str
            Server-side artifact path.
        local_path : str or Path
            Destination file path.

        Returns
        -------
        Path
            The destination path.
        """
        url = f"{self.base_url}/api/runs/{run_id}/artifacts/{artifact_path}"
        response = self._session.get(url, stream=True, timeout=self.timeout)

        if response.status_code >= 400:
            raise MLTrackerAPIError(f"Download failed: {response.text[:500]}")

        output_path = Path(local_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path

    # ---- Analysis ----

    def get_metric_history(
        self,
        run_id: str,
        metric_key: str | None = None,
    ) -> dict[str, Any]:
        """Get metric history for a run.

        Parameters
        ----------
        run_id : str
            Run ID.
        metric_key : str | None
            Filter to a specific metric. None returns all metrics.

        Returns
        -------
        dict
            Metric history keyed by metric name.
        """
        params = {"metric_key": metric_key} if metric_key else {}
        result = self._request(
            "GET", f"/api/runs/{run_id}/metrics/history", params=params
        ) or {}
        return result.get("metrics", {})

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Compare multiple runs side by side.

        Parameters
        ----------
        run_ids : list[str]
            Run IDs to compare (2-10 recommended).

        Returns
        -------
        dict
            Comparison including ``metric_keys``, ``param_keys``, and per-run values.
        """
        return self._request("POST", "/api/runs/compare", json=run_ids)

    def health_check(self) -> dict[str, Any]:
        """Check API server health.

        Returns
        -------
        dict
            Health status including ``status`` and ``version``.
        """
        return self._request("GET", "/api/health") or {}


# ---- Custom exceptions ----


class MLTrackerError(Exception):
    """Base exception for mltracker client errors."""


class MLTrackerAPIError(MLTrackerError):
    """Raised when the server returns an error response (4xx/5xx)."""


class MLTrackerTimeoutError(MLTrackerError):
    """Raised when a request times out."""


class MLTrackerConnectionError(MLTrackerError):
    """Raised when the client cannot connect to the server."""


# ---- Context manager ----


@contextmanager
def start_run(
    client: MLTrackerClient,
    experiment_name: str = "default",
    run_name: str | None = None,
):
    """Context manager for safe run lifecycle management.

    Automatically starts a run on entry and ends it with ``"FINISHED"``
    status on normal exit, or ``"FAILED"`` on exception.

    Parameters
    ----------
    client : MLTrackerClient
        The API client instance.
    experiment_name : str
        Target experiment name. Default ``"default"``.
    run_name : str | None
        Optional run name.

    Yields
    ------
    str
        The run ID.

    Examples
    --------
    >>> client = MLTrackerClient("http://localhost:8000")
    >>> with start_run(client, "my_experiment", "run_1"):
    ...     client.log_metric("accuracy", 0.95)
    """
    run_id = client.start_run(experiment_name, run_name)
    try:
        yield run_id
        client.end_run(run_id, "FINISHED")
    except Exception as e:
        client.end_run(run_id, "FAILED")
        raise
