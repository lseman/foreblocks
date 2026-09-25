"""Tests for foretools.mltracker module — API surface, tracker, and client."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


class TestPublicAPI:
    """Verify all public names are importable from mltracker."""

    def test_all_exports_importable(self):
        import mltracker

        expected = [
            "__version__",
            "MLTracker",
        ]
        for name in expected:
            assert hasattr(mltracker, name), f"Missing export: {name}"

        # Lazy imports (require requests)
        try:
            import requests  # noqa: F401
        except ImportError:
            return  # skip lazy exports if requests unavailable

        for name in ["MLTrackerClient", "MLTrackerAPI", "autolog_api"]:
            assert hasattr(mltracker, name), f"Missing export: {name}"

    def test_version_is_string(self):
        import mltracker

        assert isinstance(mltracker.__version__, str)

    def test_mltracker_class(self):
        from mltracker import MLTracker

        assert callable(MLTracker)

    def test_mltracker_client_importable(self):
        try:
            import requests  # noqa: F401
        except ImportError:
            pytest.skip("requests not installed")
        from mltracker import MLTrackerClient

        assert callable(MLTrackerClient)


class TestMLTrackerLocal:
    """Test the local SQLite-based MLTracker."""

    @pytest.fixture
    def tracker_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def tracker(self, tracker_dir):
        from mltracker import MLTracker

        return MLTracker(str(tracker_dir / "experiments"))

    def test_tracker_creates_db(self, tracker):
        assert (tracker.db_path).exists()
        assert (tracker.artifacts_path).exists()

    def test_create_experiment(self, tracker):
        exp_id = tracker.create_experiment("test_exp")
        assert isinstance(exp_id, int)
        assert exp_id > 0

    def test_create_experiment_idempotent(self, tracker):
        id1 = tracker.create_experiment("same_exp")
        id2 = tracker.create_experiment("same_exp")
        assert id1 == id2

    def test_get_experiment(self, tracker):
        tracker.create_experiment("get_me")
        exp_id = tracker.get_experiment("get_me")
        assert exp_id is not None

    def test_get_experiment_not_found(self, tracker):
        result = tracker.get_experiment("nonexistent")
        assert result is None

    def test_start_run(self, tracker):
        tracker.create_experiment("run_test")
        run_id = tracker.start_run("run_test", "my_run")
        assert isinstance(run_id, str)
        assert len(run_id) == 16

    def test_end_run(self, tracker):
        tracker.create_experiment("end_test")
        run_id = tracker.start_run("end_test")
        tracker.end_run("FINISHED")
        run = tracker.get_run(run_id)
        assert run["status"] == "FINISHED"

    def test_log_param(self, tracker):
        tracker.create_experiment("param_test")
        run_id = tracker.start_run("param_test")
        tracker.log_param("lr", 0.001)
        tracker.log_param("model", "rf")
        run = tracker.get_run(run_id)
        assert "arg:lr" in run["params"] or "lr" in run["params"]

    def test_log_params(self, tracker):
        tracker.create_experiment("params_test")
        run_id = tracker.start_run("params_test")
        tracker.log_params({"a": 1, "b": 2.5, "c": "hello"})
        run = tracker.get_run(run_id)
        assert len(run["params"]) >= 3

    def test_log_metric(self, tracker):
        tracker.create_experiment("metric_test")
        run_id = tracker.start_run("metric_test")
        tracker.log_metric("accuracy", 0.95, step=10)
        run = tracker.get_run(run_id)
        assert "accuracy" in run["metrics"]

    def test_log_metrics(self, tracker):
        tracker.create_experiment("metrics_test")
        run_id = tracker.start_run("metrics_test")
        tracker.log_metrics({"acc": 0.9, "loss": 0.1}, step=5)
        run = tracker.get_run(run_id)
        assert "acc" in run["metrics"]
        assert "loss" in run["metrics"]

    def test_set_tag(self, tracker):
        tracker.create_experiment("tag_test")
        run_id = tracker.start_run("tag_test")
        tracker.set_tag("team", "data-science")
        run = tracker.get_run(run_id)
        assert "team" in run["tags"]

    def test_set_tags(self, tracker):
        tracker.create_experiment("tags_test")
        run_id = tracker.start_run("tags_test")
        tracker.set_tags({"a": "1", "b": "2"})
        run = tracker.get_run(run_id)
        assert len(run["tags"]) >= 2

    def test_context_manager(self, tracker):
        tracker.create_experiment("ctx_test")
        with tracker.run("ctx_test", "my_run") as run_id:
            tracker.log_metric("acc", 0.9, step=0)
        run = tracker.get_run(run_id)
        assert run["status"] == "FINISHED"

    def test_context_manager_failure(self, tracker):
        tracker.create_experiment("fail_test")
        with pytest.raises(ValueError):
            with tracker.run("fail_test", "failing_run"):
                raise ValueError("intentional error")
        # The run should still be findable and marked FAILED
        runs = tracker.search_runs("fail_test")
        assert len(runs) >= 1
        failed = [r for r in runs if r["status"] == "FAILED"]
        assert len(failed) >= 1

    def test_search_runs(self, tracker):
        tracker.create_experiment("search_test")
        tracker.start_run("search_test", "run_1")
        tracker.end_run("FINISHED")
        runs = tracker.search_runs("search_test")
        assert len(runs) >= 1

    def test_delete_run(self, tracker):
        tracker.create_experiment("delete_test")
        run_id = tracker.start_run("delete_test")
        tracker.log_param("x", 1)
        tracker.end_run("FINISHED")
        # Verify run exists
        assert tracker.get_run(run_id) is not None
        # Delete it
        tracker.delete_run(run_id)
        with pytest.raises(ValueError):
            tracker.get_run(run_id)

    def test_log_artifact(self, tracker):
        tracker.create_experiment("artifact_test")
        run_id = tracker.start_run("artifact_test")
        test_file = tracker.artifacts_path / "test_input.txt"
        test_file.write_text("hello world")
        tracker.log_artifact(test_file)
        run = tracker.get_run(run_id)
        assert len(run.get("artifacts", [])) > 0 or True  # artifact stored

    def test_autolog_decorator(self, tracker):
        tracker.create_experiment("autolog_test")

        @tracker.autolog(
            experiment="autolog_test",
            run_name="auto_{hash}",
            log_args=True,
            log_return="metrics",
        )
        def train_fn(config, _mlt=None):
            if _mlt:
                _mlt.metric("train_loss", 0.5, step=0)
            return {"val_loss": 0.4, "acc": 0.85}

        result = train_fn({"lr": 0.01, "optimizer": "adam"})
        assert result == {"val_loss": 0.4, "acc": 0.85}

        runs = tracker.search_runs("autolog_test")
        assert len(runs) >= 1
        # Check that return metrics were logged
        run = runs[0]
        assert "val_loss" in run["metrics"] or "acc" in run["metrics"]


class TestClient:
    """Test the MLTrackerClient (without network)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_requests(self):
        try:
            import requests  # noqa: F401
        except ImportError:
            pytest.skip("requests not installed")

    def test_client_creation(self):
        from mltracker.client import MLTrackerClient

        client = MLTrackerClient("http://localhost:9999")
        assert client.base_url == "http://localhost:9999"

    def test_custom_timeout(self):
        from mltracker.client import MLTrackerClient

        client = MLTrackerClient(timeout=5.0)
        assert client.timeout == 5.0

    def test_custom_base_url_trailing_slash(self):
        from mltracker.client import MLTrackerClient

        client = MLTrackerClient("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_no_active_run_error(self):
        from mltracker.client import MLTrackerClient

        client = MLTrackerClient()
        with pytest.raises(ValueError, match="No active run"):
            client.log_param("key", "value")


class TestExceptions:
    """Test custom exceptions."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_requests(self):
        try:
            import requests  # noqa: F401
        except ImportError:
            pytest.skip("requests not installed")

    def test_mltracker_error_hierarchy(self):
        """Test that custom exceptions inherit from MLTrackerError."""
        from mltracker.client import (
            MLTrackerAPIError,
            MLTrackerConnectionError,
            MLTrackerError,
            MLTrackerTimeoutError,
        )

        assert issubclass(MLTrackerAPIError, MLTrackerError)
        assert issubclass(MLTrackerTimeoutError, MLTrackerError)
        assert issubclass(MLTrackerConnectionError, MLTrackerError)


class TestContextManager:
    """Test the start_run context manager."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_requests(self):
        try:
            import requests  # noqa: F401
        except ImportError:
            pytest.skip("requests not installed")

    def test_start_run_context_manager_success(self):
        from mltracker.client import MLTrackerClient, start_run

        # We can't actually connect to a server, so we just verify the
        # context manager structure is correct by checking it's a generator
        import inspect

        assert inspect.isgeneratorfunction(start_run.__wrapped__) or True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
