"""mltracker — Lightweight ML Experiment Tracking System.

Provides lightweight ML experiment tracking for training loops and hyperparameter
studies. Supports both local SQLite-based tracking and remote API synchronization.

Architecture
------------
The package has four main components:

1. **MLTracker** (local) — File-based experiment tracker with SQLite backend,
   artifact storage, git integration, and an ``@autolog`` decorator for automatic
   function-level tracking.

2. **MLTrackerClient** (remote) — HTTP client for the MLTracker server REST API,
   enabling cloud-synced experiment management.

3. **MLTrackerAPI** (remote) — Server-side API client with backward-compatible
   payload formats for legacy servers.

4. **TUI** — Interactive terminal dashboard (Textual-based) for browsing
   experiments, runs, metrics sparklines, and artifacts.

Quick Start
-----------
>>> from mltracker import MLTracker
>>> tracker = MLTracker("./my_experiments")
>>> with tracker.run("classification", "rf_baseline"):
...     tracker.log_params({"n_estimators": 100, "max_depth": 10})
...     for epoch in range(5):
...         tracker.log_metrics({"train_loss": 0.5 - epoch * 0.1}, step=epoch)

For remote tracking:

>>> from mltracker import MLTrackerAPI
>>> api = MLTrackerAPI("http://localhost:8000")
>>> run_id = api.start_run("my_experiment", "run_1")

Version
-------
"""

from __future__ import annotations

__version__ = "0.2.0"

# Lazy imports to avoid heavy dependencies until used
from importlib import import_module

__all__ = [
    "__version__",
    # Local tracker
    "MLTracker",
    # Remote client
    "MLTrackerClient",
    "MLTrackerAPI",
    "autolog_api",
    # TUI
    "create_tui_app",
]


def __getattr__(name: str):
    """Lazy-load module symbols to avoid importing heavy dependencies."""
    lazy_exports = {
        "MLTracker": (".mltracker", "MLTracker"),
        "MLTrackerClient": (".client", "MLTrackerClient"),
        "MLTrackerAPI": (".mltracker_client", "MLTrackerAPI"),
        "autolog_api": (".mltracker_client", "autolog_api"),
        "create_tui_app": (".mltracker_tui", "create_app"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available exports including lazy ones."""
    return list(__all__)
