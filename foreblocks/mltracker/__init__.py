"""foreblocks.mltracker.

Experiment tracking package with file-based and API-backed modes.

Provides lightweight ML experiment tracking for training loops and hyperparameter
studies. Supports both local SQLite-based tracking and remote API synchronization.

Core API:
- MLTracker: local file-based experiment tracker
- MLTrackerAPI: remote API client
- autolog_api: remote autologging decorator
- create_tui_app: TUI dashboard for experiment inspection

"""

from importlib import import_module

__all__ = ["MLTracker", "MLTrackerAPI", "autolog_api", "create_tui_app"]


def __getattr__(name):
    lazy_exports = {
        "MLTracker": (".mltracker", "MLTracker"),
        "MLTrackerAPI": (".mltracker_client", "MLTrackerAPI"),
        "autolog_api": (".mltracker_client", "autolog_api"),
        "create_tui_app": (".mltracker_tui", "create_app"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
