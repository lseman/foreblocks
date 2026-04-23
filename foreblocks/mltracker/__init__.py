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
