from importlib import import_module

__all__ = ["ModelEvaluator"]


def __getattr__(name):
    if name == "ModelEvaluator":
        module = import_module(".model_evaluator", __name__)
        return module.ModelEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
