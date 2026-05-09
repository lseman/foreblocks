import importlib


def compute_params(computer, model):
    """Parameter count using fvcore if available."""

    def _compute():
        try:
            fvcore_nn = importlib.import_module("fvcore.nn")
            parameter_count = getattr(fvcore_nn, "parameter_count")
            count_dict = parameter_count(model)
            return sum(v for v in count_dict.values())
        except Exception:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return computer._compute_safely(_compute)
