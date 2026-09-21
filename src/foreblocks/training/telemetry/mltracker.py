"""MLTracker telemetry integration for training.

MLTracker and MoE logging helpers extracted from the Trainer.

All functions are static or instance helpers that log training parameters,
per-epoch metrics, model architecture metadata, and end-of-training summaries
to the MLTracker backend. Also provides last-run routing for logging eval
metrics back to the training run without reopening it.

Core API:
- init_mltracker_run_context: create an MLTracker run context (nullcontext fallback)
- get_mltracker_params: extract all parameters from a config object or dict
- log_mltracker_metrics: log per-epoch metrics to the active MLTracker run
- log_mltracker_model_info: log model architecture metadata as params and system/git tags
- log_model_to_last_run: log model artifacts to the most recent training run
- last_run_context: context manager for routing MLTracker calls to a previous run

"""

from __future__ import annotations

import contextlib
import datetime
import tempfile
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass  # type: ignore


# ---------------------------------------------------------------------------
# MLTracker helpers
# ---------------------------------------------------------------------------


def init_mltracker_run_context(
    mltracker: Any, run_name: str | None
) -> tuple[Any, str | None]:
    if not mltracker:
        return contextlib.nullcontext(), run_name

    try:
        exp_name = "default_experiment"
        if run_name is None:
            run_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_context = mltracker.run(experiment_name=exp_name, run_name=run_name)
    except Exception as e:
        print(f"[MLTracker] Warning: Failed to initialize run context: {e}")
        run_context = contextlib.nullcontext()

    return run_context, run_name


def get_mltracker_params(config: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if hasattr(config, "__dict__"):
        params.update(config.__dict__)
    elif isinstance(config, dict):
        params.update(config)
    return params


def build_mltracker_metrics(
    train_loss: float,
    lr: float,
    components: dict[str, float],
    val_loss: float | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {"train_loss": train_loss, "lr": lr}
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    for k, v in components.items():
        metrics[f"comp/{k}"] = float(v)
    return metrics


# ---------------------------------------------------------------------------
# Instance logging (must be called from Trainer)
# ---------------------------------------------------------------------------


def log_mltracker_params(mltracker: Any, config: Any) -> None:
    if not mltracker:
        return
    try:
        mltracker.log_params(get_mltracker_params(config))
    except Exception as e:
        print(f"[MLTracker] Warning: Failed to log params: {e}")


def log_mltracker_metrics(
    mltracker: Any,
    epoch: int,
    train_loss: float,
    lr: float,
    components: dict[str, float],
    val_loss: float | None,
) -> None:
    if not mltracker:
        return
    try:
        metrics = build_mltracker_metrics(train_loss, lr, components, val_loss)
        mltracker.log_metrics(metrics, step=epoch)
    except Exception as e:
        print(f"[MLTracker] Warning: Failed to log metrics: {e}")


def log_mltracker_model_info(
    mltracker: Any, model: torch.nn.Module, device: torch.device
) -> None:
    if not mltracker:
        return
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mltracker.log_params(
            {
                "model/class": type(model).__name__,
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/device": str(device),
            }
        )
    except Exception as e:
        print(f"[MLTracker] Warning: Failed to log model info: {e}")

    # System + git tags (best-effort)
    try:
        from mltracker.mltracker import _maybe_git_info, _sys_info

        mltracker.set_tags({f"sys:{k}": v for k, v in _sys_info().items()})
        git = _maybe_git_info()
        if git:
            mltracker.set_tags({f"git:{k}": v for k, v in git.items()})
    except Exception:
        pass


def log_mltracker_final(
    mltracker: Any,
    total_epochs: int,
    stopped_early: bool,
    best_val_loss: float,
) -> None:
    if not mltracker:
        return
    try:
        summary: dict[str, Any] = {"epochs_completed": total_epochs}
        if best_val_loss < float("inf"):
            summary["best_val_loss"] = best_val_loss
        mltracker.log_metrics(summary, step=total_epochs)
        mltracker.set_tags(
            {
                "trainer/early_stopped": str(stopped_early),
                "trainer/device": "cuda",  # Could be passed as param
                "trainer/amp": "true",
            }
        )
    except Exception as e:
        print(f"[MLTracker] Warning: Failed to log final summary: {e}")


# ---------------------------------------------------------------------------
# Last-run routing (for eval metrics logged back to the training run)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def last_run_context(mltracker: Any, last_run_id: Any):
    if not mltracker or not last_run_id:
        yield False
        return
    previous = mltracker._active_run
    try:
        mltracker._active_run = last_run_id
        yield True
    finally:
        mltracker._active_run = previous


def log_to_last_run(
    mltracker: Any,
    last_run_id: Any,
    metrics: dict[str, float],
    step: int | None,
    prefix: str,
) -> None:
    try:
        with last_run_context(mltracker, last_run_id) as active:
            if not active:
                return
            prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
            mltracker.log_metrics(prefixed, step=step if step is not None else 0)
    except Exception as e:
        print(f"[MLTracker] Warning: Failed to log eval metrics: {e}")


def log_model_to_last_run(
    mltracker: Any,
    last_run_id: Any,
    model: torch.nn.Module,
    model_name: str = "model",
) -> None:
    try:
        with last_run_context(mltracker, last_run_id) as active:
            if not active:
                return
            mltracker.log_model(model, model_name=model_name)
    except Exception as e:
        print(f"[MLTracker] Warning: Failed to log model artifacts: {e}")


def log_figure_to_last_run(
    mltracker: Any,
    last_run_id: Any,
    fig: Any,
    artifact_path: str = "plots",
) -> None:
    try:
        with last_run_context(mltracker, last_run_id) as active:
            if not active:
                return
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
                mltracker.log_artifact(tmp_path, artifact_path=artifact_path)
            finally:
                with contextlib.suppress(OSError):
                    import os

                    os.unlink(tmp_path)
    except Exception:
        pass
