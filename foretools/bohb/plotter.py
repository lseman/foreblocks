from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover - optional dependency
    plt = None


@dataclass
class PlotStyle:
    dpi: int = 150
    figsize: Tuple[float, float] = (6.4, 4.0)
    grid: bool = True
    best_color: str = "#1f77b4"
    scatter_color: str = "#444444"
    accent_color: str = "#e15759"


class OptimizationPlotter:
    """
    Lightweight plotting utilities for BOHB/TPE runs.
    Designed for paper-ready figures with minimal dependencies.
    """

    def __init__(
        self,
        history: List[Dict[str, Any]],
        config_space: Optional[Dict[str, Tuple]] = None,
        style: Optional[PlotStyle] = None,
    ) -> None:
        self.history = list(history)
        self.config_space = config_space or {}
        self.style = style or PlotStyle()

    @classmethod
    def from_bohb(cls, bohb: Any, style: Optional[PlotStyle] = None) -> "OptimizationPlotter":
        return cls(bohb.get_optimization_history(), bohb.config_space, style=style)

    # ------------------------------------------------------------------
    # Core plots
    # ------------------------------------------------------------------

    def plot_optimization_history(
        self,
        ax: Optional[Any] = None,
        show_best: bool = True,
        title: str = "Optimization History",
        save_path: Optional[str] = None,
    ) -> Any:
        self._ensure_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        losses = [h.get("loss") for h in self.history if _is_finite(h.get("loss"))]
        xs = list(range(1, len(losses) + 1))
        ax.plot(xs, losses, color=self.style.scatter_color, linewidth=1.2, alpha=0.7)

        if show_best and losses:
            best = np.minimum.accumulate(np.asarray(losses, dtype=float))
            ax.plot(xs, best, color=self.style.best_color, linewidth=2.0, label="best so far")
            ax.legend(loc="best")

        ax.set_title(title)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Loss")
        if self.style.grid:
            ax.grid(True, alpha=0.25)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return ax

    def plot_budget_vs_loss(
        self,
        ax: Optional[Any] = None,
        title: str = "Budget vs Loss",
        log_budget: bool = True,
        save_path: Optional[str] = None,
    ) -> Any:
        self._ensure_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        budgets = [h.get("budget") for h in self.history if _is_finite(h.get("loss"))]
        losses = [h.get("loss") for h in self.history if _is_finite(h.get("loss"))]
        ax.scatter(budgets, losses, s=16, color=self.style.scatter_color, alpha=0.6)

        if log_budget:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Budget")
        ax.set_ylabel("Loss")
        if self.style.grid:
            ax.grid(True, alpha=0.25)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return ax

    def plot_bracket_best(
        self,
        ax: Optional[Any] = None,
        title: str = "Best Loss by Bracket",
        save_path: Optional[str] = None,
    ) -> Any:
        self._ensure_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        by_bracket: Dict[int, List[float]] = {}
        for h in self.history:
            loss = h.get("loss")
            bracket = h.get("bracket")
            if not _is_finite(loss) or bracket is None:
                continue
            by_bracket.setdefault(int(bracket), []).append(float(loss))

        if not by_bracket:
            return ax

        brackets = sorted(by_bracket.keys())
        best = [min(by_bracket[b]) for b in brackets]
        ax.plot(brackets, best, marker="o", color=self.style.best_color, linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Bracket")
        ax.set_ylabel("Best Loss")
        if self.style.grid:
            ax.grid(True, alpha=0.25)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return ax

    # ------------------------------------------------------------------
    # Parameter effect plots
    # ------------------------------------------------------------------

    def plot_param_effect(
        self,
        param: str,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        self._ensure_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        xs, ys = self._param_series(param)
        if not xs:
            return ax

        typ = self._param_type(param)
        if typ in ("float", "int"):
            ax.scatter(xs, ys, s=16, color=self.style.scatter_color, alpha=0.6)
            ax.set_xlabel(param)
            ax.set_ylabel("Loss")
        else:
            # categorical: boxplot by category
            cats = sorted(set(xs))
            data = [np.asarray([y for x, y in zip(xs, ys) if x == c], dtype=float) for c in cats]
            ax.boxplot(data, labels=[str(c) for c in cats], showfliers=False)
            ax.set_xlabel(param)
            ax.set_ylabel("Loss")

        ax.set_title(title or f"Effect of {param}")
        if self.style.grid:
            ax.grid(True, axis="y", alpha=0.25)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return ax

    def rank_params_by_importance(self) -> List[Tuple[str, float]]:
        """
        Return a simple importance proxy:
        - numeric: absolute Spearman correlation with loss
        - categorical: eta^2 (between-group variance / total variance)
        """
        losses = np.asarray(
            [h.get("loss") for h in self.history if _is_finite(h.get("loss"))],
            dtype=float,
        )
        if losses.size == 0:
            return []

        results: List[Tuple[str, float]] = []
        for param in self.config_space.keys():
            xs, ys = self._param_series(param)
            if not xs:
                continue
            typ = self._param_type(param)
            if typ in ("float", "int"):
                score = abs(_spearman(xs, ys))
            else:
                score = _eta_squared(xs, ys)
            results.append((param, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def plot_param_importance(
        self,
        ax: Optional[Any] = None,
        title: str = "Parameter Importance (Proxy)",
        top_k: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        self._ensure_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        ranked = self.rank_params_by_importance()
        if top_k is not None:
            ranked = ranked[: int(top_k)]
        if not ranked:
            return ax

        labels = [r[0] for r in ranked][::-1]
        scores = [r[1] for r in ranked][::-1]
        ax.barh(labels, scores, color=self.style.best_color, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Importance (proxy)")
        if self.style.grid:
            ax.grid(True, axis="x", alpha=0.25)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return ax

    def plot_parallel_coordinates(
        self,
        ax: Optional[Any] = None,
        title: str = "Parallel Coordinates",
        max_points: int = 200,
        save_path: Optional[str] = None,
    ) -> Any:
        self._ensure_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=self.style.dpi)
        else:
            fig = ax.figure

        params = list(self.config_space.keys())
        if not params:
            return ax

        # Collect rows (config + loss)
        rows = []
        for h in self.history:
            cfg = h.get("config", {})
            loss = h.get("loss")
            if not _is_finite(loss):
                continue
            row = []
            for p in params:
                if p in cfg:
                    row.append(cfg[p])
                else:
                    row.append(None)
            rows.append((row, float(loss)))

        if not rows:
            return ax

        # Subsample for clarity
        if len(rows) > max_points:
            idx = np.linspace(0, len(rows) - 1, max_points).astype(int)
            rows = [rows[i] for i in idx]

        # Build scaling per param
        scalers = []
        for i, p in enumerate(params):
            vals = [r[0][i] for r in rows if r[0][i] is not None]
            if not vals:
                scalers.append((0.0, 1.0, "float", {}))
                continue
            typ = self._param_type(p)
            if typ in ("float", "int"):
                v = np.asarray(vals, dtype=float)
                lo, hi = float(np.min(v)), float(np.max(v))
                if abs(hi - lo) < 1e-12:
                    hi = lo + 1.0
                scalers.append((lo, hi, typ, {}))
            else:
                cats = sorted({v for v in vals})
                idx_map = {c: j for j, c in enumerate(cats)}
                scalers.append((0.0, max(1, len(cats) - 1), "choice", idx_map))

        losses = np.asarray([r[1] for r in rows], dtype=float)
        if losses.size == 0:
            return ax
        lo_l, hi_l = float(np.min(losses)), float(np.max(losses))
        if abs(hi_l - lo_l) < 1e-12:
            hi_l = lo_l + 1.0

        # Plot each row
        for row, loss in rows:
            xs = list(range(len(params)))
            ys = []
            for i, val in enumerate(row):
                lo, hi, typ, idx_map = scalers[i]
                if val is None:
                    ys.append(0.0)
                    continue
                if typ in ("float", "int"):
                    y = (float(val) - lo) / (hi - lo)
                else:
                    y = float(idx_map.get(val, 0)) / max(1.0, (hi - lo))
                ys.append(y)
            t = (loss - lo_l) / (hi_l - lo_l)
            color = plt.cm.viridis(1.0 - t)
            ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.8)

        ax.set_xticks(list(range(len(params))))
        ax.set_xticklabels(params, rotation=30, ha="right")
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["low", "mid", "high"])
        ax.set_title(title)
        if self.style.grid:
            ax.grid(True, axis="y", alpha=0.25)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return ax

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _param_series(self, param: str) -> Tuple[List[Any], List[float]]:
        xs, ys = [], []
        for h in self.history:
            cfg = h.get("config", {})
            if param not in cfg:
                continue
            loss = h.get("loss")
            if not _is_finite(loss):
                continue
            xs.append(cfg[param])
            ys.append(float(loss))
        return xs, ys

    def _param_type(self, param: str) -> str:
        spec = self.config_space.get(param)
        if not spec:
            return "float"
        return str(spec[0])

    def _ensure_matplotlib(self) -> None:
        if plt is None:
            raise RuntimeError("matplotlib is required for plotting.")


def _is_finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _spearman(xs: Iterable[Any], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    if np.std(x_rank) < 1e-12 or np.std(y_rank) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    temp = np.argsort(x)
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(x), dtype=float)
    return ranks


def _eta_squared(categories: Iterable[Any], values: Iterable[float]) -> float:
    cats = list(categories)
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return 0.0
    overall_mean = float(np.mean(vals))
    total_var = float(np.sum((vals - overall_mean) ** 2))
    if total_var <= 0:
        return 0.0

    between_var = 0.0
    for c in set(cats):
        group_vals = vals[[i for i, cc in enumerate(cats) if cc == c]]
        if group_vals.size == 0:
            continue
        mean_c = float(np.mean(group_vals))
        between_var += group_vals.size * (mean_c - overall_mean) ** 2

    return float(between_var / total_var)
