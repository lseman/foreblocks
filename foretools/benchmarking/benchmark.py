# -*- coding: utf-8 -*-
"""
Reusable NumPy → NeuralForecast Benchmark

- Input: np.ndarray [T, N] (rows=time, cols=series)
- Handles: NaN forward-fill, train-only detrend/normalize, long-format building
- Uses: NeuralForecast cross_validation
- Reports: metrics on normalized space AND original scale (with proper inverse transform)
- Extras: model factory for n_series-aware models, nice summaries + LaTeX table,
  critical difference diagram from per-window RMSE
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NeuralForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    GRU,
    LSTM,
    NBEATS,
    NHITS,
    TCN,
    TFT,
    BiTCN,
    DeepAR,
    FEDformer,
    Informer,
    NBEATSx,
    PatchTST,
    TiDE,
    TimesNet,
)

# Metric helpers
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# CD diagram deps (optional)
try:
    import scikit_posthocs as sp
    from scipy.stats import friedmanchisquare, studentized_range

    _HAS_POSTHOCS = True
except Exception:
    _HAS_POSTHOCS = False

# Try newer models if available in your install
try:
    from neuralforecast.models import (
        Autoformer,
        StemGNN,
        TimeMixer,
        TimeXer,
        VanillaTransformer,
        iTransformer,
    )

    _NEW_MODELS = [
        Autoformer,
        VanillaTransformer,
        iTransformer,
        TimeMixer,
        TimeXer,
        StemGNN,
    ]
except Exception:
    _NEW_MODELS = []

warnings.filterwarnings("ignore")


def _ffill_numpy_colwise(arr: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(arr)
    df_ffill = df.ffill(axis=0)
    return df_ffill.to_numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse_v = mean_squared_error(y_true, y_pred)
    rmse_v = float(np.sqrt(mse_v))
    mae_v = mean_absolute_error(y_true, y_pred)
    try:
        mape_v = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    except Exception:
        mape_v = np.nan
    return {
        "RMSE": rmse_v,
        "MSE": float(mse_v),
        "MAE": float(mae_v),
        "MAPE": float(mape_v),
    }


def _find_cliques(
    avg_ranks: pd.Series,
    nemenyi_pvalues: pd.DataFrame,
    alpha: float,
) -> List[Tuple[float, float]]:
    models = avg_ranks.sort_values().index.tolist()
    ranks = avg_ranks.loc[models].values.astype(float)
    K = len(models)

    not_sig = np.ones((K, K), dtype=bool)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            mi, mj = models[i], models[j]
            not_sig[i, j] = bool(nemenyi_pvalues.loc[mi, mj] > alpha)

    segments: List[Tuple[int, int]] = []
    for i in range(K):
        for j in range(i + 1, K):
            ok = True
            for a in range(i, j + 1):
                for b in range(a + 1, j + 1):
                    if not not_sig[a, b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                segments.append((i, j))

    maximal: List[Tuple[int, int]] = []
    for i, j in segments:
        is_subset = False
        for i2, j2 in segments:
            if (i2 <= i and j <= j2) and (i2 < i or j < j2):
                is_subset = True
                break
        if not is_subset:
            maximal.append((i, j))

    out: List[Tuple[float, float]] = [
        (float(ranks[i]), float(ranks[j])) for i, j in maximal
    ]
    out.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    return out


def _plot_cd_diagram(
    ranks_df: pd.DataFrame,
    alpha: float = 0.05,
    title: str = "Critical Difference Diagram",
    figsize: Tuple[float, float] = (10, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    if not _HAS_POSTHOCS:
        raise ImportError(
            "scikit-posthocs + scipy required. Install: pip install scikit-posthocs scipy"
        )

    N, K = ranks_df.shape
    model_names = list(ranks_df.columns)

    ranks = ranks_df.rank(axis=1, method="average", ascending=True)
    avg_ranks = ranks.mean(axis=0).sort_values(ascending=True)

    stat, p_value = friedmanchisquare(*[ranks_df[c].values for c in model_names])

    nemenyi = sp.posthoc_nemenyi_friedman(ranks_df.values)
    nemenyi.index = model_names
    nemenyi.columns = model_names

    df = max(1, (K - 1) * (N - 1))
    q_alpha = float(studentized_range.ppf(1 - alpha, K, df))
    cd = q_alpha * np.sqrt(K * (K + 1) / (6.0 * N))

    fig, ax = plt.subplots(figsize=figsize)
    lowv, highv = 1.0, float(K)
    ax.set_xlim(lowv - 0.5, highv + 0.5)
    ax.set_ylim(0.0, 1.0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", direction="out", pad=5)
    ax.set_xticks(range(1, K + 1))
    ax.set_xlabel("Average Rank (lower is better)", fontsize=11)

    axis_y = 0.70
    ax.hlines(axis_y, lowv, highv, color="black", linewidth=1.4)
    for i in range(1, K + 1):
        ax.vlines(i, axis_y - 0.02, axis_y + 0.02, color="black", linewidth=1.2)

    cd_y = 0.86
    cd_start = lowv
    cd_end = min(highv, lowv + cd)
    ax.hlines(cd_y, cd_start, cd_end, color="black", linewidth=2)
    ax.vlines(cd_start, cd_y - 0.02, cd_y + 0.02, color="black", linewidth=2)
    ax.vlines(cd_end, cd_y - 0.02, cd_y + 0.02, color="black", linewidth=2)
    ax.text(
        (cd_start + cd_end) / 2,
        cd_y + 0.04,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    sorted_models = avg_ranks.index.tolist()
    sorted_vals = avg_ranks.values.astype(float)
    half = (K + 1) // 2
    left_models, left_vals = sorted_models[:half], sorted_vals[:half]
    right_models, right_vals = sorted_models[half:], sorted_vals[half:]

    text_y_left = np.linspace(0.56, 0.12, len(left_models)) if left_models else []
    text_y_right = np.linspace(0.56, 0.12, len(right_models)) if right_models else []

    for y, m, r in zip(text_y_left, left_models, left_vals):
        ax.plot([r, r], [axis_y, y], "k-", linewidth=0.8)
        ax.plot([r, lowv - 0.30], [y, y], "k-", linewidth=0.8)
        ax.text(lowv - 0.35, y, f"{m} ({r:.2f})", ha="right", va="center", fontsize=9)
        ax.plot(r, axis_y, "ko", markersize=5)

    for y, m, r in zip(text_y_right, right_models, right_vals):
        ax.plot([r, r], [axis_y, y], "k-", linewidth=0.8)
        ax.plot([r, highv + 0.30], [y, y], "k-", linewidth=0.8)
        ax.text(highv + 0.35, y, f"({r:.2f}) {m}", ha="left", va="center", fontsize=9)
        ax.plot(r, axis_y, "ko", markersize=5)

    clique_y = axis_y - 0.09
    cliques = _find_cliques(avg_ranks, nemenyi, alpha)
    for start_r, end_r in cliques:
        ax.hlines(clique_y, start_r, end_r, color="red", linewidth=3, alpha=0.7)
        clique_y -= 0.06

    ax.set_title(f"{title}\n(Friedman p={p_value:.4g})", fontsize=12, pad=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@dataclass
class NPTimeseriesNFBenchmark:
    # Core data
    time_series_original: np.ndarray  # [T, N]
    series_names: Optional[List[str]] = None  # len N (optional)
    start_date: str = "2018-01-01"
    freq: str = "D"

    # Split & modeling
    data_split: float = 0.2  # fraction for test
    input_size: int = 50  # lookback
    horizon: int = 5  # forecast horizon
    max_steps: int = 100  # epochs per model
    early_stop_patience_steps: int = 0
    val_size: Optional[int] = None  # None -> auto (horizon if early-stop enabled)

    # Transforms
    normalize: bool = False
    detrend: bool = False

    # Internals (filled during build)
    _T: int = field(init=False, default=0)
    _N: int = field(init=False, default=0)
    _split_point: int = field(init=False, default=0)
    _date_index: Optional[pd.DatetimeIndex] = field(init=False, default=None)
    _ds_to_tidx: Optional[pd.Series] = field(init=False, default=None)
    _norm_params: Dict[str, Dict[str, object]] = field(init=False, default_factory=dict)
    _params_df: Optional[pd.DataFrame] = field(init=False, default=None)

    # DataFrames on both spaces
    _df_norm_full: Optional[pd.DataFrame] = field(
        init=False, default=None
    )  # long: [unique_id, ds, y] (normalized)
    _df_orig_full: Optional[pd.DataFrame] = field(
        init=False, default=None
    )  # long: [unique_id, ds, y] (original)

    # Splits (long)
    _train_df: Optional[pd.DataFrame] = field(init=False, default=None)
    _test_df: Optional[pd.DataFrame] = field(init=False, default=None)

    # Results
    metrics_normalized: Optional[pd.DataFrame] = field(init=False, default=None)
    metrics_original: Optional[pd.DataFrame] = field(init=False, default=None)
    _per_window_rmse_norm: Optional[pd.DataFrame] = field(init=False, default=None)
    _per_window_rmse_orig: Optional[pd.DataFrame] = field(init=False, default=None)

    # Models that require n_series arg
    MODELS_REQUIRE_NSERIES: Tuple[str, ...] = (
        "iTransformer",
        "TimeMixer",
        "TimeXer",
        "StemGNN",
    )

    def __post_init__(self):
        self._prepare_data()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def run(
        self,
        models: Optional[Sequence[Tuple[type, str]]] = None,
        step_size: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Trains/evaluates all models with NeuralForecast cross_validation.
        Returns (metrics_normalized_df, metrics_original_df)
        """
        if models is None:
            models = self.default_models()

        step_size = int(step_size or self.horizon)
        if step_size <= 0:
            raise ValueError("step_size must be > 0")

        # Dicts for row-wise assembly
        m_norm: Dict[str, Dict[str, float]] = {}
        m_orig: Dict[str, Dict[str, float]] = {}
        per_win_rmse_norm: Dict[str, Dict[str, float]] = {}
        per_win_rmse_orig: Dict[str, Dict[str, float]] = {}

        # Full DF (train + test) on normalized space
        full_df = pd.concat([self._train_df, self._test_df], ignore_index=True)
        # Test length per series
        test_len = len(
            self._test_df[self._test_df["unique_id"] == self.series_names[0]]
        )
        # Align window count with requested step_size.
        n_windows = max(1, (test_len - self.horizon) // step_size + 1)

        for ModelClass, model_name in models:
            if verbose:
                print(f"▶ {model_name} ... ", end="", flush=True)

            try:
                start = time.time()
                model = self._make_model(ModelClass, model_name)
                nf = NeuralForecast(models=[model], freq=self.freq)

                cv_kwargs = dict(
                    df=full_df,
                    n_windows=n_windows,
                    step_size=step_size,
                )
                val_size_eff = self._resolve_val_size()
                if val_size_eff > 0:
                    cv_kwargs["val_size"] = val_size_eff
                try:
                    cv_df = nf.cross_validation(**cv_kwargs)
                except TypeError:
                    cv_kwargs.pop("val_size", None)
                    cv_df = nf.cross_validation(**cv_kwargs)
                elapsed = time.time() - start

                # Predictions on normalized space
                y_true_norm = cv_df["y"].to_numpy(dtype=np.float32)
                y_pred_norm = cv_df[model_name].to_numpy(dtype=np.float32)
                m_norm_row = _metrics(y_true_norm, y_pred_norm)
                m_norm_row.update({"Time": elapsed, "N_Points": len(y_true_norm)})
                m_norm[model_name] = m_norm_row
                if "cutoff" in cv_df.columns:
                    g_norm = (
                        cv_df.assign(
                            y_true=cv_df["y"].astype(np.float64),
                            y_pred=cv_df[model_name].astype(np.float64),
                        )
                        .groupby("cutoff", sort=True)
                        .apply(
                            lambda d: float(
                                np.sqrt(mean_squared_error(d["y_true"], d["y_pred"]))
                            )
                        )
                    )
                    per_win_rmse_norm[model_name] = {
                        str(k): float(v) for k, v in g_norm.items()
                    }
                else:
                    per_win_rmse_norm[model_name] = {}

                # Compute original-scale metrics:
                # 1) denormalize predictions per (unique_id, ds)
                # 2) fetch original y_true via merge with _df_orig_full
                cv_core = cv_df[["unique_id", "ds"]].copy()
                if "cutoff" in cv_df.columns:
                    cv_core["cutoff"] = cv_df["cutoff"].to_numpy()
                cv_core["y_pred_norm"] = y_pred_norm

                # Denormalize vectorized (faster and less notebook overhead)
                cv_core["y_pred_orig"] = self._denorm_vectorized(
                    unique_id=cv_core["unique_id"],
                    ds=cv_core["ds"],
                    y_pred_norm=cv_core["y_pred_norm"],
                )

                # Now attach original y_true using ds+uid
                orig_map = self._df_orig_full[["unique_id", "ds", "y"]].rename(
                    columns={"y": "y_true_orig"}
                )
                join_df = cv_core.merge(orig_map, on=["unique_id", "ds"], how="left")

                y_true_orig = join_df["y_true_orig"].to_numpy(dtype=np.float32)
                y_pred_orig = join_df["y_pred_orig"].to_numpy(dtype=np.float32)

                # Keep only finite
                ok = np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)
                m_orig_row = _metrics(y_true_orig[ok], y_pred_orig[ok])
                m_orig_row.update({"Time": elapsed, "N_Points": int(ok.sum())})
                m_orig[model_name] = m_orig_row
                if "cutoff" in join_df.columns:
                    d_ok = join_df.loc[ok].copy()
                    if not d_ok.empty:
                        g_orig = (
                            d_ok.assign(
                                y_true=d_ok["y_true_orig"].astype(np.float64),
                                y_pred=d_ok["y_pred_orig"].astype(np.float64),
                            )
                            .groupby("cutoff", sort=True)
                            .apply(
                                lambda d: float(
                                    np.sqrt(
                                        mean_squared_error(d["y_true"], d["y_pred"])
                                    )
                                )
                            )
                        )
                        per_win_rmse_orig[model_name] = {
                            str(k): float(v) for k, v in g_orig.items()
                        }
                    else:
                        per_win_rmse_orig[model_name] = {}
                else:
                    per_win_rmse_orig[model_name] = {}

                if verbose:
                    print(f"ok ({elapsed:.1f}s, {int(ok.sum())} pts)")

            except Exception as e:
                if verbose:
                    print(f"error: {str(e)[:180]}")
                # Mark NaNs to keep the table shape stable
                m_norm[model_name] = {
                    "RMSE": np.nan,
                    "MSE": np.nan,
                    "MAE": np.nan,
                    "MAPE": np.nan,
                    "Time": np.nan,
                    "N_Points": 0,
                }
                m_orig[model_name] = {
                    "RMSE": np.nan,
                    "MSE": np.nan,
                    "MAE": np.nan,
                    "MAPE": np.nan,
                    "Time": np.nan,
                    "N_Points": 0,
                }
                per_win_rmse_norm[model_name] = {}
                per_win_rmse_orig[model_name] = {}

        self.metrics_normalized = pd.DataFrame(m_norm).T
        self.metrics_original = pd.DataFrame(m_orig).T
        self._per_window_rmse_norm = pd.DataFrame(per_win_rmse_norm).sort_index()
        self._per_window_rmse_orig = pd.DataFrame(per_win_rmse_orig).sort_index()

        # Nice ordering
        self.metrics_normalized = self.metrics_normalized.sort_values("MSE")
        self.metrics_original = self.metrics_original.sort_values("MSE")
        return self.metrics_normalized, self.metrics_original

    def plot_critical_difference(
        self,
        space: str = "original",
        alpha: float = 0.05,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        if not _HAS_POSTHOCS:
            raise ImportError(
                "scikit-posthocs + scipy are required. Install: pip install scikit-posthocs scipy"
            )

        if str(space).lower() == "original":
            df = self._per_window_rmse_orig
            space_label = "Original Scale"
        else:
            df = self._per_window_rmse_norm
            space_label = "Normalized Space"

        if df is None:
            raise ValueError("Run the benchmark first with .run()")

        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="any")

        if df.shape[0] < 3:
            raise ValueError(
                f"Need at least 3 windows for CD diagram, got {df.shape[0]}"
            )
        if df.shape[1] < 2:
            raise ValueError(f"Need at least 2 models, got {df.shape[1]}")

        if title is None:
            title = (
                f"Critical Difference Diagram ({space_label}, {df.shape[0]} windows)"
            )

        return _plot_cd_diagram(
            ranks_df=df,
            alpha=alpha,
            title=title,
            figsize=figsize,
            save_path=save_path,
        )

    def summarize(self) -> None:
        """Pretty-print both tables + brief best-model summary."""
        if self.metrics_normalized is None or self.metrics_original is None:
            print("Run the benchmark first with .run().")
            return

        print("\n" + "=" * 80)
        print("RESULTS (Normalized Space)")
        print("=" * 80)
        print(self.metrics_normalized.to_string(float_format=lambda x: f"{x:,.4f}"))

        print("\n" + "=" * 80)
        print("RESULTS (Original Scale)")
        print("=" * 80)
        print(self.metrics_original.to_string(float_format=lambda x: f"{x:,.4f}"))

        try:
            best_norm = self.metrics_normalized.index[0]
            best_orig = self.metrics_original.index[0]
            print("\n" + "=" * 80)
            print("BEST MODELS")
            print("=" * 80)
            print(
                f"Normalized: {best_norm}  "
                f"(RMSE={self.metrics_normalized.loc[best_norm, 'RMSE']:.4f})"
            )
            print(
                f"Original:   {best_orig}  "
                f"(RMSE={self.metrics_original.loc[best_orig, 'RMSE']:.4f})"
            )
        except Exception:
            pass

    def latex_table_original(self) -> str:
        """Return a LaTeX table string for original-scale results."""
        if self.metrics_original is None or self.metrics_original.empty:
            return "% Run the benchmark first."
        df = self.metrics_original.copy()
        lines = []
        lines.append(
            "Model        & RMSE      & MSE       & MAE       & MAPE      & Time      \\\\"
        )
        lines.append("\\hline")
        for model in df.index:
            row = df.loc[model]
            lines.append(
                f"{model:16} & {row['RMSE']:.2E} & {row['MSE']:.2E} "
                f"& {row['MAE']:.2E} & {row['MAPE']:.2E} & {row['Time']:.2E} \\\\"
            )
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Configuration helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def default_models() -> List[Tuple[type, str]]:
        """Safe default list: include what’s available in this environment."""
        base = [
            (TiDE, "TiDE"),
            (NHITS, "NHITS"),
            (NBEATS, "NBEATS"),
            (NBEATSx, "NBEATSx"),
            (LSTM, "LSTM"),
            (GRU, "GRU"),
            (DeepAR, "DeepAR"),
            (TFT, "TFT"),
            (Informer, "Informer"),
            (PatchTST, "PatchTST"),
            (FEDformer, "FEDformer"),
            (TCN, "TCN"),
            (BiTCN, "BiTCN"),
            (TimesNet, "TimesNet"),
        ]
        # Add newer models if import was successful
        for cls in _NEW_MODELS:
            base.append((cls, cls.__name__))
        return base

    # ──────────────────────────────────────────────────────────────────────────
    # Internals: data prep & transforms
    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_data(self) -> None:
        arr = np.asarray(self.time_series_original, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("`time_series_original` must be 2D [T, N].")
        self._T, self._N = arr.shape

        # Names
        if self.series_names is None:
            self.series_names = [f"S{i + 1}" for i in range(self._N)]
        elif len(self.series_names) != self._N:
            raise ValueError("`series_names` length must match N.")

        # Basic NaN handling
        arr = _ffill_numpy_colwise(arr)

        # Calendar
        self._date_index = pd.date_range(
            start=self.start_date, periods=self._T, freq=str(self.freq)
        )
        self._ds_to_tidx = pd.Series(
            np.arange(self._T, dtype=np.int32), index=self._date_index
        )

        # Split
        self._split_point = int(self._T * (1 - self.data_split))
        if not (0 < self._split_point < self._T):
            raise ValueError("Invalid `data_split`—results in empty train or test.")

        # Fit transforms on TRAIN only, apply to ALL
        self._norm_params = {
            name: {"mean": 0.0, "std": 1.0, "trend": None} for name in self.series_names
        }
        x_all = np.arange(self._T)
        x_trn = np.arange(self._split_point)

        norm_space = arr.copy()
        for j, name in enumerate(self.series_names):
            y = arr[:, j]

            # Detrend
            if self.detrend:
                c = np.polyfit(
                    x_trn, y[: self._split_point], deg=1
                )  # (slope, intercept)
                trend_all = np.polyval(c, x_all)
                y_detr = y - trend_all
                self._norm_params[name]["trend"] = (float(c[0]), float(c[1]))
            else:
                y_detr = y

            # Normalize
            if self.normalize:
                mu = float(y_detr[: self._split_point].mean())
                sd = float(y_detr[: self._split_point].std(ddof=0))
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1.0
                y_norm = (y_detr - mu) / sd
                self._norm_params[name]["mean"] = mu
                self._norm_params[name]["std"] = sd
            else:
                y_norm = y_detr

            norm_space[:, j] = y_norm

        self._params_df = pd.DataFrame(
            [
                dict(
                    unique_id=name,
                    mean=float(self._norm_params[name]["mean"]),
                    std=float(self._norm_params[name]["std"]),
                    slope=(
                        float(self._norm_params[name]["trend"][0])
                        if self._norm_params[name]["trend"] is not None
                        else 0.0
                    ),
                    intercept=(
                        float(self._norm_params[name]["trend"][1])
                        if self._norm_params[name]["trend"] is not None
                        else 0.0
                    ),
                )
                for name in self.series_names
            ]
        )

        # Long DFs
        df_norm = pd.DataFrame(
            norm_space.astype(np.float32),
            index=self._date_index,
            columns=self.series_names,
        )
        df_orig = pd.DataFrame(
            arr.astype(np.float32), index=self._date_index, columns=self.series_names
        )

        self._df_norm_full = self._wide_to_long(df_norm)
        self._df_orig_full = self._wide_to_long(df_orig)

        # Split long (same cut for all series)
        self._train_df, self._test_df = self._split_long(
            self._df_norm_full, self._split_point
        )

    @staticmethod
    def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for name in df_wide.columns:
            parts.append(
                pd.DataFrame(
                    {
                        "unique_id": name,
                        "ds": df_wide.index,
                        "y": df_wide[name].to_numpy(),
                    }
                )
            )
        return (
            pd.concat(parts, ignore_index=True)
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )

    def _split_long(
        self, df_long_norm: pd.DataFrame, split_point: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_list, test_list = [], []
        for name in self.series_names:
            d = df_long_norm[df_long_norm["unique_id"] == name].reset_index(drop=True)
            train_list.append(d.iloc[:split_point])
            test_list.append(d.iloc[split_point:])
        return pd.concat(train_list, ignore_index=True), pd.concat(
            test_list, ignore_index=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internals: model construction & denorm
    # ──────────────────────────────────────────────────────────────────────────
    def _make_model(self, ModelClass: type, model_name: str):
        kwargs = dict(
            h=self.horizon,
            input_size=self.input_size,
            max_steps=self.max_steps,
            early_stop_patience_steps=int(self.early_stop_patience_steps),
        )
        if model_name in self.MODELS_REQUIRE_NSERIES:
            kwargs["n_series"] = self._N
        try:
            return ModelClass(**kwargs)
        except TypeError:
            # Some models may not expose early_stop_patience_steps in constructor.
            kwargs.pop("early_stop_patience_steps", None)
            return ModelClass(**kwargs)

    def _resolve_val_size(self) -> int:
        """Resolve validation length used by NF cross_validation fit calls."""
        if self.val_size is not None:
            v = int(self.val_size)
        elif int(self.early_stop_patience_steps) > 0:
            v = int(self.horizon)
        else:
            v = 0
        v = max(0, v)

        # Guard for short history in first window.
        min_len = len(
            self._train_df[self._train_df["unique_id"] == self.series_names[0]]
        )
        needed = int(self.input_size) + v + 1
        if min_len < needed:
            return 0
        return v

    def _denorm_row_pred(self, uid: str, ds: pd.Timestamp, y_pred_norm: float) -> float:
        """Inverse transform a single prediction using stored params and its time index."""
        p = self._norm_params[uid]
        val = float(y_pred_norm)

        # de-normalize
        if self.normalize:
            val = val * float(p["std"]) + float(p["mean"])

        # add back trend evaluated at absolute time index
        if self.detrend and p["trend"] is not None:
            # absolute index for this timestamp
            t_idx = int(
                (ds - self._date_index[0]) / pd.tseries.frequencies.to_offset(self.freq)
            )
            slope, intercept = p["trend"]
            trend = slope * t_idx + intercept
            val = val + trend

        return float(val)

    def _denorm_vectorized(
        self, unique_id: pd.Series, ds: pd.Series, y_pred_norm: pd.Series
    ) -> np.ndarray:
        """Vectorized inverse transform in original scale."""
        if self._params_df is None or self._ds_to_tidx is None:
            raise RuntimeError("Normalization parameters not prepared.")

        p = pd.DataFrame({"unique_id": unique_id.to_numpy()}).merge(
            self._params_df, on="unique_id", how="left"
        )
        y = y_pred_norm.to_numpy(dtype=np.float64)

        if self.normalize:
            y = y * p["std"].to_numpy(dtype=np.float64) + p["mean"].to_numpy(
                dtype=np.float64
            )

        if self.detrend:
            t_idx = ds.map(self._ds_to_tidx).to_numpy(dtype=np.float64)
            y = (
                y
                + p["slope"].to_numpy(dtype=np.float64) * t_idx
                + p["intercept"].to_numpy(dtype=np.float64)
            )
        return y.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo generator (disable if you pass your own array)
    USE_DEMO = False

    if USE_DEMO:
        T, N = 800, 4
        t = np.arange(T)
        A = np.array([1.0, 0.7, 1.4, 0.9])
        f = np.array([2 * np.pi / 30, 2 * np.pi / 50, 2 * np.pi / 80, 2 * np.pi / 120])
        base = np.stack([A[i] * np.sin(f[i] * t) for i in range(N)], axis=1)
        noise = 0.2 * np.random.randn(T, N)
        trends = np.array([0.001, -0.0005, 0.0008, 0.0])[None, :] * t[:, None]
        arr = (base + noise + trends).astype(np.float32)
        names = ["SANTA CLARA-PR", "G B MUNHOZ", "GOV JAYME CANET JR", "G P SOUZA"]
    else:
        # Replace with your own array and names
        # e.g., arr = processed_data  # shape [T, N]
        raise SystemExit("Provide your array in your notebook: see class usage below.")

    bench = NPTimeseriesNFBenchmark(
        time_series_original=arr,
        series_names=names,
        start_date="2018-01-01",
        freq="D",
        data_split=0.2,
        input_size=50,
        horizon=5,
        max_steps=100,
        normalize=False,
        detrend=False,
    )
    bench.run()
    bench.summarize()
    print("\nLaTeX (Original):")
    print(bench.latex_table_original())
