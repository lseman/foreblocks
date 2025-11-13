# -*- coding: utf-8 -*-
"""
Reusable NumPy → NeuralForecast Benchmark

- Input: np.ndarray [T, N] (rows=time, cols=series)
- Handles: NaN forward-fill, train-only detrend/normalize, long-format building
- Uses: NeuralForecast cross_validation
- Reports: metrics on normalized space AND original scale (with proper inverse transform)
- Extras: model factory for n_series-aware models, nice summaries + LaTeX table
"""

from __future__ import annotations
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Metric helpers
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# NeuralForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    TFT, DilatedRNN, NHITS, TCN, LSTM, RNN,
    NBEATS, NBEATSx, GRU, Informer, TiDE,
    PatchTST, FEDformer, MLP, TimesNet, DeepAR, BiTCN,
)

# Try newer models if available in your install
try:
    from neuralforecast.models import Autoformer, VanillaTransformer, iTransformer, TimeMixer, TimeXer, StemGNN
    _NEW_MODELS = [Autoformer, VanillaTransformer, iTransformer, TimeMixer, TimeXer, StemGNN]
except Exception:
    _NEW_MODELS = []

warnings.filterwarnings("ignore")


def _ffill_numpy_colwise(arr: np.ndarray) -> np.ndarray:
    """Simple forward-fill per column for NaNs (in-place-safe copy)."""
    out = arr.copy()
    T, N = out.shape
    for j in range(N):
        col = out[:, j]
        mask = np.isnan(col)
        if mask.any():
            # seed the first valid
            first_valid = np.argmax(~mask) if (~mask).any() else 0
            if mask[:first_valid].any():
                col[:first_valid] = col[first_valid]
            # indices of last seen valid
            idx = np.where(~mask, np.arange(T), 0)
            np.maximum.accumulate(idx, out=idx)
            col = col[idx]
        out[:, j] = col
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse_v = mean_squared_error(y_true, y_pred)
    rmse_v = float(np.sqrt(mse_v))
    mae_v = mean_absolute_error(y_true, y_pred)
    try:
        mape_v = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    except Exception:
        mape_v = np.nan
    return {"RMSE": rmse_v, "MSE": float(mse_v), "MAE": float(mae_v), "MAPE": float(mape_v)}


@dataclass
class NPTimeseriesNFBenchmark:
    # Core data
    time_series_original: np.ndarray                    # [T, N]
    series_names: Optional[List[str]] = None            # len N (optional)
    start_date: str = "2018-01-01"
    freq: str = "D"

    # Split & modeling
    data_split: float = 0.2                             # fraction for test
    input_size: int = 50                                # lookback
    horizon: int = 5                                    # forecast horizon
    max_steps: int = 100                                # epochs per model

    # Transforms
    normalize: bool = False
    detrend: bool = False

    # Internals (filled during build)
    _T: int = field(init=False, default=0)
    _N: int = field(init=False, default=0)
    _split_point: int = field(init=False, default=0)
    _date_index: Optional[pd.DatetimeIndex] = field(init=False, default=None)
    _norm_params: Dict[str, Dict[str, object]] = field(init=False, default_factory=dict)

    # DataFrames on both spaces
    _df_norm_full: Optional[pd.DataFrame] = field(init=False, default=None)   # long: [unique_id, ds, y] (normalized)
    _df_orig_full: Optional[pd.DataFrame] = field(init=False, default=None)   # long: [unique_id, ds, y] (original)

    # Splits (long)
    _train_df: Optional[pd.DataFrame] = field(init=False, default=None)
    _test_df: Optional[pd.DataFrame] = field(init=False, default=None)

    # Results
    metrics_normalized: Optional[pd.DataFrame] = field(init=False, default=None)
    metrics_original: Optional[pd.DataFrame] = field(init=False, default=None)

    # Models that require n_series arg
    MODELS_REQUIRE_NSERIES: Tuple[str, ...] = ("iTransformer", "TimeMixer", "TimeXer", "StemGNN")

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

        # Dicts for row-wise assembly
        m_norm: Dict[str, Dict[str, float]] = {}
        m_orig: Dict[str, Dict[str, float]] = {}

        # Full DF (train + test) on normalized space
        full_df = pd.concat([self._train_df, self._test_df], ignore_index=True)
        # Test length per series
        test_len = len(self._test_df[self._test_df["unique_id"] == self.series_names[0]])
        n_windows = max(1, test_len // self.horizon)

        for ModelClass, model_name in models:
            if verbose:
                print(f"▶ {model_name} ... ", end="", flush=True)

            try:
                start = time.time()
                model = self._make_model(ModelClass, model_name)
                nf = NeuralForecast(models=[model], freq=self.freq)

                cv_df = nf.cross_validation(
                    df=full_df,
                    n_windows=n_windows,
                    step_size=step_size,
                )
                elapsed = time.time() - start

                # Predictions on normalized space
                y_true_norm = cv_df["y"].to_numpy(dtype=np.float32)
                y_pred_norm = cv_df[model_name].to_numpy(dtype=np.float32)
                m_norm_row = _metrics(y_true_norm, y_pred_norm)
                m_norm_row.update({"Time": elapsed, "N_Points": len(y_true_norm)})
                m_norm[model_name] = m_norm_row

                # Compute original-scale metrics:
                # 1) denormalize predictions per (unique_id, ds)
                # 2) fetch original y_true via merge with _df_orig_full
                cv_core = cv_df[["unique_id", "ds"]].copy()
                cv_core["y_pred_norm"] = y_pred_norm

                # Denormalize per row using stored params
                cv_core["y_pred_orig"] = cv_core.apply(
                    lambda r: self._denorm_row_pred(
                        uid=r["unique_id"], ds=r["ds"], y_pred_norm=r["y_pred_norm"]
                    ),
                    axis=1,
                )

                # Now attach original y_true using ds+uid
                orig_map = (
                    self._df_orig_full[["unique_id", "ds", "y"]]
                    .rename(columns={"y": "y_true_orig"})
                )
                join_df = cv_core.merge(orig_map, on=["unique_id", "ds"], how="left")

                y_true_orig = join_df["y_true_orig"].to_numpy(dtype=np.float32)
                y_pred_orig = join_df["y_pred_orig"].to_numpy(dtype=np.float32)

                # Keep only finite
                ok = np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)
                m_orig_row = _metrics(y_true_orig[ok], y_pred_orig[ok])
                m_orig_row.update({"Time": elapsed, "N_Points": int(ok.sum())})
                m_orig[model_name] = m_orig_row

                if verbose:
                    print(f"ok ({elapsed:.1f}s, {int(ok.sum())} pts)")

            except Exception as e:
                if verbose:
                    print(f"error: {str(e)[:180]}")
                # Mark NaNs to keep the table shape stable
                m_norm[model_name] = {"RMSE": np.nan, "MSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Time": np.nan, "N_Points": 0}
                m_orig[model_name] = {"RMSE": np.nan, "MSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Time": np.nan, "N_Points": 0}

        self.metrics_normalized = pd.DataFrame(m_norm).T
        self.metrics_original  = pd.DataFrame(m_orig).T

        # Nice ordering
        self.metrics_normalized = self.metrics_normalized.sort_values("MSE")
        self.metrics_original  = self.metrics_original.sort_values("MSE")
        return self.metrics_normalized, self.metrics_original

    def summarize(self) -> None:
        """Pretty-print both tables + brief best-model summary."""
        if self.metrics_normalized is None or self.metrics_original is None:
            print("Run the benchmark first with .run().")
            return

        print("\n" + "="*80)
        print("RESULTS (Normalized Space)")
        print("="*80)
        print(self.metrics_normalized.to_string(float_format=lambda x: f"{x:,.4f}"))

        print("\n" + "="*80)
        print("RESULTS (Original Scale)")
        print("="*80)
        print(self.metrics_original.to_string(float_format=lambda x: f"{x:,.4f}"))

        try:
            best_norm = self.metrics_normalized.index[0]
            best_orig = self.metrics_original.index[0]
            print("\n" + "="*80)
            print("BEST MODELS")
            print("="*80)
            print(f"Normalized: {best_norm}  "
                  f"(RMSE={self.metrics_normalized.loc[best_norm, 'RMSE']:.4f})")
            print(f"Original:   {best_orig}  "
                  f"(RMSE={self.metrics_original.loc[best_orig, 'RMSE']:.4f})")
        except Exception:
            pass

    def latex_table_original(self) -> str:
        """Return a LaTeX table string for original-scale results."""
        if self.metrics_original is None or self.metrics_original.empty:
            return "% Run the benchmark first."
        df = self.metrics_original.copy()
        lines = []
        lines.append("Model        & RMSE      & MSE       & MAE       & MAPE      & Time      \\\\")
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
            (TiDE, 'TiDE'),
            (NHITS, 'NHITS'),
            (NBEATS, 'NBEATS'),
            (NBEATSx, 'NBEATSx'),
            (LSTM, 'LSTM'),
            (GRU, 'GRU'),
            (DeepAR, 'DeepAR'),
            (TFT, 'TFT'),
            (Informer, 'Informer'),
            (PatchTST, 'PatchTST'),
            (FEDformer, 'FEDformer'),
            (TCN, 'TCN'),
            (BiTCN, 'BiTCN'),
            (TimesNet, 'TimesNet'),
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
            self.series_names = [f"S{i+1}" for i in range(self._N)]
        elif len(self.series_names) != self._N:
            raise ValueError("`series_names` length must match N.")

        # Basic NaN handling
        arr = _ffill_numpy_colwise(arr)

        # Calendar
        self._date_index = pd.date_range(start=self.start_date, periods=self._T, freq=str(self.freq))

        # Split
        self._split_point = int(self._T * (1 - self.data_split))
        if not (0 < self._split_point < self._T):
            raise ValueError("Invalid `data_split`—results in empty train or test.")

        # Fit transforms on TRAIN only, apply to ALL
        self._norm_params = {name: {"mean": 0.0, "std": 1.0, "trend": None} for name in self.series_names}
        x_all = np.arange(self._T)
        x_trn = np.arange(self._split_point)

        norm_space = arr.copy()
        for j, name in enumerate(self.series_names):
            y = arr[:, j]

            # Detrend
            if self.detrend:
                c = np.polyfit(x_trn, y[:self._split_point], deg=1)  # (slope, intercept)
                trend_all = np.polyval(c, x_all)
                y_detr = y - trend_all
                self._norm_params[name]["trend"] = (float(c[0]), float(c[1]))
            else:
                y_detr = y

            # Normalize
            if self.normalize:
                mu = float(y_detr[:self._split_point].mean())
                sd = float(y_detr[:self._split_point].std(ddof=0))
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1.0
                y_norm = (y_detr - mu) / sd
                self._norm_params[name]["mean"] = mu
                self._norm_params[name]["std"] = sd
            else:
                y_norm = y_detr

            norm_space[:, j] = y_norm

        # Long DFs
        df_norm = pd.DataFrame(norm_space.astype(np.float32), index=self._date_index, columns=self.series_names)
        df_orig = pd.DataFrame(arr.astype(np.float32), index=self._date_index, columns=self.series_names)

        self._df_norm_full = self._wide_to_long(df_norm)
        self._df_orig_full = self._wide_to_long(df_orig)

        # Split long (same cut for all series)
        self._train_df, self._test_df = self._split_long(self._df_norm_full, self._split_point)

    @staticmethod
    def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for name in df_wide.columns:
            parts.append(pd.DataFrame({"unique_id": name, "ds": df_wide.index, "y": df_wide[name].to_numpy()}))
        return pd.concat(parts, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def _split_long(self, df_long_norm: pd.DataFrame, split_point: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_list, test_list = [], []
        for name in self.series_names:
            d = df_long_norm[df_long_norm["unique_id"] == name].reset_index(drop=True)
            train_list.append(d.iloc[:split_point])
            test_list.append(d.iloc[split_point:])
        return pd.concat(train_list, ignore_index=True), pd.concat(test_list, ignore_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Internals: model construction & denorm
    # ──────────────────────────────────────────────────────────────────────────
    def _make_model(self, ModelClass: type, model_name: str):
        kwargs = dict(h=self.horizon, input_size=self.input_size, max_steps=self.max_steps)
        if model_name in self.MODELS_REQUIRE_NSERIES:
            kwargs["n_series"] = self._N
        return ModelClass(**kwargs)

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
            t_idx = int((ds - self._date_index[0]) / pd.tseries.frequencies.to_offset(self.freq))
            slope, intercept = p["trend"]
            trend = slope * t_idx + intercept
            val = val + trend

        return float(val)


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
        f = np.array([2*np.pi/30, 2*np.pi/50, 2*np.pi/80, 2*np.pi/120])
        base = np.stack([A[i]*np.sin(f[i]*t) for i in range(N)], axis=1)
        noise = 0.2*np.random.randn(T, N)
        trends = (np.array([0.001, -0.0005, 0.0008, 0.0])[None, :] * t[:, None])
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
