"""Shared result formatting and inverse transforms for NeuralForecast benchmarks."""

import pandas as pd


def summarize(metrics_normalized, metrics_original):
    if metrics_normalized is None or metrics_original is None:
        print("Run the benchmark first with .run().")
        return

    print("\n" + "=" * 80)
    print("RESULTS (Normalized Space)")
    print("=" * 80)
    print(metrics_normalized.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\n" + "=" * 80)
    print("RESULTS (Original Scale)")
    print("=" * 80)
    print(metrics_original.to_string(float_format=lambda x: f"{x:,.4f}"))

    try:
        best_norm = metrics_normalized.index[0]
        best_orig = metrics_original.index[0]
        print("\n" + "=" * 80)
        print("BEST MODELS")
        print("=" * 80)
        print(
            f"Normalized: {best_norm}  "
            f"(RMSE={metrics_normalized.loc[best_norm, 'RMSE']:.4f})"
        )
        print(
            f"Original:   {best_orig}  "
            f"(RMSE={metrics_original.loc[best_orig, 'RMSE']:.4f})"
        )
    except Exception:
        pass


def latex_table_original(metrics_original):
    if metrics_original is None or metrics_original.empty:
        return "% Run the benchmark first."
    df = metrics_original.copy()
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


def denormalize_prediction(params, ds, y_pred_norm, *, normalize, detrend, date_index, freq):
    val = float(y_pred_norm)

    # de-normalize
    if normalize:
        val = val * float(params["std"]) + float(params["mean"])

    # add back trend evaluated at absolute time index
    if detrend and params["trend"] is not None:
        # absolute index for this timestamp
        t_idx = int(
            (ds - date_index[0]) / pd.tseries.frequencies.to_offset(freq)
        )
        slope, intercept = params["trend"]
        trend = slope * t_idx + intercept
        val = val + trend

    return float(val)
