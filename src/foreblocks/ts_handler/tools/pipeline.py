"""foreblocks.ts_handler.pipeline.

Pipeline runner for time-series preprocessing.

"""

from __future__ import annotations

import numpy as np


def _run_pipeline(
    x: np.ndarray,
    mode: str,
    time_stamps: np.ndarray | None,
    apply_imputation: bool,
    impute_missing_fn: callable,
    remove_outliers: bool,
    parallel_outlier_clean_fn: callable,
    apply_ewt: bool,
    apply_ewt_and_detrend_fn: callable,
    apply_filter: bool,
    filter_method: str,
    apply_filter_fn: callable,
    differencing: bool,
    set_diff_values_fn: callable,
    apply_scaling_stage_fn: callable,
    maybe_plot_fn: callable,
    vprint: callable,
    feature_state: any,
    log_transform_flags: list[bool] | None,
    log_offset: np.ndarray | None,
    scaler: any | None,
    normalize: bool,
    scaling_method: str,
    trend_component: np.ndarray | None,
    detrend: bool,
    diff_values: np.ndarray | None,
) -> np.ndarray:
    processed = np.array(x, dtype=float, copy=True)
    vprint(f"Starting {mode} pipeline (shape: {x.shape})")

    # 1) Impute (optional)
    if np.any(np.isnan(processed)):
        if apply_imputation:
            vprint(f"Applying imputation ({feature_state.impute_method})")
            processed = impute_missing_fn(processed)
            maybe_plot_fn(processed, "After Imputation", time_stamps)
        if np.any(np.isnan(processed)):
            raise ValueError(
                "NaNs remain after imputation. Enable apply_imputation or change method."
            )

    # 2) Log transform (fit learns offsets; transform uses learned)
    processed = apply_log_stage_fn(processed, mode)

    # 3) Outliers (kept: user-controlled; beware leakage but you already chose this)
    if remove_outliers:
        vprint(f"Applying outlier removal ({feature_state.outlier_method})")
        processed = parallel_outlier_clean_fn(processed)
        maybe_plot_fn(processed, "After Outlier Removal", time_stamps)
        if np.any(np.isnan(processed)) and apply_imputation:
            vprint("Re-applying imputation after outlier removal")
            processed = impute_missing_fn(processed)
        if np.any(np.isnan(processed)):
            raise ValueError(
                "NaNs remain after outlier removal. Enable imputation or adjust the outlier settings."
            )

    # 4) EWT + detrend
    if apply_ewt:
        vprint(f"Applying EWT & detrending ({feature_state.ewt_bands} bands)")
        processed = apply_ewt_and_detrend_fn(processed)

    # 5) Filtering
    if apply_filter:
        vprint(f"Applying signal filtering ({filter_method})")
        processed = apply_filter_fn(processed, method=filter_method)
        maybe_plot_fn(
            processed,
            f"After {filter_method.capitalize()} Filtering",
            time_stamps,
        )

    # 6) Differencing
    if differencing:
        vprint("Applying differencing")
        if mode == "fit":
            set_diff_values_fn(processed[0:1].copy())
        processed = np.vstack(
            [
                np.zeros_like(processed[0]),
                np.diff(processed, axis=0),
            ]
        )

    # 7) Normalization / Scaling (Adaptive)
    return apply_scaling_stage_fn(processed, mode)
