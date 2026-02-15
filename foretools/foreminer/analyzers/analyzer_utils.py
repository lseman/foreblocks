from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_numeric_columns(
    data: pd.DataFrame, exclude_cols: Optional[Iterable[str]] = None
) -> List[str]:
    """Return numeric columns excluding any provided names."""
    cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not exclude_cols:
        return cols
    exclude = {c for c in exclude_cols if c is not None}
    return [c for c in cols if c not in exclude]


def get_numeric_frame(
    data: pd.DataFrame,
    exclude_cols: Optional[Iterable[str]] = None,
    *,
    replace_inf: bool = True,
) -> pd.DataFrame:
    """Return numeric dataframe with optional excluded columns and inf cleanup."""
    cols = get_numeric_columns(data, exclude_cols=exclude_cols)
    numeric = data[cols].copy()
    if replace_inf and not numeric.empty:
        numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    return numeric


def drop_constant_numeric(
    numeric: pd.DataFrame, tol: float = 1e-12
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop constant/near-constant columns from a numeric dataframe."""
    keep_cols: List[str] = []
    for col in numeric.columns:
        arr = numeric[col].to_numpy()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        if (np.max(finite) - np.min(finite)) > tol:
            keep_cols.append(col)
    return numeric[keep_cols], keep_cols


def build_series_map(
    data: pd.DataFrame,
    columns: List[str],
    *,
    min_length: int = 10,
    max_count: Optional[int] = None,
) -> Dict[str, pd.Series]:
    """Create a cleaned series map for columns meeting minimum length."""
    selected = columns if max_count is None else columns[:max_count]
    series_map: Dict[str, pd.Series] = {}
    for col in selected:
        clean_series = data[col].dropna()
        if len(clean_series) >= min_length:
            series_map[col] = clean_series
    return series_map


def safe_call(
    fn: Callable[..., Any], *args: Any, default: Any = None, **kwargs: Any
) -> Tuple[Any, Optional[str]]:
    """Execute a callable and return (result, error_message)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return default, str(exc)
