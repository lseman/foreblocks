from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class DateTimeTransformer(BaseFeatureTransformer):
    """
    Datetime feature extraction with tz support, DST-safe cyclicals, fixed anchors,
    and optional holiday/business flags.
    """

    def __init__(
        self,
        config,
        include_cyclical: bool = True,
        include_flags: bool = True,
        include_elapsed: bool = True,
        group_key: Optional[str] = None,  # e.g., customer_id, series_id
        country_holidays: Optional[str] = None,  # e.g., "BR", "US", "GB"
    ):
        super().__init__(config)
        self.include_cyclical = include_cyclical
        self.include_flags = include_flags
        self.include_elapsed = include_elapsed
        self.group_key = group_key
        self.country_holidays = country_holidays

        self.datetime_cols_: list = []
        self._anchors_global_: Dict[str, pd.Timestamp] = {}
        self._anchors_group_: Dict[str, Dict[Union[str, int], pd.Timestamp]] = {}
        self._has_holidays = False
        self._holiday_set = None

    def _coerce_dt(self, s: pd.Series) -> pd.Series:
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce", utc=True)
        # normalize to UTC, keep as tz-aware to avoid DST misalignment; downstream uses .dt components
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        return s

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "DateTimeTransformer":
        # detect datetime columns, including tz-aware
        self.datetime_cols_ = [
            c
            for c in X.columns
            if pd.api.types.is_datetime64_any_dtype(X[c])
            or (
                pd.api.types.is_object_dtype(X[c])
                and pd.to_datetime(X[c], errors="ignore") is not X[c]
            )
        ]
        # Coerce and store anchors (global, and optionally per-group)
        if not self.datetime_cols_:
            self.is_fitted = True
            return self

        if self.country_holidays:
            try:
                import holidays as _hol

                self._holiday_set = _hol.country_holidays(self.country_holidays)
                self._has_holidays = True
            except Exception:
                self._has_holidays = False

        gkey = self.group_key if (self.group_key in X.columns) else None

        for col in self.datetime_cols_:
            dt = self._coerce_dt(X[col])
            valid = dt.dropna()
            if valid.empty:
                # Anchor to a fixed epoch if column is all NaT
                self._anchors_global_[col] = pd.Timestamp("1970-01-01", tz="UTC")
                continue
            self._anchors_global_[col] = valid.min()

            if gkey is not None:
                self._anchors_group_.setdefault(col, {})
                # per-group anchor = group-specific min timestamp (prevents leakage across groups)
                grp = X[[gkey, col]].copy()
                grp[col] = self._coerce_dt(grp[col])
                a = grp.dropna().groupby(gkey)[col].min()
                self._anchors_group_[col] = a.to_dict()

        self.is_fitted = True
        return self

    def _cyc(self, x: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
        # robust cyclic transform (sin, cos), ignoring NaN
        r = 2.0 * np.pi * (x / period)
        return np.sin(r), np.cos(r)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.datetime_cols_:
            return pd.DataFrame(index=X.index)

        out = {}
        gkey = self.group_key if (self.group_key in X.columns) else None

        for col in self.datetime_cols_:
            dt = self._coerce_dt(X[col])
            # basic date parts (Int64 to keep NaNs)
            out[f"{col}_year"] = dt.dt.year.astype("Int64")
            out[f"{col}_month"] = dt.dt.month.astype("Int64")
            out[f"{col}_day"] = dt.dt.day.astype("Int64")
            out[f"{col}_weekday"] = dt.dt.weekday.astype("Int64")
            out[f"{col}_quarter"] = dt.dt.quarter.astype("Int64")
            out[f"{col}_hour"] = dt.dt.hour.astype("Int64")
            out[f"{col}_dayofyear"] = dt.dt.dayofyear.astype("Int64")

            # ISO week is safer: weeks can be 1..53
            iso = dt.dt.isocalendar()
            out[f"{col}_isoweek"] = iso.week.astype("Int64")
            out[f"{col}_isoyear"] = iso.year.astype("Int64")

            if self.include_flags:
                out[f"{col}_is_weekend"] = (dt.dt.weekday >= 5).astype("Int64")
                out[f"{col}_is_month_start"] = dt.dt.is_month_start.astype("Int64")
                out[f"{col}_is_month_end"] = dt.dt.is_month_end.astype("Int64")
                out[f"{col}_is_quarter_start"] = dt.dt.is_quarter_start.astype("Int64")
                out[f"{col}_is_quarter_end"] = dt.dt.is_quarter_end.astype("Int64")
                out[f"{col}_is_year_start"] = dt.dt.is_year_start.astype("Int64")
                out[f"{col}_is_year_end"] = dt.dt.is_year_end.astype("Int64")

                if self._has_holidays:
                    # holiday flag by date in local calendar (assume UTC dates; if you have a tz column, localize before fit/transform)
                    dates = dt.dt.date.astype("object")
                    out[f"{col}_is_holiday"] = pd.Series(
                        [
                            d in self._holiday_set if pd.notnull(d) else pd.NA
                            for d in dates
                        ],
                        index=X.index,
                        dtype="Int64",
                    )

            # elapsed (fixed anchors learned in fit)
            if self.include_elapsed:
                if gkey and col in self._anchors_group_:
                    # per-row anchor from its group, fallback to global anchor
                    anchors = pd.Series(
                        X[gkey].map(self._anchors_group_[col]), index=X.index
                    )
                    anchors = anchors.fillna(self._anchors_global_[col])
                    elapsed = (dt - anchors).dt.total_seconds()
                else:
                    elapsed = (dt - self._anchors_global_[col]).dt.total_seconds()

                out[f"{col}_elapsed_seconds"] = pd.Series(
                    elapsed, index=X.index
                ).astype("Int64")
                # day resolution as well
                out[f"{col}_elapsed_days"] = (
                    out[f"{col}_elapsed_seconds"] // 86400
                ).astype("Int64")

            # cyclical encodings (DST-safe variants)
            if self.include_cyclical:
                # hour-of-week in [0, 167] is less aliasing than hour + weekday
                how = dt.dt.weekday * 24 + dt.dt.hour
                s, c = self._cyc(how.astype(float).values, period=24.0 * 7.0)
                out[f"{col}_hourweek_sin"] = s
                out[f"{col}_hourweek_cos"] = c

                # day-of-year with 365.2425 for leap year smoothness
                doy = dt.dt.dayofyear.astype(float).values
                s, c = self._cyc(doy, period=365.2425)
                out[f"{col}_dayofyear_sin"] = s
                out[f"{col}_dayofyear_cos"] = c

                # month-of-year (1..12)
                moy = dt.dt.month.astype(float).values
                s, c = self._cyc(moy, period=12.0)
                out[f"{col}_month_sin"] = s
                out[f"{col}_month_cos"] = c

                # ISO week-of-year (1..53)
                woy = iso.week.astype(float).values
                s, c = self._cyc(woy, period=53.0)
                out[f"{col}_isoweek_sin"] = s
                out[f"{col}_isoweek_cos"] = c

        return pd.DataFrame(out, index=X.index)


