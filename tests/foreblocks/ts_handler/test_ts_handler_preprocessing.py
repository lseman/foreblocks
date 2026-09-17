import unittest

import numpy as np
import pandas as pd

from foreblocks.ts_handler import preprocessing as prep


class TestTimeSeriesPreprocessing(unittest.TestCase):
    def test_create_sequences_returns_window_major_layout(self):
        handler = prep.TimeSeriesHandler(window_size=3, horizon=2)
        data = np.arange(20, dtype=float).reshape(10, 2)

        X, y, tf = handler._create_sequences(data)

        self.assertIsNone(tf)
        self.assertEqual(X.shape, (6, 3, 2))
        self.assertEqual(y.shape, (6, 2, 2))
        np.testing.assert_array_equal(
            X[0],
            np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float),
        )
        np.testing.assert_array_equal(
            y[0],
            np.array([[6.0, 7.0], [8.0, 9.0]], dtype=float),
        )

    def test_auto_tune_short_series_avoids_saits_and_imputes(self):
        data = np.array(
            [
                [1.0, np.nan],
                [2.0, 2.0],
                [np.nan, 3.0],
                [4.0, np.nan],
                [5.0, 5.0],
                [6.0, 6.0],
            ],
            dtype=float,
        )
        handler = prep.TimeSeriesHandler(window_size=2, horizon=1, self_tune=True)

        X, y, processed, tf = handler.fit_transform(data)

        self.assertIsNone(tf)
        self.assertTrue(handler.apply_imputation)
        self.assertIn(handler.impute_method, {"interpolate", "ffill"})
        self.assertEqual(int(np.isnan(processed).sum()), 0)
        self.assertEqual(X.shape, (4, 2, 2))
        self.assertEqual(y.shape, (4, 1, 2))

    def test_apply_ewt_uses_imported_parallel_helper(self):
        captured = {}
        original = prep.apply_ewt_and_detrend_parallel

        def fake_apply(data, ewt_bands, detrend, trend_imf_idx):
            captured["args"] = (data.copy(), ewt_bands, detrend, trend_imf_idx)
            trend = np.ones_like(data)
            return data + 1.0, ["ewt"], ["bounds"], trend

        prep.apply_ewt_and_detrend_parallel = fake_apply
        try:
            handler = prep.TimeSeriesHandler(apply_ewt=True, detrend=True, ewt_bands=4)
            data = np.arange(12, dtype=float).reshape(6, 2)

            out = handler._apply_ewt_and_detrend(data)

            self.assertEqual(captured["args"][1:], (4, True, 0))
            np.testing.assert_array_equal(out, data + 1.0)
            self.assertEqual(handler.ewt_components, ["ewt"])
            self.assertEqual(handler.ewt_boundaries, ["bounds"])
            np.testing.assert_array_equal(handler.trend_component, np.ones_like(data))
        finally:
            prep.apply_ewt_and_detrend_parallel = original

    def test_auto_configure_uses_auto_filter_style_selection(self):
        original = prep.TimeSeriesHandler._auto_select_filter_method

        def fake_selector(self, data, stats):
            return {
                "best_method": "wiener",
                "apply_filter": True,
                "best_score": 0.1,
                "none_score": 0.4,
                "improvement": 0.3,
                "weights": {},
                "scores": pd.DataFrame(
                    {
                        "score": [0.1, 0.4],
                        "fidelity_mse": [0.2, 0.0],
                        "roughness": [0.1, 0.5],
                        "residual_autocorr": [0.1, 0.4],
                        "derivative_corr": [0.9, 1.0],
                    },
                    index=["wiener", "none"],
                ),
            }

        prep.TimeSeriesHandler._auto_select_filter_method = fake_selector
        try:
            rng = np.random.default_rng(0)
            t = np.linspace(0, 8 * np.pi, 256)
            data = (np.sin(t) + 0.35 * rng.normal(size=t.shape[0])).reshape(-1, 1)
            handler = prep.TimeSeriesHandler(self_tune=True)

            handler.auto_configure(data, verbose=False)

            self.assertEqual(handler.filter_method, "wiener")
            self.assertTrue(handler.apply_filter)
            self.assertEqual(handler.filter_selection_["best_method"], "wiener")
        finally:
            prep.TimeSeriesHandler._auto_select_filter_method = original

    def test_generate_time_features_defaults_to_cyclical_encoding(self):
        handler = prep.TimeSeriesHandler(generate_time_features=True)
        timestamps = pd.date_range("2026-01-01", periods=5, freq="h")

        feats = handler._generate_time_features(timestamps.to_numpy())

        self.assertEqual(feats.shape, (5, 8))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_generate_time_features_supports_legacy_mode(self):
        handler = prep.TimeSeriesHandler(
            generate_time_features=True, time_feature_mode="legacy"
        )
        timestamps = pd.date_range("2026-01-01", periods=4, freq="D")

        feats = handler._generate_time_features(timestamps.to_numpy())

        self.assertEqual(feats.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_auto_imputation_can_escalate_to_saits_for_long_gaps(self):
        original = prep.SAITSImputer

        class FakeSAITSImputer:
            def __init__(self, seq_len, epochs):
                self.seq_len = seq_len
                self.epochs = epochs

            def fit(self, data):
                self.seen = np.asarray(data, dtype=float)

            def impute(self, data):
                arr = np.asarray(data, dtype=float).copy()
                arr[np.isnan(arr)] = -1.0
                return arr

        prep.SAITSImputer = FakeSAITSImputer
        try:
            data = np.arange(200, dtype=float).reshape(100, 2)
            data[20:45, 0] = np.nan
            handler = prep.TimeSeriesHandler(
                window_size=12, horizon=3, impute_method="auto"
            )

            imputed = handler._impute_missing(data)

            self.assertEqual(handler._resolve_imputation_method(data), "saits")
            self.assertEqual(int(np.isnan(imputed).sum()), 0)
            self.assertEqual(float(imputed[20, 0]), -1.0)
        finally:
            prep.SAITSImputer = original


if __name__ == "__main__":
    unittest.main()
