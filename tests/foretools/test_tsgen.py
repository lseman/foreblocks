import unittest

import numpy as np
import pandas as pd

from foretools.tsgen import TimeSeriesGenerator


class TestTimeSeriesGenerator(unittest.TestCase):
    def test_import_from_package(self):
        self.assertEqual(TimeSeriesGenerator.__name__, "TimeSeriesGenerator")

    def test_make_returns_dataframe_and_meta(self):
        gen = TimeSeriesGenerator(random_state=42)
        df, meta = gen.make(
            n_series=2,
            n_steps=20,
            freq="D",
            trend={"type": "linear", "slope": 0.1, "intercept": 1.0},
            seasonality=[{"period": 7.0, "amplitude": 2.0}],
            noise={"sigma": 0.5},
            return_components=True,
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(meta, dict)
        self.assertEqual(df.shape[0], 40)
        self.assertTrue({"series", "time", "y"}.issubset(df.columns))
        self.assertIn("components", meta)
        self.assertIn("trend", meta["components"])
        self.assertEqual(meta["components"]["trend"].shape, (2, 20))

    def test_make_train_ready_splits(self):
        gen = TimeSeriesGenerator(random_state=0)
        ds = gen.make_train_ready(
            n_series=2,
            n_steps=24,
            horizon=4,
            trend={"type": "linear", "slope": 0.05},
            seasonality=[{"period": 7.0, "amplitude": 1.0}],
            noise={"sigma": 0.3},
        )

        self.assertEqual(set(ds.keys()), {"train", "val", "test", "meta"})
        self.assertEqual(ds["train"].shape[0], 16)
        self.assertEqual(ds["val"].shape[0], 4)
        self.assertEqual(ds["test"].shape[0], 4)
        self.assertEqual(ds["train"].shape[1], 2)
        self.assertIsInstance(ds["meta"], dict)


if __name__ == "__main__":
    unittest.main()
