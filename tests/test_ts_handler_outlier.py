import unittest
from unittest.mock import patch

import numpy as np
import torch

from foreblocks.ts_handler import outlier


class TestTranADOutlier(unittest.TestCase):
    def test_tranad_forward_predicts_last_step_only(self):
        model = outlier.TranAD(feats=3, window_size=5, n_layers=1, dropout=0.0)
        src = torch.rand(4, 5, 3)
        tgt = src[:, -1:, :]

        out1, out2 = model(src, tgt)

        self.assertEqual(tuple(out1.shape), (4, 1, 3))
        self.assertEqual(tuple(out2.shape), (4, 1, 3))

    def test_remove_outliers_tranad_uses_raw_series_and_maps_window_end_scores(self):
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
                [7.0, 7.0],
                [8.0, 8.0],
                [9.0, 9.0],
            ],
            dtype=float,
        )
        captured = {}

        class FakeDetector:
            def __init__(self, *args, **kwargs):
                captured["kwargs"] = kwargs

            def fit_predict(self, series, validation_split=0.2):
                captured["series"] = np.array(series, copy=True)
                return np.array(
                    [
                        [0.0, 0.0],
                        [10.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 8.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    dtype=float,
                )

        with patch.object(outlier, "TranADDetector", FakeDetector):
            cleaned = outlier._remove_outliers(
                data,
                "tranad",
                3.0,
                seq_len=3,
                adaptive=True,
                z_threshold=1.0,
            )

        np.testing.assert_array_equal(captured["series"], data)
        self.assertEqual(captured["kwargs"]["scaler_type"], "minmax")
        self.assertTrue(np.isnan(cleaned[3]).all())
        self.assertTrue(np.isnan(cleaned[5]).all())
        self.assertFalse(np.isnan(cleaned[2]).any())
        self.assertFalse(np.isnan(cleaned[4]).any())

    def test_repo_anomaly_score_averages_both_passes(self):
        x1 = torch.tensor([[[2.0, 1.0]]], dtype=torch.float32)
        x2 = torch.tensor([[[0.0, 3.0]]], dtype=torch.float32)
        target = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)

        scores = outlier.TranADDetector._compute_anomaly_scores(x1, x2, target)

        np.testing.assert_allclose(scores, np.array([[1.0, 2.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
