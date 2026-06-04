import unittest

import torch

from foreblocks.models.popular.crossformer import CrossFormerHeadCustom
from foreblocks.models.popular.etsformer import ETSformerHeadCustom
from foreblocks.models.popular.informer import InformerHeadCustom
from foreblocks.models.popular.timesnet import TimesNetHeadCustom


class TestPopularHeads(unittest.TestCase):
    def test_informer_forward(self):
        torch.manual_seed(0)
        model = InformerHeadCustom(
            pred_len=12,
            in_channels=3,
            out_channels=3,
            label_len=0,
            d_model=64,
            n_heads=4,
            n_layers_enc=1,
            n_layers_dec=1,
            dim_feedforward=128,
            dropout=0.0,
        )
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))

    def test_etsformer_forward(self):
        torch.manual_seed(0)
        model = ETSformerHeadCustom(
            pred_len=12,
            in_channels=3,
            out_channels=3,
            d_model=64,
            n_heads=4,
            n_layers=1,
            dim_feedforward=128,
            dropout=0.0,
            ma_kernel=5,
        )
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))

    def test_timesnet_forward(self):
        torch.manual_seed(0)
        model = TimesNetHeadCustom(pred_len=12, in_channels=3, out_channels=3, d_model=64, n_blocks=1)
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))

    def test_crossformer_forward(self):
        torch.manual_seed(0)
        model = CrossFormerHeadCustom(
            pred_len=12,
            in_channels=3,
            out_channels=3,
            d_model=64,
            n_heads=4,
            hidden=128,
            dropout=0.0,
        )
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))


if __name__ == "__main__":
    unittest.main()
