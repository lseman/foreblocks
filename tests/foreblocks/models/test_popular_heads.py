import unittest

import torch

from foreblocks.models.popular.autoformer import Autoformer
from foreblocks.models.popular.crossformer import CrossFormer
from foreblocks.models.popular.etsformer import ETSformer
from foreblocks.models.popular.informer import Informer
from foreblocks.models.popular.nonstationary import NonStationaryTransformer
from foreblocks.models.popular.timemixer import TimeMixer
from foreblocks.models.popular.timesnet import TimesNet
from foreblocks.models.popular.timexer import TimeXer


class TestPopularHeads(unittest.TestCase):
    def test_autoformer_forward(self):
        torch.manual_seed(0)
        model = Autoformer(
            pred_len=12,
            in_channels=3,
            out_channels=3,
            d_model=64,
            n_heads=4,
            n_layers_enc=1,
            n_layers_dec=1,
            dim_ff=128,
            dropout=0.0,
        )
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))

    def test_informer_forward(self):
        torch.manual_seed(0)
        model = Informer(
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
        model = ETSformer(
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
        model = TimesNet(pred_len=12, in_channels=3, out_channels=3, d_model=64, n_blocks=1)
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))

    def test_crossformer_forward(self):
        torch.manual_seed(0)
        model = CrossFormer(
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

    def test_timemixer_forward(self):
        torch.manual_seed(0)
        model = TimeMixer(
            pred_len=12,
            in_channels=3,
            out_channels=3,
            d_model=64,
            n_levels=3,
            n_layers_pms=2,
            dropout=0.0,
        )
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))

    def test_timemixer_quantiles(self):
        torch.manual_seed(0)
        model = TimeMixer(
            pred_len=12,
            in_channels=1,
            out_channels=1,
            d_model=64,
            quantiles=(0.1, 0.5, 0.9),
            n_levels=2,
            dropout=0.0,
        )
        x = torch.randn(2, 48, 1)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))
        qs = model.split_quantiles(y)
        self.assertEqual(len(qs), 3)

    def test_timemixer_channel_mixer(self):
        torch.manual_seed(0)
        model = TimeMixer(
            pred_len=12,
            in_channels=7,
            out_channels=1,
            d_model=64,
            n_levels=2,
            dropout=0.0,
        )
        x = torch.randn(2, 48, 7)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 1))

    def test_timexer_forward(self):
        torch.manual_seed(0)
        model = TimeXer(
            pred_len=12,
            in_channels=3,
            out_channels=3,
            d_model=64,
            n_heads=4,
            n_layers=1,
            dim_feedforward=128,
            patch_len=8,
            stride=4,
            dropout=0.0,
        )
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))

    def test_timexer_with_exog_forward(self):
        torch.manual_seed(0)
        model = TimeXer(
            pred_len=12,
            in_channels=2,
            out_channels=1,
            exog_channels=4,
            d_model=64,
            n_heads=4,
            n_layers=1,
            dim_feedforward=128,
            patch_len=8,
            stride=4,
            dropout=0.0,
        )
        x = torch.randn(2, 48, 2)
        exog = torch.randn(2, 48, 4)
        y = model(x, exog=exog)
        self.assertEqual(y.shape, (2, 12, 1))

    def test_nonstationary_transformer_forward(self):
        torch.manual_seed(0)
        model = NonStationaryTransformer(
            pred_len=12,
            in_channels=3,
            out_channels=3,
            d_model=64,
            n_heads=4,
            n_layers=1,
            d_layers=1,
            dim_ff=128,
            dropout=0.0,
            max_seq_len=64,
        )
        x = torch.randn(2, 48, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 3))


if __name__ == "__main__":
    unittest.main()
