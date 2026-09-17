"""Tests for the Genotype abstraction (architecture/genotype.py).

Covers the gap ``derive_final_architecture`` previously left open: there was
no serializable record of which op was chosen per cell edge, so a found
architecture could not be exported, diffed, or rebuilt without re-running
the bilevel search.
"""

import unittest

import torch

from darts.architecture.finalization import derive_final_architecture
from darts.architecture.genotype import Genotype, build_model_from_genotype
from darts.architecture.time_series_darts import TimeSeriesDARTS
from darts.utils.io import load_genotype, save_genotype


def _make_model(seed: int, **overrides) -> TimeSeriesDARTS:
    torch.manual_seed(seed)
    kwargs = dict(
        input_dim=3,
        hidden_dim=16,
        latent_dim=16,
        forecast_horizon=6,
        seq_length=12,
        num_cells=1,
        num_nodes=3,
        arch_mode="encoder_only",
    )
    kwargs.update(overrides)
    return TimeSeriesDARTS(**kwargs)


class TestDeriveFinalArchitectureReturnGenotype(unittest.TestCase):
    def test_default_return_is_unchanged(self):
        model = _make_model(0)
        result = derive_final_architecture(model)
        self.assertIsInstance(result, torch.nn.Module)

    def test_return_genotype_true_returns_tuple(self):
        model = _make_model(0)
        result = derive_final_architecture(model, return_genotype=True)
        self.assertIsInstance(result, tuple)
        fixed_model, genotype = result
        self.assertIsInstance(fixed_model, torch.nn.Module)
        self.assertIsInstance(genotype, Genotype)

    def test_genotype_records_every_cell_edge(self):
        model = _make_model(1, num_cells=2, num_nodes=3)
        _, genotype = derive_final_architecture(model, return_genotype=True)
        self.assertEqual(len(genotype.cells), 2)
        for cell in genotype.cells:
            self.assertEqual(len(cell.edges), sum(range(cell.num_nodes)))
            for edge in cell.edges:
                self.assertTrue(edge.operation)


class TestGenotypeSerialization(unittest.TestCase):
    def test_json_round_trip(self):
        model = _make_model(2)
        _, genotype = derive_final_architecture(model, return_genotype=True)
        restored = Genotype.from_json(genotype.to_json())
        self.assertEqual(restored.to_dict(), genotype.to_dict())

    def test_file_round_trip(self, tmp_path=None):
        import tempfile, os

        model = _make_model(3)
        _, genotype = derive_final_architecture(model, return_genotype=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "genotype.json")
            save_genotype(genotype, path)
            restored = load_genotype(path)
        self.assertEqual(restored.to_dict(), genotype.to_dict())


class TestBuildModelFromGenotype(unittest.TestCase):
    def test_rebuilt_model_matches_recorded_ops_exactly(self):
        source = _make_model(4)
        _, genotype = derive_final_architecture(source, return_genotype=True)

        fresh = _make_model(999)  # different seed -> different random alphas
        rebuilt = build_model_from_genotype(fresh, genotype)

        for cell, cell_geno in zip(rebuilt.cells, genotype.cells):
            ops = [type(edge.op).__name__.replace("Op", "") for edge in cell.edges]
            expected = [e.operation for e in cell_geno.edges]
            self.assertEqual(ops, expected)

    def test_rebuilt_model_forward_pass_runs(self):
        source = _make_model(5)
        _, genotype = derive_final_architecture(source, return_genotype=True)

        fresh = _make_model(1234)
        rebuilt = build_model_from_genotype(fresh, genotype)
        rebuilt.eval()

        x = torch.randn(2, 12, 3)
        with torch.no_grad():
            y = rebuilt(x)
        self.assertEqual(tuple(y.shape), (2, 6, 3))

    def test_mismatched_num_cells_raises(self):
        source = _make_model(6, num_cells=1)
        _, genotype = derive_final_architecture(source, return_genotype=True)

        mismatched = _make_model(7, num_cells=2)
        with self.assertRaises(ValueError):
            build_model_from_genotype(mismatched, genotype)


if __name__ == "__main__":
    unittest.main()
