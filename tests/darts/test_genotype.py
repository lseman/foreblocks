"""Tests for the Genotype abstraction (architecture/genotype.py).

Covers the gap ``derive_final_architecture`` previously left open: there was
no serializable record of which op was chosen per cell edge, so a found
architecture could not be exported, diffed, or rebuilt without re-running
the bilevel search.
"""

import unittest

import torch

from darts.architecture.common.inspector import ArchitectureInspector
from darts.architecture.common.norms import NORM_NAMES
from darts.architecture.darts.finalization import (
    derive_final_architecture,
    select_architecture,
)
from darts.architecture.darts.genotype import (
    Genotype,
    build_model_from_genotype,
    resolve_op_name,
)
from darts.architecture.darts.time_series_darts import TimeSeriesDARTS
from darts.architecture.ops.registry import OP_REGISTRY, build_op
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
            ops = [type(edge.op) for edge in cell.edges]
            expected = [
                type(build_op(e.operation, 3, 16, 12)) for e in cell_geno.edges
            ]
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


# Ops whose class name minus "Op" differs from their registry key; recording
# the class-derived name made these genotypes impossible to rebuild.
_RENAMED_OPS = ["TimeMixer", "SwiGLU", "GeGLU", "GatedGELU"]


def _pin_norm(model: TimeSeriesDARTS, name: str) -> None:
    with torch.no_grad():
        model.norm_alpha.fill_(-5.0)
        model.norm_alpha[NORM_NAMES.index(name)] = 5.0


class TestNormalizationModes(unittest.TestCase):
    def test_every_norm_mode_is_recorded_by_its_own_name(self):
        for name in NORM_NAMES:
            with self.subTest(norm=name):
                model = _make_model(10)
                _pin_norm(model, name)
                self.assertEqual(select_architecture(model).norm, name)

    def test_rms_norm_survives_derive_and_rebuild(self):
        source = _make_model(11)
        _pin_norm(source, "rms_norm")
        fixed, genotype = derive_final_architecture(source, return_genotype=True)
        self.assertEqual(fixed.selected_norm, "rms_norm")
        self.assertEqual(ArchitectureInspector(source).extract_normalization(), "rms_norm")

        rebuilt = build_model_from_genotype(_make_model(12), genotype)
        self.assertEqual(rebuilt.selected_norm, "rms_norm")
        self.assertEqual(
            int(rebuilt.norm_alpha.argmax()), NORM_NAMES.index("rms_norm")
        )

    def test_unknown_norm_in_genotype_raises(self):
        source = _make_model(13)
        genotype = select_architecture(source)
        genotype.norm = "norm_3"
        with self.assertRaises(ValueError):
            build_model_from_genotype(_make_model(14), genotype)


class TestOperationNames(unittest.TestCase):
    def _renamed_op_model(self, seed: int) -> TimeSeriesDARTS:
        return _make_model(seed, num_nodes=4, selected_ops=list(_RENAMED_OPS))

    def test_genotype_records_registry_names(self):
        model = self._renamed_op_model(20)
        genotype = select_architecture(model)
        recorded = {e.operation for c in genotype.cells for e in c.edges}
        self.assertTrue(recorded)
        self.assertTrue(recorded <= set(_RENAMED_OPS), recorded)

    def test_renamed_ops_rebuild_from_json(self):
        source = self._renamed_op_model(21)
        _, genotype = derive_final_architecture(source, return_genotype=True)
        restored = Genotype.from_json(genotype.to_json())
        rebuilt = build_model_from_genotype(self._renamed_op_model(22), restored)
        rebuilt.eval()
        with torch.no_grad():
            y = rebuilt(torch.randn(2, 12, 3))
        self.assertEqual(tuple(y.shape), (2, 6, 3))

    def test_legacy_class_derived_names_still_resolve(self):
        for name in OP_REGISTRY:
            legacy = type(build_op(name, 3, 16, 12)).__name__.removesuffix("Op")
            self.assertEqual(resolve_op_name(legacy), name)
        with self.assertRaises(ValueError):
            resolve_op_name("NotAnOp")


class TestSelectionIsSeparateFromConversion(unittest.TestCase):
    def test_select_architecture_does_not_modify_model(self):
        model = _make_model(30, num_cells=2)
        model.train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        edge_types = [type(e) for c in model.cells for e in c.edges]

        select_architecture(model)

        self.assertEqual([type(e) for c in model.cells for e in c.edges], edge_types)
        self.assertFalse(hasattr(model, "selected_norm"))
        self.assertTrue(all(m.training for m in model.modules()))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)

    def test_derive_matches_select_and_keeps_trained_op_weights(self):
        source = _make_model(31)
        expected = select_architecture(source)
        fixed, genotype = derive_final_architecture(source, return_genotype=True)
        self.assertEqual(genotype.to_dict(), expected.to_dict())

        for src_cell, fixed_cell, cell_geno in zip(
            source.cells, fixed.cells, genotype.cells
        ):
            for src_edge, fixed_edge, edge_geno in zip(
                src_cell.edges, fixed_cell.edges, cell_geno.edges
            ):
                trained = src_edge.ops[src_edge.available_ops.index(edge_geno.operation)]
                src_state = trained.state_dict()
                for k, v in fixed_edge.op.state_dict().items():
                    self.assertTrue(torch.equal(v, src_state[k]), k)


if __name__ == "__main__":
    unittest.main()
