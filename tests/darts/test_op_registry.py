"""Tests for the central operation registry (op_registry.py).

Guards against the metadata-drift bug this registry was introduced to fix:
SwiGLU/GeGLU/GatedGELU were reachable through ``MixedOp.op_map`` but were
missing from ``MixedOp.op_efficiency`` and from ``config.py``'s
``DEFAULT_OPS``/``DEFAULT_OP_FAMILIES`` entirely, because those were four
independent literals instead of one source of truth.
"""

import unittest

import torch

from darts.architecture.darts.mixed_op import MixedOp
from darts.architecture.ops.registry import DEFAULT_OP_NAMES, FAMILY_TO_OPS, OP_REGISTRY, build_op
from darts.config import DEFAULT_OP_FAMILIES, DEFAULT_OPS


class TestOpRegistryConsistency(unittest.TestCase):
    def test_registry_nonempty_and_families_cover_all_ops(self):
        self.assertGreater(len(OP_REGISTRY), 0)
        all_family_ops = {op for ops in FAMILY_TO_OPS.values() for op in ops}
        self.assertEqual(all_family_ops, set(OP_REGISTRY.keys()))

    def test_every_op_has_valid_efficiency_prior(self):
        for name, spec in OP_REGISTRY.items():
            self.assertTrue(
                0.0 <= spec.efficiency <= 1.0,
                f"{name} has out-of-range efficiency {spec.efficiency}",
            )

    def test_config_default_ops_matches_registry(self):
        # config.DEFAULT_OPS must contain exactly the registry's op names —
        # this is the assertion that would have caught SwiGLU/GeGLU/GatedGELU
        # being absent from config.py's search-space defaults.
        self.assertEqual(set(DEFAULT_OPS), set(DEFAULT_OP_NAMES))
        self.assertEqual(set(DEFAULT_OPS), set(OP_REGISTRY.keys()))

    def test_config_op_families_only_reference_known_ops(self):
        family_ops = {op for ops in DEFAULT_OP_FAMILIES.values() for op in ops}
        self.assertTrue(family_ops.issubset(OP_REGISTRY.keys()))
        # Every non-Identity op must be reachable through some family, or
        # candidate sampling (search/candidate_config.py) can never select it.
        non_identity_ops = set(OP_REGISTRY.keys()) - {"Identity"}
        self.assertEqual(family_ops, non_identity_ops)

    def test_gated_ffn_ops_have_tuned_efficiency(self):
        # Previously missing from op_efficiency entirely (silently fell back
        # to the generic 0.5 default wherever read defensively).
        for name in ("SwiGLU", "GeGLU", "GatedGELU"):
            self.assertIn(name, OP_REGISTRY)
            self.assertNotEqual(OP_REGISTRY[name].efficiency, 0.5)

    def test_ssm_family_has_a_member(self):
        # search/candidate_config.py special-cases "ssm" as a family kept
        # even when empty; it should no longer be empty.
        self.assertIn("ssm", FAMILY_TO_OPS)
        self.assertGreater(len(FAMILY_TO_OPS["ssm"]), 0)

    def test_mixed_op_derives_from_registry(self):
        m = MixedOp(input_dim=4, latent_dim=16, seq_length=24, available_ops=None)
        self.assertEqual(set(m.op_map.keys()), set(OP_REGISTRY.keys()))
        self.assertEqual(
            {op for ops in m.operation_groups.values() for op in ops},
            set(OP_REGISTRY.keys()),
        )
        for name in OP_REGISTRY:
            self.assertIn(name, m.op_efficiency)

    def test_build_op_instantiates_and_runs(self):
        for name in OP_REGISTRY:
            op = build_op(name, input_dim=3, latent_dim=8, seq_length=16)
            x = torch.randn(2, 16, 3)
            y = op(x)
            self.assertEqual(y.shape, (2, 16, 8), msg=f"op={name}")


if __name__ == "__main__":
    unittest.main()
