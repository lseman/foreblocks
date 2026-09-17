"""Guards against config-dataclass defaults drifting from what the actual
search functions accept.

``RobustPoolSearchConfig.robustness_mode`` previously defaulted to
``"spearman"``, a value ``search/robust_pool.py`` has never accepted (it
only recognises ``"topk_freq"``, ``"avg_rank"``, ``"worst_rank"`` and raises
``ValueError`` otherwise) — so building a ``RobustPoolSearchConfig()`` and
threading its default straight through would have crashed. This is the same
class of bug the operation registry (op_registry.py) was introduced to fix:
a config default drifting out of sync with its consumer.
"""

import inspect
import unittest

from darts.config import RobustPoolSearchConfig
from darts.trainer import DARTSTrainer


class TestRobustPoolSearchConfigDefaults(unittest.TestCase):
    def test_robustness_mode_is_a_value_the_implementation_accepts(self):
        valid_modes = {"topk_freq", "avg_rank", "worst_rank"}
        self.assertIn(RobustPoolSearchConfig().robustness_mode, valid_modes)

    def test_robustness_mode_matches_trainer_default(self):
        sig = inspect.signature(DARTSTrainer.robust_initial_pool_over_op_pools)
        trainer_default = sig.parameters["robustness_mode"].default
        self.assertEqual(RobustPoolSearchConfig().robustness_mode, trainer_default)


if __name__ == "__main__":
    unittest.main()
