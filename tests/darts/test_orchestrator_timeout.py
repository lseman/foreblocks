"""Tests for the phase-1 candidate-collection timeout being configurable.

Previously ``candidate_timeout`` was a hardcoded 120.0 constant buried in
``orchestrator.py``/``trainer.py`` with no way to tune it for a larger
search space or epoch budget, and ``MultiFidelitySearchConfig`` had no
corresponding field at all.
"""

import time
import unittest

from darts.config import MultiFidelitySearchConfig
from darts.search.orchestrator import run_parallel_candidate_collection


class TestMultiFidelitySearchConfigTimeout(unittest.TestCase):
    def test_default_matches_historical_constant(self):
        self.assertEqual(MultiFidelitySearchConfig().candidate_timeout, 120.0)

    def test_field_is_overridable(self):
        cfg = MultiFidelitySearchConfig(candidate_timeout=5.0)
        self.assertEqual(cfg.candidate_timeout, 5.0)


class TestRunParallelCandidateCollectionTimeout(unittest.TestCase):
    def test_short_timeout_yields_no_results(self):
        # ThreadPoolExecutor cannot cancel a thread already mid-sleep, so a
        # short candidate_timeout doesn't cut wall time short — it bounds
        # which results get *collected* (none, here, since nothing finishes
        # inside the budget).
        def slow_candidate(cid: int) -> dict:
            time.sleep(0.3)
            return {"success": True, "candidate_id": cid, "score": 1.0}

        results = run_parallel_candidate_collection(
            num_candidates=3,
            candidate_fn=slow_candidate,
            max_workers=3,
            candidate_timeout=0.02,  # total budget = 0.02 * 3 = 0.06s
        )
        self.assertEqual(results, [])

    def test_generous_timeout_collects_all_candidates(self):
        def fast_candidate(cid: int) -> dict:
            return {"success": True, "candidate_id": cid, "score": float(cid)}

        results = run_parallel_candidate_collection(
            num_candidates=4,
            candidate_fn=fast_candidate,
            max_workers=4,
            candidate_timeout=30.0,
        )
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
