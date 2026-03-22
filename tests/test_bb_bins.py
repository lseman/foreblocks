import importlib.util

import numpy as np

from foretools.aux.bb_bins import BayesianBlocks


ASTROPY_AVAILABLE = importlib.util.find_spec("astropy") is not None

if ASTROPY_AVAILABLE:
    from astropy.stats import bayesian_blocks


def _assert_matches_astropy(t, x=None, sigma=None, **kwargs):
    ours = BayesianBlocks(**kwargs).fit_edges(t, x, sigma)
    ref = bayesian_blocks(t, x=x, sigma=sigma, **kwargs)
    assert np.allclose(ours, ref, atol=1e-10, rtol=1e-10)


def test_bb_bins_events_matches_astropy_reference():
    if not ASTROPY_AVAILABLE:
        return

    rng = np.random.default_rng(0)
    t = rng.choice(np.arange(20), size=80, replace=True).astype(float)
    _assert_matches_astropy(t, fitness="events")


def test_bb_bins_measures_matches_astropy_reference():
    if not ASTROPY_AVAILABLE:
        return

    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 10, size=40))
    x = np.sin(t) + 0.1 * rng.normal(size=t.size)
    sigma = 0.2 + 0.05 * rng.random(size=t.size)
    _assert_matches_astropy(t, x=x, sigma=sigma, fitness="measures")


def test_bb_bins_regular_events_matches_astropy_reference():
    if not ASTROPY_AVAILABLE:
        return

    rng = np.random.default_rng(2)
    t = np.arange(0, 20, 1.0)
    x = (rng.random(size=t.size) > 0.6).astype(int)
    _assert_matches_astropy(t, x=x, fitness="regular_events", dt=1.0)


def test_bb_fit_bins_caps_to_max_bins():
    class StubBayesianBlocks(BayesianBlocks):
        def fit_edges(self, t, x=None, sigma=None):
            return np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    bins = StubBayesianBlocks(fitness="events").fit_bins([0.0, 1.0], max_bins=2)
    assert bins == 2


def test_bb_fit_bins_uses_requested_fallback():
    class FailingBayesianBlocks(BayesianBlocks):
        def fit_edges(self, t, x=None, sigma=None):
            raise RuntimeError("boom")

    t = np.linspace(0.0, 1.0, 16)
    assert FailingBayesianBlocks().fit_bins(t, fallback="sturges") == 5
    assert FailingBayesianBlocks().fit_bins(t, fallback="sqrt") == 4
    assert FailingBayesianBlocks().fit_bins(t, fallback="fd") >= 1
