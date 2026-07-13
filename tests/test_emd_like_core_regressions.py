from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("pyfftw")

from foretools.emd_like.config import VMDOptions, VMDParameters
from foretools.emd_like.core import VMDCore
from foretools.emd_like.pipeline import FastVMD, VMDOptimizer
from foretools.emd_like.support.fft import FFTWManager


@pytest.fixture(scope="module")
def core(tmp_path_factory: pytest.TempPathFactory) -> VMDCore:
    wisdom = tmp_path_factory.mktemp("vmd") / "wisdom.dat"
    return VMDCore(FFTWManager(str(wisdom), num_threads=1))


def _decompose(core: VMDCore, signal: np.ndarray, **kwargs):
    params = {
        "alpha": 1000.0,
        "tau": 0.0,
        "K": 2,
        "DC": 0,
        "init": 1,
        "tol": 1e-5,
        "max_iter": 40,
        "boundary_method": "none",
    }
    params.update(kwargs)
    return core.decompose(signal, **params)


def test_odd_length_is_preserved(core: VMDCore) -> None:
    n = 129
    x = np.sin(2.0 * np.pi * 0.1 * np.arange(n))
    modes, spectrum, omega = _decompose(core, x)
    assert modes.shape == (2, n)
    assert spectrum.shape == (n, 2)
    assert omega.shape == (2,)
    assert np.all(np.isfinite(modes))


@pytest.mark.parametrize(
    "signal, message",
    [
        (np.array([]), "at least two"),
        (np.array([1.0]), "at least two"),
        (np.array([1.0, np.nan]), "finite"),
        (np.ones((2, 2)), "one-dimensional"),
    ],
)
def test_invalid_signals_are_rejected(
    core: VMDCore, signal: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _decompose(core, signal)


def test_warm_start_shapes_are_validated(core: VMDCore) -> None:
    x = np.sin(2.0 * np.pi * 0.1 * np.arange(64))
    bad_state = {
        "omega": np.zeros(1),
        "u_hat": np.zeros((64, 2), dtype=np.complex128),
    }
    with pytest.raises(ValueError, match="warm_start_state shapes"):
        _decompose(core, x, init=5, warm_start_state=bad_state)


def test_explicit_nondefault_keyword_wins_over_options(core: VMDCore) -> None:
    x = np.sin(2.0 * np.pi * 0.1 * np.arange(64))
    options = VMDOptions(boundary_method="mirror")
    modes, spectrum, _ = _decompose(
        core, x, boundary_method="none", options=options, max_iter=1
    )
    assert modes.shape[1] == x.size
    assert spectrum.shape[0] == x.size


def test_cache_key_separates_signal_rate_and_configuration() -> None:
    p1 = VMDParameters(max_iter=10)
    p2 = VMDParameters(max_iter=11)
    x = np.arange(16.0)
    key = VMDOptimizer._candidate_cache_key
    assert key(x, 1.0, p1, 2, 10) != key(x + 1.0, 1.0, p1, 2, 10)
    assert key(x, 1.0, p1, 2, 10) != key(x, 2.0, p1, 2, 10)
    assert key(x, 1.0, p1, 2, 10) != key(x, 1.0, p2, 2, 10)


def test_fft_thread_count_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOREBLOCKS_FFT_THREADS", "999999")
    resolved = FFTWManager._resolve_num_threads(None)
    assert 1 <= resolved <= (__import__("os").cpu_count() or 1)


def test_clear_cache_clears_univariate_and_multivariate() -> None:
    facade = FastVMD.__new__(FastVMD)
    facade.opt = type("Optimizer", (), {"_cache": {1: 1}, "_cache_mv": {2: 2}})()
    facade.clear_cache()
    assert facade.opt._cache == {}
    assert facade.opt._cache_mv == {}
