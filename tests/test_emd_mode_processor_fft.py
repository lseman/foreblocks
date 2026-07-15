import numpy as np
import pytest

from foretools.emd_like.analysis.mode_processor import ModeProcessor


def test_dominant_frequency_uses_numpy_fft_without_pyfftw() -> None:
    fs = 256.0
    time = np.arange(256) / fs
    signal = np.sin(2 * np.pi * 32.0 * time)
    assert ModeProcessor.dominant_frequency(signal, fs) == pytest.approx(32.0)


def test_batched_cost_matches_individual_entropy_spectra() -> None:
    fs = 256.0
    time = np.arange(256) / fs
    modes = [
        np.sin(2 * np.pi * 16.0 * time),
        0.5 * np.sin(2 * np.pi * 48.0 * time),
    ]
    signal = np.sum(modes, axis=0)
    cost = ModeProcessor.cost_signal(modes, signal, fs)
    assert np.isfinite(cost)
    assert cost >= 0.0


def test_torch_fft_backend_matches_numpy() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    signal = rng.normal(size=257)
    expected = ModeProcessor._rfft(signal, backend="numpy")
    actual = ModeProcessor._rfft(signal, backend="torch", device="cpu")
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
