"""Engineered signal features: names, layout, and physical sanity."""

import numpy as np
import pytest
from scipy import stats

from foreblocks.features import SignalFeatures, extract_signal_features

FS = 8_000


def _tones(freqs, length=2_000, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / FS
    return np.stack([np.sin(2 * np.pi * f * t) + rng.normal(scale=noise, size=length) for f in freqs])


def test_names_align_with_columns_and_default_bands_cover_nyquist():
    extractor = SignalFeatures(fs=FS)
    names = list(extractor.get_feature_names_out())
    features = extractor.transform(_tones([500, 3_000]))
    assert features.shape == (2, len(names)) and len(set(names)) == len(names)
    assert names[11:15] == ["band_0_1k", "band_1k_2k", "band_2k_3k", "band_3k_4k"]
    band_total = features[:, 11:15].sum(axis=1)
    np.testing.assert_allclose(band_total, 1.0, atol=1e-6)


def test_spectral_features_track_the_tone():
    x = _tones([500, 3_000])
    features = SignalFeatures(fs=FS).transform(x)
    names = list(SignalFeatures(fs=FS).get_feature_names_out())
    dominant = features[:, names.index("dominant_freq_hz")]
    np.testing.assert_allclose(dominant, [500, 3_000], atol=FS / 2_000)
    bands = features[:, names.index("band_0_1k") : names.index("band_3k_4k") + 1]
    assert bands.argmax(axis=1).tolist() == [0, 3]


def test_moments_match_scipy():
    x = np.random.default_rng(0).gamma(2.0, size=(3, 1_000))
    features = extract_signal_features(x, FS)
    names = list(SignalFeatures(fs=FS).get_feature_names_out())
    centered = x - np.median(x, axis=1, keepdims=True)
    np.testing.assert_allclose(features[:, names.index("kurtosis")], stats.kurtosis(centered, axis=1), rtol=1e-4)
    np.testing.assert_allclose(features[:, names.index("skewness")], stats.skew(centered, axis=1), rtol=1e-4)


def test_multichannel_is_channel_major_concatenation():
    x = _tones([500, 1_500, 2_500, 3_500]).reshape(2, 2, -1)
    extractor = SignalFeatures(fs=FS, wavelet_level=3).fit(x)
    features = extractor.transform(x)
    names = extractor.get_feature_names_out()
    per_channel = SignalFeatures(fs=FS, wavelet_level=3).transform(x.reshape(4, -1))
    np.testing.assert_allclose(features, per_channel.reshape(2, -1))
    assert names[0] == "ch0_log_rms" and names[len(names) // 2] == "ch1_log_rms"
    assert features.shape[1] == len(names)


def test_custom_bands_names_and_validation():
    extractor = SignalFeatures(fs=FS, bands=[(0, 1_000), (1_000, 4_001)], band_names=["low", "high"])
    names = list(extractor.get_feature_names_out())
    assert "low" in names and "high" in names
    assert extractor.transform(np.empty((0, 64))).shape == (0, len(names))
    with pytest.raises(ValueError, match="band_names"):
        SignalFeatures(fs=FS, bands=[(0, 1)], band_names=["a", "b"]).fit()
    with pytest.raises(ValueError, match="fs"):
        SignalFeatures(fs=0).transform(_tones([500]))
