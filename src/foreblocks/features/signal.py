"""Hand-engineered amplitude, spectral, wavelet and pulse features.

Vectorized across series: every stage runs on the whole batch except peak
picking, which stays per series. Multivariate input `[N, C, L]` is
featurized per channel and concatenated channel-major.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pywt
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

from foreblocks.features._validation import as_panel

AMPLITUDE_FEATURES = ("log_rms", "log_peak", "crest_factor", "kurtosis", "skewness")
SPECTRAL_FEATURES = (
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_rolloff_hz",
    "dominant_freq_hz",
)
PULSE_FEATURES = ("pulse_rate_hz", "pulse_occupancy", "pulse_gap_cv")


def _format_hz(hz: float) -> str:
    return f"{hz / 1000:g}k" if hz >= 1000 else f"{hz:g}"


class SignalFeatures(TransformerMixin, BaseEstimator):
    """Per-series engineered features, in this order:

    - amplitude: log1p RMS, log1p peak, crest factor, kurtosis (Fisher),
      skewness, all on the median-centered series
    - spectral (Welch PSD): centroid, bandwidth, normalized entropy,
      flatness, `rolloff`-quantile frequency, dominant frequency
    - band power fractions for each `bands` `[lo, hi)` Hz range
    - wavelet (`wavelet`, `wavelet_level` levels): normalized entropy and
      relative energy per coefficient band (approximation first)
    - pulse: rate (Hz) and occupancy of samples above `pulse_threshold`
      robust (MAD) noise floors, and the coefficient of variation of the
      gaps between peaks at least `pulse_min_distance_s` apart

    `bands` defaults to four equal-width bands spanning 0 to Nyquist. The
    transform is stateless; `fit` only validates parameters.
    """

    def __init__(
        self,
        fs: float,
        bands: Sequence[tuple[float, float]] | None = None,
        band_names: Sequence[str] | None = None,
        nperseg: int = 4096,
        rolloff: float = 0.95,
        wavelet: str = "db4",
        wavelet_level: int = 4,
        pulse_threshold: float = 6.0,
        pulse_min_distance_s: float = 0.0005,
    ):
        self.fs = fs
        self.bands = bands
        self.band_names = band_names
        self.nperseg = nperseg
        self.rolloff = rolloff
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.pulse_threshold = pulse_threshold
        self.pulse_min_distance_s = pulse_min_distance_s

    def _bands(self) -> list[tuple[float, float]]:
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("fs must be positive and finite.")
        if self.bands is not None:
            return [(float(lo), float(hi)) for lo, hi in self.bands]
        edges = np.linspace(0, self.fs / 2, 5)
        edges[-1] = np.nextafter(edges[-1], np.inf)  # include Nyquist
        return list(pairwise(edges))

    def _band_names(self) -> list[str]:
        bands = self._bands()
        if self.band_names is not None:
            if len(self.band_names) != len(bands):
                raise ValueError("band_names must match bands in length.")
            return list(self.band_names)
        return [f"band_{_format_hz(lo)}_{_format_hz(hi)}" for lo, hi in bands]

    def _base_names(self) -> list[str]:
        return [
            *AMPLITUDE_FEATURES,
            *SPECTRAL_FEATURES,
            *self._band_names(),
            "wavelet_entropy",
            *(f"wavelet_energy_{i}" for i in range(self.wavelet_level + 1)),
            *PULSE_FEATURES,
        ]

    def fit(self, X=None, y=None) -> SignalFeatures:
        self._band_names()
        if X is not None:
            self.n_channels_in_ = as_panel(X, allow_empty=True).shape[1]
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = self._base_names()
        n_channels = getattr(self, "n_channels_in_", 1)
        if n_channels > 1:
            names = [f"ch{c}_{name}" for c in range(n_channels) for name in names]
        return np.asarray(names, dtype=object)

    def transform(self, X) -> np.ndarray:
        x = as_panel(X, allow_empty=True)
        n, n_channels, length = x.shape
        bands = self._bands()
        n_base = len(self._base_names())
        if n == 0:
            return np.empty((0, n_channels * n_base), dtype=np.float32)
        flat = x.reshape(n * n_channels, length).astype(np.float64)
        features = self._features(flat, bands)
        return features.reshape(n, n_channels * n_base).astype(np.float32)

    def _features(self, x: np.ndarray, bands) -> np.ndarray:
        fs = self.fs
        eps = np.finfo(float).eps
        length = x.shape[1]
        centered = x - np.median(x, axis=1, keepdims=True)
        magnitude = np.abs(centered)
        rms = np.sqrt(np.mean(centered**2, axis=1))
        peak = magnitude.max(axis=1)

        freqs, psd = signal.welch(
            centered, fs=fs, nperseg=min(self.nperseg, length), axis=-1
        )
        power = psd.sum(axis=1)
        has_power = power > 0
        p = np.divide(
            psd, power[:, None], out=np.zeros_like(psd), where=has_power[:, None]
        )
        centroid = p @ freqs
        bandwidth = np.sqrt(np.sum((freqs - centroid[:, None]) ** 2 * p, axis=1))
        entropy = -np.sum(p * np.log(p + eps), axis=1) / np.log(p.shape[1])
        flatness = np.where(
            has_power,
            np.exp(np.mean(np.log(psd + eps), axis=1))
            / np.maximum(psd.mean(axis=1), eps),
            0.0,
        )
        reached = np.cumsum(p, axis=1) >= self.rolloff
        rolloff_index = np.where(
            reached.any(axis=1), reached.argmax(axis=1), len(freqs) - 1
        )
        rolloff = np.where(has_power, freqs[rolloff_index], 0.0)
        dominant = freqs[np.argmax(psd, axis=1)]
        band_fractions = [
            p[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in bands
        ]

        coeffs = pywt.wavedec(
            centered, self.wavelet, level=self.wavelet_level, mode="periodization", axis=-1
        )
        energy = np.stack([np.einsum("ij,ij->i", c, c) for c in coeffs], axis=1)
        energy /= np.maximum(energy.sum(axis=1, keepdims=True), eps)
        wavelet_entropy = -np.sum(energy * np.log(energy + eps), axis=1) / np.log(
            energy.shape[1]
        )

        # Signal-relative floor: a fixed absolute floor erases pulses in
        # normalized or low-gain data.
        noise_floor = np.maximum(
            1.4826 * np.median(magnitude, axis=1), eps * np.maximum(peak, 1.0)
        )
        height = self.pulse_threshold * noise_floor
        pulse_occupancy = np.mean(magnitude > height[:, None], axis=1)
        distance = max(1, round(self.pulse_min_distance_s * fs))
        pulse_rate = np.empty(len(x))
        pulse_gap_cv = np.zeros(len(x))
        for i, row in enumerate(magnitude):
            peaks, _ = signal.find_peaks(row, height=height[i], distance=distance)
            pulse_rate[i] = len(peaks) * fs / length
            if len(peaks) > 2:
                gaps = np.diff(peaks) / fs
                pulse_gap_cv[i] = gaps.std() / max(gaps.mean(), eps)

        # Biased central moments, as `scipy.stats.skew`/`kurtosis` (Fisher)
        # compute them, without scipy's slow generic moment path.
        deviation = centered - centered.mean(axis=1, keepdims=True)
        squared = deviation**2
        m2 = squared.mean(axis=1)
        m3 = np.mean(squared * deviation, axis=1)
        m4 = np.mean(squared**2, axis=1)
        spread = np.sqrt(m2) > eps
        safe_m2 = np.where(spread, m2, 1.0)
        kurtosis = np.where(spread, m4 / safe_m2**2 - 3.0, 0.0)
        skewness = np.where(spread, m3 / safe_m2**1.5, 0.0)

        return np.column_stack(
            [
                np.log1p(rms),
                np.log1p(peak),
                peak / np.maximum(rms, eps),
                kurtosis,
                skewness,
                centroid,
                bandwidth,
                entropy,
                flatness,
                rolloff,
                dominant,
                *band_fractions,
                wavelet_entropy,
                energy,
                pulse_rate,
                pulse_occupancy,
                pulse_gap_cv,
            ]
        )


def extract_signal_features(x, fs: float, **params) -> np.ndarray:
    """Functional form of `SignalFeatures(fs, **params).transform(x)`."""
    return SignalFeatures(fs, **params).transform(x)
