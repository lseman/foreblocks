from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _choose_nearest_candidate(
    value: float,
    candidates: list[int],
    max_value: float = float("inf"),
) -> int:
    viable = [candidate for candidate in candidates if candidate <= max_value]
    pool = viable if viable else candidates
    return min(pool, key=lambda candidate: abs(candidate - value))


def _to_1d_array(series: Any) -> np.ndarray:
    if isinstance(series, torch.Tensor):
        array = series.detach().cpu().numpy()
    else:
        array = np.asarray(series)

    if array.ndim == 0:
        raise ValueError("series must contain at least one observation")
    if array.ndim == 1:
        values = array.astype(np.float64, copy=False)
    elif array.ndim == 2:
        if 1 in array.shape:
            values = array.reshape(-1).astype(np.float64, copy=False)
        else:
            values = array.mean(axis=-1).astype(np.float64, copy=False)
    else:
        leading = array.shape[0]
        values = array.reshape(leading, -1).mean(axis=-1).astype(np.float64, copy=False)

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError("series must contain at least one finite observation")
    return values[finite_mask]


def _standard_deviation(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=0))


def _difference(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.empty(0, dtype=np.float64)
    return np.diff(values)


def _linear_detrend(values: np.ndarray) -> tuple[np.ndarray, float]:
    if values.size <= 2:
        return values - np.mean(values), 0.0
    timeline = np.arange(values.size, dtype=np.float64)
    slope, intercept = np.polyfit(timeline, values, deg=1)
    trend = slope * timeline + intercept
    residual = values - trend
    total_var = float(np.var(values))
    residual_var = float(np.var(residual))
    trend_strength = (
        0.0 if total_var <= 1e-12 else _clamp(1.0 - residual_var / total_var, 0.0, 1.0)
    )
    return residual, trend_strength


def _autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered))
    if variance <= 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float64)
    corr = np.correlate(centered, centered, mode="full")
    mid = corr.size // 2
    acf = corr[mid : mid + max_lag + 1] / variance
    return acf.astype(np.float64, copy=False)


def _find_acf_peaks(acf: np.ndarray, max_peaks: int = 6) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    if acf.size <= 3:
        return peaks
    for lag in range(2, acf.size - 1):
        value = float(acf[lag])
        if value < 0.1:
            continue
        if value >= float(acf[lag - 1]) and value >= float(acf[lag + 1]):
            peaks.append((lag, value))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks[:max_peaks]


def _spectral_entropy(normalized_power: np.ndarray) -> float:
    if normalized_power.size == 0:
        return 1.0
    power = normalized_power[normalized_power > 0]
    if power.size == 0:
        return 1.0
    entropy = -float(np.sum(power * np.log(power)))
    return entropy / max(np.log(power.size), 1e-8)


@dataclass(frozen=True)
class SpectralPeak:
    period: int
    normalized_power: float


@dataclass(frozen=True)
class ScaleBandShare:
    band: str
    share: float
    detail: str


@dataclass(frozen=True)
class DecompositionRecommendation:
    method: str
    suitability: str
    rationale: str
    periods: tuple[int, ...] = ()


@dataclass(frozen=True)
class TransformerTuningReport:
    available: bool
    series_length: int
    patching_label: str
    patching_score: int
    recommended_patch_len: int
    recommended_patch_stride: int
    recommended_patch_set: tuple[int, ...]
    dominant_period: float
    dominant_periods: tuple[SpectralPeak, ...]
    acf_periods: tuple[int, ...]
    repeatability: float
    roughness: float
    forecastability_score: float
    spectral_entropy: float
    multiscale_label: str
    multiscale_score: int
    hierarchical_label: str
    hierarchical_score: int
    recommended_hierarchical_periods: tuple[int, ...]
    scale_band_shares: tuple[ScaleBandShare, ...]
    recommended_attention_mode: str
    recommended_preprocessing_heads: tuple[str, ...]
    recommended_decompositions: tuple[DecompositionRecommendation, ...]
    recommended_model_kwargs: dict[str, Any]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModernTransformerTuner:
    """Analyze a time series and recommend transformer tokenization choices.

    The tuner is intentionally dependency-light and uses only NumPy-based
    autocorrelation and FFT heuristics so it can run in the core package.
    """

    def __init__(
        self,
        candidate_patch_lengths: Optional[Iterable[int]] = None,
        candidate_strides: Optional[Iterable[int]] = None,
    ) -> None:
        self.candidate_patch_lengths = list(
            candidate_patch_lengths or [4, 6, 8, 12, 16, 24, 32, 48, 64]
        )
        self.candidate_strides = list(
            candidate_strides or [1, 2, 4, 6, 8, 12, 16, 24, 32]
        )

    def analyze(
        self,
        series: Any,
        seasonal_periods: Optional[Iterable[int]] = None,
    ) -> TransformerTuningReport:
        values = _to_1d_array(series)
        count = int(values.size)
        known_periods = [
            int(period) for period in (seasonal_periods or []) if int(period) >= 2
        ]

        if count < 36:
            return TransformerTuningReport(
                available=False,
                series_length=count,
                patching_label="insufficient",
                patching_score=0,
                recommended_patch_len=0,
                recommended_patch_stride=0,
                recommended_patch_set=(),
                dominant_period=0.0,
                dominant_periods=(),
                acf_periods=(),
                repeatability=0.0,
                roughness=0.0,
                forecastability_score=0.0,
                spectral_entropy=1.0,
                multiscale_label="unknown",
                multiscale_score=0,
                hierarchical_label="unknown",
                hierarchical_score=0,
                recommended_hierarchical_periods=(),
                scale_band_shares=(),
                recommended_attention_mode="standard",
                recommended_preprocessing_heads=(),
                recommended_decompositions=(),
                recommended_model_kwargs={},
                notes=(
                    "Need at least 36 observations before patching and scale recommendations become reliable.",
                ),
            )

        series_std = _standard_deviation(values)
        innovations = _difference(values)
        innovation_std = _standard_deviation(innovations)
        roughness = innovation_std / max(series_std, 1e-8)

        detrended, trend_strength = _linear_detrend(values)
        max_lag = max(8, min(count // 2, 128))
        acf = _autocorrelation(values, max_lag=max_lag)
        acf_peaks = _find_acf_peaks(acf)
        repeatability = max((abs(value) for _, value in acf_peaks), default=0.0)
        acf_periods = tuple(lag for lag, _ in acf_peaks)

        centered = detrended - np.mean(detrended)
        spectrum = np.abs(np.fft.rfft(centered))[1:] ** 2
        if spectrum.size == 0 or float(np.sum(spectrum)) <= 1e-12:
            normalized_power = np.zeros(0, dtype=np.float64)
            peak_candidates: list[SpectralPeak] = []
        else:
            normalized_power = spectrum / np.sum(spectrum)
            periods = count / np.arange(1, normalized_power.size + 1, dtype=np.float64)
            spectral_pairs = [
                SpectralPeak(
                    period=max(2, int(round(period))), normalized_power=float(power)
                )
                for period, power in zip(periods, normalized_power)
                if period >= 2
            ]
            spectral_pairs.sort(key=lambda item: item.normalized_power, reverse=True)
            deduped: list[SpectralPeak] = []
            seen: set[int] = set()
            for peak in spectral_pairs:
                if peak.period in seen:
                    continue
                deduped.append(peak)
                seen.add(peak.period)
                if len(deduped) >= 6:
                    break
            peak_candidates = deduped

        dominant_period = (
            float(peak_candidates[0].period)
            if peak_candidates
            else float(known_periods[0])
            if known_periods
            else float(acf_periods[0])
            if acf_periods
            else 12.0
        )
        dominant_periods = tuple(peak_candidates[:4])
        dominant_share = (
            dominant_periods[0].normalized_power if dominant_periods else 0.0
        )
        second_share = (
            dominant_periods[1].normalized_power if len(dominant_periods) > 1 else 0.0
        )
        effective_periods = [
            peak.period for peak in dominant_periods if peak.normalized_power >= 0.08
        ]

        short_band_share = (
            float(
                np.sum(
                    normalized_power[
                        (count / np.arange(1, normalized_power.size + 1)) <= 8
                    ]
                )
            )
            if normalized_power.size
            else 0.0
        )
        medium_mask = (
            np.logical_and(
                (count / np.arange(1, normalized_power.size + 1)) > 8,
                (count / np.arange(1, normalized_power.size + 1)) <= 24,
            )
            if normalized_power.size
            else np.array([], dtype=bool)
        )
        medium_band_share = (
            float(np.sum(normalized_power[medium_mask]))
            if normalized_power.size
            else 0.0
        )
        long_band_share = (
            float(
                np.sum(
                    normalized_power[
                        (count / np.arange(1, normalized_power.size + 1)) > 24
                    ]
                )
            )
            if normalized_power.size
            else 0.0
        )
        scale_spread = len(
            [
                share
                for share in (short_band_share, medium_band_share, long_band_share)
                if share >= 0.22
            ]
        )

        peak_concentration = float(
            sum(peak.normalized_power for peak in dominant_periods[:3])
        )
        spectral_entropy = float(_spectral_entropy(normalized_power))
        forecastability_score = float(round(100.0 * (1.0 - spectral_entropy), 2))
        forecastability_penalty = (
            10 if forecastability_score < 35 else 4 if forecastability_score < 55 else 0
        )

        max_patch_length = max(4, min(max(self.candidate_patch_lengths), count // 3))
        base_patch_length = (
            dominant_period / 3 if dominant_period >= 24 else dominant_period / 2
        )
        if roughness > 1.05:
            base_patch_length *= 0.75
        elif roughness < 0.55:
            base_patch_length *= 1.15

        recommended_patch_len = _choose_nearest_candidate(
            base_patch_length,
            self.candidate_patch_lengths,
            max_value=max_patch_length,
        )
        recommended_patch_stride = min(
            recommended_patch_len,
            _choose_nearest_candidate(
                max(1.0, recommended_patch_len / 2),
                self.candidate_strides,
                max_value=recommended_patch_len,
            ),
        )

        patching_score = int(
            _clamp(
                round(
                    100.0
                    * (
                        0.30 * min(1.0, peak_concentration / 0.22)
                        + 0.25 * min(1.0, repeatability / 0.45)
                        + 0.25 * (1.0 - min(1.0, roughness / 1.45))
                        + 0.20 * min(1.0, count / 240.0)
                    )
                )
                - forecastability_penalty,
                0,
                100,
            )
        )
        patching_label = (
            "good"
            if patching_score >= 68
            else "moderate"
            if patching_score >= 48
            else "weak"
        )

        multiscale_score = int(
            _clamp(
                round(
                    100.0
                    * (
                        0.45 * min(1.0, scale_spread / 3.0)
                        + 0.35 * min(1.0, max(0, len(effective_periods) - 1) / 3.0)
                        + 0.20
                        * min(
                            1.0,
                            1.0
                            if short_band_share > 0.14 and long_band_share > 0.14
                            else medium_band_share / 0.35,
                        )
                    )
                )
                - (25 if dominant_share >= 0.58 and second_share <= 0.14 else 0),
                0,
                100,
            )
        )
        multiscale_label = (
            "high"
            if multiscale_score >= 65
            else "moderate"
            if multiscale_score >= 45
            else "low"
        )

        harmonic_pairs = 0
        hierarchy_periods: list[int] = []
        period_pool = list(effective_periods)
        for period in known_periods:
            if period not in period_pool:
                period_pool.append(period)
        for period in acf_periods[:3]:
            if period not in period_pool:
                period_pool.append(period)
        period_pool = sorted(period for period in period_pool if period >= 2)
        for index, period in enumerate(period_pool):
            for later in period_pool[index + 1 :]:
                ratio = later / max(period, 1)
                nearest_integer = max(1, round(ratio))
                if nearest_integer >= 2 and abs(ratio - nearest_integer) <= 0.2:
                    harmonic_pairs += 1
                    if period not in hierarchy_periods:
                        hierarchy_periods.append(period)
                    if later not in hierarchy_periods:
                        hierarchy_periods.append(later)

        hierarchical_score = int(
            _clamp(
                round(
                    100.0
                    * (
                        0.40 * min(1.0, max(0, len(period_pool) - 1) / 3.0)
                        + 0.35 * min(1.0, harmonic_pairs / 2.0)
                        + 0.25 * min(1.0, scale_spread / 3.0)
                    )
                ),
                0,
                100,
            )
        )
        hierarchical_label = (
            "high"
            if hierarchical_score >= 65
            else "moderate"
            if hierarchical_score >= 45
            else "low"
        )

        recommended_patch_set = tuple(
            sorted(
                {
                    _choose_nearest_candidate(
                        max(4.0, recommended_patch_len / 2),
                        self.candidate_patch_lengths,
                        max_value=max_patch_length,
                    ),
                    recommended_patch_len,
                    _choose_nearest_candidate(
                        max(4.0, dominant_period),
                        self.candidate_patch_lengths,
                        max_value=max_patch_length,
                    ),
                }
            )
        )

        decomposition_recommendations: list[DecompositionRecommendation] = []
        if trend_strength >= 0.2 and dominant_period >= 8:
            decomposition_recommendations.append(
                DecompositionRecommendation(
                    method="trend_seasonal",
                    suitability="recommended",
                    rationale="A visible low-frequency trend and a stable seasonal period make trend/season separation worthwhile before transformer tokenization.",
                    periods=(int(round(dominant_period)),),
                )
            )
        if len(period_pool) >= 2 and multiscale_label in {"moderate", "high"}:
            decomposition_recommendations.append(
                DecompositionRecommendation(
                    method="mstl",
                    suitability=multiscale_label,
                    rationale="Multiple separated seasonal periods suggest a multi-seasonal decomposition or explicit multi-period conditioning.",
                    periods=tuple(period_pool[:3]),
                )
            )
        if roughness >= 0.95 and short_band_share >= 0.18:
            decomposition_recommendations.append(
                DecompositionRecommendation(
                    method="wavelet",
                    suitability="recommended",
                    rationale="Short-scale energy and rough local transitions suggest a wavelet-style decomposition or localized multiscale filtering path.",
                )
            )
        if multiscale_label == "high" and roughness >= 0.9:
            decomposition_recommendations.append(
                DecompositionRecommendation(
                    method="emd",
                    suitability="optional",
                    rationale="Broadband multi-scale content can benefit from data-adaptive mode decomposition when fixed seasonal blocks underfit transients.",
                )
            )

        preprocessing_heads = ["RevINHead"]
        if any(
            item.method in {"trend_seasonal", "mstl"}
            for item in decomposition_recommendations
        ):
            preprocessing_heads.append("DecompositionBlock")
        if multiscale_label in {"moderate", "high"} or roughness >= 0.95:
            preprocessing_heads.append("MultiScaleConvHead")

        if count >= 768 and patching_label != "weak":
            recommended_attention_mode = (
                "hybrid" if multiscale_label in {"moderate", "high"} else "linear"
            )
        elif multiscale_label == "high":
            recommended_attention_mode = "hybrid"
        else:
            recommended_attention_mode = "standard"

        notes = [
            f"Patch tokenization looks {patching_label} with score {patching_score}/100 across {count} observations.",
            f"Start near patch length {recommended_patch_len} and stride {recommended_patch_stride}; dominant period is {dominant_period:.1f} steps.",
            f"Scale energy shares are short={short_band_share * 100:.1f}%, medium={medium_band_share * 100:.1f}%, long={long_band_share * 100:.1f}%.",
        ]
        if hierarchical_label in {"moderate", "high"} and hierarchy_periods:
            notes.append(
                "Hierarchical or nested patching is worth benchmarking because the signal contains harmonic structure across periods "
                + ", ".join(str(period) for period in hierarchy_periods[:4])
                + "."
            )
        if decomposition_recommendations:
            notes.append(
                "Suggested decomposition paths: "
                + ", ".join(rec.method for rec in decomposition_recommendations)
                + "."
            )

        model_kwargs = {
            "patch_encoder": patching_label != "weak",
            "patch_decoder": False,
            "patch_len": int(recommended_patch_len),
            "patch_stride": int(recommended_patch_stride),
            "attention_mode": recommended_attention_mode,
            "recommended_patch_set": list(recommended_patch_set),
            "use_multiscale": multiscale_label in {"moderate", "high"},
            "use_hierarchical": hierarchical_label in {"moderate", "high"},
        }

        return TransformerTuningReport(
            available=True,
            series_length=count,
            patching_label=patching_label,
            patching_score=patching_score,
            recommended_patch_len=int(recommended_patch_len),
            recommended_patch_stride=int(recommended_patch_stride),
            recommended_patch_set=recommended_patch_set,
            dominant_period=round(float(dominant_period), 2),
            dominant_periods=dominant_periods,
            acf_periods=acf_periods,
            repeatability=round(float(repeatability), 3),
            roughness=round(float(roughness), 3),
            forecastability_score=forecastability_score,
            spectral_entropy=round(spectral_entropy, 4),
            multiscale_label=multiscale_label,
            multiscale_score=multiscale_score,
            hierarchical_label=hierarchical_label,
            hierarchical_score=hierarchical_score,
            recommended_hierarchical_periods=tuple(hierarchy_periods[:4]),
            scale_band_shares=(
                ScaleBandShare(
                    band="Short", share=round(short_band_share, 4), detail="period <= 8"
                ),
                ScaleBandShare(
                    band="Medium",
                    share=round(medium_band_share, 4),
                    detail="8 < period <= 24",
                ),
                ScaleBandShare(
                    band="Long", share=round(long_band_share, 4), detail="period > 24"
                ),
            ),
            recommended_attention_mode=recommended_attention_mode,
            recommended_preprocessing_heads=tuple(preprocessing_heads),
            recommended_decompositions=tuple(decomposition_recommendations),
            recommended_model_kwargs=model_kwargs,
            notes=tuple(notes),
        )


TransformerTuner = ModernTransformerTuner


__all__ = [
    "ScaleBandShare",
    "SpectralPeak",
    "DecompositionRecommendation",
    "TransformerTuningReport",
    "ModernTransformerTuner",
    "TransformerTuner",
]
