from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel  # pip install pydantic
from pydantic import Field, field_validator


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp value to [lower, upper]."""
    return max(lower, min(upper, value))


def _choose_nearest_candidate(
    value: float, candidates: list[int], max_value: float = float("inf")
) -> int:
    """Select the candidate closest to `value` without exceeding `max_value`."""
    viable = [c for c in candidates if c <= max_value]
    pool = viable if viable else candidates
    return min(pool, key=lambda c: abs(c - value))


def _to_1d_array(series: Any) -> np.ndarray:
    """Convert input (list, np.array, torch.Tensor) to 1D float64 array, handling multivariate by averaging."""
    if isinstance(series, torch.Tensor):
        arr = series.detach().cpu().numpy()
    else:
        arr = np.asarray(series, dtype=np.float64)

    if arr.ndim == 0:
        raise ValueError("series must contain at least one observation")
    if arr.ndim == 1:
        values = arr
    elif arr.ndim == 2:
        values = arr.reshape(-1) if 1 in arr.shape else arr.mean(axis=-1)
    else:
        values = arr.reshape(arr.shape[0], -1).mean(axis=-1)

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError("series must contain at least one finite observation")

    return values[finite_mask].astype(np.float64, copy=False)


def _standard_deviation(values: np.ndarray) -> float:
    return 0.0 if values.size <= 1 else float(np.std(values, ddof=0))


def _difference(values: np.ndarray) -> np.ndarray:
    return np.empty(0, dtype=np.float64) if values.size <= 1 else np.diff(values)


def _linear_detrend(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Linear detrend + compute trend strength (R²-like)."""
    if values.size <= 2:
        return values - values.mean(), 0.0

    timeline = np.arange(values.size, dtype=np.float64)
    slope, intercept = np.polyfit(timeline, values, deg=1)
    trend = slope * timeline + intercept
    residual = values - trend

    total_var = float(np.var(values))
    residual_var = float(np.var(residual))
    trend_strength = (
        0.0 if total_var <= 1e-12 else _clamp(1.0 - residual_var / total_var)
    )
    return residual, trend_strength


def _autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute ACF up to max_lag."""
    centered = values - values.mean()
    variance = float(np.dot(centered, centered))
    if variance <= 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float64)

    corr = np.correlate(centered, centered, mode="full")
    mid = corr.size // 2
    acf = corr[mid : mid + max_lag + 1] / variance
    return acf.astype(np.float64, copy=False)


def _find_acf_peaks(acf: np.ndarray, max_peaks: int = 6) -> list[tuple[int, float]]:
    """Find local maxima in ACF above threshold."""
    peaks: list[tuple[int, float]] = []
    if acf.size <= 3:
        return peaks

    for lag in range(2, acf.size - 1):
        val = float(acf[lag])
        if val < 0.1:
            continue
        if val >= acf[lag - 1] and val >= acf[lag + 1]:
            peaks.append((lag, val))

    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:max_peaks]


def _spectral_entropy(normalized_power: np.ndarray) -> float:
    """Shannon entropy of the power spectrum (higher = more unpredictable)."""
    if normalized_power.size == 0:
        return 1.0
    power = normalized_power[normalized_power > 0]
    if power.size == 0:
        return 1.0
    entropy = -float(np.sum(power * np.log(power)))
    return entropy / max(np.log(power.size), 1e-8)


class SpectralPeak(BaseModel):
    period: int
    normalized_power: float


class ScaleBandShare(BaseModel):
    band: str
    share: float
    detail: str


class DecompositionRecommendation(BaseModel):
    method: str
    suitability: Literal["recommended", "optional", "moderate", "high", "low"]
    rationale: str
    periods: tuple[int, ...] = Field(default_factory=tuple)


class TransformerTuningReport(BaseModel):
    """Modern Pydantic model for the tuning report (better serialization + validation)."""

    available: bool
    series_length: int
    context_length: int | None = None

    # Patching
    patching_label: Literal["good", "moderate", "weak", "insufficient"]
    patching_score: int = Field(ge=0, le=100)
    recommended_patch_len: int
    recommended_patch_stride: int
    recommended_patch_set: tuple[int, ...]
    estimated_num_patches: int | None = None

    # Periodicity
    dominant_period: float
    dominant_periods: tuple[SpectralPeak, ...]
    acf_periods: tuple[int, ...]

    # Signal characteristics
    repeatability: float
    roughness: float
    forecastability_score: float
    spectral_entropy: float

    # Multiscale & hierarchical
    multiscale_label: Literal["high", "moderate", "low", "unknown"]
    multiscale_score: int
    hierarchical_label: Literal["high", "moderate", "low", "unknown"]
    hierarchical_score: int
    recommended_hierarchical_periods: tuple[int, ...]

    scale_band_shares: tuple[ScaleBandShare, ...]

    recommended_attention_mode: Literal["standard", "linear", "hybrid"]
    recommended_preprocessing_heads: tuple[str, ...]
    recommended_decompositions: tuple[DecompositionRecommendation, ...]
    recommended_model_kwargs: dict[str, Any]

    notes: tuple[str, ...]

    @field_validator(
        "recommended_patch_set", "recommended_hierarchical_periods", mode="before"
    )
    @classmethod
    def ensure_tuples(cls, v: Any) -> tuple:
        return tuple(v) if isinstance(v, (list, set)) else v


class ModernTransformerTuner:
    """Analyze a time series and recommend modern transformer tokenization / preprocessing choices.

    Designed to be lightweight (NumPy + Torch only) while providing heuristics inspired by
    PatchTST, iTransformer, and recent dynamic patching research.
    """

    def __init__(
        self,
        candidate_patch_lengths: Iterable[int] | None = None,
        candidate_strides: Iterable[int] | None = None,
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
        seasonal_periods: Iterable[int] | None = None,
        context_length: int | None = None,
    ) -> TransformerTuningReport:
        values = _to_1d_array(series)
        n = int(values.size)
        known_periods = [int(p) for p in (seasonal_periods or []) if int(p) >= 2]

        effective_context = (
            context_length if context_length is not None and context_length >= 32 else n
        )

        if effective_context < 36:
            return TransformerTuningReport(
                available=False,
                series_length=n,
                context_length=context_length,
                patching_label="insufficient",
                patching_score=0,
                recommended_patch_len=0,
                recommended_patch_stride=0,
                recommended_patch_set=(),
                estimated_num_patches=None,
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
                    "Need at least 36 observations for reliable patching and scale recommendations.",
                ),
            )

        # Basic signal stats
        series_std = _standard_deviation(values)
        innovations = _difference(values)
        roughness = _standard_deviation(innovations) / max(series_std, 1e-8)

        detrended, trend_strength = _linear_detrend(values)

        # Autocorrelation
        max_lag = max(8, min(n // 2, 128))
        acf = _autocorrelation(values, max_lag)
        acf_peaks = _find_acf_peaks(acf)
        repeatability = max((abs(v) for _, v in acf_peaks), default=0.0)
        acf_periods = tuple(lag for lag, _ in acf_peaks)

        # Spectral analysis (FFT)
        centered = detrended - detrended.mean()
        spectrum = np.abs(np.fft.rfft(centered))[1:] ** 2
        total_power = float(np.sum(spectrum))

        if spectrum.size == 0 or total_power <= 1e-12:
            normalized_power = np.zeros(0, dtype=np.float64)
            dominant_periods: tuple[SpectralPeak, ...] = ()
        else:
            normalized_power = spectrum / total_power
            freqs = np.arange(1, normalized_power.size + 1, dtype=np.float64)
            periods = n / freqs

            spectral_peaks = [
                SpectralPeak(
                    period=max(2, int(round(p))), normalized_power=float(power)
                )
                for p, power in zip(periods, normalized_power)
                if p >= 2
            ]
            spectral_peaks.sort(key=lambda x: x.normalized_power, reverse=True)

            # Deduplicate by period
            seen: set[int] = set()
            deduped = []
            for peak in spectral_peaks:
                if peak.period not in seen:
                    deduped.append(peak)
                    seen.add(peak.period)
                if len(deduped) >= 6:
                    break
            dominant_periods = tuple(deduped[:4])

        dominant_period = (
            float(dominant_periods[0].period)
            if dominant_periods
            else (
                float(known_periods[0])
                if known_periods
                else (acf_periods[0] if acf_periods else 12.0)
            )
        )

        # Scale band shares (short/medium/long)
        if normalized_power.size > 0:
            periods_arr = n / np.arange(1, normalized_power.size + 1)
            short = float(np.sum(normalized_power[periods_arr <= 8]))
            medium = float(
                np.sum(normalized_power[(periods_arr > 8) & (periods_arr <= 24)])
            )
            long_ = float(np.sum(normalized_power[periods_arr > 24]))
        else:
            short = medium = long_ = 0.0

        scale_spread = sum(1 for s in (short, medium, long_) if s >= 0.22)
        peak_concentration = sum(p.normalized_power for p in dominant_periods[:3])
        spectral_entropy = _spectral_entropy(normalized_power)
        forecastability_score = round(100.0 * (1.0 - spectral_entropy), 2)

        # Patching recommendation (influenced by roughness + dominant period)
        max_patch = max(
            4,
            min(max(self.candidate_patch_lengths), effective_context // 4),
        )
        base_patch = dominant_period / (3 if dominant_period >= 24 else 2)

        if roughness > 1.05:
            base_patch *= 0.75
        elif roughness < 0.55:
            base_patch *= 1.15

        if context_length is not None and context_length >= 32:
            divisors = [
                d
                for d in self.candidate_patch_lengths
                if d <= max_patch and context_length % d == 0
            ]
            if divisors:
                base_patch = (base_patch + float(np.mean(divisors))) / 2.0

        rec_patch_len = _choose_nearest_candidate(
            base_patch, self.candidate_patch_lengths, max_patch
        )

        target_patches = max(32, min(96, effective_context // 8))
        ideal_stride = max(1, effective_context // target_patches)

        rec_stride = min(
            rec_patch_len,
            _choose_nearest_candidate(
                max(1.0, ideal_stride), self.candidate_strides, rec_patch_len
            ),
        )

        if rec_stride > 0:
            estimated_num_patches = (
                effective_context - rec_patch_len
            ) // rec_stride + 1
        else:
            estimated_num_patches = None

        forecastability_penalty = (
            10 if forecastability_score < 35 else 4 if forecastability_score < 55 else 0
        )

        patching_score = int(
            _clamp(
                round(
                    100.0
                    * (
                        0.30 * min(1.0, peak_concentration / 0.22)
                        + 0.25 * min(1.0, repeatability / 0.45)
                        + 0.25 * (1.0 - min(1.0, roughness / 1.45))
                        + 0.20 * min(1.0, effective_context / 240.0)
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

        # Multiscale & hierarchical logic (kept similar but cleaner)
        effective_periods = [
            p.period for p in dominant_periods if p.normalized_power >= 0.08
        ]

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
                            1.0 if (short > 0.14 and long_ > 0.14) else medium / 0.35,
                        )
                    )
                )
                - (
                    25
                    if dominant_periods
                    and dominant_periods[0].normalized_power >= 0.58
                    and (
                        len(dominant_periods) < 2
                        or dominant_periods[1].normalized_power <= 0.14
                    )
                    else 0
                ),
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

        # Hierarchical (harmonic) detection - simplified
        period_pool = sorted({*effective_periods, *known_periods, *acf_periods[:3]})
        hierarchy_periods: list[int] = []
        harmonic_pairs = 0

        for i, p1 in enumerate(period_pool):
            for p2 in period_pool[i + 1 :]:
                ratio = p2 / max(p1, 1)
                nearest = round(ratio)
                if nearest >= 2 and abs(ratio - nearest) <= 0.2:
                    harmonic_pairs += 1
                    if p1 not in hierarchy_periods:
                        hierarchy_periods.append(p1)
                    if p2 not in hierarchy_periods:
                        hierarchy_periods.append(p2)

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

        # Patch set for experimentation
        rec_patch_set = tuple(
            sorted(
                {
                    _choose_nearest_candidate(
                        max(4.0, rec_patch_len / 2),
                        self.candidate_patch_lengths,
                        max_patch,
                    ),
                    rec_patch_len,
                    _choose_nearest_candidate(
                        max(4.0, dominant_period),
                        self.candidate_patch_lengths,
                        max_patch,
                    ),
                }
            )
        )

        # Decomposition recommendations
        decomps: list[DecompositionRecommendation] = []
        if trend_strength >= 0.2 and dominant_period >= 8:
            decomps.append(
                DecompositionRecommendation(
                    method="trend_seasonal",
                    suitability="recommended",
                    rationale="Visible trend + stable seasonal period → worth separating before patching.",
                    periods=(int(round(dominant_period)),),
                )
            )
        if len(period_pool) >= 2 and multiscale_label in {"moderate", "high"}:
            decomps.append(
                DecompositionRecommendation(
                    method="mstl",
                    suitability=multiscale_label,  # type: ignore
                    rationale="Multiple seasonal components detected.",
                    periods=tuple(period_pool[:3]),
                )
            )
        if roughness >= 0.95 and short >= 0.18:
            decomps.append(
                DecompositionRecommendation(
                    method="wavelet",
                    suitability="recommended",
                    rationale="High short-scale energy suggests wavelet or localized multiscale processing.",
                )
            )
        if multiscale_label == "high" and roughness >= 0.9:
            decomps.append(
                DecompositionRecommendation(
                    method="emd",
                    suitability="optional",
                    rationale="Broadband signal may benefit from adaptive decomposition.",
                )
            )

        # Preprocessing heads
        heads = ["RevINHead"]
        if any(d.method in {"trend_seasonal", "mstl"} for d in decomps):
            heads.append("DecompositionBlock")
        if multiscale_label in {"moderate", "high"} or roughness >= 0.95:
            heads.append("MultiScaleConvHead")

        # Attention mode recommendation (modern defaults favor efficient variants)
        if n >= 768 and patching_label != "weak":
            attention_mode = (
                "hybrid" if multiscale_label in {"moderate", "high"} else "linear"
            )
        elif multiscale_label == "high":
            attention_mode = "hybrid"
        else:
            attention_mode = "standard"

        notes = [
            f"Patch tokenization is {patching_label} (score {patching_score}/100) for {n} observations.",
            f"Recommended starting point: patch_len={rec_patch_len}, stride={rec_stride} ({estimated_num_patches} patches for context={effective_context}).",
            f"Scale energy: short={short * 100:.1f}% (≤8), medium={medium * 100:.1f}% (8–24), long={long_ * 100:.1f}% (>24).",
        ]
        if hierarchical_label in {"moderate", "high"} and hierarchy_periods:
            notes.append(
                "Hierarchical/nested patching recommended due to harmonic structure across periods: "
                + ", ".join(map(str, hierarchy_periods[:4]))
                + "."
            )
        if context_length is not None and context_length != n:
            notes.append(
                f"Recommendations optimized for context_length={context_length} while series length is {n}."
            )

        if decomps:
            notes.append(
                "Suggested decompositions: "
                + ", ".join(d.method for d in decomps)
                + "."
            )

        model_kwargs = {
            "patch_encoder": patching_label != "weak",
            "patch_decoder": False,
            "patch_len": int(rec_patch_len),
            "patch_stride": int(rec_stride),
            "attention_mode": attention_mode,
            "recommended_patch_set": list(rec_patch_set),
            "use_multiscale": multiscale_label in {"moderate", "high"},
            "use_hierarchical": hierarchical_label in {"moderate", "high"},
            "context_length": context_length,
            "estimated_num_patches": estimated_num_patches,
        }

        return TransformerTuningReport(
            available=True,
            series_length=n,
            context_length=context_length,
            patching_label=patching_label,
            patching_score=patching_score,
            recommended_patch_len=int(rec_patch_len),
            recommended_patch_stride=int(rec_stride),
            recommended_patch_set=rec_patch_set,
            estimated_num_patches=estimated_num_patches,
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
                    band="Short", share=round(short, 4), detail="period <= 8"
                ),
                ScaleBandShare(
                    band="Medium", share=round(medium, 4), detail="8 < period <= 24"
                ),
                ScaleBandShare(
                    band="Long", share=round(long_, 4), detail="period > 24"
                ),
            ),
            recommended_attention_mode=attention_mode,
            recommended_preprocessing_heads=tuple(heads),
            recommended_decompositions=tuple(decomps),
            recommended_model_kwargs=model_kwargs,
            notes=tuple(notes),
        )


# Backward compatibility
TransformerTuner = ModernTransformerTuner

__all__ = [
    "ScaleBandShare",
    "SpectralPeak",
    "DecompositionRecommendation",
    "TransformerTuningReport",
    "ModernTransformerTuner",
    "TransformerTuner",
]
