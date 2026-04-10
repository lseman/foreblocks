import numpy as np

from foreblocks.tf.transformer_tuner import ModernTransformerTuner


def test_transformer_tuner_prefers_single_scale_patching_for_clean_daily_cycle():
    timeline = np.arange(240, dtype=np.float64)
    series = np.sin(2 * np.pi * timeline / 24)

    report = ModernTransformerTuner().analyze(series)

    assert report.available is True
    assert report.recommended_patch_len in {8, 12, 16}
    assert report.recommended_patch_stride <= report.recommended_patch_len
    assert report.multiscale_label in {"low", "moderate"}
    assert report.dominant_period >= 20
    assert "RevINHead" in report.recommended_preprocessing_heads


def test_transformer_tuner_detects_multiscale_and_hierarchical_structure():
    timeline = np.arange(384, dtype=np.float64)
    series = (
        0.9 * np.sin(2 * np.pi * timeline / 8)
        + 1.1 * np.sin(2 * np.pi * timeline / 48)
        + 0.4 * np.sin(2 * np.pi * timeline / 96)
    )

    report = ModernTransformerTuner().analyze(series)

    assert report.available is True
    assert report.multiscale_score >= 45
    assert report.hierarchical_score >= 45
    assert len(report.recommended_patch_set) >= 2
    assert len(report.recommended_hierarchical_periods) >= 2
    assert any(
        recommendation.method in {"mstl", "emd", "wavelet"}
        for recommendation in report.recommended_decompositions
    )
