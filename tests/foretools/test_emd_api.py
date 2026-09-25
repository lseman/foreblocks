"""Tests for EMD/VMD decomposition package — API, validation, and common usage."""

from __future__ import annotations

import numpy as np
import pytest

from foretools.decomposition.emd import (
    BoundaryHandler,
    EMDVariants,
    FastVMD,
    FFTWManager,
    HierarchicalParameters,
    ModeProcessor,
    SignalAnalyzer,
    VariationalVariants,
    VMDOptions,
    VMDParameters,
)
from foretools.decomposition.emd.common import (
    _energy,
    _is_imf,
    _mode_energy_ratio,
    _normalise,
    _reconstruct_error,
    _validate_signal,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_signal(
    fs: float = 1000.0,
    duration: float = 1.0,
    freqs: list[float] | None = None,
    amps: list[float] | None = None,
    noise: float = 0.0,
) -> np.ndarray:
    """Synthetic multi-tone signal with optional Gaussian noise."""
    t = np.arange(int(fs * duration)) / fs
    sig = np.zeros_like(t)
    freqs = freqs or [50.0, 150.0]
    amps = amps or [1.0, 0.5]
    for f, a in zip(freqs, amps):
        sig += a * np.sin(2 * np.pi * f * t)
    if noise > 0:
        sig += noise * np.random.default_rng(42).standard_normal(len(t))
    return sig


# ── common.py helpers ────────────────────────────────────────────────────────


class TestEnergyAndValidation:
    def test_energy_scalar(self) -> None:
        assert _energy(np.array([3.0, 4.0])) == pytest.approx(25.0)

    def test_energy_zero(self) -> None:
        assert _energy(np.zeros(10)) == 0.0

    def test_normalise_zero_mean_unit_var(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n = _normalise(x)
        assert np.mean(n) == pytest.approx(0.0, abs=1e-12)
        assert np.std(n) == pytest.approx(1.0, abs=1e-12)

    def test_reconstruct_error_perfect(self) -> None:
        sig = _make_signal(freqs=[50.0], amps=[1.0])
        err = _reconstruct_error(sig, [sig])
        assert err == pytest.approx(0.0, abs=1e-12)

    def test_reconstruct_error_empty(self) -> None:
        sig = _make_signal()
        with pytest.raises(ValueError, match="at least one"):
            _reconstruct_error(sig, [])

    def test_mode_energy_ratio(self) -> None:
        sig = _make_signal(freqs=[50.0], amps=[2.0])
        ratio = _mode_energy_ratio(sig, sig)
        assert ratio == pytest.approx(1.0)  # mode equals original → full energy

    def test_validate_1d_ok(self) -> None:
        x = np.random.default_rng(0).standard_normal(64)
        assert _validate_signal(x).shape == (64,)

    def test_validate_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            _validate_signal(np.zeros((4, 4)))

    def test_validate_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            _validate_signal(np.array([1.0, 2.0]))

    def test_validate_nonfinite_raises(self) -> None:
        x = np.ones(64)
        x[32] = np.nan
        with pytest.raises(ValueError, match="Non-finite"):
            _validate_signal(x)


class TestIsIMF:
    def test_monotonic_is_not_imf(self) -> None:
        assert not _is_imf(np.linspace(0, 1, 64))

    def test_constant_is_not_imf(self) -> None:
        assert not _is_imf(np.ones(64))

    def test_short_signal_is_not_imf(self) -> None:
        assert not _is_imf(np.array([1.0, 2.0, 3.0]))


# ── config dataclasses ───────────────────────────────────────────────────────


class TestConfigDataclasses:
    def test_vmd_parameters_defaults(self) -> None:
        p = VMDParameters()
        assert p.max_K == 6
        assert p.tol == 1e-6
        assert p.alpha_min == 500
        assert p.admm_over_relax == 1.6

    def test_vmd_options_defaults(self) -> None:
        o = VMDOptions()
        assert o.tol == 1e-6
        assert o.max_iter == 300
        assert o.use_anderson is False

    def test_vmd_options_override_decompose(self) -> None:
        """VMDOptions should be usable as a single bundled config."""
        opts = VMDOptions(tol=1e-5, use_anderson=True, gram_schmidt_every=10)
        assert opts.gram_schmidt_every == 10

    def test_hierarchical_params_defaults(self) -> None:
        hp = HierarchicalParameters()
        assert hp.max_levels == 3
        assert hp.energy_threshold == 0.01


# ── boundary handling ────────────────────────────────────────────────────────


class TestBoundaryHandler:
    def test_adaptive_extension_ratio_short(self) -> None:
        assert BoundaryHandler.adaptive_extension_ratio(np.zeros(100)) == 0.30

    def test_adaptive_extension_ratio_long(self) -> None:
        assert BoundaryHandler.adaptive_extension_ratio(np.zeros(5000)) == 0.15

    def test_extend_mirror(self) -> None:
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        extended, left, right = BoundaryHandler.extend_signal(s, "mirror", 0.2)
        assert left == right
        assert len(extended) > len(s)

    def test_extend_none(self) -> None:
        s = np.array([1.0, 2.0, 3.0])
        extended, left, right = BoundaryHandler.extend_signal(s, "none", 0.5)
        assert np.array_equal(extended, s)
        assert left == 0

    def test_taper_boundaries(self) -> None:
        modes = [np.ones(100)]
        tapered = BoundaryHandler.taper_boundaries(modes, taper_length=10)
        assert len(tapered) == 1
        # boundaries should be reduced
        assert tapered[0][0] < 1.0
        assert tapered[0][-1] < 1.0


# ── signal analysis ──────────────────────────────────────────────────────────


class TestSignalAnalyzer:
    def test_dominant_freq_simple(self) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        f = SignalAnalyzer.dominant_freq(sig, 1000.0)
        assert abs(f - 50.0) < 2.0

    def test_estimate_snr_positive(self) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0], noise=0.01)
        snr = SignalAnalyzer.estimate_snr(sig, 1000.0)
        assert snr > 0

    def test_assess_complexity_short_signal(self) -> None:
        # Short signals get minimal params
        sig = np.sin(2 * np.pi * 0.1 * np.arange(32))
        p = SignalAnalyzer.assess_complexity(sig, fs=1000.0)
        assert p.max_K <= 5  # short signal gets modest K

    def test_assess_complexity_long_signal(self) -> None:
        sig = _make_signal(fs=1000.0, duration=5.0)
        p = SignalAnalyzer.assess_complexity(sig, 1000.0)
        assert p.max_K >= 3


# ── mode processor ───────────────────────────────────────────────────────────


class TestModeProcessor:
    def test_dominant_frequency_simple(self) -> None:
        sig = _make_signal(fs=1000.0, freqs=[75.0], amps=[1.0])
        f = ModeProcessor.dominant_frequency(sig, 1000.0)
        assert abs(f - 75.0) < 2.0

    def test_merge_similar_modes(self) -> None:
        sig1 = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        sig2 = _make_signal(fs=1000.0, freqs=[52.0], amps=[0.5])
        merged = ModeProcessor.merge_similar_modes([sig1, sig2], 1000.0, freq_tol=0.1)
        assert len(merged) == 1

    def test_sort_modes_by_frequency(self) -> None:
        low = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        high = _make_signal(fs=1000.0, freqs=[200.0], amps=[1.0])
        modes, freqs = ModeProcessor.sort_modes_by_frequency([high, low], 1000.0)
        assert freqs[0] < freqs[1]

    def test_cost_signal_empty(self) -> None:
        cost = ModeProcessor.cost_signal([], _make_signal(), 1000.0)
        assert cost == 10.0


# ── EMD variants ─────────────────────────────────────────────────────────────


class TestEMDVariants:
    def test_emd_basic(self) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0, 150.0], amps=[1.0, 0.5])
        imfs = EMDVariants.emd(sig, max_imfs=3)
        assert len(imfs) >= 2  # at least 1 IMF + residual

    def test_emd_reconstruction(self) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        imfs = EMDVariants.emd(sig, max_imfs=2)
        recon = sum(imfs)
        assert np.allclose(recon, sig, atol=1e-6)

    def test_emd_too_short(self) -> None:
        sig = np.arange(5.0)
        imfs = EMDVariants.emd(sig)
        assert len(imfs) == 1

    def test_ceemdan_basic(self) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0, 150.0], amps=[1.0, 0.5])
        imfs = EMDVariants.ceemdan(sig, n_ensembles=10, max_imfs=3)
        assert len(imfs) >= 2

    def test_ceemdan_reconstruction(self) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        imfs = EMDVariants.ceemdan(sig, n_ensembles=10, max_imfs=2)
        recon = sum(imfs)
        assert np.allclose(recon, sig, atol=0.1)

    def test_iceemdan_basic(self) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        imfs = EMDVariants.iceemdan(sig, n_ensembles=10, max_imfs=2)
        assert len(imfs) >= 1

    def test_orthogonality_index_single(self) -> None:
        assert EMDVariants.compute_orthogonality_index([np.ones(64)]) == 0.0

    def test_orthogonality_index_zero(self) -> None:
        imfs = [np.zeros(64), np.zeros(64)]
        assert EMDVariants.compute_orthogonality_index(imfs) == 0.0

    def test_instantaneous_frequency(self) -> None:
        t = np.linspace(0, 1, 500)
        imf = np.sin(2 * np.pi * 50 * t)
        freqs, amps = EMDVariants.compute_instantaneous_frequency(imf, fs=1000.0)
        assert len(freqs) == 500
        assert np.mean(amps) > 0


# ── VariationalVariants ──────────────────────────────────────────────────────


class TestVariationalVariants:
    @pytest.fixture(scope="class")
    def vvar(self, tmp_path_factory: pytest.TempPathFactory):
        wisdom = tmp_path_factory.mktemp("vmd") / "wisdom.dat"
        return VariationalVariants(str(wisdom))

    def test_vmd_basic(self, vvar: VariationalVariants) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0, 150.0], amps=[1.0, 0.5])
        u, uh, omega = vvar.vmd(sig, alpha=1000.0, K=2, max_iter=20)
        assert u.shape == (2, len(sig))
        assert omega.shape == (2,)

    def test_vmd_reconstruction(self, vvar: VariationalVariants) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        u, _, _ = vvar.vmd(sig, alpha=2000.0, K=2, max_iter=60)
        recon = np.sum(u, axis=0)
        rel_err = np.mean((recon - sig) ** 2) / (np.mean(sig**2) + 1e-12)
        assert rel_err < 0.05  # relative MSE < 5%

    def test_estimate_k(self, vvar: VariationalVariants) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0, 150.0], amps=[1.0, 0.5])
        k = vvar.estimate_k(sig, K_min=2, K_max=4, max_iter=20)
        assert isinstance(k, int)
        assert 2 <= k <= 4

    def test_precompute_fft(self, vvar: VariationalVariants) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        precomp = vvar.precompute_fft(sig)
        assert "T" in precomp
        assert "f_hat_plus" in precomp

    def test_mvmd_basic(self, vvar: VariationalVariants) -> None:
        sigs = np.stack([
            _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0]),
            _make_signal(fs=1000.0, freqs=[75.0], amps=[0.8]),
        ])
        u, uh, omega = vvar.mvmd(
            sigs, alpha=2000.0, K=2, tau=0.0, DC=0, init=1, tol=1e-6,
            max_iter=30, boundary_method="mirror",
        )
        assert u.shape[0] == 2  # channels
        assert u.shape[1] == 2  # modes

    def test_ncmd_alias(self, vvar: VariationalVariants) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        result = vvar.ncmd(sig, alpha=2000.0, K=2, tau=0.0, DC=0, init=1, tol=1e-6, max_iter=10)
        assert len(result) == 4

    def test_chirp_alias(self, vvar: VariationalVariants) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        result = vvar.chirp(sig, alpha=2000.0, K=2, tau=0.0, DC=0, init=1, tol=1e-6, max_iter=10)
        assert len(result) == 4


# ── FastVMD ──────────────────────────────────────────────────────────────────


class TestFastVMD:
    @pytest.fixture(scope="class")
    def vmd(self, tmp_path_factory: pytest.TempPathFactory):
        wisdom = tmp_path_factory.mktemp("vmd") / "wisdom.dat"
        return FastVMD(str(wisdom))

    def test_decompose_standard(self, vmd: FastVMD) -> None:
        sig = _make_signal(fs=1000.0, freqs=[50.0], amps=[1.0])
        modes, freqs, info = vmd.decompose(sig, fs=1000.0, max_K=3, n_trials=5)
        assert isinstance(info, dict) or isinstance(info, tuple)

    def test_decompose_hierarchical(self, vmd: FastVMD) -> None:
        sig = _make_signal(fs=1000.0, freqs=[20.0, 100.0], amps=[1.0, 0.5])
        modes, freqs, level_info = vmd.decompose(
            sig, fs=1000.0, method="hierarchical", max_levels=2
        )
        assert isinstance(level_info, list)

    def test_clear_cache(self, vmd: FastVMD) -> None:
        vmd.opt._cache = {1: "a"}
        vmd.opt._cache_mv = {2: "b"}
        vmd.clear_cache()
        assert vmd.opt._cache == {}
        assert vmd.opt._cache_mv == {}


# ── __init__.py public API surface ──────────────────────────────────────────


class TestPublicAPI:
    """Verify that all expected names are importable from the package root."""

    def test_all_names_present(self) -> None:
        import foretools.decomposition.emd as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), f"Missing exported name: {name}"

    def test_core_classes_importable(self) -> None:
        from foretools.decomposition.emd import (
            EMDVariants,
            FastVMD,
            HierarchicalVMD,
            ModeProcessor,
            SignalAnalyzer,
            VariationalVariants,
            VMDCore,
            VMDOptimizer,
        )

        assert callable(EMDVariants.emd)
        assert callable(VariationalVariants.vmd)
        assert callable(FastVMD.decompose)
        assert callable(VMDOptimizer.optimize)

    def test_no_internal_leakage(self) -> None:
        """Private helpers should not be in __all__ except _energy."""
        import foretools.decomposition.emd as pkg

        # Only _energy is intentionally private (utility function)
        private_in_all = [n for n in pkg.__all__ if n.startswith("_") and n != "_energy"]
        assert private_in_all == [], f"Unexpected private names in __all__: {private_in_all}"
