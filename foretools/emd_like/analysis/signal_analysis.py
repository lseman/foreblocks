from __future__ import annotations

import numpy as np
import scipy.stats
import pyfftw.interfaces.numpy_fft as fftw_np

from .config import VMDParameters


class SignalAnalyzer:
    @staticmethod
    def estimate_snr(signal: np.ndarray, fs: float) -> float:
        """Improved SNR: uses lower percentile of log-spectrum for noise floor."""
        s = np.asarray(signal, dtype=np.float64)
        spec = np.abs(fftw_np.rfft(s)) + 1e-12
        spec_db = 20 * np.log10(spec)

        # Take ~bottom 40% as noise-dominated region (conservative)
        sorted_db = np.sort(spec_db)
        noise_floor_db = np.median(sorted_db[: int(0.4 * len(sorted_db))])

        # Signal power approximated by peak + some margin
        signal_power_db = np.max(spec_db)

        snr_db = float(signal_power_db - noise_floor_db)
        return np.clip(snr_db, -10.0, 60.0)

    @staticmethod
    def dominant_freq(signal: np.ndarray, fs: float) -> float:
        s = np.asarray(signal, dtype=np.float64)
        if s.size < 8:
            return 0.0
        spec = np.abs(fftw_np.rfft(s))
        freqs = fftw_np.rfftfreq(s.size, d=1 / fs)
        if spec.size == 0:
            return 0.0
        spec = spec.copy()
        spec[0] = 0.0
        return float(freqs[int(np.argmax(spec))])

    @staticmethod
    def assess_complexity(
        signal: np.ndarray,
        fs: float,
        entropy_estimator=None,  # optional: pass your _multi_scale_dispersion_entropy if available
    ) -> VMDParameters:
        s = np.asarray(signal, dtype=np.float64)
        N = s.size
        if N < 32:
            return VMDParameters(
                n_trials=8,
                max_K=3,
                tol=1e-5,
                alpha_min=1500,
                alpha_max=4000,
            )

        spec = np.abs(fftw_np.rfft(s)) + 1e-12
        freqs = fftw_np.rfftfreq(N, d=1 / fs)
        spec_norm = spec / (np.sum(spec) + 1e-12)
        spectral_entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-12))

        spec2 = spec.copy()
        spec2[0] = 0
        peak_val = np.max(spec2)
        thr = peak_val / np.sqrt(10)
        significant_bins = np.sum(spec2 > thr)
        spread = float(significant_bins) / max(1, len(freqs))

        variability = float(np.std(np.diff(s))) / (float(np.std(s)) + 1e-12)
        kurt = float(scipy.stats.kurtosis(s, fisher=True))

        de = 0.5
        if entropy_estimator is not None:
            try:
                de = entropy_estimator(s)
            except Exception:
                pass

        se_norm = np.clip(spectral_entropy / 10.0, 0.0, 1.0)
        spread_norm = np.clip(spread, 0.0, 1.0)
        var_norm = np.clip(variability / 3.0, 0.0, 1.0)
        kurt_norm = np.clip(kurt / 15.0, 0.0, 1.0)

        complexity_score = (
            0.35 * se_norm
            + 0.25 * spread_norm
            + 0.20 * var_norm
            + 0.15 * kurt_norm
            + 0.05 * de
        )
        complexity_score = np.clip(complexity_score, 0.0, 1.2)

        snr_db = SignalAnalyzer.estimate_snr(s, fs)

        base_K = 3 + int(7 * complexity_score)
        base_alpha_min = 300 + 1700 * (1 - complexity_score)
        base_alpha_max = 2500 + 7500 * complexity_score

        p = VMDParameters(
            n_trials=int(max(10, min(35, N // 120))),
            max_K=min(max(3, base_K), 10),
            tol=1e-6 * (1 + 0.5 * complexity_score),
            alpha_min=base_alpha_min,
            alpha_max=base_alpha_max,
        )

        if snr_db < 8:
            p.alpha_min *= 2.2
            p.alpha_max *= 2.8
            p.max_K = max(2, p.max_K // 2)
            p.n_trials = max(8, p.n_trials // 2)
        elif snr_db < 18:
            p.alpha_min *= 1.6
            p.alpha_max *= 1.7
        elif snr_db > 30:
            p.max_K = min(p.max_K + 2, 12)
            p.alpha_min *= 0.75
            p.alpha_max *= 0.8

        p.alpha_max = max(p.alpha_max, p.alpha_min * 1.5)

        if N > 4000:
            p.n_trials = max(10, p.n_trials // 2)
            p.tol *= 1.8

        print(
            f"complexity={complexity_score:.3f} | SNR={snr_db:.1f} dB | "
            f"dom≈{SignalAnalyzer.dominant_freq(s, fs):.2f} Hz | "
            f"kurt={kurt:.1f}"
        )
        print(
            f" => trials={p.n_trials}, max_K={p.max_K}, "
            f"alpha∈[{p.alpha_min:.0f}, {p.alpha_max:.0f}]"
        )

        return p
