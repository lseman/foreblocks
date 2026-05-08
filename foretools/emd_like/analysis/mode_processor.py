from __future__ import annotations

import numpy as np
import pyfftw.interfaces.numpy_fft as fftw_np

from .emd import EMDVariants


class ModeProcessor:
    @staticmethod
    def dominant_frequency(sig: np.ndarray, fs: float) -> float:
        x = np.asarray(sig, dtype=np.float64)
        N = x.size
        if N < 8 or np.allclose(x, 0, atol=1e-12):
            return 0.0
        freqs = fftw_np.rfftfreq(N, d=1 / fs)
        spec = np.abs(fftw_np.rfft(x))
        spec = spec.copy()
        if spec.size:
            spec[0] = 0.0
        return float(freqs[int(np.argmax(spec))]) if spec.size else 0.0

    @staticmethod
    def merge_similar_modes(
        modes: list[np.ndarray], fs: float, freq_tol: float
    ) -> list[np.ndarray]:
        if len(modes) <= 1:
            return modes
        dom = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        used = np.zeros(len(modes), dtype=bool)
        merged: list[np.ndarray] = []
        for i in range(len(modes)):
            if used[i]:
                continue
            group = [modes[i]]
            used[i] = True
            fi = dom[i]
            for j in range(i + 1, len(modes)):
                if used[j]:
                    continue
                fj = dom[j]
                if abs(fi - fj) / max(fi, 1e-6) < freq_tol:
                    group.append(modes[j])
                    used[j] = True
            merged.append(np.sum(group, axis=0))
        return merged

    @staticmethod
    def sort_modes_by_frequency(
        modes: list[np.ndarray], fs: float, low_to_high: bool = True
    ) -> tuple[list[np.ndarray], list[float]]:
        dom = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        order = np.argsort(dom)
        if not low_to_high:
            order = order[::-1]
        return [modes[i] for i in order], [float(dom[i]) for i in order]

    @staticmethod
    def cost_signal(modes: list[np.ndarray], signal: np.ndarray, fs: float) -> float:
        if len(modes) == 0:
            return 10.0
        x = np.asarray(signal, dtype=np.float64)
        total_energy = np.sum(x**2) + 1e-12
        recon = np.sum(modes, axis=0)
        residual_energy = np.sum((x - recon) ** 2) / total_energy

        dom_freqs = [ModeProcessor.dominant_frequency(m, fs) for m in modes]
        if len(dom_freqs) > 1:
            gaps = np.diff(np.sort(dom_freqs))
            overlap_penalty = float(np.mean(np.exp(-gaps / (np.median(gaps) + 1e-12))))
        else:
            overlap_penalty = 0.0

        oi = EMDVariants.compute_orthogonality_index(modes)

        entropy_vals = []
        for m in modes:
            spec = np.abs(fftw_np.rfft(m)) + 1e-12
            p = spec / np.sum(spec)
            H = -np.sum(p * np.log(p + 1e-12))
            Hn = H / np.log(p.size + 1e-12)
            entropy_vals.append(Hn)

        avg_entropy = float(np.mean(entropy_vals))
        return float(
            0.5 * residual_energy + 0.2 * overlap_penalty + 0.1 * avg_entropy + 0.2 * oi
        )

    @staticmethod
    def entropy(mode: np.ndarray) -> float:
        x = np.asarray(mode, dtype=np.float64)
        spec = np.abs(fftw_np.rfft(x)) + 1e-12
        p = spec / (np.sum(spec) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))
