from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional backend
    torch = None

from ..emd import EMDVariants


class ModeProcessor:
    @staticmethod
    def _rfft(
        x: np.ndarray,
        *,
        axis: int = -1,
        backend: str = "numpy",
        device: str = "auto",
    ) -> np.ndarray:
        """Run an optimized real FFT and return a NumPy array.

        NumPy uses pocketfft on CPU. The torch backend uses torch.fft (cuFFT
        for CUDA tensors) and is useful when sufficiently large batches
        amortize host/device transfers.
        """
        name = backend.lower()
        if name not in ("numpy", "torch", "auto"):
            raise ValueError("backend must be one of {'numpy', 'torch', 'auto'}")
        use_torch = name == "torch" or (
            name == "auto" and torch is not None and isinstance(x, torch.Tensor)
        )
        if not use_torch:
            return np.fft.rfft(np.asarray(x), axis=axis)
        if torch is None:
            raise RuntimeError("torch FFT backend requested but PyTorch is unavailable")
        if device == "auto":
            target = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        else:
            target = torch.device(device)
        tensor = torch.as_tensor(x, device=target, dtype=torch.float64)
        result = torch.fft.rfft(tensor, dim=axis)
        return result.detach().cpu().numpy()

    @staticmethod
    def dominant_frequency(
        sig: np.ndarray,
        fs: float,
        *,
        backend: str = "numpy",
        device: str = "auto",
    ) -> float:
        x = np.asarray(sig, dtype=np.float64)
        N = x.size
        if N < 8 or np.allclose(x, 0, atol=1e-12):
            return 0.0
        freqs = np.fft.rfftfreq(N, d=1 / fs)
        spec = np.abs(ModeProcessor._rfft(x, backend=backend, device=device))
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
    def cost_signal(
        modes: list[np.ndarray],
        signal: np.ndarray,
        fs: float,
        weights: dict[str, float] | None = None,
        *,
        fft_backend: str = "numpy",
        fft_device: str = "auto",
    ) -> float:
        """
        Composite cost for a set of candidate modes.

        Parameters
        ----------
        weights : dict or None
            Keys ``"residual"``, ``"overlap"``, ``"entropy"``,
            ``"orthogonality"`` with values in [0, 1].  Defaults to
            ``{"residual": 0.5, "overlap": 0.2, "entropy": 0.1,
            "orthogonality": 0.2}``.
        """
        if len(modes) == 0:
            return 10.0

        if weights is None:
            w_res = 0.5
            w_olap = 0.2
            w_ent = 0.1
            w_orth = 0.2
        else:
            w_res  = float(weights.get("residual", 0.5))
            w_olap = float(weights.get("overlap", 0.2))
            w_ent  = float(weights.get("entropy", 0.1))
            w_orth = float(weights.get("orthogonality", 0.2))

        x = np.asarray(signal, dtype=np.float64)
        total_energy = np.sum(x**2) + 1e-12
        recon = np.sum(modes, axis=0)
        residual_energy = np.sum((x - recon) ** 2) / total_energy

        mode_matrix = np.stack([np.asarray(m, dtype=np.float64) for m in modes])
        spectra = np.abs(
            ModeProcessor._rfft(
                mode_matrix,
                axis=-1,
                backend=fft_backend,
                device=fft_device,
            )
        )
        dominant_spectra = spectra.copy()
        dominant_spectra[:, 0] = 0.0
        frequency_bins = np.fft.rfftfreq(mode_matrix.shape[-1], d=1 / fs)
        dom_freqs = frequency_bins[np.argmax(dominant_spectra, axis=-1)].tolist()
        if len(dom_freqs) > 1:
            gaps = np.diff(np.sort(dom_freqs))
            overlap_penalty = float(np.mean(np.exp(-gaps / (np.median(gaps) + 1e-12))))
        else:
            overlap_penalty = 0.0

        oi = EMDVariants.compute_orthogonality_index(modes)

        spectra = spectra + 1e-12
        probabilities = spectra / np.sum(spectra, axis=-1, keepdims=True)
        entropy_vals = -np.sum(
            probabilities * np.log(probabilities + 1e-12), axis=-1
        ) / np.log(probabilities.shape[-1] + 1e-12)

        avg_entropy = float(np.mean(entropy_vals))
        return float(
            w_res * residual_energy
            + w_olap * overlap_penalty
            + w_ent * avg_entropy
            + w_orth * oi
        )

    @staticmethod
    def entropy(
        mode: np.ndarray,
        *,
        backend: str = "numpy",
        device: str = "auto",
    ) -> float:
        x = np.asarray(mode, dtype=np.float64)
        spec = np.abs(
            ModeProcessor._rfft(x, backend=backend, device=device)
        ) + 1e-12
        p = spec / (np.sum(spec) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))
