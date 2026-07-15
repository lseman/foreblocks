from __future__ import annotations

import os
import warnings

import numpy as np


# Optional torch refinement
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None


class FFTWManager:
    def __init__(
        self,
        wisdom_file: str = "vmd_fftw_wisdom.dat",
        num_threads: int | None = None,
    ):
        self.wisdom_file = wisdom_file
        self._warned_backend_keys = set()
        self.num_threads = self._resolve_num_threads(num_threads)

    @staticmethod
    def _resolve_num_threads(num_threads: int | None) -> int:
        if num_threads is None:
            raw = os.environ.get("FOREBLOCKS_FFT_THREADS", "1")
            try:
                num_threads = int(raw)
            except ValueError:
                warnings.warn(
                    "Invalid FOREBLOCKS_FFT_THREADS; using one FFT thread.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                num_threads = 1
        return max(1, min(int(num_threads), os.cpu_count() or 1))

    def load_wisdom(self):
        """Compatibility no-op retained for callers of the former FFTW backend."""

    def save_wisdom(self):
        """Compatibility no-op retained for callers of the former FFTW backend."""

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_backend_keys:
            return
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        self._warned_backend_keys.add(key)

    def resolve_backend(
        self, backend: str = "fftw", device: str = "auto"
    ) -> tuple[str, str]:
        name = str(backend).lower()
        dev = str(device).lower()

        if name == "fftw":
            name = "numpy"
        if name not in ("numpy", "torch"):
            self._warn_once(
                f"fft_backend_invalid:{name}",
                f"Unknown fft_backend={backend!r}; falling back to 'numpy'.",
            )
            return "numpy", "cpu"

        if name == "torch":
            if not TORCH_AVAILABLE or torch is None:
                self._warn_once(
                    "fft_backend_torch_missing",
                    "fft_backend='torch' requested but PyTorch is unavailable; "
                    "falling back to 'numpy'.",
                )
                return "numpy", "cpu"

            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            elif dev not in ("cpu", "cuda"):
                self._warn_once(
                    f"fft_device_invalid:{dev}",
                    f"Unknown fft_device={device!r}; using 'auto' resolution.",
                )
                dev = "cuda" if torch.cuda.is_available() else "cpu"

            if dev == "cuda" and not torch.cuda.is_available():
                self._warn_once(
                    "fft_backend_cuda_unavailable",
                    "fft_device='cuda' requested but CUDA is unavailable; "
                    "falling back to CPU torch FFT.",
                )
                dev = "cpu"

            if dev == "cuda":
                self._warn_once(
                    "fft_backend_cuda_transfer",
                    "The VMD solver stores state in NumPy, so torch CUDA FFTs "
                    "include host/device transfers and may be slower than the CPU backend.",
                )

            return "torch", dev

        return "numpy", "cpu"

    @staticmethod
    def _numpy_axes(axes, ndim: int):
        if axes is None:
            return tuple(range(ndim))
        if isinstance(axes, tuple):
            return axes
        if isinstance(axes, list):
            return tuple(axes)
        return int(axes)

    @staticmethod
    def _to_numpy(x):
        if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _to_torch(x, device: str):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, device=device)
        if isinstance(x, torch.Tensor):
            return x.to(device=device)
        return torch.as_tensor(np.asarray(x), device=device)

    def fft(self, x, axis: int = -1, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            return self._to_numpy(torch.fft.fft(xt, dim=axis))
        return np.fft.fft(np.asarray(x), axis=axis)

    def ifft(self, x, axis: int = -1, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            return self._to_numpy(torch.fft.ifft(xt, dim=axis))
        return np.fft.ifft(np.asarray(x), axis=axis)

    def rfft(self, x, axis: int = -1, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            return self._to_numpy(torch.fft.rfft(xt, dim=axis))
        return np.fft.rfft(np.asarray(x), axis=axis)

    def fftshift(self, x, axes=None, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            dims = self._numpy_axes(axes, xt.ndim)
            return self._to_numpy(torch.fft.fftshift(xt, dim=dims))
        return np.fft.fftshift(np.asarray(x), axes=axes)

    def ifftshift(self, x, axes=None, backend: str = "fftw", device: str = "auto"):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            xt = self._to_torch(x, dev)
            dims = self._numpy_axes(axes, xt.ndim)
            return self._to_numpy(torch.fft.ifftshift(xt, dim=dims))
        return np.fft.ifftshift(np.asarray(x), axes=axes)

    def fftfreq(
        self, n: int, d: float = 1.0, backend: str = "fftw", device: str = "auto"
    ):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            return self._to_numpy(
                torch.fft.fftfreq(int(n), d=float(d), device=dev, dtype=torch.float64)
            )
        return np.fft.fftfreq(int(n), d=float(d))

    def rfftfreq(
        self, n: int, d: float = 1.0, backend: str = "fftw", device: str = "auto"
    ):
        name, dev = self.resolve_backend(backend, device)
        if name == "torch":
            return self._to_numpy(
                torch.fft.rfftfreq(int(n), d=float(d), device=dev, dtype=torch.float64)
            )
        return np.fft.rfftfreq(int(n), d=float(d))
