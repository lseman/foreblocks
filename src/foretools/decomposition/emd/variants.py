"""Convenience facade for variational decomposition methods."""

from __future__ import annotations

from typing import Any

import numpy as np


try:
    from .common import FFTWManager
    from .core import VMDCore
except Exception:
    from vmd_common import FFTWManager
    from vmd_core import VMDCore


class VariationalVariants:
    """
    Thin namespace around :class:`VMDCore` for the variational family.

    This mirrors the discoverability role that ``EMDVariants`` serves for the
    empirical family, but keeps VMD/VNCMD/MVMD methods separate from EMD.
    """

    def __init__(
        self,
        wisdom_file: str = "vmd_fftw_wisdom.dat",
        fftw: FFTWManager | None = None,
    ):
        self.fftw = fftw if fftw is not None else FFTWManager(wisdom_file)
        self.core = VMDCore(self.fftw)

    def precompute_fft(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.core.precompute_fft(*args, **kwargs)

    def estimate_k(self, *args: Any, **kwargs: Any) -> int:
        return self.core.estimate_K(*args, **kwargs)

    def vmd(
        self, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.core.decompose(*args, **kwargs)

    def vncmd(
        self, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.core.decompose_vncmd(*args, **kwargs)

    def ncmd(
        self, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.core.decompose_ncmd(*args, **kwargs)

    def chirp(
        self, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.core.decompose_chirp(*args, **kwargs)

    def mvmd(
        self, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.core.decompose_multivariate(*args, **kwargs)

    def save_wisdom(self) -> None:
        self.fftw.save_wisdom()
