"""Convenience facade for variational decomposition methods."""

from __future__ import annotations

from typing import Any

import numpy as np


try:
    from .support.fft import FFTWManager
    from .core import VMDCore
except Exception:
    try:
        from vmd_common import FFTWManager  # type: ignore[assignment]
        from vmd_core import VMDCore  # type: ignore[assignment]
    except Exception:
        FFTWManager = None  # type: ignore[misc]
        VMDCore = None  # type: ignore[misc]


class VariationalVariants:
    """High-level namespace for Variational Mode Decomposition (VMD) methods.

    Provides a thin wrapper around :class:`VMDCore` with convenient method names
    that mirror the ``EMDVariants`` API for consistency.

    Available methods
    -----------------
    - ``vmd()`` — Standard VMD decomposition
    - ``vncmd()`` / ``chirp()`` — Non-stationary IF tracking (VNCMD)
    - ``ncmd()`` — Alias for vncmd
    - ``mvmd()`` — Multi-channel joint VMD (MVMD)
    - ``estimate_k()`` — Auto-select the number of modes
    - ``precompute_fft()`` — Pre-compute FFT for batch decomposition

    Parameters
    ----------
    wisdom_file : str
        Path to save/load FFTW wisdom cache. Default "vmd_fftw_wisdom.dat".
    fftw : FFTWManager or None
        Reuse an existing FFT manager. Creates a new one if None.

    Examples
    --------
    >>> from foretools.decomposition.emd import VariationalVariants
    >>> v = VariationalVariants()
    >>> sig = 1*np.sin(2*np.pi*5*np.linspace(0, 1, 500)) + \
    ...       0.5*np.sin(2*np.pi*15*np.linspace(0, 1, 500))
    >>> u, uh, omega = v.vmd(sig, alpha=2000, K=3)
    >>> print(f"Centre frequencies: {omega}")  # doctest: +SKIP
    Centre frequencies: [0.0099... 0.0198... 0.1000...]
    """

    def __init__(
        self,
        wisdom_file: str = "vmd_fftw_wisdom.dat",
        fftw: FFTWManager | None = None,
    ):
        self.fftw = fftw if fftw is not None else FFTWManager(wisdom_file)  # type: ignore[call-arg]
        self.core = VMDCore(self.fftw)

    def precompute_fft(
        self,
        signal: np.ndarray,
        boundary_method: str = "mirror",
        use_soft_junction: bool = False,
        window_alpha: float | None = None,
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> dict[str, Any]:
        """Pre-compute the FFT of a signal for batch decomposition.

        Parameters
        ----------
        signal : array-like
            1-D input signal.
        boundary_method : {"mirror", "reflect", "linear", "constant", "none"}
            How to extend the signal at boundaries.
        use_soft_junction : bool
            Apply smooth transition at original/extended boundary junction.
        window_alpha : float or None
            Tukey window alpha (0–1). If given, ``boundary_method`` must be "none".
        fft_backend : {"fftw", "torch"}
            FFT backend. "fftw" maps to NumPy; "torch" uses PyTorch.
        fft_device : {"auto", "cpu", "cuda"}
            Target device for torch backend.

        Returns
        -------
        dict[str, Any]
            Pre-computed FFT data usable in subsequent ``decompose`` calls.
        """
        return self.core.precompute_fft(
            signal,
            boundary_method=boundary_method,
            use_soft_junction=use_soft_junction,
            window_alpha=window_alpha,
            fft_backend=fft_backend,
            fft_device=fft_device,
        )

    def estimate_k(
        self,
        signal: np.ndarray,
        K_min: int = 2,
        K_max: int = 10,
        alpha: float = 2000.0,
        tol: float = 1e-6,
        max_iter: int = 150,
        energy_threshold: float = 0.01,
        entropy_gain_threshold: float = 0.015,
        boundary_method: str = "mirror",
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> int:
        """Estimate the optimal number of VMD modes (K) for a signal.

        Uses an incremental approach: decompose with K=2, 3, ... and stop when
        the newest mode carries negligible energy AND the residual entropy gain
        falls below threshold.

        Parameters
        ----------
        signal : array-like
            1-D input signal.
        K_min, K_max : int
            Search range for K (inclusive).
        alpha : float
            VMD bandwidth penalty. Higher = sharper mode separation.
        tol : float
            Convergence tolerance per decomposition.
        max_iter : int
            Maximum iterations per decomposition.
        energy_threshold : float
            Stop when newest mode energy < this fraction of total.
        entropy_gain_threshold : float
            Stop when residual spectral-entropy gain < this value.
        boundary_method : str
            Boundary extension method (passed to VMD).
        fft_backend, fft_device : str
            FFT backend settings.

        Returns
        -------
        int
            Optimal K in [K_min, K_max].

        Notes
        -----
        This is more accurate than the fast Welch-PSD estimator
        (:meth:`VMDCore.estimate_K_fast`) but requires multiple VMD runs.
        Use ``estimate_K_fast`` as a cheap pre-filter if needed.
        """
        return self.core.estimate_K(
            signal,
            K_min=K_min,
            K_max=K_max,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            energy_threshold=energy_threshold,
            entropy_gain_threshold=entropy_gain_threshold,
            boundary_method=boundary_method,
            fft_backend=fft_backend,
            fft_device=fft_device,
        )

    def vmd(
        self,
        signal: np.ndarray,
        alpha: float,
        K: int,
        tau: float = 0.0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-6,
        max_iter: int = 300,
        boundary_method: str = "mirror",
        fft_backend: str = "fftw",
        fft_device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standard VMD decomposition.

        Decomposes a signal into K band-limited modes with centre frequencies
        that are determined adaptively via ADMM optimisation.

        Parameters
        ----------
        signal : array-like
            1-D input signal.
        alpha : float
            Bandwidth penalty (balance between mode sharpness and fidelity).
            Typical range: 500–5000. Higher = sharper modes.
        K : int
            Number of modes to extract.
        tau : float
            Time-step of the dual ascent. Zero for static decomposition.
        DC : int
            If non-zero, force the first mode to be the DC component (ω=0).
        init : int
            Centre-frequency initialisation method:
            1 = uniform, 2 = log-uniform random, 3 = spectral peaks,
            4 = Hilbert IF histogram (best for AM-FM), 5 = warm start.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum ADMM iterations.
        boundary_method : str
            Boundary extension method.
        fft_backend, fft_device : str
            FFT backend settings.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(u, u_hat_full, omega)`` where:
            - ``u``: (K, N) array of time-domain modes
            - ``u_hat_full``: (T, K) full FFT spectrum
            - ``omega``: (K,) centre frequencies in [0, 0.5]
        """
        return self.core.decompose(
            signal,
            alpha=alpha,
            K=K,
            tau=tau,
            DC=DC,
            init=init,
            tol=tol,
            max_iter=max_iter,
            boundary_method=boundary_method,
            fft_backend=fft_backend,
            fft_device=fft_device,
        )

    def vncmd(
        self,
        signal: np.ndarray,
        alpha: float,
        K: int,
        tau: float = 0.0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-6,
        max_iter: int = 300,
        if_window_size: int = 256,
        if_hop_size: int = 128,
        if_center_smooth: float = 0.85,
        boundary_method: str = "mirror",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """VNCMD: Variational Non-linear Chirp Mode Decomposition.

        For signals with non-stationary (time-varying) instantaneous frequency.
        Uses an alternating solver: envelope least-squares + IF-track refinement.

        Parameters
        ----------
        signal : array-like
            1-D input signal.
        alpha : float
            Envelope smoothness penalty (maps from VMD alpha).
        K : int
            Number of chirp modes.
        tol : float
            Convergence tolerance for both mode and IF-track changes.
        max_iter : int
            Maximum outer iterations.
        if_window_size, if_hop_size : int
            STFT parameters for initial IF ridge extraction.
        if_center_smooth : float
            EMA smoothing factor for IF tracks (0–1).
        boundary_method : str
            Boundary extension method.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(u, u_hat, omega, if_tracks)`` where ``if_tracks`` is (K, N).
        """
        return self.core.decompose_vncmd(
            signal,
            alpha=alpha,
            K=K,
            tau=tau,
            DC=DC,
            init=init,
            tol=tol,
            max_iter=max_iter,
            if_window_size=if_window_size,
            if_hop_size=if_hop_size,
            if_center_smooth=if_center_smooth,
            boundary_method=boundary_method,
        )

    def ncmd(
        self, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Alias for :meth:`vncmd` (non-linear chirp mode decomposition)."""
        return self.core.decompose_ncmd(*args, **kwargs)

    def chirp(
        self, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Alias for :meth:`vncmd` (backward-compatible name)."""
        return self.core.decompose_chirp(*args, **kwargs)

    def mvmd(
        self,
        signals: np.ndarray,
        alpha: float,
        K: int,
        tau: float = 0.0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-6,
        max_iter: int = 300,
        boundary_method: str = "mirror",
        use_soft_junction: bool = False,
        window_alpha: float | None = None,
        fs: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Multi-channel joint VMD (MVMD).

        Decomposes multiple channels simultaneously with shared centre
        frequencies. All channels contribute to a single Gauss-Seidel sweep.

        Parameters
        ----------
        signals : array-like
            2-D array of shape (channels, samples).
        alpha : float
            Bandwidth penalty.
        K : int
            Number of modes.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.
        boundary_method : str
            Boundary extension method.
        use_soft_junction : bool
            Smooth transition at boundary junctions.
        window_alpha : float or None
            Tukey window alpha (use with boundary_method="none").
        fs : float
            Sampling frequency (passed through).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(u, u_hat, omega)`` where ``u`` is (C, K, N).
        """
        return self.core.decompose_multivariate(
            signals,
            alpha=alpha,
            K=K,
            tau=tau,
            DC=DC,
            init=init,
            tol=tol,
            max_iter=max_iter,
            boundary_method=boundary_method,
            use_soft_junction=use_soft_junction,
            window_alpha=window_alpha,
            fs=fs,
        )

    def save_wisdom(self) -> None:
        """Save FFTW wisdom cache to disk (if using numpy backend)."""
        self.fftw.save_wisdom()
