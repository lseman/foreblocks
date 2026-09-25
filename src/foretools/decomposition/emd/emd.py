from __future__ import annotations

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, interp1d
from scipy.signal import find_peaks, hilbert


class EMDVariants:
    """Empirical Mode Decomposition (EMD) and its variants.

    Provides three decomposition algorithms:

    * ``emd`` — Classic sifting-based EMD with Akima/PCHIP/Cubic envelopes
    * ``ceemdan`` — Complete EMD with adaptive noise (better mode separation)
    * ``iceemdan`` — Improved CEEMDAN (faster convergence, lower residue)

    All methods return a list of Intrinsic Mode Functions (IMFs) followed by
    a residual trend. The sum of all returned arrays equals the original signal.

    Parameters common to all methods
    --------------------------------
    signal : array-like
        1-D real-valued input signal.
    max_imfs : int
        Maximum number of IMFs to extract (default 10).
    envelope_method : {"akima", "pchip", "cubic", "linear"}
        Interpolation method for envelope construction. Akima is most
        robust for non-stationary signals; PCHIP avoids overshoot.
    seed : int or None
        Random seed for ensemble methods (CEEMDAN/ICEEMDAN).

    Examples
    --------
    >>> import numpy as np
    >>> from foretools.decomposition.emd import EMDVariants
    >>> sig = np.sin(2*np.pi*5*np.linspace(0, 1, 500)) + \
    ...       0.5*np.sin(2*np.pi*15*np.linspace(0, 1, 500))
    >>> imfs = EMDVariants.emd(sig)
    >>> len(imfs)  # number of IMFs + residual
    4
    >>> np.allclose(sum(imfs), sig)  # perfect reconstruction
    True
    """

    @staticmethod
    def emd(
        signal: np.ndarray,
        max_imfs: int = 10,
        max_sifts: int = 100,
        sift_threshold: float = 0.05,
        energy_threshold: float = 1e-8,
        envelope_method: str = "akima",
    ) -> list[np.ndarray]:
        """Classic Empirical Mode Decomposition via sifting.

        Extracts IMFs by iteratively subtracting the local mean of upper and
        lower envelopes until the residue satisfies the IMF stationarity criterion.

        Parameters
        ----------
        signal : array-like
            1-D input signal.
        max_imfs : int
            Maximum number of IMFs to extract. The residual is always appended.
        max_sifts : int
            Maximum sifting iterations per IMF (default 100).
        sift_threshold : float
            Stopping criterion: relative L2 change between successive sifts.
        energy_threshold : float
            Stop if remaining energy < this fraction of original energy.
        envelope_method : {"akima", "pchip", "cubic", "linear"}
            Interpolation method. Akima (≥5 extrema) or fallback to PCHIP/interp1d.

        Returns
        -------
        list[np.ndarray]
            IMFs ordered from highest to lowest frequency, followed by residual.

        Notes
        -----
        For better mode separation on noisy signals, prefer ``ceemdan`` or
        ``iceemdan`` which add ensemble averaging with white noise.
        """
        x = np.asarray(signal, dtype=np.float64)
        x = np.asarray(signal, dtype=np.float64)
        N = x.size

        if N < 10 or np.allclose(x, x[0], atol=1e-12):
            return [x.copy()]

        imfs = []
        residual = x.copy()
        original_energy = np.sum(x**2) + 1e-12

        for _ in range(max_imfs):
            if np.sum(residual**2) / original_energy < energy_threshold:
                break
            if EMDVariants._is_monotonic(residual):
                break

            imf = EMDVariants._sift(
                residual,
                max_sifts=max_sifts,
                threshold=sift_threshold,
                envelope_method=envelope_method,
            )
            if imf is None:
                break

            imfs.append(imf)
            residual = residual - imf

            if np.std(imf) < 1e-10 * np.std(x):
                break

        if len(imfs) == 0 or np.sum(residual**2) > energy_threshold * original_energy:
            imfs.append(residual)

        return imfs

    @staticmethod
    def _sift(
        signal: np.ndarray,
        max_sifts: int = 100,
        threshold: float = 0.05,
        envelope_method: str = "akima",
    ) -> np.ndarray | None:
        h = signal.copy()
        N = len(h)

        for iteration in range(max_sifts):
            max_idx, max_val = EMDVariants._find_extrema(h, kind="max")
            min_idx, min_val = EMDVariants._find_extrema(h, kind="min")

            if len(max_idx) < 3 or len(min_idx) < 3:
                return h if iteration > 0 else None

            upper = EMDVariants._interpolate_envelope(
                max_idx, max_val, N, method=envelope_method
            )
            lower = EMDVariants._interpolate_envelope(
                min_idx, min_val, N, method=envelope_method
            )
            if upper is None or lower is None:
                return h if iteration > 0 else None

            mean_env = 0.5 * (upper + lower)
            h_new = h - mean_env

            diff = np.sum((h_new - h) ** 2) / (np.sum(h**2) + 1e-12)
            h = h_new

            if diff < threshold:
                break
            if np.std(mean_env) < threshold * np.std(h):
                break

        return h

    @staticmethod
    def _find_extrema(
        signal: np.ndarray, kind: str = "max"
    ) -> tuple[np.ndarray, np.ndarray]:
        x = signal
        N = len(x)

        if kind == "max":
            peaks, _ = find_peaks(x)
            if N > 2:
                if x[0] > x[1]:
                    peaks = np.concatenate([[0], peaks])
                if x[-1] > x[-2]:
                    peaks = np.concatenate([peaks, [N - 1]])
        else:
            peaks, _ = find_peaks(-x)
            if N > 2:
                if x[0] < x[1]:
                    peaks = np.concatenate([[0], peaks])
                if x[-1] < x[-2]:
                    peaks = np.concatenate([peaks, [N - 1]])

        peaks = np.unique(peaks)
        return peaks, x[peaks]

    @staticmethod
    def _interpolate_envelope(
        idx: np.ndarray,
        values: np.ndarray,
        N: int,
        method: str = "akima",
    ) -> np.ndarray | None:
        if len(idx) < 2:
            return None

        order = np.argsort(idx)
        idx = idx[order]
        values = values[order]

        unique_mask = np.concatenate([[True], np.diff(idx) > 0])
        idx = idx[unique_mask]
        values = values[unique_mask]
        if len(idx) < 2:
            return None

        x_interp = np.arange(N)

        try:
            if method == "akima" and len(idx) >= 5:
                interp_func = Akima1DInterpolator(idx, values)
                envelope = interp_func(x_interp)
            elif method == "pchip" and len(idx) >= 2:
                interp_func = PchipInterpolator(idx, values, extrapolate=True)
                envelope = interp_func(x_interp)
            elif method == "cubic" and len(idx) >= 4:
                interp_func = interp1d(
                    idx,
                    values,
                    kind="cubic",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                envelope = interp_func(x_interp)
            else:
                envelope = np.interp(x_interp, idx, values)

            if np.any(~np.isfinite(envelope)):
                envelope = np.interp(x_interp, idx, values)

            return envelope
        except Exception:
            try:
                return np.interp(x_interp, idx, values)
            except Exception:
                return None

    @staticmethod
    def _is_monotonic(signal: np.ndarray) -> bool:
        diff = np.diff(signal)
        return np.all(diff >= -1e-10) or np.all(diff <= 1e-10)

    @staticmethod
    def _precompute_noise_imf_bank(
        N: int,
        n_ensembles: int,
        max_imfs: int,
        max_sifts_noise: int = 50,
        seed: int | None = None,
        envelope_method: str = "akima",
    ) -> tuple[np.ndarray, list[list[np.ndarray]]]:
        rng = np.random.default_rng(seed)
        noises = rng.standard_normal((n_ensembles, N)).astype(np.float64, copy=False)

        bank: list[list[np.ndarray]] = []
        for i in range(n_ensembles):
            imfs_i = EMDVariants.emd(
                noises[i],
                max_imfs=max_imfs,
                max_sifts=max_sifts_noise,
                sift_threshold=0.1,
                energy_threshold=1e-10,
                envelope_method=envelope_method,
            )
            bank.append([np.asarray(imf, dtype=np.float64) for imf in imfs_i])

        return noises, bank

    @staticmethod
    def _get_noise_component(
        bank_i: list[np.ndarray], noise_fallback: np.ndarray, k: int
    ) -> np.ndarray:
        if k < len(bank_i):
            return bank_i[k]
        if len(bank_i) > 0:
            return bank_i[-1]
        return noise_fallback

    @staticmethod
    def ceemdan(
        signal: np.ndarray,
        noise_std: float = 0.2,
        n_ensembles: int = 100,
        max_imfs: int = 10,
        epsilon: float = 1e-6,
        seed: int | None = 0,
        envelope_method: str = "akima",
        max_sifts_signal: int = 100,
        max_sifts_noise: int = 40,
        **emd_kwargs,
    ) -> list[np.ndarray]:
        """Complete EMD with adaptive noise (CEEMDAN).

        Adds white noise realisations to the signal, runs EMD on each,
        and averages the results. The adaptive noise scaling (βₖ = σₙ/(k+1))
        ensures that each IMF level receives appropriate noise amplitude.

        Parameters
        ----------
        signal : array-like
            1-D input signal.
        noise_std : float
            Standard deviation of added white noise (relative to residual std).
        n_ensembles : int
            Number of noise realisations. More = better separation but slower.
        max_imfs : int
            Maximum number of IMFs per ensemble.
        epsilon : float
            Convergence threshold for stopping the IMF extraction loop.
        seed : int or None
            Random seed for reproducibility.
        envelope_method : {"akima", "pchip", "cubic", "linear"}
            Interpolation method passed to the inner EMD call.
        max_sifts_signal : int
            Max sifting iterations per IMF on the noisy signal.
        max_sifts_noise : int
            Max sifting iterations for precomputing noise IMFs.
        **emd_kwargs
            Additional keyword arguments forwarded to the inner ``emd`` call.

        Returns
        -------
        list[np.ndarray]
            IMFs + residual. Sum equals the original signal (to numerical precision).

        Notes
        -----
        CEEMDAN typically requires fewer ensembles than EEMD for good results.
        50–100 ensembles are usually sufficient. For very long signals (>10k samples),
        reduce ``max_imfs`` or increase ``epsilon`` to speed up computation.

        References
        ----------
        Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014).
        Improved complete ensemble EMD: A suitable tool for biomedical signal processing.
        *Applied Sciences*, 4(3), 698–715.
        """
        x = np.asarray(signal, dtype=np.float64)
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 20:
            return [x.copy()]

        x_std0 = float(np.std(x))
        if x_std0 < 1e-12 or np.allclose(x, x[0], atol=1e-12):
            return [x.copy()]

        n_ensembles = int(max(1, n_ensembles))
        max_imfs = int(max(1, max_imfs))

        noises, noise_bank = EMDVariants._precompute_noise_imf_bank(
            N=N,
            n_ensembles=n_ensembles,
            max_imfs=max_imfs,
            max_sifts_noise=int(max_sifts_noise),
            seed=seed,
            envelope_method=envelope_method,
        )

        imfs: list[np.ndarray] = []
        residual = x.copy()

        for k in range(max_imfs):
            res_std = float(np.std(residual))
            if res_std < float(epsilon) * x_std0:
                break
            if EMDVariants._is_monotonic(residual):
                break

            beta_k = float(noise_std) / float(k + 1)
            amp = beta_k * res_std

            imf_sum = np.zeros(N, dtype=np.float64)
            valid = 0

            for i in range(n_ensembles):
                e_k = EMDVariants._get_noise_component(noise_bank[i], noises[i], k)
                noisy = residual + amp * e_k

                trial_imfs = EMDVariants.emd(
                    noisy,
                    max_imfs=1,
                    max_sifts=int(max_sifts_signal),
                    envelope_method=envelope_method,
                    **emd_kwargs,
                )
                if len(trial_imfs) > 0:
                    imf_sum += np.asarray(trial_imfs[0], dtype=np.float64)
                    valid += 1

            if valid == 0:
                break

            imf_k = imf_sum / float(valid)
            imfs.append(imf_k)
            residual = residual - imf_k

        if np.sum(residual**2) > (float(epsilon) ** 2) * (np.sum(x**2) + 1e-12):
            imfs.append(residual)

        return imfs

    @staticmethod
    def iceemdan(
        signal: np.ndarray,
        noise_std: float = 0.2,
        n_ensembles: int = 100,
        max_imfs: int = 10,
        epsilon: float = 1e-6,
        seed: int | None = 0,
        envelope_method: str = "akima",
        max_sifts_noise: int = 40,
        **emd_kwargs,
    ) -> list[np.ndarray]:
        """Improved CEEMDAN (ICEEMDAN) with residual-based noise addition.

        Like CEEMDAN but adds noise to the *residual* at each level rather than
        to the original signal. This gives faster convergence and better mode
        separation with fewer ensembles.

        Parameters
        ----------
        signal : array-like
            1-D input signal.
        noise_std : float
            Noise amplitude scaling (relative to residual std at each level).
        n_ensembles : int
            Number of noise realisations. 50–100 is typical.
        max_imfs : int
            Maximum number of IMFs to extract.
        epsilon : float
            Stopping threshold for the IMF extraction loop.
        seed : int or None
            Random seed for reproducibility.
        envelope_method : {"akima", "pchip", "cubic", "linear"}
            Interpolation method.
        max_sifts_noise : int
            Max sifting iterations for precomputing noise IMFs.
        **emd_kwargs
            Additional keyword arguments forwarded to the inner ``emd`` call.

        Returns
        -------
        list[np.ndarray]
            IMFs + residual.

        Notes
        -----
        ICEEMDAN converges faster than CEEMDAN and typically needs fewer
        ensembles (30–50) for comparable mode quality. It is the recommended
        choice when ensemble averaging is needed.

        References
        ----------	o
        Colominas, M. A., & Schlotthauer, G. (2019).
        On the improved complete ensemble EMD and its application to intermittent
        signal detection. *Signal Processing*, 156, 246–258.
        """
        x = np.asarray(signal, dtype=np.float64)
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 20:
            return [x.copy()]

        x_std0 = float(np.std(x))
        if x_std0 < 1e-12 or np.allclose(x, x[0], atol=1e-12):
            return [x.copy()]

        n_ensembles = int(max(1, n_ensembles))
        max_imfs = int(max(1, max_imfs))

        noises, noise_bank = EMDVariants._precompute_noise_imf_bank(
            N=N,
            n_ensembles=n_ensembles,
            max_imfs=max_imfs,
            max_sifts_noise=int(max_sifts_noise),
            seed=seed,
            envelope_method=envelope_method,
        )

        imfs: list[np.ndarray] = []
        residual = x.copy()

        for k in range(max_imfs):
            res_std = float(np.std(residual))
            if res_std < float(epsilon) * x_std0:
                break
            if EMDVariants._is_monotonic(residual):
                break

            beta_k = float(noise_std) / float(k + 1)
            amp = beta_k * res_std

            local_mean_sum = np.zeros(N, dtype=np.float64)
            valid = 0

            for i in range(n_ensembles):
                e_k = EMDVariants._get_noise_component(noise_bank[i], noises[i], k)
                noisy = residual + amp * e_k

                trial_imfs = EMDVariants.emd(
                    noisy,
                    max_imfs=1,
                    max_sifts=1,
                    envelope_method=envelope_method,
                    **emd_kwargs,
                )
                if len(trial_imfs) > 0:
                    first_imf = np.asarray(trial_imfs[0], dtype=np.float64)
                    local_mean_sum += noisy - first_imf
                    valid += 1

            if valid == 0:
                break

            avg_local_mean = local_mean_sum / float(valid)
            imf_k = residual - avg_local_mean

            imfs.append(imf_k)
            residual = avg_local_mean

        if np.sum(residual**2) > (float(epsilon) ** 2) * (np.sum(x**2) + 1e-12):
            imfs.append(residual)

        return imfs

    @staticmethod
    def compute_orthogonality_index(imfs: list[np.ndarray]) -> float:
        """Compute the Orthogonality Index (OI) of a set of IMFs.

        The OI measures how orthogonal the IMF set is. For a perfect decomposition,
        all cross-correlations between modes should be zero. The OI ranges from 0
        (perfectly orthogonal) to higher values (more correlated).

        Parameters
        ----------
        imfs : list[np.ndarray]
            List of decomposed mode arrays.

        Returns
        -------
        float
            Orthogonality Index. Values < 0.01 indicate good orthogonality;
            values > 0.1 suggest significant mode mixing.

        Notes
        -----
        OI = 2·Σᵢ<ⱼ |⟨IMFᵢ, IMFⱼ⟩| / Σₖ ||IMFₖ||²

        EMD modes are not guaranteed to be orthogonal. CEEMDAN/ICEEMDAN typically
        produce more orthogonal mode sets than classic EMD.
        """
        if len(imfs) < 2:
            return 0.0
        n_imfs = len(imfs)
        total_energy = sum(np.sum(imf**2) for imf in imfs)
        if total_energy < 1e-12:
            return 0.0
        cross_terms = 0.0
        for i in range(n_imfs):
            for j in range(i + 1, n_imfs):
                cross_terms += np.abs(np.sum(imfs[i] * imfs[j]))
        oi = 2 * cross_terms / total_energy
        return float(oi)

    @staticmethod
    def compute_instantaneous_frequency(
        imf: np.ndarray, fs: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute instantaneous frequency and amplitude via the Hilbert transform.

        Parameters
        ----------
        imf : array-like
            A single IMF (must be approximately mono-component).
        fs : float
            Sampling frequency in Hz. Default 1.0 (normalised).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(inst_freq, amplitude)`` — both arrays of the same length as *imf*.
            ``inst_freq`` is clipped to [0, fs/2].

        Notes
        -----
        The Hilbert-based IF estimate is only valid for mono-component IMFs.
        Using it on multi-component modes produces meaningless results. Always
        verify that the input is a single IMF before calling this method.
        """
        analytic = hilbert(imf)
        analytic = hilbert(imf)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.gradient(phase) * fs / (2 * np.pi)
        inst_freq = np.clip(inst_freq, 0, fs / 2)
        return inst_freq, amplitude
