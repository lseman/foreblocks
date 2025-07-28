import random

import numpy as np
import pywt
from scipy.interpolate import interp1d
from scipy.signal import hilbert, resample, welch
from scipy.stats import kurtosis


class SignalFeatureEngineer:
    """
    Comprehensive feature engineering class for time-series signal analysis
    """

    def __init__(self, fs=2_000_000, frequency_bands=None, random_state=None):
        self.fs = fs
        self.frequency_bands = frequency_bands or self._get_default_bands()

        # Configuration for different signal types
        self.wavelet_name = "db4"
        self.wavelet_levels = 4

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _get_default_bands(self):
        """Get default frequency bands based on Nyquist frequency"""
        nyquist = self.fs / 2
        return [
            (0.05 * nyquist, 0.15 * nyquist),
            (0.15 * nyquist, 0.25 * nyquist),
            (0.25 * nyquist, 0.375 * nyquist),
            (0.375 * nyquist, 0.5 * nyquist),
            (0.5 * nyquist, 0.75 * nyquist),
            (0.75 * nyquist, 0.95 * nyquist),
        ]

    def extract_features(self, window):
        """
        Extract comprehensive feature set from signal window

        Args:
            window (array): Signal window to analyze

        Returns:
            list: Feature vector
        """
        features = []

        # Time-domain features
        features.extend(self._extract_time_features(window))

        # Frequency-domain features
        features.extend(self._extract_frequency_features(window))

        # Wavelet features
        features.extend(self._extract_wavelet_features(window))

        # Envelope features
        features.extend(self._extract_envelope_features(window))

        return features

    def _extract_time_features(self, window):
        """Extract time-domain statistical features"""
        features = []

        # Basic statistics
        features.extend(
            [
                np.mean(window),
                np.std(window),
                kurtosis(window),
                np.max(window),
                np.min(window),
                np.median(window),
                np.percentile(window, 25),
                np.percentile(window, 75),
                np.sqrt(np.mean(np.square(window))),  # RMS
                np.sum(np.abs(window)),  # Energy
            ]
        )

        # Derivative features
        if len(window) > 1:
            features.append(np.mean(np.abs(np.diff(window))))
        else:
            features.append(0)

        # Zero crossing rate
        if len(window) > 1:
            zero_crossings = np.where(np.diff(np.signbit(window)))[0].size
            features.append(zero_crossings / len(window))
        else:
            features.append(0)

        # Peak analysis
        if len(window) > 2:
            peak_indices = (
                np.where((window[1:-1] > window[:-2]) & (window[1:-1] > window[2:]))[0]
                + 1
            )
            if len(peak_indices) > 0:
                features.extend(
                    [
                        len(peak_indices),
                        np.mean(window[peak_indices]),
                        np.std(window[peak_indices]),
                    ]
                )
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])

        return features

    def _extract_frequency_features(self, window):
        """Extract frequency-domain features"""
        features = []

        try:
            f, Pxx = welch(window, fs=self.fs, nperseg=min(256, len(window) // 4))

            if len(Pxx) == 0:
                return [0] * (len(self.frequency_bands) * 2 + 4)

            # Band power features
            total_power = np.sum(Pxx)
            for low, high in self.frequency_bands:
                band_mask = (f >= low) & (f <= high)
                if np.any(band_mask):
                    band_power = np.sum(Pxx[band_mask])
                    features.append(band_power)
                    features.append(band_power / total_power if total_power > 0 else 0)
                else:
                    features.extend([0, 0])

            # Spectral features
            if total_power > 0:
                # Spectral entropy
                norm_pxx = Pxx / total_power
                spectral_entropy = -np.sum(norm_pxx * np.log2(norm_pxx + 1e-10))
                features.append(spectral_entropy)

                # Dominant frequency
                dominant_freq = f[np.argmax(Pxx)]
                features.append(dominant_freq)

                # Spectral centroid
                spectral_centroid = np.sum(f * Pxx) / total_power
                features.append(spectral_centroid)

                # Spectral bandwidth
                spectral_variance = (
                    np.sum(((f - spectral_centroid) ** 2) * Pxx) / total_power
                )
                features.append(np.sqrt(spectral_variance))
            else:
                features.extend([0, 0, 0, 0])

        except Exception as e:
            print(f"Frequency feature extraction failed: {e}")
            features = [0] * (len(self.frequency_bands) * 2 + 4)

        return features

    def _extract_wavelet_features(self, window):
        """Extract wavelet-based features"""
        features = []

        try:
            coeffs = pywt.wavedec(window, self.wavelet_name, level=self.wavelet_levels)

            for coeff in coeffs:
                if len(coeff) > 0:
                    features.extend(
                        [
                            np.mean(coeff),
                            np.std(coeff),
                            self._calculate_entropy(np.abs(coeff)),
                        ]
                    )
                else:
                    features.extend([0, 0, 0])

        except Exception as e:
            print(f"Wavelet feature extraction failed: {e}")
            features = [0] * ((self.wavelet_levels + 1) * 3)

        return features

    def _extract_envelope_features(self, window):
        """Extract envelope-based features using Hilbert transform"""
        features = []

        try:
            analytic_signal = hilbert(window)
            amplitude_envelope = np.abs(analytic_signal)

            features.extend([np.mean(amplitude_envelope), np.std(amplitude_envelope)])

            if len(window) > 1:
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                instantaneous_frequency = (
                    np.diff(instantaneous_phase) / (2.0 * np.pi) * self.fs
                )

                features.extend(
                    [
                        np.mean(instantaneous_frequency),
                        (
                            np.std(instantaneous_frequency)
                            if len(instantaneous_frequency) > 1
                            else 0
                        ),
                    ]
                )
            else:
                features.extend([0, 0])

        except Exception as e:
            print(f"Envelope feature extraction failed: {e}")
            features = [0, 0, 0, 0]

        return features

    def _calculate_entropy(self, signal):
        """Calculate entropy of signal"""
        if len(signal) == 0 or np.sum(signal) == 0:
            return 0

        norm_signal = signal / np.sum(signal)
        return -np.sum(norm_signal * np.log2(norm_signal + 1e-10))


class SignalAugmentor:
    """
    Signal augmentation class for data augmentation
    """

    def __init__(self, fs=2_000_000, random_state=None):
        self.fs = fs
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def time_shift(self, signal, shift_range=(-0.1, 0.1)):
        """Apply time shifting augmentation"""
        shift_factor = np.random.uniform(*shift_range)
        shift_amount = int(len(signal) * shift_factor)

        augmented = np.zeros_like(signal)
        if shift_amount > 0:
            augmented[shift_amount:] = signal[:-shift_amount]
        elif shift_amount < 0:
            augmented[:shift_amount] = signal[-shift_amount:]
        else:
            augmented = signal.copy()

        return augmented

    def add_noise(self, signal, noise_level=(0.001, 0.05)):
        """Add Gaussian noise to signal"""
        noise_factor = np.random.uniform(*noise_level)
        noise = np.random.normal(0, np.std(signal) * noise_factor, len(signal))
        return signal + noise

    def time_stretch(self, signal, stretch_range=(0.8, 1.2)):
        """Apply time stretching/compression"""
        stretch_factor = np.random.uniform(*stretch_range)
        new_len = int(len(signal) * stretch_factor)
        stretched = resample(signal, new_len)
        return resample(stretched, len(signal))

    def magnitude_warp(self, signal, n_knots=4, strength=(0.8, 1.2)):
        """Apply magnitude warping"""
        x_knots = np.linspace(0, 1, n_knots)
        y_knots = np.random.uniform(*strength, n_knots)
        warp_func = interp1d(x_knots, y_knots, kind="cubic", fill_value="extrapolate")

        x = np.linspace(0, 1, len(signal))
        return signal * warp_func(x)

    def frequency_mask(self, signal, mask_width_range=(0.05, 0.2), n_masks=2):
        """Apply frequency masking"""
        signal_f = np.fft.rfft(signal)
        n_freq = len(signal_f)

        for _ in range(n_masks):
            width = int(np.random.uniform(*mask_width_range) * n_freq)
            start = np.random.randint(0, max(1, n_freq - width))
            signal_f[start : start + width] = 0

        return np.fft.irfft(signal_f, len(signal))

    def apply_random_augmentation(self, signal, n_augmentations=2):
        """Apply random combination of augmentations"""
        augmentation_methods = [
            self.time_shift,
            self.add_noise,
            self.time_stretch,
            self.magnitude_warp,
            self.frequency_mask,
        ]

        selected_methods = random.sample(
            augmentation_methods, min(n_augmentations, len(augmentation_methods))
        )

        augmented_signal = signal.copy()
        for method in selected_methods:
            try:
                new_signal = method(augmented_signal)
                if (
                    new_signal.shape == signal.shape
                    and not np.isnan(new_signal).any()
                    and np.any(new_signal)
                ):
                    augmented_signal = new_signal
            except Exception as e:
                print(f"Augmentation method failed: {e}")
                continue

        return augmented_signal


class SignalProcessor:
    """
    Main processing class that combines feature extraction and augmentation
    """

    def __init__(self, fs=2_000_000, frequency_bands=None, random_state=None):
        self.feature_engineer = SignalFeatureEngineer(fs, frequency_bands, random_state)
        self.augmentor = SignalAugmentor(fs, random_state)
        self.fs = fs

    def process_signals(
        self,
        signals,
        labels,
        window_size=1000,
        step_size=1000,
        augment=False,
        augment_factor=2,
    ):
        """
        Process multiple signals with windowing and optional augmentation

        Args:
            signals (dict): Dictionary of signal_id -> signal_data
            labels (dict): Dictionary of signal_id -> label
            window_size (int): Size of analysis window
            step_size (int): Step size between windows
            augment (bool): Whether to apply augmentation
            augment_factor (int): Factor for augmentation

        Returns:
            tuple: (features, labels, raw_windows, window_labels)
        """
        features = []
        feature_labels = []
        raw_windows = []
        window_labels = []

        # Extract features from original signals
        for signal_id, signal_data in signals.items():
            label = labels[signal_id]

            for i in range(0, len(signal_data) - window_size + 1, step_size):
                window = signal_data[i : i + window_size]

                try:
                    # Extract features
                    window_features = self.feature_engineer.extract_features(window)
                    features.append(window_features)
                    feature_labels.append(label)

                    # Store raw window
                    raw_windows.append(window)
                    window_labels.append(label)

                except Exception as e:
                    print(f"Error processing window from {signal_id}: {e}")
                    continue

        # Convert to numpy arrays
        features = np.array(features)
        feature_labels = np.array(feature_labels)
        raw_windows = np.array(raw_windows)
        window_labels = np.array(window_labels)

        # Apply augmentation if requested
        if augment:
            features, feature_labels = self._apply_augmentation(
                raw_windows, window_labels, features, feature_labels, augment_factor
            )

        return features, feature_labels, raw_windows, window_labels

    def _apply_augmentation(
        self, raw_windows, window_labels, features, feature_labels, factor
    ):
        """Apply augmentation to balance dataset"""
        # Get class distribution
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        max_count = np.max(counts)

        augmented_features = list(features)
        augmented_labels = list(feature_labels)

        for label, count in zip(unique_labels, counts):
            if count < max_count:
                # Get samples for this class
                class_indices = np.where(window_labels == label)[0]
                class_windows = raw_windows[class_indices]

                # Calculate how many to generate
                n_to_generate = min((max_count - count) * factor, max_count - count)

                for _ in range(n_to_generate):
                    # Select random sample to augment
                    sample_idx = np.random.randint(0, len(class_windows))
                    sample = class_windows[sample_idx]

                    # Apply augmentation
                    try:
                        augmented_sample = self.augmentor.apply_random_augmentation(
                            sample
                        )
                        augmented_features_vec = self.feature_engineer.extract_features(
                            augmented_sample
                        )

                        augmented_features.append(augmented_features_vec)
                        augmented_labels.append(label)

                    except Exception as e:
                        print(f"Augmentation failed: {e}")
                        continue

        return np.array(augmented_features), np.array(augmented_labels)

    def extract_features_from_window(self, window):
        """Extract features from a single window"""
        return self.feature_engineer.extract_features(window)

    def augment_signal(self, signal, n_augmentations=2):
        """Augment a single signal"""
        return self.augmentor.apply_random_augmentation(signal, n_augmentations)
