import random
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import interp1d
from scipy.signal import hilbert, resample, welch
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .ts_fengine_gan import *
from .ts_fengine_vae import *

# =============== DIFFUSION FOR SIGNAL AUGMENTATION ===============

class SignalDiffusion:
    """Diffusion-based noise injection and denoising for signal augmentation."""
    
    def __init__(self, signal_length: int = 1000, max_steps: int = 50):
        self.signal_length = signal_length
        self.max_steps = max_steps
        
        # Noise schedule
        self.betas = torch.linspace(0.0001, 0.02, max_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.betas = self.betas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
    
    def add_noise(self, signals: np.ndarray, noise_level: float = 0.5) -> np.ndarray:
        """Add controlled noise using diffusion schedule."""
        data = torch.FloatTensor(signals).to(self.device)
        
        # Choose noise level (timestep)
        t = int(noise_level * self.max_steps)
        t = min(t, self.max_steps - 1)
        
        # Add noise according to schedule
        noise = torch.randn_like(data)
        alpha_cumprod_t = self.alphas_cumprod[t]
        
        noisy_data = torch.sqrt(alpha_cumprod_t) * data + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        return noisy_data.cpu().numpy()
    
    def denoise(self, noisy_signals: np.ndarray, steps: int = 10) -> np.ndarray:
        """Denoise signals using iterative smoothing."""
        data = torch.FloatTensor(noisy_signals).to(self.device)
        
        # Iterative denoising with smoothing filter
        for _ in range(steps):
            # Apply smoothing filter
            kernel = torch.ones(1, 1, 5, device=self.device) / 5
            if data.dim() == 2:
                data_padded = F.pad(data.unsqueeze(1), (2, 2), mode='reflect')
            else:
                data_padded = F.pad(data, (2, 2), mode='reflect')
            
            smoothed = F.conv1d(data_padded, kernel, padding=0)
            if data.dim() == 2:
                smoothed = smoothed.squeeze(1)
            
            # Mix with original
            data = 0.7 * smoothed + 0.3 * data
        
        return data.cpu().numpy()


# =============== FACTORY FUNCTIONS (with backward compatibility) ===============

def create_signal_gan(signals: np.ndarray, signal_length: int = None, train_epochs: int = 200) -> ImprovedSignalGAN:
    """Create and train an improved GAN model for signal augmentation."""
    if signal_length is None:
        signal_length = signals.shape[1] if signals.ndim > 1 else len(signals)
    
    gan = TimeSeriesWGAN_GP(input_dim=len(signals), seq_len=24, latent_dim=16)
    
    if len(signals) > 0:
        print(f"Training GAN on {len(signals)} signals for {train_epochs} epochs...")
        gan.train(signals, epochs=train_epochs)
        print("GAN training complete!")
    
    return gan


def create_signal_vae(signals: np.ndarray, signal_length: int = None, train_epochs: int = 150) -> ImprovedSignalVAE:
    """Create and train an improved VAE model for signal augmentation."""
    if signal_length is None:
        signal_length = signals.shape[1] if signals.ndim > 1 else len(signals)
    
    vae = TimeSeriesVAE(input_dim=len(signals), latent_dim=16, seq_len=24)
    
    if len(signals) > 0:
        print(f"Training VAE on {len(signals)} signals for {train_epochs} epochs...")
        vae_trainer = TimeSeriesVAETrainer(vae)
        vae_trainer.train(signals, epochs=train_epochs)
        print("VAE training complete!")
    
    return vae


def create_signal_diffusion(signal_length: int = 1000) -> SignalDiffusion:
    """Create a diffusion model for signal augmentation (no training required)."""
    return SignalDiffusion(signal_length=signal_length)


class SignalFeatureEngineer:
    """
    Comprehensive feature engineering class for time-series signal analysis
    with support for returning feature names in order.
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
        features = []
        features.extend(self._extract_time_features(window))
        features.extend(self._extract_frequency_features(window))
        features.extend(self._extract_wavelet_features(window))
        features.extend(self._extract_envelope_features(window))
        return features

    def get_feature_names(self):
        names = []

        # Time-domain
        names.extend([
            "mean", "std", "kurtosis", "max", "min", "median",
            "percentile_25", "percentile_75", "rms", "energy",
            "mean_abs_diff", "zero_crossing_rate",
            "num_peaks", "mean_peaks", "std_peaks",
            "sample_entropy"
        ])

        # Frequency-domain
        for i, (low, high) in enumerate(self.frequency_bands):
            names.append(f"band_{i+1}_power")
            names.append(f"band_{i+1}_rel_power")

        names.extend([
            "spectral_entropy",
            "dominant_frequency",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectrum_skewness",
            "spectral_flatness"
        ])

        # Wavelet features
        for level in range(self.wavelet_levels + 1):
            names.extend([
                f"wavelet_L{level}_mean",
                f"wavelet_L{level}_std",
                f"wavelet_L{level}_entropy",
                f"wavelet_L{level}_energy"
            ])

        # Envelope
        names.extend([
            "envelope_mean",
            "envelope_std",
            "instantaneous_freq_mean",
            "instantaneous_freq_std"
        ])

        return names

    # === Feature Extraction Methods ===

    def _extract_time_features(self, window):
        from scipy.signal import find_peaks

        features = [
            np.mean(window),
            np.std(window),
            kurtosis(window),
            np.max(window),
            np.min(window),
            np.median(window),
            np.percentile(window, 25),
            np.percentile(window, 75),
            np.sqrt(np.mean(np.square(window))),
            np.sum(np.abs(window)),
        ]

        # Mean absolute diff
        features.append(np.mean(np.abs(np.diff(window))) if len(window) > 1 else 0)

        # Zero-crossing rate
        zc = np.where(np.diff(np.signbit(window)))[0].size
        features.append(zc / len(window) if len(window) > 1 else 0)

        # Peaks
        peaks, _ = find_peaks(window)
        if len(peaks) > 0:
            features.extend([
                len(peaks),
                np.mean(window[peaks]),
                np.std(window[peaks])
            ])
        else:
            features.extend([0, 0, 0])

        features.append(self._calculate_sample_entropy(window))
        return features

    def _extract_frequency_features(self, window):
        features = []
        try:
            f, Pxx = welch(window, fs=self.fs, nperseg=min(256, len(window) // 4))

            if len(Pxx) == 0:
                return [0] * (len(self.frequency_bands) * 2 + 6)

            total_power = np.sum(Pxx)
            for low, high in self.frequency_bands:
                band_mask = (f >= low) & (f <= high)
                band_power = np.sum(Pxx[band_mask]) if np.any(band_mask) else 0
                rel_power = band_power / total_power if total_power > 0 else 0
                features.extend([band_power, rel_power])

            if total_power > 0:
                norm_pxx = Pxx / total_power
                entropy = -np.sum(norm_pxx * np.log2(norm_pxx + 1e-10))
                centroid = np.sum(f * Pxx) / total_power
                bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * Pxx) / total_power)

                def _spectral_flatness(pxx):
                    pxx = np.maximum(pxx, 1e-10)
                    return np.exp(np.mean(np.log(pxx))) / np.mean(pxx)

                features.extend([
                    entropy,
                    f[np.argmax(Pxx)],
                    centroid,
                    bandwidth,
                    skew(Pxx),
                    _spectral_flatness(Pxx)
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])

        except Exception as e:
            print(f"Frequency feature extraction failed: {e}")
            features = [0] * (len(self.frequency_bands) * 2 + 6)

        return features

    def _extract_wavelet_features(self, window):
        features = []
        try:
            coeffs = pywt.wavedec(window, self.wavelet_name, level=self.wavelet_levels)
            for coeff in coeffs:
                if len(coeff) > 0:
                    features.extend([
                        np.mean(coeff),
                        np.std(coeff),
                        self._calculate_entropy(np.abs(coeff)),
                        np.sum(np.square(coeff))
                    ])
                else:
                    features.extend([0, 0, 0, 0])
        except Exception as e:
            print(f"Wavelet feature extraction failed: {e}")
            features = [0] * ((self.wavelet_levels + 1) * 4)

        return features

    def _extract_envelope_features(self, window):
        features = []
        try:
            analytic_signal = hilbert(window)
            amp_env = np.abs(analytic_signal)
            features.extend([
                np.mean(amp_env),
                np.std(amp_env)
            ])
            if len(window) > 1:
                inst_phase = np.unwrap(np.angle(analytic_signal))
                inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * self.fs
                features.extend([
                    np.mean(inst_freq),
                    np.std(inst_freq) if len(inst_freq) > 1 else 0
                ])
            else:
                features.extend([0, 0])
        except Exception as e:
            print(f"Envelope feature extraction failed: {e}")
            features = [0, 0, 0, 0]
        return features

    def _calculate_entropy(self, signal):
        if len(signal) == 0 or np.sum(signal) == 0:
            return 0
        norm_signal = signal / np.sum(signal)
        return -np.sum(norm_signal * np.log2(norm_signal + 1e-10))

    def _calculate_sample_entropy(self, signal, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(signal)
        N = len(signal)
        if N < m + 2:
            return 0

        def _phi(m):
            x = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.sum(np.sum(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
            return np.sum(C) / (N - m + 1)

        try:
            return -np.log(_phi(m + 1) / (_phi(m) + 1e-10))
        except Exception:
            return 0



class SignalAugmentor:
    """
    Comprehensive signal augmentation class combining traditional and state-of-the-art techniques
    for time-series signal data augmentation.
    """
    
    def __init__(self, fs: int = 2_000_000, random_state: Optional[int] = None):
        self.fs = fs
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            random.seed(random_state)
    
    # =============== TRADITIONAL AUGMENTATION METHODS ===============
    
    def time_shift(self, signal: np.ndarray, shift_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
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

    def time_mask(self, signal: np.ndarray, mask_frac: float = 0.1) -> np.ndarray:
        """Apply time masking by zeroing out a random segment"""
        length = len(signal)
        mask_len = int(mask_frac * length)
        start = np.random.randint(0, length - mask_len)
        masked = signal.copy()
        masked[start : start + mask_len] = 0
        return masked

    def add_noise(self, signal: np.ndarray, noise_level: Tuple[float, float] = (0.001, 0.05)) -> np.ndarray:
        """Add Gaussian noise to signal"""
        noise_factor = np.random.uniform(*noise_level)
        noise = np.random.normal(0, np.std(signal) * noise_factor, len(signal))
        return signal + noise

    def time_stretch(self, signal: np.ndarray, stretch_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply time stretching/compression"""
        stretch_factor = np.random.uniform(*stretch_range)
        new_len = int(len(signal) * stretch_factor)
        stretched = resample(signal, new_len)
        return resample(stretched, len(signal))

    def magnitude_warp(self, signal: np.ndarray, n_knots: int = 4, 
                      strength: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply magnitude warping"""
        x_knots = np.linspace(0, 1, n_knots)
        y_knots = np.random.uniform(*strength, n_knots)
        warp_func = interp1d(x_knots, y_knots, kind="cubic", fill_value="extrapolate")

        x = np.linspace(0, 1, len(signal))
        return signal * warp_func(x)

    def frequency_mask(self, signal: np.ndarray, mask_width_range: Tuple[float, float] = (0.05, 0.2), 
                      n_masks: int = 2) -> np.ndarray:
        """Apply frequency masking"""
        signal_f = np.fft.rfft(signal)
        n_freq = len(signal_f)

        for _ in range(n_masks):
            width = int(np.random.uniform(*mask_width_range) * n_freq)
            start = np.random.randint(0, max(1, n_freq - width))
            signal_f[start : start + width] = 0

        return np.fft.irfft(signal_f, len(signal))
    
    # =============== MIXUP VARIANTS ===============
    
    def mixup(self, signal1: np.ndarray, signal2: np.ndarray, 
              alpha: float = 0.2) -> Tuple[np.ndarray, float]:
        """
        MixUp augmentation for signals - linear interpolation between two signals.
        
        Args:
            signal1, signal2: Input signals
            alpha: Beta distribution parameter for mixing ratio
            
        Returns:
            Mixed signal and mixing parameter lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        mixed_signal = lam * signal1 + (1 - lam) * signal2
        return mixed_signal, lam
    
    def manifold_mixup(self, signals: List[np.ndarray], features: List[np.ndarray],
                      mixing_layer: str = 'feature', alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Manifold MixUp - mix in feature space rather than input space.
        
        Args:
            signals: List of input signals
            features: List of corresponding feature vectors
            mixing_layer: Where to apply mixing ('input' or 'feature')
            alpha: Beta distribution parameter
            
        Returns:
            Mixed signal, mixed features, mixing parameter
        """
        idx1, idx2 = np.random.choice(len(signals), 2, replace=False)
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        if mixing_layer == 'input':
            mixed_signal = lam * signals[idx1] + (1 - lam) * signals[idx2]
            mixed_features = lam * features[idx1] + (1 - lam) * features[idx2]
        else:  # feature space mixing
            mixed_signal = signals[idx1]  # Keep original signal
            mixed_features = lam * features[idx1] + (1 - lam) * features[idx2]
            
        return mixed_signal, mixed_features, lam
    
    def temporal_mixup(self, signal1: np.ndarray, signal2: np.ndarray,
                      mix_type: str = 'random', alpha: float = 0.2) -> np.ndarray:
        """
        Temporal MixUp - mix signals along time dimension with different strategies.
        
        Args:
            signal1, signal2: Input signals
            mix_type: 'random', 'block', 'interleave'
            alpha: Mixing parameter
            
        Returns:
            Temporally mixed signal
        """
        length = min(len(signal1), len(signal2))
        signal1, signal2 = signal1[:length], signal2[:length]
        
        if mix_type == 'random':
            # Random temporal mixing
            mask = np.random.random(length) < alpha
            mixed = signal1.copy()
            mixed[mask] = signal2[mask]
            
        elif mix_type == 'block':
            # Block-wise mixing
            block_size = int(length * alpha)
            start_idx = np.random.randint(0, length - block_size + 1)
            mixed = signal1.copy()
            mixed[start_idx:start_idx + block_size] = signal2[start_idx:start_idx + block_size]
            
        elif mix_type == 'interleave':
            # Interleaved mixing
            mixed = signal1.copy()
            step = max(1, int(1 / alpha))
            mixed[::step] = signal2[::step]
            
        return mixed
    
    # =============== CUTMIX VARIANTS ===============
    
    def cutmix_1d(self, signal1: np.ndarray, signal2: np.ndarray,
                  beta: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        CutMix for 1D signals - replace a continuous segment.
        
        Args:
            signal1, signal2: Input signals
            beta: Beta distribution parameter for cut ratio
            
        Returns:
            Mixed signal and mixing ratio
        """
        length = min(len(signal1), len(signal2))
        signal1, signal2 = signal1[:length], signal2[:length]
        
        # Sample cut ratio
        lam = np.random.beta(beta, beta)
        cut_length = int(length * (1 - lam))
        
        if cut_length > 0:
            # Random cut position
            cut_start = np.random.randint(0, length - cut_length + 1)
            cut_end = cut_start + cut_length
            
            mixed_signal = signal1.copy()
            mixed_signal[cut_start:cut_end] = signal2[cut_start:cut_end]
            
            # Actual mixing ratio
            actual_lam = 1 - cut_length / length
        else:
            mixed_signal = signal1.copy()
            actual_lam = 1.0
            
        return mixed_signal, actual_lam
    
    def multi_cutmix(self, signal1: np.ndarray, signal2: np.ndarray,
                     n_cuts: int = 3, total_cut_ratio: float = 0.3) -> np.ndarray:
        """
        Multiple CutMix - apply multiple cuts instead of one large cut.
        
        Args:
            signal1, signal2: Input signals
            n_cuts: Number of cuts to make
            total_cut_ratio: Total proportion to cut
            
        Returns:
            Mixed signal with multiple cuts
        """
        length = min(len(signal1), len(signal2))
        signal1, signal2 = signal1[:length], signal2[:length]
        
        mixed_signal = signal1.copy()
        total_cut_length = int(length * total_cut_ratio)
        cut_length = total_cut_length // n_cuts
        
        for _ in range(n_cuts):
            if cut_length > 0:
                cut_start = np.random.randint(0, length - cut_length + 1)
                cut_end = cut_start + cut_length
                mixed_signal[cut_start:cut_end] = signal2[cut_start:cut_end]
                
        return mixed_signal
    
    def frequency_cutmix(self, signal1: np.ndarray, signal2: np.ndarray,
                        cut_ratio: float = 0.3) -> np.ndarray:
        """
        Frequency domain CutMix - cut and paste in frequency domain.
        
        Args:
            signal1, signal2: Input signals
            cut_ratio: Ratio of frequency bins to replace
            
        Returns:
            Mixed signal
        """
        # FFT
        fft1 = np.fft.rfft(signal1)
        fft2 = np.fft.rfft(signal2)
        
        n_freqs = len(fft1)
        n_cut = int(n_freqs * cut_ratio)
        
        # Random frequency bands to cut
        cut_indices = np.random.choice(n_freqs, n_cut, replace=False)
        
        mixed_fft = fft1.copy()
        mixed_fft[cut_indices] = fft2[cut_indices]
        
        # IFFT back to time domain
        mixed_signal = np.fft.irfft(mixed_fft, len(signal1))
        
        return mixed_signal
    
    # =============== ADVANCED AUGMENTATIONS ===============
    
    def cutout(self, signal: np.ndarray, mask_ratio: float = 0.1,
               n_masks: int = 1) -> np.ndarray:
        """
        Cutout augmentation - zero out random segments.
        
        Args:
            signal: Input signal
            mask_ratio: Ratio of signal to mask
            n_masks: Number of masks to apply
            
        Returns:
            Signal with cutout masks
        """
        masked_signal = signal.copy()
        length = len(signal)
        mask_length = int(length * mask_ratio)
        
        for _ in range(n_masks):
            if mask_length > 0:
                start = np.random.randint(0, length - mask_length + 1)
                masked_signal[start:start + mask_length] = 0
                
        return masked_signal
    
    def time_warp_dtw(self, signal: np.ndarray, warp_strength: float = 0.2) -> np.ndarray:
        """
        Advanced time warping using DTW-inspired approach.
        
        Args:
            signal: Input signal
            warp_strength: Strength of warping (0-1)
            
        Returns:
            Time-warped signal
        """
        length = len(signal)
        
        # Create warping path with unique points
        max_warp = int(length * warp_strength)
        n_points = max(3, length // 100)  # At least 3 points
        
        # Generate unique warp points
        warp_points = np.linspace(0, length - 1, n_points)
        # Add some randomness while maintaining uniqueness
        if n_points > 2:
            # Add random offsets to middle points, keeping them sorted
            middle_indices = slice(1, -1)
            random_offsets = np.random.uniform(-length/(4*n_points), length/(4*n_points), n_points-2)
            warp_points[middle_indices] += random_offsets
            warp_points = np.sort(warp_points)  # Ensure sorted order
            warp_points = np.clip(warp_points, 0, length - 1)
        
        # Remove any duplicates that might have been created
        warp_points = np.unique(warp_points)
        
        # Ensure we have at least start and end points
        if len(warp_points) < 2:
            warp_points = np.array([0, length - 1])
        
        # Generate warp offsets
        warp_offsets = np.random.randint(-max_warp, max_warp + 1, size=len(warp_points))
        
        # Ensure endpoints are fixed
        warp_offsets[0] = 0
        warp_offsets[-1] = 0
        
        # Create new indices with warping
        new_indices = warp_points + warp_offsets
        new_indices = np.clip(new_indices, 0, length - 1)
        
        # Ensure new_indices are also unique and sorted
        if len(np.unique(new_indices)) < len(new_indices):
            # If duplicates exist, use linear interpolation instead of cubic
            interpolation_kind = 'linear'
        else:
            interpolation_kind = 'cubic' if len(warp_points) > 3 else 'linear'
        
        try:
            # Interpolate to get full warping path
            f = interp1d(warp_points, new_indices, kind=interpolation_kind, 
                        bounds_error=False, fill_value='extrapolate')
            warped_indices = f(np.arange(length))
            warped_indices = np.clip(warped_indices, 0, length - 1)
            
            # Apply warping
            warped_signal = np.interp(np.arange(length), warped_indices, signal)
            
        except Exception as e:
            print(f"Time warping failed, returning original signal: {e}")
            warped_signal = signal.copy()
        
        return warped_signal
    
    def spectral_augment(self, signal: np.ndarray, 
                        freq_mask_ratio: float = 0.1,
                        time_mask_ratio: float = 0.1) -> np.ndarray:
        """
        SpecAugment for time-series - frequency and time masking on spectrogram.
        
        Args:
            signal: Input signal
            freq_mask_ratio: Ratio of frequency bins to mask
            time_mask_ratio: Ratio of time steps to mask
            
        Returns:
            Augmented signal
        """
        # Compute spectrogram
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(signal, fs=self.fs, nperseg=256, noverlap=128)
        
        # Frequency masking
        n_freq_bins = len(f)
        freq_mask_size = int(n_freq_bins * freq_mask_ratio)
        if freq_mask_size > 0:
            freq_start = np.random.randint(0, n_freq_bins - freq_mask_size + 1)
            Sxx[freq_start:freq_start + freq_mask_size, :] = 0
        
        # Time masking
        n_time_bins = len(t)
        time_mask_size = int(n_time_bins * time_mask_ratio)
        if time_mask_size > 0:
            time_start = np.random.randint(0, n_time_bins - time_mask_size + 1)
            Sxx[:, time_start:time_start + time_mask_size] = 0
        
        # Reconstruct signal (simplified - using overlap-add)
        # In practice, you'd use a more sophisticated reconstruction
        augmented_signal = np.sum(Sxx, axis=0)
        augmented_signal = resample(augmented_signal, len(signal))
        
        return augmented_signal
    
    def adversarial_noise(self, signal: np.ndarray, model=None, target_label=None,
                         epsilon: float = 0.01, method: str = 'fgsm', 
                         num_steps: int = 10) -> np.ndarray:
        """
        Generate adversarial noise for robustness training.
        
        Args:
            signal: Input signal
            model: Trained PyTorch model (if available)
            target_label: Target label for targeted attacks
            epsilon: Perturbation magnitude
            method: 'fgsm', 'pgd', 'random', 'targeted', 'semantic'
            num_steps: Number of steps for iterative methods
            
        Returns:
            Signal with adversarial perturbation
        """
        if method == 'fgsm' and model is not None:
            # Fast Gradient Sign Method
            return self._fgsm_attack(signal, model, epsilon, target_label)
        elif method == 'pgd' and model is not None:
            # Projected Gradient Descent
            return self._pgd_attack(signal, model, epsilon, num_steps, target_label)
        elif method == 'random':
            # Random adversarial noise
            perturbation = np.random.uniform(-epsilon, epsilon, size=signal.shape)
            return signal + perturbation
        elif method == 'semantic':
            # Semantic adversarial perturbations
            return self._semantic_perturbation(signal, epsilon)
        else:
            # Frequency-based targeted perturbation
            return self._frequency_targeted_perturbation(signal, epsilon)
    
    def _fgsm_attack(self, signal: np.ndarray, model, epsilon: float, target_label=None) -> np.ndarray:
        """Fast Gradient Sign Method attack."""
        model.eval()
        signal_tensor = torch.tensor(signal, requires_grad=True, dtype=torch.float32).unsqueeze(0)
        
        # Forward pass
        if hasattr(model, 'predict_proba'):
            # Sklearn-style model
            signal_np = signal_tensor.detach().numpy().squeeze()
            pred = model.predict_proba(signal_np.reshape(1, -1))
            loss = -np.log(pred[0, target_label] + 1e-8) if target_label is not None else np.max(pred)
            # Approximate gradient with finite differences
            grad_sign = np.random.choice([-1, 1], size=signal.shape)
        else:
            # PyTorch model
            output = model(signal_tensor)
            
            if target_label is not None:
                # Targeted attack - minimize loss for target class
                loss = F.cross_entropy(output, torch.tensor([target_label]))
                loss = -loss  # Minimize target class loss
            else:
                # Untargeted attack - maximize loss for true class
                pred_class = output.argmax(dim=1)
                loss = F.cross_entropy(output, pred_class)
            
            # Backward pass
            loss.backward()
            grad_sign = signal_tensor.grad.sign().numpy().squeeze()
        
        # Generate adversarial example
        adversarial_signal = signal + epsilon * grad_sign
        return adversarial_signal
    
    def _pgd_attack(self, signal: np.ndarray, model, epsilon: float, 
                   num_steps: int, target_label=None) -> np.ndarray:
        """Projected Gradient Descent attack."""
        model.eval()
        adversarial_signal = signal.copy()
        step_size = epsilon / num_steps
        
        for _ in range(num_steps):
            signal_tensor = torch.tensor(adversarial_signal, requires_grad=True, dtype=torch.float32).unsqueeze(0)
            
            if hasattr(model, 'predict_proba'):
                # Sklearn-style model - use finite differences
                grad_sign = np.random.choice([-1, 1], size=signal.shape)
            else:
                # PyTorch model
                output = model(signal_tensor)
                
                if target_label is not None:
                    loss = -F.cross_entropy(output, torch.tensor([target_label]))
                else:
                    pred_class = output.argmax(dim=1)
                    loss = F.cross_entropy(output, pred_class)
                
                loss.backward()
                grad_sign = signal_tensor.grad.sign().numpy().squeeze()
            
            # Update adversarial example
            adversarial_signal = adversarial_signal + step_size * grad_sign
            
            # Project back to epsilon ball
            perturbation = adversarial_signal - signal
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            adversarial_signal = signal + perturbation
        
        return adversarial_signal
    
    def _semantic_perturbation(self, signal: np.ndarray, epsilon: float) -> np.ndarray:
        """Generate semantic adversarial perturbations."""
        # Time-shift perturbation
        shift = int(epsilon * len(signal) * 0.1)
        if shift > 0:
            shifted = np.roll(signal, shift)
        else:
            shifted = signal.copy()
        
        # Frequency perturbation
        fft_signal = np.fft.rfft(shifted)
        noise_fft = epsilon * np.random.normal(0, 1, len(fft_signal)) * np.abs(fft_signal)
        perturbed_fft = fft_signal + noise_fft
        perturbed_signal = np.fft.irfft(perturbed_fft, len(signal))
        
        return perturbed_signal
    
    def _frequency_targeted_perturbation(self, signal: np.ndarray, epsilon: float) -> np.ndarray:
        """Generate frequency-targeted perturbations."""
        # Target specific frequency bands
        fft_signal = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/self.fs)
        
        # Focus perturbation on high-energy frequency bands
        power = np.abs(fft_signal)
        high_energy_mask = power > np.percentile(power, 70)
        
        perturbation_fft = np.zeros_like(fft_signal)
        perturbation_fft[high_energy_mask] = epsilon * np.random.normal(0, 1, np.sum(high_energy_mask))
        
        perturbation = np.fft.irfft(perturbation_fft, len(signal))
        return signal + perturbation
    
    # =============== GENERATIVE AUGMENTATION ===============
    
    def gan_augment(self, signal: np.ndarray, gan_model=None, noise_dim: int = 100,
                   augment_ratio: float = 0.3) -> np.ndarray:
        """
        GAN-based augmentation using a simple 1D GAN architecture.
        
        Args:
            signal: Input signal
            gan_model: Pre-trained GAN model (Generator)
            noise_dim: Dimension of noise input to generator
            augment_ratio: How much of the signal to replace with GAN output
            
        Returns:
            GAN-augmented signal
        """

        with torch.no_grad():
            noise = torch.randn(1, noise_dim)
            if hasattr(gan_model, 'generate'):
                generated = gan_model.generate(noise).numpy().squeeze()
            else:
                generated = gan_model(noise).numpy().squeeze()
            
            # Resize to match input signal length
            if len(generated) != len(signal):
                from scipy.signal import resample
                generated = resample(generated, len(signal))
        
        # Mix original signal with generated signal
        lam = 1 - augment_ratio
        augmented_signal = lam * signal + augment_ratio * generated
        
        return augmented_signal
    
    def vae_augment(self, signal: np.ndarray, vae_model=None, latent_dim: int = 50,
                   noise_scale: float = 0.1) -> np.ndarray:
        """
        VAE-based augmentation using Variational Autoencoder.
        
        Args:
            signal: Input signal
            vae_model: Pre-trained VAE model
            latent_dim: Dimension of latent space
            noise_scale: Scale of noise to add in latent space
            
        Returns:
            VAE-augmented signal
        """

        # Use provided VAE model
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            if hasattr(vae_model, 'encode') and hasattr(vae_model, 'decode'):
                # Standard VAE interface
                mu, logvar = vae_model.encode(signal_tensor)
                
                # Add noise in latent space
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std) * noise_scale
                perturbed_z = mu + eps * std
                
                # Decode back to signal space
                reconstructed = vae_model.decode(perturbed_z)
                return reconstructed.numpy().squeeze()
            else:
                # Generic forward pass
                reconstructed = vae_model(signal_tensor)
                return reconstructed.numpy().squeeze()
    
    def diffusion_augment(self, signal: np.ndarray, diffusion_model=None, 
                         noise_steps: int = 100, noise_strength: float = 0.1) -> np.ndarray:
        """
        Diffusion model-based augmentation.
        
        Args:
            signal: Input signal
            diffusion_model: Pre-trained diffusion model
            noise_steps: Number of diffusion steps
            noise_strength: Strength of noise injection
            
        Returns:
            Diffusion-augmented signal
        """
        return diffusion_model.sample(signal)
    
    # =============== BATCH AUGMENTATION ===============
    
    def batch_mixup(self, signals: np.ndarray, labels: np.ndarray,
                   alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply MixUp to a batch of signals.
        
        Args:
            signals: Batch of signals (batch_size, signal_length)
            labels: Corresponding labels
            alpha: MixUp parameter
            
        Returns:
            Mixed signals, original labels, shuffled labels, mixing ratios
        """
        batch_size = signals.shape[0]
        indices = np.random.permutation(batch_size)
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha, batch_size)
        else:
            lam = np.ones(batch_size)
            
        mixed_signals = lam.reshape(-1, 1) * signals + (1 - lam.reshape(-1, 1)) * signals[indices]
        
        return mixed_signals, labels, labels[indices], lam
    
    def apply_random_augmentation(self, signal: np.ndarray, other_signal: np.ndarray = None,
                                 n_augmentations: int = 2, 
                                 augmentation_config: dict = None,
                                 method_weights: dict = None) -> np.ndarray:
        """
        Apply random combination of traditional and modern augmentations.
        
        Args:
            signal: Primary input signal
            other_signal: Secondary signal for mixing operations
            n_augmentations: Number of augmentations to apply
            augmentation_config: Configuration for augmentations
            method_weights: Weights for different augmentation categories
            
        Returns:
            Augmented signal
        """
        config = augmentation_config or {
            'mixup_alpha': 0.2,
            'cutmix_beta': 1.0,
            'cutout_ratio': 0.1,
            'time_warp_strength': 0.2,
            'noise_level': 0.05,
            'shift_range': (-0.1, 0.1),
            'stretch_range': (0.8, 1.2),
            'mask_frac': 0.1
        }
        
        weights = method_weights or {
            'traditional': 0.4,
            'modern_single': 0.3,
            'modern_dual': 0.3
        }
        
        # Traditional single signal augmentations
        traditional_methods = [
            lambda x: self.time_shift(x, config['shift_range']),
            lambda x: self.add_noise(x, (0.001, config['noise_level'])),
            lambda x: self.time_stretch(x, config['stretch_range']),
            lambda x: self.magnitude_warp(x),
            lambda x: self.frequency_mask(x),
            lambda x: self.time_mask(x, config['mask_frac'])
        ]
        
        # Modern single signal augmentations
        modern_single_methods = [
            lambda x: self.cutout(x, config['cutout_ratio']),
            lambda x: self.time_warp_dtw(x, config['time_warp_strength']),
            lambda x: self.spectral_augment(x),
            lambda x: self.adversarial_noise(x, epsilon=config['noise_level'], method='semantic'),
            lambda x: self.gan_augment(x, augment_ratio=0.2),
            lambda x: self.vae_augment(x, noise_scale=0.1),
            lambda x: self.diffusion_augment(x, noise_steps=20, noise_strength=0.05)
        ]
        
        # Modern dual signal augmentations (require other_signal)
        modern_dual_methods = []
        if other_signal is not None:
            modern_dual_methods = [
                lambda x: self.mixup(x, other_signal, config['mixup_alpha'])[0],
                lambda x: self.cutmix_1d(x, other_signal, config['cutmix_beta'])[0],
                lambda x: self.frequency_cutmix(x, other_signal),
                lambda x: self.temporal_mixup(x, other_signal, 'random', config['mixup_alpha']),
                lambda x: self.multi_cutmix(x, other_signal, n_cuts=3, total_cut_ratio=0.3)
            ]
        
        # Weighted selection of method categories
        method_categories = []
        if traditional_methods:
            method_categories.extend(['traditional'] * int(weights['traditional'] * 10))
        if modern_single_methods:
            method_categories.extend(['modern_single'] * int(weights['modern_single'] * 10))
        if modern_dual_methods:
            method_categories.extend(['modern_dual'] * int(weights['modern_dual'] * 10))
        
        if not method_categories:
            return signal
        
        # Select methods from different categories
        selected_methods = []
        for _ in range(n_augmentations):
            category = random.choice(method_categories)
            
            if category == 'traditional':
                method = random.choice(traditional_methods)
            elif category == 'modern_single':
                method = random.choice(modern_single_methods)
            elif category == 'modern_dual':
                method = random.choice(modern_dual_methods)
            
            selected_methods.append(method)
        
        # Apply selected augmentations
        augmented_signal = signal.copy()
        for method in selected_methods:
            try:
                new_signal = method(augmented_signal)
                if (new_signal.shape == signal.shape and 
                    not np.isnan(new_signal).any() and 
                    np.any(new_signal)):
                    augmented_signal = new_signal
            except Exception as e:
                print(f"Augmentation method failed: {e}")
                continue
                
        return augmented_signal
    
    # =============== CONVENIENCE METHODS ===============
    
    def apply_traditional_augmentation(self, signal: np.ndarray, n_augmentations: int = 2) -> np.ndarray:
        """Apply only traditional augmentation methods"""
        traditional_methods = [
            self.time_shift,
            self.add_noise,
            self.time_stretch,
            self.magnitude_warp,
            self.frequency_mask,
            self.time_mask
        ]

        selected_methods = random.sample(
            traditional_methods, min(n_augmentations, len(traditional_methods))
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
                print(f"Traditional augmentation method failed: {e}")
                continue

        return augmented_signal
    
    def apply_modern_augmentation(self, signal: np.ndarray, other_signal: np.ndarray = None,
                                 n_augmentations: int = 2) -> np.ndarray:
        """Apply only modern augmentation methods"""
        single_methods = [
            lambda x: self.cutout(x),
            lambda x: self.time_warp_dtw(x),
            lambda x: self.adversarial_noise(x, method='semantic'),
            lambda x: self.gan_augment(x),
            lambda x: self.vae_augment(x)
        ]
        
        dual_methods = []
        if other_signal is not None:
            dual_methods = [
                lambda x: self.mixup(x, other_signal)[0],
                lambda x: self.cutmix_1d(x, other_signal)[0],
                lambda x: self.frequency_cutmix(x, other_signal)
            ]
        
        all_methods = single_methods + dual_methods
        
        if not all_methods:
            return signal
            
        selected_methods = random.sample(
            all_methods, min(n_augmentations, len(all_methods))
        )
        
        augmented_signal = signal.copy()
        for method in selected_methods:
            try:
                new_signal = method(augmented_signal)
                if (new_signal.shape == signal.shape and 
                    not np.isnan(new_signal).any() and 
                    np.any(new_signal)):
                    augmented_signal = new_signal
            except Exception as e:
                print(f"Modern augmentation method failed: {e}")
                continue
                
        return augmented_signal


class SignalProcessor:
    def __init__(self, fs=2_000_000, frequency_bands=None, random_state=None):
        self.feature_engineer = SignalFeatureEngineer(fs, frequency_bands, random_state)
        self.augmentor = SignalAugmentor(fs, random_state)
        self.fs = fs
        self.selected_feature_names = None  # new

    def process_signals(
        self,
        signals,
        labels,
        window_size=1000,
        step_size=1000,
        augment=False,
        augment_factor=2,
        select_features=False,   # NEW
        feature_selection_config=None  # optional config
    ):
        features = []
        feature_labels = []
        raw_windows = []
        window_labels = []

        for signal_id, signal_data in signals.items():
            label = labels[signal_id]
            for i in range(0, len(signal_data) - window_size + 1, step_size):
                window = signal_data[i: i + window_size]
                try:
                    window_features = self.feature_engineer.extract_features(window)
                    features.append(window_features)
                    feature_labels.append(label)
                    raw_windows.append(window)
                    window_labels.append(label)
                except Exception as e:
                    print(f"Error processing window from {signal_id}: {e}")
                    continue

        features = np.array(features)
        feature_labels = np.array(feature_labels)
        raw_windows = np.array(raw_windows)
        window_labels = np.array(window_labels)

        if augment:
            features, feature_labels = self._apply_augmentation(
                raw_windows, window_labels, features, feature_labels, augment_factor
            )
            features, feature_labels = self.mixup_features(features, feature_labels)

        feature_names = self.feature_engineer.get_feature_names()

        if select_features:
            config = feature_selection_config or {
                "min_variance": 1e-4,
                "max_corr": 0.95,
                "min_info_gain": 1e-3
            }
            features, feature_names = self._select_features(
                features, feature_labels, feature_names, **config
            )
            self.selected_feature_names = feature_names
        else:
            self.selected_feature_names = feature_names

        return features, feature_labels, raw_windows, window_labels

    def mixup_features(self, features, labels, alpha=0.2, n_mix=100):
        mixed_features = []
        mixed_labels = []

        for _ in range(n_mix):
            i, j = np.random.choice(len(features), 2, replace=False)
            lam = np.random.beta(alpha, alpha)

            x_mix = lam * features[i] + (1 - lam) * features[j]
            y_mix = lam * labels[i] + (1 - lam) * labels[j]  # use soft labels or round

            mixed_features.append(x_mix)
            mixed_labels.append(y_mix)

        return np.vstack([features, mixed_features]), np.hstack([labels, mixed_labels])

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



    def _select_features(self, X, y, feature_names, min_variance=1e-4, max_corr=0.95, min_info_gain=1e-3):
        # 1. Variance filter
        var_mask = np.var(X, axis=0) > min_variance
        X = X[:, var_mask]
        feature_names = [f for f, keep in zip(feature_names, var_mask) if keep]

        # 2. Correlation filter
        corr = np.corrcoef(X.T)
        upper = np.triu(np.ones_like(corr), k=1).astype(bool)
        to_remove = set()
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[1]):
                if abs(corr[i, j]) > max_corr:
                    to_remove.add(j)
        keep_idx = [i for i in range(X.shape[1]) if i not in to_remove]
        X = X[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]

        # 3. Mutual Information
        info = mutual_info_classif(MinMaxScaler().fit_transform(X), y, discrete_features='auto')
        info_mask = info > min_info_gain
        X = X[:, info_mask]
        feature_names = [f for f, keep in zip(feature_names, info_mask) if keep]

        return X, feature_names

    def get_selected_feature_names(self):
        """Return names of features used after selection (if any)"""
        return self.selected_feature_names
