# Standard Library
import warnings
from typing import Optional, Union

# Scientific Computing and Visualization
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit, prange
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..tf.transformer import TransformerDecoder, TransformerEncoder

# Optional imports
try:
    from pykalman import KalmanFilter
except ImportError:
    KalmanFilter = None

try:
    from PyEMD import EMD
except ImportError:
    EMD = None



def _remove_outliers_parallel(index, col, method, threshold):
    cleaned = _remove_outliers_wrapper((index, col, method, threshold))
    return cleaned


@njit
def fast_mad_outlier_removal(x: np.ndarray, threshold: float) -> np.ndarray:
    valid = ~np.isnan(x)
    if np.sum(valid) < 5:
        return x  # not enough data

    med = np.nanmedian(x)
    deviations = np.abs(x - med)
    mad = np.nanmedian(deviations) + 1e-8

    # Optional robustness clamp
    if mad < 1e-6:
        mad = np.nanmean(deviations) + 1e-8

    # Apply modified Z-score
    mod_z = np.abs((x - med) / mad) * 1.4826

    # Apply adaptive threshold (optional nonlinear taper)
    adapt_thresh = threshold + 0.5 * (np.std(mod_z[valid]) > 3.5)

    return np.where(mod_z > adapt_thresh, np.nan, x)


@njit
def fast_quantile_outlier_removal(
    x: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    return np.where((x < lower) | (x > upper), np.nan, x)


@njit(parallel=True)
def fast_zscore_outlier_removal(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Numba-accelerated Z-score outlier removal.
    """
    mean = np.nanmean(x)
    std = np.nanstd(x) + 1e-8
    n = x.shape[0]
    result = np.copy(x)
    for i in prange(n):
        if not np.isnan(x[i]):
            z = abs((x[i] - mean) / std)
            if z > threshold:
                result[i] = np.nan
    return result


@njit(parallel=True)
def fast_iqr_outlier_removal(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Numba-accelerated IQR outlier removal.
    """
    q1 = np.percentile(x[~np.isnan(x)], 25)
    q3 = np.percentile(x[~np.isnan(x)], 75)
    iqr = q3 - q1 + 1e-8
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    n = x.shape[0]
    result = np.copy(x)
    for i in prange(n):
        if not np.isnan(x[i]) and (x[i] < lower or x[i] > upper):
            result[i] = np.nan
    return result


def _remove_outliers(
    data_col: np.ndarray, method: str, threshold: float, **kwargs
) -> np.ndarray:
    """
    Remove outliers from a univariate or multivariate time series using the specified method.
    Replaces detected outliers with np.nan.

    Parameters:
        data_col: np.ndarray of shape (T,) or (T, D)
        method: One of ["zscore", "iqr", "mad", "quantile", "isolation_forest", "lof", "ecod", "tranad"]
        threshold: method-dependent threshold (e.g. 0.95 for percentile methods)
        **kwargs: Optional method-specific config (e.g. seq_len, epochs for tranad)

    Returns:
        np.ndarray of same shape as input, with outliers replaced by np.nan
    """
    data_col = np.asarray(data_col)
    is_multivariate = data_col.ndim == 2
    x = data_col.copy().astype(np.float64)

    if x.size == 0 or np.isnan(x).all():
        return x

    def mask_to_nan(mask: np.ndarray) -> np.ndarray:
        if is_multivariate:
            return np.where(mask[:, None], np.nan, x)
        else:
            return np.where(mask, np.nan, x)

    # === Univariate-only methods ===
    if not is_multivariate:
        if method == "zscore":
            return fast_zscore_outlier_removal(x, threshold)
        elif method == "iqr":
            return fast_iqr_outlier_removal(x, threshold)
        elif method == "mad":
            return fast_mad_outlier_removal(x, threshold)
        elif method == "quantile":
            q1, q3 = np.nanpercentile(x, [threshold * 100, 100 - threshold * 100])
            return fast_quantile_outlier_removal(x, q1, q3)

    # === Multivariate-aware methods ===
    if method == "isolation_forest":
        model = IsolationForest(contamination=threshold, random_state=42)
        pred = model.fit_predict(x if is_multivariate else x.reshape(-1, 1))
        return mask_to_nan(pred != 1)

    elif method == "lof":
        model = LocalOutlierFactor(n_neighbors=20, contamination=threshold)
        pred = model.fit_predict(x if is_multivariate else x.reshape(-1, 1))
        return mask_to_nan(pred != 1)

    elif method == "ecod":
        try:
            from pyod.models.ecod import ECOD

            model = ECOD()
            pred = model.fit(x if is_multivariate else x.reshape(-1, 1)).predict(
                x if is_multivariate else x.reshape(-1, 1)
            )
            return mask_to_nan(pred == 1)
        except ImportError:
            warnings.warn("pyod not installed. Falling back to IQR.")
            if not is_multivariate:
                Q1, Q3 = np.percentile(x, [25, 75])
                IQR = Q3 - Q1 + 1e-8
                return mask_to_nan(
                    (x < Q1 - threshold * IQR) | (x > Q3 + threshold * IQR)
                )
            else:
                raise ValueError("ECOD fallback does not support multivariate input.")

    elif method == "tranad":
        from sklearn.preprocessing import StandardScaler

        seq_len = kwargs.get("seq_len", 24)
        epochs = kwargs.get("epochs", 10)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        adaptive = kwargs.get("adaptive", True)
        min_z = kwargs.get("z_threshold", 3.0)

        # Ensure 2D format
        if data_col.ndim == 1:
            data_col = data_col.reshape(-1, 1)
        x = data_col.astype(np.float64)
        T, D = x.shape

        if T < seq_len + 5:
            return x if D > 1 else x.flatten()

        # === Normalize each feature independently ===
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # === Run TranAD ===
        detector = TranADDetector(seq_len=seq_len, epochs=epochs, device=device)
        scores = detector.fit_predict(x_scaled)  # shape (T - seq_len,)

        # === Adaptive thresholding ===
        if adaptive:
            score_z = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
            anomaly_mask = np.full(T, False)
            # Map scores to the end of each sequence window
            anomaly_mask[seq_len-1:seq_len-1+len(score_z)] = score_z > min_z
        else:
            if threshold > 1.0:
                percentile = min(max(threshold, 0), 100)
            else:
                percentile = threshold * 100
            score_thresh = np.nanpercentile(scores, percentile)
            anomaly_mask = np.full(T, False)
            # Map scores to the end of each sequence window
            anomaly_mask[seq_len-1:seq_len-1+len(scores)] = scores > score_thresh
        # === Mask out anomalies ===
        x_cleaned = x.copy()
        x_cleaned[anomaly_mask] = np.nan

        return x_cleaned if D > 1 else x_cleaned.flatten()

    else:
        raise ValueError(f"Unsupported outlier method: {method}")


def _remove_outliers_wrapper(args):
    """Wrapper function for parallel outlier removal."""
    i, col, method, threshold = args
    cleaned = _remove_outliers(col, method, threshold)
    return i, cleaned


###########################################################################
# TranAD
###########################################################################




class TranAD(nn.Module):
    def __init__(self, feats, window_size=24, d_model=None, n_heads=None, n_layers=2, dropout=0.1):
        super().__init__()
        d_model = d_model or max(64, feats * 8)
        n_heads = n_heads or max(1, d_model // 64)
        d_model = (d_model // n_heads) * n_heads

        self.encoder = TransformerEncoder(
            input_size=feats,
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers,
            dropout=dropout,
            use_adaptive_ln="layer",
            norm_strategy="pre_norm",
        )

        self.decoder1 = TransformerDecoder(
            input_size=feats,
            output_size=feats,
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers,
            dropout=dropout,
            informer_like=False,
            use_adaptive_ln="layer",
        )

        self.decoder2 = TransformerDecoder(
            input_size=feats,
            output_size=feats,
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers,
            dropout=dropout,
            informer_like=True,
        )
        
        # Pre-allocate context tensors to avoid repeated allocations
        self.register_buffer('_context_cache', torch.empty(1, 1, feats))

    def encode(self, x, context):
        # Fused operation to reduce memory allocations
        return self.encoder(x + context)

    def forward(self, src, tgt=None):
        """
        Forward pass for TranAD using dual-pass reconstruction.
        Optimized version with reduced memory allocations.
        """
        batch_size, seq_len, feats = src.shape
        
        # Reuse buffer for zero context if possible
        if (self._context_cache.shape[0] != batch_size or 
            self._context_cache.shape[1] != seq_len or 
            self._context_cache.shape[2] != feats):
            self._context_cache = torch.zeros_like(src)
        else:
            self._context_cache.zero_()
        
        # First pass: zero context
        enc1 = self.encode(src, self._context_cache)
        out1 = self.decoder1(src, enc1)

        # Second pass: reconstruction error as context (in-place to save memory)
        context2 = out1 - src
        context2.square_()  # In-place operation
        enc2 = self.encode(src, context2)
        out2 = self.decoder2(src, enc2)

        return out1, out2


class TranADDataset(TensorDataset):
    """Memory-efficient dataset that creates sequences on-the-fly"""
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.length = data.shape[0] - seq_len + 1
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len]


def create_sequences_vectorized(data: np.ndarray, seq_len: int) -> torch.Tensor:
    """Optimized sequence creation using vectorized operations"""
    if data.ndim == 1:
        data = data[:, None]
    
    n_samples = data.shape[0] - seq_len + 1
    if n_samples <= 0:
        raise ValueError(f"Data length {data.shape[0]} is too short for sequence length {seq_len}")
    
    # Use unfold for memory-efficient sequence creation
    data_tensor = torch.from_numpy(data.T).float()  # [features, time]
    sequences = data_tensor.unfold(1, seq_len, 1).permute(1, 2, 0)  # [n_samples, seq_len, features]
    return sequences


class TranADDetector:
    def __init__(
        self,
        seq_len: int = 24,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        patience: int = 10,
        device: Optional[str] = None,
        scaler_type: str = "standard",
        use_mixed_precision: bool = True,
        compile_model: bool = False,  # New: PyTorch 2.0 compilation
        memory_efficient: bool = True,  # New: Use memory-efficient sequences
    ):
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.compile_model = compile_model
        self.memory_efficient = memory_efficient

        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.model = None
        self.amp_scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None

    def _create_sequences(self, data: np.ndarray) -> torch.Tensor:
        """Optimized sequence creation"""
        return create_sequences_vectorized(data, self.seq_len)

    def _adaptive_loss(self, x1, x2, target, epoch):
        """Optimized loss computation with pre-computed alpha"""
        alpha = min(0.8, epoch / self.epochs)
        # Compute both losses in one pass when possible
        if alpha == 0:
            return F.mse_loss(x1, target)
        elif alpha == 0.8:
            return F.mse_loss(x2, target)
        else:
            return (1 - alpha) * F.mse_loss(x1, target) + alpha * F.mse_loss(x2, target)

    def _compute_anomaly_scores(self, x2, target):
        """Optimized anomaly score computation"""
        # Compute both MSE and MAE in one pass
        diff = x2 - target
        mse = (diff * diff).mean(dim=(1, 2))
        mae = diff.abs().mean(dim=(1, 2))
        # Fused weighted combination
        return (0.7 * mse + 0.3 * mae).detach().cpu().numpy()

    def fit_predict(self, series: Union[np.ndarray, torch.Tensor], validation_split: float = 0.2) -> np.ndarray:
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()
        if series.ndim == 1:
            series = series[:, None]

        series_scaled = self.scaler.fit_transform(series)
        
        # Choose dataset type based on memory efficiency setting
        if self.memory_efficient:
            # Create sequences on-the-fly to save memory
            sequences_tensor = torch.from_numpy(series_scaled).float()
            n_train = int((len(series_scaled) - self.seq_len + 1) * (1 - validation_split))
            
            train_ds = TranADDataset(sequences_tensor[:n_train + self.seq_len - 1], self.seq_len)
            val_ds = TranADDataset(sequences_tensor[n_train:], self.seq_len) if validation_split > 0 else None
        else:
            # Pre-create all sequences (original approach)
            sequences = self._create_sequences(series_scaled)
            n_train = int(len(sequences) * (1 - validation_split))
            
            train_ds = TensorDataset(sequences[:n_train])
            val_ds = TensorDataset(sequences[n_train:]) if validation_split > 0 else None

        # Optimized DataLoader settings
        num_workers = min(4, torch.get_num_threads())
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2,
            drop_last=False  # Helps with performance consistency
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2
        ) if val_ds else None

        input_size = series.shape[1]
        self.model = TranAD(
            feats=input_size,
            window_size=self.seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        # PyTorch 2.0 compilation for faster execution
        if self.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='max-autotune')

        # Optimized optimizer settings
        opt = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            fused=True if self.device == 'cuda' else False  # Fused optimizer for CUDA
        )
        
        # Cosine annealing with warm restarts for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.epochs // 4, T_mult=2, eta_min=self.lr * 0.01
        )

        best_val_loss = float("inf")
        patience_counter = 0

        # Pre-compute epoch-dependent alpha values for adaptive loss
        alphas = [min(0.8, epoch / self.epochs) for epoch in range(self.epochs)]

        with tqdm(range(self.epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                self.model.train()
                total_loss = 0.0
                n_batches = 0
                
                for batch_data in train_loader:
                    if isinstance(batch_data, tuple):
                        batch = batch_data[0]
                    else:
                        batch = batch_data
                    
                    batch = batch.to(self.device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                        x1, x2 = self.model(batch)
                        
                        # Use pre-computed alpha
                        alpha = alphas[epoch]
                        if alpha == 0:
                            loss = F.mse_loss(x1, batch)
                        elif alpha >= 0.8:
                            loss = F.mse_loss(x2, batch)
                        else:
                            loss = (1 - alpha) * F.mse_loss(x1, batch) + alpha * F.mse_loss(x2, batch)

                    opt.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    
                    if self.use_mixed_precision:
                        self.amp_scaler.scale(loss).backward()
                        self.amp_scaler.step(opt)
                        self.amp_scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    
                    total_loss += loss.item()
                    n_batches += 1

                avg_train_loss = total_loss / n_batches
                scheduler.step()

                val_loss = None
                if val_loader:
                    self.model.eval()
                    val_total = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for batch_data in val_loader:
                            if isinstance(batch_data, tuple):
                                batch = batch_data[0]
                            else:
                                batch = batch_data
                            
                            batch = batch.to(self.device, non_blocking=True)
                            
                            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                                x1, x2 = self.model(batch)
                                alpha = alphas[epoch]
                                if alpha == 0:
                                    batch_loss = F.mse_loss(x1, batch)
                                elif alpha >= 0.8:
                                    batch_loss = F.mse_loss(x2, batch)
                                else:
                                    batch_loss = (1 - alpha) * F.mse_loss(x1, batch) + alpha * F.mse_loss(x2, batch)
                                
                            val_total += batch_loss.item()
                            val_batches += 1
                    
                    val_loss = val_total / val_batches
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Use more efficient checkpoint saving
                        checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss
                        }
                        torch.save(checkpoint, "best_model.pth", _use_new_zipfile_serialization=False)
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                            checkpoint = torch.load("best_model.pth", map_location=self.device)
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                            break

                pbar.set_postfix(train_loss=f"{avg_train_loss:.6f}", val_loss=f"{val_loss:.6f}" if val_loss else "N/A")

        # Create sequences for inference
        if self.memory_efficient:
            sequences = self._create_sequences(series_scaled)
        else:
            sequences = self._create_sequences(series_scaled)
            
        return self._infer(sequences)

    def _infer(self, sequences: torch.Tensor) -> np.ndarray:
        """Optimized inference with larger batch sizes"""
        self.model.eval()
        scores = []
        
        # Use larger batch size for inference
        infer_batch_size = min(self.batch_size * 4, 1024)
        loader = DataLoader(
            TensorDataset(sequences),
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=min(2, torch.get_num_threads()),
            pin_memory=True,
            prefetch_factor=2
        )

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    _, x2 = self.model(batch)
                    batch_scores = self._compute_anomaly_scores(x2, batch)
                    scores.extend(batch_scores)
        
        return np.array(scores)

    def predict(self, series: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Optimized prediction method"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit_predict first.")
        
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()
        if series.ndim == 1:
            series = series[:, None]

        scaled = self.scaler.transform(series)
        seqs = self._create_sequences(scaled)
        return self._infer(seqs)
