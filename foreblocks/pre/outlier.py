# Standard Library
import math
import warnings
from typing import Union
import torch

# Scientific Computing and Visualization
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

# Optional imports
try:
    from pykalman import KalmanFilter
except ImportError:
    KalmanFilter = None

try:
    from PyEMD import EMD
except ImportError:
    EMD = None

from numba import njit, prange


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
            anomaly_mask[seq_len:] = score_z > min_z
        else:
            if threshold > 1.0:
                percentile = min(max(threshold, 0), 100)
            else:
                percentile = threshold * 100
            score_thresh = np.nanpercentile(scores, percentile)
            anomaly_mask = np.full(T, False)
            anomaly_mask[seq_len:] = scores > score_thresh

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TranAD(nn.Module):
    def __init__(self, feats, window_size, dropout=0.1):
        super().__init__()
        self.n_feats = feats
        self.n_window = window_size
        self.d_model = 2 * feats

        self.pos_encoder = PositionalEncoding(self.d_model, dropout, self.n_window)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=feats,
            dim_feedforward=16,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=feats,
            dim_feedforward=16,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer1, num_layers=1)

        decoder_layer2 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=feats,
            dim_feedforward=16,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2, num_layers=1)

        self.output_layer = nn.Sequential(nn.Linear(self.d_model, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # (B, T, 2F)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)

        tgt = tgt.repeat(1, 1, 2) * math.sqrt(self.n_feats)
        tgt = self.pos_encoder(tgt)
        return tgt, memory

    def forward(self, src, tgt):
        c = torch.zeros_like(src)
        tgt1, memory1 = self.encode(src, c, tgt)
        x1 = self.output_layer(self.transformer_decoder1(tgt1, memory1))

        c = (x1 - src) ** 2
        tgt2, memory2 = self.encode(src, c, tgt)
        x2 = self.output_layer(self.transformer_decoder2(tgt2, memory2))

        return x1, x2


class TranADDetector:
    def __init__(
        self,
        seq_len=24,
        epochs=20,
        batch_size=512,
        learning_rate=1e-3,
        dropout=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.device = device
        self.scaler = StandardScaler()
        self.model = None
        self.dropout = dropout

    def _create_sequences(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            np.stack(
                [data[i : i + self.seq_len] for i in range(len(data) - self.seq_len)]
            ),
            dtype=torch.float32,
        )

    def _loss(self, x1, x2, target):
        return F.mse_loss(x1, target) + F.mse_loss(x2, target)

    def _score(self, x2, target):
        return (
            F.mse_loss(x2, target, reduction="none")
            .mean(dim=(1, 2))
            .detach()
            .cpu()
            .numpy()
        )

    def fit_predict(self, series: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(series, torch.Tensor):
            series = series.detach().cpu().numpy()

        series = np.asarray(series)
        if series.ndim == 1:
            series = series[:, None]

        series_scaled = self.scaler.fit_transform(series)
        sequences = self._create_sequences(series_scaled)
        dataset = torch.utils.data.TensorDataset(sequences)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        input_size = series.shape[1]
        self.model = TranAD(
            feats=input_size, window_size=self.seq_len, dropout=self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                x1, x2 = self.model(batch, batch)
                loss = self._loss(x1, x2, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {total_loss:.4f}")

        self.model.eval()
        with torch.no_grad():
            scores = []
            for i in range(0, len(series_scaled) - self.seq_len):
                window = torch.tensor(
                    series_scaled[i : i + self.seq_len][None], dtype=torch.float32
                ).to(self.device)
                _, x2 = self.model(window, window)
                score = self._score(x2, window)
                scores.append(score[0])

        return np.array(scores)
