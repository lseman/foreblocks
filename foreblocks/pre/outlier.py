# Standard Library
import math
import warnings
from typing import Optional, Tuple, Union

# Scientific Computing and Visualization
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler
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
    """Improved positional encoding with caching and optional learnable components."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Pre-compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Optimized multi-head attention with optional flash attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)

        # Linear transformations and reshape
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Use scaled dot-product attention with optional flash attention
        if hasattr(F, "scaled_dot_product_attention"):
            # Use PyTorch's optimized attention (available in PyTorch 2.0+)
            attn_output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # Fallback to manual implementation
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)

        # Reshape and apply output projection
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.w_o(attn_output)


class TransformerBlock(nn.Module):
    """Optimized transformer block with pre-norm and better initialization."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Use GELU for better performance
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm architecture for better gradient flow
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + attn_output

        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + ff_output

        return x


class TranAD(nn.Module):
    """Improved TranAD with better architecture and training stability."""

    def __init__(
        self,
        feats: int,
        window_size: int = 10,
        d_model: int = None,
        n_heads: int = None,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_feats = feats
        self.n_window = window_size

        # Auto-configure model dimensions
        self.d_model = d_model or max(64, feats * 8)
        self.n_heads = n_heads or max(1, self.d_model // 64)

        # Ensure d_model is divisible by n_heads
        self.d_model = (self.d_model // self.n_heads) * self.n_heads

        # Input projection
        self.input_projection = nn.Linear(feats, self.d_model)
        self.context_projection = nn.Linear(feats, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, window_size)

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, self.n_heads, self.d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

        # Dual decoders
        self.decoder1_layers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, self.n_heads, self.d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

        self.decoder2_layers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, self.n_heads, self.d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

        # Output layers with residual connections
        self.output1 = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, feats),
        )

        self.output2 = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, feats),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Project inputs
        x_proj = self.input_projection(x)
        c_proj = self.context_projection(context)

        # Combine input and context
        combined = x_proj + c_proj
        combined = self.pos_encoder(combined)

        # Apply encoder layers
        for layer in self.encoder_layers:
            combined = layer(combined)

        return combined

    def decode(
        self, memory: torch.Tensor, decoder_layers: nn.ModuleList
    ) -> torch.Tensor:
        x = memory
        for layer in decoder_layers:
            x = layer(x)
        return x

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_feats = src.shape

        # First pass with zero context
        context1 = torch.zeros_like(src)
        memory1 = self.encode(src, context1)
        decoded1 = self.decode(memory1, self.decoder1_layers)
        output1 = self.output1(decoded1)

        # Second pass with reconstruction error as context
        context2 = (output1 - src) ** 2
        memory2 = self.encode(src, context2)
        decoded2 = self.decode(memory2, self.decoder2_layers)
        output2 = self.output2(decoded2)

        return output1, output2


class TranADDetector:
    """Improved TranAD detector with better training and inference."""

    def __init__(
        self,
        seq_len: int = 24,
        d_model: int = None,
        n_heads: int = None,
        n_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        patience: int = 10,
        device: str = None,
        scaler_type: str = "standard",
        use_mixed_precision: bool = True,
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

        # Choose scaler
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.model = None

        # Initialize mixed precision scaler
        if self.use_mixed_precision:
            self.amp_scaler = torch.cuda.amp.GradScaler()

    def _create_sequences(self, data: np.ndarray) -> torch.Tensor:
        """Create sliding window sequences more efficiently."""
        n_samples = len(data) - self.seq_len + 1
        sequences = np.zeros((n_samples, self.seq_len, data.shape[1]))

        for i in range(n_samples):
            sequences[i] = data[i : i + self.seq_len]

        return torch.tensor(sequences, dtype=torch.float32)

    def _adaptive_loss(
        self, x1: torch.Tensor, x2: torch.Tensor, target: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        """Adaptive loss that balances both decoders."""
        mse1 = F.mse_loss(x1, target)
        mse2 = F.mse_loss(x2, target)

        # Gradually shift focus to second decoder
        alpha = min(0.8, epoch / self.epochs)
        return (1 - alpha) * mse1 + alpha * mse2

    def _compute_anomaly_scores(
        self, x2: torch.Tensor, target: torch.Tensor
    ) -> np.ndarray:
        """Compute anomaly scores using multiple metrics."""
        # MSE-based score
        mse_scores = F.mse_loss(x2, target, reduction="none").mean(dim=(1, 2))

        # MAE-based score
        mae_scores = F.l1_loss(x2, target, reduction="none").mean(dim=(1, 2))

        # Combined score
        combined_scores = 0.7 * mse_scores + 0.3 * mae_scores

        return combined_scores.detach().cpu().numpy()

    def fit_predict(
        self, series: Union[np.ndarray, torch.Tensor], validation_split: float = 0.2
    ) -> np.ndarray:
        """Fit model and predict anomaly scores with validation."""

        # Convert to numpy if needed
        if isinstance(series, torch.Tensor):
            series = series.detach().cpu().numpy()

        series = np.asarray(series)
        if series.ndim == 1:
            series = series[:, None]

        # Scale data
        series_scaled = self.scaler.fit_transform(series)

        # Create sequences
        sequences = self._create_sequences(series_scaled)

        # Train/validation split
        n_train = int(len(sequences) * (1 - validation_split))
        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:] if validation_split > 0 else None

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_sequences)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )

        if val_sequences is not None:
            val_dataset = torch.utils.data.TensorDataset(val_sequences)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )

        # Initialize model
        input_size = series.shape[1]
        self.model = TranAD(
            feats=input_size,
            window_size=self.seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.patience // 2, verbose=True
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(self.epochs), desc="Training"):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for (batch,) in train_loader:
                batch = batch.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        x1, x2 = self.model(batch, batch)
                        loss = self._adaptive_loss(x1, x2, batch, epoch)

                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.step(optimizer)
                    self.amp_scaler.update()
                else:
                    x1, x2 = self.model(batch, batch)
                    loss = self._adaptive_loss(x1, x2, batch, epoch)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            # Validation phase
            if val_sequences is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for (batch,) in val_loader:
                        batch = batch.to(self.device, non_blocking=True)

                        if self.use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                x1, x2 = self.model(batch, batch)
                                loss = self._adaptive_loss(x1, x2, batch, epoch)
                        else:
                            x1, x2 = self.model(batch, batch)
                            loss = self._adaptive_loss(x1, x2, batch, epoch)

                        val_loss += loss.item()

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), "best_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Load best model
                    self.model.load_state_dict(torch.load("best_model.pth"))
                    break

            if epoch % 10 == 0:
                avg_train_loss = train_loss / len(train_loader)
                if val_sequences is not None:
                    print(
                        f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}"
                    )

        # Inference phase
        self.model.eval()
        scores = []

        with torch.no_grad():
            # Use sliding window for inference
            inference_loader = torch.utils.data.DataLoader(
                sequences, batch_size=self.batch_size * 2, shuffle=False
            )

            for (batch,) in inference_loader:
                batch = batch.to(self.device)

                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        _, x2 = self.model(batch, batch)
                        batch_scores = self._compute_anomaly_scores(x2, batch)
                else:
                    _, x2 = self.model(batch, batch)
                    batch_scores = self._compute_anomaly_scores(x2, batch)

                scores.extend(batch_scores)

        return np.array(scores)

    def predict(self, series: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict anomaly scores for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit_predict first.")

        if isinstance(series, torch.Tensor):
            series = series.detach().cpu().numpy()

        series = np.asarray(series)
        if series.ndim == 1:
            series = series[:, None]

        # Scale using fitted scaler
        series_scaled = self.scaler.transform(series)
        sequences = self._create_sequences(series_scaled)

        self.model.eval()
        scores = []

        with torch.no_grad():
            loader = torch.utils.data.DataLoader(
                sequences, batch_size=self.batch_size * 2, shuffle=False
            )

            for (batch,) in loader:
                batch = batch.to(self.device)

                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        _, x2 = self.model(batch, batch)
                        batch_scores = self._compute_anomaly_scores(x2, batch)
                else:
                    _, x2 = self.model(batch, batch)
                    batch_scores = self._compute_anomaly_scores(x2, batch)

                scores.extend(batch_scores)

        return np.array(scores)
