# Standard Library
import warnings

# Scientific Computing and Visualization
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from numba import prange
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
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

    # Correct modified Z-score (Iglewicz and Hoaglin)
    mod_z = 0.6745 * np.abs(x - med) / mad

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


def _hbos_outlier_removal(x: np.ndarray, threshold: float) -> np.ndarray:
    """Histogram-based Outlier Scoring for univariate or multivariate series."""
    valid = ~np.isnan(x)
    if np.sum(valid) < 5:
        return x

    if x.ndim == 1:
        values = x[valid]
    else:
        values = x[valid].reshape(-1)

    n = values.shape[0]
    bins = min(max(int(np.sqrt(n)), 10), 50)

    if x.ndim == 1:
        col = x[valid]
        hist, edges = np.histogram(col, bins=bins)
        hist = hist.astype(np.float64) + 1e-8
        dens = hist / np.sum(hist)
        idx = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, len(hist) - 1)
        scores = np.zeros_like(x, dtype=np.float64)
        scores[valid] = -np.log(dens[idx])
    else:
        nrows, ncols = x.shape
        scores = np.zeros(nrows, dtype=np.float64)
        for j in range(ncols):
            col = x[:, j]
            valid_col = ~np.isnan(col)
            if np.sum(valid_col) < 5:
                continue
            col_values = col[valid_col]
            hist, edges = np.histogram(col_values, bins=bins)
            hist = hist.astype(np.float64) + 1e-8
            dens = hist / np.sum(hist)
            idx = np.clip(
                np.searchsorted(edges, col_values, side="right") - 1, 0, len(hist) - 1
            )
            temp_scores = -np.log(dens[idx])
            scores[valid_col] += temp_scores

    threshold_value = np.percentile(scores[valid], 100.0 * (1.0 - float(threshold)))
    result = x.copy()
    if x.ndim == 1:
        result[valid & (scores > threshold_value)] = np.nan
    else:
        result[scores > threshold_value] = np.nan
    return result


def _remove_outliers(
    data_col: np.ndarray, method: str, threshold: float, **kwargs
) -> np.ndarray:
    """
    Remove outliers from a univariate or multivariate time series using the specified method.
    Replaces detected outliers with np.nan.

    Parameters:
        data_col: np.ndarray of shape (T,) or (T, D)
        method: One of ["zscore", "iqr", "mad", "hbos", "quantile", "isolation_forest", "lof", "ecod", "tranad"]
        threshold: method-dependent threshold (e.g. contamination fraction for HBOS and ECOD, 0.95 for percentile methods)
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
        elif method == "hbos":
            return _hbos_outlier_removal(x, threshold)
        elif method == "quantile":
            q1, q3 = np.nanpercentile(x, [threshold * 100, 100 - threshold * 100])
            return fast_quantile_outlier_removal(x, q1, q3)

    # === Multivariate-aware methods ===
    if method == "hbos":
        return _hbos_outlier_removal(x, threshold)
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
        seq_len = kwargs.get("seq_len", 24)
        epochs = kwargs.get("epochs", 10)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        adaptive = kwargs.get("adaptive", True)
        min_z = kwargs.get(
            "z_threshold", float(threshold) if float(threshold) > 1.0 else 3.0
        )

        # Ensure 2D format
        if data_col.ndim == 1:
            data_col = data_col.reshape(-1, 1)
        x = data_col.astype(np.float64)
        T, D = x.shape

        if T < seq_len + 5:
            return x if D > 1 else x.flatten()

        detector = TranADDetector(
            seq_len=seq_len,
            epochs=epochs,
            device=device,
            scaler_type=kwargs.get("scaler_type", "minmax"),
        )
        scores = detector.fit_predict(x)  # shape (T - seq_len + 1, D)
        if scores.ndim == 1:
            scores = scores[:, None]

        anomaly_mask = np.full(T, False)
        if adaptive:
            score_mean = np.mean(scores, axis=0, keepdims=True)
            score_std = np.std(scores, axis=0, keepdims=True) + 1e-8
            score_z = (scores - score_mean) / score_std
            anomaly_mask[seq_len - 1 : seq_len - 1 + len(score_z)] = np.any(
                score_z > float(min_z), axis=1
            )
        else:
            if threshold > 1.0:
                cutoff = np.mean(scores, axis=0, keepdims=True) + float(threshold) * (
                    np.std(scores, axis=0, keepdims=True) + 1e-8
                )
            else:
                percentile = float(np.clip(threshold * 100.0, 0.0, 100.0))
                cutoff = np.nanpercentile(scores, percentile, axis=0, keepdims=True)
            anomaly_mask[seq_len - 1 : seq_len - 1 + len(scores)] = np.any(
                scores > cutoff, axis=1
            )

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
    def __init__(
        self,
        feats,
        window_size=24,
        d_model=None,
        n_heads=None,
        n_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.n_feats = int(feats)
        self.n_window = int(window_size)
        self.d_model = int(d_model or max(2 * self.n_feats, 32))

        if n_heads is None:
            n_heads = min(max(1, self.n_feats), self.d_model)
            while self.d_model % n_heads != 0 and n_heads > 1:
                n_heads -= 1
        self.n_heads = int(max(1, n_heads))

        self.input_projection = nn.Linear(2 * self.n_feats, self.d_model)
        self.pos_encoder = _TranADPositionalEncoding(
            self.d_model, dropout=dropout, max_len=max(512, self.n_window + 2)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=max(16, self.d_model),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=max(16, self.d_model),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        decoder_layer2 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=max(16, self.d_model),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_decoder1 = nn.TransformerDecoder(
            decoder_layer1, num_layers=n_layers
        )
        self.transformer_decoder2 = nn.TransformerDecoder(
            decoder_layer2, num_layers=n_layers
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.n_feats),
            nn.Sigmoid(),
        )

    def encode(self, src, context, tgt):
        enc_in = torch.cat((src, context), dim=-1)
        enc_in = self.input_projection(enc_in)
        enc_in = self.pos_encoder(enc_in)
        memory = self.transformer_encoder(enc_in)

        dec_in = torch.cat((tgt, tgt), dim=-1)
        dec_in = self.input_projection(dec_in)
        dec_in = self.pos_encoder(dec_in)
        return dec_in, memory

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src[:, -1:, :]

        zero_context = torch.zeros_like(src)
        dec1_in, memory1 = self.encode(src, zero_context, tgt)
        out1 = self.output_projection(self.transformer_decoder1(dec1_in, memory1))

        focus = (out1 - src).pow(2)
        dec2_in, memory2 = self.encode(src, focus, tgt)
        out2 = self.output_projection(self.transformer_decoder2(dec2_in, memory2))
        return out1, out2


class _TranADPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / max(1, d_model))
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TranADDataset(TensorDataset):
    """Memory-efficient dataset that creates sequences on-the-fly"""

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.length = data.shape[0] - seq_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len]


def create_sequences_vectorized(data: np.ndarray, seq_len: int) -> torch.Tensor:
    """Optimized sequence creation using vectorized operations"""
    if data.ndim == 1:
        data = data[:, None]

    n_samples = data.shape[0] - seq_len + 1
    if n_samples <= 0:
        raise ValueError(
            f"Data length {data.shape[0]} is too short for sequence length {seq_len}"
        )

    # Use unfold for memory-efficient sequence creation
    data_tensor = torch.from_numpy(data.T).float()  # [features, time]
    sequences = data_tensor.unfold(1, seq_len, 1).permute(
        1, 2, 0
    )  # [n_samples, seq_len, features]
    return sequences


class TranADDetector:
    def __init__(
        self,
        seq_len: int = 24,
        d_model: int | None = None,
        n_heads: int | None = None,
        n_layers: int = 1,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        patience: int = 10,
        device: str | None = None,
        scaler_type: str = "minmax",
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

        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.model = None
        self.amp_scaler = (
            torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        )

    def _create_sequences(self, data: np.ndarray) -> torch.Tensor:
        """Optimized sequence creation"""
        return create_sequences_vectorized(data, self.seq_len)

    @staticmethod
    def _repo_loss(x1, x2, target, epoch_num: int):
        weight_first = 1.0 / max(1, int(epoch_num))
        weight_second = 1.0 - weight_first
        return weight_first * F.mse_loss(x1, target) + weight_second * F.mse_loss(
            x2, target
        )

    @staticmethod
    def _compute_anomaly_scores(x1, x2, target):
        diff1 = (x1 - target).pow(2).squeeze(1)
        diff2 = (x2 - target).pow(2).squeeze(1)
        return (0.5 * diff1 + 0.5 * diff2).detach().cpu().numpy()

    def fit_predict(
        self, series: np.ndarray | torch.Tensor, validation_split: float = 0.2
    ) -> np.ndarray:
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()
        if series.ndim == 1:
            series = series[:, None]

        series_scaled = self.scaler.fit_transform(series)

        # Choose dataset type based on memory efficiency setting
        if self.memory_efficient:
            # Create sequences on-the-fly to save memory
            sequences_tensor = torch.from_numpy(series_scaled).float()
            n_train = int(
                (len(series_scaled) - self.seq_len + 1) * (1 - validation_split)
            )

            train_ds = TranADDataset(
                sequences_tensor[: n_train + self.seq_len - 1], self.seq_len
            )
            val_ds = (
                TranADDataset(sequences_tensor[n_train:], self.seq_len)
                if validation_split > 0
                else None
            )
        else:
            # Pre-create all sequences (original approach)
            sequences = self._create_sequences(series_scaled)
            n_train = int(len(sequences) * (1 - validation_split))

            train_ds = TensorDataset(sequences[:n_train])
            val_ds = (
                TensorDataset(sequences[n_train:]) if validation_split > 0 else None
            )

        num_workers = min(4, torch.get_num_threads())
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": bool(num_workers > 0),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )

        val_loader = (
            DataLoader(
                val_ds,
                shuffle=False,
                **loader_kwargs,
            )
            if val_ds
            else None
        )

        input_size = series.shape[1]
        self.model = TranAD(
            feats=input_size,
            window_size=self.seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        if self.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="max-autotune")

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=True if self.device == "cuda" else False,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        with tqdm(range(self.epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                self.model.train()
                total_loss = 0.0
                n_batches = 0
                epoch_num = epoch + 1

                for batch_data in train_loader:
                    if isinstance(batch_data, tuple):
                        batch = batch_data[0]
                    else:
                        batch = batch_data

                    batch = batch.to(self.device, non_blocking=True)
                    target = batch[:, -1:, :]

                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                        x1, x2 = self.model(batch, target)
                        loss = self._repo_loss(x1, x2, target, epoch_num)

                    opt.zero_grad(set_to_none=True)

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
                            target = batch[:, -1:, :]

                            with torch.cuda.amp.autocast(
                                enabled=self.use_mixed_precision
                            ):
                                x1, x2 = self.model(batch, target)
                                batch_loss = self._repo_loss(x1, x2, target, epoch_num)

                            val_total += batch_loss.item()
                            val_batches += 1

                    val_loss = val_total / val_batches

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f"\nEarly stopping at epoch {epoch + 1}")
                            if best_state is not None:
                                self.model.load_state_dict(best_state)
                            break
                else:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }

                scheduler.step()
                pbar.set_postfix(
                    train_loss=f"{avg_train_loss:.6f}",
                    val_loss=f"{val_loss:.6f}" if val_loss else "N/A",
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        sequences = self._create_sequences(series_scaled)
        return self._infer(sequences)

    def _infer(self, sequences: torch.Tensor) -> np.ndarray:
        self.model.eval()
        scores = []

        infer_batch_size = min(self.batch_size * 4, 1024)
        num_workers = min(2, torch.get_num_threads())
        loader_kwargs = {
            "batch_size": infer_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(
            TensorDataset(sequences),
            **loader_kwargs,
        )

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device, non_blocking=True)
                target = batch[:, -1:, :]
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    x1, x2 = self.model(batch, target)
                    batch_scores = self._compute_anomaly_scores(x1, x2, target)
                    scores.append(batch_scores)

        return np.concatenate(scores, axis=0) if scores else np.empty((0, 0))

    def predict(self, series: np.ndarray | torch.Tensor) -> np.ndarray:
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
