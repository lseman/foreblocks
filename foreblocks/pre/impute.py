import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


def rolling_window_impute(
    series: np.ndarray, model_class, window_size=48, stride=24, model_kwargs=None
):
    """
    Rolling horizon imputation using overlapping windows and a pluggable model class.

    Args:
        series: np.ndarray of shape (T, D) with NaNs.
        model_class: Imputer class with .fit() and .impute() methods (e.g. SAITSImputer).
        window_size: Length of each window (rolling horizon).
        stride: Step between consecutive windows.
        model_kwargs: Optional dict of kwargs passed to model_class constructor.

    Returns:
        Imputed np.ndarray of shape (T, D).
    """
    if series.ndim == 1:
        series = series[:, None]
    T, D = series.shape

    recon = np.zeros((T, D), dtype=np.float32)
    counts = np.zeros((T, D), dtype=np.float32)

    model_kwargs = model_kwargs or {}

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        window = series[start:end]

        model = model_class(seq_len=window_size, **model_kwargs)
        try:
            model.fit(window)
            imputed_window = model.impute(window)
        except Exception as e:
            print(f"[WARN] Imputation failed at window {start}:{end} - {e}")
            imputed_window = np.nan_to_num(window)

        recon[start:end] += imputed_window
        counts[start:end] += 1

    # Handle last window if not covered
    if end < T:
        start = T - window_size
        window = series[start:T]
        model = model_class(seq_len=window_size, **model_kwargs)
        try:
            model.fit(window)
            imputed_window = model.impute(window)
        except Exception as e:
            print(f"[WARN] Final window imputation failed - {e}")
            imputed_window = np.nan_to_num(window)
        recon[start:T] += imputed_window[-(T - start) :]
        counts[start:T] += 1

    # Normalize overlapping reconstructions
    counts[counts == 0] = 1e-9
    result = recon / counts

    # Fill original values
    return np.where(np.isnan(series), result, series)


# === Optimized Loss Functions ===
def masked_mae_cal(inputs, target, mask):
    """Vectorized masked MAE calculation"""
    diff = torch.abs(inputs - target) * mask
    return diff.sum() / (mask.sum() + 1e-9)


def masked_mse_cal(inputs, target, mask):
    """Vectorized masked MSE calculation"""
    diff = torch.square(inputs - target) * mask
    return diff.sum() / (mask.sum() + 1e-9)


# === Optimized Positional Encoding with Caching ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, n_position=1000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Pre-compute positional encodings
        pe = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        pos_emb = self.pe[:, :seq_len]
        x = x + pos_emb
        return self.dropout(x) if self.dropout else x


# === Efficient Encoder Layer with Flash Attention Support ===
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        # Use torch.nn.functional.scaled_dot_product_attention for better performance
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )

        # Pre-LayerNorm for better gradient flow
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Optimized feedforward with GELU activation
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),  # GELU often performs better than ReLU
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm architecture
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.self_attn(normed_x, normed_x, normed_x)
        x = x + attn_output

        normed_x = self.norm2(x)
        ff_output = self.ff(normed_x)
        x = x + ff_output

        return x, attn_weights


# === Optimized SAITS Model ===
class SAITS(nn.Module):
    def __init__(
        self,
        input_size,
        seq_len,
        d_model=64,
        d_inner=128,
        n_head=4,
        n_groups=2,
        n_group_inner_layers=1,
        dropout=0.1,
        param_sharing_strategy="between_group",
        input_with_mask=True,
        MIT=False,
        use_layer_scale=True,
        layer_scale_init=1e-4,
        mit_mode=False,
        holdout_rate=0.15,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_with_mask = input_with_mask
        self.param_sharing_strategy = param_sharing_strategy
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.MIT = MIT
        self.mit_mode = mit_mode
        self.holdout_rate = holdout_rate
        
        self.use_layer_scale = use_layer_scale
        

        actual_input_size = input_size * 2 if input_with_mask else input_size

        # Shared embedding layers with better initialization
        self.embedding_1 = self._make_embedding_layer(actual_input_size, d_model)
        self.embedding_2 = self._make_embedding_layer(actual_input_size, d_model)

        # Output projection layers
        self.reduce_dim_z = nn.Linear(d_model, input_size)
        self.reduce_dim_beta = nn.Linear(d_model, input_size)
        self.reduce_dim_gamma = nn.Linear(input_size, input_size)
        self.weight_combine = nn.Linear(input_size + seq_len, input_size)

        # Positional encoding with dropout
        self.pos_enc = PositionalEncoding(d_model, n_position=seq_len, dropout=dropout)

        # Layer scale for better training stability
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
            self.layer_scale_2 = nn.Parameter(torch.ones(d_model) * layer_scale_init)

        def build_layers():
            return nn.ModuleList(
                [
                    EncoderLayer(d_model, d_inner, n_head, dropout)
                    for _ in range(n_group_inner_layers)
                ]
            )

        if param_sharing_strategy == "between_group":
            self.encoder_block1 = build_layers()
            self.encoder_block2 = build_layers()
        else:
            self.encoder_block1 = nn.ModuleList(
                [build_layers()[0] for _ in range(n_groups)]
            )
            self.encoder_block2 = nn.ModuleList(
                [build_layers()[0] for _ in range(n_groups)]
            )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _make_embedding_layer(self, input_size, d_model):
        """Create embedding layer with proper initialization"""
        layer = nn.Linear(input_size, d_model)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def _init_weights(self):
        """Initialize weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _run_block(self, x, layer_stack):
        attn_weights = None
        if self.param_sharing_strategy == "between_group":
            for layer in layer_stack:
                x, attn_weights = layer(x)
        else:
            for layer in layer_stack:
                for _ in range(self.n_group_inner_layers):
                    x, attn_weights = layer(x)
        return x, attn_weights

    def impute(self, X, masks):
        batch_size, seq_len, input_size = X.shape

        # First imputation stage
        input_first = (
            torch.cat([X * masks, masks], dim=2) if self.input_with_mask else X
        )
        input_first = self.embedding_1(input_first)
        enc_output = self.pos_enc(input_first)
        enc_output, _ = self._run_block(enc_output, self.encoder_block1)
        X_tilde_1 = self.reduce_dim_z(enc_output)

        # Create refined input
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # Second imputation stage
        input_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        )
        input_second = self.embedding_2(input_second)
        enc_output = self.pos_enc(input_second)
        enc_output, attn_weights = self._run_block(enc_output, self.encoder_block2)

        X_tilde_2 = self.reduce_dim_gamma(F.gelu(self.reduce_dim_beta(enc_output)))

        # Process attention weights more efficiently
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1).transpose(1, 2)

        # Combine predictions
        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )

        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = masks * X + (1 - masks) * X_tilde_3

        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage="train"):
        X, masks = inputs["X"], inputs["missing_mask"]
        X_holdout = inputs.get("X_holdout")
        indicating_mask = inputs.get("indicating_mask")

        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(X, masks)

        recon_loss = (
            masked_mae_cal(X_tilde_1, X, masks)
            + masked_mae_cal(X_tilde_2, X, masks)
            + masked_mae_cal(X_tilde_3, X, masks)
        ) / 3

        final_mae = masked_mae_cal(X_tilde_3, X, masks)

        if (self.MIT or stage == "val") and stage != "test":
            imput_mae = masked_mae_cal(X_tilde_3, X_holdout, indicating_mask)
        else:
            imput_mae = torch.tensor(0.0, device=X.device)

        loss = recon_loss + imput_mae

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": loss,
            "imputation_loss": imput_mae,
            "reconstruction_MAE": final_mae,
            "imputation_MAE": imput_mae,
        }

class SaitsDataset(Dataset):
    def __init__(self, X, mask, mit_mode=False, holdout_rate=0.15):
        self.X = X.contiguous()
        self.mask = mask.contiguous()

        self.X_holdout = self.X.clone()
        self.indicating_mask = 1.0 - self.mask

        if mit_mode:
            # Simulate a holdout mask over existing observed values
            B, T, D = X.shape
            random_mask = torch.rand(B, T, D, device=X.device) < holdout_rate
            # Only hide parts of the observed data (i.e., where mask == 1)
            holdout_mask = random_mask & (self.mask == 1.0)
            self.X_holdout[holdout_mask] = 0.0  # Or NaN depending on preference
            self.indicating_mask[holdout_mask] = 1.0
            self.mask[holdout_mask] = 0.0  # Hide them from training
    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "missing_mask": self.mask[idx],
            "X_holdout": self.X_holdout[idx],
            "indicating_mask": self.indicating_mask[idx],
        }

    def __len__(self):
        return len(self.X)


# === Optimized Trainer with Mixed Precision ===
class SAITSTrainer:
    def __init__(self, model, optimizer, device="cuda", use_amp=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0

        for batch in dataloader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            self.optimizer.zero_grad(
                set_to_none=True
            )  # More efficient than zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(batch, stage="train")
                    loss = output["reconstruction_loss"]
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(batch, stage="train")
                loss = output["reconstruction_loss"]
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        losses = []

        for batch in dataloader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            if self.use_amp:
                with autocast():
                    output = self.model(batch, stage="val")
            else:
                output = self.model(batch, stage="val")

            losses.append(output["imputation_MAE"].item())

        return np.mean(losses)

    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()
        all_imputations = []

        for batch in dataloader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            if self.use_amp:
                with autocast():
                    output = self.model(batch, stage="test")
            else:
                output = self.model(batch, stage="test")

            all_imputations.append(output["imputed_data"].cpu())

        return torch.cat(all_imputations, dim=0)


# === Optimized Imputer Wrapper ===
class SAITSImputer:
    def __init__(
        self,
        seq_len=24,
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        d_model=64,
        d_inner=128,
        n_head=4,
        n_groups=2,
        n_group_inner_layers=1,
        dropout=0.1,
        param_sharing_strategy="between_group",
        input_with_mask=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=False,
        num_workers=4,
        pin_memory=True,
        warmup_epochs=5,
        lr_scheduler="cosine",
        mit_mode=False,
        holdout_rate=0.15,
    ):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.dropout = dropout
        self.param_sharing_strategy = param_sharing_strategy
        self.input_with_mask = input_with_mask
        self.device = device
        self.use_amp = use_amp
        self.num_workers = num_workers
        self.pin_memory = pin_memory and device == "cuda"
        self.warmup_epochs = warmup_epochs
        self.lr_scheduler = lr_scheduler
        self.mit_mode = mit_mode
        self.holdout_rate = holdout_rate

        self.model = None
        self.trainer = None
        self.input_size = None

    def _create_windows_vectorized(self, data):
        """Vectorized window creation for better performance"""
        T, D = data.shape
        if T < self.seq_len:
            raise ValueError(
                f"Data length {T} is less than sequence length {self.seq_len}"
            )

        # Use stride tricks for efficient windowing
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(data, window_shape=self.seq_len, axis=0)
        return windows.transpose(0, 2, 1)  # (num_windows, seq_len, features)

    def _create_masks(self, data):
        return (~np.isnan(data)).astype(np.float32)

    def _get_scheduler(self, optimizer, num_training_steps):
        """Create learning rate scheduler"""
        if self.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        elif self.lr_scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=self.warmup_epochs
            )
        return None

    def fit(self, series: np.ndarray):
        if series.ndim == 1:
            series = series[:, None]

        self.input_size = series.shape[1]

        # Optimized window creation
        data = self._create_windows_vectorized(series)
        masks = self._create_masks(data)

        # Convert to tensors with optimal dtypes
        X = torch.tensor(np.nan_to_num(data), dtype=torch.float32)
        mask = torch.tensor(masks, dtype=torch.float32)

        # Optimized dataset
        dataset = SaitsDataset(X, mask, mit_mode=self.mit_mode, holdout_rate=self.holdout_rate)

        # Optimized dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # For consistent batch sizes
            persistent_workers=self.num_workers > 0,
        )

        # Initialize model
        self.model = SAITS(
            input_size=self.input_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            d_inner=self.d_inner,
            n_head=self.n_head,
            n_groups=self.n_groups,
            n_group_inner_layers=self.n_group_inner_layers,
            dropout=self.dropout,
            param_sharing_strategy=self.param_sharing_strategy,
            input_with_mask=self.input_with_mask,
        )

        # Optimized optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.95),  # Better betas for transformers
        )

        # Learning rate scheduler
        scheduler = self._get_scheduler(optimizer, len(dataloader) * self.epochs)

        self.trainer = SAITSTrainer(
            self.model, optimizer, device=self.device, use_amp=self.use_amp
        )

        # Training loop with progress bar
        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        with tqdm(range(self.epochs), desc="Training SAITS") as pbar:
            for epoch in pbar:
                loss = self.trainer.train_epoch(dataloader)

                if scheduler:
                    scheduler.step()

                # Early stopping
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                pbar.set_postfix(
                    {
                        "Loss": f"{loss:.4f}",
                        "Best": f"{best_loss:.4f}",
                        "LR": f'{optimizer.param_groups[0]["lr"]:.2e}',
                    }
                )

                # if patience_counter >= patience:
                #     print(f"Early stopping at epoch {epoch + 1}")
                #     break

    def impute(self, series: np.ndarray) -> np.ndarray:
        if series.ndim == 1:
            series = series[:, None]

        data = self._create_windows_vectorized(series)
        masks = self._create_masks(data)

        X = torch.tensor(np.nan_to_num(data), dtype=torch.float32)
        mask = torch.tensor(masks, dtype=torch.float32)

        dataset = SaitsDataset(X, mask)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,  # Use larger batch for inference
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        imputed_tensor = self.trainer.predict(dataloader).numpy()

        # Optimized reconstruction
        T, D = series.shape
        recon = np.zeros((T, D), dtype=np.float32)
        counts = np.zeros((T, D), dtype=np.float32)

        for i in range(len(imputed_tensor)):
            end_idx = i + self.seq_len
            recon[i:end_idx] += imputed_tensor[i]
            counts[i:end_idx] += 1

        counts = np.maximum(counts, 1e-9)  # Avoid division by zero
        imputed = recon / counts

        # Return imputed values only where original data was missing
        return np.where(np.isnan(series), imputed, series)

