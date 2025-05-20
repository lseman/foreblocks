import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
# === Masked Loss Utilities ===
def masked_mae_cal(inputs, target, mask):
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

def masked_mse_cal(inputs, target, mask):
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

import numpy as np

def rolling_window_impute(series: np.ndarray, model_class, window_size=48, stride=24, model_kwargs=None):
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
        recon[start:T] += imputed_window[-(T - start):]
        counts[start:T] += 1

    # Normalize overlapping reconstructions
    counts[counts == 0] = 1e-9
    result = recon / counts

    # Fill original values
    return np.where(np.isnan(series), result, series)

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, n_position=1000):
        super().__init__()
        pe = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# === Encoder Layer ===
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
        )

    def forward(self, x):
        attn_output, attn_weights = self.self_attn(x, x, x)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.layer_norm2(x + ff_output)
        return x, attn_weights

# === SAITS Main Model ===
class SAITS(nn.Module):
    def __init__(self, input_size, seq_len, d_model=64, d_inner=128, n_head=4,
                 n_groups=2, n_group_inner_layers=1, dropout=0.1,
                 param_sharing_strategy='between_group', input_with_mask=True, MIT=False):
        super().__init__()
        self.seq_len = seq_len
        self.input_with_mask = input_with_mask
        self.param_sharing_strategy = param_sharing_strategy
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.MIT = MIT

        actual_input_size = input_size * 2 if input_with_mask else input_size
        self.embedding_1 = nn.Linear(actual_input_size, d_model)
        self.embedding_2 = nn.Linear(actual_input_size, d_model)
        self.reduce_dim_z = nn.Linear(d_model, input_size)
        self.reduce_dim_beta = nn.Linear(d_model, input_size)
        self.reduce_dim_gamma = nn.Linear(input_size, input_size)
        self.weight_combine = nn.Linear(input_size + seq_len, input_size)
        self.pos_enc = PositionalEncoding(d_model, n_position=seq_len)

        def build_layers():
            return nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, dropout)
                                   for _ in range(n_group_inner_layers)])

        if param_sharing_strategy == 'between_group':
            self.encoder_block1 = build_layers()
            self.encoder_block2 = build_layers()
        else:
            self.encoder_block1 = nn.ModuleList([build_layers()[0] for _ in range(n_groups)])
            self.encoder_block2 = nn.ModuleList([build_layers()[0] for _ in range(n_groups)])

        self.dropout = nn.Dropout(dropout)

    def _run_block(self, x, layer_stack):
        attn_weights = None
        if self.param_sharing_strategy == 'between_group':
            for layer in layer_stack:
                x, attn_weights = layer(x)
        else:
            for layer in layer_stack:
                for _ in range(self.n_group_inner_layers):
                    x, attn_weights = layer(x)
        return x, attn_weights

    def impute(self, X, masks):
        input_first = torch.cat([X * masks, masks], dim=2) if self.input_with_mask else X
        input_first = self.embedding_1(input_first)
        enc_output = self.dropout(self.pos_enc(input_first))
        enc_output, _ = self._run_block(enc_output, self.encoder_block1)
        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        input_second = torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        input_second = self.embedding_2(input_second)
        enc_output = self.dropout(self.pos_enc(input_second))
        enc_output, attn_weights = self._run_block(enc_output, self.encoder_block2)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1).transpose(1, 2)

        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )

        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = masks * X + (1 - masks) * X_tilde_3

        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage='train'):
        X, masks = inputs['X'], inputs['missing_mask']
        X_holdout = inputs.get('X_holdout')
        indicating_mask = inputs.get('indicating_mask')

        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(X, masks)

        recon_loss = masked_mae_cal(X_tilde_1, X, masks)
        recon_loss += masked_mae_cal(X_tilde_2, X, masks)
        final_mae = masked_mae_cal(X_tilde_3, X, masks)
        recon_loss += final_mae
        recon_loss /= 3

        if (self.MIT or stage == 'val') and stage != 'test':
            imput_mae = masked_mae_cal(X_tilde_3, X_holdout, indicating_mask)
        else:
            imput_mae = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": recon_loss,
            "imputation_loss": imput_mae,
            "reconstruction_MAE": final_mae,
            "imputation_MAE": imput_mae,
        }
    

# === SAITSTrainer ===
class SAITSTrainer:
    def __init__(self, model: SAITS, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(batch, stage='train')
            loss = output['reconstruction_loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(batch, stage='val')
                losses.append(output['imputation_MAE'].item())
        return np.mean(losses)

    def predict(self, dataloader):
        self.model.eval()
        all_imputations = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(batch, stage='test')
                all_imputations.append(output['imputed_data'].cpu())
        return torch.cat(all_imputations, dim=0)


# === SAITSImputer Wrapper ===
class SAITSImputer:
    def __init__(self, seq_len=24, epochs=20, batch_size=64, learning_rate=1e-3,
                 d_model=64, d_inner=128, n_head=4, n_groups=2, n_group_inner_layers=1,
                 dropout=0.1, param_sharing_strategy='between_group', input_with_mask=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
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
        self.model = None
        self.trainer = None
        self.scaler = None
        self.input_size = None

    def _create_windows(self, data):
        T, D = data.shape
        windows = []
        for i in range(T - self.seq_len + 1):
            windows.append(data[i:i + self.seq_len])
        return np.stack(windows)

    def _create_masks(self, data):
        return (~np.isnan(data)).astype(np.float32)

    def fit(self, series: np.ndarray):
        if series.ndim == 1:
            series = series[:, None]

        self.input_size = series.shape[1]
        self.scaler = lambda x: x  # Identity if no normalization

        data = self._create_windows(series)
        masks = self._create_masks(data)

        X = torch.tensor(np.nan_to_num(data), dtype=torch.float32)
        mask = torch.tensor(masks, dtype=torch.float32)

        dataset = [{
            'X': X[i],
            'missing_mask': mask[i],
            'X_holdout': X[i],
            'indicating_mask': 1.0 - mask[i],
        } for i in range(len(X))]

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: {
            key: torch.stack([d[key] for d in x]) for key in x[0]
        })

        self.model = SAITS(
            input_size=self.input_size, seq_len=self.seq_len, d_model=self.d_model,
            d_inner=self.d_inner, n_head=self.n_head, n_groups=self.n_groups,
            n_group_inner_layers=self.n_group_inner_layers, dropout=self.dropout,
            param_sharing_strategy=self.param_sharing_strategy, input_with_mask=self.input_with_mask
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.trainer = SAITSTrainer(self.model, optimizer, device=self.device)

        # use tqdm
        for epoch in tqdm(range(self.epochs), desc="Training SAITS"):
            loss = self.trainer.train_epoch(dataloader)
            if epoch % 25 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}")
    def impute(self, series: np.ndarray) -> np.ndarray:
        if series.ndim == 1:
            series = series[:, None]

        data = self._create_windows(series)
        masks = self._create_masks(data)

        X = torch.tensor(np.nan_to_num(data), dtype=torch.float32)
        mask = torch.tensor(masks, dtype=torch.float32)

        dataset = [{
            'X': X[i],
            'missing_mask': mask[i],
            'X_holdout': X[i],
            'indicating_mask': 1.0 - mask[i],
        } for i in range(len(X))]

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: {
            key: torch.stack([d[key] for d in x]) for key in x[0]
        })

        imputed_tensor = self.trainer.predict(dataloader).numpy()

        T = series.shape[0]
        D = series.shape[1]
        recon = np.zeros((T, D))
        counts = np.zeros((T, D))

        for i in range(len(imputed_tensor)):
            recon[i:i + self.seq_len] += imputed_tensor[i]
            counts[i:i + self.seq_len] += 1

        counts[counts == 0] = 1e-9
        imputed = recon / counts
        return np.where(np.isnan(series), imputed, series)