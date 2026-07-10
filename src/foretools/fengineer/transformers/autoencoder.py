"""PyTorch autoencoder transformer for non-linear dense embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .aux import BaseFeatureTransformer, AutoencoderConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    """Configuration for AutoencoderTransformer."""

    enabled: bool = False
    latent_dim: int = 8
    encoder_arch: list[int] = field(default_factory=lambda: [64, 32])
    decoder_arch: list[int] = field(default_factory=lambda: [32, 64])
    activation: str = "relu"
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-4
    weight_decay: float = 1e-5
    use_bn: bool = True
    device: str = "auto"
    random_state: int = 42
    max_features: int = 100
    min_features: int = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _AEBlock(nn.Module):
    """Encoder/decoder block: Linear -> BN -> Activation -> Dropout."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_bn: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if use_bn and out_dim > 1:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(_act(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.block(x)


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()


class _AutoEncoder(nn.Module):
    """Builds encoder from encoder_arch and decoder from decoder_arch."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_arch: list[int],
        decoder_arch: list[int],
        activation: str = "relu",
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()
        if not encoder_arch:
            raise ValueError("encoder_arch must be non-empty")
        if encoder_arch[-1] != latent_dim:
            raise ValueError("encoder_arch[-1] must equal latent_dim")

        enc_dims = [input_dim] + encoder_arch
        enc_blocks = []
        for i_dim, o_dim in zip(enc_dims[:-1], enc_dims[1:]):
            enc_blocks.append(
                _AEBlock(i_dim, o_dim, activation, dropout, use_bn and o_dim > 1)
            )
        self.encoder = nn.Sequential(*enc_blocks)

        dec_dims = [latent_dim] + decoder_arch + [input_dim]
        dec_blocks = []
        for i_dim, o_dim in zip(dec_dims[:-1], dec_dims[1:]):
            # Last block: no BN (output layer)
            use_bn_block = use_bn and (o_dim != input_dim)
            dec_blocks.append(_AEBlock(i_dim, o_dim, activation, dropout, use_bn_block))
        self.decoder = nn.Sequential(*dec_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class AutoencoderTransformer(BaseFeatureTransformer):
    """
    Non-linear feature embedding via PyTorch autoencoder.

    Learns a compressed latent representation capturing non-linear
    relationships. Encoder output replaces original features.
    """

    def __init__(self, config: Any, ae_config: AutoencoderConfig | None = None):
        super().__init__(config)
        self.ae_config = ae_config or AutoencoderConfig()
        self._log = logging.getLogger(f"{__name__}.Autoencoder")

        self.input_cols_: list[str] = []
        self.input_medians_: dict[str, float] = {}
        self.input_scales_: dict[str, float] = {}
        self.model: _AutoEncoder | None = None
        self.latent_dim_: int = self.ae_config.latent_dim
        self.device_: str = "cpu"
        self.is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "AutoencoderTransformer":
        if not getattr(self.ae_config, "enabled", False):
            self.is_fitted = True
            return self

        _all_cols = X.select_dtypes(include=[np.number]).columns  # type: ignore[union-attr]
        num_cols = [str(c) for c in _all_cols]
        if len(num_cols) < self.ae_config.min_features:
            self.is_fitted = True
            return self

        if len(num_cols) > self.ae_config.max_features:
            var_s = X[num_cols].var()
            _nl = getattr(var_s, "nlargest")  # type: ignore[union-attr]
            if _nl is not None:
                num_cols = _nl(self.ae_config.max_features).index.tolist()
            else:
                num_cols = num_cols[: self.ae_config.max_features]

        self.input_cols_ = [str(c) for c in num_cols]
        # Convert ALL columns to float64 to handle Int64 from binning, etc.
        X_dict: dict[str, np.ndarray] = {}
        for col in self.input_cols_:
            try:
                X_dict[col] = np.asarray(
                    pd.to_numeric(X[col], errors="coerce"), dtype=np.float64
                )
            except Exception:
                X_dict[col] = np.zeros(len(X), dtype=np.float64)
        X_sub = pd.DataFrame(X_dict, index=X.index)

        for col in self.input_cols_:
            try:
                self.input_medians_[col] = float(np.nanmean(X_sub[col]))
            except Exception:
                self.input_medians_[col] = 0.0

        # Impute
        for col in self.input_cols_:
            try:
                X_sub[col] = X_sub[col].fillna(self.input_medians_.get(col, 0.0))
            except Exception:
                pass

        for col in self.input_cols_:
            try:
                col_data = X_sub[col].to_numpy(dtype=np.float32, copy=True)
            except Exception:
                col_data = np.zeros(len(X_sub), dtype=np.float32)
            try:
                m = float(np.nanmean(col_data))
                s = float(np.nanstd(col_data))
            except Exception:
                m, s = 0.0, 1.0
            self.input_scales_[col] = max(s, 1e-8)
            try:
                X_sub[col] = (X_sub[col] - m) / self.input_scales_[col]
            except Exception:
                pass

        X_np: np.ndarray = X_sub.to_numpy(dtype=np.float32, copy=True)  # type: ignore[union-attr]

        if self.ae_config.device == "auto":
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.ae_config.device
        if self.device_ == "cuda" and not torch.cuda.is_available():
            self._log.warning("CUDA requested but unavailable; falling back to CPU.")
            self.device_ = "cpu"

        self._train_model(X_np)
        self.is_fitted = True
        return self

    def _train_model(self, X: np.ndarray) -> None:
        cfg = self.ae_config
        rng = np.random.RandomState(cfg.random_state)
        torch.manual_seed(cfg.random_state)

        input_dim = X.shape[1]
        latent_dim = min(self.latent_dim_, input_dim // 2, max(2, input_dim - 1))
        self.latent_dim_ = latent_dim
        # Adjust encoder_arch to end with actual latent_dim
        encoder_arch = list(cfg.encoder_arch)
        if encoder_arch and encoder_arch[-1] != latent_dim:
            encoder_arch = encoder_arch[:-1] + [latent_dim]

        n = len(X)
        n_val = max(10, n // 10)
        indices = np.arange(n)
        rng.shuffle(indices)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=self.device_)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=self.device_)

        model = _AutoEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_arch=cfg.encoder_arch,
            decoder_arch=cfg.decoder_arch,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_bn=cfg.use_bn,
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, cfg.patience // 3)
        )
        criterion = nn.MSELoss()

        ds = TensorDataset(X_train, X_train)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

        best_state: dict[str, torch.Tensor] | None = None
        import math

        best_loss = math.inf  # float('inf')
        patience_counter = 0

        for epoch in range(cfg.epochs):
            model.train()
            train_loss = 0.0
            n_batches = 0
            for xb, _ in loader:
                optimizer.zero_grad()
                recon = model(xb)
                loss = criterion(recon, xb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            avg_train = train_loss / max(n_batches, 1)

            model.eval()
            with torch.no_grad():
                val_recon = model(X_val)
                val_loss = criterion(val_recon, X_val).item()

            scheduler.step(val_loss)

            if val_loss < best_loss - cfg.min_delta:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                self._log.info(
                    f"  Autoencoder early stop at epoch {epoch + 1}/{cfg.epochs} "
                    f"(val_loss={val_loss:.6f})"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if not self.model or not self.input_cols_:
            return pd.DataFrame(index=X.index)

        # Extract and convert ALL columns to float64
        X_sub_dict: dict[str, np.ndarray] = {}
        for col in self.input_cols_:
            if col in X.columns:
                try:
                    raw = pd.to_numeric(X[col], errors="coerce")
                    X_sub_dict[col] = np.asarray(raw, dtype=np.float64)
                except Exception:
                    X_sub_dict[col] = np.zeros(len(X), dtype=np.float64)
            else:
                try:
                    X_sub_dict[col] = np.full(
                        len(X),
                        float(self.input_medians_.get(col, 0.0)),
                        dtype=np.float64,
                    )
                except Exception:
                    X_sub_dict[col] = np.zeros(len(X), dtype=np.float64)
        X_sub = pd.DataFrame(X_sub_dict, index=X.index)

        # Impute missing values
        for col in self.input_cols_:
            try:
                X_sub[col] = X_sub[col].fillna(self.input_medians_.get(col, 0.0))
            except Exception:
                pass

        # Z-score
        for col in self.input_cols_:
            s = self.input_scales_.get(col, 1.0)
            m = self.input_medians_.get(col, 0.0)
            X_sub[col] = (X_sub[col] - m) / s

        X_np = X_sub.to_numpy(dtype=np.float32, copy=True)  # type: ignore[union-attr]
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device_)
            latent = self.model.encoder(tensor).cpu().numpy().astype(np.float32)

        cols = [f"ae_latent_{i}" for i in range(latent.shape[1])]
        return pd.DataFrame(latent, index=X.index).set_axis(cols, axis=1)

    def get_feature_names_out(self) -> list[str]:
        if not self.model:
            return []
        return [f"ae_latent_{i}" for i in range(self.latent_dim_)]
