"""foreblocks.anomaly.tranad.

This module implements the tranad pieces for its package.
It belongs to the anomaly detection and reconstruction workflows area of Foreblocks.
It exposes classes such as TranADDataset, TranADDetector.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from foreblocks.anomaly.models.tranad import TranAD


class TranADDataset(TensorDataset):
    """Memory-efficient dataset that creates sequences on-the-fly."""

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.length = data.shape[0] - seq_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len]


def create_sequences_vectorized(data: np.ndarray, seq_len: int) -> torch.Tensor:
    """Optimized sequence creation using vectorized operations."""
    if data.ndim == 1:
        data = data[:, None]

    n_samples = data.shape[0] - seq_len + 1
    if n_samples <= 0:
        raise ValueError(
            f"Data length {data.shape[0]} is too short for sequence length {seq_len}"
        )

    data_tensor = torch.from_numpy(data.T).float()
    sequences = data_tensor.unfold(1, seq_len, 1).permute(1, 2, 0)
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
        compile_model: bool = False,
        memory_efficient: bool = True,
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

        if self.memory_efficient:
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
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit_predict first.")

        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()
        if series.ndim == 1:
            series = series[:, None]

        scaled = self.scaler.transform(series)
        seqs = self._create_sequences(scaled)
        return self._infer(seqs)


__all__ = [
    "TranAD",
    "TranADDetector",
    "TranADDataset",
    "create_sequences_vectorized",
]
