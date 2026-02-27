"""
Example: AutoDA-Timeseries on synthetic time series tasks.

Demonstrates:
  1. Classification on synthetic waveform data
  2. Forecasting on synthetic autoregressive data
  3. Visualization of augmentation policies

Run: python example.py
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, "/home/claude")

from autoda_timeseries import (
    AutoDATimeseries,
    AutoDATrainer,
    extract_features,
    TRANSFORM_NAMES,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Synthetic Data Generation
# ──────────────────────────────────────────────────────────────────────

def make_classification_data(
    n_samples: int = 500,
    seq_len: int = 64,
    n_channels: int = 3,
    n_classes: int = 4,
    noise: float = 0.3,
):
    """Generate synthetic waveform classification data.

    Each class corresponds to a different frequency/amplitude pattern.
    """
    X, Y = [], []
    for i in range(n_samples):
        cls = i % n_classes
        t = np.linspace(0, 2 * np.pi, seq_len)
        channels = []
        for c in range(n_channels):
            freq = (cls + 1) * (c + 1) * 0.5
            amp = 1.0 + 0.5 * cls
            signal = amp * np.sin(freq * t + c * np.pi / 4)
            signal += noise * np.random.randn(seq_len)
            channels.append(signal)
        X.append(np.stack(channels, axis=-1))
        Y.append(cls)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


def make_forecasting_data(
    n_samples: int = 500,
    input_len: int = 96,
    pred_len: int = 24,
    n_channels: int = 1,
):
    """Generate synthetic autoregressive forecasting data."""
    total_len = input_len + pred_len
    X_input, Y_target = [], []

    for _ in range(n_samples):
        # AR(2) process + trend + seasonality
        series = np.zeros((total_len, n_channels))
        for c in range(n_channels):
            s = np.zeros(total_len)
            s[0] = np.random.randn()
            s[1] = np.random.randn()
            for t in range(2, total_len):
                s[t] = 0.6 * s[t-1] - 0.3 * s[t-2] + 0.1 * np.random.randn()
            # Add seasonality
            t_axis = np.arange(total_len)
            s += 0.5 * np.sin(2 * np.pi * t_axis / 12)
            s += 0.01 * t_axis  # trend
            series[:, c] = s

        X_input.append(series[:input_len])
        Y_target.append(series[input_len:])

    X = torch.tensor(np.array(X_input), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_target), dtype=torch.float32)
    return X, Y


# ──────────────────────────────────────────────────────────────────────
# 2. Simple Downstream Models
# ──────────────────────────────────────────────────────────────────────

class SimpleTCN(nn.Module):
    """Minimal TCN-like classifier for demonstration."""

    def __init__(self, input_channels: int, seq_len: int, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class SimpleRNN(nn.Module):
    """Minimal RNN forecaster for demonstration."""

    def __init__(self, input_channels: int, pred_len: int, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_channels, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, pred_len * input_channels)
        self.pred_len = pred_len
        self.input_channels = input_channels

    def forward(self, x):
        # x: (B, L, C)
        _, h = self.rnn(x)
        h = h[-1]  # last layer hidden state
        out = self.fc(h)  # (B, pred_len * C)
        return out.view(-1, self.pred_len, self.input_channels)


# ──────────────────────────────────────────────────────────────────────
# 3. Run Experiments
# ──────────────────────────────────────────────────────────────────────

def run_classification_experiment():
    """Classification experiment with AutoDA-Timeseries."""
    print("=" * 70)
    print("EXPERIMENT 1: Time Series Classification")
    print("=" * 70)

    # Generate data
    n_classes = 4
    X_train, Y_train = make_classification_data(400, seq_len=64, n_channels=3, n_classes=n_classes)
    X_val, Y_val = make_classification_data(100, seq_len=64, n_channels=3, n_classes=n_classes)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Baseline: NoAug ---
    print("\n--- Baseline: NoAug ---")
    model_noaug = SimpleTCN(input_channels=3, seq_len=64, n_classes=n_classes)
    opt = torch.optim.Adam(model_noaug.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model_noaug.to(device)

    for epoch in range(30):
        model_noaug.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model_noaug(xb), yb).backward()
            opt.step()

    model_noaug.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model_noaug(xb).argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    noaug_acc = correct / total
    print(f"  NoAug Validation Accuracy: {noaug_acc:.4f}")

    # --- AutoDA-Timeseries ---
    print("\n--- AutoDA-Timeseries ---")
    autoda = AutoDATimeseries(
        num_layers=3,
        hidden_dim=64,
        init_temperature=1.0,
        raw_bias=0.1,
    )
    downstream = SimpleTCN(input_channels=3, seq_len=64, n_classes=n_classes)

    trainer = AutoDATrainer(
        autoda=autoda,
        downstream_model=downstream,
        task="classification",
        lr=1e-3,
        aug_lr=5e-4,
        device=device,
    )

    history = trainer.fit(
        train_loader, val_loader,
        epochs=30, log_interval=10,
    )

    final_acc = history.get("val_accuracy", [0])[-1]
    print(f"\n  AutoDA-Timeseries Final Validation Accuracy: {final_acc:.4f}")
    print(f"  NoAug Validation Accuracy:                   {noaug_acc:.4f}")
    print(f"  Improvement: {(final_acc - noaug_acc) * 100:+.2f}%")

    # Print final augmentation policy
    print("\n  Final Augmentation Policy:")
    policy = autoda.get_policy_summary(
        *trainer._get_sample_policy(train_loader)
    )
    for layer_name, info in policy.items():
        print(f"    {layer_name} (temp={info['temperature']:.3f}):")
        probs = info["avg_probabilities"]
        top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
        for name, p in top3:
            print(f"      {name}: {p:.3f}")

    return history


def run_forecasting_experiment():
    """Forecasting experiment with AutoDA-Timeseries."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Time Series Forecasting")
    print("=" * 70)

    input_len, pred_len = 96, 24
    X_train, Y_train = make_forecasting_data(400, input_len, pred_len, n_channels=1)
    X_val, Y_val = make_forecasting_data(100, input_len, pred_len, n_channels=1)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Baseline: NoAug ---
    print("\n--- Baseline: NoAug ---")
    model_noaug = SimpleRNN(input_channels=1, pred_len=pred_len)
    opt = torch.optim.Adam(model_noaug.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model_noaug.to(device)

    for epoch in range(30):
        model_noaug.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model_noaug(xb), yb).backward()
            opt.step()

    model_noaug.eval()
    val_mse = 0
    count = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_mse += loss_fn(model_noaug(xb), yb).item()
            count += 1
    noaug_mse = val_mse / count
    print(f"  NoAug Validation MSE: {noaug_mse:.4f}")

    # --- AutoDA-Timeseries ---
    print("\n--- AutoDA-Timeseries ---")
    autoda = AutoDATimeseries(
        num_layers=3,
        hidden_dim=64,
        init_temperature=1.0,
        raw_bias=0.1,
    )
    downstream = SimpleRNN(input_channels=1, pred_len=pred_len)

    trainer = AutoDATrainer(
        autoda=autoda,
        downstream_model=downstream,
        task="forecasting",
        lr=1e-3,
        aug_lr=5e-4,
        device=device,
    )

    history = trainer.fit(
        train_loader, val_loader,
        epochs=30, log_interval=10,
    )

    final_mse = history["val_loss"][-1]
    print(f"\n  AutoDA-Timeseries Final Validation MSE: {final_mse:.4f}")
    print(f"  NoAug Validation MSE:                   {noaug_mse:.4f}")
    improvement = (noaug_mse - final_mse) / noaug_mse * 100
    print(f"  Improvement: {improvement:+.2f}%")

    return history


def demonstrate_augmentations():
    """Demonstrate individual augmentation transformations."""
    print("\n" + "=" * 70)
    print("DEMO: Augmentation Transformations")
    print("=" * 70)

    from autoda_timeseries.transformations import TRANSFORMATIONS, TRANSFORM_NAMES

    # Create a simple sine wave
    t = torch.linspace(0, 4 * np.pi, 100)
    x = torch.sin(t).unsqueeze(0).unsqueeze(-1)  # (1, 100, 1)
    intensity = torch.tensor([0.3])

    print(f"\nInput shape: {x.shape}")
    print(f"Intensity: {intensity.item():.2f}\n")

    for name, transform_fn in zip(TRANSFORM_NAMES, TRANSFORMATIONS):
        x_aug = transform_fn(x.clone(), intensity)
        diff = (x_aug - x).abs().mean().item()
        print(f"  {name:12s} -> mean abs difference: {diff:.4f}")


def demonstrate_features():
    """Demonstrate feature extraction."""
    print("\n" + "=" * 70)
    print("DEMO: Feature Extraction")
    print("=" * 70)

    # Create different types of time series
    t = torch.linspace(0, 4 * np.pi, 200)
    sine = torch.sin(t).unsqueeze(0).unsqueeze(-1)         # (1, 200, 1)
    noisy = (torch.sin(t) + 0.5 * torch.randn(200)).unsqueeze(0).unsqueeze(-1)
    trend = (0.01 * t + torch.sin(t)).unsqueeze(0).unsqueeze(-1)

    for name, x in [("Sine", sine), ("Noisy Sine", noisy), ("Trend + Sine", trend)]:
        features = extract_features(x)
        print(f"\n  {name} -> feature shape: {features.shape}")
        print(f"    Mean: {features[0, 0]:.4f}, Std: {features[0, 1]:.4f}, "
              f"Skew: {features[0, 2]:.4f}, Kurt: {features[0, 3]:.4f}")
        print(f"    AC1: {features[0, 8]:.4f}, AC2: {features[0, 9]:.4f}, "
              f"SpectralEnt: {features[0, 17]:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    demonstrate_augmentations()
    demonstrate_features()
    run_classification_experiment()
    run_forecasting_experiment()

    print("\n" + "=" * 70)
    print("All experiments completed.")
    print("=" * 70)
