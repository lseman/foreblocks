# Standard library
import copy
import importlib
import math
import time
import warnings

# Scientific computing and data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Machine learning and preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Typing
from typing import Dict, List, Optional, Tuple, Union

class TimeSeriesSeq2Seq:
    """Complete pipeline for time series forecasting with Seq2Seq models"""

    def __init__(
        self,
        model_type="lstm",
        model_params=None,
        training_params=None,
        device="cuda",
        input_processor=None,
        output_processor=None,
        output_block=None,
        input_normalization=None,
        output_normalization=None,
    ):
        """
        Initialize the time series forecasting pipeline

        Args:
            model_type: Type of model to use ('lstm', 'gru', 'transformer', 'hybrid')
            model_params: Dictionary of model parameters
            training_params: Dictionary of training parameters
            device: Device to use
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        self.device = device
        self.model = None
        self.scaler = None
        self.input_size = self.model_params.get("input_size", 1)
        self.output_size = self.model_params.get("output_size", 1)
        self.use_attention = self.model_params.get("use_attention", False)
        self.output_size = self.model_params.get("output_size", 1)
        self.hidden_size = self.model_params.get("hidden_size", 64)

        # Set default parameters if not provided
        if "hidden_size" not in self.model_params:
            self.model_params["hidden_size"] = 64
        if "d_model" not in self.model_params and model_type in [
            "transformer",
            "hybrid",
        ]:
            self.model_params["d_model"] = 128

        self.input_processor = input_processor
        self.output_processor = output_processor
        self.output_block = output_block
        self.input_normalization = input_normalization
        self.output_normalization = output_normalization
        # Initialize model
        self._build_model()

    def _build_model(self):
        """Build the appropriate model based on model_type"""
        # Optional input preprocessing with Fourier features
        input_preprocessor = self.input_processor

        # Optional output postprocessing with attention
        output_postprocessor = self.output_processor

        output_block = self.output_block

        attention_module = None
        if self.use_attention:
            attention_module = AttentionLayer(
                method=self.model_params.get("attention_method", "dot"),
                attention_backend=self.model_params.get(
                    "attention_backend", "xformers"
                ),
                encoder_hidden_size=self.hidden_size,
                decoder_hidden_size=self.hidden_size,
            )
        encoder = None
        decoder = None
        input_normalization = self.input_normalization
        output_normalization = self.output_normalization

        if self.model_type == "lstm":
            encoder = LSTMEncoder(
                input_size=self.model_params.get("input_processor_output_size", 1),
                hidden_size=self.hidden_size,
                num_layers=1,
                dropout=0.2,
                bidirectional=False,
            )
            # Create decoder
            decoder = LSTMDecoder(
                input_size=self.output_size,  # Previous output becomes input
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=1,
                dropout=0.2,
            )
        elif self.model_type == "gru":
            encoder = GRUEncoder(
                input_size=self.model_params.get("input_processor_output_size", 1),
                hidden_size=self.hidden_size,
                num_layers=1,
                dropout=0.2,
                bidirectional=False,
            )
            # Create decoder
            decoder = GRUDecoder(
                input_size=self.output_size,  # Previous output becomes input
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=1,
                dropout=0.2,
            )
        elif self.model_type == "transformer":
            encoder = TransformerEncoder(
                input_size=self.model_params.get("input_processor_output_size", 1),
                hidden_size=self.model_params.get("hidden_size", 64),
                nhead=self.model_params.get("nhead", 4),
                num_layers=self.model_params.get("num_encoder_layers", 1),
                dropout=self.model_params.get("dropout", 0.1),
                dim_feedforward=self.model_params.get("dim_feedforward", 2048),
            )

            # Create transformer decoder
            decoder = TransformerDecoder(
                input_size=self.model_params.get("input_processor_output_size", 1),
                hidden_size=self.model_params.get("hidden_size", 64),
                output_size=self.output_size,
                nhead=self.model_params.get("nhead", 4),
                num_layers=self.model_params.get("num_decoder_layers", 1),
                dropout=self.model_params.get("dropout", 0.1),
                dim_feedforward=self.model_params.get("dim_feedforward", 2048),
            )
        elif self.model_type == "informer":
            encoder = InformerEncoder(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                nhead=self.model_params.get("nhead", 8),
                num_layers=self.model_params.get("num_encoder_layers", 3),
                dropout=self.model_params.get("dropout", 0.1),
                distil=self.model_params.get("distil", True),
            )

            decoder = InformerDecoder(
                input_size=self.output_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                nhead=self.model_params.get("nhead", 8),
                num_layers=self.model_params.get("num_decoder_layers", 3),
                dropout=self.model_params.get("dropout", 0.1),
            )

        elif self.model_type == "vae":
            # Step 1: Base RNN encoder
            base_encoder = LSTMEncoder(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.model_params.get("num_layers", 1),
                dropout=self.model_params.get("dropout", 0.1),
                bidirectional=False,
            )

            # Step 2: Variational encoder wrapper
            encoder = VariationalEncoderWrapper(
                base_encoder=base_encoder,
                latent_dim=self.model_params.get("latent_size", 32),
            )

            # Step 3: Base RNN decoder
            base_decoder = LSTMDecoder(
                input_size=self.output_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.model_params.get("num_layers", 1),
                dropout=self.model_params.get("dropout", 0.1),
            )

            # Step 4: Latent-conditioned decoder wrapper
            decoder = LatentConditionedDecoder(
                base_decoder=base_decoder,
                latent_dim=self.model_params.get("latent_size", 32),
                hidden_size=self.hidden_size,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Create the Seq2Seq model
        self.model = ForecastingModel(
            encoder=encoder,
            decoder=decoder,
            input_preprocessor=input_preprocessor,
            output_postprocessor=output_postprocessor,
            output_block=output_block,
            attention_module=attention_module,
            teacher_forcing_ratio=0.5,
            target_len=self.model_params.get("target_len", 10),
            input_normalization=input_normalization,
            output_normalization=output_normalization,
            forecasting_strategy=self.model_params.get("strategy", "seq2seq"),
            model_type=self.model_type,
            input_processor_output_size=self.model_params.get(
                "input_processor_output_size", 16
            ),
            multi_encoder_decoder=self.model_params.get("multi_encoder_decoder", False),
        )
        self.model.to(self.device)

    def train(
        self, train_dataloader, val_dataloader=None, optimizer=None, criterion=None
    ):
        """
        Train the model

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            optimizer: Optional optimizer (if None, Adam will be used)
            criterion: Optional loss function (if None, MSELoss will be used)

        Returns:
            history: Dictionary containing training history
        """
        # Set up optimizer and criterion if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.training_params.get("learning_rate", 0.001),
            )

        if criterion is None:
            criterion = nn.MSELoss()
            # criterion= nn.SmoothL1Loss()

        # Training parameters
        num_epochs = self.training_params.get("num_epochs", 10)
        patience = self.training_params.get("patience", 5)
        use_amp = self.training_params.get("use_amp", True)

        # Train the model
        self.model, train_losses, val_losses = train_model(
            model=self.model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=num_epochs,
            device=self.device,
            use_amp=use_amp,
            patience=patience,
        )

        # Return training history
        history = {"train_losses": train_losses, "val_losses": val_losses}

        return history

    def predict(self, test_dataloader, plot=True):
        all_preds = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model.forward(X_batch)  # (batch_size, horizon, 1)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        all_preds = torch.cat(all_preds).squeeze(-1).numpy()  # (num_samples, horizon)
        all_targets = torch.cat(all_targets).squeeze(-1).numpy()

        mse = np.mean((all_preds - all_targets) ** 2)
        print(f"Mean Squared Error: {mse:.4f}")

        # plot
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(all_targets.flatten(), label="Target", alpha=0.5)
            plt.plot(all_preds.flatten(), label="Prediction", alpha=0.5)
            plt.xlabel("Time")
            plt.ylabel("Feature Value")
            plt.legend()
            plt.grid(True)
            plt.show()

        return all_targets, all_preds

    def save_model(self, path):
        """Save model to file"""
        torch.save(
            {
                "model_type": self.model_type,
                "model_params": self.model_params,
                "training_params": self.training_params,
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load_model(cls, path, device="cuda"):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=device)

        # Create instance with saved parameters
        instance = cls(
            model_type=checkpoint["model_type"],
            model_params=checkpoint["model_params"],
            training_params=checkpoint["training_params"],
            device=device,
        )

        # Load model weights
        instance.model.load_state_dict(checkpoint["model_state_dict"])

        return instance