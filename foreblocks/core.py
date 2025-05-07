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

# EWT
import ewtpy
from ewtpy import EWT1D

from .aux import *
#from extra import *
from .enc_dec import *
from .att import *


class ForecastingModel(nn.Module):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        target_len=5,
        forecasting_strategy="seq2seq",  # seq2seq | autoregressive | direct
        input_preprocessor=None,
        output_postprocessor=None,
        attention_module=None,
        teacher_forcing_ratio=0.5,
        scheduled_sampling_fn=None,
        output_size=None,
        output_block=None,
        input_normalization=None,
        output_normalization=None,
        model_type="lstm",
        input_skip_connection=False,
        multi_encoder_decoder=False,
        input_processor_output_size=16,
        hidden_size=64,
        enc_embbedding=None,
        dec_embedding=None,
    ):
        super(ForecastingModel, self).__init__()

        if multi_encoder_decoder:
            assert (
                input_processor_output_size is not None
            ), "input_size must be provided for multi_encoder_decoder"
            assert (
                encoder is not None and decoder is not None
            ), "You must provide a base encoder and decoder module"

            # Automatically clone encoder/decoder for each input feature
            self.encoder = nn.ModuleList(
                [
                    self._clone_module(encoder)
                    for _ in range(input_processor_output_size)
                ]
            )
            self.decoder = nn.ModuleList(
                [
                    self._clone_module(decoder)
                    for _ in range(input_processor_output_size)
                ]
            )
            self.decoder_aggregator = nn.Linear(
                input_processor_output_size, 1, bias=False
            )  # input_size = num decoders

        else:
            self.encoder = encoder
            self.decoder = decoder

        self.multi_encoder_decoder = multi_encoder_decoder
        self.hidden_size = hidden_size
        self.strategy = forecasting_strategy
        self.target_len = target_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduled_sampling_fn = scheduled_sampling_fn
        self.model_type = model_type

        self.attention_module = attention_module
        self.use_attention = self.attention_module is not None

        self.output_size = output_size or (decoder.output_size if decoder else None)
        self.pred_len = output_size or target_len
        self.output_block = output_block or nn.Identity()

        self.input_preprocessor = input_preprocessor or nn.Identity()
        self.output_postprocessor = output_postprocessor or nn.Identity()
        self.input_normalization = input_normalization or nn.Identity()
        self.output_normalization = output_normalization or nn.Identity()
        self.start_token = (
            nn.Linear(self.hidden_size, self.output_size) if decoder else None
        )
        self.time_features_dim = 1  # Default to 1, can be changed based on your needs
        print(decoder)
        self.label_len = self.decoder.output_size if decoder else None

        self.input_skip_connection = input_skip_connection

        if model_type == "transformer":
            if enc_embbedding is not None:
                self.enc_embedding = enc_embbedding
            else:
                self.enc_embedding = TimeSeriesEncoder(encoder.input_size, encoder.hidden_size)
            if dec_embedding is not None:
                self.dec_embedding = dec_embedding
            else:
                self.dec_embedding = TimeSeriesEncoder(encoder.input_size, encoder.hidden_size)

        if self.use_attention:
            self.output_layer = nn.Linear(
                decoder.output_size + encoder.hidden_size, self.output_size
            )
        elif decoder:
            self.output_layer = nn.Linear(decoder.output_size, self.output_size)
        else:
            self.output_layer = nn.Identity()

        self.input_size = encoder.input_size

        self.project_output = nn.Linear(
            self.input_size, self.output_size
        )


    def _clone_module(self, module):
        return copy.deepcopy(module)

    def forward(self, src, targets=None, epoch=None):
        if self.strategy == "direct":
            # Direct models like Informer take a tuple of inputs
            return self._forward_direct(src)

        # seq2seq or autoregressive: src is a tensor
        if self.input_preprocessor is not None:
            if self.input_skip_connection:
                src = self.input_preprocessor(src) + src
            else:
                src = self.input_preprocessor(src)

        src = self.input_normalization(src)

        if self.strategy == "seq2seq":
            if self.model_type == "transformer":
                return self._forward_transformer_seq2seq(src, targets, epoch)
            if self.multi_encoder_decoder:
                return self._forward_seq2seq_multi(src, targets, epoch)  # NEW
            return self._forward_seq2seq(src, targets, epoch)

        elif self.strategy == "transformer_seq2seq":
            return self._forward_transformer_seq2seq(src, targets, epoch)
        elif self.strategy == "autoregressive":
            return self._forward_autoregressive(src, targets, epoch)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def _forward_seq2seq(self, src, targets, epoch):
        batch_size, _, _ = src.shape
        device = src.device

        encoder_outputs, encoder_hidden = self.encoder(src)

        if encoder_outputs.dim() == 3 and encoder_outputs.shape[1] != src.shape[1]:
            encoder_outputs = encoder_outputs.transpose(0, 1)

        # Handle VAE latent outputs
        if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
            z, mu, logvar = encoder_hidden
            decoder_hidden = (z,)
            self._kl = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
            )
        else:
            decoder_hidden = self._prepare_decoder_hidden(encoder_hidden)
            self._kl = None

        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)
        outputs = torch.zeros(
            batch_size, self.target_len, self.output_size, device=device
        )

        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            if self.use_attention:
                query = self._get_attention_query(decoder_output, decoder_hidden)
                context, _ = self.attention_module(query, encoder_outputs)
                decoder_output = self.output_layer(
                    torch.cat((decoder_output, context), dim=-1)
                )
            else:
                decoder_output = self.output_layer(decoder_output)

            decoder_output = self.output_block(decoder_output)
            decoder_output = self.output_normalization(decoder_output)
            outputs[:, t : t + 1] = decoder_output.unsqueeze(1)

            if targets is not None:
                use_tf = torch.rand(1).item() < (
                    self.scheduled_sampling_fn(epoch)
                    if self.scheduled_sampling_fn
                    else self.teacher_forcing_ratio
                )
                decoder_input = (
                    targets[:, t : t + 1] if use_tf else decoder_output.unsqueeze(1)
                )
            else:
                decoder_input = decoder_output.unsqueeze(1)

        return self.output_postprocessor(outputs)

    def _forward_seq2seq_multi(self, src, targets, epoch):
        batch_size, seq_len, input_size = src.shape
        device = src.device

        decoder_outputs_list = []

        for i in range(input_size):
            x_i = src[:, :, i].unsqueeze(-1)  # [B, T, 1]
            encoder_i = self.encoder[i]
            decoder_i = self.decoder[i]

            encoder_outputs, encoder_hidden = encoder_i(x_i)

            # Align encoder_outputs if needed
            if encoder_outputs.dim() == 3 and encoder_outputs.shape[1] != seq_len:
                encoder_outputs = encoder_outputs.transpose(0, 1)

            # Handle VAE-style encoder
            if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
                z, mu, logvar = encoder_hidden
                decoder_hidden = (z,)
                self._kl = (
                    -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
                )
            else:
                decoder_hidden = self._prepare_decoder_hidden(encoder_hidden)
                self._kl = None

            # Decoder outputs for this feature
            decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)
            feature_outputs = torch.zeros(
                batch_size, self.target_len, self.output_size, device=device
            )

            for t in range(self.target_len):
                decoder_output, decoder_hidden = decoder_i(
                    decoder_input, decoder_hidden
                )

                if self.use_attention:
                    query = self._get_attention_query(decoder_output, decoder_hidden)
                    context, _ = self.attention_module(query, encoder_outputs)
                    decoder_output = self.output_layer(
                        torch.cat((decoder_output, context), dim=-1)
                    )
                else:
                    decoder_output = self.output_layer(decoder_output)

                decoder_output = self.output_block(decoder_output)
                decoder_output = self.output_normalization(decoder_output)

                feature_outputs[:, t : t + 1] = decoder_output.unsqueeze(1)

                if targets is not None:
                    use_tf = torch.rand(1).item() < (
                        self.scheduled_sampling_fn(epoch)
                        if self.scheduled_sampling_fn
                        else self.teacher_forcing_ratio
                    )
                    decoder_input = (
                        targets[:, t : t + 1] if use_tf else decoder_output.unsqueeze(1)
                    )
                else:
                    decoder_input = decoder_output.unsqueeze(1)

            decoder_outputs_list.append(feature_outputs)  # shape [B, T, output_size]

        # Aggregate across features
        # Stack: [num_decoders, B, T, output_size] → [B, T, num_decoders, output_size]
        stacked = (
            torch.stack(decoder_outputs_list, dim=0).permute(1, 2, 0, 3).squeeze(3)
        )
        # print(f"Stacked shape: {stacked.shape}")

        # Apply linear aggregation over decoder dimension → [B, T, 1, output_size]
        outputs = self.decoder_aggregator(stacked).squeeze(
            1
        )  # shape [B, T, 1, output_size]
        # print(f"Aggregated shape: {outputs.shape}")

        return self.output_postprocessor(outputs)

    def _forward_transformer_seq2seq(self, src, targets, epoch):
        batch_size, src_seq_len, _ = src.shape
        device = src.device

        # Encode the source
        enc_out = self.enc_embedding(src)
        enc_out = self.encoder(enc_out)

        if self.training and targets is not None:
            # === Scheduled Sampling: Use TF with probability
            use_tf = torch.rand(1).item() < (
                self.scheduled_sampling_fn(epoch)
                if self.scheduled_sampling_fn
                else self.teacher_forcing_ratio
            )
            if use_tf:
                # === Teacher Forcing ===
                # Pad targets if output_size < input_size
                if self.output_size != self.input_size:
                    pad_size = self.input_size - self.output_size
                    padding = torch.zeros(targets.size(0), targets.size(1), pad_size, device=targets.device)
                    targets_padded = torch.cat([targets, padding], dim=-1)
                else:
                    targets_padded = targets

                x_dec = torch.cat([src[:, -self.label_len:, :], targets_padded], dim=1)

                tgt_mask = self._generate_square_subsequent_mask(x_dec.size(1)).to(device)
                dec_out = self.dec_embedding(x_dec)
                output = self.decoder(dec_out, enc_out, tgt_mask=tgt_mask)
                output = self.output_layer(output)
                return output[:, -self.pred_len:, :]

            else:
                # === Autoregressive decoding (no teacher forcing)
                preds = []
                x_dec_so_far = src[:, -self.label_len:, :]

                for step in range(self.pred_len):
                    tgt_mask = self._generate_square_subsequent_mask(x_dec_so_far.size(1)).to(device)

                    dec_embed = self.dec_embedding(x_dec_so_far)
                    out = self.decoder(dec_embed, enc_out, tgt_mask=tgt_mask)
                    pred_t = out[:, -1:, :]
                    pred_t = self.output_layer(pred_t)  # shape: [B, 1, output_size]
                    preds.append(pred_t)

                    # === Pad pred_t to match input_size ===
                    if self.output_size != self.input_size:
                        pad_size = self.input_size - self.output_size
                        padding = torch.zeros(pred_t.size(0), 1, pad_size, device=pred_t.device)
                        pred_t_padded = torch.cat([pred_t, padding], dim=-1)
                    else:
                        pred_t_padded = pred_t

                    x_dec_so_far = torch.cat([x_dec_so_far, pred_t_padded], dim=1)

                return torch.cat(preds, dim=1)


        else:
            # Inference
            preds = []
            x_dec_so_far = src[:, -self.label_len:, :]  # [B, label_len, input_size]

            for step in range(self.pred_len):
                tgt_mask = self._generate_square_subsequent_mask(x_dec_so_far.size(1)).to(device)
                dec_embed = self.dec_embedding(x_dec_so_far)  # [B, T, d_model]
                out = self.decoder(dec_embed, enc_out, tgt_mask=tgt_mask)
                pred_t = out[:, -1:, :]  # [B, 1, d_model]
                pred_t = self.output_layer(pred_t)  # [B, 1, output_size]
                preds.append(pred_t)

                # Pad pred_t to match input_size
                if self.output_size != self.input_size:
                    pad_size = self.input_size - self.output_size
                    padding = torch.zeros(pred_t.size(0), 1, pad_size, device=pred_t.device)
                    pred_t_padded = torch.cat([pred_t, padding], dim=-1)
                else:
                    pred_t_padded = pred_t

                x_dec_so_far = torch.cat([x_dec_so_far, pred_t_padded], dim=1)

            return torch.cat(preds, dim=1)


    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the decoder to prevent attending to future time steps.
        """
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        mask.fill_diagonal_(0)
        return mask


    def _generate_time_features(self, seq_len, device):
        """
        Generate simple positional features for time series.
        For more complex time features, you could incorporate day of week,
        hour of day, etc. if that information is available.
        """
        batch_size = 1  # Will be broadcast to all batches

        # Create simple positional features
        pos = torch.arange(0, seq_len, device=device).unsqueeze(0).unsqueeze(-1).float()
        pos = pos / seq_len  # Normalize to [0, 1]

        # Option: Create more complex features for the time dimension
        # For example, sine and cosine features like in positional encoding
        time_features = torch.zeros(
            batch_size, seq_len, self.time_features_dim, device=device
        )

        # Simple increment
        time_features[:, :, 0] = pos.squeeze(-1)

        # Sine features at different frequencies (similar to positional encoding)
        if self.time_features_dim > 1:
            for i in range(1, self.time_features_dim):
                if i % 2 == 1:
                    # Sine with different frequencies
                    time_features[:, :, i] = torch.sin(pos * (2 ** (i // 2)))
                else:
                    # Cosine with different frequencies
                    time_features[:, :, i] = torch.cos(pos * (2 ** (i // 2 - 1)))

        return time_features

    def _forward_autoregressive(self, src, targets, epoch):
        batch_size, _, _ = src.shape
        decoder_input = src[:, -1:, :]  # last time step
        outputs = []

        for t in range(self.target_len):
            decoder_output = self.decoder(decoder_input)
            decoder_output = self.output_normalization(decoder_output)
            outputs.append(decoder_output)

            if targets is not None:
                use_tf = torch.rand(1).item() < (
                    self.scheduled_sampling_fn(epoch)
                    if self.scheduled_sampling_fn
                    else self.teacher_forcing_ratio
                )
                decoder_input = targets[:, t : t + 1] if use_tf else decoder_output
            else:
                decoder_input = decoder_output

        return self.output_postprocessor(torch.cat(outputs, dim=1))

    def _forward_direct(self, src):
        output = self.decoder(src)
        output = self.output_normalization(output)
        return self.output_postprocessor(output)

    def _prepare_decoder_hidden(self, encoder_hidden):
        if isinstance(encoder_hidden, tuple):
            h_n, c_n = encoder_hidden
            if hasattr(self.encoder, "bidirectional") and self.encoder.bidirectional:
                if self.encoder.num_layers != self.decoder.num_layers:
                    h_n = h_n[-self.decoder.num_layers :]
                    c_n = c_n[-self.decoder.num_layers :]
                if h_n.size(0) % 2 == 0:
                    h_n = h_n.view(
                        self.encoder.num_layers, 2, -1, self.encoder.hidden_size
                    ).sum(dim=1)
                    c_n = c_n.view(
                        self.encoder.num_layers, 2, -1, self.encoder.hidden_size
                    ).sum(dim=1)
                return (h_n, c_n)
            return encoder_hidden
        else:
            h_n = encoder_hidden
            if hasattr(self.encoder, "bidirectional") and self.encoder.bidirectional:
                if self.encoder.num_layers != self.decoder.num_layers:
                    h_n = h_n[-self.decoder.num_layers :]
                if h_n.size(0) % 2 == 0:
                    h_n = h_n.view(
                        self.encoder.num_layers, 2, -1, self.encoder.hidden_size
                    ).sum(dim=1)
                return h_n
            return encoder_hidden

    def _get_attention_query(self, decoder_output, decoder_hidden):
        if hasattr(self.decoder, "is_transformer") and self.decoder.is_transformer:
            return decoder_hidden.permute(1, 0, 2)
        else:
            return (
                decoder_hidden[0][-1]
                if isinstance(decoder_hidden, tuple)
                else decoder_hidden[-1]
            )

    def get_kl(self):
        return getattr(self, "_kl", None)


# Example usage and complete pipeline
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



class TimeSeriesPreprocessor:
    """State-of-the-art preprocessing for time series data."""

    def __init__(
        self,
        normalize=True,
        differencing=False,
        detrend=True,
        apply_ewt=True,
        window_size=24,
        horizon=10,
        remove_outliers=True,
        outlier_threshold=0.05,
        outlier_method="iqr",
        impute_method="auto",
        ewt_bands=5,
        trend_imf_idx=0,
        log_transform=False,
        filter_window=5,
        filter_polyorder=2,
        apply_filter=True,
    ):

        self.normalize = normalize
        self.differencing = differencing
        self.detrend = detrend
        self.apply_ewt = apply_ewt
        self.window_size = window_size
        self.horizon = horizon
        self.ewt_bands = ewt_bands
        self.trend_imf_idx = trend_imf_idx
        self.log_transform = log_transform
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.apply_filter = apply_filter
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.outlier_method = outlier_method
        self.impute_method = impute_method

        self.scaler = None
        self.log_offset = None
        self.diff_values = None
        self.trend_component = None
        self.ewt_components = None
        self.ewt_boundaries = None

    def _plot_comparison(self, original, cleaned, title="Preprocessing Comparison", time_stamps=None):
        # Ensure 2D shape
        original = np.atleast_2d(original)
        cleaned = np.atleast_2d(cleaned)

        #print(original.shape, cleaned.shape)

        # Transpose if needed (1, n) → (n, 1)
        if original.shape[0] == 1:
            original = original.T
        if cleaned.shape[0] == 1:
            cleaned = cleaned.T

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x = time_stamps if time_stamps is not None else np.arange(original.shape[0])

        axs[0].set_title("Original", fontsize=12)
        axs[1].set_title("Cleaned", fontsize=12)

        n_features = original.shape[1]
        for i in range(n_features):
            axs[0].plot(x, original[:, i], label=f"Feature {i}", linewidth=1.5, alpha=0.8)
            axs[1].plot(x, cleaned[:, i], label=f"Feature {i}", linewidth=1.5, alpha=0.8)

        axs[0].legend(loc="upper right", fontsize=9)
        axs[1].legend(loc="upper right", fontsize=9)

        axs[0].grid(True)
        axs[1].grid(True)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    def generate_time_features(self, timestamps, freq='h'):
        df = pd.DataFrame({'ts': pd.to_datetime(timestamps)})
        df['month'] = df.ts.dt.month / 12.0
        df['day'] = df.ts.dt.day / 31.0
        df['weekday'] = df.ts.dt.weekday / 6.0
        df['hour'] = df.ts.dt.hour / 23.0 if freq in ['h', 'H'] else 0.0
        return df[['month', 'day', 'weekday', 'hour']].values.astype(np.float32)


    def adaptive_filter(self, data):
        return savgol_filter(data, self.filter_window, self.filter_polyorder, axis=0)

    def _remove_outliers(self, data):
        method = self.outlier_method
        data = np.asarray(data)
        is_1d = data.ndim == 1
        data = data.reshape(-1, 1) if is_1d else data
        print(
            f"[Outlier Removal] Method: {method}, Threshold: {self.outlier_threshold}"
        )

        if method == "iqr":
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower = Q1 - self.outlier_threshold * IQR
            upper = Q3 + self.outlier_threshold * IQR
            return np.clip(data, lower, upper)

        elif method == "zscore":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            z = (data - mean) / std
            return np.where(
                np.abs(z) <= self.outlier_threshold,
                data,
                np.clip(
                    data,
                    mean - self.outlier_threshold * std,
                    mean + self.outlier_threshold * std,
                ),
            )

        elif method == "mad":
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            z = np.abs((data - med) / (mad + 1e-6))
            return np.where(z <= self.outlier_threshold * 1.4826, data, med)

        elif method == "quantile":
            lower = np.percentile(data, self.outlier_threshold * 100, axis=0)
            upper = np.percentile(data, (1 - self.outlier_threshold) * 100, axis=0)
            return np.clip(data, lower, upper)

        elif method == "isolation_forest":
            df = pd.DataFrame(data)
            for col in df.columns:
                df[f"{col}_mean"] = df[col].rolling(5, min_periods=1).mean()
                df[f"{col}_std"] = df[col].rolling(5, min_periods=1).std().fillna(0)
            X = StandardScaler().fit_transform(
                df.fillna(method="ffill").fillna(method="bfill")
            )
            preds = IsolationForest(
                contamination=self.outlier_threshold, random_state=42
            ).fit_predict(X)
            result = np.where(preds[:, None] == 1, data, np.nan)
            return result

        elif method == "lof":
            preds = LocalOutlierFactor(
                n_neighbors=20, contamination=self.outlier_threshold
            ).fit_predict(data)
            return np.where(preds[:, None] == 1, data, np.nan)

        elif method == "ecod":
            from pyod.models.ecod import ECOD

            preds = ECOD().fit(data).predict(data)
            return np.where(preds[:, None] == 0, data, np.nan)

        else:
            raise ValueError(f"Unsupported outlier removal method: {method}")

    def _impute_missing(self, data):
        df = pd.DataFrame(data)
        method = self.impute_method
        print(f"[Imputation] Method: {method}")

        if method == "auto":
            missing = df.isna().mean()
            result = pd.DataFrame(index=df.index)
            for col in df.columns:
                col_missing = missing[col]
                if col_missing < 0.05:
                    print(f" - Feature {col}: interpolate")
                    result[col] = df[col].interpolate().ffill().bfill()
                elif col_missing < 0.2 and df.corr()[col].drop(col).abs().max() > 0.6:
                    print(f" - Feature {col}: knn")
                    result[col] = (
                        KNNImputer(n_neighbors=3).fit_transform(df[[col]]).ravel()
                    )
                else:
                    try:
                        from fancyimpute import IterativeImputer

                        print(f" - Feature {col}: iterative")
                        result[col] = (
                            IterativeImputer().fit_transform(df[[col]]).ravel()
                        )
                    except ImportError:
                        print(f" - Feature {col}: mean (fallback)")
                        result[col] = df[col].fillna(df[col].mean())
            return result.values

        elif method == "interpolate":
            return df.interpolate().ffill().bfill().values
        elif method == "mean":
            return df.fillna(df.mean()).values
        elif method == "ffill":
            return df.ffill().bfill().values
        elif method == "bfill":
            return df.bfill().ffill().values
        elif method == "knn":
            return KNNImputer(n_neighbors=5).fit_transform(df)
        elif method == "iterative":
            try:
                from fancyimpute import IterativeImputer

                return IterativeImputer().fit_transform(df)
            except ImportError:
                warnings.warn("fancyimpute not installed. Using mean instead.")
                return df.fillna(df.mean()).values
        else:
            raise ValueError(f"Unsupported impute method: {self.impute_method}")

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size - self.horizon + 1):
            X.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size : i + self.window_size + self.horizon])
        return np.array(X), np.array(y)
    
    def fit_transform(self, data, time_stamps=None, feats=None):
        processed = data.copy()
        original = data.copy()

        if self.log_transform:
            min_val = processed.min(axis=0)
            self.log_offset = np.where(min_val <= 0, np.abs(min_val) + 1.0, 0.0)
            processed = np.log(processed + self.log_offset)

        imputed = self._impute_missing(processed)
        self._plot_comparison(processed, imputed, title="After Imputation")
        processed = imputed
        
        if self.remove_outliers:
            cleaned = np.zeros_like(processed)
            for i in range(processed.shape[1]):
                _clean = self._remove_outliers(processed[:, i])
                cleaned[:, i] = _clean.ravel()
                
            self._plot_comparison(processed, cleaned, title="After Outlier Removal")
            processed = cleaned

        if self.apply_ewt:
            self.ewt_components = []
            self.ewt_boundaries = []
            if self.detrend:
                self.trend_component = np.zeros_like(processed)

            for i in range(processed.shape[1]):
                signal = processed[:, i]
                ewt, _, boundaries = EWT1D(signal, N=self.ewt_bands)
                self.ewt_components.append(ewt)
                self.ewt_boundaries.append(boundaries)

                if self.detrend:
                    trend = ewt[:, self.trend_imf_idx]
                    self.trend_component[:, i] = trend
                    detrended = signal - trend

                    # Plotting
                    x = time_stamps if time_stamps is not None else np.arange(len(signal))
                    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

                    axs[0].plot(x, signal, label="Original", alpha=0.8)
                    axs[0].plot(x, trend, label="Trend", linestyle="-", linewidth=2.5)
                    axs[0].set_title(f"Feature {i} - Original and Trend", fontsize=12)
                    axs[0].legend()
                    axs[0].grid(True)

                    axs[1].plot(x, detrended, label="Detrended", color="tab:orange", linewidth=2)
                    axs[1].set_title(f"Feature {i} - Detrended", fontsize=12)
                    axs[1].legend()
                    axs[1].grid(True)

                    plt.tight_layout()
                    plt.show()

                    processed[:, i] = detrended

        if self.apply_filter:
            filtered = self.adaptive_filter(processed)
            self._plot_comparison(processed, filtered, title="After Adaptive Filtering")
            processed = filtered

        if self.differencing:
            self.diff_values = processed[0:1].copy()
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0

        if self.normalize:
            self.scaler = StandardScaler()
            processed = self.scaler.fit_transform(processed)

        if time_stamps is not None:
            time_features = self.generate_time_features(time_stamps)
            processed = np.concatenate((processed, time_features), axis=1)

        return self._create_sequences(processed, feats) + (processed,)

    def transform(self, data):
        """
        Transform data using fitted parameters

        Args:
            data: Time series data of shape [samples, features]

        Returns:
            X: Input sequences
            y: Target sequences (if possible to create)
        """
        # Check if the preprocessor has been fitted
        if self.normalize and self.scaler is None:
            raise ValueError(
                "Preprocessor has not been fitted yet. Call fit_transform first."
            )

        # Make a copy to avoid modifying the original data
        processed_data = data.copy()

        # Apply log transform if specified
        if self.log_transform:
            if self.log_offset is None:
                raise ValueError("Log offset not set. Call fit_transform first.")
            processed_data = np.log(processed_data + self.log_offset)

        # If multivariate, process each feature independently
        n_samples, n_features = processed_data.shape

        # Store original data for EWT processing
        original_data = processed_data.copy()

        # Apply EWT if specified
        if self.apply_ewt:
            if self.ewt_boundaries is None:
                raise ValueError("EWT boundaries not set. Call fit_transform first.")

            # Process each feature independently for EWT
            for i in range(n_features):
                # Apply EWT to the current feature using stored boundaries
                feature_data = original_data[:, i]

                # For transform, use the boundaries detected during fit
                ewt, _, _ = ewtpy.EWT1D(
                    feature_data,
                    N=len(self.ewt_boundaries[i]),
                    detect="given_bounds",
                    boundaries=self.ewt_boundaries[i],
                )

                # If detrending is enabled, remove the trend component
                if self.detrend:
                    # Extract the trend component
                    trend = ewt[:, self.trend_imf_idx]
                    plt.plot(trend)
                    plt.title(f"Trend Component for Feature {i}")
                    plt.show()

                    # Remove trend from the data
                    processed_data[:, i] = processed_data[:, i] - trend
                    plt.plot(processed_data[:, i])
                    plt.title(f"Processed Feature {i} after Detrending")
                    plt.show()

        # Apply differencing if specified
        if self.differencing:
            if self.diff_values is None:
                raise ValueError(
                    "Differencing values not set. Call fit_transform first."
                )

            # Store the first value for inverse transform
            prev_value = processed_data[0:1].copy()

            # Apply differencing
            processed_data[1:] = np.diff(processed_data, axis=0)
            processed_data[0] = 0

        # Apply normalization if specified
        if self.normalize:
            processed_data = self.scaler.transform(processed_data)

        # Apply adaptive filtering if specified
        if self.apply_filter:
            processed_data = self.adaptive_filter(processed_data)

        # Create input/output sequences
        X, y = self._create_sequences(processed_data)

        return X, y

    def inverse_transform(self, predictions):
        """
        Inverse transform forecasted values back to original scale

        Args:
            predictions: Forecasted values

        Returns:
            Predictions in the original scale
        """
        # Inverse normalization
        if self.normalize:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit_transform first.")

            predictions = self.scaler.inverse_transform(predictions)

        # Inverse differencing
        if self.differencing:
            if self.diff_values is None:
                raise ValueError(
                    "Differencing values not set. Call fit_transform first."
                )

            # Initialize with the last known value
            last_value = self.diff_values[-1]
            result = np.zeros_like(predictions)

            # Integrate the differences
            for i in range(len(predictions)):
                last_value = last_value + predictions[i]
                result[i] = last_value

            predictions = result

        # Add back trend component if detrending was applied
        if self.detrend and self.trend_component is not None:
            # For simplicity, we'll extrapolate the trend from the training data
            # In a production system, you might want to use a more sophisticated approach

            # Check if we have enough trend data
            if len(self.trend_component) >= len(predictions):
                # Use the corresponding portion of the trend
                trend_to_add = self.trend_component[: len(predictions)]
            else:
                # Extrapolate trend for longer predictions
                # This is a simple linear extrapolation
                n_samples, n_features = predictions.shape
                trend_to_add = np.zeros_like(predictions)

                for i in range(n_features):
                    # Extract trend for the current feature
                    feature_trend = self.trend_component[:, i]

                    # Compute slope for extrapolation (using last few points)
                    window = min(10, len(feature_trend) // 2)
                    slope = (feature_trend[-1] - feature_trend[-window - 1]) / window

                    # Extrapolate trend
                    for j in range(n_samples):
                        if j < len(feature_trend):
                            trend_to_add[j, i] = feature_trend[j]
                        else:
                            # Linear extrapolation
                            trend_to_add[j, i] = feature_trend[-1] + slope * (
                                j - len(feature_trend) + 1
                            )

            # Add trend back to predictions
            predictions = predictions + trend_to_add

        # Inverse log transform if applied
        if self.log_transform:
            if self.log_offset is None:
                raise ValueError("Log offset not set. Call fit_transform first.")
            predictions = np.exp(predictions) - self.log_offset

        return predictions
    
    def _create_sequences(self, data, feats=None):
        """
        Create input/output sequences for training or inference

        Args:
            data: Preprocessed time series data
            feats: List of column indices to extract for y

        Returns:
            X: Input sequences of shape [n_sequences, window_size, n_features]
            y: Target sequences of shape [n_sequences, horizon, len(feats)]
        """
        n_samples, n_features = data.shape
        feats = list(range(n_features)) if feats is None else feats
        X, y = [], []

        for i in range(n_samples - self.window_size - self.horizon + 1):
            X.append(data[i : i + self.window_size])  # full features
            y_seq = data[i + self.window_size : i + self.window_size + self.horizon]
            y.append(y_seq[:, feats])  # select only specified features
        print(f"X shape: {np.array(X).shape}, y shape: {np.array(y).shape}")
        return np.array(X), np.array(y)


    def get_ewt_components(self):
        """
        Get the decomposed EWT components (IMFs)

        Returns:
            List of IMF components if EWT was applied, None otherwise
        """
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self):
        """
        Get the trend component extracted during detrending

        Returns:
            Trend component if detrending was applied, None otherwise
        """
        return self.trend_component if self.detrend else None


# Example DataLoader for time series
class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset for time series data"""

    def __init__(self, X, y=None):
        """
        Initialize dataset

        Args:
            X: Input sequences of shape [n_sequences, seq_len, n_features]
            y: Target sequences of shape [n_sequences, horizon, n_features]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=32):
    """
    Create PyTorch DataLoaders for training and validation

    Args:
        X_train: Training input sequences
        y_train: Training target sequences
        X_val: Validation input sequences
        y_val: Validation target sequences
        batch_size: Batch size

    Returns:
        train_dataloader: DataLoader for training
        val_dataloader: DataLoader for validation (if validation data provided)
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Create validation dataloader if validation data provided
    if X_val is not None and y_val is not None:
        val_dataset = TimeSeriesDataset(X_val, y_val)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        return train_dataloader, val_dataloader

    return train_dataloader, None
