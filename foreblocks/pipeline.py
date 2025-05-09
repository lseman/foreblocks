# Core typing and system
from typing import Optional, Dict, Any

# Torch
import torch

from .preprocessing import TimeSeriesPreprocessor

# Model components (these must be implemented or imported from your package)
from .core import ForecastingModel
from .enc_dec import (
    LSTMEncoder, LSTMDecoder,
    GRUEncoder, GRUDecoder,
    TransformerEncoder, TransformerDecoder,
    VariationalEncoderWrapper, LatentConditionedDecoder,
)
from .att import AttentionLayer

from .utils import Trainer  # Your existing Trainer implementation


class TimeSeriesSeq2Seq:
    """End-to-end pipeline for time series forecasting with Seq2Seq-compatible models."""

    def __init__(
        self,
        model_type="lstm",
        model_params=None,
        training_params=None,
        device="cuda",
        input_preprocessor=None,
        output_postprocessor=None,
        output_block=None,
        input_normalization=None,
        output_normalization=None,
        attention_module=None,
        enc_embedding=None,
        dec_embedding=None,
        scheduled_sampling_fn=None,
        encoder=None,
        decoder=None,
    ):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        self.device = device

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.output_block = output_block
        self.input_normalization = input_normalization
        self.output_normalization = output_normalization
        self.attention_module = attention_module
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.scheduled_sampling_fn = scheduled_sampling_fn
        self.encoder = encoder
        self.decoder = decoder

        self.model = None
        self.trainer = None
        self.history = None

        self._auto_configure()
        self._build_model()

    def _auto_configure(self):
        """Set default parameters if missing."""
        mp = self.model_params
        mp.setdefault("input_size", 1)
        mp.setdefault("output_size", 1)
        mp.setdefault("input_processor_output_size", mp["input_size"])
        mp.setdefault("hidden_size", 64)
        mp.setdefault("target_len", 10)
        mp.setdefault("strategy", "seq2seq")
        mp.setdefault("teacher_forcing_ratio", 0.5)
        mp.setdefault("input_skip_connection", False)
        mp.setdefault("multi_encoder_decoder", False)

    def _build_model(self):
        """Instantiate encoder, decoder, and forecasting model."""
        mp = self.model_params
        hs = mp["hidden_size"]
        os = mp["output_size"]
        isize = mp["input_size"]
        iproj = mp["input_processor_output_size"]

        # Encoder-decoder selection
        registry = {
            "lstm": (LSTMEncoder, LSTMDecoder),
            "gru": (GRUEncoder, GRUDecoder),
            "transformer": (TransformerEncoder, TransformerDecoder),
        }

        if self.model_type in registry:
            Enc, Dec = registry[self.model_type]

            enc_kwargs = dict(
                input_size=iproj,
                hidden_size=hs,
                num_layers=mp.get("num_encoder_layers", 1),
                dropout=mp.get("dropout", 0.2),
            )

            # Only RNNs support bidirectional
            if self.model_type in ["lstm", "gru"]:
                enc_kwargs["bidirectional"] = False

            encoder = Enc(**enc_kwargs)
            decoder = Dec(
                input_size=os,
                hidden_size=hs,
                output_size=os,
                num_layers=mp.get("num_decoder_layers", 1),
                dropout=mp.get("dropout", 0.2),
            )
        elif self.model_type == "vae":
            base_enc = LSTMEncoder(
                input_size=isize,
                hidden_size=hs,
                num_layers=mp.get("num_layers", 1),
                dropout=mp.get("dropout", 0.1),
            )
            encoder = VariationalEncoderWrapper(
                base_encoder=base_enc,
                latent_dim=mp.get("latent_size", 32),
            )
            base_dec = LSTMDecoder(
                input_size=os,
                hidden_size=hs,
                output_size=os,
                num_layers=mp.get("num_layers", 1),
                dropout=mp.get("dropout", 0.1),
            )
            decoder = LatentConditionedDecoder(
                base_decoder=base_dec,
                latent_dim=mp.get("latent_size", 32),
                hidden_size=hs,
            )
        else:
            encoder = self.encoder
            decoder = self.decoder

        # Forecasting model
        self.model = ForecastingModel(
            encoder=encoder,
            decoder=decoder,
            target_len=mp["target_len"],
            forecasting_strategy=mp["strategy"],
            input_preprocessor=self.input_preprocessor,
            output_postprocessor=self.output_postprocessor,
            attention_module=self.attention_module,
            teacher_forcing_ratio=mp["teacher_forcing_ratio"],
            scheduled_sampling_fn=self.scheduled_sampling_fn,
            output_size=mp["output_size"],
            output_block=self.output_block,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            model_type=self.model_type,
            input_skip_connection=mp["input_skip_connection"],
            multi_encoder_decoder=mp["multi_encoder_decoder"],
            input_processor_output_size=iproj,
            hidden_size=hs,
            enc_embbedding=self.enc_embedding,
            dec_embedding=self.dec_embedding,
        )
        self.model.to(self.device)

    def train_model(self, train_loader, val_loader=None, callbacks=None, plot_curves=True, num_epochs=10,
                   early_stopping=None, patience=10):
        self.trainer = Trainer(self.model, config=self.training_params, device=self.device)
        if num_epochs is not None:
            self.trainer.set_config("num_epochs", num_epochs)
        self.history = self.trainer.train(train_loader, val_loader=val_loader, callbacks=callbacks)

        if plot_curves:
            self.trainer.plot_learning_curves()
        return self.history

    def evaluate_model(self, X_val, y_val):
        return self.trainer.metrics(X_val, y_val)
        
    def preprocess(self, X, **preprocessor_kwargs):
        self.input_preprocessor = TimeSeriesPreprocessor(**preprocessor_kwargs)
        return self.input_preprocessor.fit_transform(X)


    def plot_prediction(self, X_val, y_val, full_series=None, offset=0):
        self.trainer.plot_prediction(X_val, y_val, full_series=full_series, offset=offset)
