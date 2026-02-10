# Core typing and system
from typing import Optional, Tuple

# Torch
import torch
from torch import nn

from .aux import (  # Your existing ModelConfig and TrainingConfig implementations
    ModelConfig, TrainingConfig)
# Model components (these must be implemented or imported from your package)
from .blocks.enc_dec import (
    GRUDecoder,
    GRUEncoder,
    LatentConditionedDecoder,
    LSTMDecoder,
    LSTMEncoder,
    VariationalEncoderWrapper,
)
from .core import ForecastingModel
from .preprocessing import TimeSeriesPreprocessor
from .tf.transformer import TransformerDecoder, TransformerEncoder
from .training import Trainer  # Trainer package entrypoint


class TimeSeriesSeq2Seq:
    """
    Comprehensive sequence-to-sequence model for time series forecasting.

    This class provides a unified interface for building, training, and using
    various types of sequence-to-sequence models for time series forecasting,
    including LSTM, GRU, Transformer, and VAE-based models.
    """

    # Dictionary mapping model types to their encoder-decoder classes
    MODEL_TYPES = {
        "lstm": (LSTMEncoder, LSTMDecoder),
        "gru": (GRUEncoder, GRUDecoder),
        "transformer": (TransformerEncoder, TransformerDecoder),
    }

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        device: str = "cuda",
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
        """
        Initialize the TimeSeriesSeq2Seq model.

        Args:
            model_config: Configuration for the model architecture
            training_config: Configuration for training parameters
            device: Device to run the model on ('cuda' or 'cpu')
            input_preprocessor: Module for preprocessing input data
            output_postprocessor: Module for postprocessing output predictions
            output_block: Additional processing block for outputs
            input_normalization: Normalization for inputs
            output_normalization: Normalization for outputs
            attention_module: Attention mechanism for sequence-to-sequence models
            enc_embedding: Embedding module for encoder (transformer)
            dec_embedding: Embedding module for decoder (transformer)
            scheduled_sampling_fn: Function that returns teacher forcing ratio based on epoch
            encoder: Custom encoder (if not using built-in model types)
            decoder: Custom decoder (if not using built-in model types)
        """
        # Initialize configurations
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        # Store processing components
        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.output_block = output_block
        self.input_normalization = input_normalization
        self.output_normalization = output_normalization
        self.attention_module = attention_module
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.scheduled_sampling_fn = scheduled_sampling_fn

        # Store custom encoder/decoder if provided
        self.encoder = encoder
        self.decoder = decoder

        # Initial model state
        self.model = None
        self.trainer = None
        self.history = None

        # Build the model
        self._auto_configure()
        self._build_model()

    def _auto_configure(self) -> None:
        """Ensure derived configuration values are consistent."""
        mc = self.model_config
        if mc.input_processor_output_size is None:
            mc.input_processor_output_size = mc.input_size

    def _build_model(self) -> None:
        """Build the forecasting model based on the configuration."""
        mc = self.model_config

        # Get encoder and decoder classes for the specified model type
        if self.encoder is None or self.decoder is None:
            if mc.model_type == "vae":
                # Special case for VAE models
                encoder, decoder = self._build_vae_components()
            else:
                # Regular sequence-to-sequence models
                encoder, decoder = self._build_seq2seq_components()
        else:
            # Use custom encoder and decoder
            encoder, decoder = self.encoder, self.decoder

        # Create the complete forecasting model
        self.model = ForecastingModel(
            encoder=encoder,
            decoder=decoder,
            target_len=mc.target_len,
            forecasting_strategy=mc.strategy,
            teacher_forcing_ratio=mc.teacher_forcing_ratio,
            input_preprocessor=self.input_preprocessor,
            output_postprocessor=self.output_postprocessor,
            attention_module=self.attention_module,
            scheduled_sampling_fn=self.scheduled_sampling_fn,
            output_size=mc.output_size,
            output_block=self.output_block,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            model_type=mc.model_type,
            input_skip_connection=mc.input_skip_connection,
            multi_encoder_decoder=mc.multi_encoder_decoder,
            input_processor_output_size=mc.input_processor_output_size,
            hidden_size=mc.hidden_size,
            enc_embbedding=self.enc_embedding,
            dec_embedding=self.dec_embedding,
        ).to(self.device)

    def _build_vae_components(self) -> Tuple[nn.Module, nn.Module]:
        """Build encoder and decoder components for VAE models."""
        mc = self.model_config

        # Create base encoder (usually LSTM)
        base_enc = LSTMEncoder(
            mc.input_size, mc.hidden_size, mc.num_encoder_layers, mc.dropout
        )

        # Wrap with variational encoder
        encoder = VariationalEncoderWrapper(
            base_encoder=base_enc, latent_dim=mc.latent_size
        )

        # Create base decoder
        base_dec = LSTMDecoder(
            mc.output_size,
            mc.hidden_size,
            mc.output_size,
            mc.num_decoder_layers,
            mc.dropout,
        )

        # Wrap with latent-conditioned decoder
        decoder = LatentConditionedDecoder(
            base_decoder=base_dec, latent_dim=mc.latent_size, hidden_size=mc.hidden_size
        )

        return encoder, decoder

    def _build_seq2seq_components(
        self,
    ) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
        """Build encoder and decoder components for standard sequence-to-sequence models."""
        mc = self.model_config

        # Get encoder and decoder classes
        EncoderClass, DecoderClass = self.MODEL_TYPES.get(mc.model_type, (None, None))
        if EncoderClass is None or DecoderClass is None:
            return None, None

        # Configure encoder arguments based on model type
        enc_kwargs = {
            "input_size": mc.input_processor_output_size,
            "hidden_size": mc.hidden_size,
            "num_layers": mc.num_encoder_layers,
            "dropout": mc.dropout,
        }

        # Add bidirectional option for RNN-based encoders
        if mc.model_type in ["lstm", "gru"]:
            enc_kwargs["bidirectional"] = False  # Could make configurable

        # Add transformer-specific parameters
        if mc.model_type == "transformer":
            enc_kwargs["nhead"] = mc.nheads
            enc_kwargs["dim_feedforward"] = mc.dim_feedforward

        # Create encoder
        encoder = EncoderClass(**enc_kwargs)

        # Configure decoder arguments based on model type
        dec_kwargs = {
            "input_size": mc.output_size,
            "hidden_size": mc.hidden_size,
            "output_size": mc.output_size,
            "num_layers": mc.num_decoder_layers,
            "dropout": mc.dropout,
        }

        # Add transformer-specific parameters
        if mc.model_type == "transformer":
            dec_kwargs["nhead"] = mc.nheads
            dec_kwargs["dim_feedforward"] = mc.dim_feedforward

        # Create decoder
        decoder = DecoderClass(**dec_kwargs)

        return encoder, decoder

    def train_model(
        self, train_loader, val_loader=None, callbacks=None, plot_curves=True
    ):
        """
        Train the sequence-to-sequence model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            callbacks: List of training callbacks
            plot_curves: Whether to plot learning curves after training

        Returns:
            Training history
        """
        # Initialize trainer with dataclass fields converted to dict
        config_dict = {**vars(self.training_config)}

        # Handle specific parameters from our training config
        config_dict["early_stopping_patience"] = self.training_config.patience
        config_dict["early_stopping_delta"] = self.training_config.min_delta

        self.trainer = Trainer(self.model, config=config_dict, device=self.device)

        # Train the model
        self.history = self.trainer.train(
            train_loader, val_loader=val_loader, callbacks=callbacks
        )

        # Plot learning curves if requested
        if plot_curves:
            self.trainer.plot_learning_curves()

        return self.history

    def evaluate_model(self, X_val, y_val):
        """
        Evaluate the model on validation data.

        Args:
            X_val: Validation inputs
            y_val: Validation targets

        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ValueError(
                "Model must be trained before evaluation. Call train_model first."
            )

        return self.trainer.metrics(X_val, y_val)

    def preprocess(self, X, **preprocessor_kwargs):
        """
        Preprocess input data using TimeSeriesPreprocessor.

        Args:
            X: Input time series data
            **preprocessor_kwargs: Keyword arguments for TimeSeriesPreprocessor

        Returns:
            Preprocessed data
        """
        self.input_preprocessor = TimeSeriesPreprocessor(**preprocessor_kwargs)
        return self.input_preprocessor.fit_transform(X)

    def plot_prediction(self, X_val, y_val, full_series=None, offset=0):
        """
        Plot model predictions against ground truth.

        Args:
            X_val: Validation inputs
            y_val: Validation targets
            full_series: Complete time series (optional)
            offset: Time offset for plotting
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before plotting predictions.")

        self.trainer.plot_prediction(
            X_val, y_val, full_series=full_series, offset=offset
        )

    def predict(self, X):
        """
        Make predictions with the trained model.

        Args:
            X: Input data

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        # Ensure model is in evaluation mode
        self.model.eval()

        # Convert to tensor if necessary
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X = X.to(self.device)

        # Get predictions
        with torch.no_grad():
            predictions = self.model(X)

        # Return as numpy array
        return predictions.cpu().numpy()

    def save_model(self, path):
        """
        Save the model to a file.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "training_config": self.training_config,
            },
            path,
        )

    def load_model(self, path):
        """
        Load a model from a file.

        Args:
            path: Path to load the model from
        """
        # Load saved model
        checkpoint = torch.load(path, map_location=self.device)

        # Update configurations
        self.model_config = checkpoint["model_config"]
        self.training_config = checkpoint["training_config"]

        # Rebuild model with updated config
        self._auto_configure()
        self._build_model()

        # Load state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
