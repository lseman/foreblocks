"""Forecasting model construction settings."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_type: str = "lstm"
    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 64
    seq_len: int = 10
    target_len: int = 10
    strategy: str = "seq2seq"
    teacher_forcing_ratio: float = 0.5
    input_processor_output_size: int | None = None
    input_skip_connection: bool = False
    dim_feedforward: int = 512
    multi_encoder_decoder: bool = False
    dropout: float = 0.2
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    latent_size: int | None = 32  # for VAE
    nheads: int = 8
