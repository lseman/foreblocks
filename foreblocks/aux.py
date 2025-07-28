from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for the sequence-to-sequence model architecture."""

    model_type: str = "lstm"
    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 64
    seq_len: int = 10
    target_len: int = 10
    strategy: str = "seq2seq"
    teacher_forcing_ratio: float = 0.5
    input_processor_output_size: Optional[int] = None
    input_skip_connection: bool = False
    dim_feedforward: int = 512
    multi_encoder_decoder: bool = False
    dropout: float = 0.2
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    latent_size: Optional[int] = 32  # for VAE
    nheads: int = 8


@dataclass
class TrainingConfig:
    """Configuration for model training parameters."""

    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_delta: float = 1e-4
    use_amp: bool = True
    optimizer: str = "adam"
    criterion: str = "mse"
    batch_size: int = 32
    gradient_clip_val: Optional[float] = None
    lr_scheduler: Optional[str] = None
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    lambda_kl: float = 0.1  # For VAE
