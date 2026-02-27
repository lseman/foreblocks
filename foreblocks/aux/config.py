from dataclasses import dataclass
from typing import Optional


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
    """Type-safe training configuration with NAS support"""

    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 32
    patience: int = 10
    min_delta: float = 1e-4
    use_amp: bool = True
    gradient_clip_val: Optional[float] = None
    gradient_accumulation_steps: int = 1
    l1_regularization: float = 0.0
    kl_weight: float = 1.0
    scheduler_type: Optional[str] = None
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    min_lr: float = 1e-6
    verbose: bool = True
    log_interval: int = 10
    save_best_model: bool = True
    save_model_path: Optional[str] = None

    # MoE logging toggles
    moe_logging: bool = False
    moe_log_latency: bool = False
    moe_condition_name: Optional[str] = None
    moe_condition_cardinality: Optional[int] = None

    # ── NEW: NAS training toggles ──
    train_nas: bool = False  # Enable two-step NAS optimization
    nas_alpha_lr: float = 3e-4  # Learning rate for α parameters
    nas_alpha_weight_decay: float = 1e-3  # Weight decay for α
    nas_warmup_epochs: int = 5  # Epochs before starting α optimization
    nas_alternate_steps: int = 1  # Steps of α opt per weight opt step
    nas_use_val_for_alpha: bool = True  # Use validation loss for α updates
    nas_discretize_at_end: bool = True  # Discretize alphas after training
    nas_discretize_threshold: float = 0.5  # Threshold for discretization
    nas_log_alphas: bool = True  # Log alpha values during training

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config key '{key}' not found")
