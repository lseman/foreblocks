import copy
from typing import Callable, Dict, Optional, Tuple, Union, List
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.quantize_fx import prepare_qat_fx


class ForecastingModel(nn.Module):
    """
    Unified forecasting model with quantization and knowledge distillation support.
    Supports both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).
    Also supports knowledge distillation for model compression.
    """

    VALID_STRATEGIES = ["seq2seq", "autoregressive", "direct", "transformer_seq2seq"]
    VALID_MODEL_TYPES = ["lstm", "transformer", "informer-like"]
    VALID_QUANTIZATION_MODES = ["none", "ptq", "qat", "dynamic"]
    VALID_DISTILLATION_MODES = ["none", "output", "feature", "attention", "comprehensive"]

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        target_len: int = 5,
        forecasting_strategy: str = "seq2seq",
        model_type: str = "lstm",
        # Processing modules
        input_preprocessor: nn.Module = None,
        output_postprocessor: nn.Module = None,
        input_normalization: nn.Module = None,
        output_normalization: nn.Module = None,
        output_block: nn.Module = None,
        # Architecture options
        attention_module: nn.Module = None,
        output_size: int = None,
        hidden_size: int = 64,
        input_skip_connection: bool = False,
        # Multi-encoder
        multi_encoder_decoder: bool = False,
        input_processor_output_size: int = 16,
        # Training
        teacher_forcing_ratio: float = 0.5,
        scheduled_sampling_fn: Callable = None,
        # Time embeddings
        time_feature_embedding_enc: nn.Module = None,
        time_feature_embedding_dec: nn.Module = None,
        # Quantization options
        quantization_mode: str = "none",
        quantization_backend: str = "fbgemm",  # fbgemm (x86) or qnnpack (ARM)
        bit_width: int = 8,
        enable_fake_quantization: bool = False,
        # Knowledge distillation options
        distillation_mode: str = "none",
        teacher_model: nn.Module = None,
        distillation_temperature: float = 4.0,
        distillation_alpha: float = 0.7,
        feature_distillation_layers: List[str] = None,
        attention_distillation_layers: List[str] = None,
    ):
        super().__init__()

        # Validate inputs
        assert forecasting_strategy in self.VALID_STRATEGIES, f"Invalid strategy: {forecasting_strategy}"
        assert model_type in self.VALID_MODEL_TYPES, f"Invalid model type: {model_type}"
        assert quantization_mode in self.VALID_QUANTIZATION_MODES, f"Invalid quantization mode: {quantization_mode}"
        assert distillation_mode in self.VALID_DISTILLATION_MODES, f"Invalid distillation mode: {distillation_mode}"

        # Core parameters
        self.strategy = forecasting_strategy
        self.model_type = model_type
        self.target_len = target_len
        self.pred_len = target_len  # Compatibility
        self.hidden_size = hidden_size
        self.multi_encoder_decoder = multi_encoder_decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduled_sampling_fn = scheduled_sampling_fn
        self.input_skip_connection = input_skip_connection

        # Quantization parameters
        self.quantization_mode = quantization_mode
        self.quantization_backend = quantization_backend
        self.bit_width = bit_width
        self.enable_fake_quantization = enable_fake_quantization
        self.is_quantized = False

        # Knowledge distillation parameters
        self.distillation_mode = distillation_mode
        self.teacher_model = teacher_model
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha
        self.feature_distillation_layers = feature_distillation_layers or []
        self.attention_distillation_layers = attention_distillation_layers or []
        self.feature_hooks = {}
        self.attention_hooks = {}
        self.teacher_features = {}
        self.teacher_attentions = {}
        self.student_features = {}
        self.student_attentions = {}

        # Processing modules (use Identity as default)
        self.input_preprocessor = input_preprocessor or nn.Identity()
        self.output_postprocessor = output_postprocessor or nn.Identity()
        self.input_normalization = input_normalization or nn.Identity()
        self.output_normalization = output_normalization or nn.Identity()
        self.output_block = output_block or nn.Identity()

        # Time embeddings
        self.time_feature_embedding_enc = time_feature_embedding_enc
        self.time_feature_embedding_dec = time_feature_embedding_dec

        # Quantization stubs (for PTQ/QAT)
        if quantization_mode in ["ptq", "qat"]:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

        # Setup architecture
        self._setup_architecture(encoder, decoder, input_processor_output_size)

        # Sizes
        self.input_size = getattr(encoder, "input_size", None) if encoder else None
        self.output_size = output_size or (getattr(decoder, "output_size", None) if decoder else None)
        self.label_len = getattr(decoder, "output_size", None) if decoder else None

        # Attention
        self.use_attention = attention_module is not None
        self.attention_module = attention_module

        # Setup output layers
        self._setup_output_layers()

        # Decoder input projection
        if encoder and self.output_size:
            encoder_dim = getattr(encoder, "hidden_size", self.hidden_size)
            self.init_decoder_input_layer = nn.Linear(encoder_dim, self.output_size)

        self._kl = None

        # Initialize quantization if needed
        if quantization_mode != "none":
            self._setup_quantization()
        
        # Initialize knowledge distillation if needed
        if distillation_mode != "none":
            self._setup_distillation()

    def _setup_architecture(self, encoder, decoder, input_processor_output_size):
        """Setup encoder/decoder architecture with quantization support"""
        if self.multi_encoder_decoder:
            self.encoder = nn.ModuleList([copy.deepcopy(encoder) for _ in range(input_processor_output_size)])
            self.decoder = nn.ModuleList([copy.deepcopy(decoder) for _ in range(input_processor_output_size)])
            self.decoder_aggregator = nn.Linear(input_processor_output_size, 1, bias=False)
        else:
            self.encoder = encoder
            self.decoder = decoder

    def _setup_output_layers(self):
        """Setup output projection layers"""
        if not self.encoder or not self.decoder:
            self.output_layer = nn.Identity()
            self.project_output = nn.Identity()
            return

        # Get dimensions
        encoder_list = isinstance(self.encoder, nn.ModuleList)
        encoder_hidden = getattr(self.encoder[0] if encoder_list else self.encoder, "hidden_size", self.hidden_size)
        decoder_hidden = getattr(self.decoder[0] if encoder_list else self.decoder, "hidden_size", self.hidden_size)
        decoder_output = getattr(self.decoder[0] if encoder_list else self.decoder, "output_size", self.output_size)

        # Output layer
        output_dim = decoder_output + encoder_hidden if self.use_attention else decoder_output
        self.output_layer = nn.Linear(decoder_hidden * 2, self.output_size)

        # Project output if needed
        self.project_output = (nn.Linear(self.input_size, self.output_size) 
                              if self.input_size and self.input_size != self.output_size 
                              else nn.Identity())

    # ==================== QUANTIZATION METHODS ====================

    def _setup_quantization(self):
        """Setup quantization configuration"""
        if self.quantization_mode == "none":
            return

        # Set quantization backend
        torch.backends.quantized.engine = self.quantization_backend
        
        # Prepare quantization configuration
        if self.quantization_mode in ["ptq", "qat"]:
            # For eager mode quantization, we need to fuse modules first
            self._fuse_modules()

    def _fuse_modules(self):
        """Fuse modules for better quantization performance"""
        # This is a simplified version - you might need to customize based on your specific architecture
        try:
            # Common fusion patterns
            if hasattr(self, 'output_layer') and isinstance(self.output_layer, nn.Linear):
                # Fuse linear layers with activation functions if they exist
                pass
            
            # For LSTM/GRU, fusion is handled internally by PyTorch
            # For transformer layers, we might need custom fusion
            if self.model_type == "transformer":
                self._fuse_transformer_modules()
                
        except Exception as e:
            warnings.warn(f"Module fusion failed: {e}. Proceeding without fusion.")

    def _fuse_transformer_modules(self):
        """Fuse transformer-specific modules"""
        # This would be customized based on your transformer architecture
        pass

    def prepare_for_quantization(self, example_input: torch.Tensor, calibration_data: Optional[torch.utils.data.DataLoader] = None):
        """
        Prepare model for quantization.
        
        Args:
            example_input: Example input tensor for tracing
            calibration_data: DataLoader for calibration (PTQ only)
        """
        if self.quantization_mode == "none":
            return self

        if self.quantization_mode == "dynamic":
            # Dynamic quantization - happens at runtime
            return self._apply_dynamic_quantization()
        
        elif self.quantization_mode == "ptq":
            # Post-training quantization
            return self._apply_post_training_quantization(example_input, calibration_data)
        
        elif self.quantization_mode == "qat":
            # Quantization-aware training
            return self._prepare_quantization_aware_training(example_input)

    def _apply_dynamic_quantization(self):
        """Apply dynamic quantization to linear layers"""
        # Dynamic quantization for linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            self,
            {nn.Linear, nn.LSTM, nn.GRU},  # Quantize these layer types
            dtype=torch.qint8
        )
        quantized_model.is_quantized = True
        return quantized_model

    def _apply_post_training_quantization(self, example_input: torch.Tensor, calibration_data: Optional[torch.utils.data.DataLoader] = None):
        """Apply post-training quantization"""
        # Prepare model for PTQ
        self.eval()
        
        # Get quantization configuration
        qconfig_mapping = get_default_qconfig_mapping(self.quantization_backend)
        
        # Prepare model
        model_prepared = prepare_fx(self, qconfig_mapping, example_input)
        
        # Calibrate if data is provided
        if calibration_data is not None:
            model_prepared.eval()
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    model_prepared(inputs)
        
        # Convert to quantized model
        quantized_model = convert_fx(model_prepared)
        quantized_model.is_quantized = True
        return quantized_model

    def _prepare_quantization_aware_training(self, example_input: torch.Tensor):
        """Prepare model for quantization-aware training"""
        # Get quantization configuration
        qconfig_mapping = get_default_qconfig_mapping(self.quantization_backend)
        
        # Prepare model for QAT
        model_prepared = prepare_qat_fx(self, qconfig_mapping, example_input)
        model_prepared.is_quantized = False  # Not yet quantized, still training
        return model_prepared

    def finalize_quantization(self):
        """Finalize quantization after QAT training"""
        if self.quantization_mode == "qat":
            quantized_model = convert_fx(self)
            quantized_model.is_quantized = True
            return quantized_model
        return self

    # ==================== DISTILLATION METHODS ====================

    def _setup_distillation(self):
        """Setup knowledge distillation configuration"""
        if self.distillation_mode == "none" or self.teacher_model is None:
            return
        
        # Set teacher model to evaluation mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Setup feature distillation hooks
        if self.distillation_mode in ["feature", "comprehensive"]:
            self._setup_feature_hooks()
        
        # Setup attention distillation hooks
        if self.distillation_mode in ["attention", "comprehensive"]:
            self._setup_attention_hooks()
    
    def _setup_feature_hooks(self):
        """Setup hooks for feature distillation"""
        def create_feature_hook(name, storage_dict):
            def hook(module, input, output):
                storage_dict[name] = output
            return hook
        
        # Register hooks for teacher model
        for layer_name in self.feature_distillation_layers:
            if hasattr(self.teacher_model, layer_name):
                layer = getattr(self.teacher_model, layer_name)
                hook = layer.register_forward_hook(
                    create_feature_hook(layer_name, self.teacher_features)
                )
                self.feature_hooks[f"teacher_{layer_name}"] = hook
        
        # Register hooks for student model (self)
        for layer_name in self.feature_distillation_layers:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                hook = layer.register_forward_hook(
                    create_feature_hook(layer_name, self.student_features)
                )
                self.feature_hooks[f"student_{layer_name}"] = hook
    
    def _setup_attention_hooks(self):
        """Setup hooks for attention distillation"""
        def create_attention_hook(name, storage_dict):
            def hook(module, input, output):
                # For attention modules, we typically want the attention weights
                if hasattr(module, 'attention_weights'):
                    storage_dict[name] = module.attention_weights
                else:
                    storage_dict[name] = output
            return hook
        
        # Register hooks for teacher model
        for layer_name in self.attention_distillation_layers:
            if hasattr(self.teacher_model, layer_name):
                layer = getattr(self.teacher_model, layer_name)
                hook = layer.register_forward_hook(
                    create_attention_hook(layer_name, self.teacher_attentions)
                )
                self.attention_hooks[f"teacher_{layer_name}"] = hook
        
        # Register hooks for student model (self)
        for layer_name in self.attention_distillation_layers:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                hook = layer.register_forward_hook(
                    create_attention_hook(layer_name, self.student_attentions)
                )
                self.attention_hooks[f"student_{layer_name}"] = hook

    def compute_distillation_loss(self, student_output: torch.Tensor, teacher_output: torch.Tensor, 
                                 targets: torch.Tensor, base_loss_fn: nn.Module) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute knowledge distillation loss combining multiple distillation strategies.
        
        Args:
            student_output: Student model predictions
            teacher_output: Teacher model predictions
            targets: Ground truth targets
            base_loss_fn: Base loss function (e.g., MSELoss)
        
        Returns:
            total_loss: Combined distillation loss
            loss_components: Dictionary with individual loss components
        """
        loss_components = {}
        
        # Base task loss
        task_loss = base_loss_fn(student_output, targets)
        loss_components['task_loss'] = task_loss
        
        # Output distillation loss
        if self.distillation_mode in ["output", "comprehensive"]:
            output_distill_loss = self._compute_output_distillation_loss(student_output, teacher_output)
            loss_components['output_distillation'] = output_distill_loss
        
        # Feature distillation loss
        if self.distillation_mode in ["feature", "comprehensive"]:
            feature_distill_loss = self._compute_feature_distillation_loss()
            loss_components['feature_distillation'] = feature_distill_loss
        
        # Attention distillation loss
        if self.distillation_mode in ["attention", "comprehensive"]:
            attention_distill_loss = self._compute_attention_distillation_loss()
            loss_components['attention_distillation'] = attention_distill_loss
        
        # Combine losses
        total_loss = self._combine_distillation_losses(loss_components)
        
        return total_loss, loss_components
    
    def _compute_output_distillation_loss(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """Compute output-level distillation loss using KL divergence with temperature scaling"""
        # Apply temperature scaling
        student_soft = F.softmax(student_output / self.distillation_temperature, dim=-1)
        teacher_soft = F.softmax(teacher_output / self.distillation_temperature, dim=-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(
            F.log_softmax(student_output / self.distillation_temperature, dim=-1),
            teacher_soft,
            reduction='batchmean'
        )
        
        # Scale by temperature squared (standard practice)
        return kl_loss * (self.distillation_temperature ** 2)
    
    def _compute_feature_distillation_loss(self) -> torch.Tensor:
        """Compute feature-level distillation loss"""
        feature_loss = 0.0
        num_layers = 0
        
        for layer_name in self.feature_distillation_layers:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                teacher_feat = self.teacher_features[layer_name]
                student_feat = self.student_features[layer_name]
                
                # Align dimensions if necessary
                if teacher_feat.shape != student_feat.shape:
                    student_feat = self._align_feature_dimensions(student_feat, teacher_feat.shape)
                
                # MSE loss between features
                layer_loss = F.mse_loss(student_feat, teacher_feat)
                feature_loss += layer_loss
                num_layers += 1
        
        return feature_loss / max(num_layers, 1)
    
    def _compute_attention_distillation_loss(self) -> torch.Tensor:
        """Compute attention-level distillation loss"""
        attention_loss = 0.0
        num_layers = 0
        
        for layer_name in self.attention_distillation_layers:
            if layer_name in self.teacher_attentions and layer_name in self.student_attentions:
                teacher_att = self.teacher_attentions[layer_name]
                student_att = self.student_attentions[layer_name]
                
                # Align dimensions if necessary
                if teacher_att.shape != student_att.shape:
                    student_att = self._align_attention_dimensions(student_att, teacher_att.shape)
                
                # MSE loss between attention maps
                layer_loss = F.mse_loss(student_att, teacher_att)
                attention_loss += layer_loss
                num_layers += 1
        
        return attention_loss / max(num_layers, 1)
    
    def _align_feature_dimensions(self, student_feat: torch.Tensor, teacher_shape: torch.Size) -> torch.Tensor:
        """Align student feature dimensions to match teacher"""
        if student_feat.shape[-1] != teacher_shape[-1]:
            # Use linear projection to match feature dimensions
            projection = nn.Linear(student_feat.shape[-1], teacher_shape[-1]).to(student_feat.device)
            student_feat = projection(student_feat)
        
        # Handle sequence length differences
        if len(student_feat.shape) > 1 and student_feat.shape[1] != teacher_shape[1]:
            if student_feat.shape[1] < teacher_shape[1]:
                # Pad or interpolate
                student_feat = F.interpolate(
                    student_feat.transpose(1, 2), 
                    size=teacher_shape[1], 
                    mode='linear'
                ).transpose(1, 2)
            else:
                # Truncate
                student_feat = student_feat[:, :teacher_shape[1], :]
        
        return student_feat
    
    def _align_attention_dimensions(self, student_att: torch.Tensor, teacher_shape: torch.Size) -> torch.Tensor:
        """Align student attention dimensions to match teacher"""
        if student_att.shape != teacher_shape:
            # For attention maps, typically we need to handle head differences
            if len(student_att.shape) == 4:  # [batch, heads, seq, seq]
                if student_att.shape[1] != teacher_shape[1]:  # Different number of heads
                    # Average or interpolate heads
                    if student_att.shape[1] < teacher_shape[1]:
                        # Repeat heads
                        repeat_factor = teacher_shape[1] // student_att.shape[1]
                        student_att = student_att.repeat(1, repeat_factor, 1, 1)
                    else:
                        # Average heads
                        group_size = student_att.shape[1] // teacher_shape[1]
                        student_att = student_att.view(
                            student_att.shape[0], teacher_shape[1], group_size, 
                            student_att.shape[2], student_att.shape[3]
                        ).mean(dim=2)
                
                # Handle sequence length differences
                if student_att.shape[2] != teacher_shape[2] or student_att.shape[3] != teacher_shape[3]:
                    student_att = F.interpolate(
                        student_att.view(-1, 1, student_att.shape[2], student_att.shape[3]),
                        size=(teacher_shape[2], teacher_shape[3]),
                        mode='bilinear'
                    ).view(student_att.shape[0], student_att.shape[1], teacher_shape[2], teacher_shape[3])
        
        return student_att
    
    def _combine_distillation_losses(self, loss_components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine different distillation losses with appropriate weights"""
        total_loss = 0.0
        
        # Task loss (ground truth)
        if 'task_loss' in loss_components:
            total_loss += (1.0 - self.distillation_alpha) * loss_components['task_loss']
        
        # Distillation losses
        distillation_loss = 0.0
        num_distill_losses = 0
        
        if 'output_distillation' in loss_components:
            distillation_loss += loss_components['output_distillation']
            num_distill_losses += 1
        
        if 'feature_distillation' in loss_components:
            distillation_loss += 0.5 * loss_components['feature_distillation']  # Lower weight for features
            num_distill_losses += 1
        
        if 'attention_distillation' in loss_components:
            distillation_loss += 0.3 * loss_components['attention_distillation']  # Lower weight for attention
            num_distill_losses += 1
        
        if num_distill_losses > 0:
            total_loss += self.distillation_alpha * (distillation_loss / num_distill_losses)
        
        return total_loss

    def set_teacher_model(self, teacher_model: nn.Module):
        """Set or update the teacher model for distillation"""
        self.teacher_model = teacher_model
        if teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            # Re-setup distillation if mode is not none
            if self.distillation_mode != "none":
                self._setup_distillation()
    
    def enable_distillation(self, mode: str = "output", teacher_model: nn.Module = None):
        """Enable knowledge distillation with specified mode"""
        assert mode in self.VALID_DISTILLATION_MODES, f"Invalid distillation mode: {mode}"
        
        self.distillation_mode = mode
        if teacher_model is not None:
            self.set_teacher_model(teacher_model)
        
        if mode != "none":
            self._setup_distillation()
    
    def disable_distillation(self):
        """Disable knowledge distillation"""
        self.distillation_mode = "none"
        self.teacher_model = None
        
        # Remove hooks
        for hook in self.feature_hooks.values():
            hook.remove()
        for hook in self.attention_hooks.values():
            hook.remove()
        
        self.feature_hooks.clear()
        self.attention_hooks.clear()
    
    def get_distillation_info(self) -> Dict[str, Union[str, int, float]]:
        """Get information about current distillation setup"""
        return {
            "distillation_mode": self.distillation_mode,
            "has_teacher": self.teacher_model is not None,
            "temperature": self.distillation_temperature,
            "alpha": self.distillation_alpha,
            "feature_layers": len(self.feature_distillation_layers),
            "attention_layers": len(self.attention_distillation_layers),
            "active_feature_hooks": len(self.feature_hooks),
            "active_attention_hooks": len(self.attention_hooks)
        }

    # ==================== FORWARD METHODS ====================

    def forward(self, src: torch.Tensor, targets: torch.Tensor = None, 
                time_features: torch.Tensor = None, epoch: int = None, 
                return_teacher_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Main forward pass with quantization and distillation support"""
        # Apply quantization stub if needed
        if self.quantization_mode in ["ptq", "qat"]:
            src = self.quant(src)
        
        # Clear previous feature/attention storages
        self.student_features.clear()
        self.student_attentions.clear()
        
        # Get teacher outputs if in distillation mode
        teacher_output = None
        if self.distillation_mode != "none" and self.teacher_model is not None:
            self.teacher_features.clear()
            self.teacher_attentions.clear()
            with torch.no_grad():
                teacher_output = self.teacher_model(src, targets, time_features, epoch)
        
        processed_src = self._preprocess_input(src)

        # Route to strategy
        strategy_map = {
            "direct": self._forward_direct,
            "autoregressive": self._forward_autoregressive,
            "seq2seq": self._forward_seq2seq,
            "transformer_seq2seq": self._forward_seq2seq
        }
        
        output = strategy_map[self.strategy](processed_src, targets, time_features, epoch)
        
        # Apply dequantization stub if needed
        if self.quantization_mode in ["ptq", "qat"]:
            output = self.dequant(output)
        
        if return_teacher_outputs and teacher_output is not None:
            return output, teacher_output
        return output

    def _preprocess_input(self, src: torch.Tensor) -> torch.Tensor:
        """Apply input preprocessing with optional skip connection"""
        processed = self.input_preprocessor(src)
        if self.input_skip_connection:
            processed = processed + src
        return self.input_normalization(processed)

    def _forward_direct(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Direct forecasting - single forward pass"""
        output = self.decoder(src)
        return self.output_postprocessor(self.output_normalization(output))

    def _forward_autoregressive(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Autoregressive forecasting"""
        outputs = []
        decoder_input = src[:, -1:, :]
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            output = self.output_normalization(self.decoder(decoder_input))
            outputs.append(output)

            if t < self.target_len - 1:
                decoder_input = targets[:, t:t+1, :] if use_teacher_forcing and targets is not None else output

        return self.output_postprocessor(torch.cat(outputs, dim=1))

    def _forward_seq2seq(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Seq2seq forward for all model types"""
        if self.multi_encoder_decoder:
            return self._forward_multi_encoder_decoder(src, targets, epoch)

        # Route by model type
        forward_map = {
            "informer-like": self._forward_informer_style,
            "transformer": self._forward_transformer_style
        }
        
        return forward_map.get(self.model_type, self._forward_rnn_style)(src, targets, time_features, epoch)

    def _forward_rnn_style(self, src, targets=None, time_features=None, epoch=None):
        """RNN/LSTM seq2seq"""
        batch_size = src.size(0)
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
        self._kl = kl_div

        decoder_input = src[:, -1:, :]  # Last timestep as initial input
        outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            #print(f"Decoder output shape at time {t}: {decoder_output.shape}")
            #print(f"Decoder hidden shape at time {t}: {decoder_hidden.shape}")

            # Apply attention if configured
            if self.use_attention:
                context, _ = self.attention_module(decoder_hidden, encoder_outputs)
                #print(f"Context shape at time {t}: {context.shape}")
                decoder_output = self.output_layer(torch.cat([decoder_output, context], dim=-1))

            # Post-process
            decoder_output = self.output_normalization(self.output_block(decoder_output))
            outputs.append(decoder_output.squeeze(1) if decoder_output.dim() == 3 else decoder_output)

            # Next input
            if t < self.target_len - 1:
                if use_teacher_forcing and targets is not None:
                    decoder_input = targets[:, t:t+1, :]
                else:
                    decoder_input = decoder_output.unsqueeze(1) if decoder_output.dim() == 2 else decoder_output

        return self.output_postprocessor(torch.stack(outputs, dim=1))

    def _forward_transformer_style(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Transformer with autoregressive decoding"""
        batch_size = src.size(0)
        memory = self.encoder(src)
        next_input = src[:, -self.label_len:][:, -1:, :]
        outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.pred_len):
            out = self.decoder(next_input, memory)
            pred_t = self.output_layer(out)
            outputs.append(pred_t)

            if t < self.pred_len - 1:
                if use_teacher_forcing and targets is not None:
                    next_input = targets[:, t:t+1, :]
                else:
                    next_input = pred_t
                    # Handle input/output size mismatch
                    if self.input_size != self.output_size:
                        pad_size = self.input_size - self.output_size
                        padding = torch.zeros(batch_size, 1, pad_size, device=src.device)
                        next_input = torch.cat([next_input, padding], dim=-1)

        return torch.cat(outputs, dim=1)

    def _forward_informer_style(self, src: torch.Tensor, targets=None, time_features=None, epoch=None) -> torch.Tensor:
        """Informer-style parallel decoding"""
        batch_size = src.size(0)

        # Encode
        enc_result = self.encoder(src, time_features=time_features)
        enc_out = enc_result[0] if isinstance(enc_result, tuple) else enc_result

        # Decode
        start_token = src[:, -1:, :]
        dec_input = start_token.expand(batch_size, self.pred_len, -1)
        
        dec_result = self.decoder(dec_input, enc_out)
        out = dec_result[0] if isinstance(dec_result, tuple) else dec_result

        return self.output_layer(out)

    def _forward_multi_encoder_decoder(self, src: torch.Tensor, targets=None, epoch=None) -> torch.Tensor:
        """Multi encoder-decoder processing"""
        batch_size, seq_len, input_size = src.shape
        feature_outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for i in range(input_size):
            feature_input = src[:, :, i:i+1]
            encoder_outputs, encoder_hidden = self.encoder[i](feature_input)
            decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
            self._kl = kl_div

            decoder_input = torch.zeros(batch_size, 1, self.output_size, device=src.device)
            feature_output = []

            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder[i](decoder_input, decoder_hidden)

                if self.use_attention:
                    query = self._get_attention_query(decoder_output, decoder_hidden)
                    context, _ = self.attention_module(query, encoder_outputs)
                    decoder_output = self.output_layer(torch.cat([decoder_output, context], dim=-1))
                else:
                    decoder_output = self.output_layer(decoder_output)

                decoder_output = self.output_normalization(self.output_block(decoder_output))
                feature_output.append(decoder_output.squeeze(1) if decoder_output.dim() == 3 else decoder_output)

                if t < self.target_len - 1:
                    if use_teacher_forcing and targets is not None:
                        decoder_input = targets[:, t:t+1, :]
                    else:
                        decoder_input = decoder_output.unsqueeze(1) if decoder_output.dim() == 2 else decoder_output

            feature_outputs.append(torch.stack(feature_output, dim=1))

        # Aggregate features
        feature_outputs = torch.stack(feature_outputs, dim=-1)
        aggregated = self.decoder_aggregator(feature_outputs).squeeze(-1)
        return self.output_postprocessor(aggregated)

    def _should_use_teacher_forcing(self, targets=None, epoch=None, fallback_device="cpu") -> bool:
        """Determine whether to use teacher forcing safely during FX tracing"""
        # If in eval mode or no targets, disable teacher forcing
        if (not self.training) or (targets is None):
            return False

        # Get scheduled sampling ratio
        ratio = (
            self.scheduled_sampling_fn(epoch)
            if self.scheduled_sampling_fn and epoch is not None
            else self.teacher_forcing_ratio
        )

        # Get a safe device for torch.rand (use targets.device if available)
        device = getattr(targets, "device", torch.device(fallback_device))

        # Important: FX tracing cannot handle torch.rand with dynamic device, so fallback to CPU
        if torch.fx._symbolic_trace.is_fx_tracing():
            return False  # Disable random during tracing to keep deterministic

        return torch.rand((1,), device=device).item() < ratio

    def _process_encoder_hidden(self, encoder_hidden) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process encoder hidden state, handling VAE and bidirectional cases"""
        # VAE case: (z, mu, logvar)
        if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
            z, mu, logvar = encoder_hidden
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return (z,), kl_div
        
        return self._prepare_decoder_hidden(encoder_hidden), None

    def _prepare_decoder_hidden(self, encoder_hidden):
        """Prepare encoder hidden for decoder"""
        if not getattr(self.encoder, "bidirectional", False):
            return encoder_hidden

        # Handle bidirectional case
        if isinstance(encoder_hidden, tuple):  # LSTM
            h_n, c_n = encoder_hidden
            return (self._combine_bidirectional(h_n), self._combine_bidirectional(c_n))
        else:  # GRU/RNN
            return self._combine_bidirectional(encoder_hidden)

    def _combine_bidirectional(self, hidden):
        """Combine bidirectional hidden states"""
        # Only combine if we actually have bidirectional layers
        # For bidirectional RNNs, hidden.size(0) = num_layers * 2
        num_layers = hidden.size(0) // 2
        hidden = hidden.view(num_layers, 2, *hidden.shape[1:]).sum(dim=1)
        return hidden

    def _get_attention_query(self, decoder_output, decoder_hidden):
        """Extract attention query from decoder state"""
        if getattr(self.decoder, "is_transformer", False):
            return decoder_hidden.permute(1, 0, 2)
        else:
            return decoder_hidden[0][-1] if isinstance(decoder_hidden, tuple) else decoder_hidden[-1]

    # ==================== UTILITY METHODS ====================

    def get_kl(self) -> Optional[torch.Tensor]:
        """Get KL divergence for VAE loss"""
        return self._kl

    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Get model size information"""
        param_size = sum(p.numel() for p in self.parameters())
        buffer_size = sum(b.numel() for b in self.buffers())
        
        # Estimate memory usage (in MB)
        if self.is_quantized:
            # INT8 quantized model
            size_mb = (param_size + buffer_size) * 1 / (1024 * 1024)  # 1 byte per param
        else:
            # FP32 model
            size_mb = (param_size + buffer_size) * 4 / (1024 * 1024)  # 4 bytes per param
        
        return {
            "parameters": param_size,
            "buffers": buffer_size,
            "total_elements": param_size + buffer_size,
            "size_mb": size_mb,
            "is_quantized": self.is_quantized
        }

    def benchmark_inference(self, input_tensor: torch.Tensor, num_runs: int = 100, warmup_runs: int = 10):
        """Benchmark inference speed"""
        import time
        
        self.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_samples_per_sec": 1.0 / avg_time,
            "quantization_mode": self.quantization_mode,
            "is_quantized": self.is_quantized
        }

    def attribute_forward(self, src: torch.Tensor, time_features: Optional[torch.Tensor] = None,
                         targets: Optional[torch.Tensor] = None, epoch: Optional[int] = None,
                         output_idx: Optional[int] = None) -> torch.Tensor:
        """Captum-compatible forward pass for attribution"""
        self.train()  # Enable backward for RNN
        self._disable_dropout()
        
        src = src.requires_grad_()
        out = self.forward(src, targets=targets, time_features=time_features, epoch=epoch)
        
        return out[..., output_idx] if output_idx is not None else out

    def _disable_dropout(self):
        """Disable dropout layers while keeping model in train() mode"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0