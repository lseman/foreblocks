import copy
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class ForecastingModel(nn.Module):
    """
    Unified forecasting model supporting multiple architectures and strategies.
    Optimized version maintaining all original functionality with performance improvements.
    """

    VALID_STRATEGIES = ["seq2seq", "autoregressive", "direct", "transformer_seq2seq"]
    VALID_MODEL_TYPES = ["lstm", "transformer", "informer-like"]

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
        # Multi-encoder setup
        multi_encoder_decoder: bool = False,
        input_processor_output_size: int = 16,
        # Training parameters
        teacher_forcing_ratio: float = 0.5,
        scheduled_sampling_fn: Callable = None,
        # Time embeddings
        time_feature_embedding_enc: nn.Module = None,
        time_feature_embedding_dec: nn.Module = None,
    ):
        super().__init__()

        # Validate inputs
        if forecasting_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {forecasting_strategy}. Valid: {self.VALID_STRATEGIES}"
            )
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model type: {model_type}. Valid: {self.VALID_MODEL_TYPES}"
            )

        # Core parameters
        self.strategy = forecasting_strategy
        self.model_type = model_type
        self.target_len = target_len
        self.pred_len = target_len  # Compatibility
        self.hidden_size = hidden_size
        self.multi_encoder_decoder = multi_encoder_decoder

        # Training parameters
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduled_sampling_fn = scheduled_sampling_fn

        # Setup processing modules with optimized identity checks
        self.input_preprocessor = input_preprocessor or nn.Identity()
        self.output_postprocessor = output_postprocessor or nn.Identity()
        self.input_normalization = input_normalization or nn.Identity()
        self.output_normalization = output_normalization or nn.Identity()
        self.output_block = output_block or nn.Identity()
        self.input_skip_connection = input_skip_connection

        # Cache identity module checks for faster runtime
        self._is_input_preprocessor_identity = isinstance(
            self.input_preprocessor, nn.Identity
        )
        self._is_output_postprocessor_identity = isinstance(
            self.output_postprocessor, nn.Identity
        )
        self._is_input_normalization_identity = isinstance(
            self.input_normalization, nn.Identity
        )
        self._is_output_normalization_identity = isinstance(
            self.output_normalization, nn.Identity
        )
        self._is_output_block_identity = isinstance(self.output_block, nn.Identity)

        # Time embeddings
        self.time_feature_embedding_enc = time_feature_embedding_enc
        self.time_feature_embedding_dec = time_feature_embedding_dec

        # Setup encoder/decoder
        self._setup_architecture(encoder, decoder, input_processor_output_size)

        # Infer sizes
        self.input_size = getattr(encoder, "input_size", None) if encoder else None
        self.output_size = (
            output_size or getattr(decoder, "output_size", None) if decoder else None
        )
        self.label_len = getattr(decoder, "output_size", None) if decoder else None

        # Attention
        self.use_attention = attention_module is not None
        self.attention_module = attention_module

        # Setup output layers
        self._setup_output_layers()

        # Initialize decoder input projection
        if encoder and self.output_size:
            encoder_dim = getattr(encoder, "hidden_size", self.hidden_size)
            self.init_decoder_input_layer = nn.Linear(encoder_dim, self.output_size)

        # Optimized caching
        self._cached_masks = {}
        self._cached_zeros = {}
        self._last_batch_size = None
        self._last_device = None

        # Pre-compute strategy routing for efficiency
        self._strategy_fn = self._get_strategy_function()

    def _setup_architecture(self, encoder, decoder, input_processor_output_size):
        """Setup encoder/decoder architecture (single or multi)"""
        if self.multi_encoder_decoder:
            self.encoder = nn.ModuleList(
                [copy.deepcopy(encoder) for _ in range(input_processor_output_size)]
            )
            self.decoder = nn.ModuleList(
                [copy.deepcopy(decoder) for _ in range(input_processor_output_size)]
            )
            self.decoder_aggregator = nn.Linear(
                input_processor_output_size, 1, bias=False
            )
        else:
            self.encoder = encoder
            self.decoder = decoder

    def _setup_output_layers(self):
        """Setup output projection layers"""
        if not self.encoder or not self.decoder:
            self.output_layer = nn.Identity()
            self.project_output = nn.Identity()
            return

        # Handle multi-encoder case
        if isinstance(self.encoder, nn.ModuleList):
            encoder_hidden = getattr(self.encoder[0], "hidden_size", self.hidden_size)
            decoder_output = getattr(self.decoder[0], "output_size", self.output_size)
        else:
            encoder_hidden = getattr(self.encoder, "hidden_size", self.hidden_size)
            decoder_output = getattr(self.decoder, "output_size", self.output_size)

        # Output layer (with or without attention)
        if self.use_attention:
            self.output_layer = nn.Linear(
                decoder_output + encoder_hidden, self.output_size
            )
        else:
            self.output_layer = nn.Linear(decoder_output, self.output_size)

        # Project output if needed
        if self.input_size and self.input_size != self.output_size:
            self.project_output = nn.Linear(self.input_size, self.output_size)
        else:
            self.project_output = nn.Identity()

    def _get_strategy_function(self):
        """Pre-compute strategy function for routing efficiency"""
        if self.strategy == "direct":
            return self._forward_direct
        elif self.strategy == "autoregressive":
            return self._forward_autoregressive
        elif self.strategy in ["seq2seq", "transformer_seq2seq"]:
            return self._forward_seq2seq_unified
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    @lru_cache(maxsize=64)
    def _get_cached_mask(
        self, size: int, device_str: str, dtype_str: str
    ) -> torch.Tensor:
        """Cache causal masks for transformer decoder"""
        device = torch.device(device_str)
        dtype = getattr(torch, dtype_str)
        mask = torch.triu(
            torch.full((size, size), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        mask.fill_diagonal_(0.0)
        return mask

    def _get_cached_zeros(
        self,
        batch_size: int,
        seq_len: int,
        feature_size: int,
        device: torch.device,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """Cache zero tensors to avoid repeated allocation"""
        dtype = dtype or torch.float32
        cache_key = (batch_size, seq_len, feature_size, str(device), str(dtype))

        if cache_key not in self._cached_zeros:
            # Limit cache size to prevent memory leaks
            if len(self._cached_zeros) > 50:
                oldest_keys = list(self._cached_zeros.keys())[:10]
                for k in oldest_keys:
                    del self._cached_zeros[k]

            self._cached_zeros[cache_key] = torch.zeros(
                batch_size, seq_len, feature_size, device=device, dtype=dtype
            )

        return self._cached_zeros[cache_key]

    def forward(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Main forward pass - routes to appropriate strategy"""
        # Update cache info
        self._last_batch_size = src.size(0)
        self._last_device = src.device

        # Preprocess input
        processed_src = self._preprocess_input(src)

        # Route to strategy using pre-computed function
        return self._strategy_fn(processed_src, targets, time_features, epoch)

    def _preprocess_input(self, src: torch.Tensor) -> torch.Tensor:
        """Apply input preprocessing with optional skip connection"""
        # Fast path for all identity modules
        if (
            self._is_input_preprocessor_identity
            and self._is_input_normalization_identity
            and not self.input_skip_connection
        ):
            return src

        processed = (
            self.input_preprocessor(src)
            if not self._is_input_preprocessor_identity
            else src
        )

        if self.input_skip_connection:
            processed = processed + src

        return (
            self.input_normalization(processed)
            if not self._is_input_normalization_identity
            else processed
        )

    def _forward_direct(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Direct forecasting - single forward pass"""
        output = self.decoder(src)

        if not self._is_output_normalization_identity:
            output = self.output_normalization(output)

        return (
            self.output_postprocessor(output)
            if not self._is_output_postprocessor_identity
            else output
        )

    def _forward_autoregressive(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Autoregressive forecasting with pre-allocated output tensor"""
        batch_size, seq_len, feature_size = src.shape
        device = src.device
        dtype = src.dtype

        # Pre-allocate output tensor
        outputs = torch.empty(
            batch_size,
            self.target_len,
            self.output_size,
            device=device,
            dtype=dtype,
        )

        decoder_input = src[:, -1:, :]  # Start with last timestep

        # Pre-compute teacher forcing decision
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            output = self.decoder(decoder_input)

            if not self._is_output_normalization_identity:
                output = self.output_normalization(output)

            outputs[:, t : t + 1, :] = output

            # Next input: teacher forcing or prediction (skip last iteration)
            if t < self.target_len - 1:
                decoder_input = self._get_next_input(
                    output, targets, t, use_teacher_forcing
                )

        return (
            self.output_postprocessor(outputs)
            if not self._is_output_postprocessor_identity
            else outputs
        )

    def _forward_seq2seq_unified(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Unified seq2seq forward for all model types"""
        if self.multi_encoder_decoder:
            return self._forward_multi_encoder_decoder(src, targets, epoch)

        # Route by model type
        if self.model_type == "informer-like":
            return self._forward_informer_style(src, targets, time_features, epoch)
        elif self.model_type == "transformer":
            return self._forward_transformer_style(src, targets, epoch)
        else:  # LSTM/RNN style
            return self._forward_rnn_style(src, targets, epoch)

    def _forward_rnn_style(self, src, targets=None, epoch=None):
        """Optimized RNN/LSTM seq2seq"""
        batch_size, seq_len, _ = src.shape
        device = src.device
        dtype = src.dtype

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src)

        # Process encoder hidden state
        decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
        self._kl = kl_div

        # Initialize decoder input
        if (
            hasattr(self, "init_decoder_input_layer")
            and self.init_decoder_input_layer is not None
        ):
            decoder_input = self.init_decoder_input_layer(
                encoder_outputs[:, -1, :]
            ).unsqueeze(1)
        else:
            decoder_input = self._get_cached_zeros(
                batch_size, 1, self.output_size, device, dtype
            )

        # Pre-allocate output tensor
        outputs = torch.empty(
            batch_size,
            self.target_len,
            self.output_size,
            device=device,
            dtype=dtype,
        )

        # Pre-compute teacher forcing decision
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        # Decode step by step
        for t in range(self.target_len):
            # Decode one step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Apply attention if configured
            if self.use_attention:
                context, _ = self.attention_module(decoder_hidden, encoder_outputs)
                decoder_output = self.output_layer(
                    torch.cat([decoder_output, context], dim=-1)
                )
            else:
                decoder_output = self.output_layer(decoder_output)

            # Post-process
            if not self._is_output_block_identity:
                decoder_output = self.output_block(decoder_output)
            if not self._is_output_normalization_identity:
                decoder_output = self.output_normalization(decoder_output)

            # Store output
            if decoder_output.dim() == 2:
                outputs[:, t, :] = decoder_output
            else:
                outputs[:, t, :] = decoder_output.squeeze(1)

            # Prepare next input (skip last iteration)
            if t < self.target_len - 1:
                decoder_input = self._get_next_input(
                    decoder_output, targets, t, use_teacher_forcing
                )

        return (
            self.output_postprocessor(outputs)
            if not self._is_output_postprocessor_identity
            else outputs
        )

    def _forward_transformer_style(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Optimized transformer with efficient autoregressive decoding"""
        batch_size = src.size(0)
        device = src.device
        dtype = src.dtype

        # Encode
        memory = self.encoder(src)

        # Initialize
        next_input = src[:, -self.label_len :, :][:, -1:, :]
        outputs = torch.empty(
            batch_size, self.pred_len, self.output_size, device=device, dtype=dtype
        )
        incremental_state = None

        # Pre-compute teacher forcing decision and padding info
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)
        padding_needed = self.input_size != self.output_size

        if padding_needed:
            pad_size = self.input_size - self.output_size
            padding_zeros = torch.zeros(
                batch_size, 1, pad_size, device=device, dtype=dtype
            )

        # Decode autoregressively
        for t in range(self.pred_len):
            # Single step decode
            if hasattr(self.decoder, "forward_one_step"):
                out, incremental_state = self.decoder.forward_one_step(
                    tgt=next_input, memory=memory, incremental_state=incremental_state
                )
            else:
                out = self.decoder(next_input, memory)

            pred_t = self.output_layer(out)
            outputs[:, t : t + 1, :] = pred_t

            # Prepare next input (skip last iteration)
            if t < self.pred_len - 1:
                if use_teacher_forcing and targets is not None:
                    next_input = targets[:, t : t + 1, :]
                else:
                    next_input = pred_t
                    if padding_needed:
                        next_input = torch.cat([next_input, padding_zeros], dim=-1)

        return outputs

    def _forward_informer_style(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # noqa: F821
        """
        Optimized Informer-style parallel decoding with optional aux losses.
        Returns:
            output: prediction tensor
            aux_losses: dict containing optional auxiliary loss terms (e.g., encoder, decoder aux loss)
        """
        batch_size = src.size(0)

        # Encode
        enc_result = self.encoder(src, time_features=time_features)
        if isinstance(enc_result, tuple):
            enc_out, enc_aux = enc_result
        else:
            enc_out = enc_result
            enc_aux = {}

        # Decoder input
        start_token = src[:, -1:, :]
        dec_input = start_token.expand(batch_size, self.pred_len, -1)

        # Decode
        dec_result = self.decoder(dec_input, enc_out)
        if isinstance(dec_result, tuple):
            out, dec_aux = dec_result
        else:
            out = dec_result
            dec_aux = {}

        # Merge aux losses
        aux_losses = {}
        if "aux_loss" in enc_aux:
            aux_losses["encoder_aux_loss"] = enc_aux["aux_loss"]
        if "aux_loss" in dec_aux:
            aux_losses["decoder_aux_loss"] = dec_aux["aux_loss"]
        if aux_losses:
            aux_losses["aux_loss"] = sum(aux_losses.values())

        return self.output_layer(out), aux_losses

    def _forward_multi_encoder_decoder(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Optimized multi encoder-decoder"""
        batch_size, seq_len, input_size = src.shape
        device = src.device
        dtype = src.dtype

        # Pre-allocate feature outputs
        feature_outputs = torch.empty(
            input_size,
            batch_size,
            self.target_len,
            self.output_size,
            device=device,
            dtype=dtype,
        )

        # Pre-compute teacher forcing decision
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for i in range(input_size):
            # Process each feature separately
            feature_input = src[:, :, i : i + 1]

            # Use dedicated encoder/decoder
            encoder_outputs, encoder_hidden = self.encoder[i](feature_input)
            decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
            self._kl = kl_div

            # Initialize decoder
            decoder_input = self._get_cached_zeros(
                batch_size, 1, self.output_size, device, dtype
            )

            # Decode for this feature
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder[i](
                    decoder_input, decoder_hidden
                )

                if self.use_attention:
                    query = self._get_attention_query(decoder_output, decoder_hidden)
                    context, _ = self.attention_module(query, encoder_outputs)
                    decoder_output = self.output_layer(
                        torch.cat([decoder_output, context], dim=-1)
                    )
                else:
                    decoder_output = self.output_layer(decoder_output)

                if not self._is_output_block_identity:
                    decoder_output = self.output_block(decoder_output)
                if not self._is_output_normalization_identity:
                    decoder_output = self.output_normalization(decoder_output)

                if decoder_output.dim() == 2:
                    feature_outputs[i, :, t, :] = decoder_output
                else:
                    feature_outputs[i, :, t, :] = decoder_output.squeeze(1)

                if t < self.target_len - 1:  # Skip last iteration
                    decoder_input = self._get_next_input(
                        decoder_output, targets, t, use_teacher_forcing
                    )

        # Aggregate features
        feature_outputs = feature_outputs.permute(
            1, 2, 3, 0
        )  # [B, T, output_size, num_features]
        aggregated = self.decoder_aggregator(feature_outputs).squeeze(-1)

        return (
            self.output_postprocessor(aggregated)
            if not self._is_output_postprocessor_identity
            else aggregated
        )

    def _get_next_input(
        self,
        current_output: torch.Tensor,
        targets: torch.Tensor = None,
        step: int = 0,
        use_teacher_forcing: bool = False,
    ) -> torch.Tensor:
        """Optimized next input determination"""
        if targets is None or not self.training or not use_teacher_forcing:
            return (
                current_output.unsqueeze(1)
                if current_output.dim() == 2
                else current_output
            )
        return targets[:, step : step + 1, :]

    def _should_use_teacher_forcing(
        self, targets: torch.Tensor = None, epoch: int = None
    ) -> bool:
        """Optimized teacher forcing decision"""
        if not self.training or targets is None:
            return False

        ratio = (
            self.scheduled_sampling_fn(epoch)
            if self.scheduled_sampling_fn and epoch is not None
            else self.teacher_forcing_ratio
        )
        return torch.rand(1, device=targets.device).item() < ratio

    def _process_encoder_hidden(
        self, encoder_hidden
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process encoder hidden state, handling VAE and bidirectional cases"""
        # VAE case: (z, mu, logvar)
        if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
            z, mu, logvar = encoder_hidden
            # More efficient KL computation
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return (z,), kl_div

        # Regular case
        return self._prepare_decoder_hidden(encoder_hidden), None

    def _prepare_decoder_hidden(self, encoder_hidden):
        """Optimized preparation of encoder hidden for decoder"""
        if not hasattr(self.encoder, "bidirectional") or not self.encoder.bidirectional:
            return encoder_hidden

        # Handle bidirectional case
        if isinstance(encoder_hidden, tuple):  # LSTM
            h_n, c_n = encoder_hidden
            h_n = self._combine_bidirectional(h_n)
            c_n = self._combine_bidirectional(c_n)
            return (h_n, c_n)
        else:  # GRU/RNN
            return self._combine_bidirectional(encoder_hidden)

    def _combine_bidirectional(self, hidden):
        """Efficiently combine bidirectional hidden states"""
        if hidden.size(0) % 2 == 0:
            num_layers = hidden.size(0) // 2
            # More efficient reshaping and combining
            hidden = hidden.view(num_layers, 2, *hidden.shape[1:]).sum(dim=1)
        return hidden

    def _get_attention_query(self, decoder_output, decoder_hidden):
        """Extract attention query from decoder state"""
        if hasattr(self.decoder, "is_transformer") and self.decoder.is_transformer:
            return decoder_hidden.permute(1, 0, 2)
        else:
            return (
                decoder_hidden[0][-1]
                if isinstance(decoder_hidden, tuple)
                else decoder_hidden[-1]
            )

    # Utility methods
    def get_kl(self) -> Optional[torch.Tensor]:
        """Get KL divergence for VAE loss"""
        return getattr(self, "_kl", None)

    def attribute_forward(self, src: torch.Tensor) -> torch.Tensor:
        """Captum-compatible forward for attribution analysis"""
        src = src.requires_grad_()
        return self.forward(src)

    def clear_cache(self):
        """Clear internal caches to free memory"""
        self._cached_masks.clear()
        self._cached_zeros.clear()

    def __del__(self):
        """Clean up caches on deletion"""
        if hasattr(self, "_cached_masks"):
            self._cached_masks.clear()
        if hasattr(self, "_cached_zeros"):
            self._cached_zeros.clear()
