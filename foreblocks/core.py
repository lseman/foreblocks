import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Union


class ForecastingModel(nn.Module):
    """
    Unified forecasting model supporting multiple architectures and strategies.
    Simplified version maintaining all original functionality.
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

        # Setup processing modules
        self.input_preprocessor = input_preprocessor or nn.Identity()
        self.output_postprocessor = output_postprocessor or nn.Identity()
        self.input_normalization = input_normalization or nn.Identity()
        self.output_normalization = output_normalization or nn.Identity()
        self.output_block = output_block or nn.Identity()
        self.input_skip_connection = input_skip_connection

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

    def forward(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Main forward pass - routes to appropriate strategy"""
        # Preprocess input
        processed_src = self._preprocess_input(src)

        # Route to strategy
        if self.strategy == "direct":
            return self._forward_direct(processed_src)
        elif self.strategy == "autoregressive":
            return self._forward_autoregressive(processed_src, targets, epoch)
        elif self.strategy in ["seq2seq", "transformer_seq2seq"]:
            return self._forward_seq2seq_unified(processed_src, targets, epoch)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _preprocess_input(self, src: torch.Tensor) -> torch.Tensor:
        """Apply input preprocessing"""
        processed = self.input_preprocessor(src)
        if self.input_skip_connection:
            processed = processed + src
        return self.input_normalization(processed)

    def _forward_direct(self, src: torch.Tensor) -> torch.Tensor:
        """Direct forecasting - single forward pass"""
        output = self.decoder(src)
        output = self.output_normalization(output)
        return self.output_postprocessor(output)

    def _forward_autoregressive(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Autoregressive forecasting"""
        decoder_input = src[:, -1:, :]  # Start with last timestep
        outputs = []

        for t in range(self.target_len):
            output = self.decoder(decoder_input)
            output = self.output_normalization(output)
            outputs.append(output)

            # Next input: teacher forcing or prediction
            decoder_input = self._get_next_input(output, targets, t, epoch)

        return self.output_postprocessor(torch.cat(outputs, dim=1))

    def _forward_seq2seq_unified(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Unified seq2seq forward for all model types"""
        if self.multi_encoder_decoder:
            return self._forward_multi_encoder_decoder(src, targets, epoch)

        # Route by model type
        if self.model_type == "informer-like":
            return self._forward_informer_style(src, targets, epoch)
        elif self.model_type == "transformer":
            return self._forward_transformer_style(src, targets, epoch)
        else:  # LSTM/RNN style
            return self._forward_rnn_style(src, targets, epoch)

    def _forward_rnn_style(self, src, targets=None, epoch=None):
        """Fixed RNN/LSTM seq2seq with proper output accumulation"""
        batch_size = src.size(0)
        device = src.device
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Process encoder hidden state
        decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
        self._kl = kl_div
        
        # Initialize decoder input
        if hasattr(self, 'init_decoder_input_layer'):
            decoder_input = self.init_decoder_input_layer(
                encoder_outputs[:, -1, :]
            ).unsqueeze(1)  # [B, 1, output_size]
        else:
            decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)
        
        # PRE-ALLOCATE output tensor (like original)
        outputs = torch.zeros(
            batch_size, self.target_len, self.output_size, device=device
        )
        
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
            decoder_output = self.output_block(decoder_output)
            decoder_output = self.output_normalization(decoder_output)
            
            # CRITICAL FIX: Proper assignment to output tensor
            if len(decoder_output.shape) == 2:  # [B, output_size]
                outputs[:, t, :] = decoder_output
            else:  # [B, 1, output_size]
                outputs[:, t, :] = decoder_output.squeeze(1)
            
            # Next input with proper shape handling
            if targets is not None:
                teacher_force_ratio = (
                    self.scheduled_sampling_fn(epoch)
                    if self.scheduled_sampling_fn
                    else self.teacher_forcing_ratio
                )
                use_teacher_forcing = torch.rand(1).item() < teacher_force_ratio
                
                if use_teacher_forcing:
                    decoder_input = targets[:, t:t+1, :]  # [B, 1, input_size]
                else:
                    # Ensure decoder_output has right shape for next input
                    if len(decoder_output.shape) == 2:
                        decoder_input = decoder_output.unsqueeze(1)  # [B, 1, output_size]
                    else:
                        decoder_input = decoder_output  # Already [B, 1, output_size]
            else:
                if len(decoder_output.shape) == 2:
                    decoder_input = decoder_output.unsqueeze(1)
                else:
                    decoder_input = decoder_output
        
        return self.output_postprocessor(outputs)


    def _forward_transformer_style(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Transformer with autoregressive decoding"""
        batch_size = src.size(0)
        device = src.device

        # Encode
        memory = self.encoder(src)

        # Initialize
        next_input = src[:, -self.label_len :, :][:, -1:, :]  # Last timestep
        outputs = []
        incremental_state = None

        # Check teacher forcing
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

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
            outputs.append(pred_t)

            # Next input
            if use_teacher_forcing and targets is not None:
                next_input = targets[:, t : t + 1, :]
            else:
                next_input = pred_t
                # Pad if dimensions don't match
                if self.output_size != self.input_size:
                    pad_size = self.input_size - self.output_size
                    padding = torch.zeros(batch_size, 1, pad_size, device=device)
                    next_input = torch.cat([next_input, padding], dim=-1)

        return torch.cat(outputs, dim=1)

    def _forward_informer_style(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Informer-style parallel decoding"""
        batch_size = src.size(0)

        # Encode
        enc_out = self.encoder(src)

        # Create decoder input (start token repeated)
        start_token = src[:, -1:, :]
        dec_input = start_token.expand(batch_size, self.pred_len, -1)

        # Decode all at once
        out = self.decoder(dec_input, enc_out)
        out = self.output_layer(out)

        return out

    def _forward_multi_encoder_decoder(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Multi encoder-decoder for each input feature"""
        batch_size, seq_len, input_size = src.shape
        feature_outputs = []

        for i in range(input_size):
            # Process each feature separately
            feature_input = src[:, :, i : i + 1]  # [B, T, 1]

            # Use dedicated encoder/decoder
            encoder_outputs, encoder_hidden = self.encoder[i](feature_input)
            decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
            self._kl = kl_div

            # Initialize decoder
            decoder_input = torch.zeros(
                batch_size, 1, self.output_size, device=src.device
            )
            outputs = []

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

                decoder_output = self.output_block(decoder_output)
                decoder_output = self.output_normalization(decoder_output)
                outputs.append(decoder_output)

                decoder_input = self._get_next_input(decoder_output, targets, t, epoch)

            feature_outputs.append(torch.cat(outputs, dim=1))

        # Aggregate features
        stacked = torch.stack(
            feature_outputs, dim=-1
        )  # [B, T, output_size, num_features]
        aggregated = self.decoder_aggregator(stacked).squeeze(-1)  # [B, T, output_size]

        return self.output_postprocessor(aggregated)

    def _get_next_input(
        self,
        current_output: torch.Tensor,
        targets: torch.Tensor = None,
        step: int = 0,
        epoch: int = None,
    ) -> torch.Tensor:
        """Determine next decoder input (teacher forcing or prediction)"""
        if targets is None or not self.training:
            return (
                current_output.unsqueeze(1)
                if current_output.dim() == 2
                else current_output
            )

        # Teacher forcing decision
        if self._should_use_teacher_forcing(targets, epoch):
            return targets[:, step : step + 1, :]
        else:
            return (
                current_output.unsqueeze(1)
                if current_output.dim() == 2
                else current_output
            )

    def _should_use_teacher_forcing(
        self, targets: torch.Tensor = None, epoch: int = None
    ) -> bool:
        """Determine if teacher forcing should be used"""
        if not self.training or targets is None:
            return False

        ratio = (
            self.scheduled_sampling_fn(epoch)
            if self.scheduled_sampling_fn and epoch is not None
            else self.teacher_forcing_ratio
        )
        return torch.rand(1).item() < ratio

    def _process_encoder_hidden(
        self, encoder_hidden
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process encoder hidden state, handling VAE and bidirectional cases"""
        # VAE case: (z, mu, logvar)
        if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
            z, mu, logvar = encoder_hidden
            kl_div = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
            )
            return (z,), kl_div

        # Regular case
        return self._prepare_decoder_hidden(encoder_hidden), None

    def _prepare_decoder_hidden(self, encoder_hidden):
        """Prepare encoder hidden for decoder (handle LSTM/bidirectional)"""
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
        """Combine bidirectional hidden states"""
        if hidden.size(0) % 2 == 0:
            # Reshape and sum forward/backward
            num_layers = hidden.size(0) // 2
            hidden = hidden.view(num_layers, 2, -1, hidden.size(-1)).sum(dim=1)
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

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for transformer decoder"""
        mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
        mask.fill_diagonal_(0)
        return mask
