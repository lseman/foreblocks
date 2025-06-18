import copy
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class ForecastingModel(nn.Module):
    """
    Unified forecasting model supporting multiple architectures and strategies.
    Simplified version maintaining all functionality with reduced complexity.
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

        # Processing modules
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

        # Store KL divergence for VAE models
        self._kl = None

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
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Main forward pass - routes to appropriate strategy"""
        # Preprocess input
        processed_src = self._preprocess_input(src)

        # Route to strategy
        if self.strategy == "direct":
            return self._forward_direct(processed_src, targets, time_features, epoch)
        elif self.strategy == "autoregressive":
            return self._forward_autoregressive(
                processed_src, targets, time_features, epoch
            )
        elif self.strategy in ["seq2seq", "transformer_seq2seq"]:
            return self._forward_seq2seq(processed_src, targets, time_features, epoch)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _preprocess_input(self, src: torch.Tensor) -> torch.Tensor:
        """Apply input preprocessing with optional skip connection"""
        processed = self.input_preprocessor(src)

        if self.input_skip_connection:
            processed = processed + src

        return self.input_normalization(processed)

    def _forward_direct(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Direct forecasting - single forward pass"""
        output = self.decoder(src)
        output = self.output_normalization(output)
        return self.output_postprocessor(output)

    def _forward_autoregressive(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Autoregressive forecasting"""
        batch_size, seq_len, feature_size = src.shape
        outputs = []
        decoder_input = src[:, -1:, :]  # Start with last timestep

        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            output = self.decoder(decoder_input)
            output = self.output_normalization(output)
            outputs.append(output)

            # Next input: teacher forcing or prediction
            if t < self.target_len - 1:
                if use_teacher_forcing and targets is not None:
                    decoder_input = targets[:, t : t + 1, :]
                else:
                    decoder_input = output

        outputs = torch.cat(outputs, dim=1)
        return self.output_postprocessor(outputs)

    def _forward_seq2seq(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Seq2seq forward for all model types"""
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
        """RNN/LSTM seq2seq"""
        batch_size, seq_len, _ = src.shape

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src)

        # Process encoder hidden state
        decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
        self._kl = kl_div

        # Initialize decoder input
        if hasattr(self, "init_decoder_input_layer"):
            decoder_input = self.init_decoder_input_layer(
                encoder_outputs[:, -1, :]
            ).unsqueeze(1)
        else:
            decoder_input = torch.zeros(
                batch_size, 1, self.output_size, device=src.device
            )

        outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        # Decode step by step
        for t in range(self.target_len):
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

            outputs.append(
                decoder_output.squeeze(1)
                if decoder_output.dim() == 3
                else decoder_output
            )

            # Prepare next input
            if t < self.target_len - 1:
                if use_teacher_forcing and targets is not None:
                    decoder_input = targets[:, t : t + 1, :]
                else:
                    decoder_input = (
                        decoder_output.unsqueeze(1)
                        if decoder_output.dim() == 2
                        else decoder_output
                    )

        outputs = torch.stack(outputs, dim=1)
        return self.output_postprocessor(outputs)

    def _forward_transformer_style(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Transformer with autoregressive decoding"""
        batch_size = src.size(0)
        memory = self.encoder(src)

        # Initialize
        next_input = src[:, -self.label_len :][:, -1:, :]
        outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        # Decode autoregressively
        for t in range(self.pred_len):
            out = self.decoder(next_input, memory)
            pred_t = self.output_layer(out)
            outputs.append(pred_t)

            # Prepare next input
            if t < self.pred_len - 1:
                if use_teacher_forcing and targets is not None:
                    next_input = targets[:, t : t + 1, :]
                else:
                    next_input = pred_t
                    # Handle input/output size mismatch
                    if self.input_size != self.output_size:
                        pad_size = self.input_size - self.output_size
                        padding = torch.zeros(
                            batch_size, 1, pad_size, device=src.device
                        )
                        next_input = torch.cat([next_input, padding], dim=-1)

        return torch.cat(outputs, dim=1)

    def _forward_informer_style(
        self,
        src: torch.Tensor,
        targets: torch.Tensor = None,
        time_features: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """Informer-style parallel decoding"""
        batch_size = src.size(0)

        # Encode
        enc_result = self.encoder(src, time_features=time_features)
        if isinstance(enc_result, tuple):
            enc_out, enc_aux = enc_result
        else:
            enc_out = enc_result

        # Decoder input
        start_token = src[:, -1:, :]
        dec_input = start_token.expand(batch_size, self.pred_len, -1)

        # Decode
        dec_result = self.decoder(dec_input, enc_out)
        if isinstance(dec_result, tuple):
            out, dec_aux = dec_result
        else:
            out = dec_result

        return self.output_layer(out)

    def _forward_multi_encoder_decoder(
        self, src: torch.Tensor, targets: torch.Tensor = None, epoch: int = None
    ) -> torch.Tensor:
        """Multi encoder-decoder processing"""
        batch_size, seq_len, input_size = src.shape
        feature_outputs = []
        use_teacher_forcing = self._should_use_teacher_forcing(targets, epoch)

        for i in range(input_size):
            # Process each feature separately
            feature_input = src[:, :, i : i + 1]

            # Use dedicated encoder/decoder
            encoder_outputs, encoder_hidden = self.encoder[i](feature_input)
            decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
            self._kl = kl_div

            # Initialize decoder
            decoder_input = torch.zeros(
                batch_size, 1, self.output_size, device=src.device
            )
            feature_output = []

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

                feature_output.append(
                    decoder_output.squeeze(1)
                    if decoder_output.dim() == 3
                    else decoder_output
                )

                if t < self.target_len - 1:
                    if use_teacher_forcing and targets is not None:
                        decoder_input = targets[:, t : t + 1, :]
                    else:
                        decoder_input = (
                            decoder_output.unsqueeze(1)
                            if decoder_output.dim() == 2
                            else decoder_output
                        )

            feature_outputs.append(torch.stack(feature_output, dim=1))

        # Aggregate features
        feature_outputs = torch.stack(
            feature_outputs, dim=-1
        )  # [B, T, output_size, num_features]
        aggregated = self.decoder_aggregator(feature_outputs).squeeze(-1)

        return self.output_postprocessor(aggregated)

    def _should_use_teacher_forcing(
        self, targets: torch.Tensor = None, epoch: int = None
    ) -> bool:
        """Determine whether to use teacher forcing"""
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
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return (z,), kl_div

        # Regular case
        return self._prepare_decoder_hidden(encoder_hidden), None

    def _prepare_decoder_hidden(self, encoder_hidden):
        """Prepare encoder hidden for decoder"""
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
            num_layers = hidden.size(0) // 2
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
        return self._kl

    def attribute_forward(
        self,
        src: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        output_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Captum-compatible forward pass for attribution.
        Forces train() mode to allow backward through cuDNN RNNs.
        Disables dropout manually.
        """
        self.train()  # Enable backward for RNN
        self._disable_dropout()

        src = src.requires_grad_()

        out = self.forward(src, targets=targets, time_features=time_features, epoch=epoch)

        if output_idx is not None:
            return out[..., output_idx]

        return out

    def _disable_dropout(self):
        """Disable dropout layers while keeping the model in train() mode for cuDNN backward."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
