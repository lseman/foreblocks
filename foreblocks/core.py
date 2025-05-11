# Standard library
import copy

# PyTorch
import torch
import torch.nn as nn

from .aux import *
from .enc_dec import *
from .att import *
import torch
import torch.nn as nn
import copy


class ForecastingModel(nn.Module):
    """
    A flexible sequence-to-sequence forecasting model supporting multiple architectures
    and forecasting strategies including seq2seq, autoregressive, and transformer approaches.
    """
    
    VALID_STRATEGIES = ["seq2seq", "autoregressive", "direct", "transformer_seq2seq"]
    VALID_MODEL_TYPES = ["lstm", "transformer"]
    
    def __init__(
        self,
        encoder=None,
        decoder=None,
        target_len=5,
        forecasting_strategy="seq2seq",
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
        """
        Initialize the ForecastingModel with specified components and parameters.
        
        Args:
            encoder: Neural network module for encoding input sequences
            decoder: Neural network module for decoding and generating predictions
            target_len: Length of prediction sequence to generate
            forecasting_strategy: Strategy for forecasting ("seq2seq", "autoregressive", "direct", "transformer_seq2seq")
            input_preprocessor: Module for preprocessing input data
            output_postprocessor: Module for postprocessing output predictions
            attention_module: Optional attention mechanism
            teacher_forcing_ratio: Ratio for applying teacher forcing during training
            scheduled_sampling_fn: Function that returns teacher forcing ratio based on epoch
            output_size: Size of output features (if None, inferred from decoder)
            output_block: Additional processing block for outputs
            input_normalization: Normalization for inputs
            output_normalization: Normalization for outputs
            model_type: Type of base model architecture ("lstm" or "transformer")
            input_skip_connection: Whether to use skip connection for inputs
            multi_encoder_decoder: Whether to use multiple encoder-decoders
            input_processor_output_size: Output size for input processor with multi encoder-decoder
            hidden_size: Size of hidden representations
            enc_embbedding: Embedding module for encoder (transformer)
            dec_embedding: Embedding module for decoder (transformer)
        """
        super().__init__()
        
        # Validate inputs
        self._validate_initialization(forecasting_strategy, model_type)
        
        # Store all parameters first to ensure they're available to all methods
        self.strategy = forecasting_strategy
        self.model_type = model_type
        self.target_len = target_len
        self.pred_len = self.target_len  # For compatibility
        self.hidden_size = hidden_size
        self.multi_encoder_decoder = multi_encoder_decoder
        self.output_size = output_size   # Store this early to make it available
        
        # Training parameters
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduled_sampling_fn = scheduled_sampling_fn
        
        # Input/output processing
        self._setup_preprocessing_modules(
            input_preprocessor, 
            output_postprocessor, 
            input_normalization, 
            output_normalization, 
            output_block
        )
        self.input_skip_connection = input_skip_connection
        
        # Model architecture setup
        self._setup_encoder_decoder(
            encoder, 
            decoder, 
            multi_encoder_decoder, 
            input_processor_output_size
        )
        
        # Feature dimensions
        self.input_size = encoder.input_size if encoder else None
        # Update output_size if it wasn't provided explicitly
        if self.output_size is None:
            self.output_size = decoder.output_size if decoder else None
        self.label_len = decoder.output_size if decoder else None
        
        # Now that output_size is properly set, update the start_token if it was created
        if hasattr(self, 'start_token') and self.start_token is not None and self.output_size is not None:
            self.start_token = nn.Linear(self.hidden_size, self.output_size)
        
        # Attention setup
        self.use_attention = attention_module is not None
        self.attention_module = attention_module
        
        # Transformer-specific setup
        if model_type == "transformer":
            self._setup_transformer_components(enc_embbedding, dec_embedding)
        
        # Output layers
        self._setup_output_layers()

    def _validate_initialization(self, strategy, model_type):
        """Validate the input parameters."""
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}. Valid options: {self.VALID_STRATEGIES}")
        
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}. Valid options: {self.VALID_MODEL_TYPES}")
    
    def _setup_preprocessing_modules(self, input_preprocessor, output_postprocessor, 
                                    input_normalization, output_normalization, output_block):
        """Setup input/output processing modules."""
        self.input_preprocessor = input_preprocessor or nn.Identity()
        self.output_postprocessor = output_postprocessor or nn.Identity()
        self.input_normalization = input_normalization or nn.Identity()
        self.output_normalization = output_normalization or nn.Identity()
        self.output_block = output_block or nn.Identity()
    
    def _setup_encoder_decoder(self, encoder, decoder, multi_encoder_decoder, input_processor_output_size):
        """Setup encoder and decoder architecture."""
        if multi_encoder_decoder:
            self.encoder = nn.ModuleList([self._clone_module(encoder) for _ in range(input_processor_output_size)])
            self.decoder = nn.ModuleList([self._clone_module(decoder) for _ in range(input_processor_output_size)])
            self.decoder_aggregator = nn.Linear(input_processor_output_size, 1, bias=False)
        else:
            self.encoder = encoder
            self.decoder = decoder
            
        # Start token for sequence generation
        if decoder:
            # Use self.output_size which is set later in __init__
            self.start_token = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.start_token = None
    
    def _setup_transformer_components(self, enc_embedding, dec_embedding):
        """Setup transformer-specific components."""
        self.enc_embedding = enc_embedding or TimeSeriesEncoder(self.encoder.input_size, self.encoder.hidden_size)
        self.dec_embedding = dec_embedding or TimeSeriesEncoder(self.encoder.input_size, self.encoder.hidden_size)
    
    def _setup_output_layers(self):
        """Setup output projection layers."""
        if self.use_attention and hasattr(self, 'decoder') and self.decoder:
            if not isinstance(self.decoder, nn.ModuleList):
                self.output_layer = nn.Linear(self.decoder.output_size + self.encoder.hidden_size, self.output_size)
            else:
                # For multi-encoder-decoder with attention
                self.output_layer = nn.Linear(self.decoder[0].output_size + self.encoder[0].hidden_size, self.output_size)
        elif hasattr(self, 'decoder') and self.decoder and not isinstance(self.decoder, nn.ModuleList):
            self.output_layer = nn.Linear(self.decoder.output_size, self.output_size)
        else:
            self.output_layer = nn.Identity()
            
        if hasattr(self, 'input_size') and self.input_size:
            self.project_output = nn.Linear(self.input_size, self.output_size)
    
    def _clone_module(self, module):
        """Create a deep copy of a module."""
        return copy.deepcopy(module)
    
    def forward(self, src, targets=None, epoch=None):
        """
        Forward pass for the forecasting model.
        
        Args:
            src: Source sequence [batch_size, seq_len, features]
            targets: Target sequence for teacher forcing [batch_size, target_len, features] 
            epoch: Current training epoch (for scheduled sampling)
            
        Returns:
            Forecasted sequence [batch_size, target_len, output_size]
        """
        # Process input
        processed_src = self._preprocess_input(src)
        
        # Call appropriate forward method based on strategy
        if self.strategy == "direct":
            return self._forward_direct(processed_src)
        elif self.strategy in ["seq2seq", "transformer_seq2seq"]:
            if self.model_type == "transformer":
                return self._forward_transformer_seq2seq(processed_src, targets, epoch)
            elif self.multi_encoder_decoder:
                return self._forward_seq2seq_multi(processed_src, targets, epoch)
            else:
                return self._forward_seq2seq(processed_src, targets, epoch)
        elif self.strategy == "autoregressive":
            return self._forward_autoregressive(processed_src, targets, epoch)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
    
    def _preprocess_input(self, src):
        """Apply preprocessing to input sequence."""
        if self.input_skip_connection:
            return self.input_normalization(self.input_preprocessor(src) + src)
        else:
            return self.input_normalization(self.input_preprocessor(src))
    
    def _forward_seq2seq(self, src, targets, epoch):
        """
        Forward pass for standard sequence-to-sequence forecasting.
        
        Args:
            src: Preprocessed source sequence [batch_size, seq_len, features]
            targets: Target sequence for teacher forcing
            epoch: Current epoch for scheduled sampling
            
        Returns:
            Predicted sequence [batch_size, target_len, output_size]
        """
        batch_size, _, _ = src.shape
        device = src.device
        
        # Encode the input sequence
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Ensure proper shape for encoder outputs
        if encoder_outputs.dim() == 3 and encoder_outputs.shape[1] != src.shape[1]:
            encoder_outputs = encoder_outputs.transpose(0, 1)
        
        # Handle VAE latent representations if present
        decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
        self._kl = kl_div  # Store KL divergence for potential VAE loss
        
        # Initialize decoder sequence
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)
        outputs = torch.zeros(batch_size, self.target_len, self.output_size, device=device)
        
        # Generate sequence step by step
        for t in range(self.target_len):
            # Decode one step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Apply attention if configured
            if self.use_attention:
                query = self._get_attention_query(decoder_output, decoder_hidden)
                context, _ = self.attention_module(query, encoder_outputs)
                decoder_output = self.output_layer(torch.cat((decoder_output, context), dim=-1))
            else:
                decoder_output = self.output_layer(decoder_output)
            
            # Apply output transformations
            decoder_output = self.output_block(decoder_output)
            decoder_output = self.output_normalization(decoder_output)
            outputs[:, t:t+1] = decoder_output.unsqueeze(1)
            
            # Determine next input (teacher forcing or own prediction)
            if targets is not None:
                teacher_force_ratio = self.scheduled_sampling_fn(epoch) if self.scheduled_sampling_fn else self.teacher_forcing_ratio
                use_teacher_forcing = torch.rand(1).item() < teacher_force_ratio
                decoder_input = targets[:, t:t+1] if use_teacher_forcing else decoder_output.unsqueeze(1)
            else:
                decoder_input = decoder_output.unsqueeze(1)
                
        return self.output_postprocessor(outputs)
    
    def _forward_seq2seq_multi(self, src, targets, epoch):
        """
        Forward pass for multi-encoder-decoder sequence-to-sequence forecasting.
        
        Args:
            src: Preprocessed source sequence [batch_size, seq_len, features]
            targets: Target sequence for teacher forcing
            epoch: Current epoch for scheduled sampling
            
        Returns:
            Predicted sequence [batch_size, target_len, output_size]
        """
        batch_size, seq_len, input_size = src.shape
        device = src.device
        
        # Each feature gets its own encoder-decoder
        decoder_outputs_list = []
        for i in range(input_size):
            # Extract single feature and process it
            x_i = src[:, :, i].unsqueeze(-1)  # [B, T, 1]
            encoder_i = self.encoder[i]
            decoder_i = self.decoder[i]
            
            # Encode the input
            encoder_outputs, encoder_hidden = encoder_i(x_i)
            
            # Ensure proper shape for encoder outputs
            if encoder_outputs.dim() == 3 and encoder_outputs.shape[1] != seq_len:
                encoder_outputs = encoder_outputs.transpose(0, 1)
            
            # Handle VAE latent representations if present
            decoder_hidden, kl_div = self._process_encoder_hidden(encoder_hidden)
            self._kl = kl_div  # Store KL divergence for potential VAE loss
            
            # Initialize decoder sequence for this feature
            decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)
            feature_outputs = torch.zeros(batch_size, self.target_len, self.output_size, device=device)
            
            # Generate sequence step by step for this feature
            for t in range(self.target_len):
                decoder_output, decoder_hidden = decoder_i(decoder_input, decoder_hidden)
                
                # Apply attention if configured
                if self.use_attention:
                    query = self._get_attention_query(decoder_output, decoder_hidden)
                    context, _ = self.attention_module(query, encoder_outputs)
                    decoder_output = self.output_layer(torch.cat((decoder_output, context), dim=-1))
                else:
                    decoder_output = self.output_layer(decoder_output)
                
                # Apply output transformations
                decoder_output = self.output_block(decoder_output)
                decoder_output = self.output_normalization(decoder_output)
                feature_outputs[:, t:t+1] = decoder_output.unsqueeze(1)
                
                # Determine next input (teacher forcing or own prediction)
                if targets is not None:
                    teacher_force_ratio = self.scheduled_sampling_fn(epoch) if self.scheduled_sampling_fn else self.teacher_forcing_ratio
                    use_teacher_forcing = torch.rand(1).item() < teacher_force_ratio
                    decoder_input = targets[:, t:t+1] if use_teacher_forcing else decoder_output.unsqueeze(1)
                else:
                    decoder_input = decoder_output.unsqueeze(1)
            
            decoder_outputs_list.append(feature_outputs)  # [B, T, output_size]
        
        # Aggregate outputs from all features
        stacked = torch.stack(decoder_outputs_list, dim=0).permute(1, 2, 0, 3).squeeze(3)
        outputs = self.decoder_aggregator(stacked).squeeze(1)
        
        return self.output_postprocessor(outputs)
    
    def _forward_transformer_seq2seq(self, src, targets, epoch):
        """
        Unified transformer forward pass using autoregressive decoding.
        Replaces inputs with ground truth at each step if teacher forcing is enabled.
        """
        batch_size, _, _ = src.shape
        device = src.device

        enc_out = self.enc_embedding(src)
        enc_out = self.encoder(enc_out)

        # Setup decoder input
        x_dec_so_far = src[:, -self.label_len:, :]  # Context

        preds = []
        teacher_forcing = False
        if self.training and targets is not None:
            teacher_force_ratio = self.scheduled_sampling_fn(epoch) if self.scheduled_sampling_fn else self.teacher_forcing_ratio
            teacher_forcing = torch.rand(1).item() < teacher_force_ratio

        for t in range(self.pred_len):
            # Mask for decoder
            tgt_mask = self._generate_square_subsequent_mask(x_dec_so_far.size(1)).to(device)

            # Embed and decode
            dec_embed = self.dec_embedding(x_dec_so_far)
            out = self.decoder(dec_embed, enc_out, tgt_mask=tgt_mask)
            pred_t = self.output_layer(out[:, -1:, :])  # Only use last step
            preds.append(pred_t)

            # Prepare next decoder input
            if teacher_forcing and targets is not None:
                next_input = targets[:, t:t+1, :]
            else:
                next_input = pred_t

            # Pad if needed
            if self.output_size != self.input_size:
                pad_size = self.input_size - self.output_size
                padding = torch.zeros(next_input.size(0), 1, pad_size, device=device)
                next_input = torch.cat([next_input, padding], dim=-1)

            # Append to decoder input
            x_dec_so_far = torch.cat([x_dec_so_far, next_input], dim=1)

        return torch.cat(preds, dim=1)


    def _forward_autoregressive(self, src, targets, epoch):
        """
        Forward pass for autoregressive forecasting.
        
        Args:
            src: Preprocessed source sequence [batch_size, seq_len, features]
            targets: Target sequence for teacher forcing
            epoch: Current epoch for scheduled sampling
            
        Returns:
            Predicted sequence [batch_size, target_len, output_size]
        """
        batch_size, _, _ = src.shape
        decoder_input = src[:, -1:, :]  # Use last time step as initial input
        outputs = []
        
        # Generate sequence autoregressively
        for t in range(self.target_len):
            decoder_output = self.decoder(decoder_input)
            decoder_output = self.output_normalization(decoder_output)
            outputs.append(decoder_output)
            
            # Determine next input (teacher forcing or own prediction)
            if targets is not None:
                teacher_force_ratio = self.scheduled_sampling_fn(epoch) if self.scheduled_sampling_fn else self.teacher_forcing_ratio
                use_teacher_forcing = torch.rand(1).item() < teacher_force_ratio
                decoder_input = targets[:, t:t+1] if use_teacher_forcing else decoder_output
            else:
                decoder_input = decoder_output
        
        return self.output_postprocessor(torch.cat(outputs, dim=1))
    
    def _forward_direct(self, src):
        """
        Forward pass for direct forecasting (single-step prediction).
        
        Args:
            src: Preprocessed source sequence
            
        Returns:
            Predicted sequence
        """
        output = self.decoder(src)
        output = self.output_normalization(output)
        return self.output_postprocessor(output)
    
    def _process_encoder_hidden(self, encoder_hidden):
        """
        Process encoder hidden state, handling VAE style encoders and bidirectional encoders.
        
        Args:
            encoder_hidden: Hidden state from encoder
            
        Returns:
            tuple: (processed_hidden, kl_divergence)
        """
        # Check if this is a VAE style encoder with (z, mu, logvar)
        if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
            z, mu, logvar = encoder_hidden
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
            return (z,), kl_div
        
        # Otherwise, prepare regular hidden state
        return self._prepare_decoder_hidden(encoder_hidden), None
    
    def _prepare_decoder_hidden(self, encoder_hidden):
        """
        Prepare encoder hidden state for decoder, handling LSTM and bidirectional cases.
        
        Args:
            encoder_hidden: Hidden state from encoder
            
        Returns:
            Processed hidden state ready for decoder
        """
        # Handle LSTM case (hidden, cell)
        if isinstance(encoder_hidden, tuple):
            h_n, c_n = encoder_hidden
            if hasattr(self.encoder, "bidirectional") and self.encoder.bidirectional:
                if self.encoder.num_layers != self.decoder.num_layers:
                    h_n = h_n[-self.decoder.num_layers:]
                    c_n = c_n[-self.decoder.num_layers:]
                if h_n.size(0) % 2 == 0:
                    h_n = h_n.view(self.encoder.num_layers, 2, -1, self.encoder.hidden_size).sum(dim=1)
                    c_n = c_n.view(self.encoder.num_layers, 2, -1, self.encoder.hidden_size).sum(dim=1)
                return (h_n, c_n)
            return encoder_hidden
        # Handle GRU/RNN case (just hidden)
        else:
            h_n = encoder_hidden
            if hasattr(self.encoder, "bidirectional") and self.encoder.bidirectional:
                if self.encoder.num_layers != self.decoder.num_layers:
                    h_n = h_n[-self.decoder.num_layers:]
                if h_n.size(0) % 2 == 0:
                    h_n = h_n.view(self.encoder.num_layers, 2, -1, self.encoder.hidden_size).sum(dim=1)
                return h_n
            return encoder_hidden
    
    def _get_attention_query(self, decoder_output, decoder_hidden):
        """
        Extract query vector for attention mechanism.
        
        Args:
            decoder_output: Output from decoder step
            decoder_hidden: Hidden state from decoder
            
        Returns:
            Query vector for attention
        """
        if hasattr(self.decoder, "is_transformer") and self.decoder.is_transformer:
            return decoder_hidden.permute(1, 0, 2)
        else:
            return decoder_hidden[0][-1] if isinstance(decoder_hidden, tuple) else decoder_hidden[-1]
    
    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the transformer decoder to prevent attending to future time steps.
        
        Args:
            sz: Size of the square mask
            
        Returns:
            Causal mask tensor
        """
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        mask.fill_diagonal_(0)
        return mask
    
    def get_kl(self):
        """Get the KL divergence term if using VAE."""
        return getattr(self, "_kl", None)

    def attribute_forward(self, src):
        """
        Captum-compatible forward function.
        Used for computing attribution with respect to the input.
        Args:
            src: Input sequence tensor [B, T, D]
        Returns:
            output: Forecasted sequence [B, T', D']
        """
        #self.eval()
        src = src.requires_grad_()
        out = self.forward(src)
        return out
