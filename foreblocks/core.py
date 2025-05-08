# Standard library
import copy

# PyTorch
import torch
import torch.nn as nn

from .aux import *
from .enc_dec import *
from .att import *

class ForecastingModel(nn.Module):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        target_len=5,
        forecasting_strategy="seq2seq",  # seq2seq | autoregressive | direct
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
        super(ForecastingModel, self).__init__()

        if multi_encoder_decoder:
            assert (
                input_processor_output_size is not None
            ), "input_size must be provided for multi_encoder_decoder"
            assert (
                encoder is not None and decoder is not None
            ), "You must provide a base encoder and decoder module"

            # Automatically clone encoder/decoder for each input feature
            self.encoder = nn.ModuleList(
                [
                    self._clone_module(encoder)
                    for _ in range(input_processor_output_size)
                ]
            )
            self.decoder = nn.ModuleList(
                [
                    self._clone_module(decoder)
                    for _ in range(input_processor_output_size)
                ]
            )
            self.decoder_aggregator = nn.Linear(
                input_processor_output_size, 1, bias=False
            )  # input_size = num decoders

        else:
            self.encoder = encoder
            self.decoder = decoder

        self.multi_encoder_decoder = multi_encoder_decoder
        self.hidden_size = hidden_size
        self.strategy = forecasting_strategy
        self.target_len = target_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduled_sampling_fn = scheduled_sampling_fn
        self.model_type = model_type

        self.attention_module = attention_module
        self.use_attention = self.attention_module is not None

        self.output_size = output_size or (decoder.output_size if decoder else None)
        self.pred_len = output_size or target_len
        self.output_block = output_block or nn.Identity()

        self.input_preprocessor = input_preprocessor or nn.Identity()
        self.output_postprocessor = output_postprocessor or nn.Identity()
        self.input_normalization = input_normalization or nn.Identity()
        self.output_normalization = output_normalization or nn.Identity()
        self.start_token = (
            nn.Linear(self.hidden_size, self.output_size) if decoder else None
        )
        self.time_features_dim = 1  # Default to 1, can be changed based on your needs
        self.label_len = self.decoder.output_size if decoder else None

        self.input_skip_connection = input_skip_connection

        if model_type == "transformer":
            if enc_embbedding is not None:
                self.enc_embedding = enc_embbedding
            else:
                self.enc_embedding = TimeSeriesEncoder(encoder.input_size, encoder.hidden_size)
            if dec_embedding is not None:
                self.dec_embedding = dec_embedding
            else:
                self.dec_embedding = TimeSeriesEncoder(encoder.input_size, encoder.hidden_size)

        if self.use_attention:
            self.output_layer = nn.Linear(
                decoder.output_size + encoder.hidden_size, self.output_size
            )
        elif decoder:
            self.output_layer = nn.Linear(decoder.output_size, self.output_size)
        else:
            self.output_layer = nn.Identity()

        self.input_size = encoder.input_size

        self.project_output = nn.Linear(
            self.input_size, self.output_size
        )


    def _clone_module(self, module):
        return copy.deepcopy(module)

    def forward(self, src, targets=None, epoch=None):
        if self.strategy == "direct":
            # Direct models like Informer take a tuple of inputs
            return self._forward_direct(src)

        # seq2seq or autoregressive: src is a tensor
        if self.input_preprocessor is not None:
            if self.input_skip_connection:
                src = self.input_preprocessor(src) + src
            else:
                src = self.input_preprocessor(src)

        src = self.input_normalization(src)

        if self.strategy == "seq2seq":
            if self.model_type == "transformer":
                return self._forward_transformer_seq2seq(src, targets, epoch)
            if self.multi_encoder_decoder:
                return self._forward_seq2seq_multi(src, targets, epoch)  # NEW
            return self._forward_seq2seq(src, targets, epoch)

        elif self.strategy == "transformer_seq2seq":
            return self._forward_transformer_seq2seq(src, targets, epoch)
        elif self.strategy == "autoregressive":
            return self._forward_autoregressive(src, targets, epoch)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def _forward_seq2seq(self, src, targets, epoch):
        batch_size, _, _ = src.shape
        device = src.device

        encoder_outputs, encoder_hidden = self.encoder(src)

        if encoder_outputs.dim() == 3 and encoder_outputs.shape[1] != src.shape[1]:
            encoder_outputs = encoder_outputs.transpose(0, 1)

        # Handle VAE latent outputs
        if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
            z, mu, logvar = encoder_hidden
            decoder_hidden = (z,)
            self._kl = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
            )
        else:
            decoder_hidden = self._prepare_decoder_hidden(encoder_hidden)
            self._kl = None

        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)
        outputs = torch.zeros(
            batch_size, self.target_len, self.output_size, device=device
        )

        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            if self.use_attention:
                query = self._get_attention_query(decoder_output, decoder_hidden)
                context, _ = self.attention_module(query, encoder_outputs)
                decoder_output = self.output_layer(
                    torch.cat((decoder_output, context), dim=-1)
                )
            else:
                decoder_output = self.output_layer(decoder_output)

            decoder_output = self.output_block(decoder_output)
            decoder_output = self.output_normalization(decoder_output)
            outputs[:, t : t + 1] = decoder_output.unsqueeze(1)

            if targets is not None:
                use_tf = torch.rand(1).item() < (
                    self.scheduled_sampling_fn(epoch)
                    if self.scheduled_sampling_fn
                    else self.teacher_forcing_ratio
                )
                decoder_input = (
                    targets[:, t : t + 1] if use_tf else decoder_output.unsqueeze(1)
                )
            else:
                decoder_input = decoder_output.unsqueeze(1)

        return self.output_postprocessor(outputs)

    def _forward_seq2seq_multi(self, src, targets, epoch):
        batch_size, seq_len, input_size = src.shape
        device = src.device

        decoder_outputs_list = []

        for i in range(input_size):
            x_i = src[:, :, i].unsqueeze(-1)  # [B, T, 1]
            encoder_i = self.encoder[i]
            decoder_i = self.decoder[i]

            encoder_outputs, encoder_hidden = encoder_i(x_i)

            # Align encoder_outputs if needed
            if encoder_outputs.dim() == 3 and encoder_outputs.shape[1] != seq_len:
                encoder_outputs = encoder_outputs.transpose(0, 1)

            # Handle VAE-style encoder
            if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
                z, mu, logvar = encoder_hidden
                decoder_hidden = (z,)
                self._kl = (
                    -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
                )
            else:
                decoder_hidden = self._prepare_decoder_hidden(encoder_hidden)
                self._kl = None

            # Decoder outputs for this feature
            decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)
            feature_outputs = torch.zeros(
                batch_size, self.target_len, self.output_size, device=device
            )

            for t in range(self.target_len):
                decoder_output, decoder_hidden = decoder_i(
                    decoder_input, decoder_hidden
                )

                if self.use_attention:
                    query = self._get_attention_query(decoder_output, decoder_hidden)
                    context, _ = self.attention_module(query, encoder_outputs)
                    decoder_output = self.output_layer(
                        torch.cat((decoder_output, context), dim=-1)
                    )
                else:
                    decoder_output = self.output_layer(decoder_output)

                decoder_output = self.output_block(decoder_output)
                decoder_output = self.output_normalization(decoder_output)

                feature_outputs[:, t : t + 1] = decoder_output.unsqueeze(1)

                if targets is not None:
                    use_tf = torch.rand(1).item() < (
                        self.scheduled_sampling_fn(epoch)
                        if self.scheduled_sampling_fn
                        else self.teacher_forcing_ratio
                    )
                    decoder_input = (
                        targets[:, t : t + 1] if use_tf else decoder_output.unsqueeze(1)
                    )
                else:
                    decoder_input = decoder_output.unsqueeze(1)

            decoder_outputs_list.append(feature_outputs)  # shape [B, T, output_size]

        # Aggregate across features
        # Stack: [num_decoders, B, T, output_size] → [B, T, num_decoders, output_size]
        stacked = (
            torch.stack(decoder_outputs_list, dim=0).permute(1, 2, 0, 3).squeeze(3)
        )
        # print(f"Stacked shape: {stacked.shape}")

        # Apply linear aggregation over decoder dimension → [B, T, 1, output_size]
        outputs = self.decoder_aggregator(stacked).squeeze(
            1
        )  # shape [B, T, 1, output_size]
        # print(f"Aggregated shape: {outputs.shape}")

        return self.output_postprocessor(outputs)

    def _forward_transformer_seq2seq(self, src, targets, epoch):
        batch_size, src_seq_len, _ = src.shape
        device = src.device

        # Encode the source
        enc_out = self.enc_embedding(src)
        enc_out = self.encoder(enc_out)

        if self.training and targets is not None:
            # === Scheduled Sampling: Use TF with probability
            use_tf = torch.rand(1).item() < (
                self.scheduled_sampling_fn(epoch)
                if self.scheduled_sampling_fn
                else self.teacher_forcing_ratio
            )
            if use_tf:
                # === Teacher Forcing ===
                # Pad targets if output_size < input_size
                if self.output_size != self.input_size:
                    pad_size = self.input_size - self.output_size
                    padding = torch.zeros(targets.size(0), targets.size(1), pad_size, device=targets.device)
                    targets_padded = torch.cat([targets, padding], dim=-1)
                else:
                    targets_padded = targets

                x_dec = torch.cat([src[:, -self.label_len:, :], targets_padded], dim=1)

                tgt_mask = self._generate_square_subsequent_mask(x_dec.size(1)).to(device)
                dec_out = self.dec_embedding(x_dec)
                output = self.decoder(dec_out, enc_out, tgt_mask=tgt_mask)
                output = self.output_layer(output)
                return output[:, -self.pred_len:, :]

            else:
                # === Autoregressive decoding (no teacher forcing)
                preds = []
                x_dec_so_far = src[:, -self.label_len:, :]

                for step in range(self.pred_len):
                    tgt_mask = self._generate_square_subsequent_mask(x_dec_so_far.size(1)).to(device)

                    dec_embed = self.dec_embedding(x_dec_so_far)
                    out = self.decoder(dec_embed, enc_out, tgt_mask=tgt_mask)
                    pred_t = out[:, -1:, :]
                    pred_t = self.output_layer(pred_t)  # shape: [B, 1, output_size]
                    preds.append(pred_t)

                    # === Pad pred_t to match input_size ===
                    if self.output_size != self.input_size:
                        pad_size = self.input_size - self.output_size
                        padding = torch.zeros(pred_t.size(0), 1, pad_size, device=pred_t.device)
                        pred_t_padded = torch.cat([pred_t, padding], dim=-1)
                    else:
                        pred_t_padded = pred_t

                    x_dec_so_far = torch.cat([x_dec_so_far, pred_t_padded], dim=1)

                return torch.cat(preds, dim=1)


        else:
            # Inference
            preds = []
            x_dec_so_far = src[:, -self.label_len:, :]  # [B, label_len, input_size]

            for step in range(self.pred_len):
                tgt_mask = self._generate_square_subsequent_mask(x_dec_so_far.size(1)).to(device)
                dec_embed = self.dec_embedding(x_dec_so_far)  # [B, T, d_model]
                out = self.decoder(dec_embed, enc_out, tgt_mask=tgt_mask)
                pred_t = out[:, -1:, :]  # [B, 1, d_model]
                pred_t = self.output_layer(pred_t)  # [B, 1, output_size]
                preds.append(pred_t)

                # Pad pred_t to match input_size
                if self.output_size != self.input_size:
                    pad_size = self.input_size - self.output_size
                    padding = torch.zeros(pred_t.size(0), 1, pad_size, device=pred_t.device)
                    pred_t_padded = torch.cat([pred_t, padding], dim=-1)
                else:
                    pred_t_padded = pred_t

                x_dec_so_far = torch.cat([x_dec_so_far, pred_t_padded], dim=1)

            return torch.cat(preds, dim=1)


    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the decoder to prevent attending to future time steps.
        """
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        mask.fill_diagonal_(0)
        return mask


    def _generate_time_features(self, seq_len, device):
        """
        Generate simple positional features for time series.
        For more complex time features, you could incorporate day of week,
        hour of day, etc. if that information is available.
        """
        batch_size = 1  # Will be broadcast to all batches

        # Create simple positional features
        pos = torch.arange(0, seq_len, device=device).unsqueeze(0).unsqueeze(-1).float()
        pos = pos / seq_len  # Normalize to [0, 1]

        # Option: Create more complex features for the time dimension
        # For example, sine and cosine features like in positional encoding
        time_features = torch.zeros(
            batch_size, seq_len, self.time_features_dim, device=device
        )

        # Simple increment
        time_features[:, :, 0] = pos.squeeze(-1)

        # Sine features at different frequencies (similar to positional encoding)
        if self.time_features_dim > 1:
            for i in range(1, self.time_features_dim):
                if i % 2 == 1:
                    # Sine with different frequencies
                    time_features[:, :, i] = torch.sin(pos * (2 ** (i // 2)))
                else:
                    # Cosine with different frequencies
                    time_features[:, :, i] = torch.cos(pos * (2 ** (i // 2 - 1)))

        return time_features

    def _forward_autoregressive(self, src, targets, epoch):
        batch_size, _, _ = src.shape
        decoder_input = src[:, -1:, :]  # last time step
        outputs = []

        for t in range(self.target_len):
            decoder_output = self.decoder(decoder_input)
            decoder_output = self.output_normalization(decoder_output)
            outputs.append(decoder_output)

            if targets is not None:
                use_tf = torch.rand(1).item() < (
                    self.scheduled_sampling_fn(epoch)
                    if self.scheduled_sampling_fn
                    else self.teacher_forcing_ratio
                )
                decoder_input = targets[:, t : t + 1] if use_tf else decoder_output
            else:
                decoder_input = decoder_output

        return self.output_postprocessor(torch.cat(outputs, dim=1))

    def _forward_direct(self, src):
        output = self.decoder(src)
        output = self.output_normalization(output)
        return self.output_postprocessor(output)

    def _prepare_decoder_hidden(self, encoder_hidden):
        if isinstance(encoder_hidden, tuple):
            h_n, c_n = encoder_hidden
            if hasattr(self.encoder, "bidirectional") and self.encoder.bidirectional:
                if self.encoder.num_layers != self.decoder.num_layers:
                    h_n = h_n[-self.decoder.num_layers :]
                    c_n = c_n[-self.decoder.num_layers :]
                if h_n.size(0) % 2 == 0:
                    h_n = h_n.view(
                        self.encoder.num_layers, 2, -1, self.encoder.hidden_size
                    ).sum(dim=1)
                    c_n = c_n.view(
                        self.encoder.num_layers, 2, -1, self.encoder.hidden_size
                    ).sum(dim=1)
                return (h_n, c_n)
            return encoder_hidden
        else:
            h_n = encoder_hidden
            if hasattr(self.encoder, "bidirectional") and self.encoder.bidirectional:
                if self.encoder.num_layers != self.decoder.num_layers:
                    h_n = h_n[-self.decoder.num_layers :]
                if h_n.size(0) % 2 == 0:
                    h_n = h_n.view(
                        self.encoder.num_layers, 2, -1, self.encoder.hidden_size
                    ).sum(dim=1)
                return h_n
            return encoder_hidden

    def _get_attention_query(self, decoder_output, decoder_hidden):
        if hasattr(self.decoder, "is_transformer") and self.decoder.is_transformer:
            return decoder_hidden.permute(1, 0, 2)
        else:
            return (
                decoder_hidden[0][-1]
                if isinstance(decoder_hidden, tuple)
                else decoder_hidden[-1]
            )

    def get_kl(self):
        return getattr(self, "_kl", None)
