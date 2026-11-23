"""Connect transformer stacks for autoregressive forecasting."""

from torch import nn

from foreblocks.models.forecasting import ForecastingModel
from foreblocks.nn.transformer.decoder import TransformerDecoder
from foreblocks.nn.transformer.encoder import TransformerEncoder


class EncoderDecoder(ForecastingModel):
    """Encode history once and predict target_len steps through decoder cross-attention."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        *,
        target_len: int,
        label_len: int | None = None,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            forecasting_strategy="seq2seq",
            model_type="transformer",
            target_len=target_len,
            label_len=label_len,
            output_size=decoder.output_size,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        if encoder.d_model != decoder.d_model:
            self._mem_model_bridge = nn.Linear(encoder.d_model, decoder.d_model)

    def _setup_output_layers(self) -> None:
        # TransformerDecoder already projects its hidden states into target space.
        self.output_head = nn.Identity()
        self.project_output = nn.Identity()
