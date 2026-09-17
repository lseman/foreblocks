import torch
import torch.nn as nn

from foreblocks.core.model import ForecastingModel
from foreblocks.models.transformer.runtime.outputs import (
    TransformerDecoderOutput,
    TransformerEncoderOutput,
)


def test_direct_strategy_reshapes_flat_head_output() -> None:
    model = ForecastingModel(
        head=nn.Sequential(nn.Flatten(), nn.Linear(12, 6)),
        forecasting_strategy="direct",
        model_type="head_only",
        target_len=3,
        output_size=2,
    )

    out = model(torch.randn(4, 4, 3))

    assert out.shape == (4, 3, 2)


def test_benchmark_inference_supports_parameterless_model() -> None:
    model = ForecastingModel(
        head=nn.Identity(),
        forecasting_strategy="direct",
        model_type="head_only",
        target_len=2,
    )

    result = model.benchmark_inference(
        torch.randn(2, 5, 3),
        num_runs=1,
        warmup_runs=0,
    )

    assert result["avg_inference_time_ms"] >= 0.0
    assert result["device"] == "cpu"


def test_attribute_forward_restores_dropout_probability_and_mode() -> None:
    dropout = nn.Dropout(p=0.7)
    model = ForecastingModel(
        head=nn.Sequential(dropout, nn.Identity()),
        forecasting_strategy="direct",
        model_type="head_only",
        target_len=2,
    )
    model.eval()

    out = model.attribute_forward(torch.randn(2, 5, 3))

    assert out.requires_grad
    assert not model.training
    assert dropout.p == 0.7


class _InformerEncoder(nn.Module):
    input_size = 3
    hidden_size = 5

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 5)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.proj(src)


class _InformerDecoder(nn.Module):
    input_size = 3
    hidden_size = 5
    d_model = 5

    def forward(self, dec_input: torch.Tensor, memory: torch.Tensor, **_) -> torch.Tensor:
        steps = dec_input.size(1)
        if memory.size(1) >= steps:
            return memory[:, -steps:, :]
        pad = memory[:, -1:, :].expand(-1, steps - memory.size(1), -1)
        return torch.cat([memory, pad], dim=1)


class _StructuredInformerEncoder(_InformerEncoder):
    def forward(self, src: torch.Tensor) -> TransformerEncoderOutput:
        return TransformerEncoderOutput(last_hidden_state=self.proj(src))


class _StructuredInformerDecoder(_InformerDecoder):
    def forward(
        self, dec_input: torch.Tensor, memory: torch.Tensor, **kwargs
    ) -> TransformerDecoderOutput:
        output = super().forward(dec_input, memory, **kwargs)
        return TransformerDecoderOutput(last_hidden_state=output)


def test_informer_style_projects_decoder_features_to_output_size() -> None:
    model = ForecastingModel(
        encoder=_InformerEncoder(),
        decoder=_InformerDecoder(),
        forecasting_strategy="seq2seq",
        model_type="informer-like",
        target_len=4,
        output_size=2,
        hidden_size=5,
        label_len=2,
    )

    out = model(torch.randn(3, 6, 3))

    assert out.shape == (3, 4, 2)


def test_informer_style_accepts_structured_encoder_output() -> None:
    model = ForecastingModel(
        encoder=_StructuredInformerEncoder(),
        decoder=_InformerDecoder(),
        forecasting_strategy="seq2seq",
        model_type="informer-like",
        target_len=4,
        output_size=2,
        hidden_size=5,
        label_len=2,
    )

    out = model(torch.randn(3, 6, 3))

    assert out.shape == (3, 4, 2)


def test_informer_style_accepts_structured_decoder_output() -> None:
    model = ForecastingModel(
        encoder=_StructuredInformerEncoder(),
        decoder=_StructuredInformerDecoder(),
        forecasting_strategy="seq2seq",
        model_type="informer-like",
        target_len=4,
        output_size=2,
        hidden_size=5,
        label_len=2,
    )

    out = model(torch.randn(3, 6, 3))

    assert out.shape == (3, 4, 2)
