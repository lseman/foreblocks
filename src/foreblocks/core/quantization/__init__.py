"""Quantization configurations and modules."""

from foreblocks.core.quantization.modules import (
    DynamicQuantizedLinear,
    FakeQuantize,
    ManualDeQuantStub,
    ManualQuantStub,
    QuantizationConfig,
    QuantizationObserver,
    QuantizedLinear,
    StaticQuantizedLinear,
)

__all__ = [
    "DynamicQuantizedLinear",
    "FakeQuantize",
    "ManualDeQuantStub",
    "ManualQuantStub",
    "QuantizationConfig",
    "QuantizationObserver",
    "QuantizedLinear",
    "StaticQuantizedLinear",
]
