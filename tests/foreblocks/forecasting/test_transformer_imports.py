"""Public import surface of the transformer package."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("base", "BaseTransformerLayer"),
        ("encoder", "TransformerEncoderLayer"),
        ("decoder", "TransformerDecoderLayer"),
        ("mixing", "MixingTransformer"),
    ],
)
def test_layers_are_exported_from_the_package(module_name, class_name):
    package = importlib.import_module("foreblocks.nn.transformer")
    layers = importlib.import_module("foreblocks.nn.transformer.layers")
    implementation = importlib.import_module(
        f"foreblocks.nn.transformer.layers.{module_name}"
    )
    layer_class = getattr(implementation, class_name)
    assert getattr(package, class_name) is layer_class
    assert getattr(layers, class_name) is layer_class


@pytest.mark.parametrize(
    "class_name",
    ["BaseTransformerLayer", "TransformerEncoderLayer", "TransformerDecoderLayer"],
)
def test_public_layer_import_does_not_load_model_stacks(class_name):
    code = f"""
import sys
from foreblocks.nn.transformer import {class_name}
for stack in ("base", "encoder", "decoder"):
    assert f"foreblocks.nn.transformer.{{stack}}" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
