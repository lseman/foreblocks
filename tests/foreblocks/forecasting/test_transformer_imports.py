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


def test_transformer_configuration_does_not_import_attention_implementations():
    code = """
import sys
from foreblocks.nn.transformer import TransformerConfig
config = TransformerConfig(d_model=16, n_heads=4)
assert 'foreblocks.nn.attention.multihead' not in sys.modules
assert not any(name.startswith('foreblocks.nn.attention.algorithms.') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_residual_execution_does_not_import_stack_checkpointing():
    code = """
import sys
from foreblocks.nn.transformer.runtime.execution import LayerExecutionStrategy
assert 'foreblocks.nn.transformer.runtime.checkpointing' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_moved_runtime_helpers_keep_their_former_import_locations():
    from foreblocks.nn.transformer import base
    from foreblocks.nn.transformer.runtime import (
        checkpointing,
        execution,
        forward,
        outputs,
        preparation,
    )

    for name in preparation.__all__:
        assert getattr(forward, name) is getattr(preparation, name)
    assert forward.build_decoder_output is outputs.build_decoder_output
    assert base.StackTrace is forward.StackTrace
    for name in checkpointing.__all__:
        assert getattr(execution, name) is getattr(checkpointing, name)
