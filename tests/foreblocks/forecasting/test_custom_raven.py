import json
import subprocess
import sys

import pytest


def test_custom_raven_uses_submodule_fla():
    from foreblocks.nn.sequence.raven import Raven

    assert Raven.__module__ == "foreblocks.nn.sequence.raven.blocks.raven"


def test_raven_import_and_configuration_do_not_import_transformers():
    code = """
import importlib.abc
import sys

class RejectTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'transformers' or fullname.startswith('transformers.'):
            raise AssertionError('Raven must not import transformers')

sys.meta_path.insert(0, RejectTransformers())
from foreblocks.nn.sequence.raven import Raven, RavenBlock, RavenConfig
config = RavenConfig(hidden_size=16, num_heads=2)
assert RavenConfig.from_dict(config.to_dict()) == config
assert not any(name == 'transformers' or name.startswith('transformers.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_raven_config_round_trip_and_overrides_do_not_mutate_inputs():
    from foreblocks.nn.sequence.raven import RavenConfig

    attention = {"layers": [0], "num_heads": 2}
    config = RavenConfig(
        hidden_size=16,
        num_heads=2,
        attn=attention,
        pad_token_id=0,
        bos_token_id=3,
        eos_token_id=4,
        tie_word_embeddings=True,
    )
    assert attention == {"layers": [0], "num_heads": 2}
    assert config.attn["num_kv_heads"] == 2
    assert config.attn["qkv_bias"] is False
    restored = RavenConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    updated = config.with_overrides(hidden_size=32)
    updated.attn["layers"].append(1)
    assert config.hidden_size == 16
    assert config.attn["layers"] == [0]
    snapshot = config.to_dict()
    snapshot["attn"]["layers"].append(2)
    assert config.attn["layers"] == [0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fuse_linear_cross_entropy": True}, "cannot both"),
        ({"attn": []}, "dictionary"),
        ({"attn": {"num_heads": 2}}, "Layer indices"),
        ({"attn": {"layers": [0]}}, "num_heads"),
        ({"attnres_block_size": 3}, "even integer"),
        ({"attnres_block_size": 0}, "even integer"),
    ],
)
def test_raven_config_preserves_validation(kwargs, message):
    from foreblocks.nn.sequence.raven import RavenConfig

    with pytest.raises(ValueError, match=message):
        RavenConfig(**kwargs)
