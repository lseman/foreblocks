"""Behavioral guards for shared anomaly components and import boundaries."""

import math

import numpy as np
import pytest
import torch

from foreblocks.models.anomaly import (
    COPOD,
    ECOD,
    HBOS,
    AnomalyBlockStack,
    AnomalyDetectorConfig,
    TranAD,
    TranADDetector,
)
from foreblocks.models.anomaly.backbones.tranad import (
    _TranADPositionalEncoding,
)
from foreblocks.models.anomaly.blocks import (
    BaseAnomalyBlock,
    list_blocks,
    resolve_block,
)
from foreblocks.models.anomaly.config import AnomalyDetectorConfig as Config
from foreblocks.models.anomaly.detector import AnomalyDetectorConfig as DetectorConfig
from foreblocks.models.anomaly.scorers import (
    COPOD as ScorerCOPOD,
    ECOD as ScorerECOD,
    HBOS as ScorerHBOS,
)
from foreblocks.models.anomaly.tranad_detector import TranADDetector as DedicatedTranAD
from foreblocks.models.anomaly.windows import robust_threshold


def test_public_api_uses_canonical_implementations():
    assert Config is AnomalyDetectorConfig is DetectorConfig
    assert ECOD is ScorerECOD
    assert COPOD is ScorerCOPOD
    assert HBOS is ScorerHBOS
    assert TranADDetector is DedicatedTranAD
    assert TranAD.__module__ == "foreblocks.models.anomaly.backbones.tranad"
    assert AnomalyBlockStack.__module__ == "foreblocks.models.anomaly.blocks"


@pytest.mark.parametrize("width", [1, 3, 7, 16, 32])
def test_tranad_positional_encoding_preserves_legacy_values_and_state(width):
    length = 512
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width)
    )
    legacy = torch.zeros(length, width)
    legacy[:, 0::2] = torch.sin(position * frequency)
    legacy[:, 1::2] = torch.cos(position * frequency[: legacy[:, 1::2].shape[1]])
    legacy = legacy.unsqueeze(0)
    encoder = _TranADPositionalEncoding(width, dropout=0, max_len=length)
    x = torch.randn(4, 19, width)
    torch.testing.assert_close(encoder(x), x + legacy[:, :19], rtol=0, atol=0)
    assert set(encoder.state_dict()) == {"pe"}
    encoder.load_state_dict({"pe": legacy}, strict=True)
    torch.testing.assert_close(encoder(x), x + legacy[:, :19], rtol=0, atol=0)


@pytest.mark.parametrize("name", list_blocks())
def test_registered_modes_share_decision_contract(name):
    block = resolve_block(name)
    assert isinstance(block, BaseAnomalyBlock)
    assert block.block_type() == name
    scores = np.array([[0.0, 0.1], [0.1, 0.0], [0.2, 0.1], [100.0, 10.0]])
    reduced = scores.max(axis=1)
    expected = (reduced > robust_threshold(reduced, contamination=0.25)).astype(int)
    np.testing.assert_array_equal(block.decide(scores, 0.25), expected)
    np.testing.assert_array_equal(block.decide(np.full(3, np.nan), 0.1), [0, 0, 0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("features", [None, 3])
def test_window_helpers_preserve_shapes_values_and_lazy_dataset(dtype, features):
    from foreblocks.models.anomaly import (
        TranADDataset,
        build_sliding_windows,
        create_sequences_vectorized,
    )

    raw = np.arange(24, dtype=dtype)
    data = raw if features is None else raw.reshape(-1, features)
    matrix = data[:, None] if features is None else data
    expected = np.stack([matrix[i : i + 4] for i in range(len(matrix) - 3)])
    np.testing.assert_array_equal(build_sliding_windows(data, 4), expected)
    tensor = create_sequences_vectorized(data, 4)
    torch.testing.assert_close(tensor, torch.tensor(expected, dtype=torch.float32))
    source = torch.from_numpy(matrix).float()
    lazy = TranADDataset(source, 4)
    assert len(lazy) == len(expected)
    torch.testing.assert_close(lazy[0], source[:4])
    assert lazy[0].data_ptr() == source.data_ptr()
    with pytest.raises(ValueError):
        create_sequences_vectorized(data, len(matrix) + 1)
