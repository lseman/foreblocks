"""foreblocks.models.discharge.backbones.

Waveform and envelope encoders for the discharge classifier.

Core API:
- WaveformEncoder: shared multiscale encoder over 10 ms waveform frames
- EnvelopeEncoder: dilated temporal encoder over the resampled amplitude envelope
- compute_envelope: Hilbert-magnitude envelope with antialiased resampling

"""

from foreblocks.models.discharge.backbones.envelope import (
    EnvelopeEncoder,
    compute_envelope,
)
from foreblocks.models.discharge.backbones.pooling import TemporalAttentionPool
from foreblocks.models.discharge.backbones.waveform import WaveformEncoder

__all__ = [
    "TemporalAttentionPool",
    "WaveformEncoder",
    "EnvelopeEncoder",
    "compute_envelope",
]
