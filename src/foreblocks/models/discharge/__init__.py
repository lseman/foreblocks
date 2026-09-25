"""foreblocks.models.discharge.

Two-branch discharge classifier with class-conditional novelty flagging.

Classifies acoustic partial-discharge signals into known classes (e.g.
corona, dry-band arcing, surface/puncture discharge) from raw waveform
context windows, and flags signals that don't resemble any known class well.
Follows the design recommended in
`datasets/_paper_review/model_proposal.md` after reviewing Lutfi et al.,
"Nonintrusive Ultrasonic Sensing and Deep Learning for Outdoor Ceramic
Insulator Assessment," IEEE TDEI 31(6), 2024.

Core API:
- DischargeClassifier: fit/predict orchestration for the two-branch model
- DischargeClassifierConfig: configuration
- DischargeResult: prediction output (class, probabilities, embedding,
  novelty score, unfamiliar flag, calibrated novelty tail p-value)
- ClassConditionalNovelty: native Mahalanobis, ECOD, COPOD, or cosine rejection
- TemporalAttentionPool: order-aware attention-weighted frame statistics
- split_training_indices: class-preserving recording-group development split
- WaveformEncoder, EnvelopeEncoder, compute_envelope: reusable backbones
- RocketFeatures, RocketDischargeClassifier: matched Rocket-family baseline
- extract_features, FeatureDischargeClassifier: matched engineered-feature baseline

"""

from foreblocks.models.discharge.backbones import (
    EnvelopeEncoder,
    TemporalAttentionPool,
    WaveformEncoder,
    compute_envelope,
)
from foreblocks.models.discharge.classifier import DischargeClassifier, DischargeResult
from foreblocks.models.discharge.config import DischargeClassifierConfig
from foreblocks.models.discharge.feature_baseline import (
    FeatureDischargeClassifier,
    extract_features,
)
from foreblocks.models.discharge.novelty import ClassConditionalNovelty
from foreblocks.models.discharge.rocket import RocketDischargeClassifier, RocketFeatures
from foreblocks.models.discharge.validation import split_training_indices

__all__ = [
    "TemporalAttentionPool",
    "split_training_indices",
    "DischargeClassifier",
    "DischargeClassifierConfig",
    "DischargeResult",
    "ClassConditionalNovelty",
    "WaveformEncoder",
    "EnvelopeEncoder",
    "compute_envelope",
    "RocketFeatures",
    "RocketDischargeClassifier",
    "extract_features",
    "FeatureDischargeClassifier",
]
