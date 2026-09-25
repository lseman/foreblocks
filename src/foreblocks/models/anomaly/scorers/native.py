"""Native fitted detector registry. Every implementation lives in Foreblocks."""

from foreblocks.models.anomaly.scorers.density import LODA, GaussianMixtureScorer
from foreblocks.models.anomaly.scorers.isolation import DeepIsolationForest
from foreblocks.models.anomaly.scorers.neighbors import INNE, KNNScorer
from foreblocks.models.anomaly.scorers.neural import (
    AutoEncoderScorer,
    DeepSVDD,
    VAEScorer,
)

NATIVE_MODELS = {
    "inne": INNE,
    "loda": LODA,
    "knn": KNNScorer,
    "gmm": GaussianMixtureScorer,
    "dif": DeepIsolationForest,
    "autoencoder": AutoEncoderScorer,
    "vae": VAEScorer,
    "deep_svdd": DeepSVDD,
}
