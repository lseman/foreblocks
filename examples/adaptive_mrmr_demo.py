from __future__ import annotations

import numpy as np
import pandas as pd

from foretools import AdaptiveMI, AdaptiveMRMR
from foretools.fengineer import FeatureConfig
from foretools.fengineer.selectors.feature_selector import FeatureSelector


class ToyAdaptiveMI:
    """
    Tiny deterministic scorer used only to illustrate why MID and MIQ can
    select different features from the same relevance/redundancy structure.
    """

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self._relevance = {
            "lead": 0.95,
            "candidate_mid": 0.94,
            "candidate_miq": 0.75,
        }
        self._redundancy = {
            frozenset(("lead", "candidate_mid")): 0.30,
            frozenset(("lead", "candidate_miq")): 0.12,
            frozenset(("candidate_mid", "candidate_miq")): 0.18,
        }
        self._values_to_name = {}

    def bind_columns(self, X_values):
        self._values_to_name = {
            tuple(np.asarray(X_values[:, idx], dtype=float).tolist()): name
            for idx, name in enumerate(self.feature_names)
        }

    def score_pairwise(self, X_values, y_values, return_raw_mi=False):
        self.bind_columns(X_values)
        return np.array(
            [self._relevance[name] for name in self.feature_names], dtype=float
        )

    def score(self, xa, xb, return_raw_mi=False):
        a = self._values_to_name[tuple(np.asarray(xa, dtype=float).tolist())]
        b = self._values_to_name[tuple(np.asarray(xb, dtype=float).tolist())]
        return float(self._redundancy[frozenset((a, b))])


def _print_title(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def demo_criterion_difference() -> None:
    _print_title("Toy scorer: MID and MIQ can diverge")

    X = pd.DataFrame(
        {
            "lead": [0.0, 1.0, 2.0, 3.0],
            "candidate_mid": [10.0, 11.0, 12.0, 13.0],
            "candidate_miq": [20.0, 21.0, 22.0, 23.0],
        }
    )
    y = pd.Series([0.0, 1.0, 0.0, 1.0])

    for criterion in ("mid", "miq"):
        selector = AdaptiveMRMR(
            scorer=ToyAdaptiveMI(X.columns),
            criterion=criterion,
            candidate_pool=3,
            stable_relevance=False,
            random_state=7,
        )
        selector.fit(
            X,
            y,
            min_features=2,
            max_features=2,
            mi_threshold=0.0,
            min_samples=1,
        )
        print(f"{criterion.upper()} selected: {selector.selected_features_}")
        print(selector.selection_scores_)


def demo_real_usage() -> None:
    _print_title("Real AdaptiveMRMR usage on synthetic data")

    rng = np.random.default_rng(42)
    n = 320

    lead = rng.normal(size=n)
    duplicate = lead + 0.03 * rng.normal(size=n)
    complementary = rng.normal(size=n)
    weak = rng.normal(size=n)
    y = pd.Series(1.3 * lead + 0.75 * complementary + 0.05 * rng.normal(size=n))
    X = pd.DataFrame(
        {
            "lead": lead,
            "duplicate": duplicate,
            "complementary": complementary,
            "weak": weak,
        }
    )

    for criterion in ("mid", "miq"):
        selector = AdaptiveMRMR(
            scorer=AdaptiveMI(random_state=42, rho_threshold=0.3),
            criterion=criterion,
            candidate_pool=4,
            stable_relevance=False,
            random_state=42,
        )
        selector.fit(
            X,
            y,
            min_features=2,
            max_features=2,
            mi_threshold=0.0,
        )
        print(f"{criterion.upper()} selected: {selector.selected_features_}")
        print(
            "note: on this synthetic dataset both criteria agree on the kept features"
        )
        print("relevance:")
        print(selector.relevance_scores_)
        print("selection scores:")
        print(selector.selection_scores_)

    cfg = FeatureConfig(
        selector_method="mrmr",
        selector_stable_mi=False,
        min_features=2,
        max_features=2,
        mi_threshold=0.0,
        mrmr_candidate_pool=4,
        mrmr_criterion="miq",
    )
    fs = FeatureSelector(cfg)
    fs.ami_scorer = AdaptiveMI(random_state=42, rho_threshold=0.3)
    fs.fit(X, y)
    print("FeatureSelector with mRMR/MIQ selected:", fs.selected_features_)


if __name__ == "__main__":
    demo_criterion_difference()
    demo_real_usage()
