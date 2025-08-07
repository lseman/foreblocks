from typing import Any, Dict

import numpy as np
import pandas as pd

from .foreminer_aux import *


class MissingnessAnalyzer(AnalysisStrategy):
    """SOTA analysis of missingness patterns, correlations, and MNAR behavior"""

    @property
    def name(self) -> str:
        return "missingness"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        results = {}
        missing_stats = data.isnull().mean()

        # Basic missing summary
        results["missing_rate"] = missing_stats[missing_stats > 0].sort_values(
            ascending=False
        )

        # Missingness correlation (standard + Jaccard)
        missing_mask = data.isnull().astype(int)
        results["missingness_correlation"] = missing_mask.corr()

        # Pairwise Jaccard similarity
        jaccard_matrix = missing_mask.T.dot(missing_mask)
        counts = missing_mask.sum(axis=0)
        jaccard_matrix = jaccard_matrix.div(counts, axis=0).div(counts, axis=1)
        results["missingness_jaccard"] = jaccard_matrix

        # Optionally cluster missingness patterns
        if missing_mask.shape[1] >= 2:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import pairwise_distances

            dists = pairwise_distances(missing_mask.T, metric="hamming")
            clustering = AgglomerativeClustering(
                n_clusters=min(5, len(missing_mask.columns)), linkage="average"
            )
            results["missingness_clusters"] = dict(
                zip(missing_mask.columns, clustering.fit_predict(dists))
            )

        # MNAR analysis using both numeric and categorical targets
        target_col = getattr(config, "target", None)
        if target_col and target_col in data.columns:
            results["missing_vs_target"] = self._analyze_mnar(data, target_col, config)

        return results

    def _analyze_mnar(
        self, data: pd.DataFrame, target_col: str, config: AnalysisConfig
    ) -> Dict[str, Any]:
        import warnings

        from scipy.stats import chi2_contingency, ks_2samp, ttest_ind
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import mutual_info_score, roc_auc_score

        warnings.filterwarnings("ignore")
        target = data[target_col]
        insights = {}

        for col in data.columns:
            if col == target_col or data[col].isnull().sum() == 0:
                continue

            miss_indicator = data[col].isnull().astype(int)

            try:
                if target.dtype.kind in "ifc":  # Continuous target
                    group1 = target[miss_indicator == 1].dropna()
                    group2 = target[miss_indicator == 0].dropna()
                    if len(group1) > 5 and len(group2) > 5:
                        _, pval_t = ttest_ind(group1, group2, equal_var=False)
                        _, pval_ks = ks_2samp(group1, group2)

                        # Predicting missingness with logistic regression
                        if len(set(miss_indicator)) == 2:
                            lr_model = LogisticRegression(solver="liblinear")
                            x_input = target.values.reshape(-1, 1)
                            lr_model.fit(x_input, miss_indicator)
                            auc = roc_auc_score(
                                miss_indicator, lr_model.predict_proba(x_input)[:, 1]
                            )
                        else:
                            auc = 0

                        insights[col] = {
                            "ttest_p": pval_t,
                            "ks_p": pval_ks,
                            "auc": auc,
                            "suggested_mnar": (
                                pval_t < 0.05 or pval_ks < 0.05 or auc > 0.7
                            ),
                        }

                elif target.dtype.name == "category" or target.nunique() < 15:
                    contingency = pd.crosstab(miss_indicator, target)
                    if contingency.shape[0] == 2 and contingency.shape[1] >= 2:
                        _, pval, _, _ = chi2_contingency(contingency)
                        cramer_v = np.sqrt(pval * min(contingency.shape) / len(target))
                        mi = mutual_info_score(miss_indicator, target)
                        insights[col] = {
                            "chi2_p": pval,
                            "cramers_v": cramer_v,
                            "mutual_info": mi,
                            "suggested_mnar": pval < 0.05
                            or cramer_v > 0.1
                            or mi > 0.05,
                        }

            except Exception as e:
                if config.verbose:
                    print(f"MNAR analysis failed for {col}: {e}")

        return insights

