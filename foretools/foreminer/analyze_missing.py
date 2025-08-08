from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .foreminer_aux import *


class MissingnessAnalyzer(AnalysisStrategy):
    """SOTA analysis of missingness patterns, correlation structures, and MNAR behavior."""

    @property
    def name(self) -> str:
        return "missingness"

    # --------------- Public API ---------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # 1) Basic missing rate
        missing_rate = self._missing_rate(data)
        results["missing_rate"] = missing_rate[missing_rate > 0].sort_values(ascending=False)

        # 2) Missing mask & correlation
        missing_mask = data.isna().astype(int)
        if missing_mask.shape[1] < 2:
            return results

        results["missingness_correlation"] = missing_mask.corr()

        # 3) Jaccard similarity between feature-missingness
        results["missingness_jaccard"] = self._compute_jaccard(missing_mask)

        # 4) Clustering of features by missingness pattern
        results["missingness_clusters"] = (
            self._cluster_missingness(missing_mask) if missing_mask.shape[1] >= 2 else {}
        )

        # 5) Optional MNAR analysis vs a provided target
        target_col = getattr(config, "target", None)
        if target_col and target_col in data.columns:
            results["missing_vs_target"] = self._analyze_mnar(data, target_col, config)

        return results

    # --------------- Small helpers ---------------
    @staticmethod
    def _missing_rate(df: pd.DataFrame) -> pd.Series:
        return df.isna().mean()

    @staticmethod
    def _compute_jaccard(missing_mask: pd.DataFrame) -> pd.DataFrame:
        """
        Proper Jaccard for binary columns:
            J(i,j) = |Mi ∩ Mj| / |Mi ∪ Mj|
                    = dot_ij / (sum_i + sum_j - dot_ij)
        """
        # intersections
        inter = missing_mask.T.dot(missing_mask).astype(float)  # (p x p)
        sums = missing_mask.sum(axis=0).astype(float)           # (p,)
        # broadcast sums to get unions
        union = sums.values[:, None] + sums.values[None, :] - inter.values
        with np.errstate(invalid="ignore", divide="ignore"):
            jacc = np.divide(inter.values, union, out=np.zeros_like(inter.values), where=union > 0)

        # set diagonal to 1 where a column has at least one missing; else 0
        diag = (sums.values > 0).astype(float)
        np.fill_diagonal(jacc, diag)

        return pd.DataFrame(jacc, index=missing_mask.columns, columns=missing_mask.columns)

    @staticmethod
    def _cluster_missingness(missing_mask: pd.DataFrame) -> Dict[str, int]:
        """
        Cluster features by similarity of their missingness patterns using
        AgglomerativeClustering on a precomputed Hamming distance matrix.
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import pairwise_distances

            # distance between feature-missingness vectors (columns)
            dists = pairwise_distances(missing_mask.T, metric="hamming")
            n_features = len(missing_mask.columns)
            n_clusters = int(np.clip(n_features // 2, 2, 5))

            # use precomputed distances with average linkage
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="average",
                metric="precomputed",
            )
            labels = model.fit_predict(dists)
            return dict(zip(missing_mask.columns.tolist(), labels.tolist()))
        except Exception:
            return {}

    # --------------- MNAR diagnostics ---------------
    def _analyze_mnar(self, data: pd.DataFrame, target_col: str, config: AnalysisConfig) -> Dict[str, Any]:
        from scipy.stats import chi2_contingency, ks_2samp, ttest_ind
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import mutual_info_score, roc_auc_score

        insights: Dict[str, Any] = {}
        target = data[target_col]

        for col in data.columns:
            if col == target_col:
                continue
            miss_indicator = data[col].isna().astype(int)
            msum = int(miss_indicator.sum())
            if msum < 5:  # not enough missing entries to say anything meaningful
                continue

            try:
                result = {"suggested_mnar": False, "suggested_mnar_reason": ""}

                # Numeric/continuous target
                if target.dtype.kind in "ifc":
                    g1 = target[miss_indicator == 1].dropna()
                    g0 = target[miss_indicator == 0].dropna()
                    if len(g1) > 5 and len(g0) > 5:
                        # Welch t-test + KS
                        _, p_t = ttest_ind(g1, g0, equal_var=False)
                        _, p_ks = ks_2samp(g1, g0)
                        result.update({"ttest_p": p_t, "ks_p": p_ks})

                        # Predict missingness from target (AUROC)
                        auc = np.nan
                        if miss_indicator.nunique() == 2:
                            try:
                                lr = LogisticRegression(solver="liblinear")
                                x = target.to_numpy().reshape(-1, 1)
                                lr.fit(x, miss_indicator)
                                auc = roc_auc_score(miss_indicator, lr.predict_proba(x)[:, 1])
                            except Exception:
                                pass
                        result["auc"] = auc

                        # Heuristic MNAR suggestion
                        if (p_t < 0.05) or (p_ks < 0.05) or (np.isfinite(auc) and auc > 0.70):
                            result["suggested_mnar"] = True
                            reasons = []
                            if p_t < 0.05: reasons.append("ttest")
                            if p_ks < 0.05: reasons.append("ks")
                            if np.isfinite(auc) and auc > 0.70: reasons.append("auc")
                            result["suggested_mnar_reason"] = ", ".join(reasons)

                # Categorical / low-cardinality target
                elif target.dtype.name == "category" or target.nunique(dropna=True) < 15:
                    contingency = pd.crosstab(miss_indicator, target)
                    # need both 0 and 1 rows in the indicator
                    if contingency.shape[0] == 2:
                        chi2, p_chi2, _, _ = chi2_contingency(contingency)
                        n = contingency.to_numpy().sum()
                        r, c = contingency.shape
                        # Cramér's V (bias-unadjusted), safe for r,c >= 2
                        denom = n * (min(r - 1, c - 1) if min(r, c) > 1 else 1)
                        cramers_v = float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan
                        mi = float(mutual_info_score(miss_indicator, target))

                        result.update({
                            "chi2_p": p_chi2,
                            "cramers_v": cramers_v,
                            "mutual_info": mi,
                        })

                        if (p_chi2 < 0.05) or (np.isfinite(cramers_v) and cramers_v > 0.10) or (mi > 0.05):
                            result["suggested_mnar"] = True
                            reasons = []
                            if p_chi2 < 0.05: reasons.append("chi2")
                            if np.isfinite(cramers_v) and cramers_v > 0.10: reasons.append("cramers_v")
                            if mi > 0.05: reasons.append("mutual_info")
                            result["suggested_mnar_reason"] = ", ".join(reasons)

                # record only if we computed something beyond defaults
                if len(result) > 2:
                    insights[col] = result

            except Exception as e:
                if getattr(config, "verbose", False):
                    print(f"[⚠️] MNAR analysis failed for {col}: {e}")

        return insights
