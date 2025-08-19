from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    anderson,
    chi2_contingency,
    f_oneway,
    fisher_exact,
    kruskal,
    levene,
    mannwhitneyu,
    ttest_ind,
)

from .foreminer_aux import *

# Modern scipy features with fallbacks
try:
    from scipy.stats import bootstrap
    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False

try:
    from scipy.stats import permutation_test
    HAS_PERMUTATION_TEST = True
except ImportError:
    HAS_PERMUTATION_TEST = False


class CategoricalGroupAnalyzer(AnalysisStrategy):
    """SOTA categorical group analysis with streamlined, modern statistical methods"""

    @property
    def name(self) -> str:
        return "categorical_groups"

    def __init__(self):
        self.min_group_size = 5
        self.max_groups = 20
        self.bootstrap_iterations = 1000
        self.alpha = 0.05

    # --------------------------- Auto-Detection (Simplified) ---------------------------
    def _auto_detect_categorical_column(self, data: pd.DataFrame) -> Optional[str]:
        """Detect the best categorical column for analysis"""
        candidates = []
        
        # Check object/category columns
        for col in data.select_dtypes(include=['object', 'category']).columns:
            if self._is_suitable_categorical_column(data, col):
                candidates.append((col, self._score_categorical_column(data, col)))
        
        # Check low-cardinality numeric columns
        for col in data.select_dtypes(include=[np.number]).columns:
            unique_count = data[col].nunique()
            if 2 <= unique_count <= self.max_groups:
                if self._is_suitable_categorical_column(data, col):
                    candidates.append((col, self._score_categorical_column(data, col)))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _is_suitable_categorical_column(self, data: pd.DataFrame, col: str) -> bool:
        """Check if column is suitable for group analysis"""
        try:
            unique_count = data[col].nunique()
            if unique_count < 2 or unique_count > self.max_groups:
                return False
            
            value_counts = data[col].value_counts()
            min_group_size = value_counts.min()
            
            if min_group_size < self.min_group_size:
                return False
            
            missing_pct = data[col].isnull().sum() / len(data) * 100
            if missing_pct > 50:
                return False
            
            return True
        except Exception:
            return False

    def _score_categorical_column(self, data: pd.DataFrame, col: str) -> float:
        """Score categorical column suitability"""
        try:
            score = 0.0
            
            unique_count = data[col].nunique()
            if unique_count == 2:
                score += 10
            elif 3 <= unique_count <= 5:
                score += 8
            elif 6 <= unique_count <= 10:
                score += 5
            else:
                score += 2
            
            # Group balance
            value_counts = data[col].value_counts()
            balance_ratio = value_counts.min() / value_counts.max()
            score += balance_ratio * 10
            
            # Sample size bonus
            min_group_size = value_counts.min()
            if min_group_size >= 100:
                score += 5
            elif min_group_size >= 50:
                score += 4
            elif min_group_size >= 30:
                score += 3
            else:
                score += 1
            
            # Missing data penalty
            missing_pct = data[col].isnull().sum() / len(data) * 100
            score -= missing_pct / 10
            
            return max(0, score)
        except Exception:
            return 0.0

    # --------------------------- Data Preparation ---------------------------
    def _prepare_group_data(self, data: pd.DataFrame, categorical_col: str, 
                           target_cols: List[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare and validate data for group analysis"""
        if categorical_col not in data.columns:
            raise ValueError(f"Categorical column '{categorical_col}' not found")
        
        clean_data = data[[categorical_col] + (target_cols or [])].dropna()
        if len(clean_data) == 0:
            raise ValueError("No valid data after removing missing values")
        
        group_info = {
            "categorical_column": categorical_col,
            "total_samples": len(clean_data),
            "original_samples": len(data),
            "missing_samples": len(data) - len(clean_data),
            "missing_percentage": (len(data) - len(clean_data)) / len(data) * 100
        }
        
        group_counts = clean_data[categorical_col].value_counts()
        valid_groups = [group for group in group_counts.index 
                       if group_counts[group] >= self.min_group_size]
        
        if len(valid_groups) < 2:
            raise ValueError(f"Need at least 2 groups with minimum {self.min_group_size} samples")
        
        if len(valid_groups) > self.max_groups:
            valid_groups = group_counts.head(self.max_groups).index.tolist()
            group_info["groups_truncated"] = True
        else:
            group_info["groups_truncated"] = False
        
        clean_data = clean_data[clean_data[categorical_col].isin(valid_groups)]
        group_counts = clean_data[categorical_col].value_counts()
        
        group_info.update({
            "groups": valid_groups,
            "n_groups": len(valid_groups),
            "group_sizes": dict(group_counts),
            "min_group_size": int(group_counts.min()),
            "max_group_size": int(group_counts.max()),
            "balanced_groups": group_counts.max() / group_counts.min() <= 3.0
        })
        
        grouped_data = {}
        for group in valid_groups:
            group_mask = clean_data[categorical_col] == group
            grouped_data[group] = clean_data[group_mask]
        
        return grouped_data, group_info

    # --------------------------- SOTA Assumption Testing (Streamlined) ---------------------------
    def _smart_assumption_testing(self, grouped_data: Dict[str, pd.DataFrame], 
                                 numeric_col: str) -> Dict[str, Any]:
        """Streamlined assumption testing using only the most reliable methods"""
        assumptions = {
            "normality_test": {},
            "variance_test": {},
            "sample_info": {},
            "recommended_approach": ""
        }
        
        # Extract group data
        group_arrays = {}
        for group_name, group_df in grouped_data.items():
            if numeric_col in group_df.columns:
                group_data = group_df[numeric_col].dropna().values
                if len(group_data) > 0:
                    group_arrays[group_name] = group_data
        
        if len(group_arrays) < 2:
            return assumptions
        
        all_data = np.concatenate(list(group_arrays.values()))
        smallest_group = min(len(arr) for arr in group_arrays.values())
        
        assumptions["sample_info"] = {
            "total_samples": len(all_data),
            "n_groups": len(group_arrays),
            "smallest_group": smallest_group,
            "largest_group": max(len(arr) for arr in group_arrays.values())
        }
        
        # Normality test: Use Anderson-Darling (most powerful)
        normality_passed = True
        try:
            for group_name, group_data in group_arrays.items():
                if len(group_data) >= 8:  # Minimum for reliable test
                    result = anderson(group_data, dist='norm')
                    # Use 5% critical value
                    critical_5pct = result.critical_values[2]
                    is_normal = result.statistic < critical_5pct
                    
                    if not is_normal:
                        normality_passed = False
                        break
            
            assumptions["normality_test"] = {
                "method": "Anderson-Darling",
                "all_groups_normal": normality_passed
            }
        except Exception:
            normality_passed = False
            assumptions["normality_test"] = {"method": "Anderson-Darling", "failed": True}
        
        # Variance homogeneity: Use Levene's test (robust)
        variances_equal = True
        try:
            stat, p_val = levene(*list(group_arrays.values()))
            variances_equal = p_val > self.alpha
            
            assumptions["variance_test"] = {
                "method": "Levene's test",
                "statistic": float(stat),
                "p_value": float(p_val),
                "equal_variances": variances_equal
            }
        except Exception:
            assumptions["variance_test"] = {"method": "Levene's test", "failed": True}
        
        # Smart recommendation based on sample size and assumptions
        if smallest_group >= 30 and normality_passed and variances_equal:
            assumptions["recommended_approach"] = "parametric"
        elif smallest_group >= 15:
            assumptions["recommended_approach"] = "robust_parametric"  # Welch's methods
        else:
            assumptions["recommended_approach"] = "nonparametric"
        
        return assumptions

    # --------------------------- SOTA Statistical Tests ---------------------------
    def _run_statistical_tests(self, grouped_data: Dict[str, pd.DataFrame], 
                              numeric_col: str, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Run only the most appropriate and reliable statistical tests"""
        results = {}
        
        # Extract group data
        group_arrays = {}
        for group_name, group_df in grouped_data.items():
            if numeric_col in group_df.columns:
                group_data = group_df[numeric_col].dropna().values
                if len(group_data) > 0:
                    group_arrays[group_name] = group_data
        
        if len(group_arrays) < 2:
            return results
        
        group_names = list(group_arrays.keys())
        group_data_list = list(group_arrays.values())
        
        # Method 1: Welch's methods (robust to unequal variances)
        if len(group_arrays) == 2:
            # Welch's t-test (gold standard for 2 groups)
            try:
                group1, group2 = group_data_list[0], group_data_list[1]
                stat, p_val = ttest_ind(group1, group2, equal_var=False)
                
                # Cohen's d with pooled std
                pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
                
                results["welch_t_test"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "cohens_d": float(cohens_d),
                    "effect_size": self._interpret_effect_size(abs(cohens_d), "cohens_d"),
                    "groups_compared": group_names,
                    "method": "Welch's t-test (unequal variances)",
                    "primary": True
                }
            except Exception as e:
                results["welch_t_test"] = {"error": str(e)}
        
        else:
            # Welch's ANOVA for multiple groups
            try:
                k = len(group_data_list)
                ni = np.array([len(group) for group in group_data_list])
                yi = np.array([np.mean(group) for group in group_data_list])
                si2 = np.array([np.var(group, ddof=1) for group in group_data_list])
                wi = ni / si2
                
                y_weighted = np.sum(wi * yi) / np.sum(wi)
                numerator = np.sum(wi * (yi - y_weighted)**2)
                
                sum_wi = np.sum(wi)
                sum_wi2_ni = np.sum(wi**2 / ni)
                denom = 1 + (2 * (k - 2) / (k**2 - 1)) * sum_wi2_ni / sum_wi**2
                
                welch_stat = numerator / ((k - 1) * denom)
                df1 = k - 1
                df2 = (k**2 - 1) / (3 * sum_wi2_ni / sum_wi**2)
                
                p_val = 1 - stats.f.cdf(welch_stat, df1, df2)
                
                results["welch_anova"] = {
                    "statistic": float(welch_stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "df1": float(df1),
                    "df2": float(df2),
                    "method": "Welch's ANOVA (unequal variances)",
                    "primary": True
                }
            except Exception as e:
                results["welch_anova"] = {"error": str(e)}
        
        # Method 2: Non-parametric tests (distribution-free)
        if len(group_arrays) == 2:
            # Mann-Whitney U test
            try:
                group1, group2 = group_data_list[0], group_data_list[1]
                stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                
                # Rank-biserial correlation (effect size)
                n1, n2 = len(group1), len(group2)
                rank_biserial = 1 - (2 * stat) / (n1 * n2)
                
                results["mann_whitney"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "rank_biserial": float(rank_biserial),
                    "effect_size": self._interpret_effect_size(abs(rank_biserial), "rank_biserial"),
                    "groups_compared": group_names,
                    "method": "Mann-Whitney U test"
                }
            except Exception as e:
                results["mann_whitney"] = {"error": str(e)}
        else:
            # Kruskal-Wallis test
            try:
                stat, p_val = kruskal(*group_data_list)
                
                # Eta-squared effect size
                all_data = np.concatenate(group_data_list)
                n_total = len(all_data)
                k = len(group_data_list)
                eta_squared = (stat - k + 1) / (n_total - k) if n_total > k else 0
                eta_squared = max(0, eta_squared)
                
                results["kruskal_wallis"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "eta_squared": float(eta_squared),
                    "effect_size": self._interpret_effect_size(eta_squared, "eta_squared"),
                    "method": "Kruskal-Wallis H-test",
                    "n_groups": k
                }
            except Exception as e:
                results["kruskal_wallis"] = {"error": str(e)}
        
        # Method 3: Bootstrap test (most robust)
        if len(group_arrays) == 2:
            try:
                group1, group2 = group_data_list[0], group_data_list[1]
                
                if HAS_BOOTSTRAP:
                    # Use scipy bootstrap
                    def statistic(x, y):
                        return np.mean(x) - np.mean(y)
                    
                    rng = np.random.default_rng(42)
                    res = bootstrap((group1, group2), statistic, 
                                  n_resamples=self.bootstrap_iterations,
                                  paired=False, random_state=rng)
                    
                    observed_diff = statistic(group1, group2)
                    p_value = 2 * min(
                        np.mean(res.bootstrap_distribution >= abs(observed_diff)),
                        np.mean(res.bootstrap_distribution <= -abs(observed_diff))
                    )
                    
                    results["bootstrap_test"] = {
                        "observed_difference": float(observed_diff),
                        "p_value": float(p_value),
                        "significant": p_value < self.alpha,
                        "bootstrap_iterations": self.bootstrap_iterations,
                        "method": "Bootstrap test for difference in means",
                        "groups_compared": group_names
                    }
                else:
                    # Manual bootstrap
                    observed_diff = np.mean(group1) - np.mean(group2)
                    pooled_data = np.concatenate([group1, group2])
                    n1 = len(group1)
                    
                    bootstrap_diffs = []
                    np.random.seed(42)
                    for _ in range(self.bootstrap_iterations):
                        resampled = np.random.choice(pooled_data, size=len(pooled_data), replace=True)
                        boot_sample1 = resampled[:n1]
                        boot_sample2 = resampled[n1:]
                        boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
                        bootstrap_diffs.append(boot_diff)
                    
                    bootstrap_diffs = np.array(bootstrap_diffs)
                    p_value = 2 * min(
                        np.mean(bootstrap_diffs >= abs(observed_diff)),
                        np.mean(bootstrap_diffs <= -abs(observed_diff))
                    )
                    
                    results["bootstrap_test"] = {
                        "observed_difference": float(observed_diff),
                        "p_value": float(p_value),
                        "significant": p_value < self.alpha,
                        "bootstrap_iterations": self.bootstrap_iterations,
                        "method": "Bootstrap test for difference in means",
                        "groups_compared": group_names
                    }
            except Exception as e:
                results["bootstrap_test"] = {"error": str(e)}
        
        return results

    # --------------------------- Categorical Association Tests ---------------------------
    def _categorical_association_tests(self, data: pd.DataFrame, cat_col1: str, cat_col2: str) -> Dict[str, Any]:
        """Modern categorical association testing"""
        results = {}
        
        try:
            contingency_table = pd.crosstab(data[cat_col1], data[cat_col2])
            
            if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return {"error": "Insufficient data for categorical association tests"}
            
            # Chi-square test
            chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
            
            # Effect size: Cramér's V
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            results["chi_square"] = {
                "statistic": float(chi2_stat),
                "p_value": float(chi2_p),
                "degrees_of_freedom": int(dof),
                "significant": chi2_p < self.alpha,
                "cramers_v": float(cramers_v),
                "effect_size": self._interpret_effect_size(cramers_v, "cramers_v"),
                "method": "Chi-square test of independence"
            }
            
            # Fisher's exact test for 2x2 tables
            if contingency_table.shape == (2, 2):
                try:
                    odds_ratio, fisher_p = fisher_exact(contingency_table)
                    results["fisher_exact"] = {
                        "odds_ratio": float(odds_ratio),
                        "p_value": float(fisher_p),
                        "significant": fisher_p < self.alpha,
                        "method": "Fisher's exact test"
                    }
                except Exception as e:
                    results["fisher_exact"] = {"error": str(e)}
            
            results["contingency_info"] = {
                "table_shape": contingency_table.shape,
                "total_observations": int(n),
                "min_expected_frequency": float(expected.min())
            }
            
        except Exception as e:
            results["chi_square"] = {"error": str(e)}
        
        return results

    # --------------------------- Effect Size Interpretation ---------------------------
    def _interpret_effect_size(self, effect_size: float, effect_type: str) -> str:
        """Interpret effect size magnitude"""
        abs_effect = abs(effect_size)
        
        if effect_type == "cohens_d":
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        
        elif effect_type in ["eta_squared", "epsilon_squared"]:
            if abs_effect < 0.01:
                return "negligible"
            elif abs_effect < 0.06:
                return "small"
            elif abs_effect < 0.14:
                return "medium"
            else:
                return "large"
        
        elif effect_type == "cramers_v":
            if abs_effect < 0.1:
                return "negligible"
            elif abs_effect < 0.3:
                return "small"
            elif abs_effect < 0.5:
                return "medium"
            else:
                return "large"
        
        elif effect_type == "rank_biserial":
            if abs_effect < 0.1:
                return "negligible"
            elif abs_effect < 0.3:
                return "small"
            elif abs_effect < 0.5:
                return "medium"
            else:
                return "large"
        
        return "unknown"

    # --------------------------- Descriptive Statistics ---------------------------
    def _compute_descriptives(self, grouped_data: Dict[str, pd.DataFrame], 
                             numeric_col: str) -> Dict[str, Any]:
        """Compute essential descriptive statistics"""
        descriptives = {}
        
        for group_name, group_df in grouped_data.items():
            if numeric_col not in group_df.columns:
                continue
                
            group_data = group_df[numeric_col].dropna()
            if len(group_data) == 0:
                continue
            
            # Essential statistics
            stats_dict = {
                "n": int(len(group_data)),
                "mean": float(group_data.mean()),
                "median": float(group_data.median()),
                "std": float(group_data.std(ddof=1)),
                "min": float(group_data.min()),
                "max": float(group_data.max()),
                "q1": float(group_data.quantile(0.25)),
                "q3": float(group_data.quantile(0.75)),
                "skewness": float(group_data.skew())
            }
            
            # 95% confidence interval for mean
            sem = stats_dict["std"] / np.sqrt(stats_dict["n"])
            t_critical = stats.t.ppf(0.975, stats_dict["n"] - 1)
            margin_error = t_critical * sem
            stats_dict["ci_95_lower"] = float(stats_dict["mean"] - margin_error)
            stats_dict["ci_95_upper"] = float(stats_dict["mean"] + margin_error)
            
            descriptives[group_name] = stats_dict
        
        return descriptives

    # --------------------------- Smart Recommendations ---------------------------
    def _generate_recommendations(self, test_results: Dict[str, Any], 
                                 assumptions: Dict[str, Any], 
                                 group_info: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on results"""
        recommendations = []
        
        # Find most reliable significant test
        significant_tests = []
        for test_name, result in test_results.items():
            if isinstance(result, dict) and result.get("significant", False) and "error" not in result:
                # Prioritize bootstrap > Welch's > non-parametric
                reliability = 1.0
                if "bootstrap" in test_name:
                    reliability = 3.0  # Highest reliability
                elif "welch" in test_name:
                    reliability = 2.5  # High reliability, robust
                elif "mann_whitney" in test_name or "kruskal" in test_name:
                    reliability = 2.0  # Good reliability, distribution-free
                
                significant_tests.append((test_name, result, reliability))
        
        if significant_tests:
            # Sort by reliability
            significant_tests.sort(key=lambda x: x[2], reverse=True)
            best_test_name, best_result, _ = significant_tests[0]
            
            p_val = best_result.get("p_value", 1.0)
            method = best_result.get("method", best_test_name)
            
            recommendations.append(f"🎯 **Significant difference detected** using {method} (p = {p_val:.6f})")
            
            # Effect size
            effect_info = ""
            if "cohens_d" in best_result:
                d = best_result["cohens_d"]
                effect_info = f" with {best_result.get('effect_size', 'unknown')} effect size (d = {d:.3f})"
            elif "eta_squared" in best_result:
                eta = best_result["eta_squared"]
                effect_info = f" with {best_result.get('effect_size', 'unknown')} effect size (η² = {eta:.3f})"
            elif "rank_biserial" in best_result:
                rb = best_result["rank_biserial"]
                effect_info = f" with {best_result.get('effect_size', 'unknown')} effect size (rb = {rb:.3f})"
            
            if effect_info:
                recommendations.append(f"📊 Effect size is {effect_info}")
            
            # Method-specific insights
            if "bootstrap" in best_test_name:
                recommendations.append("✅ Result confirmed by bootstrap resampling - highly reliable")
            elif "welch" in best_test_name:
                recommendations.append("🔧 Welch's method used - robust to unequal variances")
        else:
            recommendations.append("❌ **No significant difference detected** between groups")
            
            # Check for low power
            if group_info.get("min_group_size", 0) < 30:
                recommendations.append("🔍 Small sample sizes may limit statistical power")
        
        # Data quality insights
        if not group_info.get("balanced_groups", True):
            recommendations.append("⚖️ Unbalanced groups detected - robust methods recommended")
        
        # Sample size recommendations
        min_size = group_info.get("min_group_size", 0)
        if min_size < 10:
            recommendations.append("📈 Very small groups - consider collecting more data")
        elif min_size >= 100:
            recommendations.append("💪 Large sample sizes provide good statistical power")
        
        return recommendations[:5]

    # --------------------------- Main Analysis Method ---------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig, 
                categorical_col: str = None, target_cols: List[str] = None) -> Dict[str, Any]:
        """
        Modern categorical group analysis with streamlined SOTA methods
        
        Args:
            data: DataFrame containing the data
            categorical_col: Name of categorical column (auto-detected if None)
            target_cols: List of numeric columns to analyze (all numeric if None)
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Auto-detect categorical column if needed
            if categorical_col is None:
                categorical_col = self._auto_detect_categorical_column(data)
                if categorical_col is None:
                    raise ValueError("No suitable categorical column found")
            
            if not self._is_suitable_categorical_column(data, categorical_col):
                raise ValueError(f"Column '{categorical_col}' not suitable for group analysis")
            
            # Prepare target columns
            if target_cols is None:
                target_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                target_cols = [col for col in target_cols if col != categorical_col]
            
            if not target_cols:
                raise ValueError("No numeric columns found for analysis")
            
            # Prepare data
            grouped_data, group_info = self._prepare_group_data(data, categorical_col, target_cols)
            
            # Analysis results
            analysis_results = {
                "group_info": group_info,
                "variable_analyses": {},
                "categorical_associations": {},
                "summary": {},
                "recommendations": []
            }
            
            # Analyze each numeric variable
            for target_col in target_cols:
                if target_col == categorical_col:
                    continue
                
                # Smart assumption testing
                assumptions = self._smart_assumption_testing(grouped_data, target_col)
                
                # Run statistical tests
                test_results = self._run_statistical_tests(grouped_data, target_col, assumptions)
                
                # Descriptive statistics
                descriptives = self._compute_descriptives(grouped_data, target_col)
                
                # Generate recommendations
                recommendations = self._generate_recommendations(test_results, assumptions, group_info)
                
                analysis_results["variable_analyses"][target_col] = {
                    "descriptive_statistics": descriptives,
                    "assumption_tests": assumptions,
                    "statistical_tests": test_results,
                    "recommendations": recommendations
                }
            
            # Test categorical associations
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            for cat_col in categorical_cols:
                if cat_col != categorical_col and cat_col not in target_cols:
                    try:
                        assoc_results = self._categorical_association_tests(data, categorical_col, cat_col)
                        if assoc_results and "error" not in assoc_results:
                            analysis_results["categorical_associations"][cat_col] = assoc_results
                    except Exception:
                        continue
            
            # Generate overall summary
            significant_variables = []
            for var_name, var_results in analysis_results["variable_analyses"].items():
                test_results = var_results.get("statistical_tests", {})
                for test_name, result in test_results.items():
                    if isinstance(result, dict) and result.get("significant", False):
                        significant_variables.append(var_name)
                        break
            
            analysis_results["summary"] = {
                "total_variables_tested": len(analysis_results["variable_analyses"]),
                "significant_variables": significant_variables,
                "n_significant": len(significant_variables),
                "categorical_associations_tested": len(analysis_results["categorical_associations"]),
                "groups_analyzed": group_info["groups"],
                "n_groups": group_info["n_groups"],
                "total_samples": group_info["total_samples"],
                "analysis_type": "modern_group_comparison",
                "auto_detected_categorical": categorical_col
            }
            
            # Overall recommendations
            if significant_variables:
                analysis_results["recommendations"].append(
                    f"🎯 **{len(significant_variables)} variables** show significant differences between {categorical_col} groups"
                )
                analysis_results["recommendations"].append(
                    f"📊 Variables with differences: {', '.join(significant_variables)}"
                )
            else:
                analysis_results["recommendations"].append(
                    f"❌ **No significant differences** found between {categorical_col} groups"
                )
            
            if not group_info.get("balanced_groups", True):
                analysis_results["recommendations"].append(
                    "⚖️ **Unbalanced groups** - interpret results cautiously"
                )
            
            # Method used summary
            analysis_results["recommendations"].append(
                "🔬 **Modern methods used**: Welch's tests, bootstrap resampling, robust non-parametrics"
            )
            
            return analysis_results
            
        except Exception as e:
            return {
                "error": f"Categorical group analysis failed: {e}",
                "analysis_type": "modern_group_comparison"
            }
