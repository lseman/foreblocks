import warnings
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    anderson,
    bartlett,
    chi2_contingency,
    f_oneway,
    fisher_exact,
    jarque_bera,
    kruskal,
    levene,
    mannwhitneyu,
    normaltest,
    shapiro,
    trim_mean,
    ttest_ind,
)
from sklearn.preprocessing import LabelEncoder

# Remove the import that's causing issues
from .foreminer_aux import *

# Try to import newer scipy features with fallbacks
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




class CategoricalGroupAnalyzer:
    """SOTA analysis for testing differences between categorical groups with comprehensive statistical methods"""

    @property
    def name(self) -> str:
        return "categorical_groups"

    def __init__(self):
        # Performance and reliability thresholds
        self.min_group_size = 5          # Minimum samples per group for reliable testing
        self.max_groups = 20             # Maximum groups to analyze (performance)
        self.small_sample_threshold = 30 # Threshold for small sample corrections
        self.large_sample_threshold = 1000 # Threshold for asymptotic methods
        self.bootstrap_iterations = 1000 # Bootstrap resampling iterations
        self.alpha = 0.05               # Default significance level

    # --------------------------- Auto-Detection Helper Methods ---------------------------
    def _auto_detect_categorical_column(self, data: pd.DataFrame) -> Optional[str]:
        """Automatically detect the best categorical column for group analysis"""
        
        # Get potential categorical columns
        categorical_candidates = []
        
        # 1. Object and category dtype columns
        for col in data.select_dtypes(include=['object', 'category']).columns:
            if self._is_suitable_categorical_column(data, col):
                categorical_candidates.append((col, self._score_categorical_column(data, col)))
        
        # 2. Low-cardinality numeric columns (might be encoded categories)
        for col in data.select_dtypes(include=[np.number]).columns:
            unique_count = data[col].nunique()
            if 2 <= unique_count <= self.max_groups:
                if self._is_suitable_categorical_column(data, col):
                    categorical_candidates.append((col, self._score_categorical_column(data, col)))
        
        if not categorical_candidates:
            return None
        
        # Sort by score and return the best one
        categorical_candidates.sort(key=lambda x: x[1], reverse=True)
        return categorical_candidates[0][0]
    
    def _is_suitable_categorical_column(self, data: pd.DataFrame, col: str) -> bool:
        """Check if a column is suitable for categorical group analysis"""
        try:
            # Check unique value count
            unique_count = data[col].nunique()
            if unique_count < 2 or unique_count > self.max_groups:
                return False
            
            # Check group sizes
            value_counts = data[col].value_counts()
            min_group_size = value_counts.min()
            
            # Each group must have at least minimum required samples
            if min_group_size < self.min_group_size:
                return False
            
            # Check for too much missing data
            missing_pct = data[col].isnull().sum() / len(data) * 100
            if missing_pct > 50:  # More than 50% missing
                return False
            
            return True
            
        except Exception:
            return False
    
    def _score_categorical_column(self, data: pd.DataFrame, col: str) -> float:
        """Score a categorical column for its suitability (higher = better)"""
        try:
            score = 0.0
            
            # 1. Number of groups (prefer 2-5 groups)
            unique_count = data[col].nunique()
            if unique_count == 2:
                score += 10  # Binary is often most interpretable
            elif 3 <= unique_count <= 5:
                score += 8   # Good range
            elif 6 <= unique_count <= 10:
                score += 5   # Acceptable
            else:
                score += 2   # Too many groups
            
            # 2. Group balance (prefer balanced groups)
            value_counts = data[col].value_counts()
            max_size = value_counts.max()
            min_size = value_counts.min()
            balance_ratio = min_size / max_size if max_size > 0 else 0
            score += balance_ratio * 10  # 0-10 points for balance
            
            # 3. Sample sizes (prefer larger minimum group size)
            min_group_size = value_counts.min()
            if min_group_size >= 100:
                score += 5
            elif min_group_size >= 50:
                score += 4
            elif min_group_size >= 30:
                score += 3
            elif min_group_size >= 10:
                score += 2
            else:
                score += 1
            
            # 4. Missing data penalty
            missing_pct = data[col].isnull().sum() / len(data) * 100
            score -= missing_pct / 10  # Lose 1 point per 10% missing
            
            # 5. Data type bonus
            if data[col].dtype in ['object', 'category']:
                score += 3  # Bonus for explicit categorical types
            
            # 6. Interpretable names bonus (heuristic)
            if any(keyword in col.lower() for keyword in 
                   ['gender', 'sex', 'type', 'category', 'class', 'group', 'status', 
                    'level', 'grade', 'treatment', 'condition', 'department']):
                score += 2
            
            return max(0, score)  # Ensure non-negative
            
        except Exception:
            return 0.0

    # --------------------------- Data Preparation ---------------------------
    def _prepare_group_data(
        self, data: pd.DataFrame, categorical_col: str, target_cols: List[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare and validate data for group analysis"""
        
        if categorical_col not in data.columns:
            raise ValueError(f"Categorical column '{categorical_col}' not found in data")
        
        # Remove missing values
        clean_data = data[[categorical_col] + (target_cols or [])].dropna()
        
        if len(clean_data) == 0:
            raise ValueError("No valid data after removing missing values")
        
        # Get group information
        group_info = {
            "categorical_column": categorical_col,
            "total_samples": len(clean_data),
            "original_samples": len(data),
            "missing_samples": len(data) - len(clean_data),
            "missing_percentage": (len(data) - len(clean_data)) / len(data) * 100
        }
        
        # Analyze groups
        group_counts = clean_data[categorical_col].value_counts()
        groups = group_counts.index.tolist()
        
        # Filter out groups that are too small
        valid_groups = [group for group in groups if group_counts[group] >= self.min_group_size]
        
        if len(valid_groups) < 2:
            raise ValueError(f"Need at least 2 groups with minimum {self.min_group_size} samples each")
        
        if len(valid_groups) > self.max_groups:
            # Keep top N groups by size
            valid_groups = group_counts.head(self.max_groups).index.tolist()
            group_info["groups_truncated"] = True
            group_info["truncated_count"] = len(groups) - len(valid_groups)
        else:
            group_info["groups_truncated"] = False
        
        # Filter data to valid groups
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
        
        # Prepare grouped data
        grouped_data = {}
        for group in valid_groups:
            group_mask = clean_data[categorical_col] == group
            grouped_data[group] = clean_data[group_mask]
        
        return grouped_data, group_info

    # --------------------------- Normality and Assumptions Testing ---------------------------
    def _test_assumptions(self, grouped_data: Dict[str, pd.DataFrame], numeric_col: str) -> Dict[str, Any]:
        """Comprehensive assumption testing for statistical methods"""
        assumptions = {
            "normality_tests": {},
            "homogeneity_tests": {},
            "sample_characteristics": {},
            "recommended_methods": []
        }
        
        # Extract numeric data for each group
        group_arrays = {}
        for group_name, group_df in grouped_data.items():
            if numeric_col in group_df.columns:
                group_data = group_df[numeric_col].dropna().values
                if len(group_data) > 0:
                    group_arrays[group_name] = group_data
        
        if len(group_arrays) < 2:
            return assumptions
        
        # Sample characteristics
        all_data = np.concatenate(list(group_arrays.values()))
        assumptions["sample_characteristics"] = {
            "total_samples": len(all_data),
            "n_groups": len(group_arrays),
            "smallest_group": min(len(arr) for arr in group_arrays.values()),
            "largest_group": max(len(arr) for arr in group_arrays.values()),
            "overall_mean": float(np.mean(all_data)),
            "overall_std": float(np.std(all_data, ddof=1)),
            "overall_skewness": float(stats.skew(all_data)),
            "overall_kurtosis": float(stats.kurtosis(all_data))
        }
        
        # Normality tests for each group
        normality_results = {}
        all_normal = True
        
        for group_name, group_data in group_arrays.items():
            if len(group_data) < 3:
                continue
                
            group_normal = {}
            
            # Shapiro-Wilk (best for small samples)
            if len(group_data) <= 5000:
                try:
                    stat, p_val = shapiro(group_data)
                    group_normal["shapiro_wilk"] = {
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "normal": p_val > self.alpha
                    }
                    if p_val <= self.alpha:
                        all_normal = False
                except Exception:
                    pass
            
            # Jarque-Bera (good for larger samples)
            if len(group_data) >= 20:
                try:
                    stat, p_val = jarque_bera(group_data)
                    group_normal["jarque_bera"] = {
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "normal": p_val > self.alpha
                    }
                    if p_val <= self.alpha:
                        all_normal = False
                except Exception:
                    pass
            
            # Anderson-Darling (most powerful)
            try:
                result = anderson(group_data, dist='norm')
                critical_value = result.critical_values[2]  # 5% significance level
                group_normal["anderson_darling"] = {
                    "statistic": float(result.statistic),
                    "critical_value": float(critical_value),
                    "normal": result.statistic < critical_value
                }
                if result.statistic >= critical_value:
                    all_normal = False
            except Exception:
                pass
                
            normality_results[group_name] = group_normal
        
        assumptions["normality_tests"] = normality_results
        assumptions["all_groups_normal"] = all_normal
        
        # Homogeneity of variance tests
        group_data_list = list(group_arrays.values())
        
        if len(group_data_list) >= 2:
            # Levene's test (robust to non-normality)
            try:
                stat, p_val = levene(*group_data_list)
                assumptions["homogeneity_tests"]["levene"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "equal_variances": p_val > self.alpha
                }
            except Exception:
                pass
            
            # Bartlett's test (assumes normality)
            if all_normal:
                try:
                    stat, p_val = bartlett(*group_data_list)
                    assumptions["homogeneity_tests"]["bartlett"] = {
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "equal_variances": p_val > self.alpha
                    }
                except Exception:
                    pass
        
        # Method recommendations based on assumptions
        sample_size = assumptions["sample_characteristics"]["smallest_group"]
        equal_variances = assumptions["homogeneity_tests"].get("levene", {}).get("equal_variances", True)
        
        if all_normal and equal_variances:
            assumptions["recommended_methods"].append("anova")
            assumptions["recommended_methods"].append("t_test")
        
        if not all_normal or sample_size < 30:
            assumptions["recommended_methods"].append("kruskal_wallis")
            assumptions["recommended_methods"].append("mann_whitney")
        
        if not equal_variances:
            assumptions["recommended_methods"].append("welch_anova")
            assumptions["recommended_methods"].append("welch_t_test")
        
        assumptions["recommended_methods"].append("bootstrap")
        assumptions["recommended_methods"].append("permutation")
        
        return assumptions

    # --------------------------- Parametric Tests ---------------------------
    def _parametric_tests(self, grouped_data: Dict[str, pd.DataFrame], numeric_col: str, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive parametric statistical tests"""
        results = {}
        
        # Extract numeric data for each group
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
        
        # One-way ANOVA
        if len(group_arrays) >= 2:
            try:
                stat, p_val = f_oneway(*group_data_list)
                
                # Effect size (eta-squared)
                all_data = np.concatenate(group_data_list)
                ss_total = np.sum((all_data - np.mean(all_data))**2)
                
                group_means = [np.mean(arr) for arr in group_data_list]
                group_sizes = [len(arr) for arr in group_data_list]
                overall_mean = np.mean(all_data)
                
                ss_between = sum(n * (mean - overall_mean)**2 for n, mean in zip(group_sizes, group_means))
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                results["anova"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "eta_squared": float(eta_squared),
                    "effect_size": self._interpret_effect_size(eta_squared, "eta_squared"),
                    "method": "One-way ANOVA",
                    "assumption_met": "anova" in assumptions.get("recommended_methods", [])
                }
                
            except Exception as e:
                results["anova"] = {"error": str(e)}
        
        # Welch's ANOVA (unequal variances)
        if len(group_arrays) >= 2:
            try:
                # Manual Welch's ANOVA implementation
                k = len(group_data_list)
                ni = np.array([len(group) for group in group_data_list])
                yi = np.array([np.mean(group) for group in group_data_list])
                si2 = np.array([np.var(group, ddof=1) for group in group_data_list])
                wi = ni / si2
                
                y_weighted = np.sum(wi * yi) / np.sum(wi)
                numerator = np.sum(wi * (yi - y_weighted)**2)
                
                # Approximate degrees of freedom
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
                    "assumption_met": "welch_anova" in assumptions.get("recommended_methods", [])
                }
                
            except Exception as e:
                results["welch_anova"] = {"error": str(e)}
        
        # Pairwise t-tests (for 2 groups or post-hoc)
        if len(group_arrays) == 2:
            group1_data, group2_data = group_data_list[0], group_data_list[1]
            group1_name, group2_name = group_names[0], group_names[1]
            
                    # Standard t-test
            try:
                stat, p_val = ttest_ind(group1_data, group2_data, equal_var=True)
                
                # Cohen's d effect size
                pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                                    (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                                   (len(group1_data) + len(group2_data) - 2))
                cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std if pooled_std > 0 else 0
                
                results["t_test"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "cohens_d": float(cohens_d),
                    "effect_size": self._interpret_effect_size(abs(cohens_d), "cohens_d"),
                    "groups_compared": [group1_name, group2_name],
                    "method": "Independent t-test (equal variances)",
                    "assumption_met": "t_test" in assumptions.get("recommended_methods", [])
                }
                
            except Exception as e:
                results["t_test"] = {"error": str(e)}
            
            # Welch's t-test (unequal variances)
            try:
                stat, p_val = ttest_ind(group1_data, group2_data, equal_var=False)
                
                results["welch_t_test"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "groups_compared": [group1_name, group2_name],
                    "method": "Welch's t-test (unequal variances)",
                    "assumption_met": "welch_t_test" in assumptions.get("recommended_methods", [])
                }
                
            except Exception as e:
                results["welch_t_test"] = {"error": str(e)}
        
        elif len(group_arrays) > 2:
            # Post-hoc pairwise comparisons with Bonferroni correction
            pairwise_results = {}
            n_comparisons = len(group_names) * (len(group_names) - 1) // 2
            bonferroni_alpha = self.alpha / n_comparisons
            
            for i, group1_name in enumerate(group_names):
                for j, group2_name in enumerate(group_names[i+1:], i+1):
                    group1_data = group_arrays[group1_name]
                    group2_data = group_arrays[group2_name]
                    
                    try:
                        stat, p_val = ttest_ind(group1_data, group2_data, equal_var=True)
                        
                        # Cohen's d
                        pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                                            (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                                           (len(group1_data) + len(group2_data) - 2))
                        cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std if pooled_std > 0 else 0
                        
                        pairwise_results[f"{group1_name}_vs_{group2_name}"] = {
                            "statistic": float(stat),
                            "p_value": float(p_val),
                            "p_value_bonferroni": float(min(p_val * n_comparisons, 1.0)),
                            "significant": p_val < bonferroni_alpha,
                            "cohens_d": float(cohens_d),
                            "effect_size": self._interpret_effect_size(abs(cohens_d), "cohens_d")
                        }
                        
                    except Exception:
                        continue
            
            if pairwise_results:
                results["pairwise_t_tests"] = {
                    "comparisons": pairwise_results,
                    "n_comparisons": n_comparisons,
                    "bonferroni_alpha": float(bonferroni_alpha),
                    "method": "Pairwise t-tests with Bonferroni correction"
                }
        
        return results

    # --------------------------- Non-parametric Tests ---------------------------
    def _nonparametric_tests(self, grouped_data: Dict[str, pd.DataFrame], numeric_col: str) -> Dict[str, Any]:
        """Comprehensive non-parametric statistical tests"""
        results = {}
        
        # Extract numeric data for each group
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
        
        # Kruskal-Wallis H-test
        if len(group_arrays) >= 2:
            try:
                stat, p_val = kruskal(*group_data_list)
                
                # Effect size (epsilon-squared)
                all_data = np.concatenate(group_data_list)
                n_total = len(all_data)
                k = len(group_data_list)
                
                epsilon_squared = (stat - k + 1) / (n_total - k) if n_total > k else 0
                epsilon_squared = max(0, epsilon_squared)  # Ensure non-negative
                
                results["kruskal_wallis"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "epsilon_squared": float(epsilon_squared),
                    "effect_size": self._interpret_effect_size(epsilon_squared, "epsilon_squared"),
                    "method": "Kruskal-Wallis H-test",
                    "n_groups": k,
                    "total_n": n_total
                }
                
            except Exception as e:
                results["kruskal_wallis"] = {"error": str(e)}
        
        # Mann-Whitney U test (for 2 groups)
        if len(group_arrays) == 2:
            group1_data, group2_data = group_data_list[0], group_data_list[1]
            group1_name, group2_name = group_names[0], group_names[1]
            
            try:
                stat, p_val = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                
                # Effect size (rank-biserial correlation)
                n1, n2 = len(group1_data), len(group2_data)
                rank_biserial = 1 - (2 * stat) / (n1 * n2)
                
                results["mann_whitney"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": p_val < self.alpha,
                    "rank_biserial_correlation": float(rank_biserial),
                    "effect_size": self._interpret_effect_size(abs(rank_biserial), "rank_biserial"),
                    "groups_compared": [group1_name, group2_name],
                    "method": "Mann-Whitney U test"
                }
                
            except Exception as e:
                results["mann_whitney"] = {"error": str(e)}
        
        elif len(group_arrays) > 2:
            # Post-hoc pairwise Mann-Whitney tests with Bonferroni correction
            pairwise_results = {}
            n_comparisons = len(group_names) * (len(group_names) - 1) // 2
            bonferroni_alpha = self.alpha / n_comparisons
            
            for i, group1_name in enumerate(group_names):
                for j, group2_name in enumerate(group_names[i+1:], i+1):
                    group1_data = group_arrays[group1_name]
                    group2_data = group_arrays[group2_name]
                    
                    try:
                        stat, p_val = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                        
                        # Effect size
                        n1, n2 = len(group1_data), len(group2_data)
                        rank_biserial = 1 - (2 * stat) / (n1 * n2)
                        
                        pairwise_results[f"{group1_name}_vs_{group2_name}"] = {
                            "statistic": float(stat),
                            "p_value": float(p_val),
                            "p_value_bonferroni": float(min(p_val * n_comparisons, 1.0)),
                            "significant": p_val < bonferroni_alpha,
                            "rank_biserial_correlation": float(rank_biserial),
                            "effect_size": self._interpret_effect_size(abs(rank_biserial), "rank_biserial")
                        }
                        
                    except Exception:
                        continue
            
            if pairwise_results:
                results["pairwise_mann_whitney"] = {
                    "comparisons": pairwise_results,
                    "n_comparisons": n_comparisons,
                    "bonferroni_alpha": float(bonferroni_alpha),
                    "method": "Pairwise Mann-Whitney tests with Bonferroni correction"
                }
        
        return results

    # --------------------------- SOTA Modern Methods ---------------------------
    def _modern_statistical_methods(self, grouped_data: Dict[str, pd.DataFrame], numeric_col: str) -> Dict[str, Any]:
        """State-of-the-art modern statistical methods"""
        results = {}
        
        # Extract numeric data for each group
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
        
        # Bootstrap hypothesis testing
        if HAS_BOOTSTRAP:
            try:
                def bootstrap_mean_diff_scipy(data1, data2):
                    """Bootstrap test using scipy.stats.bootstrap (if available)"""
                    def statistic(x, y):
                        return np.mean(x) - np.mean(y)
                    
                    # Use scipy bootstrap if available
                    rng = np.random.default_rng(42)
                    res = bootstrap((data1, data2), statistic, n_resamples=self.bootstrap_iterations,
                                  paired=False, random_state=rng)
                    
                    observed_diff = statistic(data1, data2)
                    p_value = 2 * min(
                        np.mean(res.bootstrap_distribution >= abs(observed_diff)),
                        np.mean(res.bootstrap_distribution <= -abs(observed_diff))
                    )
                    
                    return {
                        "observed_difference": float(observed_diff),
                        "p_value": float(p_value),
                        "significant": p_value < self.alpha,
                        "bootstrap_iterations": self.bootstrap_iterations,
                        "method": "Bootstrap test for difference in means (scipy)"
                    }
                
                if len(group_arrays) == 2:
                    boot_result = bootstrap_mean_diff_scipy(group_data_list[0], group_data_list[1])
                    boot_result["groups_compared"] = group_names
                    results["bootstrap_test"] = boot_result
                
            except Exception as e:
                # Fallback to manual bootstrap
                pass
        
        # Manual bootstrap implementation (fallback or when scipy.bootstrap not available)
        if "bootstrap_test" not in results:
            try:
                def bootstrap_mean_diff_manual(data1, data2, n_bootstrap=self.bootstrap_iterations):
                    """Manual bootstrap test for difference in means"""
                    observed_diff = np.mean(data1) - np.mean(data2)
                    
                    # Pool the data under null hypothesis
                    pooled_data = np.concatenate([data1, data2])
                    n1, n2 = len(data1), len(data2)
                    
                    bootstrap_diffs = []
                    np.random.seed(42)  # For reproducibility
                    for _ in range(n_bootstrap):
                        # Resample under null hypothesis
                        resampled = np.random.choice(pooled_data, size=len(pooled_data), replace=True)
                        boot_sample1 = resampled[:n1]
                        boot_sample2 = resampled[n1:n1+n2]
                        boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
                        bootstrap_diffs.append(boot_diff)
                    
                    bootstrap_diffs = np.array(bootstrap_diffs)
                    p_value = 2 * min(
                        np.mean(bootstrap_diffs >= abs(observed_diff)),
                        np.mean(bootstrap_diffs <= -abs(observed_diff))
                    )
                    
                    return {
                        "observed_difference": float(observed_diff),
                        "p_value": float(p_value),
                        "significant": p_value < self.alpha,
                        "bootstrap_iterations": n_bootstrap,
                        "method": "Bootstrap test for difference in means (manual)"
                    }
                
                if len(group_arrays) == 2:
                    boot_result = bootstrap_mean_diff_manual(group_data_list[0], group_data_list[1])
                    boot_result["groups_compared"] = group_names
                    results["bootstrap_test"] = boot_result
                
            except Exception as e:
                results["bootstrap_test"] = {"error": str(e)}
        
        # Permutation test
        if HAS_PERMUTATION_TEST:
            try:
                def permutation_test_scipy(data1, data2):
                    """Permutation test using scipy.stats.permutation_test (if available)"""
                    def statistic(x, y):
                        return np.mean(x) - np.mean(y)
                    
                    rng = np.random.default_rng(42)
                    res = permutation_test((data1, data2), statistic, n_resamples=self.bootstrap_iterations,
                                         random_state=rng)
                    
                    observed_diff = statistic(data1, data2)
                    
                    return {
                        "observed_difference": float(observed_diff),
                        "p_value": float(res.pvalue),
                        "significant": res.pvalue < self.alpha,
                        "permutation_iterations": self.bootstrap_iterations,
                        "method": "Permutation test for difference in means (scipy)"
                    }
                
                if len(group_arrays) == 2:
                    perm_result = permutation_test_scipy(group_data_list[0], group_data_list[1])
                    perm_result["groups_compared"] = group_names
                    results["permutation_test"] = perm_result
                
            except Exception as e:
                # Fallback to manual permutation
                pass
        
        # Manual permutation test (fallback or when scipy.permutation_test not available)
        if "permutation_test" not in results:
            try:
                def permutation_test_manual(data1, data2, n_permutations=self.bootstrap_iterations):
                    """Manual permutation test for difference in means"""
                    observed_diff = np.mean(data1) - np.mean(data2)
                    
                    # Combine all data
                    combined_data = np.concatenate([data1, data2])
                    n1 = len(data1)
                    
                    permutation_diffs = []
                    np.random.seed(42)  # For reproducibility
                    for _ in range(n_permutations):
                        # Randomly permute the combined data
                        permuted = np.random.permutation(combined_data)
                        perm_group1 = permuted[:n1]
                        perm_group2 = permuted[n1:]
                        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
                        permutation_diffs.append(perm_diff)
                    
                    permutation_diffs = np.array(permutation_diffs)
                    p_value = np.mean(np.abs(permutation_diffs) >= abs(observed_diff))
                    
                    return {
                        "observed_difference": float(observed_diff),
                        "p_value": float(p_value),
                        "significant": p_value < self.alpha,
                        "permutation_iterations": n_permutations,
                        "method": "Permutation test for difference in means (manual)"
                    }
                
                if len(group_arrays) == 2:
                    perm_result = permutation_test_manual(group_data_list[0], group_data_list[1])
                    perm_result["groups_compared"] = group_names
                    results["permutation_test"] = perm_result
                
            except Exception as e:
                results["permutation_test"] = {"error": str(e)}
        
        # Bayesian estimation (using bootstrap for credible intervals)
        try:
            def bayesian_estimation(data_groups, n_bootstrap=self.bootstrap_iterations):
                """Bayesian-inspired estimation with credible intervals"""
                group_results = {}
                
                for group_name, group_data in zip(group_names, data_groups):
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        boot_sample = np.random.choice(group_data, size=len(group_data), replace=True)
                        bootstrap_means.append(np.mean(boot_sample))
                    
                    bootstrap_means = np.array(bootstrap_means)
                    
                    group_results[group_name] = {
                        "posterior_mean": float(np.mean(bootstrap_means)),
                        "credible_interval_95": [
                            float(np.percentile(bootstrap_means, 2.5)),
                            float(np.percentile(bootstrap_means, 97.5))
                        ],
                        "posterior_std": float(np.std(bootstrap_means))
                    }
                
                # Probability of differences
                if len(data_groups) == 2:
                    diff_probs = []
                    for _ in range(n_bootstrap):
                        boot1 = np.random.choice(data_groups[0], size=len(data_groups[0]), replace=True)
                        boot2 = np.random.choice(data_groups[1], size=len(data_groups[1]), replace=True)
                        diff_probs.append(np.mean(boot1) > np.mean(boot2))
                    
                    group_results["difference_analysis"] = {
                        "prob_group1_greater": float(np.mean(diff_probs)),
                        "prob_group2_greater": float(1 - np.mean(diff_probs)),
                        "groups_compared": group_names
                    }
                
                return group_results
            
            bayesian_result = bayesian_estimation(group_data_list)
            results["bayesian_estimation"] = {
                "group_estimates": bayesian_result,
                "method": "Bayesian-inspired estimation with bootstrap credible intervals"
            }
            
        except Exception as e:
            results["bayesian_estimation"] = {"error": str(e)}
        
        # Robust statistics (using trimmed means and Winsorized variance)
        try:
            def robust_group_comparison(data_groups, trim_percent=0.1):
                """Robust comparison using trimmed statistics"""
                
                robust_stats = {}
                
                for group_name, group_data in zip(group_names, data_groups):
                    # Trimmed mean (remove extreme values)
                    sorted_data = np.sort(group_data)
                    n = len(sorted_data)
                    trim_count = int(n * trim_percent / 2)
                    
                    if trim_count > 0:
                        trimmed_data = sorted_data[trim_count:-trim_count]
                        trimmed_mean = np.mean(trimmed_data)
                    else:
                        trimmed_mean = np.mean(sorted_data)
                    
                    # Winsorized variance
                    if trim_count > 0:
                        winsorized = sorted_data.copy()
                        winsorized[:trim_count] = sorted_data[trim_count]
                        winsorized[-trim_count:] = sorted_data[-(trim_count+1)]
                    else:
                        winsorized = sorted_data
                    
                    winsorized_var = np.var(winsorized, ddof=1)
                    
                    robust_stats[group_name] = {
                        "robust_mean": float(trimmed_mean),
                        "robust_variance": float(winsorized_var),
                        "robust_std": float(np.sqrt(winsorized_var)),
                        "trim_percentage": trim_percent * 100
                    }
                
                # Robust effect size between groups
                if len(data_groups) == 2:
                    robust_diff = robust_stats[group_names[0]]["robust_mean"] - robust_stats[group_names[1]]["robust_mean"]
                    pooled_robust_std = np.sqrt(
                        (robust_stats[group_names[0]]["robust_variance"] + 
                         robust_stats[group_names[1]]["robust_variance"]) / 2
                    )
                    robust_effect_size = robust_diff / pooled_robust_std if pooled_robust_std > 0 else 0
                    
                    robust_stats["robust_comparison"] = {
                        "robust_difference": float(robust_diff),
                        "robust_effect_size": float(robust_effect_size),
                        "effect_size_interpretation": self._interpret_effect_size(abs(robust_effect_size), "cohens_d"),
                        "groups_compared": group_names
                    }
                
                return robust_stats
            
            robust_result = robust_group_comparison(group_data_list)
            results["robust_statistics"] = {
                "group_statistics": robust_result,
                "method": "Robust statistics with trimmed means and Winsorized variance"
            }
            
        except Exception as e:
            results["robust_statistics"] = {"error": str(e)}
        
        return results

    # --------------------------- Categorical vs Categorical Tests ---------------------------
    def _categorical_association_tests(self, data: pd.DataFrame, cat_col1: str, cat_col2: str) -> Dict[str, Any]:
        """Tests for association between two categorical variables"""
        results = {}
        
        # Create contingency table
        try:
            contingency_table = pd.crosstab(data[cat_col1], data[cat_col2])
            
            if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return {"error": "Insufficient data for categorical association tests"}
            
            # Chi-square test
            chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
            
            # Effect sizes
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            # Phi coefficient (for 2x2 tables)
            phi = None
            if contingency_table.shape == (2, 2):
                phi = np.sqrt(chi2_stat / n)
            
            results["chi_square"] = {
                "statistic": float(chi2_stat),
                "p_value": float(chi2_p),
                "degrees_of_freedom": int(dof),
                "significant": chi2_p < self.alpha,
                "cramers_v": float(cramers_v),
                "phi": float(phi) if phi is not None else None,
                "effect_size": self._interpret_effect_size(cramers_v, "cramers_v"),
                "method": "Chi-square test of independence"
            }
            
            # Fisher's exact test (for 2x2 tables)
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
            
            # Contingency table information
            results["contingency_info"] = {
                "table_shape": contingency_table.shape,
                "total_observations": int(n),
                "contingency_table": contingency_table.to_dict(),
                "expected_frequencies": expected.tolist(),
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
        
        elif effect_type == "eta_squared" or effect_type == "epsilon_squared":
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
        
        else:
            return "unknown"

    # --------------------------- Descriptive Statistics ---------------------------
    def _comprehensive_descriptives(self, grouped_data: Dict[str, pd.DataFrame], numeric_col: str) -> Dict[str, Any]:
        """Comprehensive descriptive statistics for each group"""
        descriptives = {}
        
        for group_name, group_df in grouped_data.items():
            if numeric_col not in group_df.columns:
                continue
                
            group_data = group_df[numeric_col].dropna()
            
            if len(group_data) == 0:
                continue
            
            # Basic statistics
            stats_dict = {
                "n": int(len(group_data)),
                "mean": float(group_data.mean()),
                "median": float(group_data.median()),
                "std": float(group_data.std(ddof=1)),
                "var": float(group_data.var(ddof=1)),
                "min": float(group_data.min()),
                "max": float(group_data.max()),
                "range": float(group_data.max() - group_data.min()),
                "q1": float(group_data.quantile(0.25)),
                "q3": float(group_data.quantile(0.75)),
                "iqr": float(group_data.quantile(0.75) - group_data.quantile(0.25)),
                "skewness": float(group_data.skew()),
                "kurtosis": float(group_data.kurtosis()),
                "sem": float(group_data.sem())  # Standard error of mean
            }
            
            # Confidence interval for mean (95%)
            t_critical = stats.t.ppf(0.975, len(group_data) - 1)
            margin_error = t_critical * stats_dict["sem"]
            stats_dict["ci_95_lower"] = float(stats_dict["mean"] - margin_error)
            stats_dict["ci_95_upper"] = float(stats_dict["mean"] + margin_error)
            
            # Robust statistics
            try:
                stats_dict["mad"] = float(np.median(np.abs(group_data - group_data.median())))  # Median absolute deviation
                stats_dict["trimmed_mean_10"] = float(trim_mean(group_data, 0.1))
            except Exception:
                stats_dict["mad"] = float(np.median(np.abs(group_data - group_data.median())))
                # Fallback trimmed mean calculation
                sorted_data = np.sort(group_data)
                trim_count = int(len(sorted_data) * 0.05)  # 5% from each end
                if trim_count > 0:
                    trimmed_data = sorted_data[trim_count:-trim_count]
                    stats_dict["trimmed_mean_10"] = float(np.mean(trimmed_data))
                else:
                    stats_dict["trimmed_mean_10"] = float(np.mean(sorted_data))
            
            descriptives[group_name] = stats_dict
        
        return descriptives

    # --------------------------- Smart Recommendations ---------------------------
    def _generate_recommendations(
        self, test_results: Dict[str, Any], assumptions: Dict[str, Any], 
        group_info: Dict[str, Any], descriptives: Dict[str, Any]
    ) -> List[str]:
        """Generate intelligent recommendations based on analysis results"""
        recommendations = []
        
        # Find the most reliable significant test
        significant_tests = []
        for test_name, result in test_results.items():
            if isinstance(result, dict) and result.get("significant", False) and "error" not in result:
                method_type = result.get("method", test_name)
                p_value = result.get("p_value", 1.0)
                assumption_met = result.get("assumption_met", True)
                
                # Score tests by reliability
                reliability_score = 1.0
                if not assumption_met:
                    reliability_score *= 0.7
                if "bootstrap" in test_name or "permutation" in test_name:
                    reliability_score *= 1.2  # Bonus for robust methods
                if "welch" in test_name:
                    reliability_score *= 1.1  # Bonus for robust to assumptions
                
                significant_tests.append((test_name, method_type, p_value, reliability_score, result))
        
        # Sort by reliability and significance
        significant_tests.sort(key=lambda x: (x[3], -x[2]), reverse=True)
        
        if significant_tests:
            best_test = significant_tests[0]
            test_name, method_type, p_value, _, result = best_test
            
            recommendations.append(f"🎯 **Significant difference detected** using {method_type} (p = {p_value:.6f})")
            
            # Effect size interpretation
            effect_size_info = ""
            if "cohens_d" in result:
                effect_size_info = f" with {result.get('effect_size', 'unknown')} effect size (d = {result['cohens_d']:.3f})"
            elif "eta_squared" in result:
                effect_size_info = f" with {result.get('effect_size', 'unknown')} effect size (η² = {result['eta_squared']:.3f})"
            elif "cramers_v" in result:
                effect_size_info = f" with {result.get('effect_size', 'unknown')} effect size (V = {result['cramers_v']:.3f})"
            
            if effect_size_info:
                recommendations.append(f"📊 Effect size is {effect_size_info}")
            
            # Method-specific insights
            if "bootstrap" in test_name or "permutation" in test_name:
                recommendations.append("✅ Result confirmed by robust resampling method - highly reliable")
            elif not result.get("assumption_met", True):
                recommendations.append("⚠️ Consider robust methods as statistical assumptions may be violated")
        
        else:
            recommendations.append("❌ **No significant difference detected** between groups")
            
            # Check if this might be due to low power
            smallest_group = group_info.get("min_group_size", 0)
            if smallest_group < 30:
                recommendations.append("🔍 Small sample sizes may limit power to detect differences")
        
        # Sample size and balance recommendations
        if not group_info.get("balanced_groups", True):
            recommendations.append("⚖️ Unbalanced groups detected - consider robust methods or resampling")
        
        if group_info.get("min_group_size", 0) < 10:
            recommendations.append("📈 Very small groups - consider collecting more data for reliable results")
        
        # Assumption violations
        if not assumptions.get("all_groups_normal", True):
            recommendations.append("📊 Non-normal distributions detected - non-parametric tests recommended")
        
        levene_result = assumptions.get("homogeneity_tests", {}).get("levene", {})
        if not levene_result.get("equal_variances", True):
            recommendations.append("🔧 Unequal variances detected - use Welch's tests or robust methods")
        
        # Practical significance
        if descriptives:
            group_means = [stats["mean"] for stats in descriptives.values()]
            if len(group_means) >= 2:
                mean_diff_pct = abs(max(group_means) - min(group_means)) / np.mean(group_means) * 100
                if mean_diff_pct < 5:
                    recommendations.append("🤔 Difference may be statistically significant but practically small")
                elif mean_diff_pct > 25:
                    recommendations.append("💡 Large practical difference detected - likely meaningful impact")
        
        return recommendations[:6]  # Limit to most important recommendations

    # --------------------------- Main Analysis Method ---------------------------
    def analyze(self, data: pd.DataFrame,  config: AnalysisConfig, categorical_col: str = None, 
                target_cols: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of differences between categorical groups
        
        Args:
            data: DataFrame containing the data
            categorical_col: Name of the categorical column (e.g., 'Gender'). If None, will auto-detect.
            target_cols: List of numeric columns to analyze (if None, analyzes all numeric columns)
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            # Auto-detect categorical column if not provided
            if categorical_col is None:
                categorical_col = self._auto_detect_categorical_column(data)
                if categorical_col is None:
                    raise ValueError("No suitable categorical column found. Please specify categorical_col parameter.")
            
            # Validate categorical column
            if categorical_col not in data.columns:
                raise ValueError(f"Categorical column '{categorical_col}' not found in data")
            
            # Check if the column is actually categorical
            if not self._is_suitable_categorical_column(data, categorical_col):
                raise ValueError(f"Column '{categorical_col}' is not suitable for group analysis. "
                               f"It should have 2-{self.max_groups} distinct values with adequate sample sizes.")
            
            # Prepare target columns
            if target_cols is None:
                target_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                # Remove the categorical column if it's numeric
                target_cols = [col for col in target_cols if col != categorical_col]
            
            if not target_cols:
                raise ValueError("No numeric columns found for analysis")
            
            # Prepare group data
            grouped_data, group_info = self._prepare_group_data(data, categorical_col, target_cols)
            
            # Results container
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
                
                variable_results = {
                    "descriptive_statistics": self._comprehensive_descriptives(grouped_data, target_col),
                    "assumption_tests": self._test_assumptions(grouped_data, target_col),
                    "parametric_tests": {},
                    "nonparametric_tests": {},
                    "modern_methods": {},
                    "recommendations": []
                }
                
                # Run assumption tests
                assumptions = variable_results["assumption_tests"]
                
                # Run parametric tests
                if assumptions.get("recommended_methods"):
                    variable_results["parametric_tests"] = self._parametric_tests(grouped_data, target_col, assumptions)
                
                # Run non-parametric tests
                variable_results["nonparametric_tests"] = self._nonparametric_tests(grouped_data, target_col)
                
                # Run modern methods
                variable_results["modern_methods"] = self._modern_statistical_methods(grouped_data, target_col)
                
                # Combine all test results for recommendations
                all_tests = {}
                all_tests.update(variable_results["parametric_tests"])
                all_tests.update(variable_results["nonparametric_tests"])
                all_tests.update(variable_results["modern_methods"])
                
                # Generate recommendations for this variable
                variable_results["recommendations"] = self._generate_recommendations(
                    all_tests, assumptions, group_info, variable_results["descriptive_statistics"]
                )
                
                analysis_results["variable_analyses"][target_col] = variable_results
            
            # Test associations between categorical variables
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
                # Check if any test shows significance
                all_tests = {}
                all_tests.update(var_results.get("parametric_tests", {}))
                all_tests.update(var_results.get("nonparametric_tests", {}))
                all_tests.update(var_results.get("modern_methods", {}))
                
                for test_name, result in all_tests.items():
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
                "analysis_type": "comprehensive_group_comparison",
                "auto_detected_categorical": categorical_col
            }
            
            # Generate overall recommendations
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
                    "⚖️ **Unbalanced groups** - results should be interpreted cautiously"
                )
            
            return analysis_results
            
        except Exception as e:
            return {
                "error": f"Categorical group analysis failed: {e}",
                "analysis_type": "comprehensive_group_comparison"
            }
