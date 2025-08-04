import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

# Core libraries
from scipy.stats import (
    anderson,
    entropy,
    jarque_bera,
    ks_2samp,
    kurtosis,
    normaltest,
    shapiro,
    skew,
)
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

# Optional imports with graceful fallbacks
OPTIONAL_IMPORTS = {}
for lib_name, import_stmt in [
    ('phik', 'import phik'),
    ('networkx', 'import networkx as nx'),
    ('umap', 'from umap import UMAP'),
    ('hdbscan', 'from hdbscan import HDBSCAN'),
    ('hierarchy', 'from scipy.cluster.hierarchy import dendrogram, linkage, fcluster')
]:
    try:
        exec(import_stmt)
        OPTIONAL_IMPORTS[lib_name] = True
    except ImportError:
        OPTIONAL_IMPORTS[lib_name] = False


def requires_library(lib_name: str):
    """Decorator to check if optional library is available"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OPTIONAL_IMPORTS.get(lib_name, False):
                raise ImportError(f"{lib_name} not available. Install with: pip install {lib_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


@dataclass
class AnalysisConfig:
    """Centralized configuration for analysis parameters"""
    confidence_level: float = 0.05
    max_clusters: int = 10
    sample_size_threshold: int = 5000
    correlation_threshold: float = 0.7
    outlier_contamination: float = 0.1
    time_series_min_periods: int = 24
    plot_style: str = 'seaborn-v0_8'
    figure_size: Tuple[int, int] = (12, 8)
    random_state: int = 42


class CorrelationStrategy(ABC):
    """Strategy pattern for correlation methods"""
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class PearsonCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.corr(method='pearson')


class SpearmanCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.corr(method='spearman')


class MutualInfoCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        n_features = len(data.columns)
        mi_matrix = np.zeros((n_features, n_features))
        
        for i, col_i in enumerate(data.columns):
            for j, col_j in enumerate(data.columns):
                if i == j:
                    mi_matrix[i, j] = 1.0
                else:
                    X = data[[col_i]].fillna(data[col_i].median())
                    y = data[col_j].fillna(data[col_j].median())
                    mi_matrix[i, j] = mutual_info_regression(X, y, random_state=42)[0]
        
        return pd.DataFrame(mi_matrix, index=data.columns, columns=data.columns)


class DistanceCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        def dcorr(X: np.ndarray, Y: np.ndarray) -> float:
            """Distance correlation computation"""
            n = len(X)
            a = squareform(pdist(X.reshape(-1, 1), 'euclidean'))
            b = squareform(pdist(Y.reshape(-1, 1), 'euclidean'))
            
            A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
            
            dcov2_xy = (A * B).sum() / (n * n)
            dcov2_xx = (A * A).sum() / (n * n)
            dcov2_yy = (B * B).sum() / (n * n)
            
            return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx * dcov2_yy)) if dcov2_xx * dcov2_yy > 0 else 0
        
        n_features = len(data.columns)
        matrix = np.zeros((n_features, n_features))
        
        for i, col_i in enumerate(data.columns):
            for j, col_j in enumerate(data.columns):
                matrix[i, j] = dcorr(data[col_i].dropna().values, data[col_j].dropna().values)
        
        return pd.DataFrame(matrix, index=data.columns, columns=data.columns)


@requires_library('phik')
class PhiKCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.phik_matrix()


class DatasetAnalyzer:
    """State-of-the-art dataset analyzer with comprehensive statistical analysis"""
    
    def __init__(self, df: pd.DataFrame, time_col: Optional[str] = None, 
                 config: Optional[AnalysisConfig] = None, verbose: bool = True):
        self.df = df.copy()
        self.time_col = time_col
        self.config = config or AnalysisConfig()
        self.verbose = verbose
        
        # Initialize internal state
        self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self._cache = {}
        
        # Setup matplotlib style
        plt.style.use(self.config.plot_style)
        
        self._setup_time_index()
        self._log(f"Initialized analyzer with {len(self._numeric_cols)} numeric and {len(self._categorical_cols)} categorical features")
    
    def _setup_time_index(self) -> None:
        """Setup time index if time column is provided"""
        if self.time_col and self.time_col in self.df.columns:
            self.df.set_index(self.time_col, inplace=True)
            self.df.index = pd.to_datetime(self.df.index)
    
    def _log(self, message: str) -> None:
        """Centralized logging utility"""
        if self.verbose:
            print(f"🔍 {message}")
    
    @lru_cache(maxsize=32)
    def _get_clean_numeric_data(self) -> pd.DataFrame:
        """Get cached clean numeric data"""
        return self.df[self._numeric_cols].dropna()
    
    # === DISTRIBUTION ANALYSIS ===
    def analyze_distributions(self) -> pd.DataFrame:
        """Comprehensive distribution analysis with statistical tests"""
        summary_data = []
        
        for col in self._numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) < 8:
                continue
            
            # Basic statistics
            stats_dict = {
                'feature': col,
                'count': len(col_data),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'range': col_data.max() - col_data.min(),
                'cv': col_data.std() / abs(col_data.mean()) if col_data.mean() != 0 else np.inf,
                'skewness': skew(col_data),
                'kurtosis': kurtosis(col_data),
                'entropy': entropy(np.histogram(col_data, bins=30)[0] + 1e-10)
            }
            
            # Quartiles and IQR
            q1, q2, q3 = col_data.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            stats_dict.update({
                'q1': q1, 'median': q2, 'q3': q3, 'iqr': iqr
            })
            
            # Outlier detection
            outliers_iqr = ((col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)).sum()
            stats_dict.update({
                'outliers_count': outliers_iqr,
                'outliers_pct': outliers_iqr / len(col_data) * 100
            })
            
            # Normality tests
            if len(col_data) >= 8:
                _, p_norm = normaltest(col_data)
                sample_size = min(5000, len(col_data))
                _, p_shapiro = shapiro(col_data.sample(sample_size, random_state=self.config.random_state))
                
                stats_dict.update({
                    'normaltest_p': p_norm,
                    'shapiro_p': p_shapiro,
                    'is_gaussian': p_norm > self.config.confidence_level and abs(stats_dict['skewness']) < 1,
                    'is_heavy_tailed': abs(stats_dict['kurtosis']) > 3,
                    'is_skewed': abs(stats_dict['skewness']) > 1
                })
            
            summary_data.append(stats_dict)
        
        return pd.DataFrame(summary_data)
    
    def plot_distributions(self, ncols: int = 3, figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot feature distributions with enhanced visualization"""
        if not self._numeric_cols:
            self._log("No numeric columns to plot")
            return
        
        n_features = len(self._numeric_cols)
        nrows = (n_features + ncols - 1) // ncols
        figsize = figsize or (5 * ncols, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for i, col in enumerate(self._numeric_cols):
            col_data = self.df[col].dropna()
            
            # Enhanced histogram with KDE
            sns.histplot(col_data, kde=True, stat='density', alpha=0.7, ax=axes[i])
            
            # Add mean and median lines
            axes[i].axvline(col_data.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
            axes[i].axvline(col_data.median(), color='orange', linestyle='--', alpha=0.8, label='Median')
            
            axes[i].set_title(f'{col}\n(Skew: {skew(col_data):.2f}, Kurt: {kurtosis(col_data):.2f})', fontsize=10)
            axes[i].legend(fontsize=8)
        
        # Hide unused subplots
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # === CORRELATION ANALYSIS ===
    def analyze_correlations(self, methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Compute multiple correlation measures using strategy pattern"""
        methods = methods or ['pearson', 'spearman', 'mutual_info']
        
        strategies = {
            'pearson': PearsonCorrelation(),
            'spearman': SpearmanCorrelation(),
            'mutual_info': MutualInfoCorrelation(),
            'distance': DistanceCorrelation(),
            'phik': PhiKCorrelation() if OPTIONAL_IMPORTS['phik'] else None
        }
        
        clean_data = self._get_clean_numeric_data()
        if clean_data.empty:
            return {}
        
        results = {}
        for method in methods:
            if method in strategies and strategies[method] is not None:
                self._log(f"Computing {method} correlation...")
                try:
                    results[method] = strategies[method].compute(clean_data)
                except Exception as e:
                    self._log(f"Failed to compute {method} correlation: {e}")
        
        return results
    
    def plot_correlation_matrix(self, methods: Optional[List[str]] = None, 
                              figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot multiple correlation heatmaps in a grid"""
        correlations = self.analyze_correlations(methods)
        if not correlations:
            return
        
        n_methods = len(correlations)
        figsize = figsize or (6 * n_methods, 5)
        
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        axes = np.atleast_1d(axes)
        
        for i, (method, corr_matrix) in enumerate(correlations.items()):
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                       cmap="RdBu_r", center=0, ax=axes[i], 
                       cbar_kws={'shrink': 0.8})
            axes[i].set_title(f"{method.replace('_', ' ').title()} Correlation")
        
        plt.tight_layout()
        plt.show()
    
    @requires_library('networkx')
    def plot_correlation_network(self, method: str = 'pearson', 
                               threshold: Optional[float] = None) -> None:
        """Create interactive correlation network graph"""
        threshold = threshold or self.config.correlation_threshold
        correlations = self.analyze_correlations([method])
        
        if method not in correlations:
            self._log(f"Method {method} not available")
            return
        
        corr_matrix = correlations[method]
        G = nx.Graph()
        
        # Add edges above threshold
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], 
                             weight=corr_val, correlation=corr_matrix.iloc[i, j])
        
        if not G.edges():
            self._log(f"No correlations above threshold {threshold}")
            return
        
        plt.figure(figsize=self.config.figure_size)
        pos = nx.spring_layout(G, k=1, iterations=50, seed=self.config.random_state)
        
        # Draw edges with thickness proportional to correlation
        edges = G.edges()
        weights = [G[u][v]['weight'] * 4 for u, v in edges]
        colors = ['red' if G[u][v]['correlation'] < 0 else 'blue' for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color=colors)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1500, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(f"Correlation Network ({method.title()}, threshold={threshold})")
        plt.axis('off')
        plt.show()
    
    # === OUTLIER DETECTION ===
    def detect_outliers(self, methods: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Multiple outlier detection algorithms"""
        methods = methods or ['isolation_forest', 'elliptic_envelope']
        clean_data = self._get_clean_numeric_data()
        
        if clean_data.empty:
            return {}
        
        # Scale data for outlier detection
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(clean_data)
        
        detectors = {
            'isolation_forest': IsolationForest(
                contamination=self.config.outlier_contamination,
                random_state=self.config.random_state
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=self.config.outlier_contamination
            )
        }
        
        results = {}
        for method in methods:
            if method in detectors:
                self._log(f"Detecting outliers with {method}...")
                try:
                    outliers = detectors[method].fit_predict(scaled_data) == -1
                    results[method] = outliers
                    outlier_pct = outliers.sum() / len(outliers) * 100
                    self._log(f"Found {outliers.sum()} outliers ({outlier_pct:.1f}%)")
                except Exception as e:
                    self._log(f"Failed outlier detection with {method}: {e}")
        
        return results
    
    def plot_outliers_pca(self, method: str = 'isolation_forest') -> None:
        """Visualize outliers using PCA projection"""
        outliers_dict = self.detect_outliers([method])
        
        if method not in outliers_dict:
            self._log(f"Outlier method {method} not available")
            return
        
        clean_data = self._get_clean_numeric_data()
        outliers = outliers_dict[method]
        
        # PCA projection
        pca = PCA(n_components=2, random_state=self.config.random_state)
        pca_data = pca.fit_transform(StandardScaler().fit_transform(clean_data))
        
        plt.figure(figsize=self.config.figure_size)
        colors = ['red' if x else 'blue' for x in outliers]
        labels = ['Outlier' if x else 'Normal' for x in outliers]
        
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.6, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'Outlier Detection: {method.replace("_", " ").title()}')
        
        # Create legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Normal'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Outlier')]
        plt.legend(handles=handles)
        
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # === CLUSTERING ANALYSIS ===
    def analyze_clusters(self, methods: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Comprehensive clustering analysis with multiple algorithms"""
        methods = methods or ['kmeans', 'spectral']
        if OPTIONAL_IMPORTS['hdbscan']:
            methods.append('hdbscan')
        
        clean_data = self._get_clean_numeric_data()
        if clean_data.empty:
            return {}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)
        
        results = {}
        
        for method in methods:
            self._log(f"Running {method} clustering...")
            
            try:
                if method == 'kmeans':
                    results[method] = self._kmeans_analysis(scaled_data)
                elif method == 'spectral':
                    results[method] = self._spectral_analysis(scaled_data)
                elif method == 'hdbscan':
                    results[method] = self._hdbscan_analysis(scaled_data)
            except Exception as e:
                self._log(f"Failed clustering with {method}: {e}")
        
        return results
    
    def _kmeans_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """K-means clustering with optimal k selection"""
        scores = []
        k_range = range(2, min(self.config.max_clusters + 1, len(data) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=self.config.random_state)
            labels = kmeans.fit_predict(data)
            
            scores.append({
                'k': k,
                'silhouette': silhouette_score(data, labels),
                'calinski_harabasz': calinski_harabasz_score(data, labels),
                'davies_bouldin': davies_bouldin_score(data, labels),
                'inertia': kmeans.inertia_
            })
        
        # Select best k based on silhouette score
        best_k = max(scores, key=lambda x: x['silhouette'])['k']
        final_model = KMeans(n_clusters=best_k, n_init='auto', random_state=self.config.random_state)
        final_labels = final_model.fit_predict(data)
        
        return {
            'labels': final_labels,
            'model': final_model,
            'scores': scores,
            'best_k': best_k,
            'centers': final_model.cluster_centers_
        }
    
    def _spectral_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Spectral clustering analysis"""
        n_clusters = min(5, len(data) // 10)  # Adaptive cluster count
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=self.config.random_state)
        labels = spectral.fit_predict(data)
        
        return {
            'labels': labels,
            'model': spectral,
            'n_clusters': n_clusters,
            'silhouette': silhouette_score(data, labels)
        }
    
    @requires_library('hdbscan')
    def _hdbscan_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """HDBSCAN clustering analysis"""
        min_cluster_size = max(5, len(data) // 50)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = hdbscan_model.fit_predict(data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        result = {
            'labels': labels,
            'model': hdbscan_model,
            'n_clusters': n_clusters,
            'noise_points': (labels == -1).sum()
        }
        
        if n_clusters > 1:
            # Only calculate silhouette if we have multiple clusters
            valid_labels = labels[labels != -1]
            valid_data = data[labels != -1]
            if len(set(valid_labels)) > 1:
                result['silhouette'] = silhouette_score(valid_data, valid_labels)
        
        return result
    
    def plot_clusters(self, methods: Optional[List[str]] = None) -> None:
        """Visualize clustering results using PCA"""
        clustering_results = self.analyze_clusters(methods)
        if not clustering_results:
            return
        
        clean_data = self._get_clean_numeric_data()
        
        # PCA for visualization
        pca = PCA(n_components=2, random_state=self.config.random_state)
        pca_data = pca.fit_transform(StandardScaler().fit_transform(clean_data))
        
        n_methods = len(clustering_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        axes = np.atleast_1d(axes)
        
        for i, (method, result) in enumerate(clustering_results.items()):
            scatter = axes[i].scatter(pca_data[:, 0], pca_data[:, 1], 
                                    c=result['labels'], cmap='viridis', alpha=0.7, s=50)
            
            title = f"{method.title()}\n"
            if 'best_k' in result:
                title += f"k={result['best_k']}"
            elif 'n_clusters' in result:
                title += f"clusters={result['n_clusters']}"
            
            if 'silhouette' in result:
                title += f"\nSilhouette: {result['silhouette']:.3f}"
            
            axes[i].set_title(title)
            axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            
            plt.colorbar(scatter, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
    
    # === DIMENSIONALITY REDUCTION ===
    def reduce_dimensions(self, methods: Optional[List[str]] = None, 
                         n_components: int = 2) -> Dict[str, np.ndarray]:
        """Multiple dimensionality reduction techniques"""
        methods = methods or ['pca', 'ica', 'tsne']
        if OPTIONAL_IMPORTS['umap']:
            methods.append('umap')
        
        clean_data = self._get_clean_numeric_data()
        if clean_data.empty:
            return {}
        
        # Sample if dataset is too large
        if len(clean_data) > self.config.sample_size_threshold:
            sample_idx = np.random.choice(len(clean_data), self.config.sample_size_threshold, 
                                        replace=False)
            clean_data = clean_data.iloc[sample_idx]
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)
        
        results = {}
        reducers = {
            'pca': PCA(n_components=n_components, random_state=self.config.random_state),
            'ica': FastICA(n_components=n_components, random_state=self.config.random_state),
            'tsne': TSNE(n_components=n_components, random_state=self.config.random_state, perplexity=30)
        }
        
        if OPTIONAL_IMPORTS['umap']:
            reducers['umap'] = UMAP(n_components=n_components, random_state=self.config.random_state)
        
        for method in methods:
            if method in reducers:
                self._log(f"Computing {method.upper()}...")
                try:
                    results[method] = reducers[method].fit_transform(scaled_data)
                except Exception as e:
                    self._log(f"Failed dimensionality reduction with {method}: {e}")
        
        return results
    
    def plot_reductions(self, methods: Optional[List[str]] = None) -> None:
        """Plot dimensionality reduction results"""
        reductions = self.reduce_dimensions(methods)
        if not reductions:
            return
        
        n_methods = len(reductions)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.atleast_1d(axes).flatten()
        
        for i, (method, reduced_data) in enumerate(reductions.items()):
            scatter = axes[i].scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                    alpha=0.6, s=30, c=range(len(reduced_data)), 
                                    cmap='viridis')
            axes[i].set_title(f"{method.upper()}")
            axes[i].set_xlabel("Component 1")
            axes[i].set_ylabel("Component 2")
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # === TIME SERIES ANALYSIS ===
    def analyze_stationarity(self) -> pd.DataFrame:
        """Comprehensive stationarity testing"""
        if not self._numeric_cols:
            return pd.DataFrame()
        
        results = []
        for col in self._numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) < 10:
                continue
            
            try:
                # Augmented Dickey-Fuller test
                adf_result = adfuller(col_data, autolag='AIC')
                
                # KPSS test
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    kpss_result = kpss(col_data, regression='c', nlags='auto')
                
                results.append({
                    'feature': col,
                    'adf_statistic': adf_result[0],
                    'adf_pvalue': adf_result[1],
                    'kpss_statistic': kpss_result[0],
                    'kpss_pvalue': kpss_result[1],
                    'is_stationary_adf': adf_result[1] < self.config.confidence_level,
                    'is_stationary_kpss': kpss_result[1] > self.config.confidence_level,
                    'is_stationary': (adf_result[1] < self.config.confidence_level and 
                                    kpss_result[1] > self.config.confidence_level)
                })
            except Exception as e:
                self._log(f"Stationarity test failed for {col}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def decompose_series(self, column: str, period: Optional[int] = None) -> Any:
        """STL decomposition with automatic period detection"""
        if column not in self._numeric_cols:
            raise ValueError(f"Column {column} not found in numeric columns")
        
        series = self.df[column].dropna()
        if len(series) < 24:
            raise ValueError("Series too short for decomposition (minimum 24 points)")
        
        # Auto-detect period if not provided
        if period is None:
            period = min(max(2, len(series) // 10), 24)
        
        stl = STL(series, period=period, seasonal=7)
        decomposition = stl.fit()
        
        # Enhanced plotting
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=axes[0], title=f'Original Series: {column}')
        decomposition.trend.plot(ax=axes[1], title='Trend', color='orange')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='green')  
        decomposition.resid.plot(ax=axes[3], title='Residual', color='red')
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
    
    def plot_autocorrelations(self, column: str, lags: int = 40) -> None:
        """Enhanced ACF and PACF plots"""
        if column not in self._numeric_cols:
            raise ValueError(f"Column {column} not found in numeric columns")
        
        series = self.df[column].dropna()
        if len(series) < lags + 10:
            lags = max(10, len(series) // 3)
        
        # Compute ACF and PACF
        acf_vals = acf(series, nlags=lags, fft=True)
        pacf_vals = pacf(series, nlags=lags, method='ols')
        
        # Confidence intervals (approximate)
        n = len(series)
        ci = 1.96 / np.sqrt(n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ACF plot
        ax1.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axhline(y=ci, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
        ax1.fill_between(range(len(acf_vals)), -ci, ci, alpha=0.2, color='red')
        ax1.set_title(f'Autocorrelation Function - {column}')
        ax1.set_xlabel('Lags')
        ax1.set_ylabel('ACF')
        
        # PACF plot
        ax2.stem(range(len(pacf_vals)), pacf_vals, basefmt=" ")
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=ci, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
        ax2.fill_between(range(len(pacf_vals)), -ci, ci, alpha=0.2, color='red')
        ax2.set_title(f'Partial Autocorrelation Function - {column}')
        ax2.set_xlabel('Lags')
        ax2.set_ylabel('PACF')
        
        plt.tight_layout()
        plt.show()
    
    def suggest_lag_features(self, max_lags: int = 20, top_k: int = 3) -> Dict[str, List[int]]:
        """Intelligent lag feature suggestions based on PACF"""
        suggestions = {}
        
        for col in self._numeric_cols:
            series = self.df[col].dropna()
            if len(series) < max_lags + 10:
                continue
            
            try:
                pacf_vals = pacf(series, nlags=max_lags, method='ols')[1:]  # Skip lag 0
                
                # Find significant lags (above confidence interval)
                n = len(series)
                ci = 1.96 / np.sqrt(n)
                significant_lags = []
                
                for i, val in enumerate(pacf_vals, 1):
                    if abs(val) > ci:
                        significant_lags.append((i, abs(val)))
                
                # Sort by absolute PACF value and take top k
                significant_lags.sort(key=lambda x: x[1], reverse=True)
                top_lags = [lag for lag, _ in significant_lags[:top_k]]
                
                if top_lags:
                    suggestions[col] = top_lags
                    
            except Exception as e:
                self._log(f"Lag suggestion failed for {col}: {e}")
        
        return suggestions
    
    # === ADVANCED PATTERN DETECTION ===
    def detect_patterns(self) -> Dict[str, Any]:
        """Comprehensive pattern detection and anomaly analysis"""
        patterns = {}
        
        # Feature archetypes
        patterns['feature_types'] = self._classify_features()
        
        # Relationship patterns
        patterns['relationships'] = self._analyze_relationships()
        
        # Anomaly patterns
        patterns['anomalies'] = self._detect_anomaly_patterns()
        
        # Distribution fitting
        patterns['distributions'] = self._fit_distributions()
        
        # Temporal patterns (if time series)
        if self.time_col:
            patterns['temporal'] = self._analyze_temporal_patterns()
        
        return patterns
    
    def _classify_features(self) -> Dict[str, List[str]]:
        """Classify features into different archetypes"""
        dist_summary = self.analyze_distributions()
        
        classification = {
            'gaussian': [],
            'log_normal_candidates': [],
            'bounded': [],  # 0-1 or 0-100
            'count_like': [],  # Non-negative integers
            'continuous': [],
            'highly_skewed': [],
            'heavy_tailed': []
        }
        
        for _, row in dist_summary.iterrows():
            feature = row['feature']
            
            # Basic classifications
            if row.get('is_gaussian', False):
                classification['gaussian'].append(feature)
            
            if row.get('is_heavy_tailed', False):
                classification['heavy_tailed'].append(feature)
            
            if abs(row['skewness']) > 2:
                classification['highly_skewed'].append(feature)
            
            # Log-normal candidates
            if row['skewness'] > 1 and row['mean'] > 0:
                classification['log_normal_candidates'].append(feature)
            
            # Analyze actual data for bounded/count features
            col_data = self.df[feature].dropna()
            min_val, max_val = col_data.min(), col_data.max()
            
            # Bounded features
            if (0 <= min_val <= max_val <= 1) or (0 <= min_val <= max_val <= 100 and 
                                                 col_data.nunique() <= 101):
                classification['bounded'].append(feature)
            
            # Count-like features
            elif min_val >= 0 and all(col_data % 1 == 0):
                classification['count_like'].append(feature)
            
            else:
                classification['continuous'].append(feature)
        
        return classification
    
    def _analyze_relationships(self) -> Dict[str, Any]:
        """Advanced relationship pattern mining"""
        if len(self._numeric_cols) < 2:
            return {}
        
        correlations = self.analyze_correlations(['pearson', 'spearman', 'mutual_info'])
        
        patterns = {}
        
        # Non-linear relationships
        if 'pearson' in correlations and 'spearman' in correlations:
            pearson = correlations['pearson']
            spearman = correlations['spearman']
            
            nonlinear_pairs = []
            for i in range(len(pearson.columns)):
                for j in range(i + 1, len(pearson.columns)):
                    p_corr = abs(pearson.iloc[i, j])
                    s_corr = abs(spearman.iloc[i, j])
                    
                    # Strong non-linearity indicator
                    if s_corr > p_corr + 0.2 and s_corr > 0.5:
                        nonlinear_pairs.append({
                            'feature1': pearson.columns[i],
                            'feature2': pearson.columns[j],
                            'pearson': p_corr,
                            'spearman': s_corr,
                            'nonlinearity_score': s_corr - p_corr
                        })
            
            patterns['nonlinear'] = sorted(nonlinear_pairs, 
                                         key=lambda x: x['nonlinearity_score'], 
                                         reverse=True)[:10]
        
        # Complex relationships via mutual information
        if 'mutual_info' in correlations and 'pearson' in correlations:
            mi_matrix = correlations['mutual_info']
            pearson = correlations['pearson']
            
            complex_pairs = []
            for i in range(len(mi_matrix.columns)):
                for j in range(i + 1, len(mi_matrix.columns)):
                    mi_val = mi_matrix.iloc[i, j]
                    p_val = abs(pearson.iloc[i, j])
                    
                    # High MI but low Pearson suggests complex relationship
                    if mi_val > 0.3 and p_val < 0.3:
                        complex_pairs.append({
                            'feature1': mi_matrix.columns[i],
                            'feature2': mi_matrix.columns[j],
                            'mutual_info': mi_val,
                            'pearson': p_val,
                            'complexity_score': mi_val - p_val
                        })
            
            patterns['complex'] = sorted(complex_pairs,
                                       key=lambda x: x['complexity_score'],
                                       reverse=True)[:10]
        
        return patterns
    
    def _detect_anomaly_patterns(self) -> Dict[str, Any]:
        """Comprehensive anomaly pattern detection"""
        patterns = {}
        
        # Outlier analysis
        outliers = self.detect_outliers()
        
        if outliers:
            primary_method = list(outliers.keys())[0]
            outlier_mask = outliers[primary_method]
            
            if outlier_mask.sum() > 0:
                clean_data = self._get_clean_numeric_data()
                outlier_data = clean_data[outlier_mask]
                normal_data = clean_data[~outlier_mask]
                
                # Feature contribution to outliers
                contributions = {}
                for col in self._numeric_cols:
                    if col in outlier_data.columns and col in normal_data.columns:
                        outlier_mean = outlier_data[col].mean()
                        normal_mean = normal_data[col].mean()
                        normal_std = normal_data[col].std()
                        
                        if normal_std > 0:
                            z_score = abs(outlier_mean - normal_mean) / normal_std
                            contributions[col] = z_score
                
                patterns['outlier_drivers'] = sorted(contributions.items(),
                                                   key=lambda x: x[1], reverse=True)[:10]
        
        # Distribution change detection (concept drift)
        if len(self.df) > 100:
            patterns['distribution_shifts'] = self._detect_distribution_shifts()
        
        return patterns
    
    def _detect_distribution_shifts(self) -> List[Dict[str, Any]]:
        """Detect potential distribution shifts in the data"""
        shifts = []
        
        for col in self._numeric_cols[:5]:  # Limit for performance
            data = self.df[col].dropna()
            if len(data) < 50:
                continue
            
            # Compare first and last thirds
            n = len(data)
            first_third = data.iloc[:n//3]
            last_third = data.iloc[-n//3:]
            
            try:
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = ks_2samp(first_third, last_third)
                
                if ks_p < 0.05:  # Significant shift
                    shifts.append({
                        'feature': col,
                        'test': 'Kolmogorov-Smirnov',
                        'statistic': ks_stat,
                        'p_value': ks_p,
                        'interpretation': 'Significant distribution change detected',
                        'early_mean': first_third.mean(),
                        'late_mean': last_third.mean(),
                        'mean_change': last_third.mean() - first_third.mean()
                    })
                    
            except Exception as e:
                continue
        
        return shifts
    
    def _fit_distributions(self) -> Dict[str, Dict[str, Any]]:
        """Fit statistical distributions to features"""
        from scipy import stats
        
        distributions = [
            ('normal', stats.norm),
            ('lognormal', stats.lognorm),
            ('exponential', stats.expon),
            ('gamma', stats.gamma),
            ('beta', stats.beta)
        ]
        
        fits = {}
        
        for col in self._numeric_cols[:10]:  # Limit for performance
            data = self.df[col].dropna()
            if len(data) < 30:
                continue
            
            best_fit = None
            best_aic = np.inf
            
            for name, dist in distributions:
                try:
                    # Fit distribution
                    params = dist.fit(data)
                    
                    # Calculate AIC
                    log_likelihood = np.sum(dist.logpdf(data, *params))
                    k = len(params)  # Number of parameters
                    aic = 2 * k - 2 * log_likelihood
                    
                    if aic < best_aic and not np.isnan(aic):
                        best_aic = aic
                        best_fit = {
                            'distribution': name,
                            'parameters': params,
                            'aic': aic,
                            'log_likelihood': log_likelihood
                        }
                        
                except Exception:
                    continue
            
            if best_fit:
                fits[col] = best_fit
        
        return fits
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in time series data"""
        patterns = {}
        
        for col in self._numeric_cols[:5]:  # Limit for performance
            series = self.df[col].dropna()
            if len(series) < 24:
                continue
            
            # Trend analysis
            x = np.arange(len(series))
            slope = np.polyfit(x, series.values, 1)[0]
            trend = 'increasing' if slope > series.std() * 0.01 else 'decreasing' if slope < -series.std() * 0.01 else 'stable'
            
            patterns[f'{col}_trend'] = {
                'direction': trend,
                'slope': slope,
                'slope_normalized': slope / series.std()
            }
            
            # Seasonality detection
            if len(series) >= 24:
                try:
                    # Use STL decomposition
                    period = min(24, len(series) // 4)
                    stl = STL(series, period=period, seasonal=7)
                    decomp = stl.fit()
                    
                    # Measure seasonal strength
                    seasonal_var = np.var(decomp.seasonal)
                    total_var = np.var(series)
                    seasonal_strength = seasonal_var / total_var if total_var > 0 else 0
                    
                    patterns[f'{col}_seasonality'] = {
                        'strength': seasonal_strength,
                        'classification': 'strong' if seasonal_strength > 0.3 else 'moderate' if seasonal_strength > 0.1 else 'weak'
                    }
                    
                except Exception:
                    patterns[f'{col}_seasonality'] = {'strength': 0, 'classification': 'unknown'}
            
            # Volatility clustering (for financial-like data)
            returns = series.pct_change().dropna()
            if len(returns) > 10:
                volatility = returns.rolling(window=min(5, len(returns)//3)).std()
                if len(volatility.dropna()) > 1:
                    vol_autocorr = volatility.dropna().autocorr(lag=1)
                    patterns[f'{col}_volatility_clustering'] = {
                        'autocorr': vol_autocorr,
                        'present': abs(vol_autocorr) > 0.3 if not np.isnan(vol_autocorr) else False
                    }
        
        return patterns
    
    # === FEATURE ENGINEERING SUGGESTIONS ===
    def suggest_feature_engineering(self) -> Dict[str, List[str]]:
        """Generate intelligent feature engineering suggestions"""
        suggestions = {}
        dist_summary = self.analyze_distributions()
        
        # Per-feature transformations
        for _, row in dist_summary.iterrows():
            feature = row['feature']
            transforms = []
            
            # Skewness-based suggestions
            if abs(row['skewness']) > 1.5:
                if row['min'] > 0:
                    transforms.append(f"np.log1p({feature})")
                transforms.extend([
                    f"scipy.stats.boxcox({feature})[0]",
                    f"scipy.stats.yeojohnson({feature})[0]"
                ])
            
            # Outlier handling
            if row['outliers_pct'] > 5:
                transforms.extend([
                    f"scipy.stats.mstats.winsorize({feature}, limits=(0.05, 0.05))",
                    f"RobustScaler().fit_transform({feature}.values.reshape(-1, 1))"
                ])
            
            # Heavy-tailed distributions
            if row.get('is_heavy_tailed', False):
                transforms.append(f"scipy.stats.rankdata({feature})")
            
            # Binning for highly variable features
            if row['cv'] > 2:  # High coefficient of variation
                transforms.append(f"pd.qcut({feature}, q=5, labels=False)")
            
            if transforms:
                suggestions[feature] = transforms[:3]  # Limit suggestions
        
        # Interaction features
        correlations = self.analyze_correlations(['pearson'])
        if 'pearson' in correlations:
            interactions = []
            corr_matrix = correlations['pearson']
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    
                    if 0.3 < corr_val < 0.8:  # Moderate correlation
                        interactions.extend([
                            f"{feat1} * {feat2}",
                            f"{feat1} / {feat2}" if f"{feat2}_zero_protected" not in interactions else None,
                            f"np.sqrt({feat1}**2 + {feat2}**2)"
                        ])
                        
                        if len(interactions) >= 15:  # Limit interactions
                            break
                if len(interactions) >= 15:
                    break
            
            if interactions:
                suggestions['interactions'] = [x for x in interactions if x is not None][:10]
        
        return suggestions

    # === ENHANCED INSIGHTS PRINTING ===
    def print_detailed_insights(self) -> None:
        """Print comprehensive detailed insights with specific recommendations"""
        print("=" * 100)
        print("🔬 COMPREHENSIVE DATASET ANALYSIS WITH DETAILED INSIGHTS")
        print("=" * 100)
        
        # === BASIC OVERVIEW ===
        print(f"\n📊 DATASET OVERVIEW")
        print("-" * 40)
        print(f"   Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"   Numeric features: {len(self._numeric_cols)}")
        print(f"   Categorical features: {len(self._categorical_cols)}")
        print(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        missing_pct = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        print(f"   Missing values: {missing_pct:.2f}% of total dataset")
        
        # === DETAILED DISTRIBUTION INSIGHTS ===
        dist_summary = self.analyze_distributions()
        if not dist_summary.empty:
            print(f"\n📈 DETAILED DISTRIBUTION ANALYSIS")
            print("-" * 40)
            
            gaussian_count = dist_summary['is_gaussian'].sum()
            skewed_count = dist_summary['is_skewed'].sum()
            heavy_tailed_count = dist_summary['is_heavy_tailed'].sum()
            
            print(f"   ✓ Gaussian features: {gaussian_count}/{len(dist_summary)} ({gaussian_count/len(dist_summary)*100:.1f}%)")
            if gaussian_count > 0:
                gaussian_features = dist_summary[dist_summary['is_gaussian']]['feature'].tolist()
                print(f"     → {', '.join(gaussian_features)}")
            
            print(f"   ⚠️  Skewed features: {skewed_count}/{len(dist_summary)} ({skewed_count/len(dist_summary)*100:.1f}%)")
            if skewed_count > 0:
                skewed_features = dist_summary[dist_summary['is_skewed']]['feature'].tolist()
                print(f"     → {', '.join(skewed_features)}")
                
            print(f"   📏 Heavy-tailed features: {heavy_tailed_count}/{len(dist_summary)} ({heavy_tailed_count/len(dist_summary)*100:.1f}%)")
            if heavy_tailed_count > 0:
                heavy_tailed_features = dist_summary[dist_summary['is_heavy_tailed']]['feature'].tolist()
                print(f"     → {', '.join(heavy_tailed_features)}")
            
            # Outlier insights
            high_outlier_features = dist_summary[dist_summary['outliers_pct'] > 10]
            if not high_outlier_features.empty:
                print(f"\n   🎯 HIGH OUTLIER FEATURES (>10% outliers):")
                for _, row in high_outlier_features.iterrows():
                    print(f"     → {row['feature']}: {row['outliers_pct']:.1f}% outliers ({row['outliers_count']} points)")
        
        # === DETAILED CORRELATION INSIGHTS ===
        correlations = self.analyze_correlations(['pearson', 'spearman'])
        if correlations:
            print(f"\n🔗 DETAILED CORRELATION ANALYSIS")
            print("-" * 40)
            
            if 'pearson' in correlations:
                corr_matrix = correlations['pearson']
                
                # Strong positive correlations
                strong_pos_pairs = []
                # Strong negative correlations
                strong_neg_pairs = []
                # Moderate correlations
                moderate_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        
                        if corr_val > 0.7:
                            strong_pos_pairs.append((feat1, feat2, corr_val))
                        elif corr_val < -0.7:
                            strong_neg_pairs.append((feat1, feat2, corr_val))
                        elif 0.4 <= abs(corr_val) <= 0.7:
                            moderate_pairs.append((feat1, feat2, corr_val))
                
                if strong_pos_pairs:
                    print(f"   💚 STRONG POSITIVE CORRELATIONS (r > 0.7):")
                    for feat1, feat2, corr in strong_pos_pairs[:5]:
                        print(f"     → {feat1} ↔ {feat2}: {corr:.3f}")
                        
                if strong_neg_pairs:
                    print(f"   💔 STRONG NEGATIVE CORRELATIONS (r < -0.7):")
                    for feat1, feat2, corr in strong_neg_pairs[:5]:
                        print(f"     → {feat1} ↔ {feat2}: {corr:.3f}")
                        
                if moderate_pairs:
                    print(f"   🟡 MODERATE CORRELATIONS (0.4 ≤ |r| ≤ 0.7): {len(moderate_pairs)} pairs")
                    for feat1, feat2, corr in moderate_pairs[:3]:
                        print(f"     → {feat1} ↔ {feat2}: {corr:.3f}")
        
        # === DETAILED PATTERN INSIGHTS ===
        patterns = self.detect_patterns()
        if patterns:
            print(f"\n🔍 ADVANCED PATTERN DETECTION")
            print("-" * 40)
            
            # Feature type classification
            if 'feature_types' in patterns:
                feature_types = patterns['feature_types']
                print("   📋 FEATURE TYPE CLASSIFICATION:")
                for ftype, features in feature_types.items():
                    if features:
                        type_name = ftype.replace('_', ' ').title()
                        print(f"     → {type_name}: {len(features)} features")
                        if len(features) <= 5:
                            print(f"       ∘ {', '.join(features)}")
                        else:
                            print(f"       ∘ {', '.join(features[:3])} ... (+{len(features)-3} more)")
            
            # Relationship patterns
            if 'relationships' in patterns:
                rel_patterns = patterns['relationships']
                
                if 'nonlinear' in rel_patterns and rel_patterns['nonlinear']:
                    print(f"\n   🌀 NON-LINEAR RELATIONSHIPS DETECTED:")
                    for rel in rel_patterns['nonlinear'][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(f"       ∘ Pearson: {rel['pearson']:.3f}, Spearman: {rel['spearman']:.3f}")
                        print(f"       ∘ Non-linearity score: {rel['nonlinearity_score']:.3f}")
                
                if 'complex' in rel_patterns and rel_patterns['complex']:
                    print(f"\n   🧬 COMPLEX RELATIONSHIPS (High Mutual Info, Low Linear Correlation):")
                    for rel in rel_patterns['complex'][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(f"       ∘ Mutual Info: {rel['mutual_info']:.3f}, Pearson: {rel['pearson']:.3f}")
            
            # Distribution fitting results
            if 'distributions' in patterns and patterns['distributions']:
                print(f"\n   📊 BEST-FIT DISTRIBUTIONS:")
                dist_fits = patterns['distributions']
                for feature, fit_info in list(dist_fits.items())[:5]:
                    print(f"     → {feature}: {fit_info['distribution'].title()} distribution")
                    print(f"       ∘ AIC: {fit_info['aic']:.2f}")
        
        # === DETAILED CLUSTERING INSIGHTS ===
        clustering = self.analyze_clusters()
        if clustering:
            print(f"\n🔍 DETAILED CLUSTERING ANALYSIS")
            print("-" * 40)
            
            for method, result in clustering.items():
                print(f"   🎯 {method.upper()} CLUSTERING:")
                
                if 'best_k' in result:
                    best_score = max(result['scores'], key=lambda x: x['silhouette'])
                    print(f"     → Optimal clusters: {result['best_k']}")
                    print(f"     → Silhouette score: {best_score['silhouette']:.3f}")
                    print(f"     → Calinski-Harabasz: {best_score['calinski_harabasz']:.2f}")
                    
                    # Show cluster sizes
                    labels = result['labels']
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    print(f"     → Cluster sizes: {dict(zip(unique_labels, counts))}")
                    
                elif 'n_clusters' in result:
                    print(f"     → Clusters detected: {result['n_clusters']}")
                    if 'silhouette' in result:
                        print(f"     → Silhouette score: {result['silhouette']:.3f}")
                    if 'noise_points' in result:
                        noise_pct = result['noise_points'] / len(result['labels']) * 100
                        print(f"     → Noise points: {result['noise_points']} ({noise_pct:.1f}%)")
        
        # === DETAILED OUTLIER INSIGHTS ===
        outliers = self.detect_outliers()
        if outliers:
            print(f"\n🎯 DETAILED OUTLIER ANALYSIS")
            print("-" * 40)
            
            for method, outlier_mask in outliers.items():
                outlier_count = outlier_mask.sum()
                outlier_pct = outlier_count / len(outlier_mask) * 100
                method_name = method.replace('_', ' ').title()
                
                print(f"   🔍 {method_name}:")
                print(f"     → Outliers detected: {outlier_count} ({outlier_pct:.1f}%)")
                
                # Show which features contribute most to outliers
                if patterns and 'anomalies' in patterns and 'outlier_drivers' in patterns['anomalies']:
                    drivers = patterns['anomalies']['outlier_drivers']
                    if drivers:
                        print(f"     → Top outlier-driving features:")
                        for feature, z_score in drivers[:3]:
                            print(f"       ∘ {feature}: Z-score = {z_score:.2f}")
        
        # === DETAILED FEATURE ENGINEERING SUGGESTIONS ===
        suggestions = self.suggest_feature_engineering()
        if suggestions:
            print(f"\n🛠️  DETAILED FEATURE ENGINEERING RECOMMENDATIONS")
            print("-" * 40)
            
            # Count different types of suggestions
            transform_count = sum(len(transforms) for k, transforms in suggestions.items() 
                                if k != 'interactions' and isinstance(transforms, list))
            
            print(f"   📊 TRANSFORMATION SUGGESTIONS: {transform_count} total")
            
            # Show specific transformations for problematic features
            feature_suggestions = {k: v for k, v in suggestions.items() if k != 'interactions'}
            if feature_suggestions:
                print("   🔄 RECOMMENDED TRANSFORMATIONS:")
                for feature, transforms in list(feature_suggestions.items())[:5]:
                    print(f"     → {feature}:")
                    for i, transform in enumerate(transforms[:2], 1):
                        print(f"       {i}. {transform}")
            
            # Show interaction suggestions
            if 'interactions' in suggestions:
                interactions = suggestions['interactions']
                print(f"\n   🤝 INTERACTION FEATURES: {len(interactions)} suggested")
                print("   🔗 TOP INTERACTION RECOMMENDATIONS:")
                for i, interaction in enumerate(interactions[:5], 1):
                    print(f"     {i}. {interaction}")
        
        # === TIME SERIES INSIGHTS (if applicable) ===
        if self.time_col:
            print(f"\n⏰ TIME SERIES ANALYSIS")
            print("-" * 40)
            
            try:
                stationarity = self.analyze_stationarity()
                if not stationarity.empty:
                    stationary_count = stationarity['is_stationary'].sum()
                    total_features = len(stationarity)
                    
                    print(f"   📈 STATIONARITY ANALYSIS:")
                    print(f"     → Stationary features: {stationary_count}/{total_features} ({stationary_count/total_features*100:.1f}%)")
                    
                    non_stationary = stationarity[~stationarity['is_stationary']]
                    if not non_stationary.empty:
                        print(f"     → Non-stationary features requiring differencing:")
                        for _, row in non_stationary.iterrows():
                            print(f"       ∘ {row['feature']} (ADF p-value: {row['adf_pvalue']:.4f})")
                
                # Lag suggestions
                lag_suggestions = self.suggest_lag_features()
                if lag_suggestions:
                    print(f"\n   🔄 LAG FEATURE SUGGESTIONS:")
                    for feature, lags in list(lag_suggestions.items())[:5]:
                        print(f"     → {feature}: Create lags {lags}")
                        
                # Temporal patterns
                if patterns and 'temporal' in patterns:
                    temporal_patterns = patterns['temporal']
                    print(f"\n   📊 TEMPORAL PATTERNS DETECTED:")
                    
                    for pattern_key, pattern_info in temporal_patterns.items():
                        if '_trend' in pattern_key:
                            feature = pattern_key.replace('_trend', '')
                            direction = pattern_info['direction']
                            slope = pattern_info['slope_normalized']
                            print(f"     → {feature}: {direction.title()} trend (slope: {slope:.3f})")
                            
                        elif '_seasonality' in pattern_key:
                            feature = pattern_key.replace('_seasonality', '')
                            strength = pattern_info['strength']
                            classification = pattern_info['classification']
                            print(f"     → {feature}: {classification.title()} seasonality (strength: {strength:.3f})")
                            
            except Exception as e:
                print(f"     ⚠️  Time series analysis failed: {e}")
        
        # === DATA QUALITY ASSESSMENT ===
        print(f"\n✅ DATA QUALITY ASSESSMENT")
        print("-" * 40)
        
        completeness = (1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        uniqueness = len(self.df.drop_duplicates()) / len(self.df) * 100
        
        print(f"   📊 Completeness: {completeness:.1f}%")
        if completeness < 95:
            missing_by_col = self.df.isnull().sum().sort_values(ascending=False)
            high_missing = missing_by_col[missing_by_col > 0][:5]
            if not high_missing.empty:
                print("     → Columns with missing values:")
                for col, missing_count in high_missing.items():
                    missing_pct = missing_count / len(self.df) * 100
                    print(f"       ∘ {col}: {missing_count} ({missing_pct:.1f}%)")
        
        print(f"   🔍 Uniqueness: {uniqueness:.1f}%")
        if uniqueness < 100:
            duplicate_count = len(self.df) - len(self.df.drop_duplicates())
            print(f"     → Duplicate rows: {duplicate_count}")
        
        # Cardinality analysis for categorical features
        if self._categorical_cols:
            print(f"   📋 Categorical Feature Cardinality:")
            for col in self._categorical_cols[:5]:
                unique_count = self.df[col].nunique()
                unique_pct = unique_count / len(self.df) * 100
                print(f"     → {col}: {unique_count} unique values ({unique_pct:.1f}%)")
        
        overall_quality = (completeness + uniqueness) / 2
        quality_status = "Excellent" if overall_quality > 95 else "Good" if overall_quality > 85 else "Fair" if overall_quality > 70 else "Poor"
        quality_emoji = "🟢" if overall_quality > 95 else "🟡" if overall_quality > 85 else "🟠" if overall_quality > 70 else "🔴"
        
        print(f"   {quality_emoji} Overall Quality Score: {overall_quality:.1f}% ({quality_status})")
        
        # === ACTIONABLE RECOMMENDATIONS ===
        print(f"\n🚀 ACTIONABLE RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = []
        
        # Distribution-based recommendations
        if not dist_summary.empty:
            high_skew_features = dist_summary[abs(dist_summary['skewness']) > 2]['feature'].tolist()
            if high_skew_features:
                recommendations.append(f"Apply log transformation to highly skewed features: {', '.join(high_skew_features[:3])}")
            
            high_outlier_features = dist_summary[dist_summary['outliers_pct'] > 15]['feature'].tolist()
            if high_outlier_features:
                recommendations.append(f"Consider outlier treatment for: {', '.join(high_outlier_features[:3])}")
        
        # Correlation-based recommendations
        if correlations and 'pearson' in correlations:
            corr_matrix = correlations['pearson']
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                recommendations.append(f"Consider removing redundant features due to high correlation: {high_corr_pairs[0]}")
        
        # Clustering-based recommendations
        if clustering:
            for method, result in clustering.items():
                if 'best_k' in result and result['best_k'] > 1:
                    recommendations.append(f"Dataset shows natural grouping into {result['best_k']} clusters - consider cluster-based analysis")
                    break
        
        # Missing data recommendations
        if completeness < 95:
            recommendations.append("Address missing values through imputation or removal strategies")
        
        # Feature engineering recommendations
        if suggestions:
            if len(feature_suggestions) > 0:
                recommendations.append("Apply suggested transformations to improve feature distributions")
            if 'interactions' in suggestions:
                recommendations.append("Create interaction features to capture non-linear relationships")
        
        # Time series recommendations
        if self.time_col and lag_suggestions:
            recommendations.append("Create lag features for time series forecasting")
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations[:8], 1):
                print(f"   {i}. {rec}")
        else:
            print("   ✅ Dataset appears to be in good shape - no critical recommendations")
        
        # === NEXT STEPS ===
        print(f"\n🎯 SUGGESTED NEXT STEPS")
        print("-" * 40)
        print("   1. Review and implement feature engineering suggestions")
        print("   2. Handle outliers based on domain knowledge")
        print("   3. Apply recommended transformations")
        print("   4. Consider dimensionality reduction if many features")
        print("   5. Use clustering insights for exploratory analysis")
        print("   6. Validate findings with domain experts")
        
        print("\n" + "=" * 100)
    
    # === COMPREHENSIVE REPORTING ===
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        self._log("Generating comprehensive analysis report...")
        
        report = {
            'metadata': {
                'shape': self.df.shape,
                'numeric_features': len(self._numeric_cols),
                'categorical_features': len(self._categorical_cols),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024**2),
                'analysis_timestamp': pd.Timestamp.now()
            }
        }
        
        # Core analyses
        report['distributions'] = self.analyze_distributions()
        report['correlations'] = self.analyze_correlations()
        report['outliers'] = self.detect_outliers()
        report['clusters'] = self.analyze_clusters()
        report['dimensionality'] = self.reduce_dimensions()
        report['patterns'] = self.detect_patterns()
        report['feature_suggestions'] = self.suggest_feature_engineering()
        
        # Time series specific
        if self.time_col:
            report['stationarity'] = self.analyze_stationarity()
            report['lag_suggestions'] = self.suggest_lag_features()
        
        return report
    
    def create_summary_dashboard(self) -> None:
        """Create comprehensive visual dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribution summary
        plt.subplot(3, 4, 1)
        dist_summary = self.analyze_distributions()
        if not dist_summary.empty:
            plt.bar(range(len(dist_summary)), dist_summary['skewness'], alpha=0.7)
            plt.title('Feature Skewness Distribution')
            plt.xticks(range(len(dist_summary)), dist_summary['feature'], rotation=45)
            plt.ylabel('Skewness')
        
        # 2. Correlation heatmap
        plt.subplot(3, 4, 2)
        correlations = self.analyze_correlations(['pearson'])
        if 'pearson' in correlations:
            corr = correlations['pearson']
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, cbar=False, square=True)
            plt.title('Pearson Correlations')
        
        # 3. PCA projection
        plt.subplot(3, 4, 3)
        reductions = self.reduce_dimensions(['pca'])
        if 'pca' in reductions:
            pca_data = reductions['pca']
            plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6, s=30)
            plt.title('PCA Projection')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
        
        # 4. Outlier summary
        plt.subplot(3, 4, 4)
        outliers = self.detect_outliers()
        if outliers:
            method = list(outliers.keys())[0]
            outlier_count = outliers[method].sum()
            normal_count = len(outliers[method]) - outlier_count
            
            plt.pie([normal_count, outlier_count], 
                   labels=['Normal', 'Outliers'], 
                   autopct='%1.1f%%',
                   colors=['skyblue', 'red'])
            plt.title(f'Outlier Detection\n({method})')
        
        # 5. Missing values heatmap
        plt.subplot(3, 4, 5)
        missing_data = self.df.isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data, cbar=True, cmap='viridis')
            plt.title('Missing Values Pattern')
        else:
            plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Missing Values: None')
        
        # 6. Feature types distribution
        plt.subplot(3, 4, 6)
        patterns = self.detect_patterns()
        if 'feature_types' in patterns:
            feature_types = patterns['feature_types']
            type_counts = {k: len(v) for k, v in feature_types.items() if v}
            
            if type_counts:
                plt.bar(type_counts.keys(), type_counts.values(), alpha=0.7)
                plt.title('Feature Type Distribution')
                plt.xticks(rotation=45)
                plt.ylabel('Count')
        
        # 7. Clustering results
        plt.subplot(3, 4, 7)
        clustering = self.analyze_clusters(['kmeans'])
        if 'kmeans' in clustering:
            scores = clustering['kmeans'].get('scores', [])
            if scores:
                k_values = [s['k'] for s in scores]
                silhouette_scores = [s['silhouette'] for s in scores]
                plt.plot(k_values, silhouette_scores, 'bo-')
                plt.title('K-means Elbow Curve')
                plt.xlabel('Number of Clusters (k)')
                plt.ylabel('Silhouette Score')
        
        # 8. Data quality metrics
        plt.subplot(3, 4, 8)
        completeness = (1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        uniqueness = len(self.df.drop_duplicates()) / len(self.df) * 100
        
        metrics = ['Completeness', 'Uniqueness']
        values = [completeness, uniqueness]
        colors = ['green' if v > 90 else 'orange' if v > 70 else 'red' for v in values]
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('Data Quality Metrics')
        plt.ylabel('Percentage')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def print_insights_summary(self) -> None:
        """Print human-readable insights summary (legacy method)"""
        # This method now calls the enhanced version
        self.print_detailed_insights()
    
    # === COMPLETE ANALYSIS WORKFLOW ===
    def analyze_everything(self) -> None:
        """Run complete analysis with all visualizations and insights"""
        print("🚀 Starting comprehensive dataset analysis...")
        
        # 1. Print detailed insights
        self.print_detailed_insights()
        
        # 2. Distribution analysis
        print("\n📊 Analyzing distributions...")
        self.plot_distributions()
        
        # 3. Correlation analysis  
        print("\n🔗 Analyzing correlations...")
        self.plot_correlation_matrix()
        
        # 4. Outlier analysis
        print("\n🎯 Analyzing outliers...")
        outliers = self.detect_outliers()
        if outliers:
            self.plot_outliers_pca()
        
        # 5. Clustering analysis
        print("\n🔍 Analyzing clusters...")
        self.plot_clusters()
        
        # 6. Dimensionality analysis
        print("\n📐 Analyzing dimensionality...")
        self.plot_reductions()
        
        # 7. Time series analysis (if applicable)
        if self.time_col and self._numeric_cols:
            print("\n⏰ Analyzing time series patterns...")
            try:
                stationarity = self.analyze_stationarity()
                if not stationarity.empty:
                    print("Stationarity Analysis:")
                    print(stationarity[['feature', 'is_stationary', 'adf_pvalue', 'kpss_pvalue']])
                
                # Plot ACF/PACF for first numeric column
                first_col = self._numeric_cols[0]
                print(f"\nAutocorrelation analysis for {first_col}:")
                self.plot_autocorrelations(first_col)
                
                # Show lag suggestions
                lag_suggestions = self.suggest_lag_features()
                if lag_suggestions:
                    print("Suggested lag features:")
                    for feature, lags in list(lag_suggestions.items())[:5]:
                        print(f"  {feature}: lags {lags}")
                        
            except Exception as e:
                print(f"Time series analysis failed: {e}")
        
        # 8. Network analysis (if available)
        if OPTIONAL_IMPORTS['networkx'] and len(self._numeric_cols) > 2:
            print("\n🕸️  Creating correlation network...")
            try:
                self.plot_correlation_network()
            except Exception as e:
                print(f"Network analysis failed: {e}")
        
        # 9. Summary dashboard
        print("\n📋 Creating summary dashboard...")
        self.create_summary_dashboard()
        
        print("\n🎉 Analysis complete! Use the generated report for detailed insights.")
    
    def quick_analysis(self) -> None:
        """Quick analysis for rapid insights"""
        print("⚡ Quick Analysis Mode")
        print("-" * 40)
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Numeric features: {len(self._numeric_cols)}")
        
        # Quick distribution check
        dist_summary = self.analyze_distributions()
        if not dist_summary.empty:
            print(f"Gaussian features: {dist_summary['is_gaussian'].sum()}")
            print(f"Skewed features: {dist_summary['is_skewed'].sum()}")
        
        # Quick correlation check
        correlations = self.analyze_correlations(['pearson'])
        if 'pearson' in correlations:
            corr_matrix = correlations['pearson']
            high_corr = (corr_matrix.abs() > 0.8).sum().sum() - len(corr_matrix)  # Exclude diagonal
            print(f"High correlations (|r| > 0.8): {high_corr // 2}")  # Divide by 2 for symmetry
        
        # Quick outlier check
        outliers = self.detect_outliers(['isolation_forest'])
        if 'isolation_forest' in outliers:
            outlier_pct = outliers['isolation_forest'].sum() / len(outliers['isolation_forest']) * 100
            print(f"Outliers detected: {outlier_pct:.1f}%")
        
        # Quick plot
        if len(self._numeric_cols) <= 6:
            self.plot_distributions(ncols=3)
        
        print("\nℹ️  Use analyze_everything() for comprehensive analysis.")
        print("ℹ️  Use print_detailed_insights() for detailed written insights.")
