"""
Complete Modularized TuRBO Architecture - Full Feature Compatibility
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Optional dependencies with graceful fallback
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.spatial.distance import cdist, pdist
    from scipy.stats import entropy, norm, qmc

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ==================== NUMBA OPTIMIZED FUNCTIONS ====================


@njit
def _compute_sample_cov(centered_points: np.ndarray) -> np.ndarray:
    n_points = centered_points.shape[0]
    divisor = max(n_points - 1, 1)
    return centered_points.T @ centered_points / divisor


@njit
def spherical_sample(
    center: np.ndarray, radius: float, samples: np.ndarray
) -> np.ndarray:
    samples = 2 * samples - 1
    return np.clip(center[np.newaxis, :] + radius * samples, 0, 1)


@njit
def is_in_region(point: np.ndarray, center: np.ndarray, radius: float) -> bool:
    diff = point - center
    return np.linalg.norm(diff) <= radius * 1.1


@njit
def project_to_region_kernel(
    point: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    point = np.clip(point, 0.0, 1.0)
    diff = point - center
    distance = np.sqrt(np.sum(diff**2))
    if distance > radius:
        direction = diff / distance
        point = center + radius * direction
    return np.clip(point, 0.0, 1.0)


# ==================== CONFIGURATION ====================

@dataclass
class TuRBOConfig:
    """Enhanced configuration for adaptive trust region management with overlap network"""
    
    # Core parameters
    n_trust_regions: int = 10
    init_radius: float = 0.2
    min_radius: float = 0.01
    max_radius: float = 0.5
    max_regions: int = 100
    expand_factor: float = 1.3
    contract_factor: float = 0.7
    success_tolerance: int = 3
    failure_tolerance: int = 5
    min_improvement: float = 1e-6
    max_restarts: int = 10
    enable_acquisition_lbfgs: bool = True
    
    # Adaptive parameters
    enable_adaptive_regions: bool = True
    min_trust_regions: int = 5
    max_trust_regions: int = 50
    success_rate_threshold: float = 0.3
    failure_rate_threshold: float = 0.7
    adaptation_window: int = 20
    min_trials_before_adaptation: int = 10
    region_spawn_cooldown: int = 5
    stagnation_threshold: int = 25
    exploration_boost_factor: float = 1.5
    
    # TuRBO-M specific parameters
    max_regions_per_center: int = 3
    min_separation_distance: float = 0.05
    region_overlap_tolerance: float = 0.8
    center_merging_threshold: float = 0.1
    
    # Entropy-aware exploration
    entropy_weight: float = 0.3
    entropy_window: int = 50
    min_entropy_threshold: float = 0.1
    exploration_entropy_boost: float = 1.5
    
    # Per-region acquisition optimization
    acquisition_optimization_budget: int = 100
    acquisition_multi_start: int = 5
    acquisition_local_search_steps: int = 20
    
    # Advanced region management
    region_quality_threshold: float = 0.2
    center_dominance_threshold: float = 0.7
    adaptive_overlap_scaling: bool = True
    region_birth_probability: float = 0.1
    enable_multiscale_regions: bool = True
    multiscale_scales: List[float] = field(default_factory=lambda: [0.5, 0.25])
    min_trials_before_multiscale: int = 10
    success_rate_threshold_multiscale: float = 0.7
    max_multiscale_regions_per_center: int = 3
    
    # ==================== OVERLAP NETWORK PARAMETERS ====================
    
    # Network similarity computation
    overlap_sigma: float = 0.1
    """Gaussian kernel bandwidth for region similarity computation"""
    
    overlap_threshold: float = 0.3
    """Minimum similarity threshold for considering regions as overlapping (adjacency)"""
    
    merge_threshold: float = 0.7
    """Minimum similarity threshold for automatically merging regions"""
    
    exploration_bias: float = 2.0
    """Multiplicative boost factor for regions in disconnected components"""
    
    # Network analysis control
    enable_network_merging: bool = True
    """Enable automatic merging of highly similar regions"""
    
    enable_network_exploration: bool = True
    """Enable spawning of exploration regions in disconnected areas"""
    
    network_update_freq: int = 10
    """Update network analysis every N trials"""
    
    network_adaptation_freq: int = 25
    """Apply network-based adaptations (merging/spawning) every N trials"""
    
    # Advanced network parameters
    max_network_merges_per_cycle: int = 2
    """Maximum number of region merges to perform per adaptation cycle"""
    
    max_network_spawns_per_cycle: int = 2
    """Maximum number of exploration regions to spawn per adaptation cycle"""
    
    min_component_separation: float = 0.1
    """Minimum distance between components to consider spawning between them"""
    
    network_entropy_weight: float = 0.2
    """Weight for network entropy in region selection scoring"""
    
    spectral_clustering_max_clusters: int = 5
    """Maximum number of clusters for spectral clustering analysis"""
    
    similarity_cache_size: int = 1000
    """Size of similarity computation cache for performance"""
    
    # Network debugging and visualization
    enable_network_logging: bool = True
    """Enable detailed logging of network operations"""
    
    save_network_matrices: bool = False
    """Save similarity/adjacency matrices for visualization"""
    
    network_analysis_verbose: bool = True
    """Print detailed network analysis information"""
    
    # Feature toggles
    verbose: bool = True

# ==================== CORE DATA STRUCTURES ====================

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RegionState(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CONVERGED = "converged"
    FAILED = "failed"


@dataclass
class RegionMetrics:
    """Encapsulates region performance metrics"""

    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    trials_in_region: int = 0
    age: int = 0
    last_success_trial: int = -1
    recent_improvements: List[float] = field(default_factory=list)
    local_success_rate: float = 0.0
    contribution_score: float = 0.0
    exploration_score: float = 1.0

    def update_success_rate(self):
        """Update local success rate based on recent performance"""
        total_trials = self.success_count + self.failure_count
        self.local_success_rate = self.success_count / max(total_trials, 1)


@dataclass
class RegionData:
    """Encapsulates region training data"""

    training_points: List[np.ndarray] = field(default_factory=list)
    training_values: List[float] = field(default_factory=list)
    success_points: List[np.ndarray] = field(default_factory=list)
    max_training_points: int = 50
    max_success_points: int = 20

    def add_training_point(self, point: np.ndarray, value: float):
        """Add training point with automatic pruning"""
        self.training_points.append(point.copy())
        self.training_values.append(value)

        if len(self.training_points) > self.max_training_points:
            self.training_points = self.training_points[-self.max_training_points :]
            self.training_values = self.training_values[-self.max_training_points :]

    def add_success_point(self, point: np.ndarray):
        """Add success point with automatic pruning"""
        self.success_points.append(point.copy())
        if len(self.success_points) > self.max_success_points:
            self.success_points = self.success_points[-self.max_success_points :]


@dataclass
class TrustRegion:
    """Comprehensive trust region with all features"""

    center: np.ndarray
    radius: float
    active: bool = True
    age: int = 0
    trials_in_region: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_success_trial: int = -1
    best_value: float = float("inf")
    state: RegionState = RegionState.ACTIVE

    # Encapsulated components
    metrics: RegionMetrics = field(default_factory=RegionMetrics)
    data: RegionData = field(default_factory=RegionData)

    # Performance metrics
    local_success_rate: float = 0.0
    contribution_score: float = 0.0
    exploration_score: float = 1.0

    # MAB attributes
    mab_visits: int = 1
    mab_reward: float = 0.0
    sub_regions: List["TrustRegion"] = field(default_factory=list)
    parent_id: Optional[int] = None

    # Collections
    success_history: List[int] = field(default_factory=list)
    failure_history: List[int] = field(default_factory=list)
    recent_improvements: Any = field(
        default_factory=lambda: __import__("collections").deque(maxlen=20)
    )

    # TuRBO-M specific attributes
    center_id: int = -1
    overlap_regions: Any = field(default_factory=set)
    local_entropy: float = 1.0
    acquisition_cache: Dict = field(default_factory=dict)
    last_acquisition_optimization: int = 0
    region_quality_score: float = 0.5
    exploration_bias: float = 1.0
    overlap_penalty: float = 0.0

    # Ellipsoidal region attributes
    cov_matrix: Optional[np.ndarray] = None
    inv_cov_matrix: Optional[np.ndarray] = None
    cov_chol: Optional[np.ndarray] = None
    success_points: List[np.ndarray] = field(default_factory=list)
    min_points_for_cov: int = 5
    use_ellipsoidal: bool = True

    # Clustering attributes
    cluster_id: Optional[int] = None
    merge_candidates: Any = field(default_factory=set)

    # Local surrogate attributes
    local_surrogate: Any = None
    training_data_x: List[np.ndarray] = field(default_factory=list)
    training_data_y: List[float] = field(default_factory=list)
    min_points_for_surrogate: int = 4
    surrogate_last_update: int = 0
    surrogate_performance_score: float = 0.0

    # Gradient sampling attributes
    gradient_step_size: float = 0.01
    gradient_sampling_active: bool = False

    # Acquisition function samples
    ei_samples: List[np.ndarray] = field(default_factory=list)
    ucb_samples: List[np.ndarray] = field(default_factory=list)
    pi_samples: List[np.ndarray] = field(default_factory=list)
    entropy_drop_streak: int = 0
    is_subregion: bool = False
    max_subregions: int = 2
    subregion_depth: int = 0
    allow_subregion_expansion: bool = True

    @property
    def active_prop(self) -> bool:
        return self.state == RegionState.ACTIVE

    @active_prop.setter
    def active_prop(self, value: bool):
        self.state = RegionState.ACTIVE if value else RegionState.INACTIVE


@dataclass
class RegionCenter:
    """Represents a center that can host multiple overlapping trust regions"""

    location: np.ndarray
    regions: List[int] = field(default_factory=list)
    creation_time: int = 0
    last_updated: int = 0
    dominance_score: float = 0.0
    entropy_history: deque = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class TrustRegionManagerState:
    """Encapsulates the state needed for region management decisions"""

    active_regions: List[int]
    global_trial_count: int
    global_best_value: float
    overall_success_rate: float
    stagnation_counter: int
    last_adaptation_trial: int
    restart_count: int
    config: TuRBOConfig
    regions: List[TrustRegion]

    def get_active_region_stats(self) -> Dict:
        """Get statistics about active regions"""
        if not self.active_regions:
            return {}

        active_region_objs = [self.regions[i] for i in self.active_regions]
        return {
            "avg_radius": np.mean([r.radius for r in active_region_objs]),
            "avg_success_rate": np.mean(
                [r.local_success_rate for r in active_region_objs]
            ),
            "min_radius": min(r.radius for r in active_region_objs),
            "max_consecutive_failures": max(
                r.consecutive_failures for r in active_region_objs
            ),
        }


# ==================== SAMPLING ENGINE ====================
import numpy as np
from scipy.stats import qmc


class SamplingEngine:
    """Handles hybrid sampling strategies with fallbacks"""

    def __init__(self, n_dims: int, method: str = "sobol", seed: int = None):
        self.n_dims = n_dims
        self.method = method.lower()
        self.seed = seed
        self._init_engines()

    def _init_engines(self):
        """Initialize all engines safely"""
        self.engines = {}

        try:
            self.engines["sobol"] = qmc.Sobol(
                d=self.n_dims, scramble=True, seed=self.seed
            )
        except Exception:
            pass

        try:
            self.engines["lhs"] = qmc.LatinHypercube(d=self.n_dims)
        except Exception:
            pass

        try:
            self.engines["halton"] = qmc.Halton(
                d=self.n_dims, scramble=True, seed=self.seed
            )
        except Exception:
            pass

    def _safe_sample(self, engine_name: str, n_points: int) -> np.ndarray:
        engine = self.engines.get(engine_name)
        if engine:
            try:
                return np.clip(engine.random(n_points), 1e-8, 1 - 1e-8)
            except Exception as e:
                print(f"⚠️ Sampling error with {engine_name}: {e}")
        return np.random.rand(n_points, self.n_dims)

    def get_unit_samples(self, n_points: int) -> np.ndarray:
        """Hybrid sampling logic with fallback"""

        if self.method == "hybrid":
            parts = n_points // 3
            extra = n_points - 3 * parts
            sobol = self._safe_sample("sobol", parts)
            lhs = self._safe_sample("lhs", parts)
            halton = self._safe_sample("halton", parts + extra)
            return np.vstack([sobol, lhs, halton])

        if self.method in self.engines:
            return self._safe_sample(self.method, n_points)

        return np.random.rand(n_points, self.n_dims)


# ==================== STRATEGY INTERFACES ====================


class RegionSelectionStrategy(ABC):
    """Strategy for selecting which regions to use for sampling"""

    @abstractmethod
    def select_regions(
        self, regions: List[TrustRegion], max_regions: int = None
    ) -> List[int]:
        pass


class RegionAdaptationStrategy(ABC):
    """Strategy for adapting the number and configuration of regions"""

    @abstractmethod
    def should_add_region(self, manager_state: TrustRegionManagerState) -> bool:
        pass

    @abstractmethod
    def should_remove_region(self, manager_state: TrustRegionManagerState) -> bool:
        pass

    @abstractmethod
    def get_region_to_remove(self, regions: List[TrustRegion]) -> int:
        pass


class SamplingStrategy(ABC):
    """Strategy for sampling candidates within regions"""

    @abstractmethod
    def sample_candidates(
        self, region: TrustRegion, n_candidates: int, context: Dict
    ) -> np.ndarray:
        pass


class AcquisitionStrategy(ABC):
    """Strategy for optimizing acquisition functions"""

    @abstractmethod
    def optimize_acquisition(
        self, region: TrustRegion, acquisition_type: str, context: Dict
    ) -> np.ndarray:
        pass


from sklearn.covariance import OAS

# ==================== GEOMETRIC REGION MANAGEMENT ====================
from sklearn.decomposition import PCA

class RegionGeometryManager:
    """State-of-the-art geometry manager for ellipsoidal and spherical regions"""
    
    def __init__(self, n_dims: int, enable_ellipsoidal: bool = True):
        self.n_dims = n_dims
        self.enable_ellipsoidal = enable_ellipsoidal
        
    def initialize_region_geometry(self, region: TrustRegion):
        """Initialize region with spherical geometry"""
        radius_sq = region.radius**2
        region.cov_matrix = np.eye(self.n_dims) * radius_sq
        region.inv_cov_matrix = np.eye(self.n_dims) / radius_sq
        region.cov_chol = np.eye(self.n_dims) * region.radius
        region.success_points = []
        region.min_points_for_cov = max(self.n_dims + 1, 5)
        region.use_ellipsoidal = False
        
    def update_covariance(self, region: TrustRegion):
        """Update covariance with improved scaling and blending"""
        if (not self.enable_ellipsoidal or 
            len(region.success_points) < region.min_points_for_cov):
            return
            
        try:
            X = np.array(region.success_points)
            X_centered = X - region.center[np.newaxis, :]
            
            # Estimate covariance with better methods
            cov = self._estimate_covariance(X_centered)
            
            # Improved scaling and blending strategy
            cov_scaled = self._scale_and_blend_covariance(region, cov, X_centered)
            
            # Update region matrices
            region.cov_matrix = cov_scaled
            region.inv_cov_matrix = np.linalg.inv(cov_scaled)
            region.cov_chol = np.linalg.cholesky(cov_scaled + 1e-8 * np.eye(self.n_dims))
            region.use_ellipsoidal = True
            
        except Exception as e:
            print(f"⚠️ Covariance update failed: {e}")
            self.reset_to_spherical(region)
    
    def _estimate_covariance(self, X_centered: np.ndarray) -> np.ndarray:
        """Estimate covariance using best available method"""
        n_samples, n_dims = X_centered.shape
        
        try:
            if n_samples > n_dims * 2:
                # Use OAS for well-conditioned case
                from sklearn.covariance import OAS
                cov_estimator = OAS().fit(X_centered)
                return cov_estimator.covariance_
            elif n_samples > n_dims:
                # Use shrinkage for moderate sample sizes
                from sklearn.covariance import LedoitWolf
                cov_estimator = LedoitWolf().fit(X_centered)
                return cov_estimator.covariance_
            else:
                # Use PCA for low sample sizes
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(n_dims, n_samples - 1))
                pca.fit(X_centered)
                return pca.get_covariance()
                
        except Exception:
            # Fallback to sample covariance with regularization
            cov = np.cov(X_centered, rowvar=False)
            if cov.ndim == 0:  # Single dimension case
                cov = np.array([[cov]])
            return cov + 1e-6 * np.eye(cov.shape[0])
    
    def _scale_and_blend_covariance(self, region: TrustRegion, new_cov: np.ndarray, 
                                   X_centered: np.ndarray) -> np.ndarray:
        """Improved covariance scaling and blending strategy"""
        n_samples = len(X_centered)
        
        # 1. Scale covariance to match region size
        current_scale = np.trace(region.cov_matrix) / self.n_dims
        new_scale = np.trace(new_cov) / self.n_dims
        
        # Prevent covariance from becoming too small or too large
        min_scale = (region.radius * 0.1) ** 2  # At least 10% of radius
        max_scale = (region.radius * 3.0) ** 2  # At most 3x radius
        
        if new_scale < min_scale:
            scaling_factor = min_scale / max(new_scale, 1e-12)
            new_cov = new_cov * scaling_factor
        elif new_scale > max_scale:
            scaling_factor = max_scale / new_scale
            new_cov = new_cov * scaling_factor
        
        # 2. Adaptive blending based on confidence
        # More samples = more confidence in empirical covariance
        confidence = min(1.0, n_samples / (3 * self.n_dims))
        
        # Progressive blending: start conservative, become more aggressive
        base_alpha = 0.7  # Much higher base blending rate
        alpha = base_alpha * confidence
        
        # 3. Condition number check - prevent ill-conditioning
        eigenvals = np.linalg.eigvals(new_cov)
        condition_number = np.max(eigenvals) / np.max(np.min(eigenvals), 1e-12)
        
        if condition_number > 1000:  # Too ill-conditioned
            # Regularize towards spherical
            regularization = np.eye(self.n_dims) * np.mean(eigenvals)
            reg_strength = min(0.5, condition_number / 2000)
            new_cov = (1 - reg_strength) * new_cov + reg_strength * regularization
        
        # 4. Final blending
        blended_cov = (1 - alpha) * region.cov_matrix + alpha * new_cov
        
        # 5. Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(blended_cov)
        eigenvals = np.maximum(eigenvals, (region.radius * 0.05) ** 2)  # Minimum eigenvalue
        blended_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return blended_cov
    
    def reset_to_spherical(self, region: TrustRegion):
        """Reset region to spherical geometry"""
        radius_sq = region.radius**2
        region.cov_matrix = np.eye(self.n_dims) * radius_sq
        region.inv_cov_matrix = np.eye(self.n_dims) / radius_sq
        region.cov_chol = np.eye(self.n_dims) * region.radius
        region.use_ellipsoidal = False
        
    def sample_ellipsoidal(self, region: TrustRegion, n_points: int, sampling_engine) -> np.ndarray:
        """Sample from ellipsoidal region with proper scaling"""
        try:
            # Generate standard normal samples
            if hasattr(sampling_engine, 'get_normal_samples'):
                z = sampling_engine.get_normal_samples(n_points)
            else:
                # Fallback: convert uniform to normal
                unit = sampling_engine.get_unit_samples(n_points)
                z = norm.ppf(np.clip(unit, 1e-8, 1 - 1e-8))
            
            # Transform through Cholesky decomposition
            scaled = z @ region.cov_chol.T
            
            # Translate and clip to unit hypercube
            samples = region.center + scaled
            return np.clip(samples, 0, 1)
            
        except Exception as e:
            print(f"⚠️ Ellipsoidal sampling failed: {e}, falling back to spherical")
            return self.sample_spherical(region, n_points, sampling_engine)
            
    def sample_spherical(self, region: TrustRegion, n_points: int, sampling_engine) -> np.ndarray:
        """Sample from spherical region"""
        unit = sampling_engine.get_unit_samples(n_points)
        return spherical_sample(region.center, region.radius, unit)
    
    def get_ellipse_parameters(self, region: TrustRegion, dims: tuple = (0, 1)) -> dict:
        """Extract ellipse parameters for visualization"""
        if not region.use_ellipsoidal:
            return None
            
        try:
            # Extract 2D covariance submatrix
            cov_2d = region.cov_matrix[np.ix_(dims, dims)]
            
            # Eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
            
            # Sort by eigenvalue magnitude
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            
            # Ellipse parameters (2-sigma confidence ellipse)
            width = 2 * np.sqrt(eigenvals[0])
            height = 2 * np.sqrt(eigenvals[1])
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            return {
                'center': region.center[list(dims)],
                'width': width,
                'height': height,
                'angle': angle,
                'eigenvals': eigenvals,
                'eigenvecs': eigenvecs
            }
            
        except Exception:
            return None
    
    def compute_region_volume(self, region: TrustRegion) -> float:
        """Compute region volume (for spherical and ellipsoidal)"""
        if region.use_ellipsoidal:
            # Volume of ellipsoid: (4/3)π * sqrt(det(cov)) for n=3, generalized for n-dims
            det_cov = np.linalg.det(region.cov_matrix)
            if det_cov <= 0:
                return 0.0
            
            # n-dimensional volume formula
            from scipy.special import gamma
            volume_coeff = (np.pi ** (self.n_dims / 2)) / gamma(self.n_dims / 2 + 1)
            return volume_coeff * np.sqrt(det_cov)
        else:
            # Volume of n-dimensional sphere
            from scipy.special import gamma
            volume_coeff = (np.pi ** (self.n_dims / 2)) / gamma(self.n_dims / 2 + 1)
            return volume_coeff * (region.radius ** self.n_dims)
    
    def check_point_in_ellipsoid(self, point: np.ndarray, region: TrustRegion) -> bool:
        """Check if point is inside ellipsoidal region"""
        if not region.use_ellipsoidal:
            # Use spherical check
            return np.linalg.norm(point - region.center) <= region.radius
        
        try:
            diff = point - region.center
            mahalanobis_dist_sq = diff.T @ region.inv_cov_matrix @ diff
            # Use 2-sigma threshold (95% confidence)
            return mahalanobis_dist_sq <= 4.0
        except Exception:
            # Fallback to spherical
            return np.linalg.norm(point - region.center) <= region.radius
    
    def get_region_statistics(self, region: TrustRegion) -> dict:
        """Get comprehensive region geometry statistics"""
        stats = {
            'use_ellipsoidal': region.use_ellipsoidal,
            'volume': self.compute_region_volume(region),
            'n_success_points': len(region.success_points),
            'min_points_needed': region.min_points_for_cov
        }
        
        if region.use_ellipsoidal:
            eigenvals = np.linalg.eigvals(region.cov_matrix)
            stats.update({
                'eigenvals': eigenvals.tolist(),
                'condition_number': np.max(eigenvals) / np.max(np.min(eigenvals), 1e-12),
                'trace': np.trace(region.cov_matrix),
                'determinant': np.linalg.det(region.cov_matrix),
                'aspect_ratio': np.sqrt(np.max(eigenvals) / np.max(np.min(eigenvals), 1e-12))
            })
        else:
            stats.update({
                'radius': region.radius,
                'spherical_volume': self.compute_region_volume(region)
            })
            
        return stats
# ==================== SURROGATE MODEL MANAGEMENT ====================


class SurrogateModelManager:
    """Manages local and global surrogate models"""

    def __init__(self, surrogate_type: str = "gp", enable_local: bool = True):
        self.surrogate_type = surrogate_type.lower()
        self.enable_local = enable_local
        self.global_surrogate = None
        self.surrogate_manager = None  # External surrogate manager if available

    def set_external_manager(self, manager):
        """Set external surrogate manager"""
        self.surrogate_manager = manager

    def create_local_surrogate(self, X: np.ndarray, y: np.ndarray):
        """Create local surrogate model"""
        if len(X) < 3:
            return None

        surrogate_kwargs = {"X_train": X, "y_train": y}
        model = self.surrogate_manager.create_model(
            surrogate_type=self.surrogate_type,
            n_trials=len(X),
            n_parallel=1,
            **surrogate_kwargs,
        )

        if model is not None and hasattr(model, "fit"):
            try:
                model.fit(X, y.ravel() if y.ndim > 1 else y)
            except Exception:
                return None

        return model

    def predict_surrogate(
        self, surrogate, X: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Unified surrogate prediction interface"""
        if surrogate is None:
            return None, None

        try:
            if hasattr(surrogate, "predict"):
                # Try GP-style prediction first
                try:
                    mean, std = surrogate.predict(X, return_std=True)
                    return mean, std
                except (TypeError, ValueError):
                    # Fallback to regular prediction
                    predictions = surrogate.predict(X)
                    uncertainties = self._estimate_uncertainties(
                        surrogate, X, predictions
                    )
                    return predictions, uncertainties
            elif hasattr(surrogate, "__call__"):
                predictions = surrogate(X)
                uncertainties = np.ones(len(X)) * 0.1
                return predictions, uncertainties
            return None, None
        except Exception:
            return None, None

    def _estimate_uncertainties(
        self, surrogate, X: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """Estimate uncertainties for models without native uncertainty"""
        try:
            if hasattr(surrogate, "estimators_"):  # Random Forest
                tree_predictions = np.array(
                    [tree.predict(X) for tree in surrogate.estimators_]
                )
                return np.std(tree_predictions, axis=0)
            return np.ones(len(X)) * 0.1
        except Exception:
            return np.ones(len(X)) * 0.1

    def update_region_surrogate(self, region: TrustRegion, global_trial_count: int):
        """Update local surrogate model if needed"""
        if not self.enable_local:
            return

        update_interval = max(5, len(region.center))

        if (
            global_trial_count - region.surrogate_last_update >= update_interval
            and len(region.training_data_x) >= region.min_points_for_surrogate
        ):

            X = np.array(region.training_data_x, dtype=np.float32)
            y = np.array(region.training_data_y, dtype=np.float32)

            region.local_surrogate = self.create_local_surrogate(X, y)
            region.surrogate_last_update = global_trial_count

            # Evaluate performance
            if region.local_surrogate is not None:
                pred, _ = self.predict_surrogate(region.local_surrogate, X)
                if pred is not None:
                    mse = np.mean((pred.ravel() - y.ravel()) ** 2)
                    region.surrogate_performance_score = 1.0 / (1.0 + mse)
                else:
                    region.surrogate_performance_score = 0.0
            else:
                region.surrogate_performance_score = 0.0


from typing import Optional

import numpy as np
from numba import njit


@njit
def compute_fd_from_preds(
    f_plus: np.ndarray, f_minus: np.ndarray, epsilon: float
) -> np.ndarray:
    return (f_plus - f_minus) / (2 * epsilon)

class GradientSamplingManager:
    """Enhanced gradient sampling and refinement with fallback strategies."""

    def __init__(self, enable_gradient: bool = True, ratio: float = 0.7, verbose: bool = False):
        self.enable_gradient = enable_gradient
        self.gradient_sampling_ratio = ratio
        self.verbose = verbose

    def _get_predict_fn(self, model):
        """Extract prediction function safely."""
        return (
            getattr(model, "predict", None)
            or getattr(model, "forward", None)
            or getattr(model, "__call__", None)
        )

    def compute_gradients(self, model, X: np.ndarray, epsilon: float = 1e-6) -> Optional[np.ndarray]:
        """Compute gradients using SHAP if available, otherwise numerical."""
        if not self.enable_gradient or model is None:
            return None

        # try:
        #     # Use SHAP if the model has compute_input_gradients
        #     if hasattr(model, "compute_input_gradients"):
        #         grads = model.compute_input_gradients(X)
        #         if grads is not None and grads.shape == X.shape:
        #             if self.verbose:
        #                 print("✅ Using SHAP-based gradients")
        #             return grads
        # except Exception as e:
        #     if self.verbose:
        #         print(f"⚠️ SHAP gradient fallback: {e}")

        # Fallback to numerical gradients
        return self.compute_numerical_gradients(model, X, epsilon)

    def compute_numerical_gradients(self, model, X: np.ndarray, epsilon: float = 1e-6) -> Optional[np.ndarray]:
        """Compute numerical gradients using central differences."""
        if not self.enable_gradient or model is None:
            return None

        gradients = np.zeros_like(X)
        try:
            predict_fn = self._get_predict_fn(model)
            if predict_fn is None:
                raise ValueError("No valid prediction function found.")

            base = predict_fn(X)
            base = base.reshape(-1, 1) if base.ndim == 1 else base

            for i in range(X.shape[1]):
                X_plus = X.copy()
                X_plus[:, i] += epsilon
                X_minus = X.copy()
                X_minus[:, i] -= epsilon

                f_plus = predict_fn(X_plus).reshape(-1, 1)
                f_minus = predict_fn(X_minus).reshape(-1, 1)

                gradients[:, i] = ((f_plus - f_minus) / (2 * epsilon)).ravel()

        except Exception as e:
            if self.verbose:
                print(f"[⚠️ Numerical gradient failed] {e}")
            return None
        return gradients

    def gradient_descent_step(
        self,
        region,
        start: np.ndarray,
        max_steps: int = 5,
        min_improvement: float = 1e-4,
    ) -> np.ndarray:
        """Perform refined gradient-based descent within trust region."""
        if not hasattr(region, "local_surrogate") or region.local_surrogate is None:
            return start

        x = start.copy()
        surrogate = region.local_surrogate
        predict_fn = self._get_predict_fn(surrogate)
        if predict_fn is None:
            return x

        base_value = predict_fn(x[None, :])[0]
        success_count = 0

        for step in range(max_steps):
            grad = self.compute_gradients(surrogate, x[None, :])
            if grad is None or np.allclose(grad, 0):
                break

            grad = grad[0]
            norm = np.linalg.norm(grad)
            if norm < 1e-8:
                break

            improved = False
            for scale in [1.0, 0.5, 0.25, 0.1]:
                step_size = getattr(region, "gradient_step_size", 0.1) * scale
                x_new = project_to_region_kernel(
                    x - step_size * grad / norm, region.center, region.radius
                )
                new_value = predict_fn(x_new[None, :])[0]

                if (base_value - new_value) > min_improvement:
                    x = x_new
                    base_value = new_value
                    improved = True
                    success_count += 1
                    if self.verbose:
                        print(
                            f"✔️ GD step {step} (scale={scale:.2f}) improved: {new_value:.5f}"
                        )
                    break

            if not improved:
                if self.verbose:
                    print(f"⚠️ No improvement at GD step {step}, stopping.")
                break

        if self.verbose and success_count == 0:
            print(f"🛑 Gradient descent did not yield improvement.")

        return x


from collections import deque
from typing import Optional

from scipy.spatial.distance import pdist
from scipy.stats import entropy as scipy_entropy


class EntropyCalculationManager:
    """Advanced entropy manager for region-based exploration"""

    def __init__(self, config: TuRBOConfig):
        self.config = config

    def compute_local_entropy(self, region: TrustRegion) -> float:
        points = getattr(region, "training_data_x", None)
        spatial_entropy = (
            self._estimate_spatial_entropy(points[-20:])
            if points and len(points) > 5
            else self._fallback_entropy(region)
        )

        model_entropy = getattr(region, "predictive_entropy", 0.0)
        local_entropy = 0.5 * spatial_entropy + 0.5 * model_entropy

        # Update entropy history
        if not hasattr(region, "entropy_history"):
            region.entropy_history = deque(maxlen=5)
        region.entropy_history.append(local_entropy)

        # Track entropy drops
        if len(region.entropy_history) >= 3:
            hist = list(region.entropy_history)
            decreasing = hist[-3] > hist[-2] > hist[-1]
            region.entropy_drop_streak = (
                region.entropy_drop_streak + 1 if decreasing else 0
            )

        region.local_entropy = max(local_entropy, self.config.min_entropy_threshold)
        return region.local_entropy

    def _estimate_spatial_entropy(self, points: np.ndarray) -> float:
        try:
            distances = pdist(points)
            if np.all(distances == 0):
                return 0.0
            hist, _ = np.histogram(distances, bins=10, density=True)
            hist += 1e-8  # Avoid log(0)
            hist /= hist.sum()
            return min(scipy_entropy(hist) / np.log(10), 1.0)
        except Exception:
            return 0.0

    def _fallback_entropy(self, region: TrustRegion) -> float:
        age = getattr(region, "age", 1.0)
        success_rate = max(1e-3, region.local_success_rate)
        return 0.5 * min(age / 50.0, 1.0) + 0.5 * (1.0 - success_rate)

    def should_boost_exploration(self, region: TrustRegion) -> bool:
        if region.local_entropy >= self.config.min_entropy_threshold:
            return False

        stagnating = region.consecutive_failures > max(
            2, self.config.failure_tolerance // 2
        )
        drop_streak = getattr(region, "entropy_drop_streak", 0)
        overlaps_strong = (
            sum(
                1
                for r in getattr(region, "overlap_regions", [])
                if r.local_success_rate > 0.5
            )
            > 2
        )

        return stagnating or drop_streak >= 3 or overlaps_strong


from scipy.optimize import minimize

# ==================== ACQUISITION FUNCTION MANAGER ====================
from scipy.stats import norm


class AcquisitionFunctionManager:
    """Modular and TuRBO-M-aware acquisition function manager."""

    def __init__(self, config: TuRBOConfig, surrogate_manager: SurrogateModelManager):
        self.config = config
        self.surrogate_manager = surrogate_manager
        self.acquisition_registry = {
            "ei": self._expected_improvement,
            "ucb": self._upper_confidence_bound,
            "pi": self._probability_of_improvement,
        }

    def evaluate_acquisition_function(
        self, region, point, acquisition_type, current_best
    ) -> float:
        """Evaluate acquisition function at a single point with entropy scaling (TuRBO-M)."""
        if not hasattr(region, "local_surrogate") or region.local_surrogate is None:
            # Fallback to center bias
            distance_to_center = np.linalg.norm(point - region.center)
            return -(distance_to_center / region.radius)

        try:
            point_2d = point.reshape(1, -1)
            preds, uncs = self.surrogate_manager.predict_surrogate(
                region.local_surrogate, point_2d
            )
            if preds is None or uncs is None:
                return 0.0
            score = self.evaluate_single(
                preds[0], uncs[0], acquisition_type, current_best
            )

            # TuRBO-M: boost exploration in high-entropy or young regions
            entropy_boost = 1.0 + 0.2 * getattr(region, "local_entropy", 0.0)
            age_decay = np.exp(-region.age / 50.0)
            return score * (0.5 + 0.5 * age_decay) * entropy_boost

        except Exception:
            return 0.0

    def evaluate_single(self, pred, unc, acq_type: str, current_best: float) -> float:
        """Modular acquisition evaluation from registry."""
        return self.acquisition_registry.get(acq_type, self._fallback)(
            pred, unc, current_best
        )

    def _expected_improvement(self, pred, unc, best) -> float:
        improvement = best - pred
        if unc < 1e-8:
            return max(improvement, 0.0)
        z = improvement / unc
        return improvement * norm.cdf(z) + unc * norm.pdf(z)

    def _upper_confidence_bound(self, pred, unc, best) -> float:
        return -(pred - 2.0 * unc)

    def _probability_of_improvement(self, pred, unc, best) -> float:
        improvement = best - pred
        if unc < 1e-8:
            return 1.0 if improvement > 0 else 0.0
        return norm.cdf(improvement / unc)

    def _fallback(self, pred, unc, best) -> float:
        return 0.0

    def optimize_acquisition_multi_start(
        self, region, acquisition_type, sampling_engine, geometry_manager
    ) -> np.ndarray:
        """Multi-start acquisition optimization with optional L-BFGS refinement."""
        starts = geometry_manager.sample_spherical(
            region, self.config.acquisition_multi_start, sampling_engine
        )
        current_best = region.best_value if region.best_value != float("inf") else 0.0

        if starts.ndim == 1:
            starts = starts.reshape(1, -1)

        best_point = None
        best_score = -float("inf")

        for start in starts:
            local_opt = self._local_acquisition_search(
                region, start, acquisition_type, current_best
            )
            score = self.evaluate_acquisition_function(
                region, local_opt, acquisition_type, current_best
            )
            if score > best_score:
                best_score = score
                best_point = local_opt.copy()

        return best_point if best_point is not None else region.center

    def _local_acquisition_search(
        self, region, start, acq_type: str, current_best: float
    ) -> np.ndarray:
        """Local random-direction acquisition search followed by optional L-BFGS refinement."""
        current = start.copy()
        step_size = region.radius * 0.1

        for step in range(self.config.acquisition_local_search_steps):
            candidates = []
            for _ in range(5):
                direction = np.random.normal(0, 1, len(current))
                direction /= np.linalg.norm(direction)
                candidate = current + step_size * direction
                candidate = project_to_region_kernel(
                    candidate, region.center, region.radius
                )
                candidates.append(candidate)

            best_candidate = current
            best_score = self.evaluate_acquisition_function(
                region, current, acq_type, current_best
            )

            for candidate in candidates:
                score = self.evaluate_acquisition_function(
                    region, candidate, acq_type, current_best
                )
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if np.linalg.norm(best_candidate - current) < 1e-6:
                break

            current = best_candidate
            step_size *= 0.9

        # Optional L-BFGS-B refinement
        if self.config.enable_acquisition_lbfgs:
            current = self._refine_with_lbfgs(region, current, acq_type, current_best)

        return current

    def _refine_with_lbfgs(
        self, region, start, acq_type: str, current_best: float
    ) -> np.ndarray:
        """Optional final optimization with L-BFGS inside region bounds."""

        def neg_acq(x):
            return -self.evaluate_acquisition_function(
                region, x, acq_type, current_best
            )

        bounds = [(0.0, 1.0)] * len(start)
        res = minimize(neg_acq, start, method="L-BFGS-B", bounds=bounds)
        return project_to_region_kernel(res.x, region.center, region.radius)


import numpy as np

# ==================== CLUSTERING MANAGER ====================
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from collections import deque
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
import numba
import logging

logger = logging.getLogger(__name__)

# ==================== STRATEGY IMPLEMENTATIONS ====================
class MultiArmedBanditSelection(RegionSelectionStrategy):
    """Multi-armed bandit selection with UCB + entropy-aware scoring"""

    def __init__(
        self,
        entropy_weight: float = 0.1,
        decay_factor: float = 1.0,
        verbose: bool = False,
    ):
        self.entropy_weight = entropy_weight
        self.decay_factor = decay_factor
        self.verbose = verbose

    def select_regions(
        self, regions: List[TrustRegion], max_regions: int = None
    ) -> List[int]:
        active_idxs = [i for i, r in enumerate(regions) if r.active]
        if not active_idxs or (max_regions and len(active_idxs) <= max_regions):
            return active_idxs

        # Optional decay
        if self.decay_factor < 1.0:
            for i in active_idxs:
                r = regions[i]
                r.mab_reward *= self.decay_factor
                r.mab_visits = max(1, r.mab_visits * self.decay_factor)

        total_visits = sum(regions[i].mab_visits for i in active_idxs)
        log_term = np.log(total_visits + 1e-8)

        scores = [(i, self._score_region(regions[i], log_term)) for i in active_idxs]
        scores.sort(key=lambda x: x[1], reverse=True)

        if self.verbose:
            print("[Bandit Scores]", [(i, round(s, 4)) for i, s in scores])

        return [i for i, _ in scores[:max_regions]]

    def _score_region(self, region: TrustRegion, log_term: float) -> float:
        reward = region.mab_reward
        visits = max(region.mab_visits, 1)
        exploitation = reward / visits
        exploration = np.sqrt(2 * log_term / visits)
        entropy_bonus = self.entropy_weight * getattr(region, "entropy", 0.0)
        return exploitation + exploration + entropy_bonus


class AdaptiveRegionManagement(RegionAdaptationStrategy):
    """Adaptive trust region management with dynamic spawning and pruning"""

    def should_add_region(self, state: TrustRegionManagerState) -> bool:
        if (
            len(state.active_regions) >= state.config.max_trust_regions
            or state.global_trial_count - state.last_adaptation_trial
            < state.config.region_spawn_cooldown
        ):
            return False

        stats = state.get_active_region_stats()
        low_success = state.overall_success_rate < state.config.success_rate_threshold
        stagnation = state.stagnation_counter > state.config.stagnation_threshold
        tight_radius = (
            stats.get("avg_radius", 1.0) < state.config.init_radius * 0.3
            if stats
            else False
        )

        return low_success or stagnation or tight_radius

    def should_remove_region(self, state: TrustRegionManagerState) -> bool:
        return len(state.active_regions) > state.config.min_trust_regions

    def get_region_to_remove(self, regions: List[TrustRegion]) -> int:
        active_idxs = [i for i, r in enumerate(regions) if r.active]
        if not active_idxs:
            return -1

        scores = [(i, self._score_region(regions[i])) for i in active_idxs]
        scores.sort(key=lambda x: x[1])

        if len(scores) > 1 and scores[0][1] < 0.5 * scores[1][1]:
            return scores[0][0]
        return -1

    def _score_region(self, region: TrustRegion) -> float:
        """Lower score = worse performance"""
        weights = {
            "success": 0.3,
            "contribution": 0.3,
            "exploration": 0.2,
            "quality": 0.1,
            "entropy": 0.1,
        }
        values = {
            "success": region.local_success_rate,
            "contribution": region.contribution_score,
            "exploration": region.exploration_score,
            "quality": getattr(region, "region_quality_score", 0.0),
            "entropy": getattr(region, "entropy", 0.0),
        }
        return sum(weights[k] * values.get(k, 0.0) for k in weights)


class HybridSamplingStrategy(SamplingStrategy):
    """Hybrid strategy combining geometric and gradient-based sampling"""

    def __init__(
        self,
        geometry_manager: RegionGeometryManager,
        gradient_manager: GradientSamplingManager,
        sampling_engine: SamplingEngine,
    ):
        self.geometry_manager = geometry_manager
        self.gradient_manager = gradient_manager
        self.sampling_engine = sampling_engine
        self.min_radius = 1e-6  # Minimum radius for fallback sampling

    def sample_candidates(
        self, region: TrustRegion, n_candidates: int, context: Dict
    ) -> np.ndarray:
        """Sample points using a blend of geometric and gradient-based strategies"""
        use_ellipsoidal = context.get(
            "use_ellipsoidal", getattr(region, "use_ellipsoidal", False)
        )
        use_gradient = context.get("use_gradient", False)
        gradient_ratio = context.get("gradient_ratio", 0.3)

        n_gradient = int(n_candidates * gradient_ratio) if use_gradient else 0
        n_traditional = n_candidates - n_gradient

        # Traditional samples (spherical or ellipsoidal)
        traditional_samples = (
            self._sample_traditional(region, n_traditional, use_ellipsoidal)
            if n_traditional > 0
            else []
        )

        # Gradient-guided samples
        gradient_samples = (
            self._sample_gradient_guided(region, n_gradient) if n_gradient > 0 else []
        )

        # Fallback if nothing worked
        # Final fallback if no samples generated
        if traditional_samples.size == 0 and gradient_samples.size == 0:
            return self.geometry_manager.sample_spherical(
                region, n_candidates, self.sampling_engine
            )

        return np.vstack([s for s in [traditional_samples, gradient_samples] if len(s)])

    def _sample_traditional(
        self, region: TrustRegion, n: int, ellipsoidal: bool
    ) -> np.ndarray:
        try:
            if ellipsoidal:
                samples = self.geometry_manager.sample_ellipsoidal(
                    region, n, self.sampling_engine
                )
            else:
                samples = self.geometry_manager.sample_spherical(
                    region, n, self.sampling_engine
                )
            return samples.reshape(-1, region.center.shape[0])
        except Exception as e:
            print(f"[⚠️ Traditional sampling failed] {e}")
            return np.empty((0, region.center.shape[0]))

    def _sample_gradient_guided(self, region: TrustRegion, n: int) -> np.ndarray:
        if not hasattr(region, "local_surrogate") or region.local_surrogate is None:
            return np.empty((0, region.center.shape[0]))

        try:
            seeds = self.geometry_manager.sample_spherical(
                region, n, self.sampling_engine
            )
            seeds = seeds.reshape(-1, region.center.shape[0])
            optimized = [
                self.gradient_manager.gradient_descent_step(region, seed)
                for seed in seeds
            ]
            return np.array(optimized)
        except Exception as e:
            print(f"[⚠️ Gradient sampling failed] {e}")
            return np.empty((0, region.center.shape[0]))


class TurboMAcquisitionStrategy(AcquisitionStrategy):
    """TuRBO-M specific acquisition optimization strategy"""

    def __init__(
        self,
        acquisition_manager: AcquisitionFunctionManager,
        entropy_manager: EntropyCalculationManager,
        sampling_engine: SamplingEngine,
        geometry_manager: RegionGeometryManager,
    ):
        self.acquisition_manager = acquisition_manager
        self.entropy_manager = entropy_manager
        self.sampling_engine = sampling_engine
        self.geometry_manager = geometry_manager
        self.min_radius = 1e-6  # Minimum radius for fallback sampling

    def optimize_acquisition(
        self, region: TrustRegion, acquisition_type: str, context: Dict
    ) -> np.ndarray:
        """Optimize acquisition function with TuRBO-M enhancements"""
        config = context.get("config")
        global_trial_count = context.get("global_trial_count", 0)

        region.acquisition_cache = getattr(region, "acquisition_cache", {})
        region.last_acquisition_optimization = getattr(
            region, "last_acquisition_optimization", -999
        )

        update_needed = (
            global_trial_count - region.last_acquisition_optimization
            > config.acquisition_optimization_budget // 10
        )

        if not update_needed and acquisition_type in region.acquisition_cache:
            cached_point = region.acquisition_cache[acquisition_type]
            radius = max(region.radius, self.min_radius)
            noise = np.random.normal(0, radius * 0.1, size=len(region.center))
            noisy_point = cached_point + noise
            return project_to_region_kernel(noisy_point, region.center, region.radius)

        # Optimize acquisition function
        best_point = self.acquisition_manager.optimize_acquisition_multi_start(
            region, acquisition_type, self.sampling_engine, self.geometry_manager
        )

        # Optional entropy-boosted exploration
        if self.entropy_manager.should_boost_exploration(region):
            entropy_scale = max(region.local_entropy, 1e-3)
            exploration_noise = np.random.normal(
                0, region.radius * 0.2 * entropy_scale, len(best_point)
            )
            best_point += exploration_noise

        # Ensure it's valid
        best_point = project_to_region_kernel(best_point, region.center, region.radius)

        # Cache result and update metadata
        region.acquisition_cache[acquisition_type] = best_point.copy()
        region.last_acquisition_optimization = global_trial_count

        # Optional: store acquisition score
        try:
            score = self.acquisition_manager.evaluate_acquisition_function(
                region, best_point, acquisition_type, region.best_value
            )
            region.acquisition_scores = getattr(region, "acquisition_scores", {})
            region.acquisition_scores[acquisition_type] = score
        except Exception:
            pass

        return best_point
