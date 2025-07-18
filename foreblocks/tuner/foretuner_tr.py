"""
Complete Modularized TuRBO Architecture - Full Feature Compatibility
"""

# ─── Standard Library ─────────────────────────────────────────────────────
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# ─── Logging Setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Core Scientific Stack ────────────────────────────────────────────────
import numpy as np
from numpy.linalg import norm, inv, pinv, LinAlgError

# ─── Optional Acceleration ────────────────────────────────────────────────
try:
    from numba import njit, jit, vectorize, guvectorize
    from numba.types import float64, int64, boolean
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ─── Optional ML and Optimization ─────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
    from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.spatial.distance import cdist, pdist
    from scipy.stats import entropy, norm, rankdata, qmc
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ─── Optional Graph Tools ─────────────────────────────────────────────────
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# ─── Internal Modules ─────────────────────────────────────────────────────
from .foretuner_tr_aux import *
from .foretuner_surrogate import SurrogateManager


@dataclass
class RegionMetrics:
    """Metrics for region performance tracking"""
    success_rate: float = 0.0
    trial_count: int = 0
    consecutive_failures: int = 0
    age: int = 0
    entropy: float = 0.0
    last_improvement_iter: int = 0
    volume: float = 1.0
    hypervolume_contribution: float = 0.0

class TrustRegion:
    """Advanced trust region with adaptive geometry and multi-fidelity support"""
    
    def __init__(self, center, radius, best_value=float('inf'), region_id=None):
        self.center = np.array(center)
        self.radius = radius
        self.best_value = best_value
        self.region_id = region_id
        self.active = True
        self.trial_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        
        # Advanced geometry
        dim = len(center)
        self.cov = np.eye(dim) * (radius ** 2)
        self.cov_inv = np.eye(dim) / (radius ** 2)
        self.eigenvals = np.ones(dim)
        self.eigenvecs = np.eye(dim)
        
        # Adaptive parameters
        self.expansion_factor = 1.2
        self.contraction_factor = 0.7
        self.min_radius = 1e-8
        self.max_radius = 0.5

    def update(self, value, point, gradient=None):
        """Update region with new trial result and optional gradient info"""
        self.trial_count += 1
        old_center = self.center.copy()
        
        if value < self.best_value:
            improvement = self.best_value - value
            self.best_value = value
            self.success_count += 1
            self.consecutive_failures = 0
            
            # Adaptive center update with momentum
            momentum = 0.1 if self.trial_count > 5 else 0.3
            self.center = momentum * np.array(point) + (1 - momentum) * self.center
            
            # Adaptive expansion based on improvement magnitude
            if improvement > 0.01:
                expansion = min(self.expansion_factor, 1 + improvement * 10)
                self.radius = min(self.radius * expansion, self.max_radius)
            
            # Update covariance with rank-1 update
            self._update_covariance(old_center, point, success=True)
            
        else:
            self.consecutive_failures += 1
            
            # Contraction schedule based on failure count
            if self.consecutive_failures >= 2:
                contraction = self.contraction_factor ** min(self.consecutive_failures - 1, 3)
                self.radius = max(self.radius * contraction, self.min_radius)
                
                if self.radius <= self.min_radius:
                    self.active = False
            
            # Update covariance to avoid bad directions
            self._update_covariance(old_center, point, success=False)

    def _update_covariance(self, old_center, new_point, success=True):
        """Update covariance matrix based on search direction success"""
        try:
            direction = np.array(new_point) - old_center
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 1e-10:
                return
                
            direction = direction / direction_norm
            
            # Rank-1 covariance update
            if success:
                # Expand along successful direction
                self.cov += 0.1 * np.outer(direction, direction)
            else:
                # Contract along unsuccessful direction
                self.cov -= 0.05 * np.outer(direction, direction)
            
            # Ensure positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(self.cov)
            eigenvals = np.maximum(eigenvals, 1e-8)
            self.eigenvals = eigenvals
            self.eigenvecs = eigenvecs
            self.cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            self.cov_inv = eigenvecs @ np.diag(1/eigenvals) @ eigenvecs.T
            
        except Exception:
            # Fallback to isotropic
            dim = len(self.center)
            self.cov = np.eye(dim) * (self.radius ** 2)
            self.cov_inv = np.eye(dim) / (self.radius ** 2)
    
    @property
    def success_rate(self):
        return self.success_count / max(1, self.trial_count)
    
    @property
    def volume(self):
        """Compute region volume"""
        try:
            return np.sqrt(np.prod(self.eigenvals)) * (self.radius ** len(self.center))
        except:
            return self.radius ** len(self.center)

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import pdist, cdist
from scipy.stats import qmc
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class Trial:
    """Trial data structure"""
    def __init__(self, params, value, trial_id, constraint_violations=None):
        self.params = params
        self.value = value
        self.trial_id = trial_id
        self.constraint_violations = constraint_violations or []
        
    @property
    def is_feasible(self):
        return len(self.constraint_violations) == 0 or all(v <= 0 for v in self.constraint_violations)

class TrustRegion:
    """Simplified trust region"""
    def __init__(self, center, radius, best_value=float('inf'), region_id=0):
        self.center = np.array(center)
        self.radius = radius
        self.best_value = best_value
        self.region_id = region_id
        self.trial_count = 0
        self.success_count = 0
        self.age = 0
        self.active = True
        
        # Initialize covariance as identity
        self.cov = np.eye(len(center))
        self.cov_inv = np.eye(len(center))
        
    def update(self, value, point):
        """Update region with new trial"""
        self.trial_count += 1
        self.age += 1
        
        if value < self.best_value:
            self.best_value = value
            self.success_count += 1
            # Move center slightly toward good point
            self.center = 0.9 * self.center + 0.1 * np.array(point)
        
    @property
    def success_rate(self):
        return self.success_count / max(1, self.trial_count)
    
    @property
    def volume(self):
        return self.radius ** len(self.center)

class RegionManager:
    """Streamlined region manager with fast adaptive partitioning"""

    def __init__(self, config):
        self.config = config
        self.regions = []
        self.region_counter = 0
        
        # Core parameters only
        self.max_regions = getattr(config, 'max_regions', 6)  # Reduced from 8
        self.min_regions = getattr(config, 'min_regions', 2)
        self.merge_threshold = getattr(config, 'merge_threshold', 0.2)
        self.elimination_age = getattr(config, 'elimination_age', 30)  # Simplified
        
        # Fast initialization
        self._sobol_sampler = None
        self._init_sobol()

    def _init_sobol(self):
        """Initialize Sobol sampler once"""
        try:
            self._sobol_sampler = qmc.Sobol(d=getattr(self.config, 'n_dims', 10), scramble=True)
        except:
            self._sobol_sampler = None

    def initialize_regions(self, trials, param_converter):
        """Fast initialization using k-means++"""
        feasible_trials = [t for t in trials if t.is_feasible]
        if len(feasible_trials) < 3:
            self._add_sobol_regions(min(self.max_regions, 4), param_converter)
            return
            
        # Quick k-means++ initialization
        points = np.array([param_converter.to_array(t.params) for t in feasible_trials])
        values = np.array([t.value for t in feasible_trials])
        
        n_centers = min(self.max_regions, len(points))
        centers = self._fast_kmeans_init(points, values, n_centers)
        
        for center in centers:
            distances = np.linalg.norm(points - center, axis=1)
            best_idx = np.argmin(distances)
            self._create_region(center, values[best_idx])

    def _fast_kmeans_init(self, points, values, n_centers):
        """Optimized k-means++ initialization"""
        centers = [points[np.argmin(values)]]  # Start with best point
        
        for _ in range(1, n_centers):
            # Vectorized distance computation
            dists = cdist(points, centers).min(axis=1)
            # Weight by distance and inverse objective
            weights = dists ** 2 * np.exp(-(values - values.min()) / (values.max() - values.min() + 1e-6))
            weights /= weights.sum()
            next_idx = np.random.choice(len(points), p=weights)
            centers.append(points[next_idx])
        
        return centers

    def update_regions(self, recent_trials, param_converter):
        """Fast Voronoi assignment with vectorized operations"""
        if not recent_trials or not self.regions:
            return
            
        # Vectorized region assignment
        trial_points = np.array([param_converter.to_array(t.params) 
                               for t in recent_trials if t.is_feasible])
        if len(trial_points) == 0:
            return
            
        region_centers = np.array([r.center for r in self.regions])
        
        # Compute all distances at once
        distances = cdist(trial_points, region_centers)
        closest_regions = np.argmin(distances, axis=1)
        
        # Update regions
        for i, trial in enumerate(t for t in recent_trials if t.is_feasible):
            region_idx = closest_regions[i]
            self.regions[region_idx].update(trial.value, trial_points[i])

    def manage_regions(self, trials, param_converter, model=None):
        """Streamlined region management"""
        if len(self.regions) < 2:
            return
            
        initial_count = len(self.regions)
        
        # Fast merge and elimination
        self._fast_merge_similar()
        self._fast_eliminate_stale()
        
        # Add exploration regions if needed
        feasible_trials = [t for t in trials if t.is_feasible]
        target_regions = max(self.min_regions, min(self.max_regions, len(feasible_trials) // 15))
        
        while len(self.regions) < target_regions and len(feasible_trials) > 10:
            self._add_diverse_region(feasible_trials, param_converter)
        
        if len(self.regions) != initial_count:
            print(f"🔄 Regions: {initial_count} → {len(self.regions)}")

    def _fast_merge_similar(self):
        """Fast region merging using vectorized operations"""
        if len(self.regions) <= self.min_regions:
            return
            
        centers = np.array([r.center for r in self.regions])
        distances = pdist(centers)
        
        # Find pairs to merge
        merge_pairs = []
        dist_matrix = np.zeros((len(self.regions), len(self.regions)))
        idx = 0
        for i in range(len(self.regions)):
            for j in range(i + 1, len(self.regions)):
                dist_matrix[i, j] = distances[idx]
                if distances[idx] < self.merge_threshold:
                    merge_pairs.append((i, j))
                idx += 1
        
        # Merge closest pair only (to avoid over-merging)
        if merge_pairs:
            i, j = merge_pairs[0]
            self._merge_regions(self.regions[i], self.regions[j])

    def _fast_eliminate_stale(self):
        """Fast elimination of underperforming regions"""
        if len(self.regions) <= self.min_regions:
            return
            
        # Simple scoring: age penalty + success rate
        scores = []
        for region in self.regions:
            age_penalty = max(0, 1 - region.age / self.elimination_age)
            score = 0.7 * region.success_rate + 0.3 * age_penalty
            scores.append(score)
        
        # Keep top performers
        region_scores = list(zip(self.regions, scores))
        region_scores.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(self.min_regions, len(self.regions) - 2)
        self.regions = [r for r, _ in region_scores[:keep_count]]

    def _add_diverse_region(self, trials, param_converter):
        """Add region in most diverse location"""
        if not trials:
            return
            
        points = np.array([param_converter.to_array(t.params) for t in trials])
        centers = np.array([r.center for r in self.regions])
        
        # Find point farthest from all regions
        distances_to_regions = cdist(points, centers).min(axis=1)
        farthest_idx = np.argmax(distances_to_regions)
        
        self._create_region(points[farthest_idx])

    def _add_sobol_regions(self, n_regions, param_converter):
        """Add regions using Sobol sampling"""
        if self._sobol_sampler is not None:
            try:
                centers = self._sobol_sampler.random(n_regions)
                for center in centers:
                    self._create_region(center)
                return
            except:
                pass
        
        # Fallback to random
        for _ in range(n_regions):
            center = np.random.rand(self.config.n_dims)
            self._create_region(center)

    def _create_region(self, center, best_value=float('inf')):
        """Create region with minimal overhead"""
        region = TrustRegion(center, self.config.init_radius, best_value, self.region_counter)
        self.region_counter += 1
        self.regions.append(region)
        return region

    def _merge_regions(self, r1, r2):
        """Simple region merging"""
        # Weighted center
        total_count = r1.success_count + r2.success_count + 1e-6
        new_center = (r1.center * r1.success_count + r2.center * r2.success_count) / total_count
        new_best = min(r1.best_value, r2.best_value)
        
        # Create merged region
        merged = self._create_region(new_center, new_best)
        merged.radius = max(r1.radius, r2.radius)
        
        # Remove old regions
        if r1 in self.regions:
            self.regions.remove(r1)
        if r2 in self.regions:
            self.regions.remove(r2)

    def get_active_regions(self):
        return [r for r in self.regions if r.active]

    def get_statistics(self):
        active = self.get_active_regions()
        if not active:
            return {"active_regions": 0, "avg_success_rate": 0.0}
        
        return {
            "active_regions": len(active),
            "avg_success_rate": np.mean([r.success_rate for r in active])
        }


class TuRBOController:
    """Optimized state-of-the-art TuRBO controller"""

    def __init__(self, config, device="auto"):
        self.config = config
        self.device = device
        self.n_dims = None
        
        # Core components
        self.region_manager = RegionManager(config)
        self.global_model = None
        self.local_models = {}
        
        # Streamlined configuration
        self.acquisition = getattr(config, "acquisition", "ei")
        self.batch_size = getattr(config, "batch_size", 1)
        self.ucb_beta = getattr(config, "ucb_beta", 2.0)
        self.update_freq = getattr(config, "update_frequency", 8)  # More frequent
        self.mgmt_freq = getattr(config, "management_frequency", 20)
        self.min_local_samples = getattr(config, "min_local_samples", 5)
        
        # Fast samplers
        self._sobol_sampler = None
        self._lhs_sampler = None

    def optimize(self, objective, parameters, initial_trials, n_trials, constraints, param_converter, patience):
        """Optimized main loop with adaptive exploration"""
        self.n_dims = len(parameters)
        self._param_converter = param_converter
        self._init_samplers()
        
        # Initialize
        self.region_manager.initialize_regions(initial_trials, param_converter)
        self._setup_global_model(initial_trials)
        
        trials = list(initial_trials)
        no_improve_count = 0
        best_value = self._get_best_value(trials)
        stagnation_count = 0
        
        for iteration in range(len(trials), n_trials):
            # Adaptive update frequency
            if iteration % max(5, self.update_freq - iteration // 40) == 0:
                self._fast_update_models(trials)
                recent_trials = trials[-self.update_freq:] if len(trials) >= self.update_freq else trials
                self.region_manager.update_regions(recent_trials, param_converter)
            
            # Progress reporting every 10 trials
            if iteration % 10 == 0:
                self._report_progress(trials, iteration)
            
            # Region management
            if iteration % max(10, self.mgmt_freq - iteration // 30) == 0:
                self.region_manager.manage_regions(trials, param_converter, self.global_model)
            
            # Generate candidates with adaptive strategy
            exploration_weight = self._compute_exploration_weight(iteration, n_trials, stagnation_count)
            candidates = self._fast_generate_candidates(trials, exploration_weight)
            
            # Evaluate
            new_trials = self._fast_evaluate_candidates(objective, candidates, iteration, constraints)
            valid_trials = [t for t in new_trials if t is not None]
            trials.extend(valid_trials)
            
            # Check improvement with better reporting
            current_best = self._get_best_value(trials)
            if current_best < best_value - abs(best_value) * 1e-5:  # Relative threshold
                improvement = best_value - current_best
                best_value = current_best
                no_improve_count = 0
                stagnation_count = max(0, stagnation_count - 1)
                print(f"🎯 Trial {iteration}: New best = {best_value:.6f} (↑{improvement:.6f})")
            else:
                no_improve_count += 1
                if no_improve_count > 4:
                    stagnation_count += 1
            
            # Early stopping with adaptive patience
            adaptive_patience = patience + int(exploration_weight * 15)
            if no_improve_count >= adaptive_patience:
                print(f"🛑 Early stopping: {no_improve_count} iterations without improvement")
                break
        
        return trials

    def _init_samplers(self):
        """Initialize samplers once"""
        try:
            self._sobol_sampler = qmc.Sobol(d=self.n_dims, scramble=True)
            self._lhs_sampler = qmc.LatinHypercube(d=self.n_dims)
        except:
            self._sobol_sampler = None
            self._lhs_sampler = None

    def _setup_global_model(self, trials):
        """Setup global surrogate model"""
        feasible_trials = [t for t in trials if t.is_feasible]
        if len(feasible_trials) < 3:
            return
            
        X = np.array([self._param_converter.to_array(t.params) for t in feasible_trials])
        y = np.array([t.value for t in feasible_trials])
        
        from .foretuner_surrogate import SurrogateManager
        manager = SurrogateManager()
        model_type = "ensemble" if len(X) > 20 else "xgboost" if len(X) > 10 else "linear"
        
        # Create and fit model
        self.global_model = manager.create_model(
            model_type=model_type, 
            n_trials=len(X), 
            X_train=X, 
            y_train=y, 
            n_parallel=self.batch_size,
            verbose=False
        )



    def _fast_generate_candidates(self, trials, exploration_weight):
        """Fast candidate generation with smart sampling"""
        active_regions = self.region_manager.get_active_regions()
        if not active_regions:
            return [self._sobol_sample() for _ in range(self.batch_size)]
        
        candidates = []
        for _ in range(self.batch_size):
            if np.random.random() < exploration_weight:
                # Exploration: 50% Sobol, 50% boundary
                if np.random.random() < 0.5:
                    candidate = self._sobol_sample()
                else:
                    region = self._select_exploration_region(active_regions)
                    candidate = self._sample_region_boundary(region)
            else:
                # Exploitation: fast acquisition optimization
                region = self._select_exploitation_region(active_regions)
                candidate = self._fast_optimize_acquisition(region, candidates)
            
            candidates.append(candidate)
        
        return candidates

    def _select_exploration_region(self, regions):
        """Select region for exploration (prefer underexplored)"""
        weights = []
        for region in regions:
            # Favor large volume and low trial count
            weight = region.volume / (1 + region.trial_count)
            weights.append(weight)
        
        weights = np.array(weights)
        weights /= weights.sum()
        return np.random.choice(regions, p=weights)

    def _select_exploitation_region(self, regions):
        """Select region for exploitation (prefer successful)"""
        weights = []
        for region in regions:
            # Favor high success rate and good values
            weight = region.success_rate + 0.1
            weights.append(weight)
        
        weights = np.array(weights)
        weights /= weights.sum()
        return np.random.choice(regions, p=weights)

    def _fast_optimize_acquisition(self, region, existing_candidates):
        """Fast acquisition optimization with limited candidates"""
        try:
            # Generate small candidate pool
            pool_size = 20  # Reduced from 50
            candidates = []
            
            # Half from region, half quasi-random
            for _ in range(pool_size // 2):
                candidates.append(self._sample_in_region(region))
            
            if self._sobol_sampler is not None:
                try:
                    quasi_samples = self._sobol_sampler.random(pool_size // 2)
                    candidates.extend(quasi_samples)
                except:
                    candidates.extend([np.random.rand(self.n_dims) for _ in range(pool_size // 2)])
            else:
                candidates.extend([np.random.rand(self.n_dims) for _ in range(pool_size // 2)])
            
            X_pool = np.array(candidates)
            
            # Fast prediction
            mean, std = self._fast_predict(X_pool, region)
            
            # Compute acquisition scores
            scores = self._compute_acquisition_scores(mean, std, region.best_value)
            
            # Simple diversity penalty
            if existing_candidates:
                for i, candidate in enumerate(X_pool):
                    min_dist = min(np.linalg.norm(candidate - existing) for existing in existing_candidates)
                    scores[i] -= 0.1 * np.exp(-min_dist * 5)  # Light penalty
            
            best_idx = np.argmax(scores)
            return X_pool[best_idx]
            
        except Exception as e:
            print(f"⚠️ Fast acquisition failed: {e}")
            return self._sample_in_region(region)

    def _fast_predict(self, X, region):
        """Fast prediction using best available model"""
        try:
            # Try local model first (usually faster)
            if region.region_id in self.local_models:
                try:
                    model = self.local_models[region.region_id]
                    return self._predict_with_model(model, X)
                except:
                    pass
            
            # Fall back to global model
            if self.global_model is not None:
                try:
                    return self._predict_with_model(self.global_model, X)
                except:
                    pass
            
            # Final fallback
            n = len(X)
            return np.random.rand(n), np.ones(n) * 0.1
            
        except Exception:
            n = len(X)
            return np.random.rand(n), np.ones(n) * 0.1

    def _fast_update_models(self, trials):
        """Fast model updates with smart data management using SurrogateManager"""
        feasible_trials = [t for t in trials if t.is_feasible]
        if len(feasible_trials) < 3:
            return
        
        X = np.array([self._param_converter.to_array(t.params) for t in feasible_trials])
        y = np.array([t.value for t in feasible_trials])
        
        # Global model update using SurrogateManager
        if self.global_model is None:
            from .foretuner_surrogate import SurrogateManager
            manager = SurrogateManager()
            model_type = "ensemble" if len(X) > 20 else "xgboost" if len(X) > 10 else "linear"
            
            self.global_model = manager.create_model(
                model_type=model_type,
                n_trials=len(X),
                X_train=X,
                y_train=y,
                n_parallel=self.batch_size,
                verbose=False
            )
        else:
            # Update existing model
            if hasattr(self.global_model, 'fit'):
                # Use recent data for large datasets
                if len(X) > 100:
                    recent_idx = -50
                    self.global_model.fit(X[recent_idx:], y[recent_idx:])
                else:
                    self.global_model.fit(X, y)
        
        # Fast local model updates
        self._fast_update_local_models(trials)

    def _fast_update_local_models(self, trials):
        """Fast local model updates using SurrogateManager with relaxed requirements"""
        from .foretuner_surrogate import SurrogateManager
        manager = SurrogateManager()
        
        for region in self.region_manager.get_active_regions():
            region_trials = self._get_region_trials(region, trials)
            
            if len(region_trials) >= self.min_local_samples:
                X = np.array([self._param_converter.to_array(t.params) for t in region_trials])
                y = np.array([t.value for t in region_trials])
                
                # Choose appropriate model type based on local data size
                if len(X) < 8:
                    model_type = "linear"  # Simple for very small datasets
                elif len(X) < 15:
                    model_type = "xgboost"  # Good for medium datasets
                else:
                    model_type = "gp"  # GP for larger local datasets
                
                model = manager.create_model(
                    model_type=model_type,
                    n_trials=len(X),
                    X_train=X,
                    y_train=y,
                    n_parallel=1,
                    verbose=False
                )
                self.local_models[region.region_id] = model

    def _get_region_trials(self, region, trials):
        """Fast region trial assignment"""
        region_trials = []
        for trial in trials:
            if not trial.is_feasible:
                continue
            x = self._param_converter.to_array(trial.params)
            if np.linalg.norm(x - region.center) <= region.radius * 1.5:
                region_trials.append(trial)
        return region_trials

    def _sample_in_region(self, region):
        """Fast region sampling"""
        try:
            # Simple isotropic sampling
            direction = np.random.randn(self.n_dims)
            direction /= np.linalg.norm(direction)
            radius = region.radius * np.random.random() ** (1.0 / self.n_dims)
            point = region.center + direction * radius
            return np.clip(point, 0, 1)
        except:
            return np.random.rand(self.n_dims)

    def _sample_region_boundary(self, region):
        """Sample on region boundary for exploration"""
        try:
            direction = np.random.randn(self.n_dims)
            direction /= np.linalg.norm(direction)
            # Sample at 2-4x radius for exploration
            exploration_radius = region.radius * np.random.uniform(2.0, 4.0)
            point = region.center + direction * exploration_radius
            return np.clip(point, 0, 1)
        except:
            return np.random.rand(self.n_dims)

    def _sobol_sample(self):
        """Get single Sobol sample"""
        try:
            if self._sobol_sampler is not None:
                return self._sobol_sampler.random(1)[0]
        except:
            pass
        return np.random.rand(self.n_dims)

    def _compute_exploration_weight(self, iteration, n_trials, stagnation):
        """Adaptive exploration weight"""
        progress = iteration / n_trials
        base = 0.5 * (1 - 0.3 * progress)  # Decay from 0.5 to 0.35
        stagnation_boost = min(0.3, stagnation * 0.08)
        return np.clip(base + stagnation_boost, 0.2, 0.7)

    def _fast_evaluate_candidates(self, objective, candidates, start_id, constraints):
        """Fast parallel evaluation"""
        def eval_single(i, x):
            try:
                params = self._param_converter.to_params(x)
                value = objective(params)
                
                violations = []
                if constraints:
                    for constraint in constraints:
                        try:
                            violation = constraint(params)
                            if violation > 0:
                                violations.append(violation)
                        except:
                            violations.append(float('inf'))
                
                return Trial(params, value, start_id + i, constraint_violations=violations)
            except:
                return None
        
        # Parallel evaluation
        if len(candidates) > 1:
            results = []
            with ThreadPoolExecutor(max_workers=min(len(candidates), 4)) as executor:
                futures = {executor.submit(eval_single, i, x): i for i, x in enumerate(candidates)}
                for future in as_completed(futures, timeout=120):
                    try:
                        result = future.result(timeout=30)
                        if result:
                            results.append(result)
                    except:
                        pass
            return results
        else:
            return [eval_single(0, candidates[0])]

    def _predict_with_model(self, model, X):
        """Fast robust prediction"""
        try:
            if hasattr(model, 'predict'):
                mean = model.predict(X)
                std = getattr(model, 'predict_std', lambda x: np.ones(len(x)) * 0.1)(X)
            else:
                mean = model(X)
                std = np.ones(len(X)) * 0.1
            
            mean = np.asarray(mean).flatten()
            std = np.asarray(std).flatten()
            
            # Ensure shapes match
            n = len(X)
            if len(mean) != n:
                mean = mean[:n] if len(mean) > n else np.concatenate([mean, np.random.rand(n - len(mean))])
            if len(std) != n:
                std = std[:n] if len(std) > n else np.concatenate([std, np.ones(n - len(std)) * 0.1])
            
            return mean, np.maximum(std, 1e-6)
            
        except Exception:
            n = len(X)
            return np.random.rand(n), np.ones(n) * 0.1

    def _compute_acquisition_scores(self, mean, std, best_value):
        """Fast acquisition computation"""
        if self.acquisition == "ei":
            improvement = best_value - mean
            z = improvement / (std + 1e-9)
            # Fast approximation of normal CDF/PDF
            cdf = 0.5 * (1 + np.tanh(0.8 * z))
            pdf = np.exp(-0.5 * z**2) / 2.507  # sqrt(2*pi) ≈ 2.507
            return improvement * cdf + std * pdf
        
        elif self.acquisition == "ucb":
            return -mean + self.ucb_beta * std
        
        else:  # Default EI
            improvement = best_value - mean
            z = improvement / (std + 1e-9)
            cdf = 0.5 * (1 + np.tanh(0.8 * z))
            pdf = np.exp(-0.5 * z**2) / 2.507
            return improvement * cdf + std * pdf

    def _get_best_value(self, trials):
        """Fast best value extraction"""
        feasible_values = [t.value for t in trials if t.is_feasible and np.isfinite(t.value)]
        return min(feasible_values) if feasible_values else float('inf')

    def _report_progress(self, trials, iteration):
        """Enhanced progress reporting every 10 trials"""
        feasible_trials = [t for t in trials if t.is_feasible]
        best_value = min(t.value for t in feasible_trials) if feasible_trials else float('inf')
        feasibility_rate = len(feasible_trials) / max(1, len(trials))
        
        # Calculate recent performance
        recent_trials = trials[-10:] if len(trials) >= 10 else trials
        recent_feasible = [t for t in recent_trials if t.is_feasible]
        recent_best = min(t.value for t in recent_feasible) if recent_feasible else float('inf')
        
        # Get exploration weight
        exploration_weight = self._compute_exploration_weight(iteration, len(trials) + 100, 0)
        
        stats = self.region_manager.get_statistics()
        local_count = len(self.local_models)
        
        # More detailed progress report
        print(f"📈 Trial {iteration:3d}/{len(trials)+100} | "
              f"Best: {best_value:8.4f} | Recent: {recent_best:8.4f} | "
              f"Feasible: {feasibility_rate:4.1%} | "
              f"Regions: {stats['active_regions']:2d} | "
              f"Local: {local_count:2d} | "
              f"Explore: {exploration_weight:4.1%}")
        
        # Show current best parameters every 20 trials
        if iteration % 20 == 0 and feasible_trials:
            best_trial = min(feasible_trials, key=lambda t: t.value)
            params_str = ", ".join([f"{k}={v:.3f}" for k, v in list(best_trial.params.items())[:3]])
            if len(best_trial.params) > 3:
                params_str += "..."
            print(f"    💎 Current best params: {params_str}")

    def manage_regions(self, trials, param_converter, model=None):
        """Enhanced region management with better reporting"""
        if len(self.regions) < 2:
            return
            
        initial_count = len(self.regions)
        
        # Fast merge and elimination
        self._fast_merge_similar()
        self._fast_eliminate_stale()
        
        # Add exploration regions if needed
        feasible_trials = [t for t in trials if t.is_feasible]
        target_regions = max(self.min_regions, min(self.max_regions, len(feasible_trials) // 15))
        
        added_count = 0
        while len(self.regions) < target_regions and len(feasible_trials) > 10:
            self._add_diverse_region(feasible_trials, param_converter)
            added_count += 1
        
        final_count = len(self.regions)
        if final_count != initial_count:
            change_type = "+" if final_count > initial_count else "-"
            print(f"🔄 Regions: {initial_count} → {final_count} ({change_type}{abs(final_count-initial_count)})")

    def get_optimization_state(self):
        """Get current state"""
        return {
            'regions': len(self.region_manager.get_active_regions()),
            'local_models': len(self.local_models),
            'global_model': self.global_model is not None
        }
    
# Supporting classes remain the same
class Trial:
    def __init__(self, params, value, trial_id, constraint_violations=None):
        self.params = params
        self.value = value
        self.trial_id = trial_id
        self.constraint_violations = constraint_violations or []
    
    @property
    def is_feasible(self):
        return len(self.constraint_violations) == 0


class ParameterConverter:
    """Convert between parameter dicts and arrays"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        self.bounds = [(p.low, p.high) for p in parameters]
        self.names = [p.name for p in parameters]
        self.n_dims = len(parameters)
    
    def to_array(self, params_dict):
        """Convert parameter dict to normalized array [0,1]"""
        array = np.zeros(len(self.parameters))
        for i, (name, (low, high)) in enumerate(zip(self.names, self.bounds)):
            value = params_dict[name]
            array[i] = (value - low) / (high - low)
        return array
    
    def to_params(self, array):
        """Convert normalized array to parameter dict"""
        params = {}
        for i, (name, (low, high)) in enumerate(zip(self.names, self.bounds)):
            params[name] = low + array[i] * (high - low)
        return params
