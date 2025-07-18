# Suppress warnings globally
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

# Standard library
import time
import logging
from functools import lru_cache
from types import SimpleNamespace
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Callable, Optional, Tuple, Union

# Core scientific stack
import numpy as np
import matplotlib.pyplot as plt

# Scipy
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution

# Project-specific modules
from foretuner_config import *
from foretuner_surrogate import *
from foretuner_sample import *
from foretuner_acq import *

# Numba (optional acceleration)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# scikit-learn (optional)
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# PyTorch + BoTorch + GPyTorch (optional)
try:
    import torch
    import botorch
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
    from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
    from gpytorch.means import ConstantMean
    from gpytorch.likelihoods import GaussianLikelihood
    BOTORCH_AVAILABLE = True
    GPYTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    GPYTORCH_AVAILABLE = False


@dataclass
class Trial:
    """Enhanced trial with constraint handling"""
    params: Dict[str, Any]
    value: float
    trial_id: int
    timestamp: float = field(default_factory=time.time)
    constraint_violations: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_feasible(self) -> bool:
        return len(self.constraint_violations) == 0 or all(v <= 0 for v in self.constraint_violations)


# State and data structures
@dataclass
class TrustRegion:
    """Individual trust region state"""
    center: np.ndarray
    radius: float
    success_count: int = 0
    failure_count: int = 0
    best_value: float = float('inf')
    active: bool = True
    age: int = 0
    # New fields for adaptive radius
    success_history: list = None
    failure_history: list = None
    last_success_trial: int = -1
    consecutive_failures: int = 0
    
    def __post_init__(self):
        if self.success_history is None:
            self.success_history = []
        if self.failure_history is None:
            self.failure_history = []

class TuRBOController:
    """Decoupled TuRBO 2 controller with adaptive radius scheduling"""
    
    def __init__(self, n_dims: int, config: TuRBOConfig, device: str = 'cpu'):
        self.n_dims = n_dims
        self.config = config
        self.device = device
        self.restart_count = 0
        self.global_trial_count = 0
        
        # Initialize trust regions
        self.trust_regions = [
            TrustRegion(
                center=np.random.rand(n_dims) if device == 'cpu' else torch.rand(n_dims, device=device),
                radius=config.init_radius
            )
            for _ in range(config.n_trust_regions)
        ]
        
        self.global_best_value = float('inf')
        self.global_best_point = None
        
    def get_trust_region_bounds(self, region_idx: int) -> List[Tuple[float, float]]:
        """Get bounds for specific trust region"""
        if region_idx >= len(self.trust_regions):
            return [(0.0, 1.0)] * self.n_dims
            
        region = self.trust_regions[region_idx]
        center = region.center
        radius = region.radius
        
        # Convert to numpy if needed
        if hasattr(center, 'cpu'):  # torch tensor
            center = center.cpu().numpy()
        
        return [(max(0.0, center[i] - radius), min(1.0, center[i] + radius)) 
                for i in range(self.n_dims)]
    
    def get_active_regions(self) -> List[int]:
        """Get indices of active trust regions"""
        return [i for i, region in enumerate(self.trust_regions) if region.active]
    
    def initialize_from_trials(self, trials, params_converter):
        """Initialize trust region centers with best trials"""
        sorted_trials = sorted(trials, key=lambda t: t.value)
        
        for i, region in enumerate(self.trust_regions):
            if i < len(sorted_trials):
                center_array = params_converter(sorted_trials[i].params)
                region.center = center_array
                region.best_value = sorted_trials[i].value
            else:
                region.center = np.random.rand(self.n_dims)
                
        # Initialize global best
        if sorted_trials:
            self.global_best_value = sorted_trials[0].value
            self.global_best_point = params_converter(sorted_trials[0].params)
    
    def update_regions(self, recent_trials, params_converter):
        """Update all trust regions with adaptive radius scheduling"""
        self.global_trial_count += len(recent_trials)
        
        # Update global best
        for trial in recent_trials:
            if trial.value < self.global_best_value:
                self.global_best_value = trial.value
                self.global_best_point = params_converter(trial.params)
        
        # Update each region with adaptive radius
        for region_idx in range(len(self.trust_regions)):
            region = self.trust_regions[region_idx]
            if not region.active:
                continue
            
            # Check for improvement in this region and update radius adaptively
            self._adaptive_radius_update(region_idx, recent_trials, params_converter)
    
    def _adaptive_radius_update(self, region_idx: int, recent_trials, params_converter):
        """Adaptive radius update using success/failure windows (original TuRBO algorithm)"""
        region = self.trust_regions[region_idx]
        region.age += 1
        
        # Check if any recent trial improved this region
        region_success = False
        best_trial_in_batch = None
        
        for trial in recent_trials:
            # Check if trial is within this region
            trial_array = params_converter(trial.params)
            if self._is_point_in_region(trial_array, region_idx):
                if trial.value < region.best_value - self.config.min_improvement:
                    region_success = True
                    region.best_value = trial.value
                    best_trial_in_batch = trial
                    region.last_success_trial = self.global_trial_count
                    
                    # Update center towards better point (original TuRBO behavior)
                    region.center = 0.8 * region.center + 0.2 * trial_array
        
        # Update success/failure history
        if region_success:
            region.success_history.append(self.global_trial_count)
            region.consecutive_failures = 0
            
            # Keep only recent successes within window
            cutoff = self.global_trial_count - self.config.success_window * 10
            region.success_history = [t for t in region.success_history if t > cutoff]
        else:
            region.failure_history.append(self.global_trial_count)
            region.consecutive_failures += 1
            
            # Keep only recent failures within window
            cutoff = self.global_trial_count - self.config.failure_window * 10
            region.failure_history = [t for t in region.failure_history if t > cutoff]
        
        # Adaptive radius adjustment based on success/failure patterns
        self._adjust_radius_adaptively(region_idx, region_success)
        
        # Check for restart conditions
        self._check_restart_conditions(region_idx)
    
    def _is_point_in_region(self, point: np.ndarray, region_idx: int) -> bool:
        """Check if a point falls within a trust region"""
        region = self.trust_regions[region_idx]
        center = region.center
        if hasattr(center, 'cpu'):
            center = center.cpu().numpy()
        
        # Check if point is within radius of center
        distance = np.linalg.norm(point - center)
        return distance <= region.radius * 1.1  # Small tolerance
    
    def _adjust_radius_adaptively(self, region_idx: int, current_success: bool):
        """Adjust radius based on success/failure windows (TuRBO algorithm)"""
        region = self.trust_regions[region_idx]
        
        # Count recent successes and failures
        recent_successes = len(region.success_history)
        recent_failures = len(region.failure_history)
        
        # Original TuRBO radius adjustment logic
        if current_success:
            # Expand radius on success, but check recent history
            if recent_successes >= self.config.success_tolerance:
                # Multiple recent successes - expand more aggressively
                expansion_factor = self.config.expand_factor
                region.radius = min(
                    self.config.max_radius, 
                    region.radius * expansion_factor
                )
            else:
                # Single success - modest expansion
                region.radius = min(
                    self.config.max_radius, 
                    region.radius * np.sqrt(self.config.expand_factor)
                )
        else:
            # Contract radius on failure
            if region.consecutive_failures >= self.config.failure_tolerance:
                # Multiple consecutive failures - contract aggressively
                contraction_factor = self.config.contract_factor
                region.radius = max(
                    self.config.min_radius,
                    region.radius * contraction_factor
                )
            elif recent_failures > recent_successes:
                # More failures than successes recently - modest contraction
                region.radius = max(
                    self.config.min_radius,
                    region.radius * np.sqrt(self.config.contract_factor)
                )
        
        # Additional adaptive behavior based on region age and performance
        self._age_based_radius_adjustment(region_idx)
    
    def _age_based_radius_adjustment(self, region_idx: int):
        """Additional radius adjustment based on region age and performance"""
        region = self.trust_regions[region_idx]
        
        # If region is old and hasn't seen success recently, gradually shrink
        trials_since_success = self.global_trial_count - region.last_success_trial
        if region.last_success_trial > 0 and trials_since_success > 20:
            aging_factor = 0.95 ** (trials_since_success // 20)
            region.radius = max(
                self.config.min_radius,
                region.radius * aging_factor
            )
        
        # If region radius is too small, consider for restart
        if region.radius < self.config.restart_threshold:
            region.radius = max(self.config.min_radius, region.radius)
    
    def _check_restart_conditions(self, region_idx: int):
        """Check if trust region should be restarted"""
        region = self.trust_regions[region_idx]
        
        # Original TuRBO restart conditions
        restart_needed = False
        
        # Condition 1: Radius too small
        if region.radius < self.config.min_radius * 2:
            restart_needed = True
            
        # Condition 2: Too many consecutive failures
        if region.consecutive_failures >= self.config.failure_tolerance * 2:
            restart_needed = True
            
        # Condition 3: No improvement for too long
        trials_since_success = self.global_trial_count - region.last_success_trial
        if region.last_success_trial > 0 and trials_since_success > 50:
            restart_needed = True
        
        if restart_needed:
            self._restart_region(region_idx)
    
    def _restart_region(self, region_idx: int):
        """Restart a trust region with adaptive placement"""
        if self.restart_count >= self.config.max_restarts:
            self.trust_regions[region_idx].active = False
            return
        
        region = self.trust_regions[region_idx]
        
        # Smart restart: place near global best with some randomization
        if self.global_best_point is not None:
            # Start near global best but with some exploration
            noise_scale = self.config.init_radius * 0.3
            noise = np.random.normal(0, noise_scale, self.n_dims)
            new_center = self.global_best_point + noise
            new_center = np.clip(new_center, 0, 1)  # Keep in bounds
        else:
            # Fallback to random placement
            new_center = np.random.rand(self.n_dims)
        
        # Reset region state
        region.center = new_center
        region.radius = self.config.init_radius * 0.7  # Start slightly smaller
        region.success_count = 0
        region.failure_count = 0
        region.success_history = []
        region.failure_history = []
        region.last_success_trial = -1
        region.consecutive_failures = 0
        region.age = 0
        region.active = True
        region.best_value = float('inf')
        
        self.restart_count += 1
        
    def get_region_stats(self) -> Dict:
        """Get statistics about trust regions for monitoring"""
        active_regions = self.get_active_regions()
        
        stats = {
            'active_regions': len(active_regions),
            'total_restarts': self.restart_count,
            'global_best': self.global_best_value,
            'region_radii': [self.trust_regions[i].radius for i in active_regions],
            'avg_radius': np.mean([self.trust_regions[i].radius for i in active_regions]) if active_regions else 0,
            'min_radius': min([self.trust_regions[i].radius for i in active_regions]) if active_regions else 0,
            'max_radius': max([self.trust_regions[i].radius for i in active_regions]) if active_regions else 0,
        }
        
        return stats


# Supporting classes for cleaner separation of concerns
class ParameterConverter:
    """Handles bidirectional parameter conversion"""
    
    def __init__(self, parameters):
        self.parameters = parameters
    
    def to_params(self, array):
        """Convert normalized array to parameters"""
        # Handle torch tensors by converting to numpy
        if hasattr(array, 'detach'):  # torch tensor
            array = array.detach().cpu().numpy()
        
        array = np.asarray(array).flatten()
        params = {}
        
        for i, param in enumerate(self.parameters):
            val = np.clip(array[i] if i < len(array) else 0.5, 1e-10, 1 - 1e-10)
            
            if param.type == 'float':
                if getattr(param, 'log', False):
                    log_range = np.log(param.high) - np.log(param.low)
                    params[param.name] = param.low * np.exp(val * log_range)
                else:
                    params[param.name] = param.low + val * (param.high - param.low)
            elif param.type == 'int':
                if getattr(param, 'log', False):
                    log_range = np.log(param.high) - np.log(param.low)
                    params[param.name] = int(param.low * np.exp(val * log_range))
                else:
                    range_size = param.high - param.low + 1
                    params[param.name] = min(param.low + int(val * range_size), param.high)
            elif param.type in ['choice', 'ordinal']:
                idx = int(val * len(param.choices))
                params[param.name] = param.choices[min(idx, len(param.choices) - 1)]
        
        return params
    
    def to_array(self, params):
        """Convert parameters to normalized array"""
        array = []
        
        for param in self.parameters:
            val = params.get(param.name, 0)
            
            if param.type == 'float':
                if getattr(param, 'log', False):
                    log_val = np.log(max(val, param.low * 1e-10))
                    log_range = np.log(param.high) - np.log(param.low)
                    normalized = (log_val - np.log(param.low)) / log_range
                else:
                    normalized = (val - param.low) / (param.high - param.low)
                array.append(np.clip(normalized, 0, 1))
            elif param.type == 'int':
                if getattr(param, 'log', False):
                    log_val = np.log(max(val, param.low * 1e-10))
                    log_range = np.log(param.high) - np.log(param.low)
                    normalized = (log_val - np.log(param.low)) / log_range
                else:
                    normalized = (val - param.low) / (param.high - param.low)
                array.append(np.clip(normalized, 0, 1))
            elif param.type in ['choice', 'ordinal']:
                if val in param.choices:
                    idx = param.choices.index(val)
                    normalized = idx / (len(param.choices) - 1) if len(param.choices) > 1 else 0
                else:
                    normalized = 0
                array.append(normalized)
        
        return np.array(array)


class OptimizationState:
    """Lightweight state tracker for optimization loop"""
    __slots__ = ['best_value', 'no_improve', 'last_improvement_trial']
    
    def __init__(self, trials, initial_best):
        self.best_value = initial_best
        self.no_improve = 0
        self.last_improvement_trial = len(trials)
    
    def update(self, trials, current_best):
        """Update state with new best value"""
        if current_best < self.best_value - 1e-8:
            self.best_value = current_best
            self.no_improve = 0
            self.last_improvement_trial = len(trials)
        else:
            self.no_improve += 1


class ConstraintHandler:
    """Handles constraint analysis and constraint-aware sampling"""
    
    def __init__(self, constraints, config):
        self.constraints = constraints
        self.config = config
        self.feasibility_rate = None
    
    def analyze_difficulty(self, parameters):
        """Analyze constraint difficulty"""
        if not self.constraints:
            return
        
        if self.config.verbose:
            print("🔍 Analyzing constraint difficulty...")
        
        # Quick feasibility check
        feasible_count = 0
        rng = np.random.RandomState(42)
        
        for _ in range(1000):
            sample = rng.random(len(parameters))
            params = ParameterConverter(parameters).to_params(sample)
            
            if all(constraint(params) <= 0 for constraint in self.constraints):
                feasible_count += 1
        
        self.feasibility_rate = feasible_count / 1000
        
        if self.config.verbose:
            print(f"   Feasibility rate: {self.feasibility_rate:.1%}")
            
            if self.feasibility_rate < 0.01:
                print("   ⚠️  Very difficult constraints detected")
            elif self.feasibility_rate < 0.05:
                print("   ⚠️  Difficult constraints detected")
            else:
                print("   ✅ Constraints appear manageable")

    def generate_samples(self, objective, parameters, n_samples, param_converter):
        """Generate constraint-aware samples"""
        if not self.constraints:
            return []
        
        trials = []
        max_attempts = n_samples * 10
        attempts = 0
        bounds = [(0.2, 0.8)] * len(parameters)
        rng = np.random.RandomState(42)
        
        while len(trials) < n_samples and attempts < max_attempts:
            # Generate candidate
            sample = np.array([rng.uniform(low, high) for low, high in bounds])
            params = param_converter.to_params(sample)
            
            try:
                value = objective(params)
                violations = [max(0, c(params)) for c in self.constraints]
                trial = Trial(params=params, value=float(value), 
                            trial_id=len(trials), constraint_violations=violations)
                trials.append(trial)
                
                if trial.is_feasible and len([t for t in trials if t.is_feasible]) <= 5:
                    if self.config.verbose:
                        print(f"   ✅ Feasible point found: {trial.value:.4f}")
                        
            except Exception:
                pass
            
            attempts += 1
            
            # Gradually expand search bounds
            if attempts % 50 == 0:
                bounds = [(max(0, low - 0.1), min(1, high + 0.1)) for low, high in bounds]
        
        return trials


class Foretuner:
    """Streamlined Bayesian optimizer with progressive acquisition switching"""

    def __init__(self, surrogate='auto', acquisition='ucb', n_initial=None, n_parallel=1,
                 patience=None, turbo_config=None, device='auto', verbose=True, 
                 sparse_inducing=50, sparse_strategy='kmeans',
                 initial_sampling_strategy='adaptive', constraint_aware_sampling=True,
                 feasibility_target=0.1, enable_fantasizing=True, fantasy_samples=16,
                 progressive_acquisition_config=None):

        # Core configuration
        self.config = SimpleNamespace(
            surrogate=surrogate, acquisition=acquisition, n_parallel=n_parallel, verbose=verbose,
            sparse_inducing=sparse_inducing, sparse_strategy=sparse_strategy,
            initial_sampling_strategy=initial_sampling_strategy, 
            constraint_aware_sampling=constraint_aware_sampling,
            feasibility_target=feasibility_target, enable_fantasizing=enable_fantasizing,
            fantasy_samples=fantasy_samples
        )
        
        # Adaptive parameters (will be set in optimize())
        self.n_initial = n_initial
        self.patience = patience
        
        # Sub-configurations
        self.turbo_config = turbo_config or TuRBOConfig()
        self.progressive_config = progressive_acquisition_config or ProgressiveAcquisitionConfig(
            initial_acquisition=acquisition
        )

        # State
        self.rng = np.random.RandomState(42)
        self.model = None
        self._params = None
        self._param_converter = None
        
        # Device setup
        self.device = 'cuda' if (device == 'auto' and BOTORCH_AVAILABLE and torch.cuda.is_available()) else device

        # Component managers
        self.surrogate_manager = SurrogateManager()
        self.acquisition_manager = ProgressiveAcquisitionManager(self.progressive_config)
        self.sampling_manager = SamplingManager()
        self.turbo_controller = None
        self.constraint_handler = None

        # Logging
        self.logger = logging.getLogger('Foretuner')
        if verbose:
            self.logger.setLevel(logging.INFO)
            self._log_init()

    def _log_init(self):
        """Log initialization info"""
        print(f"🔥 Using device: {self.device}")
        if self.progressive_config.enable_switching:
            print(f"🧠 Progressive acquisition enabled: {self.progressive_config.initial_acquisition} → adaptive")

    def optimize(self, objective, parameters, n_trials=100, constraints=None, initial_trials=None):
        """Main optimization entry point"""
        # Initialize components
        self._setup_optimization(parameters, constraints, n_trials)
        
        # Get or generate initial trials
        trials = initial_trials or self._generate_initial_trials(objective, parameters, constraints)
        if not trials:
            raise ValueError("No successful initial trials - check objective function and constraints")
        
        # Initialize trust regions
        self.turbo_controller.initialize_from_trials(trials, self._param_converter.to_array)
        
        # Main optimization loop
        trials = self._optimization_loop(objective, trials, n_trials, constraints)
        
        # Return results
        best = self._get_best_trial(trials)
        if self.config.verbose:
            self._report_final_results(best, trials, n_trials)
        
        return best.params, trials

    def _setup_optimization(self, parameters, constraints, n_trials):
        """Initialize optimization components"""
        self._params = parameters
        n_dims = len(parameters)
        
        # Set adaptive defaults
        if self.n_initial is None:
            self.n_initial = max(10, min(20, 2 * n_dims))
        if self.patience is None:
            self.patience = max(20, n_dims)
        
        # Initialize components
        self._param_converter = ParameterConverter(parameters)
        self.turbo_controller = TuRBOController(n_dims, self.turbo_config, self.device)
        self.constraint_handler = ConstraintHandler(constraints, self.config) if constraints else None
        
        if self.config.verbose:
            self._log_optimization_start(n_dims, n_trials)

    def _optimization_loop(self, objective, trials, n_trials, constraints):
        """Main optimization loop with optimized state tracking"""
        state = OptimizationState(trials, self._get_best_value(trials))
        surrogate_update_interval = max(5, len(self._params))
        
        while len(trials) < n_trials:
            iteration = len(trials)
            
            # Batch updates for efficiency
            if iteration % surrogate_update_interval == 0 or self.model is None:
                self._update_surrogate(trials)
            
            #print(f"\n🔄 Iteration {iteration + 1}/{n_trials} - Best value: {state.best_value:.4f}")
            self._update_acquisition_strategy(trials, state.no_improve, n_trials)
            
            # Generate and evaluate candidates
            candidates = self._generate_candidates(trials, constraints)
            new_trials = list(filter(None, self._evaluate_candidates(objective, candidates, iteration, constraints)))
            
            if new_trials:
                trials.extend(new_trials)
                state.update(trials, self._get_best_value(trials))
                self.turbo_controller.update_regions(new_trials, self._param_converter.to_array)
                
                # Progress reporting (less frequent)
                if self.config.verbose and iteration % max(10, n_trials // 10) == 0:
                    self._report_progress(trials, iteration)
            
            # Early stopping check
            if state.no_improve >= self.patience:
                if self.config.verbose:
                    print(f"\n🛑 Early stopping after {state.no_improve} iterations without improvement")
                break
        
        return trials

    def _generate_initial_trials(self, objective, parameters, constraints):
        """Generate initial trials using configured strategy"""
        if self.config.verbose:
            print("🔍 Smart Initial Sampling")
        
        if constraints and self.config.constraint_aware_sampling:
            self.constraint_handler.analyze_difficulty(parameters)
        
        if self.config.initial_sampling_strategy == 'adaptive':
            return self._adaptive_initial_sampling(objective, parameters, constraints)
        else:
            return self._single_strategy_sampling(objective, parameters, constraints, 
                                                self.config.initial_sampling_strategy)

    def _adaptive_initial_sampling(self, objective, parameters, constraints):
        """Try multiple strategies and pick the best"""
        strategies = ['lhs', 'sobol', 'targeted', 'grid', 'random']
        best_trials, best_feasibility = [], 0
        
        for strategy in strategies:
            test_trials = self._single_strategy_sampling(
                objective, parameters, constraints, strategy, max(10, self.n_initial // 3)
            )
            
            if test_trials:
                feasibility_rate = sum(1 for t in test_trials if t.is_feasible) / len(test_trials)
                
                if self.config.verbose:
                    print(f"   {strategy.upper()}: {feasibility_rate:.1%} feasible")
                
                if feasibility_rate > best_feasibility:
                    best_feasibility = feasibility_rate
                    best_trials = test_trials
        
        # Fill remaining samples
        remaining = self.n_initial - len(best_trials)
        if remaining > 0 and constraints and self.config.constraint_aware_sampling:
            additional = self.constraint_handler.generate_samples(
                objective, parameters, remaining, self._param_converter
            )
            best_trials.extend(additional)
        
        return best_trials

    def _single_strategy_sampling(self, objective, parameters, constraints, strategy, n_samples=None):
        """Generate samples using a specific strategy"""
        n_samples = n_samples or self.n_initial
        
        try:
            samples = self.sampling_manager.generate_samples(strategy, len(parameters), n_samples, self.rng)
        except Exception:
            samples = self.sampling_manager.generate_samples('random', len(parameters), n_samples, self.rng)
        
        trials = []
        for i, sample in enumerate(samples):
            params = self._param_converter.to_params(sample)
            trial = self._evaluate_single(objective, params, len(trials), constraints)
            if trial:
                trials.append(trial)
        
        return trials

    def _update_acquisition_strategy(self, trials, no_improve_count, max_trials):
        """Update progressive acquisition strategy"""
        if not self.progressive_config.enable_switching:
            return
        
        optimization_state = {
            'trials': trials,
            'no_improve_count': no_improve_count,
            'max_trials': max_trials,
            'turbo_stats': self.turbo_controller.get_region_stats(),
            'diversity_score': self._calculate_diversity_score(trials)
        }
        
        new_acquisition = self.acquisition_manager.evaluate_switch_conditions(optimization_state)
        
        if new_acquisition:
            old_acquisition = self.acquisition_manager.current_acquisition
            if self.acquisition_manager.switch_acquisition(new_acquisition, "optimization_progress"):
                if self.config.verbose:
                    exploration = self.acquisition_manager.get_current_exploration_factor()
                    print(f"🧠 Acquisition switch: {old_acquisition.upper()} → {new_acquisition.upper()} "
                          f"(exploration: {exploration:.2f})")


    def evaluate_surrogate_model(self, model, X_val, y_val):
        pred = model.predict(X_val)

        if isinstance(pred, tuple) and len(pred) == 2:
            y_pred, _ = pred
        else:
            y_pred = pred
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        return {"mse": mse, "mae": mae}
    
    def _update_surrogate(self, trials):
        if len(trials) < 3:
            return
        
        X = np.array([self._param_converter.to_array(t.params) for t in trials])
        y = np.array([t.value for t in trials])

        candidates = ['botorch', 'rf', 'gp', 'adaptive_gp']
        performances = {}

        for surrogate_type in candidates:
            try:
                model = self.surrogate_manager.create_model(
                    surrogate_type, len(trials), self.config.n_parallel,
                    sparse_inducing=self.config.sparse_inducing,
                    sparse_strategy=self.config.sparse_strategy,
                    enable_fantasizing=self.config.enable_fantasizing
                )
                model.fit(X, y)

                val_metrics = self.evaluate_surrogate_model(model, X, y)  # or cross-validation
                performances[surrogate_type] = val_metrics['mse']
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️ Surrogate {surrogate_type} failed: {e}")
                performances[surrogate_type] = float('inf')
        
        # Select best surrogate
        best_surrogate = min(performances, key=performances.get)
        if self.config.verbose:
            print(f"🔁 Adaptive Surrogate Switch → {best_surrogate.upper()}")

        # Refit best model
        self.model = self.surrogate_manager.create_model(
            best_surrogate, len(trials), self.config.n_parallel,
            sparse_inducing=self.config.sparse_inducing,
            sparse_strategy=self.config.sparse_strategy,
            enable_fantasizing=self.config.enable_fantasizing
        )
        self.model.fit(X, y)
        self.config.surrogate = best_surrogate  # Update state


    def _generate_candidates(self, trials, constraints):
        """Generate candidate points for evaluation"""
        active_regions = self.turbo_controller.get_active_regions()
        
        if not active_regions:
            return [self._random_sample() for _ in range(self.config.n_parallel)]
        
        best_value = self._get_best_value(trials)
        candidates = []
        
        # Use fantasizing for BoTorch models with parallel evaluation
        if (isinstance(self.model, BotorchGP) and self.config.n_parallel > 1 and 
            self.config.enable_fantasizing):
            
            for region_idx in active_regions:
                if len(candidates) >= self.config.n_parallel:
                    break
                    
                bounds = self.turbo_controller.get_trust_region_bounds(region_idx)
                region_candidates = self._optimize_acquisition_batch(bounds, best_value, 
                                                                   min(self.config.n_parallel - len(candidates), 2))
                candidates.extend(region_candidates)
        else:
            # Sequential candidate generation
            for region_idx in active_regions:
                if len(candidates) >= self.config.n_parallel:
                    break
                    
                bounds = self.turbo_controller.get_trust_region_bounds(region_idx)
                candidate = self._optimize_acquisition_single(bounds, best_value, trials)
                if candidate:
                    candidates.append(candidate)
        
        # Fill remaining slots randomly
        while len(candidates) < self.config.n_parallel:
            region_idx = self.rng.choice(active_regions) if active_regions else 0
            bounds = (self.turbo_controller.get_trust_region_bounds(region_idx) 
                     if active_regions else [(0, 1)] * len(self._params))
            candidates.append(self._random_sample_bounded(bounds))
        
        return candidates[:self.config.n_parallel]

    def _tensor_to_numpy(self, tensor_or_array):
        """Safely convert tensor or array to numpy array"""
        if hasattr(tensor_or_array, 'detach'):  # torch tensor
            return tensor_or_array.detach().cpu().numpy()
        return np.asarray(tensor_or_array)

    def _optimize_acquisition_batch(self, bounds, best_value, q=1):
        """Optimize acquisition function for batch candidates"""
        try:
            bounds_tensor = torch.tensor(
                [[low for low, high in bounds], [high for low, high in bounds]],
                dtype=torch.float64, device=self.model.device
            )
            
            acq_func = self.model.get_acquisition_function(
                self.acquisition_manager.current_acquisition, best_value, q=q
            )
            candidates_tensor = self.model.optimize_acquisition(acq_func, bounds_tensor, q=q)
            
            # Use utility method for safe tensor conversion
            candidates_array = self._tensor_to_numpy(candidates_tensor)
            
            if q == 1:
                return [self._param_converter.to_params(candidates_array)]
            else:
                return [self._param_converter.to_params(candidates_array[i]) for i in range(candidates_array.shape[0])]
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Batch acquisition optimization failed: {e}")
            return []

    def _optimize_acquisition_single(self, bounds, best_value, trials):
        """Optimize acquisition function for single candidate"""
        if self.model is None:
            return self._random_sample_bounded(bounds)
        
        if isinstance(self.model, BotorchGP):
            candidates = self._optimize_acquisition_batch(bounds, best_value, q=1)
            return candidates[0] if candidates else self._random_sample_bounded(bounds)
        elif hasattr(self.model, 'predict'):
            return self._scipy_optimize_acquisition(bounds, trials)
        else:
            return self._random_sample_bounded(bounds)

    def _scipy_optimize_acquisition(self, bounds, trials):
        """Optimize acquisition using scipy"""
        try:
            x0 = np.array([self.rng.uniform(low, high) for low, high in bounds])
            
            result = minimize(
                fun=lambda x: -self._acquisition_value(x, trials),
                x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50}
            )
            
            return (self._param_converter.to_params(result.x) 
                   if result.success else self._random_sample_bounded(bounds))
                
        except Exception:
            return self._random_sample_bounded(bounds)

    def _acquisition_value(self, x_array, trials):
        """Calculate acquisition function value"""
        if self.model is None:
            return self.rng.random()
        
        try:
            # Get prediction based on model type
            if isinstance(self.model, BotorchGP):
                mean, std = self.model.predict(x_array.reshape(1, -1))
                mean, std = mean[0], std[0]
            elif hasattr(self.model, 'predict'):
                mean, std = self._get_sklearn_prediction(x_array)
            else:
                return self.rng.random()
            
            best_value = self._get_best_value(trials)
            return self.acquisition_manager.calculate_value(mean, std, best_value)
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Acquisition function failed: {e}")
            return self.rng.random()

    def _get_sklearn_prediction(self, x_array):
        """Get prediction from sklearn-like model"""
        if hasattr(self.model, 'scaler') and self.model.scaler is not None:
            x_scaled = self.model.scaler.transform([x_array])
            mean, std = self.model.predict(x_scaled, return_std=True)
            return mean[0], std[0]
        else:
            pred_result = self.model.predict([x_array])
            if isinstance(pred_result, tuple) and len(pred_result) == 2:
                mean, std = pred_result
                return (mean[0] if isinstance(mean, np.ndarray) else mean,
                       std[0] if isinstance(std, np.ndarray) else std)
            else:
                mean = pred_result[0] if isinstance(pred_result, np.ndarray) else pred_result
                return mean, 0.1

    def _evaluate_candidates(self, objective, candidates, start_id, constraints):
        """Evaluate candidates in parallel or sequential"""
        if self.config.n_parallel == 1:
            trial = self._evaluate_single(objective, candidates[0], start_id, constraints)
            return [trial] if trial else []
        
        # Parallel evaluation with timeout
        try:
            with ThreadPoolExecutor(max_workers=self.config.n_parallel) as executor:
                futures = {
                    executor.submit(self._evaluate_single, objective, candidate, start_id + i, constraints): i
                    for i, candidate in enumerate(candidates)
                }
                
                results = [None] * len(candidates)
                for future in as_completed(futures, timeout=300):
                    idx = futures[future]
                    results[idx] = future.result()
                
                return results
        except Exception:
            # Fallback to sequential
            return [self._evaluate_single(objective, candidate, start_id + i, constraints) 
                   for i, candidate in enumerate(candidates)]

    def _evaluate_single(self, objective, params, trial_id, constraints):
        """Evaluate single candidate"""
        try:
            value = objective(params)
            
            violations = []
            if constraints:
                for constraint in constraints:
                    violation = constraint(params)
                    if violation > 0:
                        violations.append(violation)
            
            return Trial(
                params=params, value=float(value), trial_id=trial_id, constraint_violations=violations
            )
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Trial {trial_id} failed: {e}")
            return None

    def _update_optimization_state(self, trials, iteration, best_value, no_improve):
        """Update optimization state and check for improvement"""
        # This method is now replaced by OptimizationState.update()
        pass

    # Utility methods (simplified and consolidated)
    def _random_sample(self):
        """Generate random sample in unit cube"""
        sample = self.rng.random(len(self._params))
        return self._param_converter.to_params(sample)
    
    def _random_sample_bounded(self, bounds):
        """Generate random sample within bounds"""
        sample = np.array([self.rng.uniform(low, high) for low, high in bounds])
        return self._param_converter.to_params(sample)

    def _get_best_value(self, trials):
        """Get best value (feasible preferred)"""
        feasible = [t for t in trials if t.is_feasible]
        source = feasible if feasible else trials
        return min(t.value for t in source) if source else float('inf')

    def _get_best_trial(self, trials):
        """Get best trial (feasible preferred)"""
        feasible = [t for t in trials if t.is_feasible]
        source = feasible if feasible else trials
        return min(source, key=lambda t: t.value)

    def _calculate_diversity_score(self, trials):
        """Calculate diversity score of recent trials"""
        if len(trials) < 2:
            return 0.0
        
        recent = trials[-min(50, len(trials)):]
        if len(recent) < 2:
            return 0.0
            
        arrays = [self._param_converter.to_array(t.params) for t in recent]
        distances = [np.linalg.norm(arrays[i] - arrays[j]) 
                    for i in range(len(arrays)) for j in range(i + 1, len(arrays))]
        
        if not distances:
            return 0.0
        
        max_distance = np.sqrt(len(arrays[0]))
        return min(1.0, np.mean(distances) / max_distance)

    def _log_optimization_start(self, n_dims, n_trials):
        """Log optimization startup"""
        print(f"\n🚀 Starting TuRBO 2 Optimization ({n_dims}D, {n_trials} trials)")
        print(f"   Trust Regions: {self.turbo_config.n_trust_regions}, Parallel: {self.config.n_parallel}")
        print(f"   Surrogate: {self.config.surrogate}, Acquisition: {self.config.acquisition}")
        print(f"   Initial Sampling: {self.config.initial_sampling_strategy}")

    def _report_progress(self, trials, iteration):
        """Report progress"""
        feasible = [t for t in trials if t.is_feasible]
        best_trial = self._get_best_trial(trials)
        feasible_rate = len(feasible) / len(trials) * 100 if trials else 0
        
        turbo_stats = self.turbo_controller.get_region_stats()
        current_acq = self.acquisition_manager.current_acquisition.upper()
        exploration = self.acquisition_manager.get_current_exploration_factor()
        
        print(f"Trial {iteration:3d}: {best_trial.value:8.4f} | "
              f"Feasible: {feasible_rate:4.1f}% | "
              f"Active: {turbo_stats['active_count']} | "
              f"Acq: {current_acq} ({exploration:.2f}) | "
              f"Radius: {turbo_stats['avg_radius']:.4f}")

    def _report_final_results(self, best_trial, trials, max_trials):
        """Report final results"""
        feasible = [t for t in trials if t.is_feasible]
        feasible_rate = len(feasible) / len(trials) * 100 if trials else 0
        
        print(f"\n🎯 TuRBO 2 Optimization Complete!")
        print(f"📊 {len(trials)} trials, {feasible_rate:.1f}% feasible")
        print(f"🏆 Best value: {best_trial.value:.6f}")
        print(f"🎪 Best params: {best_trial.params}")
        
        turbo_stats = self.turbo_controller.get_region_stats()
        print(f"\n📈 Trust Region Statistics:")
        print(f"   Active: {turbo_stats['active_count']}/{self.turbo_config.n_trust_regions}")
        print(f"   Restarts: {turbo_stats['total_restarts']}")
        print(f"   Radius: [{turbo_stats['min_radius']:.4f}, {turbo_stats['max_radius']:.4f}]")
        
        if self.progressive_config.enable_switching:
            acq_stats = self.acquisition_manager.get_switch_summary()
            print(f"\n🧠 Progressive Acquisition:")
            print(f"   Final: {acq_stats['current_acquisition'].upper()}")
            print(f"   Switches: {acq_stats['total_switches']}")
            if acq_stats['switch_history']:
                sequence = [acq_stats['switch_history'][0]['from']]
                sequence.extend(s['to'] for s in acq_stats['switch_history'])
                print(f"   Sequence: {' → '.join(s.upper() for s in sequence)}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from typing import List
from pandas.plotting import parallel_coordinates
import pandas as pd

# Optional: mplcursors for interactive plots (Jupyter)
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

def plot_optimization_results(trials: List, title: str = "Enhanced Foretuner Results"):
    """State-of-the-art optimization visualization for Foretuner trials"""
    
    feasible_trials = [t for t in trials if t.is_feasible]
    all_values = [t.value for t in trials]
    feasible_values = [t.value for t in feasible_trials]

    if not feasible_values:
        feasible_values = all_values
        print("⚠️ No feasible trials found, showing all trials")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # --- Convergence Plot ---
    ax = axes[0, 0]
    ax.plot(all_values, 'o-', alpha=0.4, label='All', color='lightblue')
    
    feasible_indices = [i for i, t in enumerate(trials) if t.is_feasible]
    ax.plot(feasible_indices, feasible_values, 'o-', alpha=0.8, label='Feasible', color='blue')

    best_values = [min(feasible_values[:i+1]) for i in range(len(feasible_values))]
    ax.plot(feasible_indices, best_values, 'r-', linewidth=3, label='Best Feasible')

    # Optional: Initial BO cutoff
    init_cutoff = len([t for t in trials if getattr(t, 'is_initial', False)])
    if init_cutoff > 0:
        ax.axvline(init_cutoff, color='gray', linestyle='--', label='Start BO')

    ax.set_title(f"{title} - Convergence")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate best point
    if feasible_values:
        best_idx = feasible_indices[np.argmin(feasible_values)]
        best_val = min(feasible_values)
        ax.annotate(f"Best: {best_val:.4f}", xy=(best_idx, best_val),
                    xytext=(10, -20), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="green"),
                    fontsize=9, color="green")
    
    # --- Log Convergence ---
    ax = axes[0, 1]
    best_values_pos = np.maximum(best_values, 1e-10)
    ax.semilogy(best_values_pos, 'r-', linewidth=2)
    ax.set_title("Log Convergence")
    ax.set_xlabel("Feasible Trial")
    ax.set_ylabel("Best Value (log)")
    ax.grid(True, alpha=0.3)

    # --- Value Distribution (Histogram or KDE) ---
    ax = axes[0, 2]
    if len(feasible_values) > 1:
        try:
            sns.kdeplot(feasible_values, ax=ax, fill=True, color='skyblue')
            ax.axvline(min(feasible_values), color='red', linestyle='--', linewidth=2, label=f"Best: {min(feasible_values):.4f}")
            ax.legend()
        except Exception:
            ax.hist(feasible_values, bins='auto', edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_title("Objective Value Distribution")
    else:
        ax.text(0.5, 0.5, f'Single Value:\n{feasible_values[0]:.4f}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xlabel("Objective Value")
    ax.set_ylabel("Density/Frequency")
    ax.grid(True, alpha=0.3)

    # --- Constraint Violations ---
    ax = axes[1, 0]
    constraint_counts = [len(t.constraint_violations) if hasattr(t, 'constraint_violations') else 0 for t in trials]
    if any(constraint_counts):
        ax.plot(constraint_counts, 'o-', color='orange', alpha=0.7)
        ax.set_title("Constraint Violations")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Violations")
    else:
        ax.text(0.5, 0.5, 'No Constraints\nor All Feasible',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Constraint Status")
    ax.grid(True, alpha=0.3)

    # --- Improvement Rate ---
    ax = axes[1, 1]
    if len(best_values) > 10:
        window = min(10, len(best_values) // 4)
        improvements = [(best_values[i - window] - best_values[i]) / (abs(best_values[i - window]) + 1e-10)
                        for i in range(window, len(best_values))]
        ax.plot(range(window, len(best_values)), improvements, 'g-', linewidth=2)
        ax.set_title("Improvement Rate")
        ax.set_xlabel(f"Trial (window: {window})")
        ax.set_ylabel("Relative Improvement")
    else:
        ax.text(0.5, 0.5, 'Insufficient Data\nfor Rate Analysis',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Parameter Space (2D or message) ---
    ax = axes[1, 2]
    if trials and len(trials[0].params) >= 2:
        pnames = list(trials[0].params.keys())[:2]
        x = [t.params[pnames[0]] for t in feasible_trials]
        y = [t.params[pnames[1]] for t in feasible_trials]
        c = [t.value for t in feasible_trials]

        sc = ax.scatter(x, y, c=c, cmap='viridis_r', s=50, edgecolors='k', alpha=0.7)
        ax.set_xlabel(pnames[0])
        ax.set_ylabel(pnames[1])
        ax.set_title("2D Parameter Space")
        plt.colorbar(sc, ax=ax)

        # Best marker
        best_trial = min(feasible_trials, key=lambda t: t.value)
        ax.scatter(best_trial.params[pnames[0]], best_trial.params[pnames[1]], 
                   c='red', marker='*', s=200, edgecolors='black', linewidth=2,
                   label='Best')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Cannot visualize\nparameter space',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Parameter Space")

    plt.tight_layout()
    plt.show()

    # --- Optional: Parallel Coordinates if >2D ---
    if len(trials[0].params) > 2:
        df = pd.DataFrame([dict(**t.params, value=t.value) for t in feasible_trials])
        df['label'] = pd.qcut(df['value'], q=3, labels=["High", "Medium", "Low"])
        plt.figure(figsize=(12, 6))
        parallel_coordinates(df[['label'] + list(trials[0].params.keys())], class_column='label', colormap='coolwarm', alpha=0.6)
        plt.title("Parallel Coordinates (Parameter Patterns)")
        plt.grid(True, alpha=0.3)
        plt.show()

    # --- Optional: Parameter Correlation Heatmap ---
    df_params = pd.DataFrame([t.params for t in feasible_trials])
    if not df_params.empty:
        df_params['value'] = feasible_values
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_params.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Parameter Correlation Matrix")
        plt.tight_layout()
        plt.show()

    # --- Optional: Interactive hover ---
    if MPLCURSORS_AVAILABLE:
        mplcursors.cursor(hover=True)

    # --- Summary Statistics ---
    print("\n📊 Optimization Summary:")
    print(f"   Total trials: {len(trials)}")
    print(f"   Feasible trials: {len(feasible_trials)} ({len(feasible_trials)/len(trials)*100:.1f}%)")
    print(f"   Best value: {min(feasible_values):.6f}")
    print(f"   Value range: {max(feasible_values) - min(feasible_values):.6f}")
    if len(best_values) > 10:
        final_improv = (best_values[-10] - best_values[-1]) / (abs(best_values[-10]) + 1e-10)
        print(f"   Final convergence rate (last 10): {final_improv:.4f}")


def plot_sparse_gp_inducing_points(optimizer: Foretuner, trials: List[Trial]):
    """Visualize SparseGP inducing points if available"""
    if not hasattr(optimizer.model, 'get_inducing_points'):
        print("⚠️ Model doesn't have inducing points to visualize")
        return
    
    inducing_points = optimizer.model.get_inducing_points()
    if inducing_points is None:
        print("⚠️ No inducing points available")
        return
    
    # Get trial data
    X = np.array([optimizer._params_to_array(t.params) for t in trials])
    y = np.array([t.value for t in trials])
    
    # Only visualize if 2D or can project to 2D
    n_dims = X.shape[1]
    if n_dims == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot trials
        scatter = ax.scatter(X[:, 0], y, c=y, cmap='viridis_r', alpha=0.7, s=50, 
                           edgecolors='black', linewidth=0.5, label='Trials')
        
        # Plot inducing points
        inducing_y = np.interp(inducing_points[:, 0], 
                              sorted(range(len(y)), key=lambda i: X[i, 0]),
                              sorted(y))
        ax.scatter(inducing_points[:, 0], inducing_y, c='red', s=100, marker='^', 
                  edgecolors='black', linewidth=2, label=f'Inducing Points ({len(inducing_points)})')
        
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Objective Value')
        ax.set_title('SparseGP Inducing Points (1D)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
        
    elif n_dims >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D parameter space
        ax = axes[0]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis_r', alpha=0.7, s=50,
                           edgecolors='black', linewidth=0.5, label='Trials')
        ax.scatter(inducing_points[:, 0], inducing_points[:, 1], c='red', s=100, marker='^',
                  edgecolors='black', linewidth=2, label=f'Inducing Points ({len(inducing_points)})')
        
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title('SparseGP Inducing Points (Parameter Space)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
        
        # Distribution of inducing points vs trials
        ax = axes[1]
        ax.hist2d(X[:, 0], X[:, 1], bins=20, alpha=0.6, cmap='Blues', label='Trial Density')
        ax.scatter(inducing_points[:, 0], inducing_points[:, 1], c='red', s=100, marker='^',
                  edgecolors='black', linewidth=2, label='Inducing Points')
        
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title('Inducing Points vs Trial Density')
        ax.legend()
        
    plt.tight_layout()
    plt.show()
    
    # Print inducing point statistics
    print(f"\n🎯 SparseGP Inducing Points Analysis:")
    print(f"   Number of inducing points: {len(inducing_points)}")
    print(f"   Coverage efficiency: {len(inducing_points)/len(trials)*100:.1f}%")
    
    # Compute distances between inducing points
    if len(inducing_points) > 1:
        distances = []
        for i in range(len(inducing_points)):
            for j in range(i+1, len(inducing_points)):
                dist = np.linalg.norm(inducing_points[i] - inducing_points[j])
                distances.append(dist)
        
        print(f"   Avg distance between inducing points: {np.mean(distances):.4f}")
        print(f"   Min distance: {np.min(distances):.4f}")
        print(f"   Max distance: {np.max(distances):.4f}")