import numpy as np
import numba
from numba import jit, prange
from functools import lru_cache, partial
from dataclasses import dataclass
from typing import Tuple, Callable, List, Dict, Any
from scipy.stats import chi2, norm
import warnings
warnings.filterwarnings('ignore')
from scipy.linalg import sqrtm

import numpy as np
import torch
from typing import List, Tuple, Optional
from scipy.stats import chi2
from scipy.linalg import sqrtm
from scipy.stats.qmc import Sobol

@dataclass
class TurboConfig:
    n_init: int = 15
    max_evals: int = 100
    batch_size: int = 1
    n_regions: int = 4
    init_radius: float = 0.25
    min_radius: float = 1e-3
    max_radius: float = 0.9
    expansion_factor: float = 1.5
    contraction_factor: float = 0.8
    acquisition: str = "ts"  # Thompson Sampling as default
    ucb_beta: float = 3.0
    update_frequency: int = 5
    management_frequency: int = 10
    min_local_samples: int = 6
    max_local_data: int = 50
    n_candidates: int = 50
    # Enhanced parameters
    spawn_threshold: float = 0.6
    merge_threshold: float = 0.1
    kill_threshold: int = 15
    diversity_weight: float = 0.2
    exploration_factor: float = 0.4
    restart_threshold: int = 25
    # Thompson Sampling parameters
    n_thompson_samples: int = 10
    n_fantasy_samples: int = 5
    ts_min_variance: float = 1e-6
    # Diversity and selection
    diversity_metric: str = "kl"  # Options: "kl", "wasserstein", "euclidean"
    selection_method: str = "nsga2"  # Options: "nsga", "nsga2", "random"

# Utility functions
@jit(nopython=True)
def sobol_sequence(seed, n_points, n_dims):
    """Improved quasi-random sequence generation"""
    np.random.seed(seed)
    base_samples = np.random.uniform(0, 1, (n_points, n_dims))
    
    # Enhanced scrambling using golden ratio and prime numbers
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    golden_ratio = 0.618033988749
    
    # Apply dimension-specific scrambling
    for i in range(min(n_dims, len(primes))):
        prime_offset = primes[i] * golden_ratio
        base_samples[:, i] = (base_samples[:, i] + prime_offset) % 1.0
    
    # For dimensions beyond available primes, use simple golden ratio scrambling
    if n_dims > len(primes):
        for i in range(len(primes), n_dims):
            offset = i * golden_ratio
            base_samples[:, i] = (base_samples[:, i] + offset) % 1.0
    
    return base_samples


import numpy as np
from numba import jit

@jit(nopython=True, inline='always')
def safe_sqrt(x):
    x_real = float(np.real(x))
    return np.sqrt(x_real) if x_real > 0.0 else 0.0
from numba import jit
import numpy as np


@jit(nopython=True)
def compute_coverage(X, centers, radii):
    """Compute coverage of search space by regions"""
    n_points = X.shape[0]
    n_regions = centers.shape[0]
    covered = np.zeros(n_points, dtype=np.bool_)
    
    for i in range(n_points):
        for j in range(n_regions):
            dist = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - centers[j, k]
                dist += diff * diff
            dist = safe_sqrt(dist)
            
            if dist <= radii[j] * 2.0:
                covered[i] = True
                break
    
    return np.mean(covered.astype(np.float64))

@jit(nopython=True)
def compute_diversity_penalty(candidates, existing_candidates, weight=0.1):
    """Enhanced diversity penalty"""
    if existing_candidates.shape[0] == 0:
        return np.zeros(candidates.shape[0])
    
    penalties = np.zeros(candidates.shape[0])
    
    for i in range(candidates.shape[0]):
        min_dist = np.inf
        for j in range(existing_candidates.shape[0]):
            dist = 0.0
            for k in range(candidates.shape[1]):
                diff = candidates[i, k] - existing_candidates[j, k]
                dist += diff * diff
            dist = safe_sqrt(dist)
            if dist < min_dist:
                min_dist = dist
        
        penalties[i] = -weight * np.exp(-min_dist * 5)
    
    return penalties



class TrustRegion:
    """Enhanced trust region with improved lifecycle management"""
    
    def __init__(self, center, radius, region_id, n_dims):
        self.center = np.array(center)
        self.radius = radius
        self.region_id = region_id
        self.n_dims = n_dims
        
        # Performance tracking
        self.best_value = float('inf')
        self.trial_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.age = 0
        self.last_improvement = 0
        self.stagnation_count = 0
        
        # Adaptive geometry
        self.cov = np.eye(n_dims) * (radius**2)
        
        # Local data
        self.local_X = np.empty((0, n_dims))
        self.local_y = np.array([])
        
        # Health metrics
        self.spawn_score = 0.0
        self.exploration_bonus = 1.0

        self.gradient_sampling_ratio: float  # 0.0 to 1.0
        self.exploration_weight: float       # scales std bonus
        self.radius_adapt_factor: float      # scales expansion/contraction

    def adapt_region_parameters(self):
        # Decay exploration if stable
        if self.stagnation_count < 3 and self.success_rate > 0.3:
            self.exploration_weight *= 0.95
        else:
            self.exploration_weight *= 1.05
            self.exploration_weight = min(self.exploration_weight, 3.0)

        # Bias toward gradient if stagnating
        if self.stagnation_count > 5:
            self.gradient_sampling_ratio = min(1.0, self.gradient_sampling_ratio + 0.1)
        else:
            self.gradient_sampling_ratio *= 0.9

        # Radius adapt scaling based on last improvement
        if self.last_improvement > 10:
            self.radius_adapt_factor *= 1.1
        else:
            self.radius_adapt_factor *= 0.95


    def maybe_reset_cov(self, threshold=1e6):
        """Reset covariance matrix if it's ill-conditioned."""
        try:
            eigvals = np.linalg.eigvals(self.cov)
            cond = np.max(eigvals) / np.clip(np.min(eigvals), 1e-8, None)
            if cond > threshold:
                self.cov = np.eye(self.n_dims) * (self.radius ** 2)
        except:
            self.cov = np.eye(self.n_dims) * (self.radius ** 2)

    def update(self, point, value, config):
        """Enhanced update with better stagnation detection"""
        old_best = self.best_value
        
        # Update statistics
        self.trial_count += 1
        self.age += 1
        
        is_improvement = value < self.best_value
        improvement_amount = max(0, self.best_value - value)
        
        if is_improvement:
            self.success_count += 1
            self.consecutive_failures = 0
            self.last_improvement = 0
            self.best_value = value
            self.stagnation_count = 0
            
            # Significant improvement resets exploration bonus
            if improvement_amount > 0.01:
                self.exploration_bonus = 1.0
        else:
            self.consecutive_failures += 1
            self.last_improvement += 1
            
            # Build up stagnation
            if self.consecutive_failures > 3:
                self.stagnation_count += 1
        
        # Enhanced geometry update
        self._update_geometry(point, value, old_best, config)
        
        # Update local data
        self.local_X = np.vstack([self.local_X, point.reshape(1, -1)])
        self.local_y = np.append(self.local_y, value)
        
        # Trim data
        if len(self.local_y) > config.max_local_data:
            self.local_X = self.local_X[-config.max_local_data:]
            self.local_y = self.local_y[-config.max_local_data:]
        
        # Update spawn score and exploration bonus
        self.spawn_score = self.success_rate * (1 - self.consecutive_failures / 15)
        
        # Increase exploration bonus if stagnating
        if self.stagnation_count > 5:
            self.exploration_bonus = min(2.0, self.exploration_bonus * 1.1)

    def _update_geometry(self, point, value, old_best, config):
        """Enhanced ellipsoidal geometry update with anti-stagnation and stability fixes."""
        is_improvement = value < old_best
        improvement_amount = max(0, old_best - value)

        # Center update with adaptive momentum
        base_momentum = 0.1 if self.trial_count > 5 else 0.3
        stagnation_penalty = min(0.3, self.stagnation_count * 0.02)
        momentum = max(0.05, base_momentum - stagnation_penalty)

        if is_improvement:
            self.center = momentum * point + (1 - momentum) * self.center

        # Covariance scaling
        if is_improvement:
            if improvement_amount > 0.05:
                expansion = min(config.expansion_factor * 1.5, 2.0)
            else:
                expansion = config.expansion_factor
            scaling_factor = expansion
        elif self.consecutive_failures >= 2:
            base_contraction = config.contraction_factor
            if self.stagnation_count > 8:
                contraction = (base_contraction + 1.0) / 2
            else:
                contraction = base_contraction ** min(self.consecutive_failures - 1, 2)
            scaling_factor = contraction
        else:
            scaling_factor = 1.0

        # Apply scaling to covariance
        self.cov = self.cov * (scaling_factor ** 2)

        # Radius from covariance (mean eigen semi-axis length)
        try:
            eigvals = np.linalg.eigvals(self.cov)
            mean_axis = np.mean(np.sqrt(np.maximum(eigvals, 1e-8)))
            self.radius = np.clip(mean_axis, config.min_radius, config.max_radius)
            
            # Sync radius with covariance
            scale = self.radius / mean_axis
            self.cov = self.cov * (scale**2)
        except:
            self.radius = np.clip(self.radius * scaling_factor, config.min_radius, config.max_radius)
            self.cov = np.eye(self.n_dims) * (self.radius**2)

        # Directional update
        direction = point - self.center
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 1e-10:
            direction_normalized = direction / direction_norm
            outer_product = np.outer(direction_normalized, direction_normalized)

            base_lr = 0.1 if is_improvement else -0.03
            stagnation_boost = min(0.05, self.stagnation_count * 0.005)
            learning_rate = base_lr + (stagnation_boost if is_improvement else 0)

            lr_decay = 0.1
            self.cov = (1 - lr_decay) * self.cov + learning_rate * outer_product

            # Ensure positive definiteness
            try:
                eigvals, eigvecs = np.linalg.eigh(self.cov)
                eigvals = np.maximum(eigvals, 1e-8)
                self.cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                self.cov = np.eye(self.n_dims) * (self.radius**2)

        self.radius = np.real(self.radius)
        self.center = np.real(self.center)
        self.maybe_reset_cov()
    
    @property
    def success_rate(self):
        return self.success_count / max(1, self.trial_count)
    
    @property
    def is_active(self):
        return self.radius > 1e-8
    
    @property
    def should_kill(self):
        """Enhanced killing criteria"""
        base_kill = (self.consecutive_failures > 15 and 
                    self.last_improvement > 25 and 
                    self.trial_count > 20)
        
        return base_kill and self.success_rate < 0.05
    
    @property
    def should_restart(self):
        """Determine if region should be restarted"""
        return (self.stagnation_count > 15 and 
                self.radius < 0.05 and 
                self.consecutive_failures > 8)
    
    @property
    def health_score(self):
        """Enhanced health score"""
        base_health = self.success_rate
        recency_bonus = max(0, 1 - self.last_improvement / 30)
        exploration_factor = self.exploration_bonus - 1.0
        
        return base_health + 0.1 * recency_bonus + 0.05 * exploration_factor

import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# Sparse GP imports
from botorch.models.approximate_gp import SingleTaskVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean

# For Bayesian NN fallback
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Bayesian NN ---------------------------------

class BayesianNN(nn.Module):
    """Bayesian Neural Network surrogate using MC Dropout for uncertainty"""
    def __init__(self, input_dim, hidden_dim=64, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout_p = dropout_p

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_p, training=True)  # MC dropout
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        return self.fc_out(x)

    def predict(self, x, n_samples=20):
        preds = []
        for _ in range(n_samples):
            preds.append(self.forward(x))
        preds = torch.stack(preds, dim=0)  # (samples, N, 1)
        mean = preds.mean(0).squeeze(-1)
        std = preds.std(0).squeeze(-1)
        return mean, std

# ----------------------------- Sparse GP -----------------------------------

from botorch.models.approximate_gp import SingleTaskVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

import torch
from botorch.models.approximate_gp import SingleTaskVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from botorch.models.transforms.outcome import Standardize

class SparseGPModel(SingleTaskVariationalGP):
    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor, inducing_points: torch.Tensor):
        # Ensure correct dtype/device
        train_X = train_X.double()
        train_Y = train_Y.double()
        inducing_points = inducing_points.double()
        
        # Create variational distribution for m inducing points
        m = inducing_points.size(0)
        variational_distribution = CholeskyVariationalDistribution(m)
        
        # Call parent SingleTaskVariationalGP with parameters
        # Let the parent handle the variational strategy creation
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_points=True,
            outcome_transform=Standardize(m=1),
        )


import functools
import numpy as np

def _candidates_key(candidates: np.ndarray) -> tuple[int, bytes]:
    """Return (n_dim, raw_bytes) tuple as key"""
    return candidates.shape[1], np.round(candidates, 6).astype(np.float64).tobytes()

# ----------------------------- Surrogate Manager ---------------------------
class SurrogateManager:
    """
    Multi-backend surrogate for TuRBO-M:
    - global_backend used for the global model (exact_gp, sparse_gp, bnn)
    - local_backend used for trust-region models (exact_gp, sparse_gp, bnn)
    """

    def __init__(self, config, device="cpu",
                 global_backend="exact_gp", local_backend="exact_gp",
                 normalize_inputs=True, n_inducing=50,
                 bnn_hidden=64, bnn_dropout=0.1):
        self.config = config
        self.device = torch.device(device)

        # Separate backends
        self.global_backend = global_backend
        self.local_backend = local_backend

        self.normalize_inputs = normalize_inputs
        self.n_inducing = n_inducing
        self.bnn_hidden = bnn_hidden
        self.bnn_dropout = bnn_dropout

        # Global data + model
        self.global_X = None
        self.global_y = None
        self.global_model = None

        # Local trust-region cache
        self.local_model_cache = {}

        self._cached_posterior = self._make_cached_predict_fn()

    def _make_cached_predict_fn(self):
        @lru_cache(maxsize=128)
        def _cached_predict(n_dim: int, candidates_bytes: bytes, backend_name: str, model_version: int):
            candidates = np.frombuffer(candidates_bytes, dtype=np.float64).reshape(-1, n_dim)
            mean, std = self._predict_from_model(
                self.global_model,
                self._to_tensor(candidates),
                self.global_X,
                backend=backend_name
            )
            return mean, std
        return _cached_predict

    def predict_global_cached(self, candidates: np.ndarray):
        if self.global_model is None:
            return np.zeros(len(candidates)), np.ones(len(candidates))

        n_dim, key_bytes = _candidates_key(candidates)
        backend = self.global_backend
        version = getattr(self, "_model_version", 0)
        return self._cached_posterior(n_dim, key_bytes, backend, version)


    def clear_posterior_cache(self):
        self._cached_posterior.cache_clear()

    def _to_tensor(self, X):
        return torch.as_tensor(X, dtype=torch.double, device=self.device)

    def _normalize_inputs(self, X, ref_X):
        if not self.normalize_inputs:
            return X
        return (X - ref_X.min(0)[0]) / (ref_X.max(0)[0] - ref_X.min(0)[0] + 1e-8)

    # -------------------- GLOBAL MODEL -------------------- #
    # Inside SurrogateManager

    def update_global_data(self, X, y):
        """Update global dataset and retrain global model if enough points"""
        self.global_X = self._to_tensor(X)
        self.global_y = self._to_tensor(y).unsqueeze(-1)

        if self.global_X.shape[0] >= 5:
            self.global_model = self._fit_model(self.global_X, self.global_y,
                                                backend=self.global_backend)

    def update_data(self, X, y):
        """Alias for update_global_data for backward compatibility"""
        return self.update_global_data(X, y)
    # -------------------- MODEL FITTING -------------------- #
    def _fit_model(self, X, y, backend="exact_gp"):
        d = X.shape[-1]
        train_X = self._normalize_inputs(X, X)
        
        if backend == "exact_gp":
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=y,
                input_transform=Normalize(d=d) if self.normalize_inputs else None,
                outcome_transform=Standardize(m=1),
            ).to(self.device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            return model
                
        elif backend == "sparse_gp":
                n_inducing = min(self.n_inducing, train_X.shape[0])
                idx = torch.randperm(train_X.shape[0])[:n_inducing]
                # Make sure it's a plain tensor (no transforms)
                inducing_points = train_X[idx].clone().detach()
                
                model = SparseGPModel(
                    train_X=train_X,
                    train_Y=y,
                    inducing_points=inducing_points
                ).to(self.device)
                
                # Use VariationalELBO for variational GPs, not ExactMarginalLogLikelihood
                from gpytorch.mlls import VariationalELBO
                # Pass the underlying GP model, not the wrapper
                mll = VariationalELBO(model.likelihood, model.model, num_data=y.size(0))
                
                # Use controlled training for variational GPs
                model.train()
                model.likelihood.train()
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                
                # Quick training loop - adjust iterations as needed
                for i in range(10):  # Reduced from default to speed up
                    optimizer.zero_grad()
                    output = model(train_X)
                    loss = -mll(output, y.squeeze(-1))
                    loss.backward()
                    optimizer.step()
                    
                    # Early stopping if converged
                    if i > 10 and i % 10 == 0:
                        if abs(loss.item()) < 1e-6:
                            break
                
                model.eval()
                model.likelihood.eval()
                return model
        
        elif backend == "bnn":
            model = BayesianNN(d, hidden_dim=self.bnn_hidden, dropout_p=self.bnn_dropout).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(500):
                opt.zero_grad()
                preds = model(train_X.float())
                loss = F.mse_loss(preds, y.float())
                loss.backward()
                opt.step()
            return model
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    # -------------------- PREDICTIONS -------------------- #
    def _predict_from_model(self, model, Xq, ref_X, backend):
        Xq_norm = self._normalize_inputs(Xq, ref_X)

        if backend in ["exact_gp", "sparse_gp"]:
            posterior = model.posterior(Xq_norm)
            mean = posterior.mean.detach().cpu().numpy().flatten()
            std = posterior.variance.sqrt().detach().cpu().numpy().flatten()
            return mean, std

        elif backend == "bnn":
            model.eval()
            with torch.no_grad():
                mean_t, std_t = model.predict(Xq_norm.float(), n_samples=20)
            return mean_t.cpu().numpy(), std_t.cpu().numpy()

    def predict_global(self, X_test):
        if self.global_model is None:
            mean = np.mean(self.global_y.cpu().numpy()) if self.global_y is not None else 0.0
            return np.ones(X_test.shape[0]) * mean, np.ones(X_test.shape[0]) * 1.0

        return self._predict_from_model(self.global_model,
                                        self._to_tensor(X_test),
                                        self.global_X,
                                        backend=self.global_backend)

    def predict_local(self, X_test, local_X, local_y, region_key=None):
        if region_key in self.local_model_cache:
            model = self.local_model_cache[region_key]
        else:
            if len(local_y) < 3:
                return self.predict_global_cached(X_test)
            model = self._fit_model(self._to_tensor(local_X),
                                    self._to_tensor(local_y).unsqueeze(-1),
                                    backend=self.local_backend)
            self.local_model_cache[region_key] = model

        return self._predict_from_model(model,
                                        self._to_tensor(X_test),
                                        self._to_tensor(local_X),
                                        backend=self.local_backend)

    # -------------------- POSTERIOR SAMPLES -------------------- #
    def predict_local_with_posterior(self, X_test, local_X, local_y,
                                     seed=0, n_samples=5, region_key=None):
        torch.manual_seed(seed)
        if region_key in self.local_model_cache:
            model = self.local_model_cache[region_key]
        else:
            if len(local_y) < 3:
                mean, std = self.self.predict_global_cached(X_test)
                rng = np.random.default_rng(seed)
                return rng.normal(mean, std, size=(n_samples, len(X_test)))
            model = self._fit_model(self._to_tensor(local_X),
                                    self._to_tensor(local_y).unsqueeze(-1),
                                    backend=self.local_backend)
            self.local_model_cache[region_key] = model

        backend = self.local_backend
        Xq = self._to_tensor(X_test)
        if backend in ["exact_gp", "sparse_gp"]:
            posterior = model.posterior(self._normalize_inputs(Xq, self._to_tensor(local_X)))
            samples = posterior.rsample(sample_shape=torch.Size([n_samples]))
            return samples.detach().cpu().numpy().squeeze(-1)
        elif backend == "bnn":
            model.eval()
            preds = []
            for _ in range(n_samples):
                preds.append(model(self._normalize_inputs(Xq, self._to_tensor(local_X)).float()))
            preds = torch.stack(preds)
            return preds.detach().cpu().numpy().squeeze(-1)

    def gp_posterior_samples(self, X_test, seed=0, n_samples=1):
        """Posterior samples from global model."""
        torch.manual_seed(seed)
        Xq = self._to_tensor(X_test)
        if self.global_model is None:
            mean = np.zeros(X_test.shape[0])
            std = np.ones(X_test.shape[0])
            rng = np.random.default_rng(seed)
            return rng.normal(mean, std, size=(n_samples, len(X_test)))

        backend = self.global_backend
        Xq_norm = self._normalize_inputs(Xq, self.global_X)

        if backend in ["exact_gp", "sparse_gp"]:
            posterior = self.global_model.posterior(Xq_norm)
            samples = posterior.rsample(sample_shape=torch.Size([n_samples]))
            return samples.detach().cpu().numpy().squeeze(-1)
        elif backend == "bnn":
            self.global_model.eval()
            preds = []
            for _ in range(n_samples):
                preds.append(self.global_model(Xq_norm.float()))
            preds = torch.stack(preds)
            return preds.detach().cpu().numpy().squeeze(-1)

    def get_best_value(self):
        return self.global_y.min().item() if self.global_y is not None else float("inf")
    


# Standard acquisition functions
@jit(nopython=True)
def expected_improvement(mean, std, best_value):
    """Vectorized Expected Improvement acquisition function"""
    improvement = best_value - mean
    z = improvement / (std + 1e-9)
    ei = improvement * norm_cdf(z) + std * norm_pdf(z)
    return ei

def log_expected_improvement(mean, std, best_value, eps=1e-9):
    """
    Log-EI for numerical stability when std is very small.
    """
    from scipy.stats import norm
    diff = best_value - mean - eps
    z = diff / (std + eps)
    ei = diff * norm.cdf(z) + std * norm.pdf(z)
    # Avoid log(0) by adding eps
    return np.log(ei + eps)

@jit(nopython=True)
def norm_cdf(x):
    """Numba-compatible normal CDF approximation"""
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

@jit(nopython=True)
def norm_pdf(x):
    """Numba-compatible normal PDF"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

@jit(nopython=True)
def upper_confidence_bound(mean, std, beta):
    """Upper Confidence Bound acquisition function"""
    return -mean + beta * std

@jit(nopython=True)
def probability_improvement(mean, std, best_value):
    """Probability of Improvement acquisition function"""
    improvement = best_value - mean
    z = improvement / (std + 1e-9)
    return norm_cdf(z)

@jit(nopython=True)
def predictive_entropy_search(mean: np.ndarray, std: np.ndarray, best_value: float) -> np.ndarray:
    return std

@jit(nopython=True)
def knowledge_gradient(mean: np.ndarray, std: np.ndarray, best_value: float) -> np.ndarray:
    # Clip std to avoid division by zero
    std_safe = np.maximum(std, 1e-9)
    z = (best_value - mean) / std_safe
    kg = std_safe * norm_pdf(z) + (best_value - mean) * norm_cdf(z)
    return np.maximum(kg, 0.0)

@jit(nopython=True)
def noisy_expected_improvement(mean: np.ndarray, std: np.ndarray, best_value: float, noise: float) -> np.ndarray:
    eff_std = np.sqrt(std**2 + noise**2)
    return expected_improvement(mean, eff_std, best_value)

@jit(nopython=True)
def rbf_kernel_matrix(X: np.ndarray, lengthscale: float = 0.5) -> np.ndarray:
    n = X.shape[0]
    K = np.zeros((n, n))
    inv_lengthscale_sq = 1.0 / (lengthscale * lengthscale)
    
    for i in range(n):
        for j in range(n):
            sqdist = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - X[j, k]
                sqdist += diff * diff
            K[i, j] = np.exp(-0.5 * inv_lengthscale_sq * sqdist)
    
    return K

def dpp_select(X: np.ndarray, scores: np.ndarray, batch_size: int = 5, lengthscale: float = 0.5) -> np.ndarray:
    n = len(X)
    if batch_size >= n:
        return np.argsort(-scores)[:batch_size]
    
    selected = []
    remaining = list(range(n))
    K = rbf_kernel_matrix(X, lengthscale)
    
    for _ in range(batch_size):
        best_idx = None
        best_value = -np.inf
        for j in remaining:
            diversity_penalty = np.sum(K[j, selected]) if selected else 0.0
            gain = scores[j] - 0.1 * diversity_penalty
            if gain > best_value:
                best_value = gain
                best_idx = j
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return np.array(selected)
class AcquisitionManager:
    """State-of-the-art acquisition manager with dynamic policies and advanced batch strategies."""
    
    def __init__(self, config):
        self.config = config
        self.acquisition_type = config.acquisition
        self.iteration = 0
        
        # Pre-compute acquisition function mapping for efficiency
        self.acquisition_functions = {
            "ei": expected_improvement,
            "log_ei": log_expected_improvement,
            "ucb": lambda mean, std, best_value: upper_confidence_bound(mean, std, self.config.ucb_beta),
            "pi": probability_improvement,
            "nei": lambda mean, std, best_value: noisy_expected_improvement(mean, std, best_value, self.config.obs_noise),
            "pes": predictive_entropy_search,
            "kg": knowledge_gradient,
            "ts": lambda mean, std, best_value: np.zeros_like(mean),  # Thompson sampling handled separately
        }
        
        # Smart auto schedule based on optimization theory
        self.auto_schedule = self._create_adaptive_schedule()
        
    def _create_adaptive_schedule(self):
        """Create an adaptive acquisition schedule based on optimization phases."""
        return [
            # Phase 1: Global exploration (0-20%)
            (0.0, 0.2, "pes", 1.0),  # Pure exploration with high weight
            # Phase 2: Balanced exploration-exploitation (20-60%)
            (0.2, 0.4, "ei", 0.8),   # Standard EI with exploration bias
            (0.4, 0.6, "kg", 0.6),   # Knowledge gradient for learning
            # Phase 3: Exploitation with refinement (60-85%)
            (0.6, 0.75, "log_ei", 0.4),  # Log EI for better numerical stability
            (0.75, 0.85, "ucb", 0.2),    # UCB with decreasing exploration
            # Phase 4: Final exploitation (85-100%)
            (0.85, 1.0, "pi", 0.1),      # Probability improvement for final refinement
        ]
        
    def set_iteration(self, iteration: int):
        self.iteration = iteration
        
    def _get_progress_metrics(self):
        """Get various progress metrics for smarter scheduling."""
        progress = self.iteration / (self.config.max_evals + 1e-9)
        
        return {
            'raw_progress': progress,
            'exploration_factor': max(0, 1 - 2 * progress),  # Decreases from 1 to 0
            'exploitation_factor': min(1, 2 * progress),     # Increases from 0 to 1
        }
    
    def _select_auto_acquisition(self, metrics: Dict[str, float]) -> tuple:
        """Intelligently select acquisition function based on progress metrics."""
        progress = metrics['raw_progress']
        
        # Find appropriate phase
        for start, end, acq_type, exploration_weight in self.auto_schedule:
            if start <= progress < end:
                return acq_type, exploration_weight
                
        # Default to final phase
        return "pi", 0.1
    
    def _blend_acquisitions(self, mean, std, best_value, primary_acq: str, 
                           secondary_acq: str, blend_factor: float):
        """Blend two acquisition functions for smoother transitions."""
        primary_scores = self.acquisition_functions[primary_acq](mean, std, best_value)
        secondary_scores = self.acquisition_functions[secondary_acq](mean, std, best_value)
        
        # Normalize scores to [0, 1] for fair blending
        primary_scores = (primary_scores - np.min(primary_scores)) / (np.ptp(primary_scores) + 1e-9)
        secondary_scores = (secondary_scores - np.min(secondary_scores)) / (np.ptp(secondary_scores) + 1e-9)
        
        return blend_factor * primary_scores + (1 - blend_factor) * secondary_scores
    
    @lru_cache(maxsize=32)
    def _cached_acquisition_lookup(self, acq_type: str) -> Callable:
        """Cache acquisition function lookups for efficiency."""
        return self.acquisition_functions.get(acq_type, expected_improvement)

    def compute_acquisition_scores(self, mean, std, best_value):
        """Compute acquisition scores with intelligent function selection."""
        metrics = self._get_progress_metrics()
        
        if self.acquisition_type == "auto":
            # Intelligent auto mode
            primary_acq, exploration_weight = self._select_auto_acquisition(metrics)
            
            # Optional: Blend with secondary acquisition for smoother transitions
            if metrics['raw_progress'] > 0.1:  # After initial exploration
                # Determine secondary acquisition based on context
                if metrics['exploration_factor'] > 0.5:
                    secondary_acq = "pes"  # Keep exploring
                else:
                    secondary_acq = "ei"   # Standard exploitation
                    
                # Blend primary and secondary
                scores = self._blend_acquisitions(mean, std, best_value, 
                                                primary_acq, secondary_acq, 0.7)
            else:
                # Pure primary acquisition
                acq_func = self._cached_acquisition_lookup(primary_acq)
                scores = acq_func(mean, std, best_value)
                
            # Add exploration bonus based on progress
            exploration_bonus = exploration_weight * std * 0.1
            scores += exploration_bonus
            
        elif self.acquisition_type == "adaptive_ei":
            # Adaptive EI that adjusts based on convergence
            base_scores = expected_improvement(mean, std, best_value)
            
            # Add adaptive exploration term
            exploration_term = metrics['exploration_factor'] * std
            entropy_term = -metrics['exploitation_factor'] * np.log(np.clip(base_scores, 1e-10, None))
            
            scores = base_scores + 0.1 * exploration_term + 0.05 * entropy_term
            
        elif self.acquisition_type == "portfolio":
            # Portfolio approach: combine multiple acquisitions
            ei_scores = expected_improvement(mean, std, best_value)
            ucb_scores = upper_confidence_bound(mean, std, self.config.ucb_beta)
            kg_scores = knowledge_gradient(mean, std, best_value)
            
            # Normalize each acquisition
            ei_norm = ei_scores / (np.max(ei_scores) + 1e-9)
            ucb_norm = ucb_scores / (np.max(ucb_scores) + 1e-9)
            kg_norm = kg_scores / (np.max(kg_scores) + 1e-9)
            
            # Dynamic weights based on progress
            w_ei = 0.5 + 0.3 * metrics['exploitation_factor']
            w_ucb = 0.3 * metrics['exploration_factor']
            w_kg = 0.2
            
            scores = w_ei * ei_norm + w_ucb * ucb_norm + w_kg * kg_norm
            
        else:
            # Standard single acquisition function - fallback to original logic
            acq = self.acquisition_type
            progress = self.iteration / (self.config.max_evals + 1e-9)

            if acq == "auto":
                if progress < 0.3:
                    scores = predictive_entropy_search(mean, std, best_value)
                elif progress < 0.7:
                    scores = expected_improvement(mean, std, best_value)
                else:
                    scores = log_expected_improvement(mean, std, best_value)
            elif acq == "ei":
                scores = expected_improvement(mean, std, best_value)
            elif acq == "log_ei":
                scores = log_expected_improvement(mean, std, best_value)
            elif acq == "ucb":
                scores = upper_confidence_bound(mean, std, self.config.ucb_beta)
            elif acq == "pi":
                scores = probability_improvement(mean, std, best_value)
            elif acq == "nei":
                scores = noisy_expected_improvement(mean, std, best_value, self.config.obs_noise)
            elif acq == "pes":
                scores = predictive_entropy_search(mean, std, best_value)
            elif acq == "kg":
                scores = knowledge_gradient(mean, std, best_value)
            elif acq == "ts":
                scores = np.zeros_like(mean)  # not used directly
            else:
                scores = expected_improvement(mean, std, best_value)
            
        # Ensure numerical stability
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return scores

    def optimize_acquisition_in_region(self, region, bounds, existing_candidates, rng, surrogate_manager):
        """Optimize acquisition function in a specific region."""
        candidates = self._generate_region_candidates(region, bounds, rng)

        if region.local_y.shape[0] >= self.config.min_local_samples:
            mean, std = surrogate_manager.predict_local(candidates, region.local_X, region.local_y, region.radius)
        else:
            mean, std = surrogate_manager.predict_global_cached(candidates)

        scores = self.compute_acquisition_scores(mean, std, region.best_value)
        scores += std * region.exploration_bonus * 0.1

        if existing_candidates:
            # Handle the case where existing_candidates might have different shapes
            try:
                existing = np.stack(existing_candidates)
                selected_idx = dpp_select(candidates, scores, batch_size=self.config.batch_size)
            except ValueError:
                # Fallback if stacking fails due to shape mismatch
                selected_idx = np.argsort(-scores)[:self.config.batch_size]
        else:
            selected_idx = np.argsort(-scores)[:self.config.batch_size]

        refined = [self._refine_candidate(c, region, bounds, surrogate_manager) for c in candidates[selected_idx]]
        
        # Return only the best candidate instead of the entire batch
        refined_array = np.array(refined)
        best_refined_idx = np.argmax(scores[selected_idx])
        return refined_array[best_refined_idx]

    def _generate_region_candidates(self, region, bounds, rng):
        n = self.config.n_candidates
        entropy = region.entropy if hasattr(region, 'entropy') else 1.0
        dim = region.center.shape[0]
        samples = []

        for i in range(n):
            p = i / n
            if p < 0.4 + 0.2 * entropy:
                s = self._sample_from_covariance(region, bounds, rng)
            elif p < 0.8:
                s = self._sample_from_gradient(region, bounds, rng)
            else:
                s = self._sample_local_random(region, bounds, rng)
            samples.append(s)
        return np.stack(samples)

    def _sample_from_covariance(self, region, bounds, rng):
        eigvals, eigvecs = np.linalg.eigh(region.cov)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 1e-8, None)))
        z = rng.normal(size=region.n_dims)
        z /= np.linalg.norm(z) + 1e-9
        r = region.radius * rng.uniform() ** (1 / region.n_dims)
        sample = region.center + r * (sqrt_cov @ z)
        return np.clip(sample, bounds[:, 0], bounds[:, 1])

    def _sample_from_gradient(self, region, bounds, rng):
        dim = region.center.shape[0]
        eigvals, eigvecs = np.linalg.eigh(region.cov)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 1e-8, None)))

        if region.local_y.shape[0] >= 3:
            best_idx = np.argmin(region.local_y)
            best_point = region.local_X[best_idx]

            if region.local_y.shape[0] >= 5:
                sorted_idx = np.argsort(region.local_y)
                good = region.local_X[sorted_idx[:3]]
                bad = region.local_X[sorted_idx[-2:]]
                grad_dir = np.mean(good, axis=0) - np.mean(bad, axis=0)
            else:
                grad_dir = best_point - region.center

            grad_dir /= np.linalg.norm(grad_dir) + 1e-9
            noise = rng.normal(size=dim)
            noise /= np.linalg.norm(noise) + 1e-9
            direction = 0.7 * grad_dir + 0.3 * noise
            direction /= np.linalg.norm(direction) + 1e-9
            step = (0.4 + 0.3 * region.exploration_bonus) * region.radius
            sample = best_point + step * (sqrt_cov @ direction)
        else:
            direction = rng.normal(size=dim)
            direction /= np.linalg.norm(direction) + 1e-9
            step = 0.8 * region.radius
            sample = region.center + step * (sqrt_cov @ direction)

        return np.clip(sample, bounds[:, 0], bounds[:, 1])

    def _sample_local_random(self, region, bounds, rng):
        direction = rng.normal(size=region.center.shape)
        direction /= np.linalg.norm(direction)
        radius = region.radius * rng.uniform() ** (1 / region.n_dims)
        sample = region.center + direction * radius
        return np.clip(sample, bounds[:, 0], bounds[:, 1])

    def _refine_candidate(self, x, region, bounds, surrogate_manager, steps=5, alpha=0.4):
        """Gradient-free CMA-style refinement toward predicted minimum."""
        dim = len(x)
        x_best = x.copy()
        best_val = surrogate_manager.predict_global_cached(x[None, :])[0][0]
        for _ in range(steps):
            perturb = np.random.randn(dim)
            perturb /= np.linalg.norm(perturb) + 1e-9
            candidate = x + alpha * region.radius * perturb
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            y = surrogate_manager.predict_global_cached(candidate[None, :])[0][0]
            if y < best_val:
                x_best, best_val = candidate, y
        return x_best

    def thompson_sampling_batch(self, candidates, rng, surrogate_manager, batch_size=5, n_draws=8):
        mean, std = surrogate_manager.predict_global_cached(candidates)
        all_samples = np.stack([rng.normal(mean, std) for _ in range(n_draws)], axis=0)
        avg_samples = np.mean(all_samples, axis=0)
        indices = dpp_select(candidates, avg_samples, batch_size=batch_size)
        return candidates[indices]

    def batch_expected_improvement(self, candidates, surrogate_manager, best_value):
        """Greedy q-EI: select batch maximizing total marginal EI."""
        mean, std = surrogate_manager.predict_global_cached(candidates)
        scores = expected_improvement(mean, std, best_value)
        return candidates[np.argsort(-scores)[:self.config.batch_size]]
    
    def get_acquisition_info(self) -> Dict[str, Any]:
        """Get information about current acquisition strategy for debugging."""
        metrics = self._get_progress_metrics()
        
        info = {
            'iteration': self.iteration,
            'progress': metrics['raw_progress'],
            'exploration_factor': metrics['exploration_factor'],
            'exploitation_factor': metrics['exploitation_factor'],
            'acquisition_type': self.acquisition_type,
        }
        
        if self.acquisition_type == "auto":
            primary_acq, exploration_weight = self._select_auto_acquisition(metrics)
            info.update({
                'selected_acquisition': primary_acq,
                'exploration_weight': exploration_weight,
            })
            
        return info
import numpy as np
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class CandidateGenerator:
    """Manages candidate generation strategies with optimized batch processing"""
    
    def __init__(self, config: 'TurboConfig', acquisition_manager: 'AcquisitionManager'):
        self.config = config
        self.acquisition_manager = acquisition_manager
        self.stagnation_counter = 0
        self.iteration = 0
        self._cached_exploration_prob = None
        self._cache_iteration = -1
        
    def set_context(self, iteration: int, stagnation_counter: int):
        """Update context for adaptive sampling"""
        self.iteration = iteration
        self.stagnation_counter = stagnation_counter
        self.acquisition_manager.set_iteration(iteration)
        # Invalidate cache when context changes
        self._cache_iteration = -1
        
    def generate_candidates(self, bounds, rng, active_regions, surrogate_manager):
        """Main candidate generation entry point with optimized batch processing"""
        if not active_regions:
            return rng.uniform(bounds[:, 0], bounds[:, 1],
                            size=(self.config.batch_size, bounds.shape[0]))
        
        if self.config.acquisition == "ts":
            return self._generate_thompson_candidates_batch(bounds, rng, active_regions)
        else:
            return self._generate_adaptive_candidates_batch(bounds, rng, active_regions, surrogate_manager)
    
    def _generate_adaptive_candidates_batch(self, bounds, rng, active_regions, surrogate_manager):
        """Generate candidates using vectorized adaptive exploration/exploitation"""
        exploration_prob = self._compute_exploration_probability()
        
        # Vectorized decision making
        batch_decisions = rng.uniform(size=self.config.batch_size) < exploration_prob
        n_exploration = np.sum(batch_decisions)
        n_exploitation = self.config.batch_size - n_exploration
        
        candidates = []
        
        # Batch generate exploration candidates
        if n_exploration > 0:
            exploration_candidates = self._exploration_sampling_batch(
                bounds, active_regions, rng, n_exploration, surrogate_manager
            )
            candidates.extend(exploration_candidates)
        
        # Batch generate exploitation candidates  
        if n_exploitation > 0:
            exploitation_candidates = self._exploitation_sampling_batch(
                bounds, active_regions, rng, n_exploitation, candidates, surrogate_manager
            )
            candidates.extend(exploitation_candidates)
        
        # Shuffle to mix exploration/exploitation
        candidates = np.array(candidates)
        rng.shuffle(candidates)
        return candidates
    
    def _generate_thompson_candidates_batch(self, bounds, rng, active_regions):
        """Generate Thompson Sampling candidates with vectorized operations"""
        all_candidates = []
        
        # Vectorized per-region sampling
        per_region = max(1, self.config.n_candidates // len(active_regions))
        
        # Process regions in parallel for large batches
        if len(active_regions) > 4 and per_region > 10:
            with ThreadPoolExecutor(max_workers=min(4, len(active_regions))) as executor:
                futures = []
                for region in active_regions:
                    future = executor.submit(
                        self._sample_region_batch, region, bounds, rng, per_region
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    all_candidates.extend(future.result())
        else:
            # Sequential processing for smaller batches
            for region in active_regions:
                region_candidates = self._sample_region_batch(region, bounds, rng, per_region)
                all_candidates.extend(region_candidates)
        
        # Vectorized global exploration
        global_count = self.config.n_candidates // 4
        if global_count > 0:
            global_candidates = rng.uniform(
                bounds[:, 0], bounds[:, 1], size=(global_count, bounds.shape[0])
            )
            all_candidates.extend(global_candidates)
        
        return np.array(all_candidates)
    
    def _sample_region_batch(self, region, bounds, rng, count):
        """Sample multiple candidates from a region efficiently"""
        candidates = []
        for _ in range(count):
            candidate = self.acquisition_manager._sample_from_covariance(region, bounds, rng)
            candidates.append(candidate)
        return candidates
        
    def _compute_exploration_probability(self):
        """Compute adaptive exploration probability with caching"""
        # Cache exploration probability for the current iteration
        if self._cache_iteration == self.iteration and self._cached_exploration_prob is not None:
            return self._cached_exploration_prob
        
        base_exploration = self.config.exploration_factor
        
        # Stronger stagnation recovery
        stagnation_boost = min(0.6, 0.05 * self.stagnation_counter)
        
        # Keep minimum baseline
        progress_factor = max(0.1, 1 - (self.iteration / self.config.max_evals))
        
        # Health-based exploration boost
        health_penalty = 0.0
        if hasattr(self, "regions") and len(self.regions) > 0:
            avg_health = np.mean([r.health_score for r in self.regions])
            health_penalty = max(0.0, 0.3 - avg_health)
        
        exploration_prob = base_exploration + stagnation_boost + 0.2 * progress_factor + health_penalty
        
        # Cache result
        self._cached_exploration_prob = min(0.95, exploration_prob)
        self._cache_iteration = self.iteration
        
        return self._cached_exploration_prob

    def _exploration_sampling_batch(self, bounds, regions, rng, batch_size, surrogate_manager):
        """Batch exploration sampling with vectorized operations"""
        candidates = []
        
        # Pre-compute strategy assignments
        strategies = rng.uniform(size=batch_size)
        
        # Count each strategy type
        uniform_count = np.sum(strategies < 0.3)
        uncertainty_count = np.sum((strategies >= 0.3) & (strategies < 0.6))
        region_count = batch_size - uniform_count - uncertainty_count
        
        # Batch uniform sampling
        if uniform_count > 0:
            uniform_candidates = rng.uniform(
                bounds[:, 0], bounds[:, 1], size=(uniform_count, bounds.shape[0])
            )
            candidates.extend(uniform_candidates)
        
        # Batch uncertainty sampling
        if uncertainty_count > 0:
            uncertainty_candidates = self._uncertainty_sampling_batch(
                bounds, rng, uncertainty_count, surrogate_manager
            )
            candidates.extend(uncertainty_candidates)
        
        # Batch region-based exploration
        if region_count > 0:
            region_candidates = self._region_exploration_batch(
                bounds, regions, rng, region_count
            )
            candidates.extend(region_candidates)
        
        return candidates
    
    def _uncertainty_sampling_batch(self, bounds, rng, count, surrogate_manager):
        """Batch uncertainty-based sampling"""
        # Generate larger candidate pool for better selection
        pool_size = min(200, count * 10)
        candidate_pool = rng.uniform(
            bounds[:, 0], bounds[:, 1], size=(pool_size, bounds.shape[0])
        )
        
        # Single prediction call for entire pool
        _, std = surrogate_manager.predict_global_cached(candidate_pool)
        
        # Select top uncertainty candidates
        top_indices = np.argpartition(std, -count)[-count:]
        return candidate_pool[top_indices]
    
    def _region_exploration_batch(self, bounds, regions, rng, count):
        """Batch region-based exploration sampling"""
        candidates = []
        
        # Pre-compute region weights once
        weights = np.array([r.spawn_score * r.exploration_bonus for r in regions], dtype=np.float64)
        total = np.sum(weights, dtype=np.float64)
        
        if total <= 1e-8 or np.any(np.isnan(weights)):
            # Uniform region selection
            region_indices = rng.integers(0, len(regions), size=count)
        else:
            # Weighted region selection
            weights = np.clip(weights / total, 0, 1)
            weights /= np.sum(weights)
            region_indices = rng.choice(len(regions), size=count, p=weights)
        
        # Vectorized direction generation
        directions = rng.normal(size=(count, bounds.shape[0]))
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        # Generate candidates for each selected region
        for i, region_idx in enumerate(region_indices):
            region = regions[region_idx]
            exploration_radius = region.radius * (2.0 + region.exploration_bonus)
            
            candidate = region.center + directions[i] * exploration_radius
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            candidates.append(candidate)
        
        return candidates
    
    def _exploitation_sampling_batch(self, bounds, regions, rng, count, existing_candidates, surrogate_manager):
        """Batch exploitation sampling with optimized region selection"""
        candidates = []
        
        # Pre-compute region weights
        health_scores = np.array([r.health_score for r in regions], dtype=np.float64)
        total = np.sum(health_scores)
        
        if total <= 1e-8 or np.any(np.isnan(health_scores)) or np.all(health_scores == 0):
            region_indices = rng.integers(0, len(regions), size=count)
        else:
            region_weights = health_scores / (total + 1e-9)
            region_indices = rng.choice(len(regions), size=count, p=region_weights)
        
        # Group by region to minimize acquisition optimization calls
        region_counts = {}
        for idx in region_indices:
            region_counts[idx] = region_counts.get(idx, 0) + 1
        
        # Process each region's candidates
        for region_idx, region_count in region_counts.items():
            region = regions[region_idx]
            
            # Fallback for empty regions
            if getattr(region, "local_X", None) is None or len(region.local_X) == 0:
                fallback_candidates = rng.uniform(
                    bounds[:, 0], bounds[:, 1], size=(region_count, bounds.shape[0])
                )
                candidates.extend(fallback_candidates)
                continue
            
            # Optimize multiple candidates in this region
            for _ in range(region_count):
                candidate = self.acquisition_manager.optimize_acquisition_in_region(
                    region, bounds, existing_candidates + candidates, rng, surrogate_manager
                )
                candidates.append(candidate)
        
        return candidates

class RegionManager:
    """Manages trust regions lifecycle, selection, and optimization"""
    
    def __init__(self, config: TurboConfig):
        self.config = config
        self.regions: List[TrustRegion] = []
        self.surrogate_manager = None  # Will be set by Foretuner
        self._kl_cache = {}  # Cache for KL computations
        
    def set_surrogate_manager(self, surrogate_manager: SurrogateManager):
        """Set reference to surrogate manager"""
        self.surrogate_manager = surrogate_manager
        
    def initialize_regions(self, X, y, n_dims, rng: np.random.Generator):
        """Initialize regions from initial data"""
        best_idx = int(np.argmin(y))
        centers = [X[best_idx]]
        good_indices = np.argsort(y)[:len(y)//3]
        n_regions_to_init = min(self.config.n_regions, len(X))

        for i in range(1, n_regions_to_init):
            if len(centers) == 1:
                candidates = X[good_indices]
                distances = np.linalg.norm(candidates - centers[0], axis=1)
                probabilities = distances / np.sum(distances)
                selected_idx = rng.choice(len(candidates), p=probabilities)
                centers.append(candidates[selected_idx])
            else:
                center_array = np.stack(centers)
                distances = np.min(np.linalg.norm(X[:, None, :] - center_array[None, :, :], axis=2), axis=1)
                probabilities = distances ** 2
                probabilities = probabilities / np.sum(probabilities)
                selected_idx = rng.choice(len(X), p=probabilities)
                centers.append(X[selected_idx])

        for i, center in enumerate(centers):
            local_distances = np.linalg.norm(X - center, axis=1)
            local_radius = np.sort(local_distances)[max(1, len(local_distances) // 5)]
            initial_radius = np.clip(local_radius, self.config.min_radius, self.config.init_radius)
            region = TrustRegion(center, initial_radius, i, n_dims)
            self.regions.append(region)

    def update_regions_with_new_data(self, X_new, y_new):
        """Update regions with new evaluation data"""
        active_regions = [r for r in self.regions if r.is_active]
        if not active_regions:
            return

        region_centers = np.stack([r.center for r in active_regions])
        region_covs = np.stack([r.cov for r in active_regions])
        region_cov_invs = np.linalg.inv(region_covs)

        def mahalanobis_sq(x, c, cov_inv):
            diff = x - c
            return diff @ cov_inv @ diff

        for i, (x, y) in enumerate(zip(X_new, y_new)):
            distances = np.array([
                mahalanobis_sq(x, center, cov_inv)
                for center, cov_inv in zip(region_centers, region_cov_invs)
            ])

            closest_idx = np.argmin(distances)
            closest_region = active_regions[closest_idx]
            threshold = chi2.ppf(0.95, df=closest_region.n_dims)

            if distances[closest_idx] <= threshold:
                closest_region.update(x, float(y), self.config)
                
        # Clear KL cache when regions are updated
        self._kl_cache.clear()

    def manage_regions(self, bounds, n_dims, rng: np.random.Generator, global_X, global_y):
        """Main region management pipeline"""
        # Handle restarts and kills
        self._handle_region_lifecycle(bounds, rng)
        
        # NSGA-II based selection
        self._select_regions_nsga(global_X)
        
        # Merge operations
        self._merge_operations(bounds, global_X)
        
        # Spawn new regions
        self._spawn_new_regions(bounds, n_dims, rng, global_X, global_y)
        
        # Ensure diversity
        self._ensure_diversity(bounds, n_dims, rng, global_X, global_y)

    def _handle_region_lifecycle(self, bounds, rng):
        """Handle region expansions, restarts, replacements, and removals"""

        # ✅ 1. Mild expansion for stagnating regions
        for region in self.regions[:]:
            if region.stagnation_count > 5 and region.radius < self.config.max_radius:
                old_radius = region.radius
                region.radius = min(region.radius * 1.3, self.config.max_radius)
                region.exploration_bonus += 0.2
                print(f"[EXPAND] Region {region.region_id} expanded radius "
                    f"{old_radius:.3f} → {region.radius:.3f}")

            if region.should_restart:
                print(f"[RESTART] Stagnant region {region.region_id}")
                self._restart_region(region, bounds, rng)

        # ✅ 2. Compute avg_health
        avg_health = np.mean([r.health_score for r in self.regions]) if self.regions else 1.0
        stagnating = hasattr(self, "foretuner") and self.foretuner.stagnation_counter > 10

        # ✅ 3. If all regions are weak → replace the worst one
        if (avg_health < 0.25 or stagnating) and len(self.regions) >= self.config.n_regions:
            worst_region = min(self.regions, key=lambda r: r.health_score)
            print(f"[REPLACE] Removing weak region {worst_region.region_id} "
                f"(health={worst_region.health_score:.3f}, avg={avg_health:.3f})")
            self.regions.remove(worst_region)
            # Replace with a fresh diverse region
            self._add_diverse_region(bounds, worst_region.n_dims, rng,
                                    self.surrogate_manager.global_X,
                                    self.surrogate_manager.global_y)

        # ✅ 4. Regular cleanup of dead regions
        old_count = len(self.regions)
        self.regions = [r for r in self.regions if not r.should_kill]
        if len(self.regions) < old_count:
            print(f"[CLEANUP] Removed {old_count - len(self.regions)} hopeless regions")

    def _select_regions_nsga(self, global_X):
        """Select regions using NSGA-II"""
        if len(self.regions) > self.config.n_regions // 2:
            self.regions = self.nsga_region_selection(
                self.regions, global_X, 
                retain_k=self.config.n_regions // 2,
                diversity_metric=self.config.diversity_metric,
                method=self.config.selection_method
            )

    def _merge_operations(self, bounds, global_X):
        """Perform region merging operations with improved KL-based merging"""
        if len(self.regions) < 2:
            return

        # First do traditional distance-based merging for very close regions
        self._merge_close_regions()
        
        # Then do KL-based merging for functionally similar regions
        if len(self.regions) >= 3:
            kl_matrix = self.compute_pairwise_kl_improved(self.regions, bounds)
            
            # Adaptive threshold based on KL distribution
            all_kl = kl_matrix[np.triu_indices(len(self.regions), k=1)]
            if len(all_kl) > 0:
                # Use a more conservative threshold to avoid over-merging
                kl_threshold = np.percentile(all_kl, 5)  # Only merge very similar regions
                self.regions = self.merge_close_regions_kl_improved(
                    self.regions, kl_matrix, kl_threshold
                )

    def _spawn_new_regions(self, bounds, n_dims, rng, global_X, global_y):
        """Spawn new regions adaptively based on coverage, stagnation, and health"""
        if len(self.regions) >= self.config.n_regions:
            return

        centers = np.stack([r.center for r in self.regions])
        radii = np.array([r.radius for r in self.regions], dtype=np.float64)
        coverage = compute_coverage(global_X, centers, radii)

        avg_health = np.mean([r.health_score for r in self.regions])
        stagnating = hasattr(self, "foretuner") and self.foretuner.stagnation_counter > 5

        low_coverage = coverage < self.config.spawn_threshold
        unhealthy = avg_health < 0.25

        # ✅ Trigger spawn if any condition fires
        if low_coverage or unhealthy or stagnating:

            # ✅ Force global exploration if stagnation very high
            if hasattr(self, "foretuner") and self.foretuner.stagnation_counter > 15:
                print("[FORCE-SPAWN] Long stagnation → sampling purely unexplored region")
                new_center = rng.uniform(bounds[:, 0], bounds[:, 1])
            else:
                new_center = self._find_spawn_location(bounds, n_dims, rng, global_X, global_y)

            spawn_radius = self._compute_spawn_radius(new_center, centers)

            new_region = TrustRegion(new_center, spawn_radius, len(self.regions), n_dims)

            # ✅ Higher exploration bonus if stagnating hard
            new_region.exploration_bonus = 1.8 if stagnating else 1.5
            self.regions.append(new_region)

            print(f"[SPAWN] New region #{len(self.regions)}, "
                f"coverage={coverage:.3f}, health={avg_health:.2f}, "
                f"stagnation={stagnating}")

    def _ensure_diversity(self, bounds, n_dims, rng, global_X, global_y):
        """Ensure minimum number of diverse regions"""
        target_regions = min(self.config.n_regions, max(2, len(global_y) // 20))
        while len(self.regions) < target_regions:
            self._add_diverse_region(bounds, n_dims, rng, global_X, global_y)

    def _restart_region(self, region, bounds, rng):
        # Use _find_spawn_location for smarter restart
        new_center = self._find_spawn_location(bounds, region.n_dims, rng,
                                            self.surrogate_manager.global_X,
                                            self.surrogate_manager.global_y)
        old_center = region.center.copy()
        region.center = new_center
        region.radius = self.config.init_radius
        region.cov = np.eye(region.n_dims) * (region.radius ** 2)
        region.consecutive_failures = 0
        region.stagnation_count = 0
        region.exploration_bonus = 1.6
        print(f"[RESTART] Region {region.region_id} moved from {old_center} → {new_center}")

    def _find_spawn_location(self, bounds, n_dims, rng, global_X, global_y):
        """Find optimal location for new region using Pareto selection"""
        n_candidates = 200
        try:
            sobol = Sobol(d=n_dims, scramble=True, seed=rng.integers(1e6))
            candidates = bounds[:, 0] + sobol.random(n_candidates) * (bounds[:, 1] - bounds[:, 0])
        except ImportError:
            candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, n_dims))

        # Compute objectives: uncertainty + distance
        mean, std = self._predict_at_candidates(candidates)
        centers = np.stack([r.center for r in self.regions])
        min_distances = np.min(np.linalg.norm(
            candidates[:, None, :] - centers[None, :, :], axis=2
        ), axis=1)

        # Pareto selection
        spawn_objs = np.column_stack((-std, -min_distances))
        fronts, _ = self.fast_non_dominated_sort(spawn_objs)
        best_front = fronts[0]
        
        if len(best_front) > 1:
            best_idx = np.argmax(std[best_front])
            spawn_idx = best_front[best_idx]
        else:
            spawn_idx = best_front[0]

        return candidates[spawn_idx]

    def _compute_spawn_radius(self, new_center, existing_centers):
        """Compute appropriate radius for new region"""
        min_distances = np.min(np.linalg.norm(
            existing_centers - new_center, axis=1
        ))
        global_scale = max(0.2, 1.0 - len(self.regions) / self.config.n_regions)
        return min(min_distances * 0.3 * global_scale, self.config.init_radius)

    def _predict_at_candidates(self, candidates):
        """Predict uncertainty at candidate locations using SurrogateManager"""
        return self.surrogate_manager.predict_global_cached(candidates)

    # ========== IMPROVED KL COMPUTATION METHODS ==========
    
    def compute_pairwise_kl_improved(self, regions, bounds, n_samples=100):
        """
        Improved KL computation with better sampling strategy and numerical stability
        """
        N = len(regions)
        KL_matrix = np.zeros((N, N))
        
        # Generate consistent evaluation points for all regions
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Use stratified sampling: part global, part region-local
        n_global = n_samples // 3
        n_local = n_samples - n_global
        
        # Global samples from domain
        if n_global > 0:
            try:
                sobol = Sobol(d=bounds.shape[0], scramble=True, seed=42)
                global_samples = bounds[:, 0] + sobol.random(n_global) * (bounds[:, 1] - bounds[:, 0])
            except ImportError:
                global_samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_global, bounds.shape[0]))
        else:
            global_samples = np.empty((0, bounds.shape[0]))
        
        # Get GP predictions for each region
        region_predictions = []
        for r in regions:
            # Region-local samples using truncated normal distribution
            if n_local > 0:
                local_samples = self._sample_from_region_truncated(r, bounds, n_local, rng)
            else:
                local_samples = np.empty((0, bounds.shape[0]))
            
            # Combine global and local samples
            eval_points = np.vstack([global_samples, local_samples]) if len(global_samples) > 0 else local_samples
            
            # Get GP predictions
            if len(r.local_X) > 0:
                mu, std = self.surrogate_manager.predict_local(
                    eval_points, r.local_X, r.local_y, r.radius
                )
            else:
                # Fallback to global prediction if no local data
                mu, std = self.surrogate_manager.predict_global_cached(eval_points)
            
            # Ensure numerical stability
            std = np.maximum(std, 1e-6)
            
            region_predictions.append({
                'mean': mu,
                'std': std,
                'points': eval_points
            })
        
        # Compute pairwise KL divergences
        for i in range(N):
            for j in range(i + 1, N):
                kl_sym = self._compute_kl_between_predictions(
                    region_predictions[i], region_predictions[j]
                )
                KL_matrix[i, j] = kl_sym
                KL_matrix[j, i] = kl_sym
        
        return KL_matrix
        
    def _sample_from_region_truncated(self, region, bounds, n_samples, rng):
        n_dims = region.n_dims
        samples = []
        max_attempts = 5 * n_samples
        attempts = 0

        # ✅ Ensure center and radius are real
        center = np.real_if_close(region.center, tol=1e5)
        radius = float(np.abs(np.real_if_close(region.radius)))

        while len(samples) < n_samples and attempts < max_attempts:
            candidate = rng.normal(center, radius * 0.5, size=n_dims)

            # Clip small imaginary noise if it still exists
            if np.iscomplexobj(candidate):
                candidate = np.real(candidate)

            # ✅ Ensure within bounds
            if np.all(candidate >= bounds[:, 0]) and np.all(candidate <= bounds[:, 1]):
                samples.append(candidate)

            attempts += 1

        return np.array(samples)
    
    def _compute_kl_between_predictions(self, pred1, pred2):
        """
        Compute symmetric KL divergence between two GP predictions (same eval points required)
        """
        mu1, std1 = pred1['mean'], pred1['std']
        mu2, std2 = pred2['mean'], pred2['std']
        
        # ✅ Align sample sizes (truncate to min)
        n = min(len(mu1), len(mu2))
        mu1, std1 = mu1[:n], std1[:n]
        mu2, std2 = mu2[:n], std2[:n]

        # ✅ Clip stds to avoid log(0) or division by zero
        std1 = np.clip(std1, 1e-8, None)
        std2 = np.clip(std2, 1e-8, None)

        # Monte Carlo KL estimation
        log_p1 = -0.5 * np.log(2 * np.pi * std1**2)  # log p(x|p)
        log_p2_under_p1 = -0.5 * np.log(2 * np.pi * std2**2) - 0.5 * ((mu1 - mu2)**2 / std2**2)

        log_p2 = -0.5 * np.log(2 * np.pi * std2**2)  # log p(x|q)
        log_p1_under_p2 = -0.5 * np.log(2 * np.pi * std1**2) - 0.5 * ((mu2 - mu1)**2 / std1**2)

        # KL(p||q) and KL(q||p)
        kl_12 = np.mean(log_p1 - log_p2_under_p1)
        kl_21 = np.mean(log_p2 - log_p1_under_p2)

        # ✅ Clamp to avoid weird negatives due to numerical noise
        kl_12 = np.clip(kl_12, 0, 100)
        kl_21 = np.clip(kl_21, 0, 100)

        # Symmetric KL
        return 0.5 * (kl_12 + kl_21)

    def merge_close_regions_kl_improved(self, regions, kl_matrix, kl_threshold=0.5):
        """
        Improved KL-based region merging with better merge criteria
        """
        N = len(regions)
        merged = set()
        new_regions = []
        
        # Sort region pairs by KL divergence (most similar first)
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j, kl_matrix[i, j]))
        
        pairs.sort(key=lambda x: x[2])  # Sort by KL divergence
        
        for i, j, kl_div in pairs:
            if i in merged or j in merged:
                continue
            
            r1, r2 = regions[i], regions[j]
            
            # Additional merge criteria beyond just KL divergence
            performance_similar = abs(r1.best_value - r2.best_value) < 0.1 * (abs(r1.best_value) + 1e-6)
            radii_compatible = max(r1.radius, r2.radius) < 2 * min(r1.radius, r2.radius)
            both_mature = r1.trial_count > 3 and r2.trial_count > 3
            
            if (kl_div < kl_threshold and performance_similar and 
                radii_compatible and both_mature):
                
                # Merge regions with weighted combination
                total_weight = r1.trial_count + r2.trial_count
                w1, w2 = r1.trial_count / total_weight, r2.trial_count / total_weight
                
                new_center = w1 * r1.center + w2 * r2.center
                new_radius = max(r1.radius, r2.radius) * 1.1  # Slightly expand
                
                # Weighted covariance combination
                new_cov = w1 * r1.cov + w2 * r2.cov
                
                # Create merged region
                new_region = TrustRegion(new_center, new_radius, 
                                       max(r1.region_id, r2.region_id), r1.n_dims)
                new_region.best_value = min(r1.best_value, r2.best_value)
                new_region.trial_count = r1.trial_count + r2.trial_count
                new_region.success_count = r1.success_count + r2.success_count
                new_region.cov = new_cov
                
                # Combine local data
                if len(r1.local_X) > 0 and len(r2.local_X) > 0:
                    new_region.local_X = np.vstack([r1.local_X, r2.local_X])
                    new_region.local_y = np.hstack([r1.local_y, r2.local_y])
                elif len(r1.local_X) > 0:
                    new_region.local_X = r1.local_X.copy()
                    new_region.local_y = r1.local_y.copy()
                elif len(r2.local_X) > 0:
                    new_region.local_X = r2.local_X.copy()
                    new_region.local_y = r2.local_y.copy()
                
                new_regions.append(new_region)
                merged.add(i)
                merged.add(j)
                
                print(f"[MERGE-KL] Regions {r1.region_id} + {r2.region_id} → {new_region.region_id} "
                      f"(KL={kl_div:.4f}, perf_diff={abs(r1.best_value - r2.best_value):.4f})")
        
        # Add unmerged regions
        for i in range(N):
            if i not in merged:
                new_regions.append(regions[i])
        
        return new_regions

    # ========== REGION SELECTION AND UTILITY METHODS ==========
    
    def nsga_region_selection(self, regions, X_star, retain_k=None, diversity_metric="kl", method="nsga2"):
        """NSGA-II based region selection with improved diversity metrics"""
        perf, diversity = self.compute_region_objectives(regions, X_star, diversity_metric)
        
        if method == "nsga":
            fronts, _ = self.fast_non_dominated_sort(np.column_stack((perf, -diversity)))
            pareto_indices = fronts[0]
            print(f"[NSGA] Found {len(pareto_indices)} Pareto-optimal regions.")
            if retain_k and len(pareto_indices) > retain_k:
                pareto_indices = pareto_indices[:retain_k]
            retained = [regions[i] for i in pareto_indices]
            print(f"[NSGA] Retained {len(retained)} Pareto-optimal regions.")
            return retained
        else:
            selected_idx = self.nsga2_select(perf, diversity, retain_k)
            retained = [regions[i] for i in selected_idx]
            print(f"[NSGA-II] Retained {len(retained)} regions via Pareto fronts + crowding distance.")
            return retained

    def compute_region_objectives(self, regions, X_star, diversity_metric="kl", n_mmd_samples=50):
        """
        Compute region objectives with improved diversity computation
        """
        N = len(regions)
        perf = np.array([r.best_value for r in regions])
        
        if diversity_metric == "kl":
            # Use improved KL computation
            bounds = np.column_stack([X_star.min(axis=0), X_star.max(axis=0)])
            kl_matrix = self.compute_pairwise_kl_improved(regions, bounds, n_samples=50)
            diversity = np.sum(kl_matrix, axis=1) / (N - 1 + 1e-9)
        else:
            # Fallback to original computation for other metrics
            mu_list, cov_list, samples_list = [], [], []

            for r in regions:
                mu, std = self.surrogate_manager.predict_local(X_star, r.local_X, r.local_y, r.radius)
                cov = np.diag(std ** 2)
                mu_list.append(mu)
                cov_list.append(cov)

                if diversity_metric == "mmd":
                    samples = np.random.multivariate_normal(mu, cov, size=n_mmd_samples)
                    samples_list.append(samples)

            # Compute pairwise diversity
            pairwise_div = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    if diversity_metric == "wasserstein":
                        d = self._wasserstein_gaussians(mu_list[i], cov_list[i],
                                                       mu_list[j], cov_list[j])
                    elif diversity_metric == "mmd":
                        d = self._mmd_rbf(samples_list[i], samples_list[j])
                    else:
                        raise ValueError(f"Unknown diversity metric: {diversity_metric}")

                    pairwise_div[i, j] = d
                    pairwise_div[j, i] = d

            diversity = np.sum(pairwise_div, axis=1) / (N - 1 + 1e-9)
        
        return perf, diversity

    def fast_non_dominated_sort(self, objectives):
        """Fast non-dominated sorting (Deb et al.)"""
        N = objectives.shape[0]
        S = [[] for _ in range(N)]
        n_dom = np.zeros(N, dtype=int)
        rank = np.zeros(N, dtype=int)
        fronts = [[]]

        for p in range(N):
            for q in range(N):
                if p == q: continue
                if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                    S[p].append(q)
                elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1], rank

    def crowding_distance(self, objectives):
        """Crowding distance for a single Pareto front"""
        N, M = objectives.shape
        distances = np.zeros(N)
        for m in range(M):
            sorted_idx = np.argsort(objectives[:, m])
            distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf
            min_val, max_val = objectives[sorted_idx[0], m], objectives[sorted_idx[-1], m]
            norm = max_val - min_val + 1e-12
            for k in range(1, N - 1):
                prev_val = objectives[sorted_idx[k - 1], m]
                next_val = objectives[sorted_idx[k + 1], m]
                distances[sorted_idx[k]] += (next_val - prev_val) / norm
        return distances

    def nsga2_select(self, perf, diversity, num_select):
        """NSGA-II selection"""
        objs = np.column_stack((perf, -diversity))
        fronts, rank = self.fast_non_dominated_sort(objs)
        selected = []

        for front in fronts:
            if len(selected) + len(front) <= num_select:
                selected.extend(front)
            else:
                front_objs = objs[front]
                cd = self.crowding_distance(front_objs)
                sorted_front = [front[i] for i in np.argsort(-cd)]
                remaining = num_select - len(selected)
                selected.extend(sorted_front[:remaining])
                break

        return np.array(selected)

    def _symmetric_kl_gaussians(self, mu1, cov1, mu2, cov2, eps=1e-8):
        """Closed-form symmetric KL divergence for multivariate Gaussians"""
        d = mu1.shape[0]
        cov1 = cov1 + eps * np.eye(d)
        cov2 = cov2 + eps * np.eye(d)
        inv_cov2 = np.linalg.inv(cov2)
        inv_cov1 = np.linalg.inv(cov1)

        trace_term1 = np.trace(inv_cov2 @ cov1)
        trace_term2 = np.trace(inv_cov1 @ cov2)
        diff = mu2 - mu1
        quad_term1 = diff.T @ inv_cov2 @ diff
        quad_term2 = diff.T @ inv_cov1 @ diff
        logdet_ratio1 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
        logdet_ratio2 = -logdet_ratio1

        kl_12 = 0.5 * (trace_term1 + quad_term1 - d + logdet_ratio1)
        kl_21 = 0.5 * (trace_term2 + quad_term2 - d + logdet_ratio2)
        return 0.5 * (kl_12 + kl_21)

    def _wasserstein_gaussians(self, mu1, cov1, mu2, cov2, eps=1e-8):
        """2-Wasserstein distance between two Gaussians"""
        d = mu1.shape[0]
        cov1 = cov1 + eps * np.eye(d)
        cov2 = cov2 + eps * np.eye(d)
        diff = mu1 - mu2

        # Matrix square root computation with numerical stability
        sqrt_cov1 = sqrtm(cov1)
        inner = sqrt_cov1 @ cov2 @ sqrt_cov1
        sqrt_inner = sqrtm(inner)

        wasserstein_cov = cov1 + cov2 - 2 * sqrt_inner
        wasserstein_cov = np.real(wasserstein_cov)

        term_mean = diff.T @ diff
        term_cov = np.trace(wasserstein_cov)
        return np.sqrt(max(0, term_mean + term_cov))

    def _rbf_kernel_matrix(self, X, Y=None, gamma=None):
        """Compute RBF kernel matrix between X and Y"""
        if Y is None:
            Y = X
        XX = np.sum(X**2, axis=1)[:, None]
        YY = np.sum(Y**2, axis=1)[None, :]
        dists = XX + YY - 2 * X @ Y.T
        if gamma is None:
            gamma = 1.0 / (np.median(dists) + 1e-8)
        return np.exp(-gamma * dists)

    def _mmd_rbf(self, X, Y, gamma=None):
        """Maximum Mean Discrepancy (MMD^2) with RBF kernel"""
        Kxx = self._rbf_kernel_matrix(X, X, gamma)
        Kyy = self._rbf_kernel_matrix(Y, Y, gamma)
        Kxy = self._rbf_kernel_matrix(X, Y, gamma)
        m = X.shape[0]
        n = Y.shape[0]
        return (np.sum(Kxx) / (m*m)
                + np.sum(Kyy) / (n*n)
                - 2 * np.sum(Kxy) / (m*n))

    def _merge_close_regions(self):
        """Merge regions based on Mahalanobis distance"""
        if len(self.regions) < 2:
            return

        merge_threshold = self.config.merge_threshold * 0.5
        regions_to_remove = []
        
        for i in range(len(self.regions)):
            if i in regions_to_remove:
                continue
                
            for j in range(i + 1, len(self.regions)):
                if j in regions_to_remove:
                    continue

                r1, r2 = self.regions[i], self.regions[j]
                avg_cov = (r1.cov + r2.cov) / 2
                
                # Ensure numerical stability
                try:
                    avg_cov_inv = np.linalg.inv(avg_cov + 1e-6 * np.eye(avg_cov.shape[0]))
                except np.linalg.LinAlgError:
                    continue

                delta = r1.center - r2.center
                mahal_dist = np.sqrt(delta.T @ avg_cov_inv @ delta)

                # More restrictive merge criteria
                distance_close = mahal_dist < merge_threshold
                performance_similar = abs(r1.best_value - r2.best_value) < 0.02 * abs(r1.best_value + 1e-6)
                both_small = r1.radius < 0.05 and r2.radius < 0.05
                both_mature = r1.trial_count > 2 and r2.trial_count > 2

                if distance_close and performance_similar and both_small and both_mature:
                    # Weighted merge based on success rates
                    total_weight = r1.success_count + r2.success_count + 1e-6
                    w1 = r1.success_count / total_weight
                    w2 = r2.success_count / total_weight
                    
                    new_center = w1 * r1.center + w2 * r2.center
                    new_radius = max(r1.radius, r2.radius) * 1.1

                    new_cov = w1 * r1.cov + w2 * r2.cov

                    merged = TrustRegion(new_center, new_radius, max(r1.region_id, r2.region_id), r1.n_dims)
                    merged.best_value = min(r1.best_value, r2.best_value)
                    merged.trial_count = r1.trial_count + r2.trial_count
                    merged.success_count = r1.success_count + r2.success_count
                    merged.cov = new_cov

                    # Combine local data if available
                    if hasattr(r1, 'local_X') and hasattr(r2, 'local_X'):
                        if len(r1.local_X) > 0 and len(r2.local_X) > 0:
                            merged.local_X = np.vstack([r1.local_X, r2.local_X])
                            merged.local_y = np.hstack([r1.local_y, r2.local_y])
                        elif len(r1.local_X) > 0:
                            merged.local_X = r1.local_X.copy()
                            merged.local_y = r1.local_y.copy()
                        elif len(r2.local_X) > 0:
                            merged.local_X = r2.local_X.copy()
                            merged.local_y = r2.local_y.copy()

                    # Replace regions
                    self.regions = [r for idx, r in enumerate(self.regions) 
                                  if idx not in [i, j]]
                    self.regions.append(merged)
                    
                    print(f"[MERGE-DIST] Regions {r1.region_id} + {r2.region_id} → {merged.region_id} "
                          f"(dist={mahal_dist:.3f})")
                    return  # Only merge one pair at a time

    def _add_diverse_region(self, bounds, n_dims, rng, global_X, global_y):
        """
        Enhanced diverse region spawning with better exploration strategy
        """
        # Helper function to safely convert tensors
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        # Ensure inputs are numpy arrays
        global_X = to_numpy(global_X)
        global_y = to_numpy(global_y)
        
        if len(global_X) == 0:
            # No data yet, spawn randomly
            new_center = rng.uniform(bounds[:, 0], bounds[:, 1])
        else:
            # Sophisticated candidate generation
            n_candidates = min(1000, max(200, len(global_X) * 5))
            
            try:
                sobol = Sobol(d=n_dims, scramble=True, seed=rng.integers(1e6))
                candidates = bounds[:, 0] + sobol.random(n_candidates) * (bounds[:, 1] - bounds[:, 0])
            except ImportError:
                candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, n_dims))

            # Multi-objective candidate evaluation
            objectives = self._evaluate_spawn_candidates(candidates, global_X, global_y)
            
            # Use NSGA-II to select best candidate
            fronts, _ = self.fast_non_dominated_sort(objectives)
            best_front = fronts[0]
            
            if len(best_front) > 1:
                # Use crowding distance to pick from Pareto front
                front_objs = objectives[best_front]
                cd = self.crowding_distance(front_objs)
                best_idx_in_front = np.argmax(cd)
                selected_idx = best_front[best_idx_in_front]
            else:
                selected_idx = best_front[0]
                
            new_center = candidates[selected_idx]

        # Adaptive radius based on local density
        spawn_radius = self._compute_adaptive_radius(new_center, bounds, global_X)

        # Create new region with enhanced exploration properties
        new_region = TrustRegion(new_center, spawn_radius, len(self.regions), n_dims)
        new_region.exploration_bonus = 1.5 + 0.3 * rng.random()
        self.regions.append(new_region)

        print(f"[DIVERSE-SPAWN] Region #{len(self.regions)} at {new_center} "
              f"(radius={spawn_radius:.3f})")

    def _evaluate_spawn_candidates(self, candidates, global_X, global_y):
        """
        Evaluate spawn candidates using multiple objectives:
        1. Distance from existing regions (maximize)
        2. GP uncertainty (maximize) 
        3. Distance from evaluated points (maximize)
        4. Predicted function value (minimize for exploitation)
        """
        n_candidates = len(candidates)
        objectives = np.zeros((n_candidates, 4))
        
        # Objective 1: Distance from existing region centers
        if len(self.regions) > 0:
            centers = np.stack([r.center for r in self.regions])
            min_region_dists = np.min(np.linalg.norm(
                candidates[:, None, :] - centers[None, :, :], axis=2
            ), axis=1)
            objectives[:, 0] = -min_region_dists  # Negative for minimization
        else:
            objectives[:, 0] = -np.ones(n_candidates)  # All equally good
        
        # Objective 2 & 4: GP predictions (uncertainty and mean)
        mean, std = self._predict_at_candidates(candidates)
        objectives[:, 1] = -std  # Maximize uncertainty
        objectives[:, 3] = mean   # Minimize predicted value
        
        # Objective 3: Distance from evaluated points
        if len(global_X) > 0:
            min_eval_dists = np.min(np.linalg.norm(
                candidates[:, None, :] - global_X[None, :, :], axis=2
            ), axis=1)
            objectives[:, 2] = -min_eval_dists  # Negative for minimization
        else:
            objectives[:, 2] = -np.ones(n_candidates)
        
        return objectives

    def _compute_adaptive_radius(self, center, bounds, global_X):
        """
        Compute adaptive radius based on local point density and domain size
        """
        # Base radius from domain size
        domain_size = np.mean(bounds[:, 1] - bounds[:, 0])
        base_radius = domain_size * 0.1
        
        if len(global_X) == 0:
            return base_radius
        
        # Local density adjustment
        distances = np.linalg.norm(global_X - center, axis=1)
        k = min(10, len(global_X))
        knn_dist = np.partition(distances, k-1)[k-1]
        
        # Adaptive scaling based on density
        density_radius = max(knn_dist * 0.5, base_radius * 0.3)
        
        # Consider existing region radii
        if len(self.regions) > 0:
            avg_radius = np.mean([r.radius for r in self.regions])
            final_radius = min(density_radius, avg_radius * 1.5)
        else:
            final_radius = density_radius
        
        return np.clip(final_radius, self.config.min_radius, self.config.init_radius)



class Foretuner:
    """Enhanced TURBO-M++ optimizer with modular architecture"""
    
    def __init__(self, config: TurboConfig = None):
        self.config = config or TurboConfig()
        self.surrogate_manager = SurrogateManager(self.config)
        self.region_manager = RegionManager(self.config)
        self.acquisition_manager = AcquisitionManager(self.config)
        self.candidate_generator = CandidateGenerator(self.config, self.acquisition_manager)
        
        # Connect managers
        self.region_manager.set_surrogate_manager(self.surrogate_manager)
        
        # Core state
        self.global_X = None
        self.global_y = None
        self.iteration = 0
        self.global_best_history = []
        self.stagnation_counter = 0
        self.last_global_improvement = 0
    
    @property
    def regions(self):
        """Access regions through manager"""
        return self.region_manager.regions
    
    def optimize(self, objective_fn: Callable, bounds: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, float]:
        """Main optimization loop with cleaner separation of concerns"""
        n_dims = bounds.shape[0]
        rng = np.random.default_rng(seed)
        
        # Initialization
        self._initialize_optimization(objective_fn, bounds, rng, n_dims)
        
        # Main optimization loop
        for self.iteration in range(self.config.n_init, self.config.max_evals, self.config.batch_size):
            self._update_context()
            
            # Region management
            if self.iteration % self.config.management_frequency == 0:
                self.region_manager.manage_regions(bounds, n_dims, rng, self.global_X, self.global_y)
            
            # Generate and evaluate candidates
            candidates = self.candidate_generator.generate_candidates(
                bounds, rng, [r for r in self.regions if r.is_active], self.surrogate_manager
            )
            
            y_new = np.array([objective_fn(x) for x in candidates])
            
            # Update global data and regions
            self._update_global_data(candidates, y_new)
            self._track_progress()
            
            if self.iteration % 10 == 0:
                self._print_progress()
        
        return self._get_best_solution()
    
    def _initialize_optimization(self, objective_fn, bounds, rng, n_dims):
        """Initialize optimization state"""
        X_init = self._initialize_points(n_dims, bounds, rng)
        y_init = np.array([objective_fn(x) for x in X_init])
        self.global_X = X_init
        self.global_y = y_init
        
        # Update managers
        self.surrogate_manager.update_data(self.global_X, self.global_y)
        self.region_manager.initialize_regions(X_init, y_init, n_dims, rng)
        
        best_y = np.min(self.global_y)
        self.global_best_history.append(float(best_y))
        print(f"Initial best: {best_y:.6f}")
    
    def _update_context(self):
        """Update context for all managers"""
        self.candidate_generator.set_context(self.iteration, self.stagnation_counter)
    
    def _update_global_data(self, candidates, y_new):
        """Update global data and regions"""
        self.global_X = np.vstack([self.global_X, candidates])
        self.global_y = np.append(self.global_y, y_new)
        
        # Update managers
        self.surrogate_manager.update_data(self.global_X, self.global_y)
        self.region_manager.update_regions_with_new_data(candidates, y_new)
    
    def _track_progress(self):
        """Track optimization progress"""
        current_best_y = np.min(self.global_y)
        if len(self.global_best_history) == 0 or current_best_y < self.global_best_history[-1] - 1e-6:
            self.last_global_improvement = 0
            self.stagnation_counter = 0
            # print(f"Trial {self.iteration}: New best = {current_best_y:.6f}")
        else:
            self.last_global_improvement += 1
            #print(self.last_global_improvement, "last global improvement")
            #print(f"Trial {self.iteration}: No improvement, best = {current_best_y:.6f}")
            if self.last_global_improvement > 3:
                self.stagnation_counter += 1
        
        self.global_best_history.append(float(current_best_y))
    
    def _print_progress(self):
        """Print optimization progress"""
        best_y = np.min(self.global_y)
        active_regions = sum(1 for r in self.regions if r.is_active)
        avg_health = np.mean([r.health_score for r in self.regions])
        avg_radius = np.mean([r.radius for r in self.regions])
        print(f"Trial {self.iteration}: Best = {best_y:.6f}, "
              f"Active regions = {active_regions}, Avg health = {avg_health:.3f}, "
              f"Avg radius = {avg_radius:.4f}, Stagnation = {self.stagnation_counter}")
    
    def _get_best_solution(self):
        """Return best solution found"""
        best_idx = np.argmin(self.global_y)
        return self.global_X[best_idx], self.global_y[best_idx]
    
    def _initialize_points(self, n_dims, bounds, rng):
        """Initialize points using Sobol + random sampling"""
        sobol_part = sobol_sequence(rng.integers(0, 10000), self.config.n_init, n_dims)
        rand_part = rng.uniform(0, 1, (self.config.n_init // 4, n_dims))
        all_samples = np.vstack([sobol_part, rand_part])
        low, high = bounds[:, 0], bounds[:, 1]
        return low + all_samples * (high - low)
    

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
try:
    import seaborn as sns
    from pandas.plotting import parallel_coordinates
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

class Trial:
    """Simple trial object to hold optimization data"""
    def __init__(self, params: Dict[str, float], value: float, is_feasible: bool = True):
        self.params = params
        self.value = value
        self.is_feasible = is_feasible
        self.constraint_violations = []  # For compatibility

def plot_foretuner_results(optimizer, bounds: np.ndarray, param_names: List[str] = None, title: str = "Foretuner Optimization Results"):
    """
    Plot optimization results for Foretuner class
    
    Args:
        optimizer: Foretuner instance after optimization
        bounds: Parameter bounds array (n_dims x 2)
        param_names: List of parameter names (optional)
        title: Plot title
    """
    
    # Extract data from optimizer
    X = optimizer.global_X
    y = optimizer.global_y
    
    if param_names is None:
        param_names = [f"x{i}" for i in range(X.shape[1])]
    
    # Convert to trial objects for compatibility with existing plot function
    trials = []
    for i in range(len(X)):
        params = {param_names[j]: X[i, j] for j in range(len(param_names))}
        trial = Trial(params=params, value=y[i], is_feasible=True)
        trials.append(trial)
    
    # Use the existing plot function
    plot_optimization_results(trials, title)

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
    ax.plot(all_values, "o-", alpha=0.4, label="All", color="lightblue")

    feasible_indices = [i for i, t in enumerate(trials) if t.is_feasible]
    ax.plot(
        feasible_indices,
        feasible_values,
        "o-",
        alpha=0.8,
        label="Feasible",
        color="blue",
    )

    best_values = [min(feasible_values[: i + 1]) for i in range(len(feasible_values))]
    ax.plot(feasible_indices, best_values, "r-", linewidth=3, label="Best Feasible")

    # Optional: Initial BO cutoff
    init_cutoff = len([t for t in trials if getattr(t, "is_initial", False)])
    if init_cutoff > 0:
        ax.axvline(init_cutoff, color="gray", linestyle="--", label="Start BO")

    ax.set_title(f"{title} - Convergence")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate best point
    if feasible_values:
        best_idx = feasible_indices[np.argmin(feasible_values)]
        best_val = min(feasible_values)
        ax.annotate(
            f"Best: {best_val:.4f}",
            xy=(best_idx, best_val),
            xytext=(10, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=9,
            color="green",
        )

    # --- Log Convergence ---
    ax = axes[0, 1]
    best_values_pos = np.maximum(best_values, 1e-10)
    ax.semilogy(best_values_pos, "r-", linewidth=2)
    ax.set_title("Log Convergence")
    ax.set_xlabel("Feasible Trial")
    ax.set_ylabel("Best Value (log)")
    ax.grid(True, alpha=0.3)

    # --- Value Distribution (Histogram or KDE) ---
    ax = axes[0, 2]
    if len(feasible_values) > 1:
        try:
            if SEABORN_AVAILABLE:
                sns.kdeplot(feasible_values, ax=ax, fill=True, color="skyblue")
            else:
                ax.hist(feasible_values, bins="auto", edgecolor="black", alpha=0.7, color="skyblue")
            ax.axvline(
                min(feasible_values),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Best: {min(feasible_values):.4f}",
            )
            ax.legend()
        except Exception:
            ax.hist(
                feasible_values,
                bins="auto",
                edgecolor="black",
                alpha=0.7,
                color="skyblue",
            )
        ax.set_title("Objective Value Distribution")
    else:
        ax.text(
            0.5,
            0.5,
            f"Single Value:\n{feasible_values[0]:.4f}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    ax.set_xlabel("Objective Value")
    ax.set_ylabel("Density/Frequency")
    ax.grid(True, alpha=0.3)

    # --- Constraint Violations ---
    ax = axes[1, 0]
    constraint_counts = [
        len(t.constraint_violations) if hasattr(t, "constraint_violations") else 0
        for t in trials
    ]
    if any(constraint_counts):
        ax.plot(constraint_counts, "o-", color="orange", alpha=0.7)
        ax.set_title("Constraint Violations")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Violations")
    else:
        ax.text(
            0.5,
            0.5,
            "No Constraints\nor All Feasible",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Constraint Status")
    ax.grid(True, alpha=0.3)

    # --- Improvement Rate ---
    ax = axes[1, 1]
    if len(best_values) > 10:
        window = min(10, len(best_values) // 4)
        improvements = [
            (best_values[i - window] - best_values[i])
            / (abs(best_values[i - window]) + 1e-10)
            for i in range(window, len(best_values))
        ]
        ax.plot(range(window, len(best_values)), improvements, "g-", linewidth=2)
        ax.set_title("Improvement Rate")
        ax.set_xlabel(f"Trial (window: {window})")
        ax.set_ylabel("Relative Improvement")
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient Data\nfor Rate Analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    ax.grid(True, alpha=0.3)

    # --- Parameter Space: 1D or 2D ---
    ax = axes[1, 2]
    param_keys = list(trials[0].params.keys())

    if len(param_keys) >= 2:
        # === 2D case ===
        x_all = [t.params[param_keys[0]] for t in trials]
        y_all = [t.params[param_keys[1]] for t in trials]
        vals_all = [t.value for t in trials]
        feas_flags = [t.is_feasible for t in trials]

        x_feas = [x for x, f in zip(x_all, feas_flags) if f]
        y_feas = [y for y, f in zip(y_all, feas_flags) if f]
        val_feas = [v for v, f in zip(vals_all, feas_flags) if f]

        x_infeas = [x for x, f in zip(x_all, feas_flags) if not f]
        y_infeas = [y for y, f in zip(y_all, feas_flags) if not f]

        scatter = ax.scatter(
            x_feas,
            y_feas,
            c=val_feas,
            cmap="viridis_r",
            edgecolors="black",
            s=60,
            alpha=0.8,
            label="Feasible",
        )
        ax.scatter(
            x_infeas, y_infeas, marker="x", color="red", s=50, label="Infeasible"
        )

        if val_feas:
            best_idx = np.argmin(val_feas)
            ax.scatter(
                x_feas[best_idx],
                y_feas[best_idx],
                marker="*",
                s=200,
                c="gold",
                edgecolors="black",
                linewidths=1.5,
                label="Best",
            )

        plt.colorbar(scatter, ax=ax, label="Objective Value")
        ax.set_xlabel(param_keys[0])
        ax.set_ylabel(param_keys[1])
        ax.set_title("2D Parameter Space (colored by value)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif len(param_keys) == 1:
        # === 1D case ===
        x = [t.params[param_keys[0]] for t in trials]
        y = [t.value for t in trials]
        feas_flags = [t.is_feasible for t in trials]

        x_feas = [xi for xi, f in zip(x, feas_flags) if f]
        y_feas = [yi for yi, f in zip(y, feas_flags) if f]
        x_infeas = [xi for xi, f in zip(x, feas_flags) if not f]
        y_infeas = [yi for yi, f in zip(y, feas_flags) if not f]

        ax.scatter(
            x_feas,
            y_feas,
            c="blue",
            label="Feasible",
            edgecolors="black",
            alpha=0.7,
            s=60,
        )
        ax.scatter(x_infeas, y_infeas, c="red", marker="x", label="Infeasible", s=50)

        if y_feas:
            best_idx = np.argmin(y_feas)
            ax.scatter(
                x_feas[best_idx],
                y_feas[best_idx],
                marker="*",
                s=200,
                c="gold",
                edgecolors="black",
                linewidths=1.5,
                label="Best",
            )

        ax.set_xlabel(param_keys[0])
        ax.set_ylabel("Objective Value")
        ax.set_title("1D Parameter Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)

    else:
        ax.text(
            0.5,
            0.5,
            "No parameters to visualize",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Parameter Space")

    plt.tight_layout()
    plt.show()

    # --- Optional: Parallel Coordinates if >2D ---
    if len(trials[0].params) > 2 and SEABORN_AVAILABLE:
        try:
            df = pd.DataFrame([dict(**t.params, value=t.value) for t in feasible_trials])
            df["label"] = pd.qcut(df["value"], q=3, labels=["High", "Medium", "Low"])
            plt.figure(figsize=(12, 6))
            parallel_coordinates(
                df[["label"] + list(trials[0].params.keys())],
                class_column="label",
                colormap="coolwarm",
                alpha=0.6,
            )
            plt.title("Parallel Coordinates (Parameter Patterns)")
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"Could not create parallel coordinates plot: {e}")

    # --- Optional: Parameter Correlation Heatmap ---
    if SEABORN_AVAILABLE:
        try:
            df_params = pd.DataFrame([t.params for t in feasible_trials])
            if not df_params.empty and len(df_params.columns) > 1:
                df_params["value"] = feasible_values
                plt.figure(figsize=(10, 6))
                sns.heatmap(df_params.corr(), annot=True, fmt=".2f", cmap="coolwarm")
                plt.title("Parameter Correlation Matrix")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Could not create correlation heatmap: {e}")

    # --- Optional: Interactive hover ---
    if MPLCURSORS_AVAILABLE:
        mplcursors.cursor(hover=True)

    # --- Summary Statistics ---
    print("\n📊 Optimization Summary:")
    print(f"   Total trials: {len(trials)}")
    print(
        f"   Feasible trials: {len(feasible_trials)} ({len(feasible_trials) / len(trials) * 100:.1f}%)"
    )
    print(f"   Best value: {min(feasible_values):.6f}")
    print(f"   Value range: {max(feasible_values) - min(feasible_values):.6f}")
    if len(best_values) > 10:
        final_improv = (best_values[-10] - best_values[-1]) / (
            abs(best_values[-10]) + 1e-10
        )
        print(f"   Final convergence rate (last 10): {final_improv:.4f}")

# Example usage:
# After running optimization:
# plot_foretuner_results(optimizer, bounds, param_names=["x1", "x2"], title="My Optimization")