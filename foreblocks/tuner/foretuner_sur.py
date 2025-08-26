# ============================================
# Simplified Surrogate Manager — BoTorch Style
# ============================================
from functools import lru_cache
from typing import Optional, Tuple

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
from botorch.fit import fit_gpytorch_mll

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import fast_pred_var


class SurrogateManager:
    """
    Simplified surrogate manager following BoTorch patterns:
    - Uses BoTorch's SingleTaskGP as the primary model
    - Automatic input/output transforms
    - Simplified caching strategy  
    - Clean separation of global and local models
    - Native BoTorch acquisition function support
    """
    
    def __init__(self, config, device="cuda", normalize_inputs=True):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        self.normalize_inputs = normalize_inputs
        
        # Global model and data
        self.global_model = None
        self.global_X = None
        self.global_y = None
        self.global_mll = None
        
        # Local model cache (simple dict, no complex versioning)
        self.local_cache = {}
        self.cache_hits = 0
        
        # Simple prediction cache
        self._pred_cache = {}
        self._cache_version = 0
        
    def update_global_data(self, X, y):
        """Update global model with new data"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        
        # Convert to tensors
        self.global_X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.global_y = torch.tensor(y, dtype=self.dtype, device=self.device)
        
        # Fit global model if we have enough data
        if len(y) >= 3:
            self._fit_global_model()
            self._cache_version += 1
            self._pred_cache.clear()
    
    def _fit_global_model(self):
        """Fit global GP using BoTorch's SingleTaskGP"""
        try:
            # Create model with automatic transforms
            self.global_model = SingleTaskGP(
                train_X=self.global_X,
                train_Y=self.global_y.unsqueeze(-1),
                input_transform=Normalize(d=self.global_X.shape[-1]) if self.normalize_inputs else None,
                outcome_transform=Standardize(m=1),
            )
            
            # Set up MLL and fit
            self.global_mll = ExactMarginalLogLikelihood(
                self.global_model.likelihood, 
                self.global_model
            )
            
            # Use BoTorch's standard fitting procedure
            fit_gpytorch_mll(self.global_mll, max_attempts=3)
            
            # Set to eval mode
            self.global_model.eval()
            
        except Exception as e:
            print(f"Warning: Global GP fitting failed: {e}")
            self.global_model = None
    
    def predict_global_cached(self, X_query):
        """Predict using global model with simple caching"""
        X_query = np.asarray(X_query, dtype=np.float64)
        
        if self.global_model is None:
            # Return prior mean/std
            mean = np.mean(self.global_y.cpu().numpy()) if self.global_y is not None else 0.0
            return np.full(len(X_query), mean), np.ones(len(X_query))
        
        # Simple cache key (rounded to avoid numerical issues)
        cache_key = (self._cache_version, hash(np.round(X_query, 6).tobytes()))
        
        if cache_key in self._pred_cache:
            return self._pred_cache[cache_key]
        
        # Convert to tensor and predict
        X_tensor = torch.tensor(X_query, dtype=self.dtype, device=self.device)
        
        with torch.no_grad(), fast_pred_var():
            posterior = self.global_model.posterior(X_tensor)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            std = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1).cpu().numpy()
        
        result = (mean, std)
        
        # Cache result (limit cache size)
        if len(self._pred_cache) < 100:
            self._pred_cache[cache_key] = result
        
        return result
    
    def predict_local(self, X_query, local_X, local_y, region_radius=None):
        """Predict using local model with caching"""
        X_query = np.asarray(X_query, dtype=np.float64)
        local_X = np.asarray(local_X, dtype=np.float64)
        local_y = np.asarray(local_y, dtype=np.float64).reshape(-1)
        
        # Fall back to global if insufficient local data
        if len(local_y) < 4:
            return self.predict_global_cached(X_query)
        
        # Simple cache key based on local data
        local_key = (len(local_y), hash(np.round(local_X, 4).tobytes()))
        
        # Get or create local model
        if local_key not in self.local_cache:
            try:
                local_model = self._fit_local_model(local_X, local_y)
                # Limit cache size
                if len(self.local_cache) > 10:
                    # Remove oldest entry
                    oldest_key = next(iter(self.local_cache))
                    del self.local_cache[oldest_key]
                self.local_cache[local_key] = local_model
            except Exception:
                return self.predict_global_cached(X_query)
        else:
            local_model = self.local_cache[local_key]
            self.cache_hits += 1
        
        # Predict with local model
        try:
            X_tensor = torch.tensor(X_query, dtype=self.dtype, device=self.device)
            with torch.no_grad(), fast_pred_var():
                posterior = local_model.posterior(X_tensor)
                mean = posterior.mean.squeeze(-1).cpu().numpy()
                std = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1).cpu().numpy()
            return mean, std
        except Exception:
            return self.predict_global_cached(X_query)
    
    def _fit_local_model(self, X, y):
        """Fit a local GP model"""
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        y_tensor = torch.tensor(y, dtype=self.dtype, device=self.device)
        
        # Create local model (similar to global but potentially smaller)
        model = SingleTaskGP(
            train_X=X_tensor,
            train_Y=y_tensor.unsqueeze(-1),
            input_transform=Normalize(d=X_tensor.shape[-1]) if self.normalize_inputs else None,
            outcome_transform=Standardize(m=1),
        )
        
        # Fit with fewer iterations for speed
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, max_attempts=2, max_iter=50)
        
        model.eval()
        return model
    
    def ei_and_grad(self, X_query, best_f):
        """Expected Improvement and its gradient"""
        X_query = np.asarray(X_query, dtype=np.float64)
        
        if self.global_model is None:
            return np.zeros(len(X_query)), np.zeros_like(X_query)
        
        try:
            # Convert to tensor with gradient tracking
            X_tensor = torch.tensor(X_query, dtype=self.dtype, device=self.device, requires_grad=True)
            
            # Get posterior
            with gpytorch.settings.fast_pred_var(False):
                posterior = self.global_model.posterior(X_tensor)
                mean = posterior.mean.squeeze(-1)
                var = posterior.variance.squeeze(-1)
                std = var.clamp_min(1e-9).sqrt()
            
            # Compute EI
            z = (best_f - mean) / std
            phi = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            Phi = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
            ei = std * (z * Phi + phi)
            
            # Compute gradient
            ei_sum = ei.sum()
            grad = torch.autograd.grad(ei_sum, X_tensor)[0]
            
            return ei.detach().cpu().numpy(), grad.detach().cpu().numpy()
            
        except Exception:
            # Fallback to finite differences
            def ei_func(x):
                mean, std = self.predict_global_cached(x)
                std = np.maximum(std, 1e-9)
                z = (best_f - mean) / std
                phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
                from scipy.special import erf
                Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
                return std * (z * Phi + phi)
            
            ei_val = ei_func(X_query)
            grad = self._finite_diff_grad(X_query, ei_func)
            return ei_val, grad
    
    def _finite_diff_grad(self, X, func, eps=1e-6):
        """Simple finite difference gradient"""
        X = np.asarray(X)
        grad = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, i] += eps
            X_minus[:, i] -= eps
            
            f_plus = func(X_plus)
            f_minus = func(X_minus)
            grad[:, i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def gp_posterior_samples(self, X_query, n_samples=1, seed=None):
        """Sample from GP posterior"""
        if seed is not None:
            torch.manual_seed(seed)
            
        X_query = np.asarray(X_query, dtype=np.float64)
        
        if self.global_model is None:
            # Sample from prior
            mean = np.mean(self.global_y.cpu().numpy()) if self.global_y is not None else 0.0
            return np.random.normal(mean, 1.0, size=(n_samples, len(X_query)))
        
        try:
            X_tensor = torch.tensor(X_query, dtype=self.dtype, device=self.device)
            
            with torch.no_grad():
                posterior = self.global_model.posterior(X_tensor)
                samples = posterior.rsample(sample_shape=torch.Size([n_samples]))
                
            return samples.squeeze(-1).cpu().numpy()
            
        except Exception:
            # Fallback sampling
            mean, std = self.predict_global_cached(X_query)
            return np.random.normal(mean, std, size=(n_samples, len(X_query)))
    
    def get_lengthscales(self):
        """Extract lengthscales from global model"""
        if self.global_model is None:
            return None
            
        try:
            covar_module = self.global_model.covar_module
            if hasattr(covar_module, 'base_kernel'):
                kernel = covar_module.base_kernel
            else:
                kernel = covar_module
                
            if hasattr(kernel, 'lengthscale'):
                ls = kernel.lengthscale.detach().cpu().numpy()
                return ls.squeeze() if ls.ndim > 1 else ls
        except Exception:
            pass
        
        return None
    
    def posterior_handle(self):
        """Return a posterior handle for batch acquisition optimization"""
        return BoTorchPosteriorHandle(self)
    
    def get_best_value(self):
        """Get current best observed value"""
        if self.global_y is not None:
            return float(self.global_y.min())
        return float('inf')
    
    def clear_cache(self):
        """Clear all caches"""
        self._pred_cache.clear()
        self.local_cache.clear()
        self.cache_hits = 0
    
    # Alias methods for backward compatibility
    def update_data(self, X, y):
        return self.update_global_data(X, y)
    
    def predict_global(self, X):
        return self.predict_global_cached(X)

class BoTorchPosteriorHandle:
    """
    Complete posterior handle that wraps the surrogate manager.
    Provides full fantasy conditioning support for greedy batch acquisition.
    """
    
    def __init__(self, surrogate_manager, fantasy_X=None, fantasy_y=None):
        self.surrogate = surrogate_manager
        self.fantasy_X = fantasy_X
        self.fantasy_y = fantasy_y
        self._conditioned_model = None
        
        # If we have fantasy data, create conditioned model
        if fantasy_X is not None and fantasy_y is not None:
            self._create_conditioned_model()
    
    def _create_conditioned_model(self):
        """Create a GP model conditioned on fantasy observations"""
        if self.surrogate.global_model is None or self.fantasy_X is None:
            return
        
        try:
            # Combine real and fantasy data
            real_X = self.surrogate.global_X
            real_y = self.surrogate.global_y
            
            fantasy_X_tensor = torch.tensor(
                self.fantasy_X, 
                dtype=self.surrogate.dtype, 
                device=self.surrogate.device
            )
            fantasy_y_tensor = torch.tensor(
                self.fantasy_y, 
                dtype=self.surrogate.dtype, 
                device=self.surrogate.device
            )
            
            # Concatenate real and fantasy data
            combined_X = torch.cat([real_X, fantasy_X_tensor], dim=0)
            combined_y = torch.cat([real_y, fantasy_y_tensor], dim=0)
            
            # Create new conditioned model
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Normalize, Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
            
            self._conditioned_model = SingleTaskGP(
                train_X=combined_X,
                train_Y=combined_y.unsqueeze(-1),
                input_transform=Normalize(d=combined_X.shape[-1]) if self.surrogate.normalize_inputs else None,
                outcome_transform=Standardize(m=1),
            )
            
            # Quick fit (fewer iterations for fantasy models)
            mll = ExactMarginalLogLikelihood(
                self._conditioned_model.likelihood, 
                self._conditioned_model
            )
            fit_gpytorch_mll(mll, max_attempts=1, max_iter=25)
            self._conditioned_model.eval()
            
        except Exception as e:
            print(f"Warning: Failed to create conditioned model: {e}")
            self._conditioned_model = None
    
    def predict(self, X):
        """Predict mean and std using conditioned model if available"""
        X = np.asarray(X, dtype=np.float64)
        
        # Use conditioned model if available
        if self._conditioned_model is not None:
            try:
                X_tensor = torch.tensor(X, dtype=self.surrogate.dtype, device=self.surrogate.device)
                
                with torch.no_grad():
                    posterior = self._conditioned_model.posterior(X_tensor)
                    mean = posterior.mean.squeeze(-1).cpu().numpy()
                    std = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1).cpu().numpy()
                
                return mean, std
            except Exception:
                pass
        
        # Fallback to original surrogate
        return self.surrogate.predict_global_cached(X)
    
    def sample_y(self, X, H=8, antithetic=True, rng=None):
        """Sample from posterior with optional antithetic sampling"""
        X = np.asarray(X, dtype=np.float64)
        
        # Handle random seed
        if rng is not None and hasattr(rng, '_bit_generator'):
            seed = rng.integers(0, 2**32)
            torch.manual_seed(seed)
        elif rng is not None:
            # Assume it's an integer seed
            torch.manual_seed(int(rng))
        
        # Use conditioned model if available
        if self._conditioned_model is not None:
            try:
                X_tensor = torch.tensor(X, dtype=self.surrogate.dtype, device=self.surrogate.device)
                
                with torch.no_grad():
                    posterior = self._conditioned_model.posterior(X_tensor)
                    
                    if antithetic and H % 2 == 0:
                        # Antithetic sampling for variance reduction
                        H_half = H // 2
                        base_samples = posterior.rsample(sample_shape=torch.Size([H_half]))
                        
                        # Create antithetic pairs
                        mean = posterior.mean.unsqueeze(0)  # [1, N, 1]
                        antithetic_samples = 2 * mean - base_samples
                        
                        samples = torch.cat([base_samples, antithetic_samples], dim=0)
                    else:
                        samples = posterior.rsample(sample_shape=torch.Size([H]))
                
                # Convert to numpy with correct shape [H, N]
                samples_np = samples.squeeze(-1).cpu().numpy()
                if samples_np.ndim == 1:  # Single query point
                    samples_np = samples_np.reshape(-1, 1)
                
                return samples_np
                
            except Exception as e:
                print(f"Warning: Conditioned sampling failed: {e}")
        
        # Fallback to surrogate sampling
        samples = self.surrogate.gp_posterior_samples(X, n_samples=H, seed=None)
        
        if antithetic and H % 2 == 0:
            H_half = H // 2
            mean, _ = self.predict(X)
            # Apply antithetic transformation
            samples[H_half:] = 2 * mean - samples[:H_half]
        
        return samples
    
    def conditional(self, X_new, y_new_mean):
        """Create new posterior handle conditioned on fantasy observations"""
        X_new = np.asarray(X_new, dtype=np.float64)
        y_new_mean = np.asarray(y_new_mean, dtype=np.float64).reshape(-1)
        
        # Combine with existing fantasy data if any
        if self.fantasy_X is not None:
            combined_X = np.vstack([self.fantasy_X, X_new])
            combined_y = np.hstack([self.fantasy_y, y_new_mean])
        else:
            combined_X = X_new
            combined_y = y_new_mean
        
        # Return new handle with fantasy conditioning
        return BoTorchPosteriorHandle(
            self.surrogate, 
            fantasy_X=combined_X, 
            fantasy_y=combined_y
        )
    
    def sample_fantasy(self, X_pending):
        """Sample fantasy observations at pending points"""
        X_pending = np.asarray(X_pending, dtype=np.float64)
        
        # Sample single realization at pending points
        fantasy_samples = self.sample_y(X_pending, H=1, antithetic=False)
        fantasy_y = fantasy_samples.squeeze(0)  # Remove sample dimension
        
        return self.conditional(X_pending, fantasy_y)
    
    def get_fantasy_model(self):
        """Return the underlying conditioned model (for debugging)"""
        return self._conditioned_model
    
    def __repr__(self):
        fantasy_info = ""
        if self.fantasy_X is not None:
            fantasy_info = f" with {len(self.fantasy_X)} fantasy points"
        
        return f"BoTorchPosteriorHandle({self.surrogate.__class__.__name__}{fantasy_info})"
