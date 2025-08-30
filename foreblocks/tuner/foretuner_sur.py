from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior
from gpytorch.settings import fast_pred_var


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _round_key(arr: np.ndarray, ndig: int = 6) -> int:
    return hash(np.round(arr, ndig).tobytes())


class SurrogateManager:
    """
    Improved surrogate manager that fixes performance issues:
    - Balanced priors that don't over-constrain
    - Adaptive transforms based on data characteristics  
    - Smart model selection based on problem dimensionality
    - Better hyperparameter initialization
    """

    def __init__(
        self,
        config: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.double,
        auto_transforms: bool = True,
        adaptive_noise_floor: bool = True,
    ):
        self.config = config
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.dtype = dtype
        self.auto_transforms = auto_transforms
        self.adaptive_noise_floor = adaptive_noise_floor

        # Global data/model
        self.global_X: Optional[torch.Tensor] = None
        self.global_y: Optional[torch.Tensor] = None
        self.global_model = None
        self.global_mll = None

        # Data characteristics (for adaptive behavior)
        self._input_scales = None
        self._output_scale = None
        self._noise_estimate = None

        # Caches
        self._pred_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_version = 0
        self.local_cache: Dict[Tuple[int, int], Any] = {}
        self.cache_hits = 0
        self._local_cache_cap = int(getattr(config, "local_cache_cap", 12))

    def update_global_data(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        self.global_X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        self.global_y = torch.as_tensor(y, dtype=self.dtype, device=self.device)

        # Analyze data characteristics for adaptive behavior
        self._analyze_data_characteristics(X, y)

        if len(y) >= 3:
            self._fit_global_model()
            self._cache_version += 1
            self._pred_cache.clear()

    def _analyze_data_characteristics(self, X: np.ndarray, y: np.ndarray):
        """Analyze data to make adaptive modeling decisions"""
        # Input scale analysis
        self._input_scales = np.std(X, axis=0)
        input_range = np.ptp(X, axis=0)  # peak-to-peak
        
        # Output scale analysis  
        self._output_scale = np.std(y)
        
        # Noise estimation (using local variability)
        if len(y) > 10:
            # Estimate noise from nearest neighbor residuals
            from scipy.spatial.distance import cdist
            distances = cdist(X, X)
            np.fill_diagonal(distances, np.inf)
            
            noise_estimates = []
            for i in range(min(len(y), 50)):  # Sample for efficiency
                nearest_idx = np.argmin(distances[i])
                if distances[i, nearest_idx] < 0.1 * np.mean(input_range):
                    noise_estimates.append(abs(y[i] - y[nearest_idx]))
            
            if noise_estimates:
                self._noise_estimate = np.median(noise_estimates)
            else:
                self._noise_estimate = 0.01 * self._output_scale
        else:
            self._noise_estimate = 0.01 * self._output_scale

    def _get_adaptive_noise_floor(self) -> float:
        """Get noise floor adapted to data characteristics"""
        if not self.adaptive_noise_floor or self._noise_estimate is None:
            return 1e-6
            
        # Noise floor should be much smaller than estimated noise
        return max(1e-8, min(1e-4, self._noise_estimate / 100))

    def _should_normalize_inputs(self) -> bool:
        """Decide whether to normalize inputs based on data characteristics"""
        if not self.auto_transforms or self._input_scales is None:
            return True  # Default safe choice
            
        # Normalize if scales vary significantly across dimensions
        scale_ratio = np.max(self._input_scales) / (np.min(self._input_scales) + 1e-12)
        return scale_ratio > 3.0

    def _should_standardize_outputs(self) -> bool:
        """Decide whether to standardize outputs"""
        if not self.auto_transforms or self._output_scale is None:
            return True  # Default safe choice
            
        # Standardize if output scale is far from 1
        return abs(self._output_scale - 1.0) > 0.5

    def _make_exact_gp(self, X: torch.Tensor, y: torch.Tensor):
        """Create GP with balanced priors and adaptive transforms"""
        d = X.shape[-1]
        noise_floor = self._get_adaptive_noise_floor()
        
        # Balanced priors that don't over-constrain
        # Lengthscale prior: less restrictive, adapted to problem scale
        if d <= 5:
            # Low-D: allow wider range of lengthscales
            ls_prior = GammaPrior(1.5, 3.0)  # Less restrictive than original
        else:
            # High-D: mild preference for smaller lengthscales
            ls_prior = GammaPrior(2.0, 4.0)

        base = MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            lengthscale_prior=ls_prior,
            lengthscale_constraint=GreaterThan(1e-5),
        )
        
        # Outputscale prior: centered around reasonable values
        covar = ScaleKernel(
            base_kernel=base,
            outputscale_prior=LogNormalPrior(0.0, 0.75),  # Less broad than original
            outputscale_constraint=GreaterThan(1e-6),
        )

        # Adaptive transforms
        input_transform = Normalize(d=d) if self._should_normalize_inputs() else None
        outcome_transform = Standardize(m=1) if self._should_standardize_outputs() else None

        class _AdaptiveGP(SingleTaskGP):
            def __init__(self, X, Y, input_transform, outcome_transform, covar_module, noise_floor):
                super().__init__(
                    train_X=X,
                    train_Y=Y,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform,
                )
                self.mean_module = ConstantMean()
                self.covar_module = covar_module

                # Adaptive noise handling
                self.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(noise_floor))
                
                # Noise prior adapted to estimated noise level
                if hasattr(self, '_noise_estimate') and self._noise_estimate is not None:
                    # Prior centered around estimated noise
                    noise_prior_loc = math.log(max(noise_floor * 10, self._noise_estimate))
                    noise_prior = LogNormalPrior(noise_prior_loc, 1.0)
                else:
                    # Default prior
                    noise_prior = LogNormalPrior(-3.0, 1.0)
                    
                self.likelihood.noise_covar.register_prior("noise_prior", noise_prior, "noise")

        gp = _AdaptiveGP(X, y.unsqueeze(-1), input_transform, outcome_transform, covar, noise_floor)
        gp._noise_estimate = self._noise_estimate  # Store for noise prior
        return gp

    def _fit_global_model(self):
        """Fit global model with better initialization and error handling"""
        try:
            self.global_model = self._make_exact_gp(self.global_X, self.global_y).to(self.device, self.dtype)
            
            # Better hyperparameter initialization
            self._initialize_hyperparameters()
            
            self.global_mll = ExactMarginalLogLikelihood(self.global_model.likelihood, self.global_model)
            
            # Fit with multiple attempts and different strategies
            success = False
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # Standard fit
                        fit_gpytorch_mll(self.global_mll, max_attempts=1, max_iter=100)
                    elif attempt == 1:
                        # More conservative fit
                        fit_gpytorch_mll(self.global_mll, max_attempts=1, max_iter=50)
                        # Re-initialize if needed
                        self._initialize_hyperparameters()
                        fit_gpytorch_mll(self.global_mll, max_attempts=1, max_iter=100)
                    else:
                        # Minimal fit - just get something reasonable
                        fit_gpytorch_mll(self.global_mll, max_attempts=1, max_iter=25)
                    
                    success = True
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        print(f"[WARN] All GP fitting attempts failed: {e}")
                        continue
                    
            if not success:
                self.global_model = None
                self.global_mll = None
                return
                
            self.global_model.eval()
            
        except Exception as e:
            print(f"[WARN] Global GP creation failed: {e}")
            self.global_model = None
            self.global_mll = None

    def _initialize_hyperparameters(self):
        """Initialize hyperparameters based on data characteristics"""
        if self.global_model is None:
            return
            
        try:
            # Initialize lengthscales based on input scales
            if self._input_scales is not None:
                with torch.no_grad():
                    # Set lengthscales to reasonable fractions of input variation
                    target_ls = torch.tensor(
                        np.clip(self._input_scales * 0.3, 0.1, 2.0),
                        dtype=self.dtype,
                        device=self.device
                    )
                    self.global_model.covar_module.base_kernel.lengthscale = target_ls
                    
            # Initialize outputscale based on output variation
            if self._output_scale is not None:
                with torch.no_grad():
                    target_os = max(0.1, min(10.0, self._output_scale))
                    self.global_model.covar_module.outputscale = torch.tensor(
                        target_os, dtype=self.dtype, device=self.device
                    )
                    
            # Initialize noise based on estimate
            if self._noise_estimate is not None:
                with torch.no_grad():
                    target_noise = max(self._get_adaptive_noise_floor(), 
                                     min(0.1 * self._output_scale, self._noise_estimate))
                    self.global_model.likelihood.noise = torch.tensor(
                        target_noise, dtype=self.dtype, device=self.device
                    )
                    
        except Exception as e:
            print(f"[WARN] Hyperparameter initialization failed: {e}")

    # ----------------- Prediction (cached) -----------------
    def predict_global_cached(self, X_query) -> Tuple[np.ndarray, np.ndarray]:
        Xq = np.asarray(X_query, dtype=np.float64)
        if self.global_model is None:
            mean = float(self.global_y.min().item()) if self.global_y is not None else 0.0
            return np.full(len(Xq), mean, dtype=np.float64), np.ones(len(Xq), dtype=np.float64)

        key = (self._cache_version, _round_key(Xq, ndig=6))
        if key in self._pred_cache:
            return self._pred_cache[key]

        X_t = torch.as_tensor(Xq, dtype=self.dtype, device=self.device)
        with torch.no_grad(), fast_pred_var():
            post = self.global_model.posterior(X_t)
            mean = _to_np(post.mean.squeeze(-1))
            std = _to_np(post.variance.clamp_min(1e-12).sqrt().squeeze(-1))
            
        if len(self._pred_cache) < 256:
            self._pred_cache[key] = (mean, std)
        return mean, std

    # ----------------- Local model (reuse improved GP creation) -----------------
    def predict_local(self, X_query, local_X, local_y, region_radius=None):
        Xq = np.asarray(X_query, dtype=np.float64)
        X = np.asarray(local_X, dtype=np.float64)
        y = np.asarray(local_y, dtype=np.float64).reshape(-1)

        if len(y) < 4:
            return self.predict_global_cached(Xq)

        lkey = (len(y), _round_key(X, 4))
        if lkey not in self.local_cache:
            try:
                self.local_cache[lkey] = self._fit_local_model(X, y)
                if len(self.local_cache) > self._local_cache_cap:
                    self.local_cache.pop(next(iter(self.local_cache)))
            except Exception:
                return self.predict_global_cached(Xq)
        else:
            self.cache_hits += 1

        model = self.local_cache[lkey]
        X_t = torch.as_tensor(Xq, dtype=self.dtype, device=self.device)
        try:
            with torch.no_grad(), fast_pred_var():
                post = model.posterior(X_t)
                mean = _to_np(post.mean.squeeze(-1))
                std = _to_np(post.variance.clamp_min(1e-12).sqrt().squeeze(-1))
            return mean, std
        except Exception:
            return self.predict_global_cached(Xq)

    def _fit_local_model(self, X, y):
        """Fit local model using same improved approach as global"""
        X_t = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        y_t = torch.as_tensor(y, dtype=self.dtype, device=self.device)

        # Use simplified version of global model for local fitting
        model = self._make_exact_gp(X_t, y_t).to(self.device, self.dtype)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Quick fit for local model
        try:
            fit_gpytorch_mll(mll, max_attempts=1, max_iter=50)
        except Exception:
            # Fallback: minimal fit
            fit_gpytorch_mll(mll, max_attempts=1, max_iter=20)
            
        model.eval()
        return model

    # ----------------- Rest of methods remain the same -----------------
    def ei_and_grad(self, X_query, best_f):
        Xq = np.asarray(X_query, dtype=np.float64)
        if self.global_model is None:
            return np.zeros(len(Xq)), np.zeros_like(Xq)

        X_t = torch.as_tensor(Xq, dtype=self.dtype, device=self.device, requires_grad=True)
        try:
            with gpytorch.settings.fast_pred_var(False):
                post = self.global_model.posterior(X_t)
                m = post.mean.squeeze(-1)
                v = post.variance.clamp_min(1e-12).squeeze(-1)
                s = v.sqrt()

            z = (best_f - m) / s
            phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
            Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            ei = s * (z * Phi + phi)

            grad = torch.autograd.grad(ei.sum(), X_t, allow_unused=False)[0]
            return _to_np(ei), _to_np(grad)
        except Exception:
            # FD fallback
            def _ei(arr):
                mean, std = self.predict_global_cached(arr)
                std = np.maximum(std, 1e-12)
                z = (best_f - mean) / std
                phi = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
                Phi = 0.5 * (1.0 + torch.erf(torch.as_tensor(z / math.sqrt(2.0))).numpy())
                return std * (z * Phi + phi)

            ei = _ei(Xq)
            grad = self._finite_diff_grad(Xq, _ei)
            return ei, grad

    @staticmethod
    def _finite_diff_grad(X, func, eps=1e-6):
        X = np.asarray(X, dtype=np.float64)
        G = np.zeros_like(X)
        for j in range(X.shape[1]):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, j] += eps; Xm[:, j] -= eps
            fp, fm = func(Xp), func(Xm)
            G[:, j] = (fp - fm) / (2 * eps)
        return G

    def gp_posterior_samples(self, X_query, n_samples=1, seed: Optional[int] = None, antithetic: bool = False):
        Xq = np.asarray(X_query, dtype=np.float64)
        if self.global_model is None:
            mu = float(self.global_y.min().item()) if self.global_y is not None else 0.0
            return np.random.normal(mu, 1.0, size=(n_samples, len(Xq)))

        X_t = torch.as_tensor(Xq, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            post = self.global_model.posterior(X_t)
            if seed is not None:
                g = torch.Generator(device=self.device).manual_seed(int(seed))
            else:
                g = None

            if antithetic and n_samples % 2 == 0:
                h = n_samples // 2
                s1 = post.rsample(sample_shape=torch.Size([h]), base_samples=None, generator=g)
                s2 = 2 * post.mean.unsqueeze(0) - s1
                samples = torch.cat([s1, s2], dim=0)
            else:
                samples = post.rsample(sample_shape=torch.Size([n_samples]), base_samples=None, generator=g)
        return _to_np(samples.squeeze(-1))

    def get_lengthscales(self):
        if self.global_model is None:
            return None
        try:
            km = self.global_model.covar_module
            base = getattr(km, "base_kernel", km)
            ls = base.lengthscale
            return _to_np(ls).reshape(-1)
        except Exception:
            return None

    def posterior_handle(self):
        return BoTorchPosteriorHandle(self)

    def get_best_value(self) -> float:
        return float(self.global_y.min().item()) if self.global_y is not None else float("inf")

    def clear_cache(self):
        self._pred_cache.clear()
        self.local_cache.clear()
        self.cache_hits = 0

    # Back-compat
    def update_data(self, X, y):
        return self.update_global_data(X, y)

    def predict_global(self, X):
        return self.predict_global_cached(X)


class BoTorchPosteriorHandle:
    """Fantasy conditioning using condition_on_observations - same as before"""
    def __init__(self, surrogate, fantasy_X=None, fantasy_y=None):
        self.surrogate = surrogate
        self.device = surrogate.device
        self.dtype = surrogate.dtype

        self.fantasy_X = None if fantasy_X is None else np.asarray(fantasy_X, dtype=np.float64)
        self.fantasy_y = None if fantasy_y is None else np.asarray(fantasy_y, dtype=np.float64).reshape(-1)

        self._model = surrogate.global_model
        self._fantasy_model = None
        if self._model is not None and self.fantasy_X is not None and self.fantasy_y is not None:
            self._fantasize()

    def _fantasize(self):
        try:
            Xf = torch.as_tensor(self.fantasy_X, dtype=self.dtype, device=self.device)
            yf = torch.as_tensor(self.fantasy_y, dtype=self.dtype, device=self.device).unsqueeze(-1)
            self._fantasy_model = self._model.condition_on_observations(X=Xf, Y=yf)
            self._fantasy_model.eval()
        except Exception as e:
            print(f"[WARN] condition_on_observations failed: {e}")
            self._fantasy_model = None

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        mdl = self._fantasy_model or self._model
        if mdl is None:
            return self.surrogate.predict_global_cached(X)
        X_t = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad(), fast_pred_var():
            post = mdl.posterior(X_t)
            mean = _to_np(post.mean.squeeze(-1))
            std = _to_np(post.variance.clamp_min(1e-12).sqrt().squeeze(-1))
        return mean, std

    def sample_y(self, X, H=8, antithetic=True, rng=None):
        X = np.asarray(X, dtype=np.float64)
        seed = None
        if rng is not None:
            try:
                seed = int(rng.integers(0, 2**31 - 1))
            except Exception:
                try:
                    seed = int(rng)
                except Exception:
                    seed = None

        mdl = self._fantasy_model or self._model
        if mdl is None:
            return self.surrogate.gp_posterior_samples(X, n_samples=H, seed=seed, antithetic=antithetic)

        X_t = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            post = mdl.posterior(X_t)
            g = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
            if antithetic and H % 2 == 0:
                h = H // 2
                s1 = post.rsample(sample_shape=torch.Size([h]), generator=g)
                s2 = 2 * post.mean.unsqueeze(0) - s1
                samples = torch.cat([s1, s2], dim=0)
            else:
                samples = post.rsample(sample_shape=torch.Size([H]), generator=g)
        S = _to_np(samples.squeeze(-1))
        if S.ndim == 1:
            S = S.reshape(H, 1)
        return S

    def conditional(self, X_new, y_new_mean):
        X_new = np.asarray(X_new, dtype=np.float64)
        y_new = np.asarray(y_new_mean, dtype=np.float64).reshape(-1)
        if self.fantasy_X is None:
            fX = X_new; fy = y_new
        else:
            fX = np.vstack([self.fantasy_X, X_new])
            fy = np.hstack([self.fantasy_y, y_new])
        return BoTorchPosteriorHandle(self.surrogate, fX, fy)

    def sample_fantasy(self, X_pending):
        Xp = np.asarray(X_pending, dtype=np.float64)
        y_samp = self.sample_y(Xp, H=1, antithetic=False).squeeze(0)
        return self.conditional(Xp, y_samp)

    def get_fantasy_model(self):
        return self._fantasy_model

    def __repr__(self):
        extra = "" if self.fantasy_X is None else f" with {len(self.fantasy_X)} fantasies"
        return f"BoTorchPosteriorHandle({self.surrogate.__class__.__name__}{extra})"
