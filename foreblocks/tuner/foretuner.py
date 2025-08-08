import warnings

warnings.filterwarnings("ignore")

# ============================================
# ✅ Core Python & Concurrency
# ============================================
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

# ============================================
# ✅ GPyTorch & BoTorch (Gaussian Processes)
# ============================================
import gpytorch
# ============================================
# ✅ Visualization
# ============================================
import matplotlib.pyplot as plt
# ============================================
# ✅ Numerical & Scientific Computing
# ============================================
import numpy as np
import pandas as pd
# ============================================
# ✅ PyTorch Core
# ============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.transforms import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.models import ExactGP
from gpytorch.settings import fast_pred_var
from gpytorch.variational import CholeskyVariationalDistribution
# ============================================
# ✅ Parallelism & Performance
# ============================================
from scipy.spatial.distance import cdist
from scipy.stats import norm

# ============================================
# ✅ Project-Specific
# ============================================
from .foretuner_aux import *

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

    def __init__(
        self, params: Dict[str, float], value: float, is_feasible: bool = True
    ):
        self.params = params
        self.value = value
        self.is_feasible = is_feasible
        self.constraint_violations = []  # For compatibility


import numpy as np

# ----------------------------- Bayesian NN ---------------------------------


class BayesianNN(nn.Module):
    """
    Bayesian Neural Network surrogate using MC Dropout for epistemic uncertainty.

    ✅ Improvements:
    - Supports deeper networks & residual connections
    - Optional LayerNorm for stability
    - Flexible activation (ReLU, SiLU, Tanh)
    - MC Dropout temperature scaling
    - Stable mean/variance computation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout_p: float = 0.1,
        activation: str = "relu",
        n_hidden_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.dropout_p = dropout_p
        self.activation_fn = self._get_activation(activation)
        self.use_layernorm = use_layernorm
        self.n_hidden_layers = n_hidden_layers

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim) if use_layernorm else None

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )
        self.hidden_ln = (
            nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_hidden_layers)])
            if use_layernorm
            else None
        )

        # Output
        self.fc_out = nn.Linear(hidden_dim, 1)

    def _get_activation(self, name: str):
        if name == "relu":
            return F.relu
        elif name == "silu":
            return F.silu  # swish-like
        elif name == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _mc_dropout(self, x):
        # Force dropout during inference for MC Dropout sampling
        return F.dropout(x, p=self.dropout_p, training=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer
        x = self.activation_fn(self.input_layer(x))
        if self.input_ln:
            x = self.input_ln(x)
        x = self._mc_dropout(x)

        # Hidden layers with optional residuals
        for idx, layer in enumerate(self.hidden_layers):
            h = self.activation_fn(layer(x))
            if self.hidden_ln:
                h = self.hidden_ln[idx](h)
            h = self._mc_dropout(h)
            x = x + h * 0.5  # mild residual connection

        return self.fc_out(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int = 20):
        """
        Run MC Dropout sampling for uncertainty estimation.
        Returns mean & std for epistemic uncertainty.
        """
        preds = torch.stack(
            [self.forward(x) for _ in range(n_samples)], dim=0
        )  # (S, N, 1)

        # Stable variance computation
        mean, var = torch.var_mean(preds, dim=0, unbiased=False)
        std = torch.sqrt(var + 1e-9)

        return mean.squeeze(-1), std.squeeze(-1)


# ----------------------------- Sparse GP -----------------------------------


class SparseGPModel(SingleTaskVariationalGP):
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        inducing_points: torch.Tensor,
    ):
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


def _candidates_key(candidates: np.ndarray) -> tuple[int, bytes]:
    """Hashable key for caching: (n_dim, rounded-bytes)."""
    return candidates.shape[1], np.round(candidates, 6).astype(np.float64).tobytes()


#######################################################################
# SurrogateManager: Global & Local Surrogates for TuRBO-M
#######################################################################3


class RobustExactGP(ExactGP):
    """Exact GP with ARD Matern kernel and robust hyperparameter initialization."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # ✅ Mean
        self.mean_module = ConstantMean()

        # ✅ Kernel: ARD Matern ν=2.5 (smoother than ν=1.5 but less restrictive than RBF)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

        # ✅ Initialize hyperparameters based on data statistics
        if train_x.shape[0] > 1:
            with torch.no_grad():
                dists = torch.cdist(train_x, train_x)
                median_dist = torch.median(dists[dists > 0])
                self.covar_module.base_kernel.lengthscale = median_dist.clamp(min=1e-3)

        # ✅ Outputscale based on target variance
        y_std = train_y.std().clamp_min(1e-3)
        self.covar_module.outputscale = y_std**2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_exact_gp_model(model, likelihood, max_iter=75, patience=10):
    """More robust GP training with Adam + early stopping"""
    model.train()
    likelihood.train()

    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    best_loss = float("inf")
    patience_counter = 0

    for i in range(max_iter):
        optimizer.zero_grad()
        output = model(model.train_inputs[0])
        loss = -mll(output, model.train_targets)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            break

    model.eval()
    likelihood.eval()
    return model

from torch.autograd import grad


class SurrogateManager:
    """
    Manages global & local surrogate models for TuRBO-M.
    - Global GP used for global exploration.
    - Local GP cached per TrustRegion.
    """

    def __init__(
        self,
        config,
        device="cuda",
        global_backend="exact_gp",
        local_backend="exact_gp",
        normalize_inputs=True,
        n_inducing=50,
        bnn_hidden=64,
        bnn_dropout=0.1,
    ):
        self.config = config
        self.device = torch.device(device)

        # Backends: "exact_gp", "sparse_gp", "bnn"
        self.global_backend = global_backend
        self.local_backend = local_backend

        self.normalize_inputs = normalize_inputs
        self.n_inducing = n_inducing
        self.bnn_hidden = bnn_hidden
        self.bnn_dropout = bnn_dropout

        # Global data & model
        self.global_X = None
        self.global_y = None
        self.global_model = None

        # Cached trust-region models
        self.local_model_cache = {}

        # Versioning for cache invalidation
        self._model_version = 0

        # Cached posterior
        self._cached_posterior = self._make_cached_predict_fn()

    def gradient_global_mean(self, X_query: np.ndarray):
        """
        Compute the gradient of the GP predictive mean ∇μ(x) for each query.
        Works only for ExactGP (with autograd).
        Returns array [n_query, n_dim].
        """
        if self.global_model is None or self.global_backend != "exact_gp":
            # Fallback: no gradients
            return np.zeros_like(X_query)

        # Convert to tensor with gradients enabled
        Xq = self._to_tensor(X_query).clone().detach().requires_grad_(True)
        # Normalize like training inputs
        Xq_norm = self._normalize_inputs(Xq, self.global_X)

        # Predict posterior mean
        with torch.enable_grad():
            posterior = self.global_model.likelihood(self.global_model(Xq_norm))
            mean = posterior.mean.sum()  # sum over all for a single backward pass

        # Compute gradients ∂mean/∂X
        mean.backward()
        grads = Xq.grad.detach().cpu().numpy()
        return grads


    def ei_and_grad(self, X_np, f_best):
        """Compute EI(x) and ∇EI(x) at X for ExactGP global model."""
        if self.global_model is None:
            return np.zeros(len(X_np)), np.zeros_like(X_np)

        X_t = torch.tensor(X_np, dtype=torch.double, requires_grad=True, device=self.device)
        X_norm = self._normalize_inputs(X_t, self.global_X)

        model = self.global_model
        model.eval()
        with torch.set_grad_enabled(True):
            posterior = model.likelihood(model(X_norm))
            mean_t = posterior.mean
            std_t = posterior.variance.sqrt()
            z = (f_best - mean_t) / (std_t + 1e-9)
            ei_t = (f_best - mean_t) * torch.distributions.Normal(0, 1).cdf(z) + std_t * torch.exp(
                -0.5 * z ** 2
            ) / np.sqrt(2 * np.pi)

        ei_val = ei_t.detach().cpu().numpy()
        grads = []
        for i in range(ei_t.shape[0]):
            g = grad(ei_t[i], X_t, retain_graph=True)[0]
            grads.append(g.detach().cpu().numpy())
        return ei_val, np.stack(grads, axis=0)

    # ======================
    # Posterior Cache
    # ======================
    def _make_cached_predict_fn(self):
        @lru_cache(maxsize=128)
        def _cached_predict(
            n_dim: int, candidates_bytes: bytes, backend_name: str, model_version: int
        ):
            candidates = np.frombuffer(candidates_bytes, dtype=np.float64).reshape(
                -1, n_dim
            )
            mean, std = self._predict_from_model(
                self.global_model,
                self._to_tensor(candidates),
                self.global_X,
                backend=backend_name,
            )
            return mean, std

        return _cached_predict

    def clear_posterior_cache(self):
        self._cached_posterior.cache_clear()

    # ======================
    # Tensor utilities
    # ======================
    def _to_tensor(self, X):
        return torch.as_tensor(X, dtype=torch.double, device=self.device)

    def _normalize_inputs(self, X, ref_X):
        """Normalize X relative to ref_X (safe for small datasets)."""
        if not self.normalize_inputs:
            return X

        if ref_X is None or ref_X.shape[0] < 2:
            return X  # no normalization if no reference

        min_vals = ref_X.min(0)[0]
        max_vals = ref_X.max(0)[0]
        ranges = max_vals - min_vals
        ranges[ranges < 1e-12] = 1.0  # avoid div by zero
        return (X - min_vals) / (ranges + 1e-8)

    # ======================
    # Global Data Updates
    # ======================
    def update_global_data(self, X, y):
        """Add/update global dataset, retrain global model if enough points."""
        self.global_X = self._to_tensor(X)
        self.global_y = self._to_tensor(y).unsqueeze(-1)

        if self.global_X.shape[0] >= 5:  # only train if enough data
            self.global_model = self._fit_model(
                self.global_X, self.global_y, backend=self.global_backend
            )
            self._model_version += 1  # bump version to invalidate posterior cache
            self.clear_posterior_cache()

    def update_data(self, X, y):
        """Alias for backward compatibility."""
        return self.update_global_data(X, y)

    # ======================
    # Global Predict
    # ======================
    def predict_global_cached(self, candidates: np.ndarray):
        """Fast global GP prediction with caching."""
        if self.global_model is None:
            # fallback: mean=0 std=1
            return np.zeros(len(candidates)), np.ones(len(candidates))

        n_dim, key_bytes = _candidates_key(candidates)
        backend = self.global_backend
        version = self._model_version
        return self._cached_posterior(n_dim, key_bytes, backend, version)

    def predict_global(self, X_test):
        """Non-cached global prediction."""
        if self.global_model is None or self.global_X is None:
            mean = (
                torch.mean(self.global_y).item() if self.global_y is not None else 0.0
            )
            return np.ones(X_test.shape[0]) * mean, np.ones(X_test.shape[0])
        return self._predict_from_model(
            self.global_model,
            self._to_tensor(X_test),
            self.global_X,
            backend=self.global_backend,
        )

    # ======================
    # Local Predict
    # ======================
    def predict_local(self, X_test, local_X, local_y, region_key=None):
        """Predict within a region (local GP)."""
        # fallback for very small data
        if local_X is None or len(local_y) < 3:
            return self.predict_global_cached(X_test)

        if region_key in self.local_model_cache:
            model = self.local_model_cache[region_key]
        else:
            model = self._fit_model(
                self._to_tensor(local_X),
                self._to_tensor(local_y).unsqueeze(-1),
                backend=self.local_backend,
            )
            self.local_model_cache[region_key] = model

        return self._predict_from_model(
            model,
            self._to_tensor(X_test),
            self._to_tensor(local_X),
            backend=self.local_backend,
        )

    def predict_local_with_posterior(
        self, X_test, local_X, local_y, seed=0, n_samples=5, region_key=None
    ):
        """Draw posterior samples from local GP."""
        torch.manual_seed(seed)
        # fallback if no local data
        if local_X is None or len(local_y) < 3:
            mean, std = self.predict_global_cached(X_test)
            rng = np.random.default_rng(seed)
            return rng.normal(mean, std, size=(n_samples, len(X_test)))

        # ensure model in cache
        if region_key in self.local_model_cache:
            model = self.local_model_cache[region_key]
        else:
            model = self._fit_model(
                self._to_tensor(local_X),
                self._to_tensor(local_y).unsqueeze(-1),
                backend=self.local_backend,
            )
            self.local_model_cache[region_key] = model

        backend = self.local_backend
        Xq = self._to_tensor(X_test)
        Xq_norm = self._normalize_inputs(Xq, self._to_tensor(local_X))

        if backend in ["exact_gp", "sparse_gp"]:
            with fast_pred_var():
                with torch.autocast("cuda"):
                    posterior = model.likelihood(model(Xq_norm))

            samples = posterior.rsample(sample_shape=torch.Size([n_samples]))
            return samples.detach().cpu().numpy().squeeze(-1)

        elif backend == "bnn":
            model.eval()
            preds = []
            for _ in range(n_samples):
                preds.append(model(Xq_norm.float()))
            preds = torch.stack(preds)
            return preds.detach().cpu().numpy().squeeze(-1)

    # ======================
    # Fit Models
    # ======================
    def _fit_model(self, X, y, backend="exact_gp"):
        """Fit model using specified backend."""
        d = X.shape[-1]
        train_X = self._normalize_inputs(X, X)

        if backend == "exact_gp":
            # ✅ Robust likelihood with noise constraint
            likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6)).to(
                self.device
            )

            # ✅ Initialize noise ~ small fraction of target variance
            y_std = y.std().clamp_min(1e-3)
            likelihood.noise = (0.05 * y_std).clamp_min(1e-5)

            model = RobustExactGP(train_X, y.squeeze(-1), likelihood).to(self.device)

            # ✅ Robust training with Adam + early stopping
            model = fit_exact_gp_model(model, likelihood, max_iter=60, patience=8)

            # Attach likelihood for prediction
            model.likelihood = likelihood
            return model

        elif backend == "sparse_gp":
            # pick inducing points
            n_inducing = min(self.n_inducing, train_X.shape[0])
            idx = torch.randperm(train_X.shape[0])[:n_inducing]
            inducing_points = train_X[idx].clone().detach()
            model = SparseGPModel(
                train_X=train_X, train_Y=y, inducing_points=inducing_points
            ).to(self.device)

            mll = VariationalELBO(model.likelihood, model.model, num_data=y.size(0))

            model.train()
            model.likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

            for i in range(15):  # short training for speed
                optimizer.zero_grad()
                output = model(train_X)
                loss = -mll(output, y.squeeze(-1))
                loss.backward()
                optimizer.step()
                if loss.item() < 1e-5:
                    break

            model.eval()
            model.likelihood.eval()
            return model

        elif backend == "bnn":
            model = BayesianNN(
                d, hidden_dim=self.bnn_hidden, dropout_p=self.bnn_dropout
            ).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(300):  # fewer epochs for faster training
                opt.zero_grad()
                preds = model(train_X.float())
                loss = F.mse_loss(preds, y.float())
                loss.backward()
                opt.step()
            return model

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ======================
    # Predict from model
    # ======================
    def _predict_from_model(self, model, Xq, ref_X, backend):
        """Generic prediction (mean, std) for GP or BNN."""
        Xq_norm = self._normalize_inputs(Xq, ref_X)

        if backend in ["exact_gp", "sparse_gp"]:
            with fast_pred_var():
                posterior = model.likelihood(model(Xq_norm))
            mean = posterior.mean.detach().cpu().numpy().flatten()
            std = posterior.variance.sqrt().detach().cpu().numpy().flatten()
            return mean, std

        elif backend == "bnn":
            model.eval()
            with torch.no_grad():
                mean_t, std_t = model.predict(Xq_norm.float(), n_samples=20)
            return mean_t.cpu().numpy(), std_t.cpu().numpy()

    # ======================
    # Posterior samples from global model
    # ======================
    def gp_posterior_samples(self, X_test, seed=0, n_samples=1):
        """Posterior samples from global GP."""
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
            with fast_pred_var():
                posterior = self.global_model.likelihood(self.global_model(Xq_norm))
            samples = posterior.rsample(sample_shape=torch.Size([n_samples]))
            samples_np = samples.detach().cpu().numpy()
            # Safe squeeze only if last dim is 1
            if samples_np.ndim == 3 and samples_np.shape[-1] == 1:
                samples_np = samples_np[..., 0]
            return samples_np

        elif backend == "bnn":
            self.global_model.eval()
            preds = []
            for _ in range(n_samples):
                preds.append(self.global_model(Xq_norm.float()))
            preds = torch.stack(preds)
            return preds.detach().cpu().numpy().squeeze(-1)

    # ======================
    # Utility
    # ======================
    def get_best_value(self):
        return self.global_y.min().item() if self.global_y is not None else float("inf")


#######################################################################
# Region Management and Trust Region Optimization
#######################################################################


def compute_coverage_fraction(global_X, regions):
    """Fraction of points inside any trust region."""
    if not regions or len(global_X) == 0:
        return 0.0

    centers = np.stack([r.center for r in regions])
    radii = np.array([r.radius for r in regions])

    # Pairwise distances [N_points, N_regions]
    dists = np.linalg.norm(global_X[:, None, :] - centers[None, :, :], axis=2)

    # Point is covered if within *any* region radius
    covered_mask = (dists <= radii[None, :]).any(axis=1)

    return np.mean(covered_mask)  # fraction of points covered

def compute_mean_entropy_from_global_gp(surrogate_manager, X):
    """Compute mean predictive variance from global GP as exploration entropy."""
    if surrogate_manager is None or X is None or len(X) == 0:
        return 1.0
    mean, std = surrogate_manager.predict_global_cached(X)
    var = std ** 2
    return float(np.nan_to_num(np.mean(var), nan=1.0))


def compute_region_spread(regions):
    """Simple Wasserstein spread proxy: avg pairwise center distance."""
    if len(regions) < 2:
        return 0.0

    centers = np.stack([r.center for r in regions])
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    return np.mean(dmat[np.triu_indices(len(regions), k=1)])  # upper triangle avg


import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


class AcquisitionManager:
    def __init__(self, config, eta=0.3):
        self.config = config
        self.iteration = 0
        self.acquisition_type = config.acquisition
        self.stagnation_counter = 0

        self.acq_list = ["ei", "ucb", "kg", "pes", "ts"]
        self.n_acq = len(self.acq_list)
        self.eta = eta
        self.reward_temp = 0.5
        self.recent_rewards = np.zeros(self.n_acq)
        self.last_chosen_idx = None

        self.acquisition_functions = {
            "ei": expected_improvement,
            "log_ei": log_expected_improvement,
            "ucb": lambda m, s, b: upper_confidence_bound(m, s, config.ucb_beta),
            "pi": probability_improvement,
            "pes": predictive_entropy_search,
            "kg": knowledge_gradient,
        }

    def set_iteration(self, it):
        self.iteration = it

    def _progress(self):
        return min(1.0, self.iteration / max(1, self.config.max_evals))

    def _anneal_weights(self):
        p = self._progress()
        explore_w = max(0.05, (1.0 - p) ** 2)
        exploit_w = min(1.0, p * 1.5)
        if self.stagnation_counter > 10:
            explore_w *= 1.5
        return explore_w, exploit_w

    def _normalize(self, arr):
        arr = np.nan_to_num(arr, nan=0.0)
        return (arr - arr.min()) / (arr.ptp() + 1e-12)

    def compute_scores(self, mean, std, best_value, ts_score):
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        best_value = np.nan_to_num(best_value, nan=np.min(mean))

        explore_w, exploit_w = self._anneal_weights()
        entropy_level = np.mean(np.log1p(std))
        stagnation_factor = min(2.0, 1.0 + 0.1 * self.stagnation_counter)

        scores = {}
        for acq in self.acq_list:
            if acq == "ts":
                scores["ts"] = self._normalize(ts_score)
            else:
                scores[acq] = self._normalize(
                    self.acquisition_functions[acq](mean, std, best_value)
                )

        # Raw mixture weights
        weights = {
            "pes": explore_w * (1.0 + entropy_level),
            "ei":  exploit_w * (1.0 - 0.3 * entropy_level),
            "ucb": 0.3 * explore_w * stagnation_factor,
            "kg":  0.2 * exploit_w,
            "ts":  0.4 * (1.0 - self._progress()) + (0.3 if self.stagnation_counter > 5 else 0.0)
        }

        raw_mix = sum(weights[a] * scores[a] for a in self.acq_list)

        # Reward-weighted softmax mixture
        soft_w = np.exp(self.recent_rewards / self.reward_temp)
        soft_w /= np.sum(soft_w) + 1e-12
        soft_mix = sum(w * scores[acq] for w, acq in zip(soft_w, self.acq_list))

        final_scores = 0.5 * raw_mix + 0.5 * soft_mix
        return final_scores, "adaptive_auto"

    def optimize_in_region(self, region, bounds, rng, surrogate_manager):
        candidates = self._sample_region_candidates(region, bounds, rng)

        if region.local_y is not None and len(region.local_y) >= self.config.min_local_samples:
            X_local = np.array(region.local_X, copy=False)
            y_local = np.array(region.local_y, copy=False)
            mean, std = surrogate_manager.predict_local(candidates, X_local, y_local, region.radius)
        else:
            mean, std = surrogate_manager.predict_global_cached(candidates)

        ts_samples = surrogate_manager.gp_posterior_samples(candidates, n_samples=3)
        ts_score = np.min(ts_samples, axis=0)

        acq_scores, _ = self.compute_scores(mean, std, region.best_value, ts_score)
        acq_scores += 0.05 * std * region.exploration_bonus

        top_idx = np.argsort(-acq_scores)[: self.config.batch_size]
        return self._refine_candidate(candidates[top_idx[0]], region, bounds, surrogate_manager)

    def _sample_region_candidates(self, region, bounds, rng):
        n = int(self.config.n_candidates * (1.1 - 0.5 * self._progress()))
        dim = region.center.shape[0]
        eigvals, eigvecs = np.linalg.eigh(region.cov)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 1e-9, None)))

        sobol = i4_sobol_generate(dim, n)
        sobol_scaled = region.center + (sobol - 0.5) @ sqrt_cov.T * region.radius
        sobol_scaled = np.clip(sobol_scaled, bounds[:, 0], bounds[:, 1])
        return sobol_scaled

    def _refine_candidate(self, x, region, bounds, surrogate_manager, steps=3, step_size=0.25):
        f_best = region.best_value
        x_best = x.copy()
        ei_val, ei_grad = surrogate_manager.ei_and_grad(x_best[None], f_best)
        best_score = ei_val[0]

        for _ in range(steps):
            grad_vec = ei_grad[0]
            grad_vec /= np.linalg.norm(grad_vec) + 1e-9
            candidate = x_best + step_size * region.radius * grad_vec
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            ei_val_c, ei_grad_c = surrogate_manager.ei_and_grad(candidate[None], f_best)
            if ei_val_c[0] > best_score:
                x_best, best_score, ei_grad = candidate, ei_val_c[0], ei_grad_c

        return x_best

    def notify_iteration_result(self, improvement):
        if improvement <= 1e-9:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        if self.acquisition_type == "adaptive_auto" and self.last_chosen_idx is not None:
            self.recent_rewards[self.last_chosen_idx] = (
                0.9 * self.recent_rewards[self.last_chosen_idx] + 0.1 * improvement
            )

    def get_info(self):
        explore_w, exploit_w = self._anneal_weights()
        return {
            "iteration": self.iteration,
            "progress": self._progress(),
            "explore_w": explore_w,
            "exploit_w": exploit_w,
            "mode": self.acquisition_type,
            "stagnation": self.stagnation_counter,
            "recent_rewards": self.recent_rewards.tolist()
        }

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.spatial.distance import cdist


class CandidateGenerator:
    """
    State-of-the-art CandidateGenerator:
    ✅ Entropy + stagnation-aware exploration probability
    ✅ Sobol + uncertainty + region exploration hybrid
    ✅ Trust-region-aware exploitation with Pareto selection
    ✅ Fast k-center greedy diversity filtering
    ✅ Optional parallel sampling for scalability
    """

    def __init__(self, config, acquisition_manager):
        self.config = config
        self.acquisition_manager = acquisition_manager
        self.iteration = 0
        self.stagnation_counter = 0

        # Caching for exploration prob
        self._cached_exploration_prob = None
        self._cache_iteration = -1

    # ============================================
    # Context updates
    # ============================================
    def set_context(self, iteration, stagnation_counter):
        self.iteration = iteration
        self.stagnation_counter = stagnation_counter
        self.acquisition_manager.set_iteration(iteration)
        self._cache_iteration = -1  # invalidate exploration prob cache

    # ============================================
    # Entry point
    # ============================================
    def generate_candidates(self, bounds, rng, active_regions, surrogate_manager):
        n_dims = bounds.shape[0]
        batch_size = self.config.batch_size

        # If no active regions -> global exploration
        if not active_regions:
            return rng.uniform(bounds[:, 0], bounds[:, 1], size=(batch_size, n_dims))

        # Thompson sampling shortcut
        if self.config.acquisition == "ts":
            return self._generate_thompson_batch(bounds, rng, active_regions, surrogate_manager)

        # Default adaptive
        return self._generate_adaptive_batch(bounds, rng, active_regions, surrogate_manager)

    # ============================================
    # Adaptive exploration/exploitation blend
    # ============================================
    def _generate_adaptive_batch(self, bounds, rng, regions, surrogate_manager):
        exploration_prob = self._compute_exploration_probability()
        flags = rng.random(self.config.batch_size) < exploration_prob
        n_explore = int(flags.sum())
        n_exploit = self.config.batch_size - n_explore

        all_candidates = []

        if n_explore > 0:
            all_candidates.extend(
                self._exploration_sampling_batch(bounds, regions, rng, n_explore, surrogate_manager)
            )

        if n_exploit > 0:
            all_candidates.extend(
                self._exploitation_sampling_batch(bounds, regions, rng, n_exploit, surrogate_manager)
            )

        # ✅ Sanitize before stacking
        sanitized = []
        n_dims = bounds.shape[0]
        for arr in all_candidates:
            arr = np.asarray(arr)

            if arr.ndim == 1:
                # single candidate -> make (1,D)
                arr = arr.reshape(1, -1)

            elif arr.ndim > 2:
                # collapse extra dims
                arr = arr.reshape(-1, n_dims)

            # ensure correct D
            if arr.shape[-1] != n_dims:
                arr = arr.reshape(-1, n_dims)

            if arr.size > 0:
                sanitized.append(arr)

        if not sanitized:
            # fallback: uniform random
            return rng.uniform(bounds[:,0], bounds[:,1], size=(self.config.batch_size, n_dims))

        stacked = np.vstack(sanitized)

        # Diversity filtering
        diverse = self._kcenter_greedy(stacked, k=self.config.batch_size)
        rng.shuffle(diverse)
        return diverse


    # ============================================
    # Exploration sampling (hybrid)
    # ============================================
    def _exploration_sampling_batch(self, bounds, regions, rng, count, surrogate_manager):
        n_dims = bounds.shape[0]
        strategies = rng.random(count)

        n_sobol = (strategies < 0.3).sum()
        n_uncertainty = ((strategies >= 0.3) & (strategies < 0.6)).sum()
        n_outward = count - n_sobol - n_uncertainty

        cands = []

        # Sobol low-discrepancy
        if n_sobol > 0:
            sobol = i4_sobol_generate(n_dims, n_sobol)
            sobol_scaled = bounds[:, 0] + sobol * (bounds[:, 1] - bounds[:, 0])
            cands.append(sobol_scaled)

        # Global uncertainty sampling
        if n_uncertainty > 0:
            pool_size = min(500, n_uncertainty * 15)
            pool = rng.uniform(bounds[:, 0], bounds[:, 1], size=(pool_size, n_dims))
            _, std = surrogate_manager.predict_global_cached(pool)
            idx = np.argpartition(std, -n_uncertainty)[-n_uncertainty:]
            cands.append(pool[idx])

        # Outward trust-region sampling
        if n_outward > 0:
            cands.append(self._region_exploration_batch(bounds, regions, rng, n_outward))

        return cands

    def _region_exploration_batch(self, bounds, regions, rng, count):
        n_dims = bounds.shape[0]
        weights = np.array([max(1e-9, r.spawn_score * r.exploration_bonus) for r in regions])
        weights = weights / weights.sum() if weights.sum() > 1e-9 else np.ones_like(weights)/len(weights)

        chosen = rng.choice(len(regions), size=count, p=weights)
        dirs = rng.normal(size=(count, n_dims))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9

        cands = []
        for i, ridx in enumerate(chosen):
            r = regions[ridx]
            radius = r.radius * (1.5 + 0.5*r.exploration_bonus)
            cand = r.center + radius * dirs[i]
            cands.append(np.clip(cand, bounds[:, 0], bounds[:, 1]))
        return np.array(cands)

    # ============================================
    # Exploitation sampling (local refinement)
    # ============================================
    def _exploitation_sampling_batch(self, bounds, regions, rng, count, surrogate_manager):
        health = np.array([max(1e-9, r.health_score) for r in regions])
        health = health/health.sum() if health.sum() > 1e-9 else np.ones_like(health)/len(health)
        choices = rng.choice(len(regions), size=count, p=health)
        counts = Counter(choices)

        cands = []
        for r_idx, n_req in counts.items():
            r = regions[r_idx]

            # fallback if no local data
            if not hasattr(r, "local_X") or len(r.local_X)==0:
                fallback = rng.uniform(bounds[:,0], bounds[:,1], size=(n_req, bounds.shape[0]))
                cands.append(fallback)
                continue

            # local refinement
            for _ in range(n_req):
                cands.append(self.acquisition_manager.optimize_in_region(r, bounds, rng, surrogate_manager))

        return cands

    # ============================================
    # Thompson batch
    # ============================================
    def _generate_thompson_batch(self, bounds, rng, regions, surrogate_manager):
        n_dims = bounds.shape[0]
        per_region = max(1, self.config.n_candidates // len(regions))
        cands = []
        if len(regions)>4:
            with ThreadPoolExecutor(max_workers=min(4, len(regions))) as pool:
                futs = [pool.submit(self._sample_region_ts, r, bounds, rng, per_region) for r in regions]
                for f in as_completed(futs): cands.extend(f.result())
        else:
            for r in regions:
                cands.extend(self._sample_region_ts(r, bounds, rng, per_region))

        # global TS fallback
        global_count = max(1, self.config.n_candidates//4)
        global_samples = rng.uniform(bounds[:,0], bounds[:,1], size=(global_count,n_dims))
        cands.extend(global_samples)
        return np.array(cands).reshape(-1, n_dims)

    def _sample_region_ts(self, region, bounds, rng, count):
        return [
            self.acquisition_manager._sample_from_covariance(region, bounds, rng)
            for _ in range(count)
        ]

    # ============================================
    # Exploration probability (entropy + stagnation)
    # ============================================
    def _compute_exploration_probability(self):
        if self._cache_iteration == self.iteration and self._cached_exploration_prob is not None:
            return self._cached_exploration_prob

        progress = self.iteration / max(1, self.config.max_evals)
        prob = self.config.exploration_factor

        # stagnation boost
        prob += min(0.4, 0.03*self.stagnation_counter)

        # progress decay (more exploitation late)
        prob *= max(0.1, 1.0 - progress)

        self._cached_exploration_prob = np.clip(prob, 0.05, 0.9)
        self._cache_iteration = self.iteration
        return self._cached_exploration_prob

    # ============================================
    # Diversity filtering (k-center greedy)
    # ============================================
    def _kcenter_greedy(self, X, k=32):
        n = len(X)
        if n <= k: return X
        sel = [np.random.randint(n)]
        dmat = cdist(X, X[sel])
        for _ in range(k-1):
            idx = np.argmax(dmat.min(axis=1))
            sel.append(idx)
            dmat = np.minimum(dmat, cdist(X, X[idx:idx+1]))
        return X[sel]


def compute_coverage(X, centers, radii):
    if centers.shape[0] == 0:
        return 0.0
    # Pairwise distances [N_points, N_regions]
    diff = X[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    radius_sq = (2.0 * radii) ** 2
    covered_mask = (dist_sq <= radius_sq[None, :]).any(axis=1)
    return covered_mask.mean()


import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sobol_seq import i4_sobol_generate


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def exp_moving_avg(prev, new, alpha=0.2):
    """Exponential moving average helper."""
    return alpha * new + (1 - alpha) * prev


import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sobol_seq import i4_sobol_generate


class TrustRegion:
    """
    CMA-like TrustRegion for TuRBO-M+++:
    ✅ Ellipsoidal covariance adaptation with PCA smoothing
    ✅ Velocity- and entropy-aware radius control
    ✅ Local surrogate variance injection for uncertainty-aware expansion
    ✅ Fully NaN-safe: no invalid health scores
    """

    def __init__(self, center, radius, region_id, n_dims, dtype=np.float64):
        self.center = np.array(center, dtype=dtype)
        self.radius = float(radius)
        self.region_id = region_id
        self.n_dims = n_dims

        # Performance metrics
        self.best_value = np.inf
        self.prev_best_value = np.inf
        self.trial_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.last_improvement = 0
        self.stagnation_count = 0
        self.restarts = 0

        # Velocity
        self.stagnation_velocity = 0.0
        self.improvement_velocity = 1.0

        # Covariance ellipsoid
        self.cov = np.eye(n_dims) * (radius**2)
        self.pca_basis = np.eye(n_dims)
        self.pca_eigvals = np.ones(n_dims)
        self.cov_updates_since_reset = 0

        # Local archive
        self.local_X = []
        self.local_y = []

        # Signals
        self.local_entropy = 1.0
        self.local_uncertainty = 1.0
        self._health_score_override = None
        self.spawn_score = 0.0
        self.exploration_bonus = 1.0
        self.health_decay_factor = 1.0

    # =====================================
    # Core update per new sample
    # =====================================
    def update(self, x, y, config, surrogate_var=None):
        self.trial_count += 1
        improved = y < self.best_value
        delta = max(0.0, self.best_value - y)

        # Velocity EMA
        self.stagnation_velocity = 0.9 * self.stagnation_velocity + 0.1 * delta
        if improved:
            self.improvement_velocity = 0.8 * self.improvement_velocity + 0.2 * delta
        else:
            self.improvement_velocity *= 0.97

        # Track surrogate local uncertainty
        if surrogate_var is not None and np.isfinite(surrogate_var):
            self.local_uncertainty = (
                0.9 * self.local_uncertainty + 0.1 * abs(surrogate_var)
            )

        # Best value bookkeeping
        if improved:
            self.prev_best_value = self.best_value
            self.best_value = y
            self.success_count += 1
            self.last_improvement = 0
            self.stagnation_count = 0
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.last_improvement += 1
            if self.consecutive_failures > 4:
                self.stagnation_count += 1

        # Local archive & covariance update
        self._update_local_archive(x, y, config.max_local_data)
        self._rank_one_cov_update(x)

        # Update PCA & entropy occasionally
        if self.trial_count % max(5, self.n_dims) == 0:
            self._update_pca_from_cov()
            self.local_entropy = self._compute_entropy()

        # Adaptive ellipsoid radius
        if getattr(config, "local_radius_adaptation", True):
            self._adaptive_radius(improved, delta, config)

        # Spawn score combines age, entropy, velocity
        self._update_spawn_score(config)

    # =====================================
    # Safe rank-one covariance update
    # =====================================
    def _rank_one_cov_update(self, x, alpha=0.1):
        dx = (x - self.center).reshape(-1, 1)
        if np.all(np.isfinite(dx)):
            self.cov = (1 - alpha) * self.cov + alpha * (dx @ dx.T)
        # Numerical stabilization
        self.cov += 1e-8 * np.eye(self.n_dims)

    def _update_pca_from_cov(self):
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        eigvals = np.clip(eigvals, 1e-12, 1e6)
        idx = np.argsort(eigvals)[::-1]
        self.pca_eigvals = eigvals[idx]
        self.pca_basis = eigvecs[:, idx]

    # =====================================
    # Adaptive radius update
    # =====================================
    def _adaptive_radius(self, improved, delta, config):
        if improved:
            scale = config.expansion_factor
        else:
            stagnation_penalty = np.exp(-0.5 * self.stagnation_velocity)
            scale = config.contraction_factor * stagnation_penalty

        # Inject uncertainty bonus
        unc_bonus = 1.0 + 0.3 * np.tanh(self.local_uncertainty)
        scale *= unc_bonus

        # Expand/shrink along PCA ellipsoid
        self.cov *= scale**2
        # Update isotropic radius as mean PCA axis length
        mean_axis = np.mean(np.sqrt(self.pca_eigvals))
        self.radius = np.clip(mean_axis, config.min_radius, config.max_radius)

    # =====================================
    # Local archive & entropy
    # =====================================
    def _update_local_archive(self, x, y, max_size):
        if len(self.local_X) < max_size:
            self.local_X.append(x.copy())
            self.local_y.append(y)
        else:
            idx = np.random.randint(0, self.trial_count + 1)
            if idx < max_size:
                self.local_X[idx] = x.copy()
                self.local_y[idx] = y

    def _compute_entropy(self):
        # safe logdet of PCA eigenvalues
        logdet = np.sum(np.log(np.clip(self.pca_eigvals, 1e-12, 1e6)))
        ref = self.n_dims * np.log(self.radius + 1e-12)
        val = 0.5 * (logdet - ref)
        return float(np.nan_to_num(val, nan=1.0, posinf=1.0, neginf=1.0))

    # =====================================
    # Spawn score
    # =====================================
    def _update_spawn_score(self, config):
        age_penalty = min(1.0, self.last_improvement / max(1, config.max_age))
        vel_term = np.tanh(self.stagnation_velocity)
        entropy_term = np.tanh(self.local_entropy)
        uncertainty_term = np.tanh(self.local_uncertainty)

        val = (
            0.3 * self.success_rate
            + 0.3 * vel_term
            + 0.2 * entropy_term
            + 0.2 * uncertainty_term
            - 0.3 * age_penalty
        )
        self.spawn_score = np.clip(np.nan_to_num(val, nan=0.0), 0.0, 1.0)
        self.exploration_bonus = np.clip(
            1.0 + 0.5 * (1 - np.tanh(self.improvement_velocity)), 0.8, 2.0
        )

    # =====================================
    # Properties
    # =====================================
    @property
    def success_rate(self):
        if self.trial_count <= 0:
            return 0.0
        return self.success_count / max(1, self.trial_count)

    @property
    def is_active(self):
        return self.radius > 1e-8

    @property
    def should_restart(self):
        entropy_low = self.local_entropy < 0.05
        return (self.stagnation_count > 15 or entropy_low) and self.radius < 0.05

    @property
    def health_score(self):
        if self._health_score_override is not None:
            return self._health_score_override
        raw = (
            0.4 * self.success_rate
            + 0.3 * (1.0 - self.local_entropy)
            + 0.3 * np.tanh(self.local_uncertainty)
        )
        return float(
            np.clip(np.nan_to_num(raw * self.health_decay_factor, nan=0.0), 0.0, 1.0)
        )

    @health_score.setter
    def health_score(self, value):
        self._health_score_override = float(
            np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
        )

    def clear_health_override(self):
        self._health_score_override = None

    def decay_health(self, factor=0.98):
        self.health_decay_factor *= factor


import collections

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


def np_safe(x, default=0.0):
    """Replace NaNs/Infs with finite default."""
    return np.nan_to_num(x, nan=default, posinf=1.0, neginf=0.0)

def exp_moving_avg(prev, new_val, alpha=0.1):
    return (1 - alpha) * prev + alpha * new_val

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_coverage_fraction(global_X, regions):
    """Fraction of points covered by at least one region (approx)."""
    if len(global_X) == 0 or len(regions) == 0:
        return 0.0
    centers = np.array([r.center for r in regions])
    radii = np.array([r.radius for r in regions])
    dists = cdist(global_X, centers)
    covered = np.any(dists <= radii[None, :], axis=1)
    return np.sum(covered) / len(global_X)

def compute_mean_entropy(surrogate_manager, global_X):
    """Mean predictive variance as a proxy for entropy (if available)."""
    if surrogate_manager is None or len(global_X) == 0:
        return np.array([1.0])
    _, var = surrogate_manager.predict(global_X, return_var=True)
    return np.mean(var)


class RegionManager:
    """
    Next-gen RegionManager for TuRBO-M+++ (Entropy & Pareto-Aware)
    ✅ Ellipsoidal trust regions with PCA covariance adaptation
    ✅ Pareto-balanced spawn scoring (EI, UCB, GradNorm, Diversity)
    ✅ Dynamic region budget allocation (exploration vs exploitation)
    ✅ Adaptive pruning using health+diversity Pareto filter
    ✅ Entropy-aware spawn triggers with coverage feedback
    ✅ NaN-safe: never produces invalid weights
    """

    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose
        self.regions = []
        self.surrogate_manager = None

        # Adaptive global state
        self._entropy_buffer = collections.deque(maxlen=8)
        self._ema_progress = 0.0
        self._iteration = 0

    # ---------------------------------------------
    # Compatibility
    # ---------------------------------------------
    def set_surrogate_manager(self, surrogate_manager):
        self.surrogate_manager = surrogate_manager

    # ---------------------------------------------
    # Initialization
    # ---------------------------------------------
    def initialize_regions(self, X, y, n_dims, rng=None):
        n_init = min(self.config.n_regions, len(X))
        best_idx = int(np.argmin(y))
        selected_idx = [best_idx]

        # performance + diversity greedy selection
        D = cdist(X, X)
        perf_weights = np.exp(-0.05 * (y - y.min()))
        for _ in range(1, n_init):
            min_d = np.min(D[:, selected_idx], axis=1)
            scores = perf_weights * (min_d + 1e-8)
            scores[selected_idx] = -np.inf
            selected_idx.append(np.argmax(scores))

        for rid, center in enumerate(X[selected_idx]):
            base_r = np.percentile(np.linalg.norm(X - center, axis=1), 25)
            base_r = np.clip(base_r, self.config.min_radius, self.config.init_radius)
            self.regions.append(TrustRegion(center, base_r, rid, n_dims))

        if self.verbose:
            print(f"[INIT] {len(self.regions)} trust regions created")

    # ---------------------------------------------
    # Main lifecycle
    # ---------------------------------------------
    def manage_regions(self, bounds, n_dims, rng, global_X, global_y, iteration=0):
        self._iteration = iteration
        self._ema_progress = exp_moving_avg(
            self._ema_progress, iteration / (self.config.max_evals + 1e-9), alpha=0.1
        )

        self._adapt_all(bounds, rng)
        self._maybe_spawn(bounds, n_dims, rng, global_X, global_y)
        self._ensure_min_diversity(bounds, n_dims, rng, global_X, global_y)
        self._adaptive_prune()

    # ---------------------------------------------
    # Adaptive radius + ellipsoidal covariance
    # ---------------------------------------------
    def _adapt_all(self, bounds, rng):
        dead_regions = []
        for r in self.regions:
            # smooth improvement velocity
            r.vel_ema = exp_moving_avg(
                getattr(r, "vel_ema", 0.0),
                np_safe(r.improvement_velocity),
                alpha=0.3,
            )

            # PCA covariance adaptation (CMA-like)
            if hasattr(r, "local_X") and len(r.local_X) > max(10, 2 * r.n_dims):
                centered = np.array(r.local_X) - r.center
                cov = np.cov(centered.T) + 1e-6 * np.eye(r.n_dims)
                if np.all(np.isfinite(cov)):
                    r.cov = 0.7 * r.cov + 0.3 * cov  # low-pass filter

            # adaptive radius
            old_r = r.radius
            if r.vel_ema < 0.01:
                r.radius = min(r.radius * 1.08, self.config.max_radius)
            elif r.vel_ema > 0.05:
                r.radius = max(r.radius * 0.88, self.config.min_radius)
            else:
                r.radius *= 0.99

            # decay health: improvement + entropy + diversity
            diversity = self._region_diversity_score(r)
            entropy_local = getattr(r, "local_entropy", 0.5)
            raw_h = (
                0.5 * np_safe(r.vel_ema)
                + 0.3 * (1 - np_safe(entropy_local))
                + 0.2 * diversity
            )
            r.health_score = np.clip(np_safe(raw_h), 0.0, 1.0)
            r.decay_health(0.99)

            if self.verbose and abs(old_r - r.radius) > 1e-3:
                print(f"[ADAPT] R#{r.region_id} radius {old_r:.3f}→{r.radius:.3f}")

            if r.should_restart:
                self._restart_region(r, bounds, rng)

            if r.radius < 1.3 * self.config.min_radius and r.health_score < 0.2:
                dead_regions.append(r)

        # replace dead
        for r in dead_regions:
            self._replace_dead_region(r, bounds, rng)

    def _region_diversity_score(self, region):
        if len(self.regions) <= 1:
            return 1.0
        others = np.array([r.center for r in self.regions if r is not region])
        d = np.min(np.linalg.norm(others - region.center, axis=1))
        return np_safe(d / (region.radius + 1e-9))

    # ---------------------------------------------
    # Spawning new regions
    # ---------------------------------------------
    def _maybe_spawn(self, bounds, n_dims, rng, global_X, global_y):
        coverage = compute_coverage_fraction(global_X, self.regions)
        entropy = compute_mean_entropy_from_global_gp(self.surrogate_manager, global_X)
        self._entropy_buffer.append(np_safe(entropy))
        smoothed_entropy = np.mean(self._entropy_buffer)

        avg_health = (
            np.mean([np_safe(r.health_score) for r in self.regions])
            if self.regions else 1.0
        )
        exploration_pressure = (1 - coverage) + smoothed_entropy
        exploitation_pressure = self._ema_progress + avg_health

        trigger = sigmoid(3 * (exploration_pressure - exploitation_pressure)) > 0.45
        dynamic_cap = int(self.config.n_regions * (1.0 + 0.5 * exploration_pressure))

        if trigger and len(self.regions) < dynamic_cap:
            self._force_spawn(bounds, n_dims, rng, global_X, global_y)

    def _compute_spawn_radius(self, candidate, existing, global_X):
        """
        Compute a good initial radius for a new region.
        Uses distances to existing regions and global_X density.
        """
        # No existing regions → default init radius
        if existing is None or existing.size == 0:
            return self.config.init_radius

        # Distance to nearest existing region
        dists = np.linalg.norm(existing - candidate, axis=1)
        nearest_center_dist = np.median(dists) if len(dists) > 0 else self.config.init_radius

        # Also check density of global_X (optional)
        if global_X is not None and len(global_X) > 0:
            gx_dists = np.linalg.norm(global_X - candidate, axis=1)
            local_density = np.percentile(gx_dists, 30)  # 30%-quantile distance
        else:
            local_density = nearest_center_dist

        # Take a conservative min of density & spacing
        r = min(nearest_center_dist, local_density)
        return np.clip(r, self.config.min_radius, self.config.init_radius)

    def _estimate_grad_norms_batch(self, X, eps=1e-3):
        # If SurrogateManager supports gradients → use them
        if (
            self.surrogate_manager is not None
            and self.surrogate_manager.global_backend == "exact_gp"
            and self.surrogate_manager.global_model is not None
        ):
            grad = self.surrogate_manager.gradient_global_mean(X)
            grad_norm = np.linalg.norm(grad, axis=1)
            # Also weight by predictive uncertainty for exploration
            _, std = self.surrogate_manager.predict_global_cached(X)
            return np.nan_to_num(grad_norm * (std + 1e-6))

        # Otherwise fallback to predictive std
        return self._fallback_grad_bonus(X, eps)

    def _fallback_grad_bonus(self, X, eps=1e-3):
        # fallback = variance-only bonus
        if self.surrogate_manager is None:
            return np.ones(len(X))
        _, std = self.surrogate_manager.predict_global_cached(X)
        return np.nan_to_num(std)


    def _find_best_spawn(self, bounds, n_dims, rng, global_X, global_y):
        sobol = i4_sobol_generate(n_dims, 512)
        candidates = bounds[:, 0] + sobol * (bounds[:, 1] - bounds[:, 0])

        # ✅ Use global GP predictions
        mean, std = self.surrogate_manager.predict_global_cached(candidates)

        f_best = np.min(global_y) if len(global_y) else np.min(mean)
        z = (f_best - mean) / (std + 1e-9)
        ei = (f_best - mean) * norm.cdf(z) + std * norm.pdf(z)
        ucb = mean - 2 * std
        grad_bonus = self._estimate_grad_norms_batch(candidates, eps=2e-3)
        diversity = self._diversity_bonus(candidates)

        # normalize safely
        def safe_norm(v):
            return np_safe(v) / (np.max(np.abs(np_safe(v))) + 1e-9)
        ei, grad_bonus, ucb, diversity = map(
            safe_norm, [ei, grad_bonus, -ucb, diversity]
        )

        alpha = sigmoid(4 * (self._ema_progress - 0.5))
        score = (1 - alpha) * ei + 0.3 * grad_bonus + 0.2 * diversity + 0.2 * ucb
        return candidates[np.argmax(score)]
    
    def _force_spawn(self, bounds, n_dims, rng, global_X, global_y):
        cand = self._find_best_spawn(bounds, n_dims, rng, global_X, global_y)

        # diversity enforcement
        existing = (
            np.array([r.center for r in self.regions])
            if self.regions
            else np.empty((0, n_dims))
        )
        if existing.size > 0:
            min_dist = np.min(np.linalg.norm(existing - cand, axis=1))
            if min_dist < 0.3 * self.config.init_radius:
                cand = self._maximin_diverse(bounds, n_dims, rng, existing)
                if self.verbose:
                    print("[SPAWN] Adjusted candidate for diversity")

        radius = self._compute_spawn_radius(cand, existing, global_X)
        new_r = TrustRegion(cand, radius, len(self.regions), n_dims)
        new_r.exploration_bonus = 1.4
        self.regions.append(new_r)
        if self.verbose:
            print(f"[SPAWN] Region#{new_r.region_id} @ {np.round(cand, 3)}")


    def _diversity_bonus(self, candidates):
        if not self.regions:
            return np.ones(len(candidates))
        existing = np.array([r.center for r in self.regions])
        dists = cdist(candidates, existing).min(axis=1)
        return np_safe(dists) / (np.max(np_safe(dists)) + 1e-9)

    def _maximin_diverse(self, bounds, n_dims, rng, existing):
        samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(128, n_dims))
        min_dists = cdist(samples, existing).min(axis=1)
        return samples[np.argmax(min_dists)]

    # ---------------------------------------------
    # Adaptive pruning
    # ---------------------------------------------
    def _adaptive_prune(self):
        if not self.regions:
            return
        if getattr(self.config, "max_total_points", None):
            total_points = sum(self._len_local(r) for r in self.regions)
            if total_points > self.config.max_total_points:
                pareto_score = [
                    0.6 * np_safe(r.health_score)
                    + 0.4 * self._region_diversity_score(r)
                    for r in self.regions
                ]
                sorted_r = [r for _, r in sorted(zip(pareto_score, self.regions))]
                remove_n = max(1, int(0.2 * len(self.regions)))
                for r in sorted_r[:remove_n]:
                    self.regions.remove(r)
                    if self.verbose:
                        print(f"[PRUNE] Region#{r.region_id} pruned")

    def _len_local(self, region):
        return len(region.local_X) if getattr(region, "local_X", None) is not None else 0

    # ---------------------------------------------
    # Diversity enforcement
    # ---------------------------------------------
    def _ensure_min_diversity(self, bounds, n_dims, rng, global_X, global_y):
        """
        Guarantee at least a minimal diversity of regions exists.
        If too few regions remain, force-spawn new ones.
        """
        min_needed = max(3, int(self.config.n_regions * 0.5))
        while len(self.regions) < min_needed:
            self._force_spawn(bounds, n_dims, rng, global_X, global_y)
            if self.verbose:
                print(
                    f"[DIVERSITY] Forced spawn → {len(self.regions)}/{min_needed}"
                )

    # ---------------------------------------------
    # Safe weights for CandidateGenerator exploitation
    # ---------------------------------------------
    def safe_region_weights(self, regions):
        health = np.array(
            [np_safe(r.health_score) for r in regions]
        )
        total_h = np.sum(health)
        if total_h < 1e-12:
            return np.ones(len(regions)) / len(regions)
        return health / total_h

    # ---------------------------------------------
    # Region update assignment (unchanged but NaN-safe)
    # ---------------------------------------------
    def update_regions_with_new_data(self, X_new, y_new):
        if not self.regions:
            return
        active = [r for r in self.regions if r.is_active]
        if not active:
            return

        n_dims = active[0].n_dims
        centers = np.stack([r.center for r in active])
        cov_invs = np.stack(
            [
                np.linalg.inv(r.cov + 1e-9 * np.eye(n_dims))
                if np.all(np.isfinite(r.cov))
                else np.eye(n_dims) / (r.radius**2 + 1e-9)
                for r in active
            ]
        )

        diffs = X_new[:, None, :] - centers[None, :, :]
        tmp = np.einsum("nrd,rdk->nrk", diffs, cov_invs)
        mahal_sq = np.einsum("nrd,nrd->nr", tmp, diffs)

        avg_radius = np.mean([r.radius for r in active])
        beta = max(
            3.0, 10.0 * (self.config.init_radius / (avg_radius + 1e-9))
        )

        weights = np.exp(-beta * mahal_sq)
        weights /= np.sum(weights, axis=1, keepdims=True)

        for i, (x, y) in enumerate(zip(X_new, y_new)):
            rid = np.argmax(weights[i])
            if weights[i, rid] > 0.05:
                active[rid].update(x, float(y), self.config)


class Foretuner:
    """Enhanced TURBO-M++ optimizer with modular architecture (safe incremental improvements)."""

    def __init__(self, config: TurboConfig = None):
        self.config = config or TurboConfig()
        self.surrogate_manager = SurrogateManager(self.config)
        self.region_manager = RegionManager(self.config)
        self.acquisition_manager = AcquisitionManager(self.config)
        self.candidate_generator = CandidateGenerator(
            self.config, self.acquisition_manager
        )

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

    def optimize(
        self, objective_fn: Callable, bounds: np.ndarray, seed: int = 0
    ) -> Tuple[np.ndarray, float]:
        """Main optimization loop with cleaner separation of concerns"""
        n_dims = bounds.shape[0]
        rng = np.random.default_rng(seed)

        # === Initialization ===
        self._initialize_optimization(objective_fn, bounds, rng, n_dims)

        # === Main optimization loop ===
        for self.iteration in range(
            self.config.n_init, self.config.max_evals, self.config.batch_size
        ):
            self._update_context()

            # Manage regions periodically
            if self.iteration % self.config.management_frequency == 0:
                self.region_manager.manage_regions(
                    bounds, n_dims, rng, self.global_X, self.global_y
                )

            # Candidate generation
            active_regions = [r for r in self.regions if r.is_active]
            candidates = self.candidate_generator.generate_candidates(
                bounds, rng, active_regions, self.surrogate_manager
            )

            if candidates is None or len(candidates) == 0:
                # Safety fallback: sample random points
                candidates = rng.uniform(
                    bounds[:, 0], bounds[:, 1], size=(self.config.batch_size, n_dims)
                )

            # Evaluate objective on generated candidates
            y_new = np.array([objective_fn(x) for x in candidates])

            # Update global + surrogate + regions
            self._update_global_data(candidates, y_new)

            # Track progress
            self._track_progress()

            # Print occasional progress
            if self.iteration % 10 == 0:
                self._print_progress()

        return self._get_best_solution()

    # === Initialization ===
    def _initialize_optimization(self, objective_fn, bounds, rng, n_dims):
        """Initialize optimization state"""
        X_init = self._initialize_points(n_dims, bounds, rng)
        y_init = np.array([objective_fn(x) for x in X_init])

        self.global_X = X_init
        self.global_y = y_init

        # Update managers with initial data
        self.surrogate_manager.update_data(self.global_X, self.global_y)
        self.region_manager.initialize_regions(X_init, y_init, n_dims, rng)

        best_y = float(np.min(self.global_y))
        self.global_best_history.append(best_y)
        print(f"Initial best: {best_y:.6f}")

    def _initialize_points(self, n_dims, bounds, rng):
        """Initialize points using Sobol + random sampling (avoids duplicates)"""
        n_init = self.config.n_init

        # Sobol sequence for half
        sobol_part = sobol_sequence(rng.integers(0, 10000), n_init, n_dims)

        # Random extra points for diversity
        rand_extra = rng.uniform(0, 1, (n_init // 4, n_dims))

        # Combine and drop potential duplicates
        all_samples = np.vstack([sobol_part, rand_extra])
        all_samples = np.unique(all_samples, axis=0)

        low, high = bounds[:, 0], bounds[:, 1]
        return low + all_samples * (high - low)

    # === Iteration Context ===
    def _update_context(self):
        """Update context for all managers"""
        self.candidate_generator.set_context(self.iteration, self.stagnation_counter)

    # === Global Data Updates ===
    def _update_global_data(self, candidates, y_new):
        """Update global data and regions"""
        # Append new data
        self.global_X = np.vstack([self.global_X, candidates])
        self.global_y = np.append(self.global_y, y_new)

        # Retrain surrogate model with all data
        self.surrogate_manager.update_data(self.global_X, self.global_y)

        # Update trust regions
        self.region_manager.update_regions_with_new_data(candidates, y_new)

    # === Progress Tracking ===
    def _track_progress(self):
        """Track optimization progress & stagnation counter"""
        current_best_y = np.min(self.global_y)

        if (
            len(self.global_best_history) == 0
            or current_best_y < self.global_best_history[-1] - 1e-6
        ):
            # New improvement found
            self.last_global_improvement = 0
            self.stagnation_counter = 0
        else:
            # No improvement
            self.last_global_improvement += 1
            if self.last_global_improvement > 3:
                self.stagnation_counter = min(
                    self.stagnation_counter + 1, 1000
                )  # cap growth

        self.global_best_history.append(float(current_best_y))

    # === Logging ===
    def _print_progress(self):
        """Print optimization progress safely (handles empty regions)"""
        best_y = np.min(self.global_y)

        active_regions = sum(1 for r in self.regions if r.is_active)
        if self.regions:
            avg_health = float(np.mean([r.health_score for r in self.regions]))
            avg_radius = float(np.mean([r.radius for r in self.regions]))
        else:
            avg_health = 0.0
            avg_radius = 0.0

        print(
            f"Trial {self.iteration:4d}: Best = {best_y:.6f}, "
            f"Active regions = {active_regions}, Avg health = {avg_health:.3f}, "
            f"Avg radius = {avg_radius:.4f}, Stagnation = {self.stagnation_counter}"
        )

    # === Best Solution ===
    def _get_best_solution(self):
        """Return best solution found so far"""
        best_idx = np.argmin(self.global_y)
        return self.global_X[best_idx], self.global_y[best_idx]

    def get_trials(self) -> List[Trial]:
        """Get all trials as Trial objects for compatibility with plotting"""
        trials = []
        for i in range(len(self.global_X)):
            params = {
                f"x{j}": self.global_X[i, j] for j in range(self.global_X.shape[1])
            }
            trial = Trial(params=params, value=self.global_y[i], is_feasible=True)
            trials.append(trial)
        return trials


def plot_foretuner_results(
    optimizer,
    bounds: np.ndarray,
    param_names: List[str] = None,
    title: str = "Foretuner Optimization Results",
):
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
                ax.hist(
                    feasible_values,
                    bins="auto",
                    edgecolor="black",
                    alpha=0.7,
                    color="skyblue",
                )
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
            df = pd.DataFrame(
                [dict(**t.params, value=t.value) for t in feasible_trials]
            )
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
