# ============================================
# ✅ Core Python & Concurrency
# ============================================
from functools import lru_cache

# ============================================
# ✅ GPyTorch & BoTorch (Gaussian Processes)
# ============================================
import gpytorch

# ============================================
# ✅ Numerical & Scientific Computing
# ============================================
import numpy as np

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
from torch.autograd import grad


# ============================================
# ✅ Bayesian NN (MC Dropout)
# ============================================
class BayesianNN(nn.Module):
    """
    Bayesian Neural Network surrogate using MC Dropout for epistemic uncertainty.
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

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim) if use_layernorm else None

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )
        self.hidden_ln = (
            nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_hidden_layers)])
            if use_layernorm
            else None
        )

        self.fc_out = nn.Linear(hidden_dim, 1)

    def _get_activation(self, name: str):
        if name == "relu":
            return F.relu
        if name == "silu":
            return F.silu
        if name == "tanh":
            return torch.tanh
        raise ValueError(f"Unknown activation: {name}")

    def _mc_dropout(self, x):
        # Always active at inference for MC sampling
        return F.dropout(x, p=self.dropout_p, training=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.input_layer(x))
        if self.input_ln is not None:
            x = self.input_ln(x)
        x = self._mc_dropout(x)

        for i, layer in enumerate(self.hidden_layers):
            h = self.activation_fn(layer(x))
            if self.hidden_ln is not None:
                h = self.hidden_ln[i](h)
            h = self._mc_dropout(h)
            x = x + 0.5 * h  # mild residual

        return self.fc_out(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int = 20):
        preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)  # (S,N,1)
        mean, var = torch.var_mean(preds, dim=0, unbiased=False)
        std = torch.sqrt(var + 1e-9)
        return mean.squeeze(-1), std.squeeze(-1)


# ============================================
# ✅ Sparse GP (Variational)
# ============================================
class SparseGPModel(SingleTaskVariationalGP):
    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor, inducing_points: torch.Tensor):
        train_X = train_X.double()
        train_Y = train_Y.double()
        inducing_points = inducing_points.double()
        m = inducing_points.size(0)
        variational_distribution = CholeskyVariationalDistribution(m)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_points=True,
            outcome_transform=Standardize(m=1),
        )


# ============================================
# ✅ Exact GP (robust init)
# ============================================
class RobustExactGP(ExactGP):
    """Exact GP with ARD Matern(ν=2.5) and data-informed init."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]))

        if train_x.shape[0] > 1:
            with torch.no_grad():
                dists = torch.cdist(train_x, train_x)
                has_pos = torch.any(dists > 0)
                med = torch.median(dists[dists > 0]) if has_pos else torch.tensor(1.0, dtype=train_x.dtype, device=train_x.device)
                self.covar_module.base_kernel.lengthscale = med.clamp_min(1e-3)

        y_std = train_y.std().clamp_min(1e-3)
        self.covar_module.outputscale = (y_std ** 2).clamp_max(1e6)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def fit_exact_gp_model(model, likelihood, max_iter=75, patience=10):
    """Robust-ish GP training with Adam + early stopping."""
    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    best = float("inf")
    bad = 0
    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        out = model(model.train_inputs[0])
        loss = -mll(out, model.train_targets)
        loss.backward()
        optimizer.step()

        cur = loss.item()
        if cur < best - 1e-4:
            best = cur
            bad = 0
        else:
            bad += 1
        if bad > patience:
            break

    model.eval()
    likelihood.eval()
    return model


# ============================================
# ✅ Cached key helper
# ============================================
def _candidates_key(candidates: np.ndarray) -> tuple[int, bytes]:
    """Hashable key: (n_dim, bytes of 6-decimal rounded double array)."""
    return candidates.shape[1], np.round(candidates, 6).astype(np.float64).tobytes()


# ============================================
# ✅ Surrogate Manager (with gradient fallback)
# ============================================
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
        use_torch_compile=False,
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        # Backends: "exact_gp", "sparse_gp", "bnn"
        self.global_backend = global_backend
        self.local_backend = local_backend

        self.normalize_inputs = normalize_inputs
        self.n_inducing = n_inducing
        self.bnn_hidden = bnn_hidden
        self.bnn_dropout = bnn_dropout
        self.use_torch_compile = use_torch_compile and hasattr(torch, "compile")

        # Global data & model
        self.global_X = None  # torch.DoubleTensor [N,D]
        self.global_y = None  # torch.DoubleTensor [N,1]
        self.global_model = None

        # Local cache
        self.local_model_cache = {}  # key -> model

        # Versioning for cache invalidation
        self._model_version = 0

        # Cached posterior
        self._cached_posterior = self._make_cached_predict_fn()

    # ----------------------
    # Utilities
    # ----------------------
    def _to_tensor(self, X):
        return torch.as_tensor(X, dtype=torch.double, device=self.device)

    def _normalize_inputs(self, X, ref_X):
        if not self.normalize_inputs:
            return X
        if ref_X is None or ref_X.shape[0] < 2:
            return X
        # min-max per dim
        min_vals = ref_X.min(0).values
        max_vals = ref_X.max(0).values
        ranges = (max_vals - min_vals).clamp_min(1e-12)
        return (X - min_vals) / (ranges + 1e-9)

    # ----------------------
    # Finite differences (fallback)
    # ----------------------
    def _finite_diff(self, X_np: np.ndarray, f, eps_rel: float = 1e-4) -> np.ndarray:
        """
        Central finite differences for a vectorized scalar function f(X)->[N].
        Step size is relative to data scale (per-dimension).
        """
        X = np.asarray(X_np, dtype=float)
        n, d = X.shape
        grads = np.zeros((n, d), dtype=float)

        # scale-aware step
        if self.global_X is not None:
            scale = self.global_X.detach().cpu().numpy().astype(float)
            scale = np.std(scale, axis=0)
        else:
            scale = np.ones(d, dtype=float)
        h = eps_rel * (np.abs(scale) + 1e-6)

        for j in range(d):
            Xp = X.copy(); Xp[:, j] += h[j]
            Xm = X.copy(); Xm[:, j] -= h[j]
            fp = f(Xp)  # [N]
            fm = f(Xm)  # [N]
            grads[:, j] = (fp - fm) / (2.0 * h[j])
        return grads

    # ======================
    # Grad of global mean (tries autograd, falls back to FD)
    # ======================
    def gradient_global_mean(self, X_query: np.ndarray):
        """
        ∇_x μ_f(x) using latent GP; falls back to finite differences if autograd fails.
        """
        if self.global_model is None or self.global_backend != "exact_gp":
            return np.zeros_like(X_query, dtype=float)

        # try autograd path first
        try:
            Xq = self._to_tensor(X_query).clone().detach().requires_grad_(True)
            Xq_norm = self._normalize_inputs(Xq, self.global_X)

            self.global_model.eval()
            with gpytorch.settings.fast_pred_var(False), \
                 gpytorch.settings.detach_test_caches(False), \
                 torch.enable_grad():
                post = self.global_model(Xq_norm)  # latent posterior
                post.mean.sum().backward()

            return Xq.grad.detach().cpu().numpy().astype(float)

        except RuntimeError:
            # fallback: finite differences on predictive mean
            def f_mean(Xnp):
                m, _ = self.predict_global_cached(np.asarray(Xnp, dtype=float))
                return np.asarray(m, dtype=float)

            return self._finite_diff(np.asarray(X_query, dtype=float), f_mean, eps_rel=1e-4)

    # ======================
    # EI and grad (tries autograd, falls back to FD)
    # ======================
    def ei_and_grad(self, X_np, f_best):
        """
        EI(x) and ∇EI(x). Uses latent GP + autograd, falls back to finite differences
        on the scalar EI if autograd is not available.
        """
        X_np = np.asarray(X_np, dtype=float)
        if self.global_model is None or self.global_backend != "exact_gp":
            return np.zeros(X_np.shape[0], dtype=float), np.zeros_like(X_np, dtype=float)

        # try autograd path
        try:
            X_t = torch.tensor(X_np, dtype=torch.double, device=self.device, requires_grad=True)
            X_norm = self._normalize_inputs(X_t, self.global_X)

            self.global_model.eval()
            with gpytorch.settings.fast_pred_var(False), \
                 gpytorch.settings.detach_test_caches(False), \
                 torch.set_grad_enabled(True):
                post = self.global_model(X_norm)          # latent posterior
                mean_t = post.mean
                std_t  = post.variance.sqrt().clamp_min(1e-12)

                z   = (f_best - mean_t) / std_t
                phi = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
                Phi = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
                ei_t = (f_best - mean_t) * Phi + std_t * phi

            ei_val = ei_t.detach().cpu().numpy().astype(float)
            grads = []
            for i in range(ei_t.shape[0]):
                g = grad(ei_t[i], X_t, retain_graph=(i < ei_t.shape[0] - 1))[0]
                grads.append(g.detach().cpu().numpy())
            return ei_val, np.stack(grads, axis=0).astype(float)

        except RuntimeError:
            # fallback: finite differences on EI scalar
            from math import erf as _erf_scalar

            def _erf_np(x):
                x = np.asarray(x, dtype=float)
                # vectorize math.erf
                v = np.vectorize(_erf_scalar)
                return v(x)

            def f_ei(Xnp):
                m, s = self.predict_global_cached(np.asarray(Xnp, dtype=float))
                s = np.maximum(s, 1e-12)
                z = (f_best - m) / s
                phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
                Phi = 0.5 * (1.0 + _erf_np(z / np.sqrt(2.0)))
                return (f_best - m) * Phi + s * phi

            ei_val = f_ei(X_np).astype(float)
            ei_grad = self._finite_diff(X_np, f_ei, eps_rel=1e-4).astype(float)
            return ei_val, ei_grad

    # ======================
    # Posterior Cache
    # ======================
    def _make_cached_predict_fn(self):
        @lru_cache(maxsize=128)
        def _cached_predict(n_dim: int, candidates_bytes: bytes, backend_name: str, model_version: int):
            candidates = np.frombuffer(candidates_bytes, dtype=np.float64).reshape(-1, n_dim)
            mean, std = self._predict_from_model(self.global_model, self._to_tensor(candidates), self.global_X, backend=backend_name)
            return mean, std
        return _cached_predict

    def clear_posterior_cache(self):
        self._cached_posterior.cache_clear()

    # ======================
    # Global Data Updates
    # ======================
    def update_global_data(self, X, y):
        self.global_X = self._to_tensor(X)
        self.global_y = self._to_tensor(y).unsqueeze(-1)

        if self.global_X.shape[0] >= 5:
            self.global_model = self._fit_model(self.global_X, self.global_y, backend=self.global_backend)
            if self.use_torch_compile and hasattr(torch, "compile"):
                try:
                    self.global_model = torch.compile(self.global_model)
                except Exception:
                    pass
            self._model_version += 1
            self.clear_posterior_cache()

    def update_data(self, X, y):
        return self.update_global_data(X, y)

    # ======================
    # Global Predict
    # ======================
    def predict_global_cached(self, candidates: np.ndarray):
        if self.global_model is None:
            return np.zeros(len(candidates)), np.ones(len(candidates))
        n_dim, key_bytes = _candidates_key(candidates)
        return self._cached_posterior(n_dim, key_bytes, self.global_backend, self._model_version)

    def predict_global(self, X_test):
        if self.global_model is None or self.global_X is None:
            mean = torch.mean(self.global_y).item() if self.global_y is not None else 0.0
            return np.full(X_test.shape[0], mean), np.ones(X_test.shape[0])
        return self._predict_from_model(self.global_model, self._to_tensor(X_test), self.global_X, backend=self.global_backend)

    # ======================
    # Local Predict
    # ======================
    def _region_cache_key(self, local_X, region_key):
        if region_key is not None:
            try:
                hash(region_key)
                return ("rk", region_key)
            except Exception:
                pass
        n = len(local_X) if local_X is not None else 0
        center = np.mean(np.asarray(local_X), axis=0) if n > 0 else np.zeros(1)
        return ("auto", n, tuple(np.round(np.asarray(center, dtype=float), 6)))

    def predict_local(self, X_test, local_X, local_y, region_key=None):
        if local_X is None or len(local_y) < 3:
            return self.predict_global_cached(X_test)

        key = self._region_cache_key(local_X, region_key)
        if key in self.local_model_cache:
            model = self.local_model_cache[key]
        else:
            model = self._fit_model(self._to_tensor(local_X), self._to_tensor(local_y).unsqueeze(-1), backend=self.local_backend)
            self.local_model_cache[key] = model

        return self._predict_from_model(model, self._to_tensor(X_test), self._to_tensor(local_X), backend=self.local_backend)

    def predict_local_with_posterior(self, X_test, local_X, local_y, seed=0, n_samples=5, region_key=None):
        torch.manual_seed(seed)
        if local_X is None or len(local_y) < 3:
            mean, std = self.predict_global_cached(X_test)
            rng = np.random.default_rng(seed)
            return rng.normal(mean, std, size=(n_samples, len(X_test)))

        key = self._region_cache_key(local_X, region_key)
        if key in self.local_model_cache:
            model = self.local_model_cache[key]
        else:
            model = self._fit_model(self._to_tensor(local_X), self._to_tensor(local_y).unsqueeze(-1), backend=self.local_backend)
            self.local_model_cache[key] = model

        backend = self.local_backend
        Xq = self._to_tensor(X_test)
        Xq_norm = self._normalize_inputs(Xq, self._to_tensor(local_X))

        if backend in ["exact_gp", "sparse_gp"]:
            model.eval()
            with torch.inference_mode(), fast_pred_var():
                post = model.likelihood(model(Xq_norm))
                samp = post.rsample(sample_shape=torch.Size([n_samples]))
            arr = samp.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            return arr

        if backend == "bnn":
            model.eval()
            preds = []
            with torch.no_grad():
                for _ in range(n_samples):
                    preds.append(model(Xq_norm.float()))
            preds = torch.stack(preds)
            return preds.detach().cpu().numpy().squeeze(-1)

    # ======================
    # Fit Models
    # ======================
    def _fit_model(self, X, y, backend="exact_gp"):
        d = X.shape[-1]
        train_X = self._normalize_inputs(X, X)

        if backend == "exact_gp":
            likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6)).to(self.device)
            with torch.no_grad():
                y_std = y.std().clamp_min(1e-3)
                likelihood.noise = (0.05 * y_std).clamp_min(1e-5)

            model = RobustExactGP(train_X, y.squeeze(-1), likelihood).to(self.device)
            model = fit_exact_gp_model(model, likelihood, max_iter=60, patience=8)
            model.likelihood = likelihood
            return model

        if backend == "sparse_gp":
            n_inducing = min(self.n_inducing, train_X.shape[0])
            idx = torch.randperm(train_X.shape[0], device=train_X.device)[:n_inducing]
            inducing = train_X[idx].clone().detach()
            model = SparseGPModel(train_X=train_X, train_Y=y, inducing_points=inducing).to(self.device)

            mll = VariationalELBO(model.likelihood, model.model, num_data=y.size(0))
            model.train(); model.likelihood.train()
            opt = torch.optim.Adam(model.parameters(), lr=0.05)

            for _ in range(20):
                opt.zero_grad(set_to_none=True)
                out = model(train_X)
                loss = -mll(out, y.squeeze(-1))
                loss.backward()
                opt.step()
                if loss.item() < 1e-5:
                    break

            model.eval(); model.likelihood.eval()
            return model

        if backend == "bnn":
            model = BayesianNN(d, hidden_dim=self.bnn_hidden, dropout_p=self.bnn_dropout).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            for _ in range(300):
                opt.zero_grad(set_to_none=True)
                preds = model(train_X.float())
                loss = F.mse_loss(preds, y.float())
                loss.backward()
                opt.step()
            model.eval()
            return model

        raise ValueError(f"Unknown backend: {backend}")

    # ======================
    # Predict from model
    # ======================
    def _predict_from_model(self, model, Xq, ref_X, backend):
        Xq_norm = self._normalize_inputs(Xq, ref_X)

        if backend in ["exact_gp", "sparse_gp"]:
            model.eval()
            with torch.inference_mode(), fast_pred_var():
                post = model.likelihood(model(Xq_norm))
                mean = post.mean.detach().cpu().numpy().reshape(-1)
                std = post.variance.sqrt().detach().cpu().numpy().reshape(-1)
            return mean, std

        if backend == "bnn":
            model.eval()
            with torch.no_grad():
                m_t, s_t = model.predict(Xq_norm.float(), n_samples=20)
            return m_t.cpu().numpy(), s_t.cpu().numpy()

        # Fallback (shouldn't happen)
        return np.zeros(Xq.shape[0]), np.ones(Xq.shape[0])

    # ======================
    # Posterior samples from global model
    # ======================
    def gp_posterior_samples(self, X_test, seed=0, n_samples=1):
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
            self.global_model.eval()
            with torch.inference_mode(), fast_pred_var():
                post = self.global_model.likelihood(self.global_model(Xq_norm))
                samp = post.rsample(sample_shape=torch.Size([n_samples]))
            arr = samp.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            return arr

        if backend == "bnn":
            self.global_model.eval()
            preds = []
            with torch.no_grad():
                for _ in range(n_samples):
                    preds.append(self.global_model(Xq_norm.float()))
            preds = torch.stack(preds)
            return preds.detach().cpu().numpy().squeeze(-1)

    # ======================
    # Utility
    # ======================
    def get_best_value(self):
        return self.global_y.min().item() if self.global_y is not None else float("inf")
