import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import pdist

# Dependency checks
try:
    import botorch
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
    from gpytorch.models import ApproximateGP, ExactGP
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import ScaleKernel, MaternKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
    from sklearn.cluster import KMeans
    BOTORCH_AVAILABLE = True
    GPYTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    GPYTORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class BaseSurrogate(ABC):
    """Streamlined base class for all surrogate models"""
    
    def __init__(self, device="auto", verbose=False, **kwargs):
        # Accept but ignore extra kwargs like X_train, y_train, n_parallel, etc.
        self.device = self._setup_device(device)
        self.verbose = verbose
        self.fitted = False
        self.scaler = StandardScaler()
    
    def _setup_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pass

class LinearSurrogate(BaseSurrogate):
    """Fast linear surrogate for small datasets and fallback"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coef_ = None
        self.intercept_ = 0
        self.residual_std_ = 0.1
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        try:
            # Ridge regression with small regularization
            X_aug = np.column_stack([np.ones(len(X_scaled)), X_scaled])
            XtX = X_aug.T @ X_aug + 1e-6 * np.eye(X_aug.shape[1])  # Ridge regularization
            Xty = X_aug.T @ y
            params = np.linalg.solve(XtX, Xty)
            
            self.intercept_ = params[0]
            self.coef_ = params[1:]
            
            # Estimate residual standard deviation
            y_pred = self.intercept_ + X_scaled @ self.coef_
            residuals = y - y_pred
            self.residual_std_ = max(np.std(residuals), 0.01)
            
        except:
            # Fallback to mean
            self.intercept_ = np.mean(y)
            self.coef_ = np.zeros(X_scaled.shape[1])
            self.residual_std_ = max(np.std(y), 0.1)
        
        self.fitted = True
        self._log(f"✅ Linear model fitted with std={self.residual_std_:.4f}")
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        
        if self.coef_ is None:
            mean = np.full(len(X), self.intercept_)
        else:
            mean = self.intercept_ + X_scaled @ self.coef_
        
        std = np.full(len(X), self.residual_std_)
        
        return (mean, std) if return_std else mean

class XGBoostSurrogate(BaseSurrogate):
    """Optimized XGBoost surrogate with uncertainty estimation"""
    
    def __init__(self, n_estimators=100, max_depth=6, **kwargs):
        super().__init__(**kwargs)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required")
        
        # Optimized parameters
        self.params = {
            'objective': 'reg:squarederror',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        self.model = None
        self.quantile_models = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        # Main model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_scaled, y)
        
        # Quantile models for uncertainty (only if enough data)
        if len(X) > 20:
            for quantile in [0.1, 0.9]:
                q_params = self.params.copy()
                q_params.update({
                    'objective': 'reg:quantileerror',
                    'quantile_alpha': quantile,
                    'n_estimators': max(50, self.params['n_estimators'] // 2)  # Faster fitting
                })
                
                q_model = xgb.XGBRegressor(**q_params)
                q_model.fit(X_scaled, y)
                self.quantile_models[quantile] = q_model
        
        self.fitted = True
        self._log(f"✅ XGBoost fitted with {len(self.quantile_models)} quantile models")
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        mean = self.model.predict(X_scaled)
        
        if return_std:
            if len(self.quantile_models) == 2:
                # Use quantile models for uncertainty
                q_low = self.quantile_models[0.1].predict(X_scaled)
                q_high = self.quantile_models[0.9].predict(X_scaled)
                std = (q_high - q_low) / 2.56  # 80% interval -> std
            else:
                # Fallback: use fixed uncertainty
                std = np.full_like(mean, np.std(mean) * 0.1)
            
            return mean

class SparseGPModel(ApproximateGP if GPYTORCH_AVAILABLE else object):
    """Sparse GP model using variational inference"""
    
    def __init__(self, inducing_points):
        if not GPYTORCH_AVAILABLE:
            raise ImportError("GPyTorch required for sparse GP")
        
        if isinstance(inducing_points, np.ndarray):
            inducing_points = torch.from_numpy(inducing_points).float()
        
        n_inducing, n_dims = inducing_points.shape
        
        variational_distribution = CholeskyVariationalDistribution(n_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=n_dims))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SparseGPSurrogate(BaseSurrogate):
    """Sparse Gaussian Process using inducing points for scalability"""
    
    def __init__(self, n_inducing=50, inducing_strategy="kmeans", max_iter=50, **kwargs):
        super().__init__(**kwargs)
        if not GPYTORCH_AVAILABLE:
            raise ImportError("GPyTorch required for sparse GP")
        
        self.n_inducing = n_inducing
        self.inducing_strategy = inducing_strategy
        self.max_iter = max_iter
        self.model = None
        self.likelihood = None
    
    def _select_inducing_points(self, X: np.ndarray) -> np.ndarray:
        """Select inducing points using various strategies"""
        n_data, n_dims = X.shape
        n_inducing = min(self.n_inducing, n_data)
        
        if n_data <= n_inducing:
            return X
        
        if self.inducing_strategy == "kmeans":
            try:
                kmeans = KMeans(n_clusters=n_inducing, random_state=42, n_init=10)
                kmeans.fit(X)
                return kmeans.cluster_centers_
            except:
                pass
        
        elif self.inducing_strategy == "random":
            idx = np.random.choice(n_data, n_inducing, replace=False)
            return X[idx]
        
        elif self.inducing_strategy == "greedy":
            return self._greedy_selection(X, n_inducing)
        
        # Fallback to random
        idx = np.random.choice(n_data, n_inducing, replace=False)
        return X[idx]
    
    def _greedy_selection(self, X: np.ndarray, n_inducing: int) -> np.ndarray:
        """Greedy inducing point selection for maximum diversity"""
        if len(X) <= n_inducing:
            return X
        
        selected_idx = [0]  # Start with first point
        remaining_idx = list(range(1, len(X)))
        
        for _ in range(1, n_inducing):
            if not remaining_idx:
                break
            
            # Find point farthest from all selected points
            best_dist = -1
            best_idx = None
            
            for i in remaining_idx:
                min_dist = min(np.linalg.norm(X[i] - X[j]) for j in selected_idx)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            
            if best_idx is not None:
                selected_idx.append(best_idx)
                remaining_idx.remove(best_idx)
        
        return X[selected_idx]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        # Select inducing points
        inducing_points = self._select_inducing_points(X_scaled)
        self._log(f"Selected {len(inducing_points)} inducing points using {self.inducing_strategy}")
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        inducing_tensor = torch.from_numpy(inducing_points).float().to(self.device)
        
        # Initialize model and likelihood
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = SparseGPModel(inducing_tensor).to(self.device)
        
        # Training
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.1)
        
        mll = VariationalELBO(self.likelihood, self.model, num_data=len(y))
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(self.max_iter):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.verbose and (i % 10 == 0 or i == self.max_iter - 1):
                self._log(f"[{i:03d}] Loss: {loss.item():.5f}")
            
            if patience_counter >= 15:
                self._log(f"Early stopping at iteration {i}")
                break
        
        self.model.eval()
        self.likelihood.eval()
        self.fitted = True
        self._log("✅ Sparse GP fitted")
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        
        # Batch processing for memory efficiency
        batch_size = 1000
        means, stds = [], []
        
        for i in range(0, len(X_scaled), batch_size):
            batch = X_scaled[i:i + batch_size]
            X_tensor = torch.from_numpy(batch).float().to(self.device)
            
            with torch.no_grad():
                pred = self.likelihood(self.model(X_tensor))
                means.append(pred.mean.cpu().numpy())
                stds.append(pred.stddev.cpu().numpy())
        
        mean = np.concatenate(means)
        std = np.concatenate(stds) if return_std else None
        
        return (mean, std) if return_std else mean

class EnsembleGPSurrogate(BaseSurrogate):
    """Ensemble of Gaussian Processes for improved uncertainty quantification"""
    
    def __init__(self, n_models=3, base_model_type="gp", parallel_training=True, **kwargs):
        super().__init__(**kwargs)
        self.n_models = n_models
        self.base_model_type = base_model_type
        self.parallel_training = parallel_training
        self.models = []
        self.rng = np.random.RandomState(42)
    
    def _create_base_model(self, model_id: int):
        """Create a single base model for the ensemble"""
        if self.base_model_type == "gp":
            # Create GP with different kernels
            kernels = ["auto", "auto", "auto"]  # Will be auto-selected
            return AdaptiveGPSurrogate(
                kernel_selection=kernels[model_id % len(kernels)],
                device=self.device,
                verbose=False
            )
        
        elif self.base_model_type == "xgboost" and XGBOOST_AVAILABLE:
            # Create XGBoost with different parameters
            configs = [
                {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
                {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05},
                {"n_estimators": 80, "max_depth": 8, "learning_rate": 0.15},
            ]
            config = configs[model_id % len(configs)]
            return XGBoostSurrogate(device=self.device, verbose=False, **config)
        
        else:
            # Fallback to GP
            return AdaptiveGPSurrogate(device=self.device, verbose=False)
    
    def _train_single_model(self, args):
        """Train a single model (for parallel processing)"""
        model_id, X, y = args
        
        # Bootstrap sampling for diversity
        n_samples = len(X)
        indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[indices], y[indices]
        
        try:
            model = self._create_base_model(model_id)
            model.fit(X_boot, y_boot)
            return model, True
        except Exception as e:
            self._log(f"⚠️ Model {model_id+1} training failed: {e}")
            return None, False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.models.clear()
        
        if self.parallel_training and self.n_models > 1:
            # Parallel training
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            tasks = [(i, X, y) for i in range(self.n_models)]
            
            with ThreadPoolExecutor(max_workers=min(self.n_models, 4)) as executor:
                futures = {executor.submit(self._train_single_model, task): i for i, task in enumerate(tasks)}
                
                for future in as_completed(futures):
                    i = futures[future]
                    model, success = future.result()
                    
                    if success:
                        self.models.append(model)
                        self._log(f"✅ Ensemble model {i+1}/{self.n_models} trained")
        else:
            # Sequential training
            for i in range(self.n_models):
                model, success = self._train_single_model((i, X, y))
                if success:
                    self.models.append(model)
                    self._log(f"✅ Ensemble model {i+1}/{self.n_models} trained")
        
        if not self.models:
            raise RuntimeError("All ensemble models failed to train")
        
        self.fitted = True
        self._log(f"✅ Ensemble trained with {len(self.models)}/{self.n_models} successful models")
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("No trained models in ensemble")
        
        # Collect predictions from all models
        predictions = []
        for model in self.models:
            try:
                if return_std:
                    mean, std = model.predict(X, return_std=True)
                    predictions.append((mean, std))
                else:
                    mean = model.predict(X, return_std=False)
                    predictions.append((mean, None))
            except Exception as e:
                self._log(f"⚠️ Model prediction failed: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("All model predictions failed")
        
        # Ensemble statistics
        means = np.array([pred[0] for pred in predictions])
        ensemble_mean = np.mean(means, axis=0)
        
        if return_std:
            # Compute epistemic uncertainty (model disagreement)
            epistemic_var = np.var(means, axis=0)
            
            # Compute aleatoric uncertainty (average predictive variance)
            stds = np.array([pred[1] for pred in predictions if pred[1] is not None])
            if len(stds) > 0:
                aleatoric_var = np.mean(stds**2, axis=0)
            else:
                aleatoric_var = np.zeros_like(epistemic_var)
            
            # Total uncertainty
            total_std = np.sqrt(epistemic_var + aleatoric_var)
            
            return ensemble_mean, total_std
        
        return ensemble_mean, np.maximum(std, 0.001)  # Minimum uncertainty
        
        return mean

class AdaptiveGPSurrogate(BaseSurrogate):
    """Adaptive GP with smart kernel selection"""
    
    def __init__(self, kernel_selection="auto", **kwargs):
        super().__init__(**kwargs)
        self.kernel_selection = kernel_selection
        self.model = None
        self.selected_kernel = None
    
    def _estimate_length_scale_bounds(self, X):
        """Smart length scale bounds from data"""
        distances = pdist(X)
        if len(distances) == 0:
            return (1e-3, 1e3)
        
        nonzero_distances = distances[distances > 0]
        if len(nonzero_distances) == 0:
            return (1e-3, 1e3)
        
        min_dist = np.min(nonzero_distances)
        max_dist = np.max(distances)
        
        return (max(min_dist / 10, 1e-5), min(max_dist * 10, 1e3))
    
    def _create_kernel(self, X, y):
        """Create appropriate kernel based on data characteristics"""
        n_samples, n_dims = X.shape
        ls_bounds = self._estimate_length_scale_bounds(X)
        
        # Estimate noise level
        if n_samples > 5:
            noise_level = max(np.var(y) * 0.01, 1e-6)
        else:
            noise_level = 1e-5
        
        # Kernel selection logic
        if self.kernel_selection == "auto":
            if n_dims <= 3 and n_samples < 50:
                # Simple RBF for low-D, small data
                kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(
                    length_scale=1.0, length_scale_bounds=ls_bounds
                ) + WhiteKernel(noise_level, (1e-6, 1e-1))
                self.selected_kernel = "rbf"
            
            elif n_dims > 5 and n_samples > 30:
                # ARD Matern for high-D
                kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
                    length_scale=np.ones(n_dims), length_scale_bounds=ls_bounds, nu=2.5
                ) + WhiteKernel(noise_level, (1e-6, 1e-1))
                self.selected_kernel = "matern25_ard"
            
            else:
                # Default Matern 2.5
                kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
                    length_scale=1.0, length_scale_bounds=ls_bounds, nu=2.5
                ) + WhiteKernel(noise_level, (1e-6, 1e-1))
                self.selected_kernel = "matern25"
        
        else:
            # Default kernel
            kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
                length_scale=1.0, length_scale_bounds=ls_bounds, nu=2.5
            ) + WhiteKernel(noise_level, (1e-6, 1e-1))
            self.selected_kernel = "matern25_default"
        
        return kernel
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        kernel = self._create_kernel(X_scaled, y)
        
        # Optimize based on data size
        n_restarts = 5 if len(X) < 100 else 3
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.fitted = True
        self._log(f"✅ GP fitted with {self.selected_kernel} kernel")
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        
        if return_std:
            mean, std = self.model.predict(X_scaled, return_std=True)
            return mean, std
        else:
            return self.model.predict(X_scaled, return_std=False)

class BotorchGPSurrogate(BaseSurrogate):
    """BoTorch GP for advanced optimization"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch required")
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Scale inputs but keep on [0,1] for BoTorch
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64, device=self.device)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float64, device=self.device)
        
        # Create and fit model
        self.model = SingleTaskGP(X_tensor, y_tensor).to(self.device)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        try:
            fit_gpytorch_mll(mll, options={'maxiter': 100})
        except:
            self._log("⚠️ BoTorch hyperparameter optimization failed, using defaults")
        
        self.fitted = True
        self._log("✅ BoTorch GP fitted")
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64, device=self.device)
        
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().squeeze()
            
            if return_std:
                std = posterior.variance.sqrt().cpu().numpy().squeeze()
                return mean, std
            return mean

class RandomForestSurrogate(BaseSurrogate):
    """Fast Random Forest surrogate"""
    
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True
        self._log("✅ Random Forest fitted")
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X_scaled) for tree in self.model.estimators_
        ])
        
        mean = np.mean(tree_predictions, axis=0)
        
        if return_std:
            std = np.std(tree_predictions, axis=0)
            std = np.maximum(std, 0.001)  # Minimum uncertainty
            return mean, std
        
        return mean

class SurrogateManager:
    """Streamlined surrogate manager with smart model selection"""
    
    def __init__(self):
        self.model_registry = {
            'linear': LinearSurrogate,
            'xgboost': XGBoostSurrogate,
            'gp': AdaptiveGPSurrogate,
            'botorch': BotorchGPSurrogate,
            'rf': RandomForestSurrogate,
            'ensemble': EnsembleGPSurrogate,
            'sparse': SparseGPSurrogate,
        }
    
    def create_model(self, model_type: str, n_trials: int, **kwargs) -> BaseSurrogate:
        """Create surrogate model with smart selection"""
        
        if model_type == "auto":
            model_type = self._auto_select_type(n_trials, **kwargs)
        
        # Get model class
        model_cls = self.model_registry.get(model_type)
        if model_cls is None:
            print(f"⚠️ Unknown model type '{model_type}', using GP")
            model_cls = AdaptiveGPSurrogate
        
        # Filter kwargs to only pass what's needed for model initialization
        model_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['device', 'verbose', 'kernel_selection', 'n_estimators', 'max_depth',
                               'n_inducing', 'inducing_strategy', 'max_iter', 'n_models', 
                               'base_model_type', 'parallel_training']}
        
        # Check dependencies and fallback
        try:
            model = model_cls(**model_kwargs)
            
            # If X_train and y_train are provided, fit immediately
            if 'X_train' in kwargs and 'y_train' in kwargs:
                X_train = kwargs['X_train']
                y_train = kwargs['y_train']
                if X_train is not None and y_train is not None:
                    model.fit(X_train, y_train)
            
            return model
            
        except ImportError as e:
            print(f"⚠️ {model_type} not available: {e}")
            return self._fallback_model(**model_kwargs)
        
        except Exception as e:
            print(f"⚠️ Failed to create {model_type}: {e}")
            return self._fallback_model(**model_kwargs)
    
    def _auto_select_type(self, n_trials: int, **kwargs) -> str:
        """Smart automatic model selection"""
        n_parallel = kwargs.get('n_parallel', 1)
        X_train = kwargs.get('X_train')
        
        # Determine problem characteristics
        if X_train is not None:
            n_dims = X_train.shape[1]
            n_samples = len(X_train)
        else:
            n_dims = kwargs.get('n_dims', 5)
            n_samples = n_trials
        
        # Selection logic
        if n_samples < 10:
            return 'linear'
        elif n_samples < 30:
            return 'gp'
        elif n_samples < 100 and XGBOOST_AVAILABLE:
            return 'xgboost'
        elif n_samples < 200 and n_parallel > 1:
            return 'ensemble'  # Good for parallel optimization
        elif n_samples > 300 and GPYTORCH_AVAILABLE:
            return 'sparse'  # Scales well for large datasets
        elif n_samples < 300 and BOTORCH_AVAILABLE and n_parallel > 1:
            return 'botorch'
        elif XGBOOST_AVAILABLE:
            return 'xgboost'
        else:
            return 'gp'
    
    def _fallback_model(self, **kwargs) -> BaseSurrogate:
        """Robust fallback model selection"""
        # Try models in order of preference
        fallback_order = ['gp', 'rf', 'linear']
        
        for model_type in fallback_order:
            try:
                model_cls = self.model_registry[model_type]
                return model_cls(**kwargs)
            except:
                continue
        
        # Final fallback - create minimal linear model
        try:
            return LinearSurrogate(device=kwargs.get('device', 'auto'), 
                                 verbose=kwargs.get('verbose', False))
        except:
            # Absolute final fallback
            return LinearSurrogate()
    
    def get_available_models(self) -> list:
        """Get list of available model types"""
        available = ['linear', 'gp', 'rf']  # Always available
        
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        if BOTORCH_AVAILABLE:
            available.append('botorch')
        if GPYTORCH_AVAILABLE:
            available.extend(['sparse', 'ensemble'])
        
        return available
    
    def validate_model_type(self, model_type: str) -> bool:
        """Check if model type is valid and available"""
        if model_type == "auto":
            return True
        
        if model_type not in self.model_registry:
            return False
        
        # Check dependencies
        if model_type == 'botorch' and not BOTORCH_AVAILABLE:
            return False
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            return False
        if model_type in ['sparse', 'ensemble'] and not GPYTORCH_AVAILABLE:
            return False
        
        return True

# Example usage and testing
def test_surrogate_manager():
    """Test the surrogate manager with synthetic data"""
    np.random.seed(42)
    
    # Generate test data
    X = np.random.rand(50, 3)
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(50)
    
    manager = SurrogateManager()
    
    print("Available models:", manager.get_available_models())
    
    # Test different model types
    test_models = ['auto', 'linear', 'gp', 'rf']
    
    # Add advanced models if available
    if XGBOOST_AVAILABLE:
        test_models.append('xgboost')
    if GPYTORCH_AVAILABLE:
        test_models.extend(['sparse', 'ensemble'])
    if BOTORCH_AVAILABLE:
        test_models.append('botorch')
    
    for model_type in test_models:
        if manager.validate_model_type(model_type):
            print(f"\n🧪 Testing {model_type}...")
            try:
                model = manager.create_model(
                    model_type=model_type,
                    n_trials=len(X),
                    X_train=X,
                    y_train=y,
                    verbose=True,
                    # Additional parameters for specific models
                    n_inducing=20,  # For sparse GP
                    n_models=2,     # For ensemble (reduced for testing)
                    base_model_type='gp'  # For ensemble
                )
                
                # Test prediction
                X_test = np.random.rand(10, 3)
                mean, std = model.predict(X_test)
                
                print(f"✅ {model_type}: mean shape={mean.shape}, std shape={std.shape}")
                print(f"   Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
                print(f"   Std range: [{std.min():.3f}, {std.max():.3f}]")
                
                # Test if model remembers training
                if hasattr(model, 'fitted'):
                    print(f"   Model fitted: {model.fitted}")
                
            except Exception as e:
                print(f"❌ {model_type} failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_surrogate_manager()