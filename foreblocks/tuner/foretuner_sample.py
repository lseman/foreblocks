# Suppress warnings globally
import warnings

warnings.filterwarnings("ignore")

import logging

# Standard library
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

# Core scientific stack
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Scipy
from scipy.stats import norm

# Project-specific modules
from .foretuner_config import *
from .foretuner_surrogate import *

# Numba (optional acceleration)
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# scikit-learn (optional)
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
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# PyTorch + BoTorch + GPyTorch (optional)
try:
    import botorch
    import gpytorch
    import torch
    from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.monte_carlo import (
        qExpectedImprovement,
        qUpperConfidenceBound,
    )
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ConstantMean
    from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        VariationalStrategy,
    )

    BOTORCH_AVAILABLE = True
    GPYTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    GPYTORCH_AVAILABLE = False


class SamplingStrategy(ABC):
    @abstractmethod
    def generate_samples(
        self, n_dims: int, n_samples: int, rng: np.random.RandomState
    ) -> np.ndarray:
        pass


class LHSSamplingStrategy(SamplingStrategy):
    def __init__(self):
        try:
            from scipy.stats import qmc

            self.qmc = qmc
            self.available = True
        except ImportError:
            self.available = False

    def generate_samples(
        self, n_dims: int, n_samples: int, rng: np.random.RandomState
    ) -> np.ndarray:
        if self.available:
            sampler = self.qmc.LatinHypercube(d=n_dims, seed=rng.randint(0, 10_000))
            return sampler.random(n_samples)
        return rng.random((n_samples, n_dims))


class SobolSamplingStrategy(SamplingStrategy):
    def __init__(self):
        try:
            from scipy.stats import qmc

            self.qmc = qmc
            self.available = True
        except ImportError:
            self.available = False

    def generate_samples(
        self, n_dims: int, n_samples: int, rng: np.random.RandomState
    ) -> np.ndarray:
        if self.available:
            sampler = self.qmc.Sobol(
                d=n_dims, scramble=True, seed=rng.randint(0, 10_000)
            )
            return sampler.random(n_samples)
        return rng.random((n_samples, n_dims))


class GridSamplingStrategy(SamplingStrategy):
    def generate_samples(
        self, n_dims: int, n_samples: int, rng: np.random.RandomState
    ) -> np.ndarray:
        from itertools import product

        n_per_dim = max(2, int(np.ceil(n_samples ** (1 / n_dims))))
        coords = [np.linspace(0, 1, n_per_dim) for _ in range(n_dims)]
        grid = np.array(list(product(*coords)))
        if len(grid) > n_samples:
            grid = grid[rng.choice(len(grid), size=n_samples, replace=False)]
        return grid[:n_samples]


class TargetedSamplingStrategy(SamplingStrategy):
    def generate_samples(
        self, n_dims: int, n_samples: int, rng: np.random.RandomState
    ) -> np.ndarray:
        center_samples = int(n_samples * 0.4)
        edge_samples = int(n_samples * 0.3)
        rand_samples = n_samples - center_samples - edge_samples

        center = np.clip(
            rng.normal(loc=0.5, scale=0.1, size=(center_samples, n_dims)), 0, 1
        )
        corners = rng.choice([0.1, 0.9], size=(edge_samples, n_dims)) + rng.normal(
            0, 0.05, size=(edge_samples, n_dims)
        )
        corners = np.clip(corners, 0, 1)
        uniform = rng.random((rand_samples, n_dims))

        return np.vstack((center, corners, uniform))


class RandomSamplingStrategy(SamplingStrategy):
    def generate_samples(
        self, n_dims: int, n_samples: int, rng: np.random.RandomState
    ) -> np.ndarray:
        return rng.random((n_samples, n_dims))


class SamplingManager:
    def __init__(self):
        self.strategies = {
            "lhs": LHSSamplingStrategy(),
            "sobol": SobolSamplingStrategy(),
            "grid": GridSamplingStrategy(),
            "targeted": TargetedSamplingStrategy(),
            "random": RandomSamplingStrategy(),
        }

    def generate_samples(
        self,
        strategy_name: str,
        n_dims: int,
        n_samples: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        return self.strategies.get(
            strategy_name, self.strategies["random"]
        ).generate_samples(n_dims, n_samples, rng)
