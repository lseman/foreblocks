import math
from abc import ABC, abstractmethod
from typing import Tuple

# ================ OPTIMIZED LOSS FUNCTIONS ================
import numpy as np
from numba import njit, prange


class LossFunction(ABC):
    """Abstract base class for loss functions with optimized implementations"""

    @abstractmethod
    def grad_hess(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients and hessians"""
        pass

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss value"""
        pass


# ================ NUMBA-ACCELERATED COMPUTATION KERNELS ================


@njit(parallel=True, fastmath=True, cache=True)
def mse_grad_hess_kernel(y_true, y_pred):
    """Ultra-fast MSE gradient and hessian computation"""
    n = len(y_true)
    grad = np.empty(n, dtype=np.float64)
    hess = np.ones(n, dtype=np.float64)

    for i in prange(n):
        grad[i] = y_pred[i] - y_true[i]

    return grad, hess


@njit(parallel=True, fastmath=True, cache=True)
def mse_loss_kernel(y_true, y_pred):
    """Ultra-fast MSE loss computation"""
    n = len(y_true)
    total_loss = 0.0

    for i in prange(n):
        diff = y_true[i] - y_pred[i]
        total_loss += diff * diff

    return total_loss / n


@njit(parallel=True, fastmath=True, cache=True)
def huber_grad_hess_kernel(y_true, y_pred, delta):
    """Optimized Huber loss gradient and hessian computation"""
    n = len(y_true)
    grad = np.empty(n, dtype=np.float64)
    hess = np.empty(n, dtype=np.float64)

    for i in prange(n):
        residual = y_pred[i] - y_true[i]
        abs_residual = abs(residual)  # Use math.fabs for Numba

        if abs_residual <= delta:
            grad[i] = residual
            hess[i] = 1.0
        else:
            grad[i] = delta * (1.0 if residual > 0 else -1.0)
            hess[i] = 1e-6  # Small positive value for numerical stability

    return grad, hess


@njit(parallel=True, fastmath=True, cache=True)
def huber_loss_kernel(y_true, y_pred, delta):
    """Optimized Huber loss computation"""
    n = len(y_true)
    total_loss = 0.0
    delta_sq_half = 0.5 * delta * delta

    for i in prange(n):
        residual = y_pred[i] - y_true[i]
        abs_residual = abs(residual)

        if abs_residual <= delta:
            total_loss += 0.5 * residual * residual
        else:
            total_loss += delta * abs_residual - delta_sq_half

    return total_loss / n


@njit(parallel=True, fastmath=True, cache=True)
def quantile_grad_hess_kernel(y_true, y_pred, alpha):
    """Optimized quantile loss gradient and hessian computation"""
    n = len(y_true)
    grad = np.empty(n, dtype=np.float64)
    hess = np.full(n, 1e-6, dtype=np.float64)  # Constant small hessian
    alpha_minus_1 = alpha - 1.0

    for i in prange(n):
        residual = y_pred[i] - y_true[i]
        grad[i] = alpha if residual > 0 else alpha_minus_1

    return grad, hess


@njit(parallel=True, fastmath=True, cache=True)
def quantile_loss_kernel(y_true, y_pred, alpha):
    """Optimized quantile loss computation"""
    n = len(y_true)
    total_loss = 0.0
    alpha_minus_1 = alpha - 1.0

    for i in prange(n):
        residual = y_pred[i] - y_true[i]
        if residual > 0:
            total_loss += alpha * residual
        else:
            total_loss += alpha_minus_1 * residual

    return total_loss / n


@njit(parallel=True, fastmath=True, cache=True)
def logistic_grad_hess_kernel(y_true, y_pred):
    """Highly optimized logistic loss gradient and hessian computation"""
    n = len(y_true)
    grad = np.empty(n, dtype=np.float64)
    hess = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Clip for numerical stability
        logit = max(-250.0, min(250.0, y_pred[i]))

        # Optimized sigmoid computation
        if logit >= 0:
            exp_neg_logit = math.exp(-logit)
            prob = 1.0 / (1.0 + exp_neg_logit)
        else:
            exp_logit = math.exp(logit)
            prob = exp_logit / (1.0 + exp_logit)

        grad[i] = prob - y_true[i]
        hess_val = prob * (1.0 - prob)
        hess[i] = max(hess_val, 1e-6)  # Ensure positive definite

    return grad, hess


@njit(parallel=True, fastmath=True, cache=True)
def logistic_loss_kernel(y_true, y_pred):
    """Highly optimized logistic loss computation using log1p"""
    n = len(y_true)
    total_loss = 0.0
    eps = 1e-15

    for i in prange(n):
        # Clip for numerical stability
        logit = max(-250.0, min(250.0, y_pred[i]))

        # Numerically stable probability computation
        if logit >= 0:
            exp_neg_logit = math.exp(-logit)
            prob = 1.0 / (1.0 + exp_neg_logit)
            # Use log1p for better numerical stability
            log_prob = -math.log1p(exp_neg_logit)
            log_one_minus_prob = -logit + log_prob
        else:
            exp_logit = math.exp(logit)
            prob = exp_logit / (1.0 + exp_logit)
            log_prob = logit - math.log1p(exp_logit)
            log_one_minus_prob = -math.log1p(exp_logit)

        # Compute cross-entropy loss
        if y_true[i] > 0.5:  # Treat as binary: 0 or 1
            total_loss -= log_prob
        else:
            total_loss -= log_one_minus_prob

    return total_loss / n


# ================ OPTIMIZED LOSS FUNCTION CLASSES ================


class MSELoss(LossFunction):
    """Optimized Mean Squared Error loss with Numba acceleration"""

    def grad_hess(self, y_true, y_pred):
        """Vectorized MSE gradient and hessian computation"""
        return mse_grad_hess_kernel(y_true, y_pred)

    def loss(self, y_true, y_pred):
        """Optimized MSE loss computation"""
        return mse_loss_kernel(y_true, y_pred)


class HuberLoss(LossFunction):
    """Optimized Huber loss with enhanced numerical stability"""

    def __init__(self, delta=1.0):
        if delta <= 0:
            raise ValueError("Delta must be positive")
        self.delta = float(delta)

    def grad_hess(self, y_true, y_pred):
        """Optimized Huber gradient and hessian computation"""
        return huber_grad_hess_kernel(y_true, y_pred, self.delta)

    def loss(self, y_true, y_pred):
        """Optimized Huber loss computation"""
        return huber_loss_kernel(y_true, y_pred, self.delta)


class QuantileLoss(LossFunction):
    """Optimized Quantile loss for robust regression"""

    def __init__(self, alpha=0.5):
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = float(alpha)

    def grad_hess(self, y_true, y_pred):
        """Optimized quantile gradient and hessian computation"""
        return quantile_grad_hess_kernel(y_true, y_pred, self.alpha)

    def loss(self, y_true, y_pred):
        """Optimized quantile loss computation"""
        return quantile_loss_kernel(y_true, y_pred, self.alpha)


class LogisticLoss(LossFunction):
    """Highly optimized Logistic loss with superior numerical stability"""

    def grad_hess(self, y_true, y_pred):
        """Optimized logistic gradient and hessian computation"""
        return logistic_grad_hess_kernel(y_true, y_pred)

    def loss(self, y_true, y_pred):
        """Optimized logistic loss computation"""
        return logistic_loss_kernel(y_true, y_pred)


# ================ ADVANCED LOSS FUNCTIONS ================


class FocalLoss(LossFunction):
    """Focal Loss for addressing class imbalance (Lin et al., 2017)"""

    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def grad_hess(self, y_true, y_pred):
        """Focal loss gradients and hessians"""
        return focal_grad_hess_kernel(y_true, y_pred, self.alpha, self.gamma)

    def loss(self, y_true, y_pred):
        """Focal loss computation"""
        return focal_loss_kernel(y_true, y_pred, self.alpha, self.gamma)


@njit(parallel=True, fastmath=True, cache=True)
def focal_grad_hess_kernel(y_true, y_pred, alpha, gamma):
    """Optimized Focal loss gradient and hessian computation"""
    n = len(y_true)
    grad = np.empty(n, dtype=np.float64)
    hess = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Clip for numerical stability
        logit = max(-250.0, min(250.0, y_pred[i]))

        # Compute probability
        if logit >= 0:
            exp_neg_logit = math.exp(-logit)
            prob = 1.0 / (1.0 + exp_neg_logit)
        else:
            exp_logit = math.exp(logit)
            prob = exp_logit / (1.0 + exp_logit)

        y = y_true[i]
        p = prob

        # Focal loss terms
        if y > 0.5:  # Positive class
            pt = p
            at = alpha
        else:  # Negative class
            pt = 1.0 - p
            at = 1.0 - alpha

        # Avoid numerical issues
        pt = max(pt, 1e-8)

        # Gradient computation
        focal_weight = at * math.pow(1.0 - pt, gamma)
        if y > 0.5:
            grad[i] = focal_weight * (gamma * pt * math.log(pt) + pt - 1.0)
        else:
            grad[i] = focal_weight * (
                1.0 - gamma * (1.0 - pt) * math.log(1.0 - pt) - pt
            )

        # Simplified hessian approximation
        hess[i] = max(focal_weight * pt * (1.0 - pt), 1e-6)

    return grad, hess


@njit(parallel=True, fastmath=True, cache=True)
def focal_loss_kernel(y_true, y_pred, alpha, gamma):
    """Optimized Focal loss computation"""
    n = len(y_true)
    total_loss = 0.0

    for i in prange(n):
        # Clip for numerical stability
        logit = max(-250.0, min(250.0, y_pred[i]))

        # Compute probability
        if logit >= 0:
            exp_neg_logit = math.exp(-logit)
            prob = 1.0 / (1.0 + exp_neg_logit)
            log_prob = -math.log1p(exp_neg_logit)
        else:
            exp_logit = math.exp(logit)
            prob = exp_logit / (1.0 + exp_logit)
            log_prob = logit - math.log1p(exp_logit)

        y = y_true[i]

        # Focal loss computation
        if y > 0.5:  # Positive class
            pt = prob
            at = alpha
            log_pt = log_prob
        else:  # Negative class
            pt = 1.0 - prob
            at = 1.0 - alpha
            log_pt = math.log(max(1.0 - prob, 1e-8))

        # Focal term
        focal_weight = at * math.pow(max(1.0 - pt, 1e-8), gamma)
        total_loss -= focal_weight * log_pt

    return total_loss / n


# ================ UTILITY FUNCTIONS ================


def get_loss_function(objective: str, **kwargs) -> LossFunction:
    """Factory function to create loss functions"""
    loss_map = {
        "reg:squarederror": MSELoss,
        "reg:pseudohubererror": lambda: HuberLoss(kwargs.get("delta", 1.0)),
        "reg:quantileerror": lambda: QuantileLoss(kwargs.get("alpha", 0.5)),
        "binary:logistic": LogisticLoss,
        "binary:focal": lambda: FocalLoss(
            kwargs.get("alpha", 0.25), kwargs.get("gamma", 2.0)
        ),
    }

    if objective not in loss_map:
        raise ValueError(f"Unknown objective: {objective}")

    loss_class = loss_map[objective]
    return (
        loss_class()
        if not callable(loss_class) or loss_class in [MSELoss, LogisticLoss]
        else loss_class()
    )


@njit(cache=True)
def compute_base_score_mse(y):
    """Fast base score computation for MSE"""
    return np.mean(y)


@njit(cache=True)
def compute_base_score_huber(y):
    """Fast base score computation for Huber (median)"""
    sorted_y = np.sort(y)
    n = len(sorted_y)
    if n % 2 == 0:
        return 0.5 * (sorted_y[n // 2 - 1] + sorted_y[n // 2])
    else:
        return sorted_y[n // 2]


@njit(cache=True)
def compute_base_score_quantile(y, alpha):
    """Fast base score computation for quantile regression"""
    sorted_y = np.sort(y)
    n = len(sorted_y)
    index = alpha * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    weight = index - lower
    return sorted_y[lower] * (1 - weight) + sorted_y[upper] * weight


@njit(cache=True)
def compute_base_score_logistic(y):
    """Fast base score computation for logistic regression"""
    pos_count = 0.0
    n = len(y)

    for i in range(n):
        if y[i] > 0.5:
            pos_count += 1.0

    pos_ratio = pos_count / n
    # Avoid log(0) by clamping
    pos_ratio = max(1e-15, min(1 - 1e-15, pos_ratio))
    return math.log(pos_ratio / (1.0 - pos_ratio))
