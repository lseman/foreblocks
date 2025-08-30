# acquisitions.py (single file module)
# Pluggable acquisition strategies with a clean manager & registry.
# - EI, qEI (greedy corr-attenuated), Thompson Sampling (RF/GP) included
# - Ellipsoidal TR via ARD lengthscales (whitening)
# - Mahalanobis diversity (k++-style); Numba acceleration when available
# - Utilities shared by acquisitions via the manager (no code duplication)

from __future__ import annotations

# thompson.py
from typing import Dict, List, Optional, Type

import numpy as np
from sobol_seq import i4_sobol_generate

# ----------------------- Optional deps ---------------------------------------
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

# ----------------------- Math kernels ----------------------------------------
SQRT1_2 = 1.0 / np.sqrt(2.0)
INV_SQRT_2PI = 0.39894228040143267794

def _phi(z):   # N(0,1) pdf
    return np.exp(-0.5 * z * z) * INV_SQRT_2PI

def _Phi(z):   # exact CDF via erfc
    return 0.5 * np.erfc(-z * SQRT1_2)

if HAS_NUMBA:
    import math
    @njit(cache=True, fastmath=True, nogil=True)
    def compute_expected_improvement(mean, std, best_f):
        n = mean.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = std[i]
            if s <= 1e-12:
                out[i] = 0.0
                continue
            z = (best_f - mean[i]) / s  # minimization EI
            Phi = 0.5 * math.erfc(-z * 0.7071067811865475)
            phi = math.exp(-0.5 * z * z) * 0.3989422804014327
            out[i] = s * (z * Phi + phi)
        return out
else:
    def compute_expected_improvement(mean, std, best_f):
        std = np.maximum(std, 1e-12)
        z = (best_f - mean) / std
        return std * (z * _Phi(z) + _phi(z))

# ----------------------- Whitening & diversity utils -------------------------
class _Whitener:
    """Diagonal whitener from ARD lengthscales (ls)."""
    def __init__(self, ls):
        ls = np.asarray(ls, dtype=float).reshape(-1)
        ls = np.clip(ls, 1e-9, 1e9)
        self.L    = np.diag(ls)
        self.Linv = np.diag(1.0 / ls)

    def to_white(self, x, c):    return self.Linv @ (x - c)
    def from_white(self, xw, c): return (self.L @ xw) + c

def _unit_ball(n, d, rng):
    v = rng.normal(size=(n, d))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    r = rng.random(n) ** (1.0 / d)
    return v * r[:, None]

def _phase_jitter(points: np.ndarray,
                  bounds: np.ndarray,
                  phases=(0.0, 0.25, 0.5, 0.75),
                  strength=0.30,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Duplicate points nudged toward fractional phases (mod 1)."""
    if points.size == 0:
        return points
    lo, hi = bounds[:, 0], bounds[:, 1]
    out = []
    for p in phases:
        Z = points.copy()
        frac = Z - np.floor(Z)
        shift = (p - frac + 1.0) % 1.0
        Z = Z + strength * shift
        Z = np.minimum(hi, np.maximum(lo, Z))
        out.append(Z)
    return np.vstack(out)

if HAS_NUMBA:
    @njit(cache=True, fastmath=True, nogil=True)
    def _weighted_kpp_indices_metric(X, weights, k, wdim):
        N, d = X.shape
        k = min(k, N)
        sel = np.empty(k, dtype=np.int64)

        ww = np.maximum(weights.copy(), 1e-16)
        ww /= ww.sum()

        # first = argmax weight
        first = 0
        maxw = ww[0]
        for i in range(1, N):
            if ww[i] > maxw:
                maxw = ww[i]
                first = i
        sel[0] = first

        # init minD2 to first
        minD2 = np.empty(N, dtype=np.float64)
        x0 = X[first]
        for i in range(N):
            d2 = 0.0
            for j in range(d):
                diff = X[i, j] - x0[j]
                d2 += wdim[j] * diff * diff
            minD2[i] = d2

        for t in range(1, k):
            best_i = 0
            best_score = minD2[0] * ww[0]
            for i in range(1, N):
                s = minD2[i] * ww[i]
                if s > best_score:
                    best_score = s
                    best_i = i
            sel[t] = best_i

            xi = X[best_i]
            for i in range(N):
                d2 = 0.0
                for j in range(d):
                    diff = X[i, j] - xi[j]
                    d2 += wdim[j] * diff * diff
                if d2 < minD2[i]:
                    minD2[i] = d2
        return sel

def _pairwise_mahal_min_d2(C, picked_idx, w):
    dmin = np.full(len(C), np.inf)
    for i in range(len(C)):
        for j in picked_idx:
            diff = C[i] - C[j]
            d2 = np.sum(w * diff * diff)
            if d2 < dmin[i]:
                dmin[i] = d2
    return dmin


# ----------------------- Acquisition base & registry -------------------------
class Acquisition:
    """Strategy interface."""
    name: str = "base"

    def __init__(self, manager: AcquisitionCandidateManager):
        self.manager = manager  # injected for helpers/configs

    @classmethod
    def register(cls, registry: Dict[str, Type["Acquisition"]]):
        registry[cls.name] = cls

    def propose(self, region, bounds, rng, surrogate_manager) -> np.ndarray:
        raise NotImplementedError


class ThompsonSampling(Acquisition):
    """
    Thompson Sampling acquisition.
    - Backend: 'auto' (RF then GP), 'rf', or 'gp'
    - Proper Normal draw for RF pseudo-posterior (per-tree dispersion)
    - Optional antithetic noise (improves stability in small batches)
    - Deterministic seeding via provided rng
    - EI fallback if both backends unavailable
    """
    name = "ts"

    def __init__(self, manager, backend: str = "gp", antithetic: bool = True):
        super().__init__(manager)
        self.backend = backend
        self.antithetic = bool(antithetic)
        self.rf = RandomForestSampler()  # uses sklearn RF if available

    # ------- public API -------
    def propose(self, region, bounds, rng, surrogate_manager):
        man = self.manager
        whitener, r_white = man._get_whitened_TR(region, surrogate_manager)

        dim = int(bounds.shape[0])
        n_candidates = min(800, max(200, 40 * dim))

        C = man._candidate_pool(
            whitener, r_white, region.center, bounds, n_candidates, rng, region
        )

        # 1) Try configured backend(s)
        samples = None
        minimize = True

        if self.backend in ("auto", "rf"):
            samples = self._rf_ts_from_pool(C, surrogate_manager, rng)

        if samples is None and self.backend in ("auto", "gp"):
            samples = self._gp_ts_from_pool(C, surrogate_manager, rng)

        # 2) Fallback → EI
        if samples is None:
            best_f = float(getattr(region, "best_value", 0.0))
            mean, std = surrogate_manager.predict_global_cached(C)
            std = np.maximum(np.asarray(std, dtype=np.float64), 1e-12)
            mean = np.asarray(mean, dtype=np.float64)
            samples = compute_expected_improvement(mean, std, best_f)
            minimize = False  # EI: higher is better

        # 3) Diversity on the scores using Mahalanobis metric scales
        Ldiag = np.diag(whitener.L).astype(np.float64)  # diagonal scaling
        wdim = 1.0 / np.maximum(Ldiag * Ldiag, 1e-18)
        return man._select_diverse(C, np.asarray(samples, dtype=np.float64),
                                   minimize=minimize, metric_scales=wdim)

    # ------- RF pseudo-posterior -------
    def _rf_ts_from_pool(self, candidates, surrogate_manager, rng) -> Optional[np.ndarray]:
        try:
            if not hasattr(surrogate_manager, "global_X") or surrogate_manager.global_X is None:
                return None
            # tolerant torch/np extraction
            Xd = surrogate_manager.global_X
            yd = surrogate_manager.global_y
            Xd = Xd.detach().cpu().numpy() if hasattr(Xd, "detach") else np.asarray(Xd)
            yd = yd.detach().cpu().numpy() if hasattr(yd, "detach") else np.asarray(yd)

            if not self.rf.maybe_fit(Xd, yd):
                return None

            Xq = np.asarray(candidates, dtype=np.float32)
            preds = self.rf.per_tree_predict(Xq)  # (n_trees, N)
            if preds is None:
                return None

            # sample from Normal(mean, var_est) with antithetic noise (optional)
            mean = preds.mean(axis=0)           # (N,)
            std  = preds.std(axis=0, ddof=1)    # (N,)
            std  = np.where(np.isfinite(std), std, 0.0)

            # pure TS should NOT add an arbitrary floor; keep it optional:
            if getattr(self.rf, "std_floor", 0.0) > 0.0:
                std = np.maximum(std, float(self.rf.std_floor))

            N = len(mean)
            z = rng.standard_normal(N).astype(np.float32) if hasattr(rng, "standard_normal") else np.random.randn(N).astype(np.float32)
            if self.antithetic:
                # average a draw and its antithetic twin → lower variance without bias
                samples1 = mean + z * std
                samples2 = mean - z * std
                return 0.5 * (samples1 + samples2)
            else:
                return mean + z * std
        except Exception:
            return None

    # ------- GP posterior sampling -------
    def _gp_ts_from_pool(self, candidates, surrogate_manager, rng) -> Optional[np.ndarray]:
        """
        Use the model's posterior sampler if available.
        Expect surrogate_manager.gp_posterior_samples(X, n_samples) -> (n_samples, N) or (N,)
        """
        try:
            s = None
            # prefer a reparameterized sampler if exposed
            if hasattr(surrogate_manager, "gp_posterior_rsample"):
                # many libs accept a random seed / base_noise; pass antithetic noise if asked
                s = surrogate_manager.gp_posterior_rsample(candidates, n_samples=1, rng=rng)
            else:
                s = surrogate_manager.gp_posterior_samples(candidates, n_samples=1)
            if s is None:
                return None
            s = np.asarray(s)
            if s.ndim == 2 and s.shape[0] == 1:
                s = s[0]
            return s.squeeze()
        except Exception:
            return None


# ----------------------- RF TS helper ----------------------------------------
class RandomForestSampler:
    """
    Tiny wrapper around sklearn RandomForestRegressor for TS.
    Provides:
      - maybe_fit: incremental refit guard
      - per_tree_predict: fast per-tree predictions for variance estimate
      - std_floor: optional exploration floor (set to 0.0 to disable)
    """
    def __init__(self, std_floor: float = 0.0):
        self.model = None
        self.is_fitted = False
        self.last_data_size = 0
        self.std_floor = float(std_floor)

    def maybe_fit(self, X, y, force=False):
        if not HAS_SKLEARN or len(X) < 8:
            return False
        if (not force and self.is_fitted and
            len(X) - self.last_data_size < max(5, int(len(X) * 0.1))):
            return True
        try:
            self.model = RandomForestRegressor(
                n_estimators=64,          # a few more trees helps TS variance
                max_depth=10,
                n_jobs=1,
                random_state=42,
                bootstrap=True,
                max_features=0.5,
                min_samples_leaf=2,
            )
            Xd = np.asarray(X, dtype=np.float32)
            yd = np.asarray(y, dtype=np.float32)
            self.model.fit(Xd, yd)
            self.is_fitted = True
            self.last_data_size = len(X)
            return True
        except Exception:
            self.is_fitted = False
            return False

    def per_tree_predict(self, X_query) -> Optional[np.ndarray]:
        if not self.is_fitted:
            return None
        try:
            Xq = np.asarray(X_query, dtype=np.float32)
            # shape: (n_trees, N)
            return np.array([t.predict(Xq) for t in self.model.estimators_], dtype=np.float32)
        except Exception:
            return None


class UCB(Acquisition):
    """Upper Confidence Bound acquisition function.
    
    Balances exploration and exploitation using mean + beta * std.
    Often more robust than EI, especially in early optimization stages.
    """
    name = "ucb"

    def propose(self, region, bounds, rng, surrogate_manager):
        man = self.manager
        whitener, r_white = man._get_whitened_TR(region, surrogate_manager)
        dim = bounds.shape[0]
        n_candidates = min(600, max(120, 30 * dim))
        C = man._candidate_pool(whitener, r_white, region.center, bounds, n_candidates, rng, region)
        
        # Optional polish for periodic problems
        if getattr(man.config, "is_periodic_problem", False):
            lb, ub = bounds[:, 0], bounds[:, 1]
            K = min(8, len(C))
            idx = np.random.default_rng().choice(len(C), size=K, replace=False)
            # Use a simple UCB-based polish instead of EI
            for i in idx:
                C[i] = self._axial_ucb_polish(C[i], lb, ub, surrogate_manager, iters=2)

        mean, std = surrogate_manager.predict_global_cached(C)
        mean = mean.astype(np.float64)
        std = np.maximum(std.astype(np.float64), 1e-12)
        
        # Compute adaptive beta
        beta = self._compute_beta(man.iteration, dim, man.get_progress())
        
        # UCB = mean + beta * std (assuming maximization)
        # For minimization problems, use -mean + beta * std
        ucb_scores = mean + beta * std
        
        wdim = 1.0 / (np.diag(whitener.L) ** 2)
        return man._select_diverse(C, ucb_scores, minimize=False, metric_scales=wdim)
    
    def _compute_beta(self, iteration, dim, progress):
        """Compute adaptive confidence parameter beta.
        
        Args:
            iteration: Current optimization iteration
            dim: Problem dimensionality  
            progress: Optimization progress [0, 1]
            
        Returns:
            Beta parameter for confidence bound
        """
        # Get config parameters
        base_beta = float(getattr(self.manager.config, "ucb_base_beta", 2.0))
        decay_rate = float(getattr(self.manager.config, "ucb_decay_rate", 0.5))
        min_beta = float(getattr(self.manager.config, "ucb_min_beta", 0.5))
        
        # Theoretical: beta_t = sqrt(2 * log(t^(d/2 + 2) * pi^2 / (3 * delta)))
        # Practical adaptive version:
        t = max(iteration, 1)
        
        # Base theoretical component
        log_term = np.log(t**(dim/2 + 2) * np.pi**2 / 3)
        theoretical_beta = np.sqrt(2 * log_term)
        
        # Adaptive decay based on progress
        adaptive_beta = base_beta * np.exp(-decay_rate * progress)
        
        # Combine and ensure minimum
        beta = max(min_beta, min(theoretical_beta, adaptive_beta))
        
        # Boost exploration if stagnating
        if self.manager.stagnation_counter > 5:
            beta *= 1.3
            
        return beta
    
    def _axial_ucb_polish(self, x, lb, ub, surrogate_manager, iters=2):
        """Axial coordinate optimization using UCB criterion."""
        x = x.copy()
        D = x.size
        beta = 1.0  # Simple fixed beta for polish
        
        for _ in range(iters):
            for d in range(D):
                a, b = lb[d], ub[d]
                m1 = a + (b - a) / 3.0
                m2 = a + 2.0 * (b - a) / 3.0
                xs = np.array([
                    np.concatenate([x[:d], [m1], x[d+1:]]),
                    np.concatenate([x[:d], [m2], x[d+1:]])
                ], dtype=np.float64)
                mean, std = surrogate_manager.predict_global_cached(xs)
                ucb = mean + beta * np.maximum(std, 1e-12)
                x[d] = m1 if ucb[0] > ucb[1] else m2
        return x


class LogEI(Acquisition):
    """Logarithmic Expected Improvement acquisition function.
    
    More numerically stable than standard EI, especially for high-dimensional
    problems or when EI values become very small.
    """
    name = "logei"

    def propose(self, region, bounds, rng, surrogate_manager):
        man = self.manager
        whitener, r_white = man._get_whitened_TR(region, surrogate_manager)
        dim = bounds.shape[0]
        n_candidates = min(600, max(120, 30 * dim))
        C = man._candidate_pool(whitener, r_white, region.center, bounds, n_candidates, rng, region)
        best_f = float(getattr(region, "best_value", 0.0))

        # Optional polish for periodic problems
        if getattr(man.config, "is_periodic_problem", False):
            lb, ub = bounds[:, 0], bounds[:, 1]
            K = min(8, len(C))
            idx = np.random.default_rng().choice(len(C), size=K, replace=False)
            for i in idx:
                C[i] = self._axial_logei_polish(C[i], lb, ub, surrogate_manager, best_f, iters=2)

        mean, std = surrogate_manager.predict_global_cached(C)
        mean = mean.astype(np.float64)
        std = np.maximum(std.astype(np.float64), 1e-12)
        
        # Compute log EI
        logei = self._compute_log_expected_improvement(mean, std, best_f)
        
        wdim = 1.0 / (np.diag(whitener.L) ** 2)
        return man._select_diverse(C, logei, minimize=False, metric_scales=wdim)
    
    def _compute_log_expected_improvement(self, mean, std, best_f):
        """Compute log(EI) in a numerically stable way.
        
        Args:
            mean: Predicted means
            std: Predicted standard deviations  
            best_f: Current best function value
            
        Returns:
            Log expected improvement values
        """
        # Standardized improvement
        z = (mean - best_f) / std
        
        # For numerical stability, handle different regimes of z
        logei = np.full_like(z, -np.inf)
        
        # Regime 1: z is very negative (EI ≈ 0)
        # log(EI) ≈ log(std) + z - 0.5*z^2 (for z << 0)
        very_neg = z < -6.0
        if np.any(very_neg):
            z_neg = z[very_neg]
            logei[very_neg] = (np.log(std[very_neg]) + z_neg - 0.5 * z_neg**2)
        
        # Regime 2: z is moderately negative 
        mod_neg = (z >= -6.0) & (z < -1e-6)
        if np.any(mod_neg):
            z_mod = z[mod_neg]
            std_mod = std[mod_neg]
            # Use log-sum-exp trick: log(a*Phi(z) + b*phi(z))
            log_phi = -0.5 * z_mod**2 - 0.5 * np.log(2 * np.pi)
            log_Phi = self._log_normal_cdf(z_mod)
            
            # log(std * (z * Phi(z) + phi(z)))
            term1 = log_Phi + np.log(np.abs(z_mod) + 1e-12)  # z * Phi(z) term
            term2 = log_phi  # phi(z) term
            
            # Stable log-sum-exp
            max_term = np.maximum(term1, term2)
            logei[mod_neg] = np.log(std_mod) + max_term + np.log(
                np.exp(term1 - max_term) + np.exp(term2 - max_term)
            )
        
        # Regime 3: z is small positive (standard computation)
        small_pos = (z >= -1e-6) & (z <= 5.0)
        if np.any(small_pos):
            z_pos = z[small_pos]
            std_pos = std[small_pos]
            
            # Standard EI computation, then log
            from scipy.stats import norm
            ei = std_pos * (z_pos * norm.cdf(z_pos) + norm.pdf(z_pos))
            logei[small_pos] = np.log(np.maximum(ei, 1e-100))
        
        # Regime 4: z is very positive (EI ≈ (mean - best_f))
        very_pos = z > 5.0
        if np.any(very_pos):
            logei[very_pos] = np.log(mean[very_pos] - best_f)
            
        return logei
    
    def _log_normal_cdf(self, x):
        """Compute log(Phi(x)) in a numerically stable way."""
        # For x >= -1, use log(Phi(x)) directly
        direct = x >= -1.0
        result = np.full_like(x, -np.inf)
        
        if np.any(direct):
            from scipy.stats import norm
            result[direct] = np.log(norm.cdf(x[direct]))
        
        # For x < -1, use asymptotic expansion: log(Phi(x)) ≈ -0.5*x^2 - 0.5*log(2π) + log(-x)
        asymp = x < -1.0
        if np.any(asymp):
            x_asymp = x[asymp]
            result[asymp] = (-0.5 * x_asymp**2 - 0.5 * np.log(2 * np.pi) + 
                           np.log(-x_asymp + np.sqrt(x_asymp**2 + 2)))
            
        return result
    
    def _axial_logei_polish(self, x, lb, ub, surrogate_manager, best_f, iters=2):
        """Axial coordinate optimization using LogEI criterion."""
        x = x.copy()
        D = x.size
        
        for _ in range(iters):
            for d in range(D):
                a, b = lb[d], ub[d]
                m1 = a + (b - a) / 3.0
                m2 = a + 2.0 * (b - a) / 3.0
                xs = np.array([
                    np.concatenate([x[:d], [m1], x[d+1:]]),
                    np.concatenate([x[:d], [m2], x[d+1:]])
                ], dtype=np.float64)
                mean, std = surrogate_manager.predict_global_cached(xs)
                logei = self._compute_log_expected_improvement(
                    mean.astype(np.float64),
                    np.maximum(std, 1e-12).astype(np.float64),
                    best_f
                )
                x[d] = m1 if logei[0] > logei[1] else m2
        return x


class qNIPES(Acquisition):
    """Batch Noisy Implicit Point Evaluation Search.
    
    Information-theoretic acquisition function that learns about the optimal 
    inputs by maximizing mutual information, with robust noise handling.
    Similar to PES but designed for noisy observations and batch evaluation.
    """
    name = "qnipes"

    def __init__(self, manager):
        super().__init__(manager)
        self._optimal_cache = {}
        self._last_cache_update = -1

    def propose(self, region, bounds, rng, surrogate_manager):
        man = self.manager
        whitener, r_white = man._get_whitened_TR(region, surrogate_manager)
        dim = bounds.shape[0]
        n_candidates = min(700, max(200, 35 * dim))
        C = man._candidate_pool(whitener, r_white, region.center, bounds, n_candidates, rng, region)

        # Get optimal samples for information-theoretic computation
        optimal_inputs, optimal_outputs = self._get_optimal_samples(
            surrogate_manager, bounds, rng, n_samples=16
        )
        
        if optimal_inputs is None or len(optimal_inputs) == 0:
            # Fallback to LogEI if optimal sampling fails
            best_f = float(getattr(region, "best_value", 0.0))
            mean, std = surrogate_manager.predict_global_cached(C)
            logei = self._compute_log_expected_improvement(
                mean.astype(np.float64),
                np.maximum(std, 1e-12).astype(np.float64), 
                best_f
            )
            wdim = 1.0 / (np.diag(whitener.L) ** 2)
            return man._select_diverse(C, logei, minimize=False, metric_scales=wdim)

        # Compute qNIPES acquisition values
        acquisition_values = self._compute_qnipes(
            C, optimal_inputs, optimal_outputs, surrogate_manager
        )
        
        wdim = 1.0 / (np.diag(whitener.L) ** 2)
        return man._select_diverse(C, acquisition_values, minimize=False, metric_scales=wdim)
    
    def _get_optimal_samples(self, surrogate_manager, bounds, rng, n_samples=16):
        """Generate samples from the optimal set using model posterior."""
        # Cache optimal samples to avoid recomputation
        iteration = getattr(self.manager, 'iteration', 0)
        if (iteration == self._last_cache_update and 
            iteration in self._optimal_cache):
            return self._optimal_cache[iteration]
            
        try:
            # Strategy 1: Try to get GP posterior samples if available
            if hasattr(surrogate_manager, 'gp_posterior_samples'):
                # Generate many candidates and select best ones
                n_search = min(2000, max(500, 100 * bounds.shape[0]))
                X_search = rng.uniform(
                    bounds[:, 0], bounds[:, 1], 
                    size=(n_search, bounds.shape[0])
                ).astype(np.float64)
                
                # Get posterior samples
                samples = surrogate_manager.gp_posterior_samples(X_search, n_samples=5)
                if samples is not None and samples.size > 0:
                    # For each posterior sample, find the best points
                    optimal_inputs = []
                    optimal_outputs = []
                    
                    for i in range(samples.shape[0]):  # iterate over posterior samples
                        sample_vals = samples[i] if samples.ndim > 1 else samples
                        # Select top candidates from this posterior sample
                        n_top = max(1, n_samples // 5)
                        top_idx = np.argpartition(-sample_vals, n_top-1)[:n_top]
                        optimal_inputs.extend(X_search[top_idx])
                        optimal_outputs.extend(sample_vals[top_idx])
                    
                    optimal_inputs = np.array(optimal_inputs[:n_samples])
                    optimal_outputs = np.array(optimal_outputs[:n_samples])
                    
                    self._optimal_cache[iteration] = (optimal_inputs, optimal_outputs)
                    self._last_cache_update = iteration
                    return optimal_inputs, optimal_outputs
            
            # Strategy 2: Fallback to mean-based optimization with noise
            n_search = min(1500, max(400, 80 * bounds.shape[0]))
            X_search = rng.uniform(
                bounds[:, 0], bounds[:, 1],
                size=(n_search, bounds.shape[0])
            ).astype(np.float64)
            
            mean_pred, std_pred = surrogate_manager.predict_global_cached(X_search)
            
            # Add noise for robustness (key difference from standard PES)
            noise_scale = 0.1 * np.std(std_pred)
            noisy_mean = mean_pred + rng.normal(0, noise_scale, size=mean_pred.shape)
            
            # Select top candidates with some diversity
            top_idx = np.argpartition(-noisy_mean, n_samples-1)[:n_samples]
            optimal_inputs = X_search[top_idx]
            optimal_outputs = noisy_mean[top_idx]
            
            self._optimal_cache[iteration] = (optimal_inputs, optimal_outputs)
            self._last_cache_update = iteration
            return optimal_inputs, optimal_outputs
            
        except Exception:
            return None, None
    
    def _compute_qnipes(self, candidates, optimal_inputs, optimal_outputs, surrogate_manager):
        """Compute batch NIPES acquisition values using mutual information."""
        n_candidates = len(candidates)
        
        try:
            # Predict at candidate points
            mean_cand, std_cand = surrogate_manager.predict_global_cached(candidates)
            mean_cand = mean_cand.astype(np.float64)
            std_cand = np.maximum(std_cand.astype(np.float64), 1e-12)
            
            # Mutual information approximation for batch case
            acquisition_vals = np.zeros(n_candidates)
            
            # Noise-robust entropy computation
            noise_var = float(getattr(self.manager.config, "observation_noise_var", 0.01))
            total_var = std_cand**2 + noise_var
            
            for i in range(n_candidates):
                # Entropy of predictive distribution (with noise)
                h_pred = 0.5 * np.log(2 * np.pi * np.e * total_var[i])
                
                # Expected entropy given optimal set (approximation)
                # This approximates the conditional entropy H[y|x,X*]
                distances_to_optimals = np.array([
                    np.linalg.norm(candidates[i] - opt_x) 
                    for opt_x in optimal_inputs
                ])
                
                # Weight by proximity to optimal points (inverse distance weighting)
                weights = 1.0 / (distances_to_optimals + 1e-6)
                weights = weights / np.sum(weights)
                
                # Expected conditional entropy (heuristic approximation)
                min_distance = np.min(distances_to_optimals)
                conditional_entropy_factor = np.exp(-min_distance / np.mean(distances_to_optimals))
                h_conditional = h_pred * conditional_entropy_factor
                
                # Information gain (mutual information approximation)
                info_gain = h_pred - h_conditional
                
                # Add exploration bonus based on model uncertainty
                exploration_bonus = 0.5 * np.log(std_cand[i])
                
                acquisition_vals[i] = info_gain + exploration_bonus
            
            # Normalize for numerical stability
            if np.std(acquisition_vals) > 1e-12:
                acquisition_vals = (acquisition_vals - np.mean(acquisition_vals)) / np.std(acquisition_vals)
            
            return acquisition_vals
            
        except Exception:
            # Ultra-safe fallback to uncertainty-based acquisition
            _, std_cand = surrogate_manager.predict_global_cached(candidates)
            return np.maximum(std_cand, 1e-12)
    
    def _compute_log_expected_improvement(self, mean, std, best_f):
        """Fallback LogEI computation (reuse from LogEI class logic)."""
        z = (mean - best_f) / std
        logei = np.full_like(z, -np.inf)
        
        # Simple stable computation for fallback
        finite_mask = np.isfinite(z) & (std > 1e-12)
        if np.any(finite_mask):
            z_safe = z[finite_mask] 
            std_safe = std[finite_mask]
            
            # Use a simple approximation for speed
            logei[finite_mask] = np.log(std_safe) + np.maximum(z_safe, -10.0) - 0.5 * np.maximum(z_safe, 0.0)**2
            
        return logei



# ----------------------- Manager & registry ----------------------------------
class AcquisitionCandidateManager:
    """Holds registry + common utilities; routes to selected acquisition."""
    REGISTRY: Dict[str, Type[Acquisition]] = {}

    def __init__(self, config):
        self.config = config
        self.iteration = 0
        self.stagnation_counter = 0

        self.batch_size = int(getattr(config, "batch_size", 4))
        self.max_evals  = int(getattr(config, "max_evals", 1000))

        # selection knobs
        self.mode = getattr(config, "acquisition_mode", "adaptive")  # "ei", "qei", "ts", "adaptive"
        self.thompson_probability = float(getattr(config, "thompson_probability", 0.4))

        # bookkeeping / misc
        self.recent_improvements: List[float] = []
        self.sobol_cache: Dict[int, np.ndarray] = {}

        # instantiate strategies (lazy on first use is also fine)
        self._strategies: Dict[str, Acquisition] = {}

    # ---- context ----
    def set_context(self, iteration: int, stagnation_counter: int):
        self.iteration = iteration
        self.stagnation_counter = stagnation_counter

    def get_progress(self):
        return min(1.0, self.iteration / self.max_evals)

    # ---- public API ----
    def generate_candidates(self, bounds: np.ndarray, rng, active_regions, surrogate_manager) -> np.ndarray:
        if not active_regions:
            return self._generate_sobol_standalone(bounds, rng)
        region = active_regions[0] if len(active_regions) == 1 else self._select_best_region(active_regions)
        acq = self._pick_strategy(surrogate_manager)
        print(f"Using acquisition strategy: {acq.name}")
        return acq.propose(region, bounds, rng, surrogate_manager)

    def _pick_strategy(self, surrogate_manager) -> Acquisition:
        """Enhanced strategy selection with UCB, LogEI, and qNIPES support."""
        # explicit mode
        if self.mode in self.REGISTRY:
            return self._get_strategy(self.mode)
            
        # adaptive mode with expanded options
        p_ts = self.thompson_probability
        p_ucb = float(getattr(self.config, "ucb_probability", 0.20))
        p_logei = float(getattr(self.config, "logei_probability", 0.20))
        p_qnipes = float(getattr(self.config, "qnipes_probability", 0.15))
        
        # Early optimization: prefer exploration and information-theoretic methods
        if self.get_progress() < 0.2:
            p_ts *= 1.4
            p_ucb *= 1.2
            p_qnipes *= 1.3  # qNIPES good for early exploration
        # Late optimization: prefer exploitation  
        elif self.get_progress() > 0.8:
            p_ts *= 0.7
            p_logei *= 1.4
            p_qnipes *= 0.8  # Less information-theoretic in late stages
            
        # Stagnation: boost exploration and information-theoretic approaches
        if self.stagnation_counter > 5:
            p_ts *= 1.3
            p_ucb *= 1.4
            p_qnipes *= 1.2  # Can help escape local optima
            
        # Few data points: prefer robust methods
        n_data = getattr(surrogate_manager, "global_X", None)
        if n_data is not None and len(n_data) < 15:
            p_ts *= 0.6
            p_ucb *= 1.5
            p_logei *= 1.3
            p_qnipes *= 0.7  # Needs reasonable amount of data
        # Many data points: information-theoretic methods work better
        elif n_data is not None and len(n_data) > 50:
            p_qnipes *= 1.2
            
        # Normalize probabilities
        total_prob = p_ts + p_ucb + p_logei + p_qnipes
        remaining_prob = max(0.0, 1.0 - total_prob)
        
        # Random selection
        r = np.random.random()
        
        if r < p_ts:
            return self._get_strategy("ts")
        elif r < p_ts + p_ucb:
            return self._get_strategy("ucb") 
        elif r < p_ts + p_ucb + p_logei:
            return self._get_strategy("logei")
        elif r < p_ts + p_ucb + p_logei + p_qnipes:
            return self._get_strategy("qnipes")
        else:
            # Fall back to qEI for batch settings, LogEI for single point
            # if self.batch_size > 1:
            #     return self._get_strategy("qei")
            return self._get_strategy("logei")
        
    def _get_strategy(self, name: str) -> Acquisition:
        if name not in self._strategies:
            cls = self.REGISTRY.get(name)
            if cls is None:
                raise KeyError(f"Acquisition '{name}' is not registered.")
            self._strategies[name] = cls(self)
        return self._strategies[name]

    # ---- region pick ----
    def _select_best_region(self, regions):
        best_r, best_s = regions[0], -1e300
        vals = np.array([getattr(r, "best_value", np.inf) for r in regions], dtype=float)
        ref = np.nanmedian(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 0.0
        for r in regions:
            health = float(getattr(r, "health_score", 0.5))
            radius = float(getattr(r, "radius", 1.0))
            radius_bonus = 1.2 if 0.1 < radius < 2.0 else 0.8
            bv = float(getattr(r, "best_value", np.inf))
            improvement = max(ref - bv, 0.0)
            s = health * radius_bonus * (1.0 + improvement)
            if s > best_s:
                best_s, best_r = s, r
        return best_r

    # ---- TR whitening ----
    def _get_whitened_TR(self, region, surrogate_manager):
        dim = len(region.center)
        ls = self._extract_lengthscales(surrogate_manager, dim).astype(np.float64)
        med = np.median(ls) if np.isfinite(ls).all() else 1.0
        ls = ls / (med + 1e-12)
        return _Whitener(ls), float(getattr(region, "radius", 1.0))

    def _extract_lengthscales(self, surrogate_manager, dim):
        if hasattr(surrogate_manager, "global_model") and surrogate_manager.global_model is not None:
            try:
                model = surrogate_manager.global_model
                if hasattr(model, "covar_module") and hasattr(model.covar_module, "base_kernel"):
                    ls = model.covar_module.base_kernel.lengthscale
                    if hasattr(ls, "detach"):
                        ls_np = ls.detach().cpu().numpy().reshape(-1)
                        if ls_np.size == 1:
                            return np.full(dim, float(ls_np[0]), dtype=np.float32)
                        return (ls_np[:dim] if len(ls_np) >= dim
                                else np.full(dim, float(ls_np[0]), dtype=np.float32))
            except Exception:
                pass
        return np.ones(dim, dtype=np.float32)

    # ---- candidate pool & polishing ----
    def _candidate_pool(self, whitener, r_white, center, bounds, n_total, rng, region=None):
        d = len(center)
        n_ball = int(n_total * 0.60)
        n_loc  = int(n_total * 0.25)
        n_axis = n_total - n_ball - n_loc

        Xw = _unit_ball(n_ball, d, rng) * r_white
        if n_loc > 0:
            Xw = np.vstack([Xw, rng.normal(0.0, 0.35, size=(n_loc, d)) * r_white])

        if n_axis > 0:
            k_per = 1 if d <= 4 else (2 if d <= 12 else 3)
            alpha = 0.6 * r_white
            H = np.zeros((n_axis, d), dtype=np.float64)
            for i in range(n_axis):
                idx = rng.choice(d, size=k_per, replace=False)
                sgn = rng.choice((-1.0, 1.0), size=k_per)
                H[i, idx] = sgn * alpha
            Xw = np.vstack([Xw, H])

        jitter = (0.08 / max(1.0, np.sqrt(d))) * r_white
        Xw = Xw + jitter * np.sin(2.0 * np.pi * rng.random(size=Xw.shape))

        long = r_white
        lat  = r_white
        u = None
        if region is not None and hasattr(region, "prev_dir_white") and region.prev_dir_white is not None:
            u = np.asarray(region.prev_dir_white, dtype=np.float64).reshape(-1)
            nu = np.linalg.norm(u)
            if nu > 1e-12:
                u = u / nu
                long = float(getattr(region, "radius_long", r_white))
                lat  = float(getattr(region, "radius_lat",  r_white))
                a = Xw @ u
                ortho = Xw - np.outer(a, u)
                on = np.linalg.norm(ortho, axis=1, keepdims=True) + 1e-12
                Xw = np.outer(a * long, u) + (lat / on) * ortho

        if u is not None:
            a = Xw @ u
            ortho = Xw - np.outer(a, u)
            q = (a / max(long, 1e-12))**2 + (np.linalg.norm(ortho, axis=1) / max(lat, 1e-12))**2
            mask = q > 1.0
            if np.any(mask):
                scale = 1.0 / np.sqrt(q[mask])
                Xw[mask] *= scale[:, None]
        else:
            nr = np.linalg.norm(Xw, axis=1)
            mask = nr > r_white
            if np.any(mask):
                Xw[mask] *= (r_white / (nr[mask] + 1e-12))[:, None]

        L_diag = np.diag(whitener.L)
        X = Xw * L_diag[None, :] + center[None, :]
        lo, hi = bounds[:, 0], bounds[:, 1]
        np.clip(X, lo, hi, out=X)

        if getattr(self.config, "is_periodic_problem", False) and X.shape[0] >= 64:
            take = min(64, X.shape[0] // 4)
            idx  = np.random.default_rng().choice(X.shape[0], size=take, replace=False)
            X = np.vstack([
                X,
                _phase_jitter(
                    X[idx], bounds,
                    phases=getattr(self.config, "period_phases", (0.0, 0.25, 0.5, 0.75)),
                    strength=getattr(self.config, "phase_jitter_strength", 0.30),
                )
            ])
        return X

    def _axial_ei_polish(self, x, lb, ub, surrogate_manager, best_f, iters=2):
        x = x.copy()
        D = x.size
        for _ in range(iters):
            for d in range(D):
                a, b = lb[d], ub[d]
                m1 = a + (b - a) / 3.0
                m2 = a + 2.0 * (b - a) / 3.0
                xs = np.array([
                    np.concatenate([x[:d], [m1], x[d+1:]]),
                    np.concatenate([x[:d], [m2], x[d+1:]])
                ], dtype=np.float64)
                mean, std = surrogate_manager.predict_global_cached(xs)
                ei = compute_expected_improvement(mean.astype(np.float64),
                                                  np.maximum(std, 1e-12).astype(np.float64),
                                                  float(best_f))
                x[d] = m1 if ei[0] > ei[1] else m2
        return x

    # ---- diversity selector (shared) ----
    def _select_diverse(self, candidates, scores, minimize=False, metric_scales=None):
        if len(candidates) <= self.batch_size:
            return candidates
        sel_scores = (-scores if minimize else scores).astype(np.float64)
        s = sel_scores - np.min(sel_scores)
        s = s / (np.max(s) + 1e-12)
        s = 0.1 + 0.9 * s

        wdim = np.asarray(metric_scales, dtype=np.float64) if metric_scales is not None else None

        if HAS_NUMBA and wdim is not None:
            try:
                idx = _weighted_kpp_indices_metric(candidates.astype(np.float64), s, self.batch_size, wdim)
                return candidates[idx]
            except Exception:
                pass

        M = min(self.batch_size * 4, len(candidates))
        top = np.argpartition(-sel_scores, M - 1)[:M]
        C = candidates[top]
        picked = [int(np.argmax(sel_scores[top]))]
        while len(picked) < self.batch_size and len(picked) < len(C):
            if wdim is None:
                dmin = np.full(len(C), np.inf)
                for i in range(len(C)):
                    for j in picked:
                        d = np.linalg.norm(C[i] - C[j])
                        if d < dmin[i]:
                            dmin[i] = d
            else:
                dmin = _pairwise_mahal_min_d2(C, picked, wdim)
            picked.append(int(np.argmax(dmin)))
        return C[picked]

    # ---- exploration fallback ----
    def _generate_sobol_standalone(self, bounds, rng):
        dim = bounds.shape[0]
        try:
            skip = (self.iteration * 31) % 97
            seq = i4_sobol_generate(dim, self.batch_size + skip)[-self.batch_size:]
            return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * seq.T
        except Exception:
            return rng.uniform(bounds[:, 0], bounds[:, 1], (self.batch_size, dim))

    # ---- notifications ----
    def notify_iteration_result(self, improvement: float,
                                region: Optional[object] = None,
                                region_id: Optional[int] = None,
                                x_new: Optional[np.ndarray] = None,
                                surrogate_manager=None,
                                acquisition_used: Optional[str] = None):
        self.recent_improvements.append(float(improvement))
        if len(self.recent_improvements) > 20:
            self.recent_improvements.pop(0)
        self.stagnation_counter = (self.stagnation_counter + 1) if improvement <= 1e-9 else 0

        if region is None and region_id is not None and hasattr(self, "_region_lookup"):
            region = self._region_lookup(region_id)
        if region is None or x_new is None or surrogate_manager is None:
            return

        try:
            whitener, _ = self._get_whitened_TR(region, surrogate_manager)
        except Exception:
            return

        try:
            x_new = np.asarray(x_new, dtype=np.float64).reshape(-1)
            step_white = whitener.to_white(x_new, region.center)
            if float(improvement) > 1e-12 and hasattr(region, "update_direction"):
                region.update_direction(step_white)
            elif hasattr(region, "decay_direction"):
                region.decay_direction()
        except Exception:
            pass

    # ---- info ----
    def get_info(self):
        return {
            "iteration": self.iteration,
            "progress": self.get_progress(),
            "stagnation": self.stagnation_counter,
            "batch_size": self.batch_size,
            "mode": self.mode,
            "has_numba": HAS_NUMBA,
            "has_sklearn": HAS_SKLEARN,
        }

# ----------------------- Register built-ins -----------------------------------
#EI.register(AcquisitionCandidateManager.REGISTRY)
#QEI.register(AcquisitionCandidateManager.REGISTRY)
ThompsonSampling.register(AcquisitionCandidateManager.REGISTRY)
UCB.register(AcquisitionCandidateManager.REGISTRY)
LogEI.register(AcquisitionCandidateManager.REGISTRY)
qNIPES.register(AcquisitionCandidateManager.REGISTRY)
