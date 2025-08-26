# Acquisition & Candidate Manager (refactored)
# - Ellipsoidal TR via ARD lengthscales
# - Mahalanobis diversity
# - Single EI impl (exact Φ)
# - RF/GP TS with shared pool + EI fallback
# - Removed box-TR variants, LHS, duplicate selectors/optims

from typing import Optional

import numpy as np
from sobol_seq import i4_sobol_generate

# Optional deps
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import numba
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

SQRT1_2 = 1.0 / np.sqrt(2.0)
INV_SQRT_2PI = 0.39894228040143267794  # 1/sqrt(2π)

def _phi(z):  # N(0,1) pdf
    return np.exp(-0.5 * z * z) * INV_SQRT_2PI

def _Phi(z):  # exact CDF via erfc
    return 0.5 * np.erfc(-z * SQRT1_2)

# ---- EI (Numba/NumPy) -------------------------------------------------------
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
            z = (best_f - mean[i]) / s
            Phi = 0.5 * math.erfc(-z * 0.7071067811865475)  # 1/√2
            phi = math.exp(-0.5 * z * z) * 0.3989422804014327
            out[i] = s * (z * Phi + phi)
        return out
else:
    def compute_expected_improvement(mean, std, best_f):
        std = np.maximum(std, 1e-12)
        z = (best_f - mean) / std
        return std * (z * _Phi(z) + _phi(z))

# ---- Mahalanobis K-means++ (diversity) --------------------------------------
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

# ---- Whitening utils ---------------------------------------------------------
class _Whitener:
    """Diagonal whitener from ARD lengthscales (ls)."""
    def __init__(self, ls):
        ls = np.asarray(ls, dtype=float).reshape(-1)
        ls = np.clip(ls, 1e-9, 1e9)
        self.Linv = np.diag(1.0 / ls)  # whiten
        self.L    = np.diag(ls)        # unwhiten

    def to_white(self, x, c):   return self.Linv @ (x - c)
    def from_white(self, xw, c): return (self.L @ xw) + c

def _unit_ball(n, d, rng):
    v = rng.normal(size=(n, d))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    r = rng.random(n) ** (1.0 / d)
    return v * r[:, None]

# ---- RF Thompson sampler -----------------------------------------------------
class RandomForestSampler:
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.last_data_size = 0

    def maybe_fit(self, X, y, force=False):
        if not HAS_SKLEARN or len(X) < 8:
            return False
        if (not force and self.is_fitted and
            len(X) - self.last_data_size < max(5, int(len(X) * 0.1))):
            return True
        try:
            self.model = RandomForestRegressor(
                n_estimators=40, max_depth=8, n_jobs=1, random_state=42,
                bootstrap=True, max_features=0.5, min_samples_leaf=2
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

    def sample_posterior(self, X_query):
        if not self.is_fitted:
            return None
        try:
            Xq = np.asarray(X_query, dtype=np.float32)
            preds = np.array([t.predict(Xq) for t in self.model.estimators_])
            mean = preds.mean(axis=0)
            std  = preds.std(axis=0)
            noise = np.random.normal(0, 1, len(Xq))
            return mean + noise * np.maximum(std, 0.01)
        except Exception:
            return None
        
# --- add this helper near the top (below _unit_ball) -------------------------
def _phase_jitter(points: np.ndarray, bounds: np.ndarray,
                  phases=(0.0, 0.25, 0.5, 0.75), strength=0.30) -> np.ndarray:
    """
    Duplicate points nudged toward fractional phases (mod 1). `strength` in [0,1]
    controls how far we move toward the target phase (0=off, 1=full snap).
    """
    if points.size == 0:
        return points
    lo, hi = bounds[:, 0], bounds[:, 1]
    out = []
    for p in phases:
        Z = points.copy()
        frac = Z - np.floor(Z)               # fractional part
        shift = (p - frac + 1.0) % 1.0       # shortest mod-1 shift to phase p
        Z = Z + strength * shift             # partial move toward target phase
        Z = np.minimum(hi, np.maximum(lo, Z))
        out.append(Z)
    return np.vstack(out)

# ---- Manager -----------------------------------------------------------------
class AcquisitionCandidateManager:
    def __init__(self, config):
        self.config = config
        self.iteration = 0
        self.stagnation_counter = 0

        self.batch_size = int(getattr(config, "batch_size", 4))
        self.max_evals  = int(getattr(config, "max_evals", 1000))

        self.acquisition_mode = "adaptive"
        self.thompson_probability = 0.4

        self.rf_sampler = RandomForestSampler()

        self.recent_improvements = []
        self.sobol_cache = {}

    # context & bookkeeping
    def set_context(self, iteration: int, stagnation_counter: int):
        self.iteration = iteration
        self.stagnation_counter = stagnation_counter

    def get_progress(self):
        return min(1.0, self.iteration / self.max_evals)

    # ---- entry point ---------------------------------------------------------
    def generate_candidates(self, bounds: np.ndarray, rng, active_regions, surrogate_manager) -> np.ndarray:
        if not active_regions:
            return self._generate_sobol_standalone(bounds, rng)
        region = active_regions[0] if len(active_regions) == 1 else self._select_best_region(active_regions)
        return self._generate_for_region(region, bounds, rng, surrogate_manager)

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

    # ---- whitened TR ---------------------------------------------------------
    def _get_whitened_TR(self, region, surrogate_manager):
        dim = len(region.center)
        ls = self._extract_lengthscales(surrogate_manager, dim).astype(np.float64)
        med = np.median(ls) if np.isfinite(ls).all() else 1.0
        ls = ls / (med + 1e-12)
        return _Whitener(ls), float(getattr(region, "radius", 1.0))

    def _generate_for_region(self, region, bounds, rng, surrogate_manager):
        whitener, r_white = self._get_whitened_TR(region, surrogate_manager)
        use_ts = self._should_use_ts(surrogate_manager)
        if use_ts:
            return self._ts_batch(whitener, r_white, region, bounds, rng, surrogate_manager)
        else:
            return self._ei_batch(whitener, r_white, region, bounds, rng, surrogate_manager)
    
    def _candidate_pool(self, whitener, r_white, center, bounds, n_total, rng, region=None):
        d = len(center)
        n_ball = int(n_total * 0.60)
        n_loc  = int(n_total * 0.25)
        n_axis = n_total - n_ball - n_loc

        # --- base: unit-ball (whitened) + local Gaussian ---
        Xw = _unit_ball(n_ball, d, rng) * r_white
        if n_loc > 0:
            Xw = np.vstack([Xw, rng.normal(0.0, 0.35, size=(n_loc, d)) * r_white])

        # --- axial hops (1–3 coords depending on d) ---
        if n_axis > 0:
            k_per = 1 if d <= 4 else (2 if d <= 12 else 3)
            alpha = 0.6 * r_white
            H = np.zeros((n_axis, d), dtype=np.float64)
            for i in range(n_axis):
                idx = rng.choice(d, size=k_per, replace=False)
                sgn = rng.choice((-1.0, 1.0), size=k_per)
                H[i, idx] = sgn * alpha
            Xw = np.vstack([Xw, H])

        # --- small phase-like jitter (dimension-aware) ---
        jitter = (0.08 / max(1.0, np.sqrt(d))) * r_white
        Xw = Xw + jitter * np.sin(2.0 * np.pi * rng.random(size=Xw.shape))

        # --- optional elongation along last successful direction (whitened) ---
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
                # vectorized: project onto u, scale axial/orthogonal
                a = Xw @ u                     # (N,)
                ortho = Xw - np.outer(a, u)    # (N,d)
                on = np.linalg.norm(ortho, axis=1, keepdims=True) + 1e-12
                Xw = np.outer(a * long, u) + (lat / on) * ortho

        # --- ensure points are inside the ellipsoid (project if needed) ---
        if u is not None:
            a = Xw @ u
            ortho = Xw - np.outer(a, u)
            q = (a / max(long, 1e-12))**2 + (np.linalg.norm(ortho, axis=1) / max(lat, 1e-12))**2
            mask = q > 1.0
            if np.any(mask):
                scale = 1.0 / np.sqrt(q[mask])
                Xw[mask] *= scale[:, None]
        else:
            # isotropic ellipse: ||xw|| <= r_white
            nr = np.linalg.norm(Xw, axis=1)
            mask = nr > r_white
            if np.any(mask):
                Xw[mask] *= (r_white / (nr[mask] + 1e-12))[:, None]

        # --- map back & clip to box (vectorized; L is diagonal) ---
        # from_white: x = L @ xw + c
        L = whitener.L  # diagonal
        X = Xw * np.diag(L)[None, :] + center[None, :]
        lo, hi = bounds[:, 0], bounds[:, 1]
        np.clip(X, lo, hi, out=X)

        # --- optional periodic phase jitter (gated) ---
        if getattr(self.config, "is_periodic_problem", False) and X.shape[0] >= 64:
            take = min(64, X.shape[0] // 4)
            idx  = rng.choice(X.shape[0], size=take, replace=False)
            X = np.vstack([
                X,
                _phase_jitter(
                    X[idx], bounds,
                    phases=getattr(self.config, "period_phases", (0.0, 0.25, 0.5, 0.75)),
                    strength=getattr(self.config, "phase_jitter_strength", 0.30),
                    rng=rng,  # <- pass same rng
                )
            ])

        # If you want to keep pool size fixed, sample back to n_total here:
        # if X.shape[0] > n_total:
        #     keep = rng.choice(X.shape[0], size=n_total, replace=False)
        #     X = X[keep]

        return X


    def _axial_ei_polish(self, x, lb, ub, surrogate_manager, best_f, iters=2):
        x = x.copy()
        D = x.size
        for _ in range(iters):
            for d in range(D):
                a, b = lb[d], ub[d]
                m1 = a + (b - a) / 3.0
                m2 = a + 2.0 * (b - a) / 3.0
                xs = np.array([[*x[:d], m1, *x[d+1:]],
                            [*x[:d], m2, *x[d+1:]]], dtype=np.float64)
                ei = self._evaluate_ei(xs, surrogate_manager, best_f)
                x[d] = m1 if ei[0] > ei[1] else m2
        return x


    # ---- EI & TS batches -----------------------------------------------------
    def _ei_batch(self, whitener, r_white, region, bounds, rng, surrogate_manager):
        dim = bounds.shape[0]
        n_candidates = min(600, max(120, 30 * dim))
        C = self._candidate_pool(whitener, r_white, region.center, bounds, n_candidates, rng, region)
        best_f = float(getattr(region, "best_value", 0.0))
        
        if getattr(self.config, "is_periodic_problem", False):
            lb, ub = bounds[:, 0], bounds[:, 1]
            best_f = float(getattr(region, "best_value", 0.0))
            K = min(8, len(C))
            idx = np.random.default_rng().choice(len(C), size=K, replace=False)
            for i in idx:
                C[i] = self._axial_ei_polish(C[i], lb, ub, surrogate_manager, best_f, iters=2)
                
        ei = self._evaluate_ei(C, surrogate_manager, best_f)
        ei = self._evaluate_ei(C, surrogate_manager, best_f)
        wdim = 1.0 / (np.diag(whitener.L) ** 2)
        return self._select_diverse(C, ei, minimize=False, metric_scales=wdim)

    def _ts_batch(self, whitener, r_white, region, bounds, rng, surrogate_manager):
        dim = bounds.shape[0]
        n_candidates = min(800, max(200, 40 * dim))
        C = self._candidate_pool(whitener, r_white, region.center, bounds, n_candidates, rng, region)

        # RF TS
        samples = self._rf_ts_from_pool(C, surrogate_manager)
        if samples is None:
            # GP TS
            try:
                s = surrogate_manager.gp_posterior_samples(C, n_samples=1)
                samples = None if s is None or s.size == 0 else s.squeeze()
            except Exception:
                samples = None

        if samples is None:
            # EI fallback
            best_f = float(getattr(region, "best_value", 0.0))
            samples = self._evaluate_ei(C, surrogate_manager, best_f)
            minimize = False
        else:
            minimize = True

        wdim = 1.0 / (np.diag(whitener.L) ** 2)
        return self._select_diverse(C, samples, minimize=minimize, metric_scales=wdim)

    def _evaluate_ei(self, candidates, surrogate_manager, best_f):
        mean, std = surrogate_manager.predict_global_cached(candidates)
        return compute_expected_improvement(mean.astype(np.float64),
                                            std.astype(np.float64),
                                            float(best_f))

    def _rf_ts_from_pool(self, candidates, surrogate_manager) -> Optional[np.ndarray]:
        if not hasattr(surrogate_manager, "global_X") or surrogate_manager.global_X is None:
            return None
        try:
            Xd = surrogate_manager.global_X.detach().cpu().numpy()
            yd = surrogate_manager.global_y.detach().cpu().numpy()
            if not self.rf_sampler.maybe_fit(Xd, yd):
                return None
            return self.rf_sampler.sample_posterior(candidates)
        except Exception:
            return None

    # ---- strategy & utilities ------------------------------------------------
    def _should_use_ts(self, surrogate_manager):
        p = self.thompson_probability
        if self.get_progress() < 0.3:
            p *= 1.3
        if self.stagnation_counter > 5:
            p *= 1.2
        n_data = getattr(surrogate_manager, "global_X", None)
        if n_data is not None and len(n_data) < 15:
            p *= 0.6
        return np.random.random() < float(np.clip(p, 0.05, 0.8))

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

    def _select_diverse(self, candidates, scores, minimize=False, metric_scales=None):
        if len(candidates) <= self.batch_size:
            return candidates
        # normalize to nonneg weights (higher is better)
        sel_scores = (-scores if minimize else scores).astype(np.float64)
        s = sel_scores - np.min(sel_scores)
        s = s / (np.max(s) + 1e-12)
        s = 0.1 + 0.9 * s

        wdim = np.asarray(metric_scales, dtype=np.float64) if metric_scales is not None else None

        # Numba fast path
        if HAS_NUMBA and wdim is not None:
            try:
                idx = _weighted_kpp_indices_metric(candidates.astype(np.float64), s, self.batch_size, wdim)
                return candidates[idx]
            except Exception:
                pass

        # Python fallback on top-M
        M = min(self.batch_size * 4, len(candidates))
        top = np.argpartition(-sel_scores, M - 1)[:M]
        C = candidates[top]
        picked = [int(np.argmax(sel_scores[top]))]
        while len(picked) < self.batch_size:
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

    # ---- exploration fallback (no regions) -----------------------------------
    def _generate_sobol_standalone(self, bounds, rng):
        dim = bounds.shape[0]
        try:
            skip = (self.iteration * 31) % 97
            seq = i4_sobol_generate(dim, self.batch_size + skip)[-self.batch_size:]
            return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * seq.T
        except Exception:
            return rng.uniform(bounds[:, 0], bounds[:, 1], (self.batch_size, dim))

    # ---- diagnostics ----------------------------------------------------------
    def notify_iteration_result(
        self,
        improvement: float,
        region: Optional[object] = None,
        region_id: Optional[int] = None,
        x_new: Optional[np.ndarray] = None,
        surrogate_manager=None,
        acquisition_used: Optional[str] = None,
    ):
        """
        Call after evaluating a point from a specific region.

        Args:
        improvement: float > 0 if region best improved (y_old - y_new)
        region:      region object (preferred)
        region_id:   fallback if region not passed; requires set_region_lookup
        x_new:       evaluated point in ORIGINAL coordinates (np.ndarray)
        surrogate_manager: needed to rebuild whitener for direction update
        """
        # --- book-keeping (unchanged) ---
        self.recent_improvements.append(float(improvement))
        if len(self.recent_improvements) > 20:
            self.recent_improvements.pop(0)
        self.stagnation_counter = (self.stagnation_counter + 1) if improvement <= 1e-9 else 0

        # --- resolve region object ---
        if region is None and region_id is not None:
            region = getattr(self, "_region_lookup", None)(region_id) if hasattr(self, "_region_lookup") else None
        if region is None or x_new is None or surrogate_manager is None:
            return  # nothing else to do

        # --- build whitener exactly as candidate gen does ---
        try:
            whitener, _ = self._get_whitened_TR(region, surrogate_manager)
        except Exception:
            # if lengthscales unavailable, skip direction update safely
            return

        # --- compute step in whitened coords relative to region.center ---
        try:
            x_new = np.asarray(x_new, dtype=np.float64).reshape(-1)
            step_white = whitener.to_white(x_new, region.center)
            if float(improvement) > 1e-12:
                # success → reinforce ridge direction & elongation
                if hasattr(region, "update_direction"):
                    region.update_direction(step_white)
            else:
                # failure → relax elongation toward isotropy
                if hasattr(region, "decay_direction"):
                    region.decay_direction()
        except Exception:
            # never let direction maintenance break the loop
            pass

    def get_info(self):
        return {
            "iteration": self.iteration,
            "progress": self.get_progress(),
            "stagnation": self.stagnation_counter,
            "batch_size": self.batch_size,
            "rf_fitted": self.rf_sampler.is_fitted,
            "rf_data_size": self.rf_sampler.last_data_size,
            "recent_improvement_mean": (np.mean(self.recent_improvements) if self.recent_improvements else 0.0),
            "has_numba": HAS_NUMBA,
            "has_sklearn": HAS_SKLEARN,
        }
