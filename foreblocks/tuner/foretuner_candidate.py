from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.spatial.distance import cdist
from sobol_seq import i4_sobol_generate

try:
    from pykdtree.kdtree import KDTree as _OriginalKDTree

    # Create a wrapper that handles pykdtree's quirks
    class KDTree(_OriginalKDTree):
        def query(self, x, k=1, **kwargs):
            x = np.asarray(x, dtype=np.float64)
            single_query = x.ndim == 1
            
            # Ensure x is 2D for pykdtree compatibility
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            distances, indices = super().query(x, k=k, **kwargs)
            
            # Handle return format for single queries to match scipy behavior
            if single_query:
                if k == 1:
                    return float(distances.flatten()[0]), int(indices.flatten()[0])
                else:
                    return distances.flatten(), indices.flatten()
            
            return distances, indices
    
    HAS_KDTREE = True
except Exception:
    try:
        from scipy.spatial import KDTree
        HAS_KDTREE = True
    except Exception:
        HAS_KDTREE = False


def _safe01(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = np.max(np.abs(v))
    return v / (vmax + 1e-12)


def _pareto_mask(F):
    # lower is better in all columns
    n = F.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        le = (F <= F[i]).all(axis=1)
        lt = (F < F[i]).any(axis=1)
        if np.any(le & lt):
            nd[i] = False
    return nd



# ============================================
# ✅ Candidate Generator (V2, wired to AcquisitionManager)
# ============================================
class CandidateGenerator:
    """
    State-of-the-art CandidateGenerator V2
    - Entropy/stagnation-aware exploration probability (cached)
    - Sobol + global-uncertainty + trust-region outward hybrid
    - Trust-region exploitation with micro-pool + Pareto selection
    - Fast KDTree k-center greedy diversity filtering (fallback to NumPy)
    - Optional parallel Thompson sampling path
    - Uses AcquisitionManager.compute_scores(...) for batch scoring
    """

    def __init__(self, config, acquisition_manager):
        self.config = config
        self.acquisition_manager = acquisition_manager
        self.iteration = 0
        self.stagnation_counter = 0
        self._cached_exploration_prob = None
        self._cache_iteration = -1

    # --------------------------------------------
    # Context updates
    # --------------------------------------------
    def set_context(self, iteration, stagnation_counter):
        self.iteration = iteration
        self.stagnation_counter = stagnation_counter
        self.acquisition_manager.set_iteration(iteration)
        self._cache_iteration = -1

    # --------------------------------------------
    # Entry
    # --------------------------------------------
    def generate_candidates(self, bounds, rng, active_regions, surrogate_manager):
        n_dims = bounds.shape[0]
        B = self.config.batch_size

        if not active_regions:
            return rng.uniform(bounds[:, 0], bounds[:, 1], size=(B, n_dims))

        if str(getattr(self.config, "acquisition", "")).lower() == "ts":
            cands = self._generate_thompson_batch(bounds, rng, active_regions, surrogate_manager)
            return self._kcenter_greedy(cands, k=B)

        return self._generate_adaptive_batch(bounds, rng, active_regions, surrogate_manager)

    # --------------------------------------------
    # Adaptive blend
    # --------------------------------------------
    def _generate_adaptive_batch(self, bounds, rng, regions, surrogate_manager):
        n_dims = bounds.shape[0]
        B = self.config.batch_size
        p_exp = self._compute_exploration_probability()
        flags = rng.random(B) < p_exp
        n_explore = int(flags.sum())
        n_exploit = B - n_explore

        chunks = []
        if n_explore > 0:
            chunks.extend(self._exploration_sampling_batch(bounds, regions, rng, n_explore, surrogate_manager))
        if n_exploit > 0:
            chunks.extend(self._exploitation_sampling_batch(bounds, regions, rng, n_exploit, surrogate_manager))

        if not chunks:
            return rng.uniform(bounds[:, 0], bounds[:, 1], size=(B, n_dims))

        # sanitize & stack
        sanitized = []
        for arr in chunks:
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[-1] != n_dims:
                arr = arr.reshape(-1, n_dims)
            if arr.size:
                sanitized.append(arr)
        if not sanitized:
            return rng.uniform(bounds[:, 0], bounds[:, 1], size=(B, n_dims))

        pool = np.vstack(sanitized)

        out = self._kcenter_greedy(pool, k=B)
        rng.shuffle(out)
        return out

    # --------------------------------------------
    # Exploration: Sobol + global-uncertainty + outward
    # --------------------------------------------
    def _exploration_sampling_batch(self, bounds, regions, rng, count, surrogate_manager):
        n_dims = bounds.shape[0]
        u = rng.random(count)
        n_sobol = int(np.sum(u < 0.33))
        n_unc   = int(np.sum((u >= 0.33) & (u < 0.66)))
        n_out   = count - n_sobol - n_unc

        chunks = []

        if n_sobol > 0:
            sob = i4_sobol_generate(n_dims, n_sobol)
            sob = bounds[:, 0] + sob * (bounds[:, 1] - bounds[:, 0])
            chunks.append(sob)

        if n_unc > 0:
            pool_size = min(800, max(100, 12 * n_unc))
            pool = rng.uniform(bounds[:, 0], bounds[:, 1], size=(pool_size, n_dims))
            _, std = surrogate_manager.predict_global_cached(pool)
            idx = np.argpartition(std, -n_unc)[-n_unc:]
            chunks.append(pool[idx])

        if n_out > 0:
            chunks.append(self._region_outward_batch(bounds, regions, rng, n_out))

        return chunks

    def _region_outward_batch(self, bounds, regions, rng, count):
        """
        Ellipsoidal outward steps along principal axes (antithetic) for better coverage.
        """
        n_dims = bounds.shape[0]
        w = np.array([max(1e-12, getattr(r, "spawn_score", 1.0) * getattr(r, "exploration_bonus", 1.0))
                      for r in regions], dtype=float)
        w /= w.sum() if w.sum() > 0 else len(regions)
        chosen = rng.choice(len(regions), size=count, p=w)
        out = np.empty((count, n_dims), dtype=float)

        for i, ridx in enumerate(chosen):
            r = regions[ridx]
            C = getattr(r, "cov", None)
            if C is None or not np.all(np.isfinite(C)):
                C = (r.radius ** 2 + 1e-9) * np.eye(n_dims)
            evals, evecs = np.linalg.eigh(0.5 * (C + C.T))
            probs = _safe01(evals) + 1e-9
            probs /= probs.sum()
            a = rng.choice(n_dims, p=probs)
            direction = evecs[:, a]
            direction /= np.linalg.norm(direction) + 1e-12
            step = r.radius * (1.5 + 0.5 * getattr(r, "exploration_bonus", 1.0))
            if rng.random() < 0.5:
                direction = -direction
            cand = r.center + step * direction
            out[i] = np.clip(cand, bounds[:, 0], bounds[:, 1])
        return out

    # --------------------------------------------
    # Exploitation: local micro-pool + Pareto
    # --------------------------------------------
    def _exploitation_sampling_batch(self, bounds, regions, rng, count, surrogate_manager):
        """
        Local exploitation:
        - health-weighted region allocation
        - per-region micro-pool = {opt seeds + ellipsoid jitter}
        - vectorized scoring via AcquisitionManager.compute_scores(...)
        - Pareto filter on [ -acq, unc, prox-to-center ] (lower is better)
        - k-center greedy downselect to n_req
        """
        n_dims = bounds.shape[0]

        # Health-weighted allocation across regions
        h = np.array([max(1e-12, getattr(r, "health_score", 1.0)) for r in regions], dtype=float)
        h /= h.sum() if h.sum() > 0 else len(regions)
        picks = rng.choice(len(regions), size=count, p=h)
        by_region = Counter(picks)

        out = []
        for ridx, n_req in by_region.items():
            r = regions[ridx]

            # No local data → global fallback
            if not getattr(r, "local_X", None) or len(r.local_X) == 0:
                out.append(rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_req, n_dims)))
                continue

            # Build a small local pool: optimizer seeds + ellipsoid jitter
            pool_size = max(12, 6 * n_req)
            pool = [self.acquisition_manager.optimize_in_region(r, bounds, rng, surrogate_manager)]
            for _ in range(min(2, n_req - 1)):
                pool.append(self.acquisition_manager.optimize_in_region(r, bounds, rng, surrogate_manager))

            # Ellipsoid jitter around center using region covariance
            C = getattr(r, "cov", None)
            if C is None or not np.all(np.isfinite(C)):
                C = (r.radius ** 2 + 1e-9) * np.eye(n_dims)
            L = np.linalg.cholesky(C + 1e-12 * np.eye(n_dims))
            z = rng.normal(size=(pool_size - len(pool), n_dims))
            jitter = r.center + (z @ L.T)
            jitter = np.clip(jitter, bounds[:, 0], bounds[:, 1])
            pool.append(jitter)

            # ---- sanitize shapes before stacking ----
            san = []
            for a in pool:
                arr = np.asarray(a)
                if arr.ndim == 0:
                    # scalar → impossible here, but guard anyway
                    arr = np.full((1, n_dims), np.nan)
                elif arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                else:
                    # collapse any leading dims to get (..., D)
                    arr = arr.reshape(-1, arr.shape[-1])
                san.append(arr)

            P = np.vstack(san)
            # Optional hard safety check on dimensionality
            if P.shape[1] != n_dims:
                raise ValueError(f"Candidate dimension mismatch: got {P.shape[1]} (expected {n_dims}).")

            # Score using AcquisitionManager (vectorized)
            scores = self._score_points_compat(P, r, surrogate_manager)
            acq = -_safe01(scores["acq"])   # maximize acquisition ⇒ minimize -acq
            unc =  _safe01(scores["unc"])

            # Proximity to center (keep points inside basin)
            d_center = np.linalg.norm(P - r.center[None, :], axis=1)
            prox = _safe01(d_center)

            # Pareto filter on [acq_cost, unc_cost, prox_cost] (lower is better)
            F = np.stack([acq, unc, prox], axis=1)
            mask = _pareto_mask(F)
            PF = P[mask]

            # Downselect if too many on front
            if len(PF) > n_req:
                PF = self._kcenter_greedy(PF, k=n_req)

            out.append(PF[:n_req])

        return out


    # --------------------------------------------
    # Thompson batch
    # --------------------------------------------
    def _generate_thompson_batch(self, bounds, rng, regions, surrogate_manager):
        n_dims = bounds.shape[0]
        per_r = max(1, self.config.n_candidates // max(1, len(regions)))
        cand_list = []
        if len(regions) > 4:
            with ThreadPoolExecutor(max_workers=min(8, len(regions))) as pool:
                futs = [pool.submit(self._sample_region_ts, r, bounds, rng, per_r) for r in regions]
                for f in as_completed(futs):
                    cand_list.extend(f.result())
        else:
            for r in regions:
                cand_list.extend(self._sample_region_ts(r, bounds, rng, per_r))

        g = max(1, self.config.n_candidates // 4)
        cand_list.append(rng.uniform(bounds[:, 0], bounds[:, 1], size=(g, n_dims)))
        C = np.vstack([np.asarray(c) for c in cand_list])
        return C.reshape(-1, n_dims)

    def _sample_region_ts(self, region, bounds, rng, count):
        return [self.acquisition_manager._sample_from_covariance(region, bounds, rng) for _ in range(count)]

    # --------------------------------------------
    # Acquisition scoring via AcquisitionManager
    # --------------------------------------------
    def _score_points_compat(self, P, region, surrogate_manager):
        # local vs global prediction: mirror optimize_in_region
        if getattr(region, "local_y", None) is not None and len(region.local_y) >= getattr(self.config, "min_local_samples", 0):
            Xl = np.array(region.local_X, copy=False)
            yl = np.array(region.local_y, copy=False)
            mean, std = surrogate_manager.predict_local(P, Xl, yl, region.radius)
        else:
            mean, std = surrogate_manager.predict_global_cached(P)

        try:
            ts_samples = surrogate_manager.gp_posterior_samples(P, n_samples=3)
            ts_score = np.min(ts_samples, axis=0)  # lower is better
        except Exception:
            ts_score = np.zeros(len(P))

        acq_scores, _ = self.acquisition_manager.compute_scores(
            mean, std, getattr(region, "best_value", np.min(mean)), ts_score
        )
        acq_scores = acq_scores + 0.05 * std * getattr(region, "exploration_bonus", 1.0)

        return {"acq": np.asarray(acq_scores, dtype=float), "unc": np.asarray(std, dtype=float)}

    # --------------------------------------------
    # Exploration probability (cached)
    # --------------------------------------------
    def _compute_exploration_probability(self):
        if self._cache_iteration == self.iteration and self._cached_exploration_prob is not None:
            return self._cached_exploration_prob

        progress = self.iteration / max(1, self.config.max_evals)
        p = float(getattr(self.config, "exploration_factor", 0.5))
        p += min(0.4, 0.03 * self.stagnation_counter)   # stagnation boost
        p *= max(0.1, 1.0 - progress)                   # progress decay
        self._cached_exploration_prob = float(np.clip(p, 0.05, 0.9))
        self._cache_iteration = self.iteration
        return self._cached_exploration_prob

    # --------------------------------------------
    # Diversity filtering (k-center greedy; KDTree fallback)
    # --------------------------------------------
    def _kcenter_greedy(self, X, k=32):
        X = np.asarray(X, dtype=float)
        n = len(X)
        if n <= k:
            return X

        if HAS_KDTREE:
            sel = [np.random.randint(n)]
            d = np.linalg.norm(X - X[sel[0]], axis=1)
            for _ in range(k - 1):
                i = int(np.argmax(d))
                sel.append(i)
                d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
            return X[sel]

        sel = [np.random.randint(n)]
        dmin = cdist(X, X[sel])[:, 0]
        for _ in range(k - 1):
            i = int(np.argmax(dmin))
            sel.append(i)
            dmin = np.minimum(dmin, cdist(X, X[i:i+1])[:, 0])
        return X[sel]
