
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



# # ----------------------- Concrete acquisitions --------------------------------
# class EI(Acquisition):
#     name = "ei"

#     def propose(self, region, bounds, rng, surrogate_manager):
#         man = self.manager
#         whitener, r_white = man._get_whitened_TR(region, surrogate_manager)
#         dim = bounds.shape[0]
#         n_candidates = min(600, max(120, 30 * dim))
#         C = man._candidate_pool(whitener, r_white, region.center, bounds, n_candidates, rng, region)
#         best_f = float(getattr(region, "best_value", 0.0))

#         if getattr(man.config, "is_periodic_problem", False):
#             lb, ub = bounds[:, 0], bounds[:, 1]
#             K = min(8, len(C))
#             idx = np.random.default_rng().choice(len(C), size=K, replace=False)
#             for i in idx:
#                 C[i] = man._axial_ei_polish(C[i], lb, ub, surrogate_manager, best_f, iters=2)

#         mean, std = surrogate_manager.predict_global_cached(C)
#         ei = compute_expected_improvement(mean.astype(np.float64),
#                                           np.maximum(std, 1e-12).astype(np.float64),
#                                           best_f)
#         wdim = 1.0 / (np.diag(whitener.L) ** 2)
#         return man._select_diverse(C, ei, minimize=False, metric_scales=wdim)

# class QEI(Acquisition):
#     name = "qei"

#     def propose(self, region, bounds, rng, surrogate_manager):
#         man = self.manager
#         whitener, r_white = man._get_whitened_TR(region, surrogate_manager)
#         dim = bounds.shape[0]
#         n_candidates = min(800, max(240, 40 * dim))
#         C = man._candidate_pool(whitener, r_white, region.center, bounds, n_candidates, rng, region)

#         mean, std = surrogate_manager.predict_global_cached(C)
#         mean = mean.astype(np.float64)
#         std  = np.maximum(std.astype(np.float64), 1e-12)

#         best_f = float(getattr(region, "best_value", 0.0))
#         ei = compute_expected_improvement(mean, std, best_f)

#         # optional polish for periodic tasks
#         if getattr(man.config, "is_periodic_problem", False):
#             lb, ub = bounds[:, 0], bounds[:, 1]
#             K = min(8, len(C))
#             idx = np.random.default_rng().choice(len(C), size=K, replace=False)
#             for i in idx:
#                 C[i] = man._axial_ei_polish(C[i], lb, ub, surrogate_manager, best_f, iters=1)
#             mean, std = surrogate_manager.predict_global_cached(C)
#             mean = mean.astype(np.float64)
#             std  = np.maximum(std.astype(np.float64), 1e-12)
#             ei   = compute_expected_improvement(mean, std, best_f)

#         wdim = 1.0 / (np.diag(whitener.L) ** 2)
#         corr_pow = float(getattr(man.config, "qei_corr_power", 1.0))
#         kb_update = bool(getattr(man.config, "qei_kb_update", True))
#         kappa = float(getattr(man.config, "qei_kappa", 0.2))

#         # Normalize EI to [0.1, 1.0] for stable attenuation
#         adj = ei.copy()
#         if np.isfinite(adj).any():
#             mmin, mmax = float(np.min(adj)), float(np.max(adj))
#             if mmax > mmin:
#                 adj = 0.1 + 0.9 * (adj - mmin) / (mmax - mmin)
#             else:
#                 adj = np.full_like(adj, 0.5)

#         K = min(man.batch_size, len(C))
#         picked = []
#         for _ in range(K):
#             i = int(np.argmax(adj))
#             if not np.isfinite(adj[i]) or adj[i] <= 0:
#                 i = int(np.argmax(ei))
#             picked.append(i)

#             # attenuate near neighbors
#             corr = _mahal_correlation(C, C[i], wdim)
#             adj *= (1.0 - np.power(np.clip(corr, 0.0, 1.0), corr_pow))

#             # optional KB-style best update
#             if kb_update:
#                 bf_new = mean[i] - kappa * std[i]
#                 if bf_new < best_f:
#                     best_f = bf_new
#                     base = compute_expected_improvement(mean, std, best_f)
#                     if np.isfinite(base).any():
#                         bmin, bmax = float(np.min(base)), float(np.max(base))
#                         if bmax > bmin:
#                             base = 0.1 + 0.9 * (base - bmin) / (bmax - bmin)
#                         else:
#                             base = np.full_like(base, 0.5)
#                     eps = 1e-12
#                     att = np.maximum(1e-6, adj / (np.maximum(base, eps)))
#                     adj = base * att

#             adj[i] = -np.inf

#         Xsel = C[picked]

#         # optional light re-spacing via k++ on a top buffer
#         if len(Xsel) >= 2:
#             M = min(man.batch_size * 4, len(C))
#             top = np.argpartition(-ei, M - 1)[:M]
#             Cbuf = C[top]
#             scores = ei[top]
#             s = scores - np.min(scores)
#             s = 0.1 + 0.9 * s / (np.max(s) + 1e-12)
#             try:
#                 if HAS_NUMBA:
#                     idx = _weighted_kpp_indices_metric(Cbuf.astype(np.float64), s, man.batch_size, wdim)
#                     Xsel = Cbuf[idx]
#                 else:
#                     picked_idx = [int(np.argmax(s))]
#                     while len(picked_idx) < man.batch_size and len(picked_idx) < len(Cbuf):
#                         dmin = _pairwise_mahal_min_d2(Cbuf, picked_idx, wdim)
#                         picked_idx.append(int(np.argmax(dmin)))
#                     Xsel = Cbuf[picked_idx]
#             except Exception:
#                 pass

#         return Xsel

class ThompsonSampling(Acquisition):
    name = "ts"

    def __init__(self, manager):
        super().__init__(manager)
        self.rf = RandomForestSampler()

    def propose(self, region, bounds, rng, surrogate_manager):
        man = self.manager
        whitener, r_white = man._get_whitened_TR(region, surrogate_manager)
        dim = bounds.shape[0]
        n_candidates = min(800, max(200, 40 * dim))
        C = man._candidate_pool(whitener, r_white, region.center, bounds, n_candidates, rng, region)

        # RF TS
        samples = self._rf_ts_from_pool(C, surrogate_manager)
        minimize = True
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
            mean, std = surrogate_manager.predict_global_cached(C)
            samples = compute_expected_improvement(mean.astype(np.float64),
                                                   np.maximum(std, 1e-12).astype(np.float64),
                                                   best_f)
            minimize = False

        wdim = 1.0 / (np.diag(whitener.L) ** 2)
        return man._select_diverse(C, samples, minimize=minimize, metric_scales=wdim)

    def _rf_ts_from_pool(self, candidates, surrogate_manager) -> Optional[np.ndarray]:
        try:
            if not hasattr(surrogate_manager, "global_X") or surrogate_manager.global_X is None:
                return None
            Xd = surrogate_manager.global_X.detach().cpu().numpy()
            yd = surrogate_manager.global_y.detach().cpu().numpy()
            if not self.rf.maybe_fit(Xd, yd):
                return None
            return self.rf.sample_posterior(candidates)
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
