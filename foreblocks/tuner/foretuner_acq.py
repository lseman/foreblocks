# ============================================
# ✅ Numerical & Scientific Computing
# ============================================
import numpy as np
from sobol_seq import i4_sobol_generate

# ============================================
# ✅ Project-Specific
# ============================================
from .foretuner_aux import (
    expected_improvement,
    knowledge_gradient,
    log_expected_improvement,
    predictive_entropy_search,
    probability_improvement,
    upper_confidence_bound,
)


# ============================================
# ✅ Acquisition Manager (as provided; behavior preserved)
# ============================================
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
            "ei": exploit_w * (1.0 - 0.3 * entropy_level),
            "ucb": 0.3 * explore_w * stagnation_factor,
            "kg": 0.2 * exploit_w,
            "ts": 0.4 * (1.0 - self._progress())
            + (0.3 if self.stagnation_counter > 5 else 0.0),
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

        if (
            region.local_y is not None
            and len(region.local_y) >= self.config.min_local_samples
        ):
            X_local = np.array(region.local_X, copy=False)
            y_local = np.array(region.local_y, copy=False)
            mean, std = surrogate_manager.predict_local(
                candidates, X_local, y_local, region.radius
            )
        else:
            mean, std = surrogate_manager.predict_global_cached(candidates)

        ts_samples = surrogate_manager.gp_posterior_samples(candidates, n_samples=3)
        ts_score = np.min(ts_samples, axis=0)

        acq_scores, _ = self.compute_scores(mean, std, region.best_value, ts_score)
        acq_scores += 0.05 * std * region.exploration_bonus

        top_idx = np.argsort(-acq_scores)[: self.config.batch_size]
        return self._refine_candidate(
            candidates[top_idx[0]], region, bounds, surrogate_manager
        )

    def _sample_region_candidates(self, region, bounds, rng):
        n = int(self.config.n_candidates * (1.1 - 0.5 * self._progress()))
        dim = region.center.shape[0]
        eigvals, eigvecs = np.linalg.eigh(region.cov)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 1e-9, None)))

        sobol = i4_sobol_generate(dim, n)
        sobol_scaled = region.center + (sobol - 0.5) @ sqrt_cov.T * region.radius
        sobol_scaled = np.clip(sobol_scaled, bounds[:, 0], bounds[:, 1])
        return sobol_scaled

    def _refine_candidate(
        self, x, region, bounds, surrogate_manager, steps=3, step_size=0.25
    ):
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

        if (
            self.acquisition_type == "adaptive_auto"
            and self.last_chosen_idx is not None
        ):
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
            "recent_rewards": self.recent_rewards.tolist(),
        }

