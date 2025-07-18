import ast
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

# === Abstract Base ===


class AcquisitionStrategy(ABC):
    """Abstract base class for acquisition function strategies"""

    @abstractmethod
    def calculate(self, mean: float, std: float, best_value: float, **kwargs) -> float:
        pass

    def calculate_batch(
        self, means: np.ndarray, stds: np.ndarray, best_value: float, **kwargs
    ) -> np.ndarray:
        return np.array(
            [self.calculate(m, s, best_value, **kwargs) for m, s in zip(means, stds)]
        )

    @abstractmethod
    def get_exploration_factor(self) -> float:
        pass


# === Strategies ===


class ExpectedImprovementStrategy(AcquisitionStrategy):
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def calculate(self, mean: float, std: float, best_value: float, **kwargs) -> float:
        if std <= 1e-8:
            return 0.0

        improvement = best_value - mean - self.xi
        z = improvement / std
        return (
            improvement * norm.cdf(z) + std * norm.pdf(z)
            if improvement > 0
            else std * norm.pdf(z)
        )

    def get_exploration_factor(self) -> float:
        return 0.3


class LogExpectedImprovementStrategy(AcquisitionStrategy):
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def calculate(self, mean: float, std: float, best_value: float, **kwargs) -> float:
        if std <= 1e-8:
            return 0.0

        improvement = best_value - mean - self.xi
        z = improvement / std
        ei = (
            improvement * norm.cdf(z) + std * norm.pdf(z)
            if improvement > 0
            else std * norm.pdf(z)
        )
        return np.log1p(ei)

    def get_exploration_factor(self) -> float:
        return 0.25


class UpperConfidenceBoundStrategy(AcquisitionStrategy):
    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def calculate(self, mean: float, std: float, best_value: float, **kwargs) -> float:
        return -(mean - self.beta * std)

    def get_exploration_factor(self) -> float:
        return min(1.0, self.beta / 2.0)

    def update_beta(self, new_beta: float):
        self.beta = new_beta


class ProbabilityOfImprovementStrategy(AcquisitionStrategy):
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def calculate(self, mean: float, std: float, best_value: float, **kwargs) -> float:
        if std <= 1e-8:
            return 0.0

        improvement = best_value - mean - self.xi
        z = improvement / std
        return norm.cdf(z)

    def get_exploration_factor(self) -> float:
        return 0.2


class MaxEntropySearchStrategy(AcquisitionStrategy):
    def calculate(self, mean: float, std: float, best_value: float, **kwargs) -> float:
        if std <= 1e-8:
            return 0.0
        return np.log(1 + std**2)

    def get_exploration_factor(self) -> float:
        return 0.8


# === Future: Thompson Sampling ===
class ThompsonSamplingStrategy(AcquisitionStrategy):
    def calculate(
        self,
        mean: float,
        std: float,
        best_value: float,
        sample: Optional[float] = None,
        **kwargs,
    ) -> float:
        return sample if sample is not None else np.random.normal(mean, std)

    def get_exploration_factor(self) -> float:
        return 0.6


# === Manager with Safe, Compiled Switching ===


class ProgressiveAcquisitionManager:
    def __init__(self, config):
        self.config = config
        self.switch_history = []
        self.trials_since_switch = 0
        self.verbose = getattr(config, "verbose", False)

        # Define acquisition strategy implementations
        self.strategies = {
            "ei": ExpectedImprovementStrategy(xi=config.ei_xi),
            "logei": LogExpectedImprovementStrategy(xi=config.ei_xi),
            "ucb": UpperConfidenceBoundStrategy(beta=config.ucb_beta_start),
            "pi": ProbabilityOfImprovementStrategy(xi=config.ei_xi),
            "mes": MaxEntropySearchStrategy(),
            "thompson": ThompsonSamplingStrategy(),
        }

        # Resolve 'auto' acquisition choice
        if config.initial_acquisition == "auto":
            # Smart resolution based on config or external hints
            context = {
                "n_dims": getattr(config, "n_dims", 5),
                "n_trials": getattr(config, "n_trials", 0),
                "noisy": getattr(config, "noisy", False),
                "multi_modal": getattr(config, "multi_modal", False),
            }
            self.current_acquisition = self.resolve_auto_acquisition(context)
            if getattr(config, "verbose", False):
                print(
                    f"[AcquisitionManager] Resolved 'auto' → '{self.current_acquisition}' using context {context}"
                )
        else:
            self.current_acquisition = config.initial_acquisition

        # Compile switch conditions
        for rule in config.switch_conditions.values():
            try:
                rule["compiled"] = compile(
                    ast.parse(rule["condition"], mode="eval"), "<string>", "eval"
                )
            except:
                rule["compiled"] = None

    def resolve_auto_acquisition(self, context: Dict[str, Any]) -> str:
        """Smart resolution of 'auto' acquisition based on task context"""
        n_dims = context.get("n_dims", 5)
        n_trials = context.get("n_trials", 0)
        noisy = context.get("noisy", False)
        multi_modal = context.get("multi_modal", False)

        if n_trials < 10:
            return "ucb"  # Explore early
        if noisy and n_dims <= 10:
            return "logei"  # More numerically stable for high noise
        if multi_modal and n_dims > 10:
            return "mes"  # Entropy-based for complex landscapes
        if n_dims <= 5:
            return "ei"  # Exploit in low-D space
        return "ucb"

    def evaluate_switch_conditions(self, optimization_state: Dict) -> Optional[str]:
        if (
            not self.config.enable_switching
            or self.trials_since_switch < self.config.min_trials_before_switch
        ):
            #print("trials since switch:", self.trials_since_switch)
            #print("min trials before switch:", self.config.min_trials_before_switch)
            #print("Switching disabled or not enough trials since last switch.")
            return None

        trials = optimization_state.get("trials", [])
        feasible_trials = [t for t in trials if getattr(t, "is_feasible", True)]
        n_trials = len(trials)

        if n_trials == 0:
            print("No trials available for switching evaluation.")
            return None

        values = [t.value for t in feasible_trials or trials]
        max_improvement = max(values) - min(values) if len(values) > 1 else 0
        feasible_rate = len(feasible_trials) / n_trials
        trials_without_improvement = optimization_state.get("no_improve_count", 0)
        turbo_stats = optimization_state.get("turbo_stats", {})
        restart_count = turbo_stats.get("total_restarts", 0)
        avg_radius = turbo_stats.get("avg_radius", 1.0)
        progress_ratio = n_trials / optimization_state.get("max_trials", 100)
        diversity_score = optimization_state.get("diversity_score", 0.5)

        # Build context for switching rule evaluation
        context = {
            "feasible_rate": feasible_rate,
            "max_improvement": max_improvement,
            "trials_without_improvement": trials_without_improvement,
            "restart_count": restart_count,
            "avg_radius": avg_radius,
            "progress_ratio": progress_ratio,
            "diversity_score": diversity_score,
        }

        for rule in self.config.switch_conditions.values():
            from_modes = rule.get("from", [])
            if isinstance(from_modes, str):
                from_modes = [from_modes]
            if self.current_acquisition not in from_modes:
                continue

            condition_fn = rule.get("condition")
            if not callable(condition_fn):
                continue

            try:
                if condition_fn(context):
                    if self.verbose:
                        print(
                            f"🔁 Switching from '{self.current_acquisition}' to '{rule['to']}' by rule."
                        )
                    return rule["to"]
            except Exception as e:
                if self.verbose:
                    print(
                        f"⚠️ Failed to evaluate rule '{rule.get('name', 'unknown')}': {e}"
                    )

        return None

    def switch_acquisition(self, new_acquisition: str, reason: str = ""):
        if new_acquisition == self.current_acquisition:
            return False

        self.switch_history.append(
            {
                "from": self.current_acquisition,
                "to": new_acquisition,
                "reason": reason,
                "trial": self.trials_since_switch,
            }
        )

        self.current_acquisition = new_acquisition
        self.trials_since_switch = 0
        self._update_strategy_parameters()
        return True

    def _update_strategy_parameters(self):
        ucb = self.strategies.get("ucb")
        if isinstance(ucb, UpperConfidenceBoundStrategy):
            ucb.update_beta(
                max(self.config.ucb_beta_end, ucb.beta * self.config.ucb_beta_decay)
            )

    def calculate_value(
        self, mean: float, std: float, best_value: float, **kwargs
    ) -> float:
        strategy = self.strategies.get(self.current_acquisition, self.strategies["ei"])
        self.trials_since_switch += 1
        return strategy.calculate(mean, std, best_value, **kwargs)

    def calculate_batch(
        self, means: np.ndarray, stds: np.ndarray, best_value: float, **kwargs
    ) -> np.ndarray:
        strategy = self.strategies.get(self.current_acquisition, self.strategies["ei"])
        return strategy.calculate_batch(means, stds, best_value, **kwargs)

    def get_current_exploration_factor(self) -> float:
        return self.strategies[self.current_acquisition].get_exploration_factor()

    def get_switch_summary(self) -> Dict:
        return {
            "current_acquisition": self.current_acquisition,
            "total_switches": len(self.switch_history),
            "switch_history": self.switch_history,
            "trials_since_switch": self.trials_since_switch,
            "exploration_factor": self.get_current_exploration_factor(),
        }
