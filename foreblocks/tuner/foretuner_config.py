
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Parameter:
    """Enhanced hyperparameter definition with validation"""
    name: str
    type: str  # 'float', 'int', 'choice', 'ordinal'
    low: float = None
    high: float = None
    choices: List[Any] = None
    log: bool = False
    
    def __post_init__(self):
        if self.type in ['float', 'int']:
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter {self.name}: low and high must be specified for {self.type}")
            if self.low >= self.high:
                raise ValueError(f"Parameter {self.name}: low must be < high")
        elif self.type in ['choice', 'ordinal']:
            if not self.choices:
                raise ValueError(f"Parameter {self.name}: choices must be specified for {self.type}")


@dataclass
class TuRBOConfig:
    """TuRBO configuration parameters with adaptive radius settings"""
    n_trust_regions: int = 3
    init_radius: float = 0.8
    success_tolerance: int = 3
    failure_tolerance: int = 5
    max_restarts: int = 10
    
    # Adaptive radius parameters (from original TuRBO)
    success_window: int = 3  # Number of recent successes to track
    failure_window: int = 5  # Number of recent failures to track
    min_radius: float = 0.005**0.5  # Minimum trust region radius
    max_radius: float = 0.8  # Maximum trust region radius
    
    # Radius adjustment factors
    expand_factor: float = 2.0  # Expansion factor for successful iterations
    contract_factor: float = 0.5  # Contraction factor for failed iterations
    
    # Success/failure thresholds
    min_improvement: float = 1e-8  # Minimum improvement to consider success
    restart_threshold: float = 0.01  # Radius threshold for restart
from dataclasses import dataclass, field
from typing import Any, Dict, List



# === Switching Rule Callables ===
def to_ei_after_feasibility(stats):
    return stats.get('feasible_rate', 0.0) >= 0.8 and stats.get('trials_without_improvement', 0) >= 5

def to_logei_on_large_jump(stats):
    return stats.get('max_improvement', 0.0) >= 1.0

def ei_stagnation_to_ucb(stats):
    return stats.get('trials_without_improvement', 0) >= 10 and stats.get('avg_radius', 1.0) > 0.4

def fine_tuning(stats):
    return stats.get('progress_ratio', 0.0) >= 0.8 and stats.get('avg_radius', 1.0) < 0.1

def to_mes_if_flat_and_restart(stats):
    return stats.get('restart_count', 0) > 2 and stats.get('diversity_score', 1.0) < 0.3
# === Additional Switching Rule Callables ===

def convergence_to_ucb(stats):
    """Switch to UCB when progress is slow but we're still feasible"""
    return (stats.get('progress_ratio', 0.0) < 0.1 and 
            stats.get('feasible_rate', 0.0) >= 0.6 and
            stats.get('trials_without_improvement', 0) >= 8)

def high_diversity_to_ei(stats):
    """Switch to EI when diversity is high (good exploration coverage)"""
    return (stats.get('diversity_score', 0.0) >= 0.7 and 
            stats.get('feasible_rate', 0.0) >= 0.5)

def low_feasibility_to_ucb(stats):
    """Switch to UCB when feasibility is low (need more exploration)"""
    return (stats.get('feasible_rate', 0.0) < 0.3 and 
            stats.get('trials_without_improvement', 0) >= 5)

def rapid_improvement_to_logei(stats):
    """Switch to LogEI when making rapid progress (ride the wave)"""
    return (stats.get('max_improvement', 0.0) >= 0.5 and 
            stats.get('progress_ratio', 0.0) >= 0.3 and
            stats.get('avg_radius', 1.0) < 0.5)

def restart_recovery_to_mes(stats):
    """Switch to MES after restarts to find new regions"""
    return (stats.get('restart_count', 0) >= 1 and 
            stats.get('diversity_score', 1.0) >= 0.5 and
            stats.get('feasible_rate', 0.0) < 0.7)

def focused_search_to_ei(stats):
    """Switch to EI when search is focused (small radius) and feasible"""
    return (stats.get('avg_radius', 1.0) < 0.2 and 
            stats.get('feasible_rate', 0.0) >= 0.7 and
            stats.get('trials_without_improvement', 0) < 8)

def exploration_fatigue_to_logei(stats):
    """Switch to LogEI when exploration hasn't found improvements"""
    return (stats.get('trials_without_improvement', 0) >= 15 and 
            stats.get('diversity_score', 0.0) >= 0.6 and
            stats.get('feasible_rate', 0.0) >= 0.4)

def early_exploitation_to_ei(stats):
    """Switch to EI early if we find a good region quickly"""
    return (stats.get('max_improvement', 0.0) >= 0.3 and 
            stats.get('progress_ratio', 0.0) >= 0.2 and
            stats.get('restart_count', 0) == 0)

def mes_stagnation_to_ucb(stats):
    """Switch from MES to UCB if not finding diverse solutions"""
    return (stats.get('trials_without_improvement', 0) >= 12 and 
            stats.get('diversity_score', 1.0) < 0.4 and
            stats.get('avg_radius', 1.0) > 0.3)

def balanced_exploration_to_logei(stats):
    """Switch to LogEI when we have balanced exploration/exploitation"""
    return (stats.get('feasible_rate', 0.0) >= 0.5 and 
            stats.get('diversity_score', 0.0) >= 0.4 and
            stats.get('avg_radius', 1.0) >= 0.2 and
            stats.get('avg_radius', 1.0) <= 0.6)

def desperation_to_mes(stats):
    """Switch to MES when all else fails"""
    return (stats.get('trials_without_improvement', 0) >= 20 and 
            stats.get('restart_count', 0) >= 3 and
            stats.get('progress_ratio', 0.0) < 0.1)

def precision_tuning_to_ei(stats):
    """Switch to EI for precision tuning in good regions"""
    return (stats.get('feasible_rate', 0.0) >= 0.8 and 
            stats.get('avg_radius', 1.0) < 0.15 and
            stats.get('max_improvement', 0.0) >= 0.2)

# === Extended Rule Registry ===
SWITCH_RULES = {
    # Original rules
    'to_ei_after_feasibility': {
        'from': ['ucb'],
        'to': 'ei',
        'condition': to_ei_after_feasibility,
        'min_hold': 3,
    },
    'to_logei_on_large_jump': {
        'from': ['ucb', 'ei'],
        'to': 'logei',
        'condition': to_logei_on_large_jump,
        'min_hold': 3,
    },
    'ei_stagnation_to_ucb': {
        'from': ['ei', 'logei'],
        'to': 'ucb',
        'condition': ei_stagnation_to_ucb,
        'min_hold': 5,
    },
    'fine_tuning': {
        'from': ['logei'],
        'to': 'ei',
        'condition': fine_tuning,
        'min_hold': 5,
    },
    'to_mes_if_flat_and_restart': {
        'from': ['ei'],
        'to': 'mes',
        'condition': to_mes_if_flat_and_restart,
        'min_hold': 5,
    },
    
    # New rules
    'convergence_to_ucb': {
        'from': ['ei', 'logei', 'mes'],
        'to': 'ucb',
        'condition': convergence_to_ucb,
        'min_hold': 4,
    },
    'high_diversity_to_ei': {
        'from': ['ucb', 'mes'],
        'to': 'ei',
        'condition': high_diversity_to_ei,
        'min_hold': 3,
    },
    'low_feasibility_to_ucb': {
        'from': ['ei', 'logei', 'mes'],
        'to': 'ucb',
        'condition': low_feasibility_to_ucb,
        'min_hold': 4,
    },
    'rapid_improvement_to_logei': {
        'from': ['ucb', 'ei'],
        'to': 'logei',
        'condition': rapid_improvement_to_logei,
        'min_hold': 3,
    },
    'restart_recovery_to_mes': {
        'from': ['ucb', 'ei', 'logei'],
        'to': 'mes',
        'condition': restart_recovery_to_mes,
        'min_hold': 4,
    },
    'focused_search_to_ei': {
        'from': ['ucb', 'logei'],
        'to': 'ei',
        'condition': focused_search_to_ei,
        'min_hold': 3,
    },
    'exploration_fatigue_to_logei': {
        'from': ['ucb', 'mes'],
        'to': 'logei',
        'condition': exploration_fatigue_to_logei,
        'min_hold': 4,
    },
    'early_exploitation_to_ei': {
        'from': ['ucb'],
        'to': 'ei',
        'condition': early_exploitation_to_ei,
        'min_hold': 2,
    },
    'mes_stagnation_to_ucb': {
        'from': ['mes'],
        'to': 'ucb',
        'condition': mes_stagnation_to_ucb,
        'min_hold': 5,
    },
    'balanced_exploration_to_logei': {
        'from': ['ucb', 'ei'],
        'to': 'logei',
        'condition': balanced_exploration_to_logei,
        'min_hold': 3,
    },
    'desperation_to_mes': {
        'from': ['ucb', 'ei', 'logei'],
        'to': 'mes',
        'condition': desperation_to_mes,
        'min_hold': 6,
    },
    'precision_tuning_to_ei': {
        'from': ['logei', 'ucb'],
        'to': 'ei',
        'condition': precision_tuning_to_ei,
        'min_hold': 3,
    },
}


@dataclass
class ProgressiveAcquisitionConfig:
    """Configuration for progressive acquisition switching"""
    
    # === Initial Acquisition ===
    initial_acquisition: str = 'ucb'  # Start with exploration
    
    # === Switching control ===
    enable_switching: bool = True
    min_trials_before_switch: int = 3
    
    # === Acquisition switching thresholds ===
    feasible_region_threshold: float = 0.1        # ≥10% feasible → consider exploitation
    improvement_stagnation_threshold: int = 5    # Switch if no improvement
    use_logei_threshold: float = 10.0             # Large jump → LogEI
    
    # === UCB Parameters ===
    ucb_beta_start: float = 3.0     # Initial exploration
    ucb_beta_end: float = 1.0       # Final beta after decay
    ucb_beta_decay: float = 0.95    # Multiplicative decay per switch
    
    # === EI Parameters ===
    ei_xi: float = 0.01
    
    # === Dynamic Switch Rules ===
    #switch_conditions: Dict[str, Dict[str, Any]] = field(default_factory=dict)


    switch_conditions = SWITCH_RULES
