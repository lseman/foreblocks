from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

try:  # optional
    from scipy.stats import qmc
except Exception:  # pragma: no cover
    qmc = None
try:  # optional
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None
try:  # optional
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None
    RBF = None
    WhiteKernel = None
from scipy.stats import multivariate_normal, norm, yeojohnson_normmax

__all__ = [
    "TPE",
    "TPEConf",
    "TPEConfig",
]

from .acquisition import AcquisitionStrategy, LogRatioAcquisition
from .batch import (
    BatchSelector,
    FantasizedBatchSelector,
    GreedyDiversitySelector,
    LocalPenalizationSelector,
    ThompsonSamplingSelector,
)
from .gamma import GammaStrategy, build_gamma_strategy
from .observation import ObservationStore
from .param_models import CatModel, FloatModel, IntModel
from .utils import (
    _clamp,
    _reflect_into_bounds,
    _robust_scale_1d,
    make_positive_definite,
    safe_log,
    safe_normalize,
    yeojohnson_forward,
)

# -----------------------------------------------------------------------------
# Sampler Options (quick reference)
# -----------------------------------------------------------------------------
# Core: gamma (callable or float), gamma_strategy, dual_gamma, gamma_exploit,
#       gamma_explore, exploit_candidate_frac, n_startup_trials, n_ei_candidates
# Bandwidth: bandwidth_factor, min_bandwidth, min_bandwidth_factor,
#            bandwidth_exponent, consider_magic_clip, covariance_shrinkage
# Priors: prior_weight, prior_mix_weight, consider_prior
# Categorical: categorical_bw, categorical_kernel, categorical_distance,
#              categorical_embedding_dim, categorical_embedding_min_obs,
#              categorical_graphs, categorical_distance_func,
#              combinatorial_mode, combinatorial_min_obs,
#              combinatorial_keep_frac, combinatorial_min_keep,
#              combinatorial_sharpen_max
# Ints: int_discrete_threshold
# Budgets: max_budget, split_budget_correction, weight_power, weights_func,
#          time_weight_decay
# Joint: joint_conditional, multivariate, group_multivariate
#        joint_per_component_bandwidth, joint_component_bw_rule,
#        joint_component_bw_neighbors
# Constraints: hard_constraints, soft_constraints, soft_penalty_weight,
#              constraint_max_attempts, ctpe_constraints,
#              constraint_repair_attempts, constraint_rejection_max,
#              constraint_violation_penalty, constraint_logging
# Batch: batch_strategy, penalization_power, acq_cache_size, constant_liar,
#        liar_strategy, liar_quantile, liar_adaptive, liar_adaptive_threshold,
#        liar_adaptive_strategy_small
#        use_qlogei, use_qnei, q_fantasies, q_fantasy_noise, q_fantasy_weight
# Other: consider_endpoints, seed, verbose, local_bandwidth, local_bandwidth_k,
#        multi_scale, recent_frac, exploit_frac, explore_frac, copula_joint,
#        max_bandwidth_factor, kde_cache_size, smooth_startup, use_ei, ei_k,
#        use_ucb, ucb_kappa, gp_ucb_max_obs, acq_log_space, acq_softplus,
#        quasi_random_startup,
#        quasi_random_method, hybrid_local_search, local_search_top_k,
#        local_search_steps, blocking_unblock_after, blocking_min_obs


@dataclass
class _ParamInfo:
    name: str
    typ: str  # 'float' | 'int' | 'choice'
    rng: Any
    transform: Optional[str] = None
    log_range: Optional[Tuple[float, float]] = None
    condition_parent: Optional[str] = None
    condition_values: Optional[set] = None
    joint_group: Optional[str] = None
    embedding: Optional[Dict[Any, np.ndarray]] = None


@dataclass
class TrustRegion:
    center: np.ndarray
    length: float  # side length as fraction of domain
    success_count: int = 0
    failure_count: int = 0
    success_tolerance: int = 3
    failure_tolerance: int = 5
    min_length: float = 0.01
    expand_factor: float = 2.0
    shrink_factor: float = 0.5

    def update(self, improved: bool) -> None:
        if improved:
            self.success_count += 1
            self.failure_count = 0
            if self.success_count >= self.success_tolerance:
                self.length = min(float(self.length * self.expand_factor), 1.0)
                self.success_count = 0
        else:
            self.failure_count += 1
            self.success_count = 0
            if self.failure_count >= self.failure_tolerance:
                self.length = max(
                    float(self.length * self.shrink_factor), float(self.min_length)
                )
                self.failure_count = 0


@dataclass
class TPEConf:
    """
    Single config container for TPE.
    Use with `TPE.from_config(config_space, conf)`.
    """

    gamma: Dict[str, Any] = field(
        default_factory=lambda: {
            "gamma": 0.15,
            "gamma_strategy": "sqrt",
            "dual_gamma": False,
            "gamma_exploit": 0.10,
            "gamma_explore": 0.30,
            "exploit_candidate_frac": 0.7,
        }
    )
    bandwidth: Dict[str, Any] = field(
        default_factory=lambda: {
            "bandwidth_factor": 0.8,
            "min_bandwidth": 1e-3,
            "min_bandwidth_factor": 0.05,
            "bandwidth_exponent": 0.7,
            "consider_magic_clip": True,
            "covariance_shrinkage": "none",
            "max_bandwidth_factor": None,
            "local_bandwidth": False,
            "local_bandwidth_k": 7,
        }
    )
    prior: Dict[str, Any] = field(
        default_factory=lambda: {
            "prior_weight": 0.1,
            "prior_mix_weight": 0.15,
            "prior_mix_min": 0.01,
            "prior_mix_decay": 25.0,
            "consider_prior": True,
        }
    )
    constraints: Dict[str, Any] = field(
        default_factory=lambda: {
            "hard_constraints": None,
            "soft_constraints": None,
            "soft_penalty_weight": 1.0,
            "constraint_max_attempts": 30,
            "ctpe_constraints": False,
            "constraint_repair_attempts": 30,
            "constraint_rejection_max": 120,
            "constraint_violation_penalty": None,
            "constraint_logging": False,
        }
    )
    batch: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_strategy": "diversity",
            "penalization_power": 2.0,
            "acq_cache_size": 0,
            "constant_liar": False,
            "liar_strategy": "mean",
            "liar_quantile": 0.8,
            "liar_adaptive": True,
            "liar_adaptive_threshold": 10,
            "liar_adaptive_strategy_small": "worst",
            "use_qlogei": False,
            "use_qnei": False,
            "q_fantasies": 16,
            "q_fantasy_noise": 1e-6,
            "q_fantasy_weight": 1.0,
        }
    )
    trust_region: Dict[str, Any] = field(
        default_factory=lambda: {
            "trust_region_enabled": True,
            "trust_region_init_length": 1.0,
            "trust_region_min_length": 0.01,
            "trust_region_success_tolerance": 3,
            "trust_region_failure_tolerance": 5,
            "trust_region_expand_factor": 2.0,
            "trust_region_shrink_factor": 0.5,
            "trust_region_restart_on_min_length": True,
            "trust_region_restart_center": "best",
        }
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out.update(dict(self.gamma))
        out.update(dict(self.bandwidth))
        out.update(dict(self.prior))
        out.update(dict(self.constraints))
        out.update(dict(self.batch))
        out.update(dict(self.trust_region))
        out.update(dict(self.extra))
        return out


# Backward compatibility alias
TPEConfig = TPEConf


class TPE:
    """
    Enhanced TPE with:
    - Proper sampling from l(x) + ranking by l(x)/g(x)
    - Robust bandwidth & boundary handling
    - Discrete/int/choice handling
    - Multi-fidelity weighting
    - Per-parameter prior mixture
    - Budget-corrected good/bad split
    - Optional joint modeling (conditional or global multivariate)
    - Adaptive gamma strategies
    - Basic batched diversity for n_candidates > 1
    """

    @classmethod
    def from_config(
        cls,
        config_space: Dict[str, Tuple],
        cfg: TPEConf,
        **overrides: Any,
    ) -> "TPE":
        if not overrides:
            return cls(config_space=config_space, conf=cfg)
        merged = TPEConf(
            gamma=dict(cfg.gamma),
            bandwidth=dict(cfg.bandwidth),
            prior=dict(cfg.prior),
            constraints=dict(cfg.constraints),
            batch=dict(cfg.batch),
            trust_region=dict(cfg.trust_region),
            extra=dict(cfg.extra),
        )
        merged.extra.update(overrides)
        return cls(config_space=config_space, conf=merged)

    def __init__(
        self,
        config_space: Dict[str, Tuple],
        conf: Optional[TPEConf] = None,
    ):
        cfg = TPEConf() if conf is None else conf
        opts = cfg.to_kwargs()

        gamma = opts.get("gamma", 0.15)
        gamma_strategy = opts.get("gamma_strategy", "sqrt")
        dual_gamma = opts.get("dual_gamma", False)
        gamma_exploit = opts.get("gamma_exploit", 0.10)
        gamma_explore = opts.get("gamma_explore", 0.30)
        exploit_candidate_frac = opts.get("exploit_candidate_frac", 0.7)
        n_startup_trials = opts.get("n_startup_trials", 10)
        quasi_random_startup = opts.get("quasi_random_startup", False)
        quasi_random_method = opts.get("quasi_random_method", "lhs")
        n_ei_candidates = opts.get("n_ei_candidates", 64)
        bandwidth_factor = opts.get("bandwidth_factor", 0.8)
        min_bandwidth = opts.get("min_bandwidth", 1e-3)
        min_bandwidth_factor = opts.get("min_bandwidth_factor", 0.05)
        bandwidth_exponent = opts.get("bandwidth_exponent", 0.7)
        weight_power = opts.get("weight_power", 2.0)
        weight_strategy = opts.get("weight_strategy", "time_decay")
        weights_func = opts.get("weights_func", None)
        time_weight_decay = opts.get("time_weight_decay", 0.02)
        prior_weight = opts.get("prior_weight", 0.1)
        categorical_bw = opts.get("categorical_bw", 1.0)
        categorical_distance = opts.get("categorical_distance", "index")
        categorical_embedding_dim = opts.get("categorical_embedding_dim", 8)
        categorical_embedding_min_obs = opts.get("categorical_embedding_min_obs", 20)
        categorical_graphs = opts.get("categorical_graphs", None)
        categorical_kernel = opts.get("categorical_kernel", "exponential")
        int_discrete_threshold = opts.get("int_discrete_threshold", 256)
        prior_mix_weight = opts.get("prior_mix_weight", 0.15)
        prior_mix_min = opts.get("prior_mix_min", 0.01)
        prior_mix_decay = opts.get("prior_mix_decay", 25.0)
        consider_prior = opts.get("consider_prior", True)
        consider_magic_clip = opts.get("consider_magic_clip", True)
        covariance_shrinkage = opts.get("covariance_shrinkage", "none")
        cma_cov_adaptation = opts.get("cma_cov_adaptation", True)
        cma_cov_learning_rate = opts.get("cma_cov_learning_rate", 0.1)
        cma_path_decay = opts.get("cma_path_decay", 0.9)
        max_budget = opts.get("max_budget", None)
        split_budget_correction = opts.get("split_budget_correction", 0.25)
        joint_conditional = opts.get("joint_conditional", True)
        multivariate = opts.get("multivariate", "auto")
        group_multivariate = opts.get("group_multivariate", False)
        joint_per_component_bandwidth = opts.get("joint_per_component_bandwidth", True)
        joint_component_bw_rule = opts.get("joint_component_bw_rule", "scott")
        joint_component_bw_neighbors = opts.get("joint_component_bw_neighbors", None)
        warping = opts.get("warping", True)
        atpe = opts.get("atpe", True)
        atpe_params = opts.get("atpe_params", None)
        blocking_threshold = opts.get("blocking_threshold", 0.8)
        hard_constraints = opts.get("hard_constraints", None)
        soft_constraints = opts.get("soft_constraints", None)
        soft_penalty_weight = opts.get("soft_penalty_weight", 1.0)
        constraint_max_attempts = opts.get("constraint_max_attempts", 30)
        ctpe_constraints = opts.get("ctpe_constraints", False)
        constraint_repair_attempts = opts.get("constraint_repair_attempts", 30)
        constraint_rejection_max = opts.get("constraint_rejection_max", 120)
        constraint_violation_penalty = opts.get("constraint_violation_penalty", None)
        constraint_logging = opts.get("constraint_logging", False)
        verbose = opts.get("verbose", False)
        acq_cache_size = opts.get("acq_cache_size", 0)
        batch_strategy = opts.get("batch_strategy", "diversity")
        penalization_power = opts.get("penalization_power", 2.0)
        constant_liar = opts.get("constant_liar", False)
        liar_strategy = opts.get("liar_strategy", "mean")
        liar_quantile = opts.get("liar_quantile", 0.8)
        use_qlogei = opts.get("use_qlogei", False)
        use_qnei = opts.get("use_qnei", False)
        q_fantasies = opts.get("q_fantasies", 16)
        q_fantasy_noise = opts.get("q_fantasy_noise", 1e-6)
        q_fantasy_weight = opts.get("q_fantasy_weight", 1.0)
        categorical_distance_func = opts.get("categorical_distance_func", None)
        combinatorial_mode = opts.get("combinatorial_mode", False)
        combinatorial_min_obs = opts.get("combinatorial_min_obs", 50)
        combinatorial_keep_frac = opts.get("combinatorial_keep_frac", 0.5)
        combinatorial_min_keep = opts.get("combinatorial_min_keep", 3)
        combinatorial_sharpen_max = opts.get("combinatorial_sharpen_max", 1.8)
        consider_endpoints = opts.get("consider_endpoints", False)
        seed = opts.get("seed", None)
        liar_adaptive = opts.get("liar_adaptive", True)
        liar_adaptive_threshold = opts.get("liar_adaptive_threshold", 10)
        liar_adaptive_strategy_small = opts.get("liar_adaptive_strategy_small", "worst")
        local_bandwidth = opts.get("local_bandwidth", False)
        local_bandwidth_k = opts.get("local_bandwidth_k", 7)
        multi_scale = opts.get("multi_scale", False)
        recent_frac = opts.get("recent_frac", 0.5)
        exploit_frac = opts.get("exploit_frac", 0.7)
        explore_frac = opts.get("explore_frac", 0.25)
        copula_joint = opts.get("copula_joint", False)
        max_bandwidth_factor = opts.get("max_bandwidth_factor", None)
        kde_cache_size = opts.get("kde_cache_size", 10000)
        smooth_startup = opts.get("smooth_startup", True)
        use_ei = opts.get("use_ei", False)
        ei_k = opts.get("ei_k", 10)
        hybrid_local_search = opts.get("hybrid_local_search", False)
        local_search_top_k = opts.get("local_search_top_k", 4)
        local_search_steps = opts.get("local_search_steps", 5)
        use_ucb = opts.get("use_ucb", False)
        ucb_kappa = opts.get("ucb_kappa", 1.5)
        gp_ucb_max_obs = opts.get("gp_ucb_max_obs", 500)
        acq_log_space = opts.get("acq_log_space", True)
        acq_softplus = opts.get("acq_softplus", True)
        blocking_unblock_after = opts.get("blocking_unblock_after", 0)
        blocking_min_obs = opts.get("blocking_min_obs", 10)
        trust_region_enabled = opts.get("trust_region_enabled", True)
        trust_region_init_length = opts.get("trust_region_init_length", 1.0)
        trust_region_min_length = opts.get("trust_region_min_length", 0.01)
        trust_region_success_tolerance = opts.get("trust_region_success_tolerance", 3)
        trust_region_failure_tolerance = opts.get("trust_region_failure_tolerance", 5)
        trust_region_expand_factor = opts.get("trust_region_expand_factor", 2.0)
        trust_region_shrink_factor = opts.get("trust_region_shrink_factor", 0.5)
        trust_region_restart_on_min_length = opts.get(
            "trust_region_restart_on_min_length", True
        )
        trust_region_restart_center = opts.get("trust_region_restart_center", "best")

        self.config_space = config_space
        self.gamma = gamma
        self.gamma_strategy = str(gamma_strategy).lower()
        self.gamma_strategy_obj: GammaStrategy = build_gamma_strategy(
            gamma=self.gamma,
            gamma_strategy=self.gamma_strategy,
            adaptive_gamma_fn=self._adaptive_gamma,
        )
        self.dual_gamma = bool(dual_gamma)
        self.gamma_exploit = float(gamma_exploit)
        self.gamma_explore = float(gamma_explore)
        self.exploit_candidate_frac = float(exploit_candidate_frac)
        self.n_startup_trials = int(n_startup_trials)
        self.quasi_random_startup = bool(quasi_random_startup)
        self.quasi_random_method = str(quasi_random_method or "lhs").lower()
        self.n_ei_candidates = int(n_ei_candidates)
        self.bandwidth_factor = bandwidth_factor
        self.min_bandwidth = float(min_bandwidth)
        self.min_bandwidth_factor = float(min_bandwidth_factor)
        self.bandwidth_exponent = float(bandwidth_exponent)
        self.weight_power = float(weight_power)
        self.weight_strategy = str(weight_strategy or "time_decay").lower()
        self.weights_func = weights_func
        self.time_weight_decay = (
            None if time_weight_decay is None else float(time_weight_decay)
        )
        self.prior_weight = float(prior_weight)
        self.categorical_bw = categorical_bw
        self.categorical_kernel = str(categorical_kernel or "exponential").lower()
        self.categorical_distance = str(categorical_distance or "index").lower()
        self.categorical_embedding_dim = int(categorical_embedding_dim)
        self.categorical_embedding_min_obs = int(categorical_embedding_min_obs)
        self.categorical_graphs = categorical_graphs or {}
        self.int_discrete_threshold = int(int_discrete_threshold)
        self.prior_mix_weight = prior_mix_weight
        self.prior_mix_min = float(max(0.0, prior_mix_min))
        self.prior_mix_decay = float(max(1e-6, prior_mix_decay))
        self.consider_prior = consider_prior
        self.consider_magic_clip = bool(consider_magic_clip)
        self.covariance_shrinkage = str(covariance_shrinkage or "none").lower()
        self.cma_cov_adaptation = bool(cma_cov_adaptation)
        self.cma_cov_learning_rate = float(min(max(cma_cov_learning_rate, 1e-6), 1.0))
        self.cma_path_decay = float(min(max(cma_path_decay, 0.0), 0.9999))
        self._cma_cov_state: Dict[
            Tuple[str, Tuple[str, ...]], Dict[str, np.ndarray]
        ] = {}
        self.local_bandwidth = bool(local_bandwidth)
        self.local_bandwidth_k = int(local_bandwidth_k)
        self.multi_scale = bool(multi_scale)
        self.recent_frac = float(recent_frac)
        self.exploit_frac = float(exploit_frac)
        self.explore_frac = float(explore_frac)
        self.copula_joint = bool(copula_joint)
        self.max_bandwidth_factor = max_bandwidth_factor
        self.kde_cache_size = int(kde_cache_size)
        self._kde_cache: "OrderedDict[Tuple[Any, ...], float]" = OrderedDict()
        self.smooth_startup = bool(smooth_startup)
        self.use_ei = bool(use_ei)
        self.ei_k = int(ei_k)
        self.hybrid_local_search = bool(hybrid_local_search)
        self.local_search_top_k = int(local_search_top_k)
        self.local_search_steps = int(local_search_steps)
        self.use_ucb = bool(use_ucb)
        self.ucb_kappa = float(ucb_kappa)
        self.gp_ucb_max_obs = int(gp_ucb_max_obs)
        self.acq_log_space = bool(acq_log_space)
        self.acq_softplus = bool(acq_softplus)
        self.blocking_unblock_after = int(blocking_unblock_after)
        self.blocking_min_obs = int(blocking_min_obs)
        self.trust_region_enabled = bool(trust_region_enabled)
        self.trust_region_init_length = float(
            min(max(trust_region_init_length, 1e-6), 1.0)
        )
        self.trust_region_min_length = float(max(trust_region_min_length, 1e-6))
        self.trust_region_success_tolerance = int(
            max(1, trust_region_success_tolerance)
        )
        self.trust_region_failure_tolerance = int(
            max(1, trust_region_failure_tolerance)
        )
        self.trust_region_expand_factor = float(max(1.0, trust_region_expand_factor))
        self.trust_region_shrink_factor = float(
            min(max(trust_region_shrink_factor, 1e-6), 1.0)
        )
        self.trust_region_restart_on_min_length = bool(
            trust_region_restart_on_min_length
        )
        self.trust_region_restart_center = str(
            trust_region_restart_center or "best"
        ).lower()
        self.max_budget = None if max_budget is None else float(max_budget)
        self.split_budget_correction = float(split_budget_correction)
        self.observation_store = ObservationStore(
            max_budget=self.max_budget,
            split_budget_correction=self.split_budget_correction,
        )
        self.joint_conditional = bool(joint_conditional)
        self.multivariate_arg = multivariate
        self.group_multivariate = bool(group_multivariate)
        self.joint_per_component_bandwidth = bool(joint_per_component_bandwidth)
        self.joint_component_bw_rule = str(joint_component_bw_rule or "scott").lower()
        self.joint_component_bw_neighbors = (
            None
            if joint_component_bw_neighbors is None
            else int(joint_component_bw_neighbors)
        )
        self.warping = bool(warping)
        self.atpe = bool(atpe)
        self.atpe_params = atpe_params or {}
        self.blocking_threshold = float(blocking_threshold)
        self.hard_constraints = list(hard_constraints or [])
        self.soft_constraints = list(soft_constraints or [])
        self.soft_penalty_weight = float(soft_penalty_weight)
        self.constraint_max_attempts = int(constraint_max_attempts)
        self.constraint_logging = bool(constraint_logging)
        self.constraint_stats = {"hard_attempts": 0, "hard_rejects": 0}
        self.ctpe_constraints = bool(ctpe_constraints)
        self.constraint_repair_attempts = int(constraint_repair_attempts)
        self.constraint_rejection_max = int(constraint_rejection_max)
        self.constraint_violation_penalty = (
            None
            if constraint_violation_penalty is None
            else float(constraint_violation_penalty)
        )
        self.verbose = bool(verbose)
        self.acq_cache_size = int(acq_cache_size)
        self._acq_cache: Dict[str, float] = {}
        self.acquisition_strategy: AcquisitionStrategy = LogRatioAcquisition(
            log_likelihood_fn=self._log_likelihood,
            soft_constraint_violation_fn=self._soft_constraint_violation,
            predict_mu_sigma_fn=self._predict_mu_sigma,
            gp_ucb_score_fn=self._gp_ucb_score,
            observations_fn=lambda: self.observations,
            soft_constraints_enabled=bool(self.soft_constraints),
            ctpe_constraints=self.ctpe_constraints,
            constraint_violation_penalty=self.constraint_violation_penalty,
            soft_penalty_weight=self.soft_penalty_weight,
            use_ei=self.use_ei,
            ei_k=self.ei_k,
            use_ucb=self.use_ucb,
            ucb_kappa=self.ucb_kappa,
            acq_log_space=self.acq_log_space,
            acq_softplus=self.acq_softplus,
        )
        self.batch_strategy = str(batch_strategy or "diversity").lower()
        self.penalization_power = float(penalization_power)
        self.use_qlogei = bool(use_qlogei)
        self.use_qnei = bool(use_qnei)
        self.q_fantasies = max(1, int(q_fantasies))
        self.q_fantasy_noise = float(max(0.0, q_fantasy_noise))
        self.q_fantasy_weight = float(max(0.0, q_fantasy_weight))
        self.batch_selector: BatchSelector = self._build_batch_selector()
        self.fantasized_batch_selector = FantasizedBatchSelector(
            use_qlogei=self.use_qlogei,
            use_qnei=self.use_qnei,
            q_fantasies=self.q_fantasies,
            q_fantasy_noise=self.q_fantasy_noise,
            q_fantasy_weight=self.q_fantasy_weight,
        )
        self.constant_liar = bool(constant_liar)
        self.liar_strategy = str(liar_strategy or "mean").lower()
        self.liar_quantile = float(liar_quantile)
        self.categorical_distance_func = categorical_distance_func or {}
        self._categorical_embeddings: Dict[str, Dict[Any, np.ndarray]] = {}
        self._categorical_embedding_version: Dict[str, int] = defaultdict(int)
        self._categorical_kernel_cache: Dict[
            Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]
        ] = {}
        self.combinatorial_mode = bool(combinatorial_mode)
        self.combinatorial_min_obs = int(combinatorial_min_obs)
        self.combinatorial_keep_frac = float(combinatorial_keep_frac)
        self.combinatorial_min_keep = int(combinatorial_min_keep)
        self.combinatorial_sharpen_max = float(combinatorial_sharpen_max)
        self.consider_endpoints = bool(consider_endpoints)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.liar_adaptive = bool(liar_adaptive)
        self.liar_adaptive_threshold = int(liar_adaptive_threshold)
        self.liar_adaptive_strategy_small = str(
            liar_adaptive_strategy_small or "worst"
        ).lower()
        self.observations: List[Tuple[Dict[str, Any], float, Optional[float]]] = []
        self.param_names = list(config_space.keys())
        self.param_info: Dict[str, _ParamInfo] = {}
        self._best_loss: float = float("inf")
        self._best_config: Optional[Dict[str, Any]] = None
        self._trust_region_restart_count: int = 0

        for param, spec in config_space.items():
            typ, rng = spec[0], spec[1]
            info = _ParamInfo(name=param, typ=typ, rng=rng)
            if len(spec) >= 3 and isinstance(spec[2], dict):
                self._apply_param_options(info, spec[2])
            if (
                typ == "float"
                and isinstance(rng, tuple)
                and len(rng) == 3
                and rng[2] == "log"
            ):
                lo, hi, _ = rng
                info.transform = "log"
                info.log_range = (math.log10(float(lo)), math.log10(float(hi)))
            if typ == "choice" and self.categorical_distance in {
                "embedding",
                "learned",
            }:
                choices = list(rng)
                if self.categorical_distance == "embedding":
                    emb = self._rng.normal(
                        size=(len(choices), self.categorical_embedding_dim)
                    )
                    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
                    info.embedding = dict(zip(choices, emb))
            self.param_info[param] = info

        self.param_names = self._topo_sort_params(self.param_names)
        self.ancestral_order = self._build_ancestral_order()
        self.joint_groups = self._collect_joint_groups()
        self._tr_param_names = [
            p for p in self.param_names if self.param_info[p].typ in {"float", "int"}
        ]
        self.trust_region: Optional[TrustRegion] = None
        if self.trust_region_enabled and self._tr_param_names:
            self.trust_region = TrustRegion(
                center=np.zeros(len(self._tr_param_names), dtype=float),
                length=self.trust_region_init_length,
                success_tolerance=self.trust_region_success_tolerance,
                failure_tolerance=self.trust_region_failure_tolerance,
                min_length=self.trust_region_min_length,
                expand_factor=self.trust_region_expand_factor,
                shrink_factor=self.trust_region_shrink_factor,
            )

        # Add global multivariate joint group if enabled and no explicit joints
        # (Defer logic to suggest() or _build_joint_models for auto detection)
        if (
            self.multivariate_arg is not False
            and not self.group_multivariate
            and not self.joint_groups
        ):
            cont_params = [
                p
                for p in self.param_names
                if self.param_info[p].typ in ("float", "int")
                and (
                    self.param_info[p].typ == "float"
                    or (
                        self.param_info[p].rng[1] - self.param_info[p].rng[0] + 1
                        > self.int_discrete_threshold
                    )
                )
            ]
            if len(cont_params) >= 2:
                self.joint_groups["__all_continuous__"] = cont_params

        if self.atpe and self.verbose:
            print("DEBUG: ATPE enabled")

    def _apply_blocked_params(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        if not (self.atpe and hasattr(self, "blocked_params") and self.blocked_params):
            return cand
        updated = dict(cand)
        for p, val in self.blocked_params.items():
            if p in updated:
                updated[p] = val
        return updated

    def _generate_candidate(
        self,
        sampler: Callable[[], Dict[str, Any]],
        attempts: int,
    ) -> Dict[str, Any]:
        """
        Helper to sample a candidate with constraints handling.
        """
        wrapped_sampler = lambda: self._apply_blocked_params(
            self._apply_trust_region(sampler())
        )

        if self.ctpe_constraints:
            cand = self._sample_with_constraints_ctpe(wrapped_sampler)
            if self.hard_constraints and not self._hard_constraints_satisfied(cand):
                retried = self._sample_with_constraints(
                    wrapped_sampler, attempts=attempts
                )
                if retried is not None:
                    return retried
                prior_retry = self._sample_with_constraints(
                    lambda: self._apply_blocked_params(self._sample_prior()),
                    attempts=attempts,
                )
                if prior_retry is not None:
                    return prior_retry
                return self._apply_blocked_params(cand)
        else:
            cand = self._sample_with_constraints(wrapped_sampler, attempts=attempts)
            if cand is None:
                cand = wrapped_sampler()
        return cand

    def _numeric_bounds(self, param: str) -> Tuple[float, float]:
        info = self.param_info[param]
        if info.typ == "float":
            return self._float_bounds_in_model_space(param)
        lo, hi = info.rng
        return float(lo), float(hi)

    def _numeric_value_in_model_space(
        self, config: Dict[str, Any], param: str
    ) -> float:
        info = self.param_info[param]
        if param in config:
            if info.typ == "float":
                return self._transform(float(config[param]), param)
            return float(config[param])
        lo, hi = self._numeric_bounds(param)
        return 0.5 * (lo + hi)

    def _numeric_center_from_config(self, config: Dict[str, Any]) -> np.ndarray:
        center = np.zeros(len(self._tr_param_names), dtype=float)
        for i, p in enumerate(self._tr_param_names):
            center[i] = self._numeric_value_in_model_space(config, p)
        return center

    def _random_trust_region_center(self) -> np.ndarray:
        center = np.zeros(len(self._tr_param_names), dtype=float)
        for i, p in enumerate(self._tr_param_names):
            lo, hi = self._numeric_bounds(p)
            center[i] = float(self._rng.uniform(lo, hi))
        return center

    def _restart_trust_region(self) -> None:
        if not self.trust_region_enabled or self.trust_region is None:
            return
        tr = self.trust_region
        if self.trust_region_restart_center == "best" and self._best_config is not None:
            tr.center = self._numeric_center_from_config(self._best_config)
        else:
            tr.center = self._random_trust_region_center()
        tr.length = self.trust_region_init_length
        tr.success_count = 0
        tr.failure_count = 0
        self._trust_region_restart_count += 1

    def _update_trust_region(self, improved: bool, config: Dict[str, Any]) -> None:
        if not self.trust_region_enabled or self.trust_region is None:
            return
        tr = self.trust_region
        prev_length = float(tr.length)
        self.trust_region.center = self._numeric_center_from_config(config)
        self.trust_region.update(improved)
        # Restart after collapse (TuRBO-style), when a shrink step lands at min length.
        collapsed = (
            (not improved)
            and prev_length > tr.min_length + 1e-12
            and tr.length <= tr.min_length + 1e-12
        )
        if self.trust_region_restart_on_min_length and collapsed:
            self._restart_trust_region()

    def _apply_trust_region(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        if not self.trust_region_enabled or self.trust_region is None:
            return cand
        if not self._tr_param_names:
            return cand
        tr = self.trust_region
        updated = dict(cand)
        for i, p in enumerate(self._tr_param_names):
            if p not in updated:
                continue
            info = self.param_info[p]
            lo, hi = self._numeric_bounds(p)
            span = max(hi - lo, 1e-12)
            half = 0.5 * float(tr.length) * span
            center = float(tr.center[i]) if i < tr.center.size else 0.5 * (lo + hi)
            tr_lo = max(lo, center - half)
            tr_hi = min(hi, center + half)
            if tr_hi < tr_lo:
                tr_lo, tr_hi = tr_hi, tr_lo

            if info.typ == "float":
                x = self._transform(float(updated[p]), p)
                x = _clamp(x, tr_lo, tr_hi)
                x = _clamp(x, lo, hi)
                updated[p] = self._inv_transform(float(x), p)
            else:
                x = float(updated[p])
                x = _clamp(x, tr_lo, tr_hi)
                x = _clamp(x, lo, hi)
                updated[p] = int(round(x))
        return updated

    # ────────────────────────────────────────────────────────────────────────────
    #   Core methods (observe, suggest)
    # ────────────────────────────────────────────────────────────────────────────

    def observe(
        self, config: Dict[str, Any], loss: float, budget: Optional[float] = None
    ) -> None:
        loss_f = float(loss)
        self.observations.append(
            (config, loss_f, None if budget is None else float(budget))
        )
        improved = loss_f < self._best_loss
        if improved:
            self._best_loss = loss_f
            self._best_config = dict(config)
        if self.trust_region_enabled and self.trust_region is not None:
            cfg_for_center = (
                self._best_config if self._best_config is not None else config
            )
            self._update_trust_region(improved=improved, config=cfg_for_center)

    def suggest(
        self,
        n_candidates: int = 1,
        budget: Optional[float] = None,
        pending_configs: Optional[List[Dict[str, Any]]] = None,
        return_scores: bool | str = False,
    ) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], np.ndarray]:
        n_candidates = int(n_candidates)
        stat_before = dict(self.constraint_stats)

        # 1. Startup phase
        if len(self.observations) < self.n_startup_trials:
            attempts = self._constraint_attempts(n_candidates)
            if self.quasi_random_startup:
                result = self._sample_quasi_random(n_candidates, attempts=attempts)
            else:
                result = []
                for _ in range(n_candidates):
                    cand = self._generate_candidate(self._sample_prior, attempts)
                    result.append(cand)
            self._maybe_log_constraint_stats(stat_before)
            if return_scores:
                return result, np.zeros(len(result), dtype=float)
            return result

        # 2. Fit models
        losses = np.array([o[1] for o in self.observations])
        loss_scale = float(np.median(losses)) if losses.size > 0 else 1.0
        if self.trust_region_enabled and self.trust_region is not None:
            if self._best_config is None and self.observations:
                best_obs = min(self.observations, key=lambda o: float(o[1]))
                self._best_config = dict(best_obs[0])
                self._best_loss = float(best_obs[1])
            if self._best_config is not None:
                self.trust_region.center = self._numeric_center_from_config(
                    self._best_config
                )

        # Compute correlations for blocking
        self.blocked_params = self._blocked_params(self.observations)

        sorted_obs = self.observation_store.sort_for_split(
            self.observations,
            loss_scale,
            atpe_filter=self._atpe_filter_observations if self.atpe else None,
        )
        n_obs = len(sorted_obs)

        # Gamma logic
        n_good = self._calculate_n_good(n_obs)
        if self.atpe:
            # ATPE Dynamic Gamma
            n_good = self._adaptive_gamma_atpe(n_obs, n_good)

        good_obs, bad_obs = self.observation_store.split_good_bad(sorted_obs, n_good)
        if self.atpe:
            bad_obs = self._filter_bad_observations_atpe(list(bad_obs))

        # Handle splitting logic (dual gamma vs standard vs multi-scale)
        exploit_good_obs = None
        exploit_bad_obs = None
        explore_good_obs = None
        explore_bad_obs = None

        if self.dual_gamma:
            n_good_exploit = max(1, int(self.gamma_exploit * n_obs))
            n_good_explore = max(1, int(self.gamma_explore * n_obs))
            exploit_good_obs = sorted_obs[:n_good_exploit]
            exploit_bad_obs = (
                sorted_obs[n_good_exploit:] if n_good_exploit < n_obs else []
            )
            explore_good_obs = sorted_obs[:n_good_explore]
            explore_bad_obs = (
                sorted_obs[n_good_explore:] if n_good_explore < n_obs else []
            )
            if self.atpe:
                exploit_bad_obs = self._filter_bad_observations_atpe(
                    list(exploit_bad_obs)
                )
                explore_bad_obs = self._filter_bad_observations_atpe(
                    list(explore_bad_obs)
                )

        if self.constant_liar and n_candidates > 1 and pending_configs:
            liar_loss = self._liar_value([o[1] for o in self.observations])
            liar_obs = []
            for cfg in pending_configs:
                if cfg is None:
                    continue
                if not self._hard_constraints_satisfied(cfg):
                    continue
                liar_obs.append((cfg, float(liar_loss), None))
            if liar_obs:
                bad_obs = list(bad_obs) + liar_obs
                if self.dual_gamma:
                    if exploit_bad_obs is not None:
                        exploit_bad_obs = list(exploit_bad_obs) + liar_obs
                    if explore_bad_obs is not None:
                        explore_bad_obs = list(explore_bad_obs) + liar_obs

        # Weights
        w_good = self._compute_weights(good_obs, budget)
        w_bad = self._compute_weights(bad_obs, budget)

        good_models = self._build_models([o[0] for o in good_obs], w_good, "good")
        bad_models = self._build_models([o[0] for o in bad_obs], w_bad, "bad")

        exploit_models = None
        explore_models = None
        if (
            self.dual_gamma
            and exploit_good_obs is not None
            and explore_good_obs is not None
        ):
            w_exploit_good = self._compute_weights(exploit_good_obs, budget)
            w_exploit_bad = self._compute_weights(exploit_bad_obs or [], budget)
            w_explore_good = self._compute_weights(explore_good_obs, budget)
            w_explore_bad = self._compute_weights(explore_bad_obs or [], budget)
            exploit_models = (
                self._build_models(
                    [o[0] for o in exploit_good_obs], w_exploit_good, "exploit_good"
                ),
                self._build_models(
                    [o[0] for o in exploit_bad_obs or []], w_exploit_bad, "exploit_bad"
                ),
            )
            explore_models = (
                self._build_models(
                    [o[0] for o in explore_good_obs], w_explore_good, "explore_good"
                ),
                self._build_models(
                    [o[0] for o in explore_bad_obs or []], w_explore_bad, "explore_bad"
                ),
            )

        recent_good_models = None
        recent_bad_models = None
        if self.multi_scale and n_obs >= 2:
            recent_n = max(2, int(self.recent_frac * n_obs))
            recent_obs = sorted_obs[-recent_n:]
            n_recent_good = self._calculate_n_good(len(recent_obs))
            recent_good_obs = recent_obs[:n_recent_good]
            recent_bad_obs = recent_obs[n_recent_good:]
            if self.atpe:
                recent_bad_obs = self._filter_bad_observations_atpe(
                    list(recent_bad_obs)
                )
            w_rg = self._compute_weights(recent_good_obs, budget)
            w_rb = self._compute_weights(recent_bad_obs, budget)
            recent_good_models = self._build_models(
                [o[0] for o in recent_good_obs], w_rg, "recent_good"
            )
            recent_bad_models = self._build_models(
                [o[0] for o in recent_bad_obs], w_rb, "recent_bad"
            )

        # 3. Sampling
        pool_size = max(self._effective_n_ei_candidates(), n_candidates * 8)
        prior_blend = self._startup_blend_weight(n_obs) if self.smooth_startup else 0.0
        attempts = self._constraint_attempts(pool_size)

        candidates: List[Dict[str, Any]] = []
        score_list: List[float] = []

        # Strategy definitions: List of (n_samples, good_source, bad_source)
        # good_source: models dict OR None (implies prior)
        strategies = []

        if self.multi_scale:
            n_exploit = int(pool_size * self.exploit_frac)
            n_explore = int(pool_size * self.explore_frac)
            n_global = max(0, pool_size - n_exploit - n_explore)

            gm_exploit, bm_exploit = good_models, bad_models
            if exploit_models:
                gm_exploit, bm_exploit = exploit_models

            strategies.append((n_exploit, gm_exploit, bm_exploit))

            if recent_good_models and recent_bad_models:
                strategies.append((n_explore, recent_good_models, recent_bad_models))
            elif explore_models:
                gm_explore, bm_explore = explore_models
                strategies.append((n_explore, gm_explore, bm_explore))
            else:
                # fallback if explore not avail: add to global
                n_global += n_explore

            if n_global > 0:
                strategies.append((n_global, None, None))  # Prior

        elif self.dual_gamma and exploit_models and explore_models:
            n_exploit = max(0, int(pool_size * self.exploit_candidate_frac))
            n_explore = max(0, pool_size - n_exploit)
            strategies.append((n_exploit, exploit_models[0], exploit_models[1]))
            strategies.append((n_explore, explore_models[0], explore_models[1]))
        else:
            strategies.append((pool_size, good_models, bad_models))

        # Execute strategies
        for n_samp, gm, bm in strategies:
            if n_samp <= 0:
                continue

            if gm is None:
                # Prior sampling
                sampler = self._sample_prior
                # For scoring global samples, we use main models, scaled down
                models_for_score = (good_models, bad_models)
                score_factor = 0.5
            else:
                # Model sampling
                sampler = lambda: self._sample_startup_blend(gm, prior_blend)
                models_for_score = (gm, bm)
                score_factor = 1.0

            for _ in range(n_samp):
                cand = self._generate_candidate(sampler, attempts)
                score = self._acq_ratio(cand, models_for_score[0], models_for_score[1])
                candidates.append(cand)
                score_list.append(score * score_factor)

        scores = np.array(score_list, dtype=float)

        if self.hybrid_local_search and minimize is not None and len(candidates) > 0:
            candidates, scores = self._local_search_candidates(
                candidates, scores, good_models, bad_models
            )

        if n_candidates > 1:
            if self.use_qlogei or self.use_qnei:
                top_idx = self.fantasized_batch_selector.select(
                    candidates,
                    scores,
                    n_candidates,
                    observations=self.observations,
                    budget=budget,
                    acq_fn=lambda c, gm, bm: float(self._acq_ratio(c, gm, bm)),
                    refit_fn=self._split_and_build_models,
                    predict_fn=self._predict_mu_sigma_from_obs,
                    rng=self._rng,
                    ei_k=self.ei_k,
                )
            else:
                top_idx = self.batch_selector.select(candidates, scores, n_candidates)
        else:
            # simple top-k
            # scores are high is good (l(x)/g(x)?) wait acq_ratio is l(x)/g(x) typically
            # Wait, standard TPE maximizes EI ~ l(x)/g(x).
            # Let's check _acq_ratio direction.
            # _acq_ratio returns l(x)/g(x)?
            # In original code: `top_idx = list(np.argsort(scores)[-n_candidates:][::-1])`
            # So HIGHER is better.
            if len(scores) < n_candidates:
                top_idx = list(range(len(scores)))
            else:
                top_idx = list(np.argsort(scores)[-n_candidates:][::-1])

        result = [candidates[j] for j in top_idx]
        self._maybe_log_constraint_stats(stat_before)
        if return_scores:
            if return_scores == "aligned":
                aligned = np.array([scores[j] for j in top_idx], dtype=float)
                return result, aligned
            # If plain return_scores is True/string, return raw pool scores?
            # Original code: `return result, scores`.
            # But `scores` was size of pool.
            # Wait, `suggest` usually expects scores for the returned candidates if aligned.
            # But the signature says `Tuple[List, np.ndarray]`.
            # If `return_scores` is plain True, maybe it returns scores of result?
            # Original behavior: `return result, scores` where `scores` matches `pool`.
            # This seems dangerous if caller expects scores aligned with result.
            # But logic says: `if return_scores == "aligned": ... else: return result, scores`.
            # So "aligned" specifically requests result-aligned scores.
            # Default True returns all pool scores (maybe for debugging?).
            # I will keep original behavior.
            return result, scores
        return result

    def _split_and_build_models(
        self,
        obs: List[Tuple[Dict[str, Any], float, Optional[float]]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not obs:
            return {}, {}
        losses = np.asarray([o[1] for o in obs], dtype=float)
        loss_scale = float(np.median(losses)) if losses.size > 0 else 1.0
        sorted_obs = self.observation_store.sort_for_split(
            obs,
            loss_scale,
            atpe_filter=self._atpe_filter_observations if self.atpe else None,
        )
        n_obs = len(sorted_obs)
        if n_obs == 0:
            return {}, {}
        n_good = self._calculate_n_good(n_obs)
        if self.atpe:
            n_good = self._adaptive_gamma_atpe(n_obs, n_good)
        good_obs, bad_obs = self.observation_store.split_good_bad(sorted_obs, n_good)
        if self.atpe:
            bad_obs = self._filter_bad_observations_atpe(list(bad_obs))
        good_models = self._build_models([o[0] for o in good_obs], None, "good")
        bad_models = self._build_models([o[0] for o in bad_obs], None, "bad")
        return good_models, bad_models

    def _predict_mu_sigma_from_obs(
        self,
        config: Dict[str, Any],
        obs: List[Tuple[Dict[str, Any], float, Optional[float]]],
        k: int = 10,
    ) -> Tuple[float, float]:
        if not obs:
            return 0.0, 1.0
        dists = []
        losses = []
        for cfg, loss, _ in obs:
            dists.append(self._config_distance(config, cfg))
            losses.append(float(loss))
        dists = np.asarray(dists, dtype=float)
        losses = np.asarray(losses, dtype=float)
        k = max(1, min(int(k), len(losses)))
        idx = np.argpartition(dists, k - 1)[:k]
        sel_d = dists[idx]
        sel_l = losses[idx]
        tau = float(np.median(sel_d)) + 1e-8
        w = np.exp(-(sel_d**2) / (tau**2))
        w = safe_normalize(w)
        mu = float(np.sum(w * sel_l))
        var = float(np.sum(w * (sel_l - mu) ** 2))
        sigma = math.sqrt(max(var, 1e-12))
        return mu, sigma

    def _blocked_params(self, obs: List[Any]) -> Dict[str, Any]:
        if not self.atpe:
            return {}

        losses = np.array([o[1] for o in obs])
        if len(losses) < max(5, self.blocking_min_obs):
            return {}
        if self.blocking_unblock_after and len(losses) >= self.blocking_unblock_after:
            return {}

        blocked = {}
        # Try import scipy
        try:
            from scipy.stats import spearmanr
        except ImportError:
            return {}

        for param in self.param_names:
            p_info = self.param_info[param]

            # Extract values
            vals = []
            valid_losses = []
            for o in obs:
                if param in o[0]:
                    vals.append(o[0][param])
                    valid_losses.append(o[1])

            if len(vals) < max(5, self.blocking_min_obs):
                continue

            valid_losses = np.array(valid_losses)

            best_val = None

            if p_info.typ in ("float", "int"):
                # Spearman
                try:
                    corr, _ = spearmanr(vals, valid_losses)
                    if abs(corr) > self.blocking_threshold:
                        # Strong correlation.
                        idx_best = np.argmin(valid_losses)
                        best_val = vals[idx_best]
                        blocked[param] = best_val
                except Exception:
                    pass

            elif p_info.typ == "choice":
                # ANOVA-ish (more aggressive): between-group variance vs total variance
                val_groups = defaultdict(list)
                for v, l in zip(vals, valid_losses):
                    val_groups[v].append(l)

                if len(val_groups) < 2:
                    continue

                total_var = np.var(valid_losses)
                if total_var < 1e-12:
                    continue

                values_list = list(val_groups.keys())
                group_means = np.array([np.mean(val_groups[v]) for v in values_list])
                group_counts = np.array(
                    [len(val_groups[v]) for v in values_list], dtype=float
                )
                overall_mean = float(np.mean(valid_losses))
                between_var = float(
                    np.sum(group_counts * (group_means - overall_mean) ** 2)
                    / max(np.sum(group_counts), 1.0)
                )
                ratio = between_var / max(total_var, 1e-12)
                if ratio > self.blocking_threshold:
                    best_cat_idx = int(np.argmin(group_means))
                    best_cat = values_list[best_cat_idx]
                    blocked[param] = best_cat
        return blocked

    def _calculate_n_good(self, n_obs: int) -> int:
        return int(self.gamma_strategy_obj.n_good(n_obs))

    def _effective_n_ei_candidates(self) -> int:
        dim = max(1, len(self.param_names))
        scaled = int(self.n_ei_candidates * max(1.0, math.sqrt(dim)))
        return max(self.n_ei_candidates, min(scaled, 4096))

    def _sample_quasi_random(
        self, n: int, attempts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if attempts is None:
            attempts = self._constraint_attempts(n)
        if qmc is None or self.quasi_random_method != "lhs":
            return [
                self._generate_candidate(self._sample_prior, attempts) for _ in range(n)
            ]

        d = len(self.param_names)
        sampler = qmc.LatinHypercube(d=d, seed=self._rng)
        u = sampler.random(n)
        results: List[Dict[str, Any]] = []
        for i in range(n):
            cfg = self._config_from_unit(u[i])
            if self.hard_constraints and not self._hard_constraints_satisfied(cfg):
                cfg = self._generate_candidate(self._sample_prior, attempts)
            results.append(cfg)
        return results

    def _config_from_unit(self, u: np.ndarray) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for i, param in enumerate(self.ancestral_order):
            info = self.param_info[param]
            ui = float(u[i]) if i < len(u) else float(self._rng.random())
            if info.condition_parent is not None and not self._is_param_active(
                cfg, param
            ):
                continue
            if info.typ == "float":
                lo, hi = self._float_bounds_in_model_space(param)
                x = lo + ui * (hi - lo)
                cfg[param] = self._inv_transform(x, param)
            elif info.typ == "int":
                lo, hi = int(info.rng[0]), int(info.rng[1])
                idx = int(math.floor(ui * (hi - lo + 1)))
                cfg[param] = int(_clamp(lo + idx, lo, hi))
            else:
                support = list(info.rng)
                if not support:
                    continue
                idx = int(math.floor(ui * len(support)))
                idx = max(0, min(idx, len(support) - 1))
                cfg[param] = support[idx]
        return cfg

    def _adaptive_gamma(self, n_obs: int) -> int:
        if self.gamma_strategy == "watanabe2023":
            if n_obs <= 20:
                return max(1, int(0.25 * n_obs))
            if n_obs <= 100:
                return max(1, int(0.15 * n_obs))
            return max(1, int(0.10 * n_obs))
        if n_obs < 20:
            return max(1, int(0.18 * n_obs))

        recent_losses = [o[1] for o in self.observations[-12:]]
        if len(recent_losses) < 4:
            return max(1, int(0.18 * n_obs))

        best_recent = min(recent_losses)
        if best_recent < np.percentile(recent_losses, 35):
            gamma_frac = 0.22 + 0.04 * self._rng.random()
        elif best_recent > np.percentile(recent_losses, 65):
            gamma_frac = 0.10 + 0.06 * self._rng.random()
        else:
            gamma_frac = 0.15 + 0.05 * self._rng.random()

        return max(1, int(gamma_frac * n_obs))

    def _local_search_candidates(
        self,
        candidates: List[Dict[str, Any]],
        scores: np.ndarray,
        good_models: Dict[str, Any],
        bad_models: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        if minimize is None:
            return candidates, scores
        if len(candidates) == 0:
            return candidates, scores
        k = max(1, min(self.local_search_top_k, len(candidates)))
        top_idx = np.argsort(scores)[-k:][::-1]

        for idx in top_idx:
            base = candidates[idx]
            float_params = [
                p
                for p in self.ancestral_order
                if p in base and self.param_info[p].typ == "float"
            ]
            if not float_params:
                continue

            x0 = []
            bounds = []
            for p in float_params:
                lo, hi = self._float_bounds_in_model_space(p)
                x0.append(self._transform(float(base[p]), p))
                bounds.append((lo, hi))
            x0 = np.asarray(x0, dtype=float)

            def _obj(x: np.ndarray) -> float:
                cfg = dict(base)
                for j, p in enumerate(float_params):
                    cfg[p] = self._inv_transform(float(x[j]), p)
                if self.hard_constraints and not self._hard_constraints_satisfied(cfg):
                    return 1e9
                return -float(self._acq_ratio(cfg, good_models, bad_models))

            try:
                res = minimize(
                    _obj,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max(1, int(self.local_search_steps))},
                )
            except Exception:
                continue

            if not res.success:
                continue

            new_cfg = dict(base)
            for j, p in enumerate(float_params):
                new_cfg[p] = self._inv_transform(float(res.x[j]), p)
            if self.hard_constraints and not self._hard_constraints_satisfied(new_cfg):
                continue

            new_score = float(self._acq_ratio(new_cfg, good_models, bad_models))
            if new_score > scores[idx]:
                candidates[idx] = new_cfg
                scores[idx] = new_score

        return candidates, scores

    def _gp_ucb_score(self, config: Dict[str, Any]) -> Optional[float]:
        if GaussianProcessRegressor is None or RBF is None or WhiteKernel is None:
            return None
        n_obs = len(self.observations)
        if n_obs < 8 or n_obs > self.gp_ucb_max_obs:
            return None

        # Build dataset from continuous params only
        cont_params = self._continuous_params()
        if not cont_params:
            return None

        X = []
        y = []
        for cfg, loss, _ in self.observations:
            vec = self._config_to_vector(cfg, cont_params)
            if vec is None:
                continue
            X.append(vec)
            y.append(float(loss))

        if len(X) < 8:
            return None

        try:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            kernel = RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(
                noise_level=1e-3
            )
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
            gp.fit(X, y)
            xq = self._config_to_vector(config, cont_params)
            if xq is None:
                return None
            mu, std = gp.predict(np.asarray([xq], dtype=float), return_std=True)
            if std is None or len(std) == 0:
                return None
            # For minimization, lower mean is better: convert to "score"
            return float(std[0])
        except Exception:
            return None

    def _startup_blend_weight(self, n_obs: int) -> float:
        if n_obs < 12:
            return float(max(0.0, 0.7 - 0.05 * n_obs))
        if n_obs < 25:
            return float(max(0.0, 0.25 - 0.012 * (n_obs - 12)))
        return 0.0

    def _sample_startup_blend(
        self, good_models: Dict[str, Any], prior_weight: float
    ) -> Dict[str, Any]:
        if prior_weight <= 0:
            return self._sample_from_models(good_models)
        if self._rng.random() < prior_weight:
            return self._sample_prior()
        return self._sample_from_models(good_models)

    def _liar_value(self, losses: List[float]) -> float:
        if not losses:
            return 1.0
        arr = np.asarray(losses, dtype=float)
        if self.liar_adaptive and arr.size < max(1, self.liar_adaptive_threshold):
            strategy = self.liar_adaptive_strategy_small
        else:
            strategy = self.liar_strategy
        if strategy == "optuna_worst":
            strategy = "worst"
        if strategy == "hypervolume":
            strategy = "worst"
        if strategy == "median":
            return float(np.median(arr))
        if strategy == "worst":
            return float(np.max(arr))
        if strategy == "best":
            return float(np.min(arr))
        if strategy == "quantile":
            q = float(min(max(self.liar_quantile, 0.0), 1.0))
            return float(np.quantile(arr, q))
        if strategy == "random":
            lo, hi = float(np.min(arr)), float(np.max(arr))
            return float(self._rng.uniform(lo, hi))
        return float(np.mean(arr))

    # ────────────────────────────────────────────────────────────────────────────
    #   New helper: normalized config distance for diversity
    # ────────────────────────────────────────────────────────────────────────────

    def _config_distance(self, c1: Dict, c2: Dict) -> float:
        vec1, vec2 = [], []
        for p in self.param_names:
            if p in c1 and p in c2:
                info = self.param_info[p]
                if info.typ == "float":
                    lo, hi = self._float_bounds_in_model_space(p)
                    v1 = (self._transform(c1[p], p) - lo) / max(hi - lo, 1e-12)
                    v2 = (self._transform(c2[p], p) - lo) / max(hi - lo, 1e-12)
                elif info.typ == "int":
                    lo, hi = info.rng
                    v1 = (c1[p] - lo) / max(hi - lo, 1)
                    v2 = (c2[p] - lo) / max(hi - lo, 1)
                else:  # choice
                    v1 = 0.0 if c1[p] == c2[p] else 1.0
                    v2 = 0.0
                vec1.append(v1)
                vec2.append(v2)
        if not vec1:
            return 0.0
        return float(np.linalg.norm(np.array(vec1) - np.array(vec2)))

    def _local_penalization_score(
        self,
        candidate: Dict[str, Any],
        selected_so_far: List[Dict[str, Any]],
        base_score: float,
    ) -> float:
        penalty = 1.0
        for prev in selected_so_far:
            dist = self._config_distance(candidate, prev)
            penalty *= 1.0 / (1.0 + dist**self.penalization_power)
        return float(base_score) * float(penalty)

    def _build_batch_selector(self) -> BatchSelector:
        key = self.batch_strategy
        if key == "ts":
            return ThompsonSamplingSelector()
        if key in {"local_penalization", "lp"}:
            return LocalPenalizationSelector(
                self._config_distance, self.penalization_power
            )
        # default diversity
        return GreedyDiversitySelector(self._config_distance)

    def _sample_with_constraints(
        self, sampler: Callable[[], Dict[str, Any]], attempts: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        if not self.hard_constraints:
            return sampler()
        if attempts is None:
            attempts = self._constraint_attempts(None)
        else:
            attempts = max(1, int(attempts))
        for _ in range(attempts):
            cfg = sampler()
            self.constraint_stats["hard_attempts"] += 1
            if self._hard_constraints_satisfied(cfg):
                return cfg
            self.constraint_stats["hard_rejects"] += 1
        if self.verbose and self.constraint_logging:
            print(f"[TPE] hard-constraint attempts exhausted ({attempts})")
        return None

    def _sample_with_constraints_ctpe(
        self, sampler: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not self.hard_constraints and not self.soft_constraints:
            return sampler()
        max_attempts = max(1, int(self.constraint_rejection_max))
        repair_attempts = max(0, int(self.constraint_repair_attempts))
        for _ in range(max_attempts):
            cfg = sampler()
            self.constraint_stats["hard_attempts"] += 1
            if self._hard_constraints_satisfied(cfg):
                return cfg
            self.constraint_stats["hard_rejects"] += 1
            if repair_attempts <= 0:
                continue
            repaired = self._repair_candidate(cfg, repair_attempts)
            if repaired is not None and self._hard_constraints_satisfied(repaired):
                return repaired
        if self.verbose and self.constraint_logging:
            print("[TPE] c-TPE constraints: falling back to prior")
        return self._sample_prior()

    def _hard_constraints_satisfied(self, config: Dict[str, Any]) -> bool:
        if not self.hard_constraints:
            return True
        return all(
            self._eval_constraint(fn, config) <= 0 for fn in self.hard_constraints
        )

    def _soft_constraint_violation(self, config: Dict[str, Any]) -> float:
        if not self.soft_constraints:
            return 0.0
        return float(
            sum(
                max(0.0, self._eval_constraint(fn, config))
                for fn in self.soft_constraints
            )
        )

    def _eval_constraint(self, fn: Callable, config: Dict[str, Any]) -> float:
        try:
            return float(fn(config))
        except Exception:
            return float("inf")

    def _repair_candidate(
        self, config: Dict[str, Any], attempts: int
    ) -> Optional[Dict[str, Any]]:
        if attempts <= 0:
            return None
        cfg = dict(config)
        for _ in range(attempts):
            p = self._rng.choice(self.param_names)
            cfg[p] = self._sample_prior_value(p)
            if self._hard_constraints_satisfied(cfg):
                return cfg
        return None

    def _maybe_log_constraint_stats(self, stat_before: Dict[str, int]) -> None:
        if not self.constraint_logging or not self.hard_constraints or not self.verbose:
            return
        attempts = self.constraint_stats["hard_attempts"] - stat_before.get(
            "hard_attempts", 0
        )
        rejects = self.constraint_stats["hard_rejects"] - stat_before.get(
            "hard_rejects", 0
        )
        if attempts <= 0:
            return
        rate = rejects / max(attempts, 1)
        print(f"[TPE] hard-constraint rejects: {rejects}/{attempts} ({rate:.1%})")

    def _constraint_attempts(self, pool_size: Optional[int]) -> int:
        base = max(1, int(self.constraint_max_attempts))
        if pool_size is None:
            return base
        adaptive = min(100, max(base, int(pool_size * 2)))
        return adaptive

    # -------------------------- Internals ---------------------------------

    def _compute_weights(
        self,
        obs: List[Tuple[Dict[str, Any], float, Optional[float]]],
        budget: Optional[float],
    ) -> Optional[np.ndarray]:
        if self.weights_func is not None:
            try:
                w = np.asarray(self.weights_func(obs), dtype=float)
                if w.size != len(obs):
                    return None
                return self._normalize_weights(w)
            except Exception:
                return None
        base_w = None

        # Non-uniform weighting strategy (recommended over uniform for TPE).
        if self.weight_strategy in {"loss_rank", "ei", "ei_weighted"} and len(obs) > 0:
            losses = np.asarray([o[1] for o in obs], dtype=float)
            rank = np.empty_like(losses, dtype=float)
            rank[np.argsort(losses)] = np.arange(losses.size, dtype=float)
            tau = max(1.0, losses.size / 5.0)
            base_w = np.exp(-rank / tau)

        if budget is not None:
            budgets = [o[2] for o in obs]
            if not any(b is None for b in budgets):
                b = np.asarray(budgets, dtype=float)
                if b.size > 0:
                    if self.max_budget is not None and self.max_budget > 0:
                        mx = float(self.max_budget)
                    else:
                        mx = float(np.max(b))
                    if mx > 0:
                        b_w = (b / mx) ** self.weight_power
                        base_w = b_w if base_w is None else (base_w * b_w)
        if base_w is None:
            if self.weight_strategy == "uniform":
                return None
            base_w = np.ones(len(obs), dtype=float)

        if self.time_weight_decay is not None:
            idx_map = {id(o): i for i, o in enumerate(self.observations)}
            n_total = max(1, len(self.observations))
            decay = float(self.time_weight_decay)
            decay_w = np.ones(len(obs), dtype=float)
            for i, o in enumerate(obs):
                idx = idx_map.get(id(o), n_total - 1)
                age = (n_total - 1) - idx
                decay_w[i] = math.exp(-decay * max(0, age))
            base_w = base_w * decay_w

        if np.sum(base_w) <= 0:
            return None
        return safe_normalize(base_w).astype(float)

    def _apply_param_options(self, info: _ParamInfo, options: Dict[str, Any]) -> None:
        condition = options.get("condition")
        parent = options.get("parent")
        values = options.get("values")
        value = options.get("value")

        if condition is not None and isinstance(condition, dict):
            parent = condition.get("parent", parent)
            values = condition.get("values", values)
            if value is None and "value" in condition:
                value = condition.get("value")

        if parent is not None:
            info.condition_parent = str(parent)
            if values is None and value is not None:
                values = [value]
            if values is not None:
                info.condition_values = set(
                    values if isinstance(values, (list, tuple, set)) else [values]
                )

        joint = options.get("joint")
        if info.condition_parent is not None and joint:
            if isinstance(joint, str):
                info.joint_group = joint
            else:
                info.joint_group = f"{info.condition_parent}__joint"

    def _topo_sort_params(self, params: List[str]) -> List[str]:
        remaining = set(params)
        ordered: List[str] = []
        while remaining:
            progressed = False
            for p in list(remaining):
                parent = self.param_info[p].condition_parent
                if parent is None or parent in ordered:
                    ordered.append(p)
                    remaining.remove(p)
                    progressed = True
            if not progressed:
                ordered.extend([p for p in params if p in remaining])
                break
        return ordered

    def _build_ancestral_order(self) -> List[str]:
        params = list(self.param_names)
        graph = {p: [] for p in params}
        indegree = {p: 0 for p in params}
        for p in params:
            parent = self.param_info[p].condition_parent
            if parent is not None and parent in graph:
                graph[parent].append(p)
                indegree[p] += 1

        order: List[str] = []
        queue = [p for p in params if indegree[p] == 0]
        queue.sort(key=lambda x: params.index(x))
        while queue:
            current = queue.pop(0)
            order.append(current)
            for child in graph.get(current, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
            queue.sort(key=lambda x: params.index(x))

        if len(order) != len(params):
            remaining = [p for p in params if p not in order]
            order.extend(remaining)
        return order

    def _collect_joint_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = defaultdict(list)
        if not self.joint_conditional:
            return {}
        for name, info in self.param_info.items():
            if info.joint_group:
                groups[info.joint_group].append(name)
        return dict(groups)

    def _prior_mix_for_param(self, param: str) -> float:
        cp = self.consider_prior
        if isinstance(cp, dict):
            cp = cp.get(param, True)
        if not bool(cp):
            return 0.0
        w = self.prior_mix_weight
        if isinstance(w, dict):
            w = w.get(param, 0.2)
        try:
            base = float(w)
        except Exception:
            base = 0.0
        base = float(min(max(base, 0.0), 1.0))
        if base <= 0.0:
            return 0.0

        n = len(self.observations)
        # Faster decay than previous default to reduce prior dominance late.
        decay = max(self.prior_mix_min, float(np.exp(-n / self.prior_mix_decay)))
        w = base * decay
        return float(min(max(w, self.prior_mix_min), base))

    def _condition_satisfied(self, config: Dict[str, Any], info: _ParamInfo) -> bool:
        if info.condition_parent is None:
            return True
        if info.condition_parent not in config:
            return False
        parent_val = config[info.condition_parent]
        if info.condition_values is None:
            return True
        return parent_val in info.condition_values

    def _is_param_active(self, config: Dict[str, Any], param: str) -> bool:
        return self._condition_satisfied(config, self.param_info[param])

    def _filter_configs(
        self,
        configs: List[Dict[str, Any]],
        weights: Optional[np.ndarray],
        param: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        idxs = [
            i
            for i, c in enumerate(configs)
            if self._is_param_active(c, param) and param in c
        ]
        filtered = [configs[i] for i in idxs]
        if weights is None:
            return filtered, None
        w = np.asarray([weights[i] for i in idxs], dtype=float)
        w = self._normalize_weights(w)
        return filtered, w

    def _log_uniform_pdf(self, lo: float, hi: float) -> float:
        span = max(float(hi) - float(lo), 1e-12)
        return -safe_log(span)

    def _sample_prior_value(self, param: str) -> Any:
        info = self.param_info[param]
        if info.typ == "float":
            if info.transform == "log":
                lo, hi = info.log_range
                v = 10.0 ** self._rng.uniform(lo, hi)
                return float(v)
            lo, hi = float(info.rng[0]), float(info.rng[1])
            return float(self._rng.uniform(lo, hi))
        if info.typ == "int":
            lo, hi = int(info.rng[0]), int(info.rng[1])
            return int(self._rng.integers(lo, hi + 1))
        if info.typ == "choice":
            choices = list(info.rng)
            return choices[int(self._rng.integers(0, len(choices)))]
        raise ValueError(f"Unknown type: {info.typ}")

    def _sample_prior_model_space(self, param: str) -> float:
        info = self.param_info[param]
        if info.typ == "float":
            if info.transform == "log":
                lo, hi = info.log_range
                return float(self._rng.uniform(lo, hi))
            lo, hi = float(info.rng[0]), float(info.rng[1])
            return float(self._rng.uniform(lo, hi))
        if info.typ == "int":
            lo, hi = int(info.rng[0]), int(info.rng[1])
            return float(self._rng.uniform(lo, hi))
        raise ValueError(f"Prior model space not defined for type: {info.typ}")

    def _log_prior_model_space(self, param: str) -> float:
        info = self.param_info[param]
        if info.typ == "float":
            lo, hi = self._float_bounds_in_model_space(param)
            return self._log_uniform_pdf(lo, hi)
        if info.typ == "int":
            lo, hi = int(info.rng[0]), int(info.rng[1])
            span = max(hi - lo + 1, 1)
            return -safe_log(float(span))
        return 0.0

    def _transform(self, x: float, param: str) -> float:
        info = self.param_info[param]
        if info.transform == "log":
            return float(math.log10(float(x)))
        return float(x)

    def _inv_transform(self, x: float, param: str) -> float:
        info = self.param_info[param]
        if info.transform == "log":
            return float(10.0 ** float(x))
        return float(x)

    def _float_bounds_in_model_space(self, param: str) -> Tuple[float, float]:
        info = self.param_info[param]
        rng = info.rng
        if info.transform == "log":
            assert info.log_range is not None
            return info.log_range
        lo, hi = float(rng[0]), float(rng[1])
        return lo, hi

    def _sample_prior(self) -> Dict[str, Any]:
        """
        Prior sampler for startup (uniform, log-uniform, categorical uniform).
        """
        config: Dict[str, Any] = {}
        for param in self.param_names:
            info = self.param_info[param]
            if info.condition_parent is not None:
                continue
            config[param] = self._sample_prior_value(param)

        for param in self.param_names:
            info = self.param_info[param]
            if info.condition_parent is None:
                continue
            if not self._is_param_active(config, param):
                continue
            config[param] = self._sample_prior_value(param)
        return config

    def _build_models(
        self,
        configs: List[Dict[str, Any]],
        weights: Optional[np.ndarray],
        model_tag: str = "default",
    ) -> Dict[str, Any]:
        """
        Build per-parameter density models for TPE:
        - float: Gaussian mixture centered at observations (bandwidth shared)
        - int (small domain): discrete pmf smoothing
        - int (large domain): treat like float in transformed space then round
        - choice: categorical smoothed
        """
        models: Dict[str, Any] = {}

        for param in self.param_names:
            info = self.param_info[param]
            cfgs, w = self._filter_configs(configs, weights, param)
            prior_w = self._prior_mix_for_param(param)

            if info.typ == "float":
                models[param] = self._build_float_model(param, cfgs, w, prior_w)
            elif info.typ == "choice":
                models[param] = self._build_choice_model(param, cfgs, w)
            elif info.typ == "int":
                models[param] = self._build_int_model(param, cfgs, w, prior_w)
            else:
                raise ValueError(f"Unknown type: {info.typ}")

        joint_models = self._build_joint_models(configs, weights, model_tag=model_tag)
        if joint_models:
            models["__joint__"] = joint_models

        return models

    def _float_model_kwargs(
        self, param: str, prior_w: float, **overrides: Any
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "param": param,
            "prior_w": float(prior_w),
            "mu": None,
            "vals": None,
            "bw": None,
            "w": None,
            "min_bandwidth": self.min_bandwidth,
            "local_bandwidth": self.local_bandwidth,
            "local_bandwidth_k": self.local_bandwidth_k,
            "transform": self._transform,
            "inv_transform": self._inv_transform,
            "float_bounds": self._float_bounds_in_model_space,
            "sample_prior_model_space": self._sample_prior_model_space,
            "log_prior_model_space": self._log_prior_model_space,
            "sample_mixture_1d": self._sample_mixture_1d,
            "local_bw_func": self._local_bandwidth,
            "rng": self._rng,
        }
        kwargs.update(overrides)
        return kwargs

    def _int_model_kwargs(
        self, param: str, lo: int, hi: int, prior_w: float, **overrides: Any
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "param": param,
            "lo": int(lo),
            "hi": int(hi),
            "prior_w": float(prior_w),
            "mu": None,
            "vals": None,
            "bw": None,
            "w": None,
            "local_bandwidth": self.local_bandwidth,
            "local_bandwidth_k": self.local_bandwidth_k,
            "sample_prior_model_space": self._sample_prior_model_space,
            "log_prior_model_space": self._log_prior_model_space,
            "sample_mixture_1d": self._sample_mixture_1d,
            "local_bw_func": self._local_bandwidth,
            "rng": self._rng,
        }
        kwargs.update(overrides)
        return kwargs

    def _build_float_model(
        self,
        param: str,
        cfgs: List[Dict[str, Any]],
        w: Optional[np.ndarray],
        prior_w: float,
    ) -> FloatModel:
        vals = np.array(
            [self._transform(float(c[param]), param) for c in cfgs],
            dtype=float,
        )
        if vals.size == 0:
            return FloatModel(
                kind="prior_only_float", **self._float_model_kwargs(param, prior_w)
            )
        if vals.size == 1:
            return FloatModel(
                kind="single_float",
                **self._float_model_kwargs(param, prior_w, mu=float(vals[0])),
            )

        if self.consider_endpoints:
            lo, hi = self._float_bounds_in_model_space(param)
            vals = np.concatenate([vals, [lo, hi]])
            if w is not None:
                mean_w = float(np.mean(w)) if len(w) > 0 else 1.0
                w = np.concatenate([w, [mean_w, mean_w]])
                w = self._normalize_weights(w)

        lmbda = None
        if self.warping and vals.size >= 10:
            try:
                # Find optimal lambda
                lmbda = float(yeojohnson_normmax(vals))
                # Restrict lambda to reasonable range for stability
                lmbda = float(max(min(lmbda, 3.0), -3.0))
                vals = yeojohnson_forward(vals, lmbda)
            except Exception:
                lmbda = None

        bw = self._bandwidth(vals, param) * self._get_bandwidth_factor(param)
        bw = max(float(bw), self.min_bandwidth)
        return FloatModel(
            kind="kde_float",
            **self._float_model_kwargs(
                param, prior_w, vals=vals, bw=bw, w=w, warp_lmbda=lmbda
            ),
        )

    def _build_choice_model(
        self, param: str, cfgs: List[Dict[str, Any]], w: Optional[np.ndarray]
    ) -> CatModel:
        info = self.param_info[param]
        probs = self._kernel_categorical_probs(
            [c[param] for c in cfgs], list(info.rng), w, param
        )
        return CatModel(kind="cat", probs=probs, rng=self._rng)

    def _build_int_model(
        self,
        param: str,
        cfgs: List[Dict[str, Any]],
        w: Optional[np.ndarray],
        prior_w: float,
    ) -> IntModel:
        info = self.param_info[param]
        lo, hi = int(info.rng[0]), int(info.rng[1])
        domain_size = hi - lo + 1
        if domain_size <= self.int_discrete_threshold:
            probs = self._smoothed_categorical_probs(
                [int(c[param]) for c in cfgs],
                list(range(lo, hi + 1)),
                w,
            )
            return IntModel(
                kind="int_discrete",
                probs=probs,
                **self._int_model_kwargs(param, lo, hi, 0.0),
            )

        vals = np.array([float(c[param]) for c in cfgs], dtype=float)
        if vals.size == 0:
            return IntModel(
                kind="prior_only_int", **self._int_model_kwargs(param, lo, hi, prior_w)
            )
        if vals.size == 1:
            return IntModel(
                kind="single_int",
                mu=float(vals[0]),
                **self._int_model_kwargs(param, lo, hi, prior_w),
            )

        bw = self._bandwidth(
            vals, param, numeric_bounds=(lo, hi)
        ) * self._get_bandwidth_factor(param)
        bw = max(float(bw), 1.0)
        return IntModel(
            kind="kde_int",
            vals=vals,
            bw=bw,
            w=w,
            **self._int_model_kwargs(param, lo, hi, prior_w),
        )

    def _smoothed_categorical_probs(
        self, values: List[Any], support: List[Any], weights: Optional[np.ndarray]
    ) -> Dict[Any, float]:
        counts = defaultdict(float)
        total = 0.0
        if weights is None:
            for v in values:
                counts[v] += 1.0
            total = float(len(values))
        else:
            for v, w in zip(values, weights):
                counts[v] += float(w)
            total = float(np.sum(weights))

        # smoothing prior
        prior = float(self.prior_weight)
        for s in support:
            counts[s] += prior
        total += prior * float(len(support))

        # normalize
        probs = {k: (float(v) / total) for k, v in counts.items()}
        # ensure all support keys exist
        for s in support:
            probs.setdefault(s, prior / total)
        return probs

    def _categorical_bw_for_param(self, param: str) -> float:
        bw = self.categorical_bw
        if isinstance(bw, dict):
            bw = bw.get(param, 1.0)
        try:
            bw = float(bw)
        except Exception:
            bw = 1.0
        return float(max(bw, 1e-6))

    def _categorical_kernel_row(self, d: np.ndarray, bw: float) -> np.ndarray:
        if self.categorical_kernel == "epanechnikov":
            u = d / bw
            k = 0.75 * (1.0 - u**2)
            k[u > 1.0] = 0.0
            return k
        return np.exp(-d / bw)

    def _kernel_categorical_probs(
        self,
        values: List[Any],
        support: List[Any],
        weights: Optional[np.ndarray],
        param: str,
    ) -> Dict[Any, float]:
        """
        Distance-aware categorical kernel smoothing using index distance.
        Kernel: exp(-|i-j| / bw), properly normalized per location.
        """
        if not support:
            return {}

        idx_of = {v: i for i, v in enumerate(support)}
        n_cats = len(support)
        bw = self._categorical_bw_for_param(param)

        # Precompute kernel rows and row sums
        kernel_rows = np.zeros((n_cats, n_cats), dtype=float)

        dist_func = self.categorical_distance_func.get(param)

        def _distance_mode() -> str:
            if dist_func is not None:
                return "custom"
            mode = self.categorical_distance
            if mode in {"embedding", "learned"}:
                if len(values) < self.categorical_embedding_min_obs:
                    return "index"
            return mode

        mode = _distance_mode()
        emb_version = self._categorical_embedding_version.get(param, 0)
        support_key = tuple(support)

        cache_key = (
            param,
            support_key,
            mode,
            float(bw),
            self.categorical_kernel,
            emb_version,
        )

        cached = self._categorical_kernel_cache.get(cache_key)
        if cached is not None:
            kernel_rows, kernel_sums = cached
        else:

            def _dist_vector(i: int) -> np.ndarray:
                if dist_func is not None:
                    return np.array(
                        [dist_func(support[i], support[j]) for j in range(n_cats)],
                        dtype=float,
                    )

                if mode == "hamming":
                    return np.array([0.0 if i == j else 1.0 for j in range(n_cats)])

                if mode == "levenshtein":

                    def _lev(a: Any, b: Any) -> float:
                        if not isinstance(a, str) or not isinstance(b, str):
                            return float("inf")
                        if a == b:
                            return 0.0
                        la, lb = len(a), len(b)
                        dp = list(range(lb + 1))
                        for ia in range(1, la + 1):
                            prev = dp[0]
                            dp[0] = ia
                            for ib in range(1, lb + 1):
                                cur = dp[ib]
                                cost = 0 if a[ia - 1] == b[ib - 1] else 1
                                dp[ib] = min(
                                    dp[ib] + 1,  # deletion
                                    dp[ib - 1] + 1,  # insertion
                                    prev + cost,  # substitution
                                )
                                prev = cur
                        return float(dp[lb])

                    return np.array(
                        [_lev(support[i], support[j]) for j in range(n_cats)],
                        dtype=float,
                    )

                if mode == "permutation_l1":

                    def _perm_l1(a: Any, b: Any) -> float:
                        if not isinstance(a, (list, tuple)) or not isinstance(
                            b, (list, tuple)
                        ):
                            return float("inf")
                        if len(a) != len(b):
                            return float("inf")
                        try:
                            return float(
                                sum(abs(int(x) - int(y)) for x, y in zip(a, b))
                            )
                        except Exception:
                            return float("inf")

                    return np.array(
                        [_perm_l1(support[i], support[j]) for j in range(n_cats)],
                        dtype=float,
                    )

                if mode == "graph":
                    # Optional: only attempt if graph exists
                    if hasattr(self, "categorical_graphs"):
                        graph = self.categorical_graphs.get(param)
                        if graph is not None:
                            try:
                                import networkx as nx

                                return np.array(
                                    [
                                        float(
                                            nx.shortest_path_length(
                                                graph, support[i], support[j]
                                            )
                                        )
                                        for j in range(n_cats)
                                    ],
                                    dtype=float,
                                )
                            except Exception:
                                pass  # fall back to index distance

                # embedding / learned
                if mode in {"embedding", "learned"}:
                    emb = self._categorical_embeddings.get(param)
                    if (
                        emb is None
                        and mode == "learned"
                        and len(values) >= self.categorical_embedding_min_obs
                    ):
                        self._learn_categorical_embedding(param, values)
                        emb = self._categorical_embeddings.get(param)
                    if emb is not None:
                        ei = emb.get(support[i])
                        if ei is not None:
                            return np.array(
                                [
                                    float(np.linalg.norm(ei - emb.get(s, ei)))
                                    for s in support
                                ],
                                dtype=float,
                            )

                # default: index distance
                return np.abs(np.arange(n_cats) - i).astype(float)

            # Compute kernels
            for i in range(n_cats):
                d = _dist_vector(i)
                kernel_rows[i] = self._categorical_kernel_row(d, bw)

            kernel_sums = np.sum(kernel_rows, axis=1)
            self._categorical_kernel_cache[cache_key] = (kernel_rows, kernel_sums)

        # Weighted kernel density
        density = np.zeros(n_cats, dtype=float)

        if weights is None:
            weights_iter = [1.0] * len(values)
        else:
            weights_iter = [float(w) for w in weights]

        for v, w in zip(values, weights_iter):
            if v not in idx_of or w <= 0:
                continue
            i = idx_of[v]
            kernel = kernel_rows[i]
            density += w * (kernel / kernel_sums[i])  # normalize per location

        # Laplace smoothing (prior pseudo-counts)
        prior = float(self.prior_weight)
        density += prior
        total = float(np.sum(density))

        if total <= 0:
            uniform = 1.0 / n_cats
            return {s: uniform for s in support}

        probs = {support[i]: float(density[i] / total) for i in range(n_cats)}

        # Optional combinatorial mode: sharpen + narrow support
        if (
            hasattr(self, "combinatorial_mode")
            and self.combinatorial_mode
            and len(values) >= getattr(self, "combinatorial_min_obs", 20)
        ):
            if n_cats > 1:
                # Sharpen distribution
                scale = min(
                    getattr(self, "combinatorial_sharpen_max", 5.0),
                    1.0 + 0.25 * math.log(n_cats + 1.0),
                )
                for k in probs:
                    probs[k] = float(max(probs[k], 1e-12) ** scale)

                s = float(sum(probs.values()))
                if s > 0:
                    for k in probs:
                        probs[k] /= s

                # Range-narrowing: keep only top-probability categories
                keep_frac = getattr(self, "combinatorial_keep_frac", 0.3)
                min_keep = getattr(self, "combinatorial_min_keep", 3)
                keep = max(int(math.ceil(keep_frac * n_cats)), min_keep)
                keep = min(max(keep, 1), n_cats)

                top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:keep]
                keep_set = {k for k, _ in top}

                for k in list(probs.keys()):
                    if k not in keep_set:
                        probs[k] = 0.0

                s = float(sum(probs.values()))
                if s > 0:
                    for k in probs:
                        probs[k] /= s

        return probs

    def _learn_categorical_embedding(
        self, param: str, observed_values: List[Any]
    ) -> None:
        unique = list({v for v in observed_values if v is not None})
        if len(unique) < 2:
            return
        dim = max(2, int(self.categorical_embedding_dim))
        emb = self._rng.normal(size=(len(unique), dim)) * 0.01
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

        cooccur = defaultdict(lambda: defaultdict(int))
        for cfg, _, _ in self.observations:
            vals = [
                cfg.get(p)
                for p in self.param_names
                if self.param_info[p].typ == "choice"
            ]
            for i, v1 in enumerate(vals):
                if v1 is None:
                    continue
                for v2 in vals[i + 1 :]:
                    if v2 is None:
                        continue
                    if v1 in unique and v2 in unique:
                        cooccur[v1][v2] += 1
                        cooccur[v2][v1] += 1

        lr = 0.5
        for _ in range(10):
            grad = np.zeros_like(emb)
            for i, v1 in enumerate(unique):
                for j, v2 in enumerate(unique):
                    if i == j:
                        continue
                    x = emb[i] - emb[j]
                    d = np.linalg.norm(x)
                    if d < 1e-8:
                        continue
                    sim = math.exp(cooccur[v1][v2])
                    grad[i] += sim * x / (d + 1e-8)
            emb -= lr * grad
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

        self._categorical_embeddings[param] = dict(zip(unique, emb))
        self._categorical_embedding_version[param] = (
            self._categorical_embedding_version.get(param, 0) + 1
        )

    def _build_joint_models(
        self,
        configs: List[Dict[str, Any]],
        weights: Optional[np.ndarray],
        model_tag: str = "default",
    ) -> Dict[str, Any]:
        multivariate_enabled = False
        if self.multivariate_arg == "auto":
            # Heuristic: enable if we have enough observations relative to dimensionality
            n_obs = len(configs)
            dim = len(self.param_names)
            # Need decent data to estimate covariance (e.g. 5x - 10x dims)
            if n_obs >= max(30, 5 * dim):
                multivariate_enabled = True
        else:
            multivariate_enabled = bool(self.multivariate_arg)

        if not self.joint_conditional and not multivariate_enabled:
            return {}

        joint_groups: Dict[str, Any] = {}
        joint_params: set = set()

        # ─── 1. Explicit conditional joint groups ───────────────────────────────
        for group_name, params in self.joint_groups.items():
            if group_name == "__all_continuous__":
                continue  # handled below
            if not params:
                continue
            parent = self.param_info[params[0]].condition_parent
            if parent is None:
                continue
            if any(self.param_info[p].condition_parent != parent for p in params):
                continue

            cont_params = [
                p
                for p in params
                if self.param_info[p].typ in ("float", "int")
                and (
                    self.param_info[p].typ == "float"
                    or (
                        self.param_info[p].rng[1] - self.param_info[p].rng[0] + 1
                        > self.int_discrete_threshold
                    )
                )
            ]

            if len(cont_params) < 2:
                continue

            joint_params.update(cont_params)
            groups = self._collect_grouped_vectors(
                configs, cont_params, parent, weights
            )
            models_by_val = {}
            for parent_val, (vecs, vec_ws) in groups.items():
                if not vecs:
                    models_by_val[parent_val] = {
                        "kind": "prior_only",
                        "params": cont_params,
                    }
                    continue
                cov_key = f"{model_tag}:{group_name}:{parent_val}"
                model = self._build_multivariate_density(
                    cont_params, vecs, vec_ws, cov_key=cov_key
                )
                models_by_val[parent_val] = model

            joint_groups[group_name] = {
                "parent": parent,
                "params": cont_params,
                "values": models_by_val,
            }

        # ─── 2. Global multivariate (when multivariate=True and no explicit joints) ───
        if multivariate_enabled and "__all_continuous__" in self.joint_groups:
            cont_params = self.joint_groups["__all_continuous__"]
            joint_params.update(cont_params)

            vecs, vec_ws = self._collect_joint_vectors(configs, cont_params, weights)

            if vecs:
                cov_key = f"{model_tag}:__all_continuous__:global"
                model = self._build_multivariate_density(
                    cont_params, vecs, vec_ws, cov_key=cov_key
                )

                joint_groups["__all_continuous__"] = {
                    "parent": None,
                    "params": cont_params,
                    "values": {"global": model},
                }

        # ─── 3. Grouped multivariate (when multivariate=True and group_multivariate=True) ───
        if multivariate_enabled and self.group_multivariate:
            cont_params = self._continuous_params()
            remaining = [p for p in cont_params if p not in joint_params]
            subspaces = self._maximal_active_subspaces(configs, remaining)
            for idx, params in enumerate(subspaces):
                if len(params) < 2:
                    continue
                cov_key = f"{model_tag}:__group_{idx}__:global"
                model = self._build_multivariate_model(
                    params, configs, weights, cov_key=cov_key
                )
                if model is None:
                    continue
                name = f"__group_{idx}__"
                joint_groups[name] = {
                    "parent": None,
                    "params": params,
                    "values": {"global": model},
                }
                joint_params.update(params)

        if not joint_groups:
            return {}

        return {"groups": joint_groups, "params": joint_params}

    def _continuous_params(self) -> List[str]:
        return [
            p
            for p in self.param_names
            if self.param_info[p].typ in ("float", "int")
            and (
                self.param_info[p].typ == "float"
                or (
                    self.param_info[p].rng[1] - self.param_info[p].rng[0] + 1
                    > self.int_discrete_threshold
                )
            )
        ]

    def _config_to_vector(
        self, cfg: Dict[str, Any], cont_params: List[str]
    ) -> Optional[List[float]]:
        vec: List[float] = []
        for p in cont_params:
            if p not in cfg or not self._is_param_active(cfg, p):
                return None
            info = self.param_info[p]
            if info.typ == "float":
                vec.append(self._transform(float(cfg[p]), p))
            else:
                vec.append(float(cfg[p]))
        return vec

    def _collect_joint_vectors(
        self,
        configs: List[Dict[str, Any]],
        cont_params: List[str],
        weights: Optional[np.ndarray],
    ) -> Tuple[List[List[float]], Optional[List[float]]]:
        vecs: List[List[float]] = []
        vec_ws: List[float] = []
        for idx, cfg in enumerate(configs):
            vec = self._config_to_vector(cfg, cont_params)
            if vec is None:
                continue
            vecs.append(vec)
            if weights is not None:
                vec_ws.append(weights[idx])
        return vecs, (vec_ws if weights is not None else None)

    def _collect_grouped_vectors(
        self,
        configs: List[Dict[str, Any]],
        cont_params: List[str],
        parent: str,
        weights: Optional[np.ndarray],
    ) -> Dict[Any, Tuple[List[List[float]], Optional[List[float]]]]:
        # Helper for conditional groups (including nested conditionals).
        by_val: Dict[Any, List[List[float]]] = defaultdict(list)
        by_val_w: Dict[Any, List[float]] = defaultdict(list)
        for idx, cfg in enumerate(configs):
            if parent not in cfg or not self._condition_satisfied(
                cfg, self.param_info[cont_params[0]]
            ):
                continue
            vec = self._config_to_vector(cfg, cont_params)
            if vec is None:
                continue
            by_val[cfg[parent]].append(vec)
            if weights is not None:
                by_val_w[cfg[parent]].append(float(weights[idx]))

        grouped: Dict[Any, Tuple[List[List[float]], Optional[List[float]]]] = {}
        for parent_val, vecs in by_val.items():
            vec_ws = by_val_w[parent_val] if weights is not None else None
            grouped[parent_val] = (vecs, vec_ws)
        return grouped

    @staticmethod
    def _normalize_weights(weights: Optional[Any]) -> Optional[np.ndarray]:
        if weights is None:
            return None
        w = np.asarray(weights, dtype=float)
        if w.size == 0:
            return None
        return safe_normalize(w)

    def _local_bandwidth(
        self,
        x_query: float,
        centers: np.ndarray,
        base_bw: float,
        k: int = 7,
    ) -> float:
        centers = np.asarray(centers, dtype=float)
        if centers.size <= k:
            return float(base_bw)
        dists = np.abs(centers - x_query)
        knn_dist = np.partition(dists, k)[k]
        denom = float(np.median(dists)) if np.median(dists) > 0 else 1.0
        return float(base_bw * (knn_dist / denom + 0.1))

    def _filter_bad_observations_atpe(self, bad_obs: List[Any]) -> List[Any]:
        if not bad_obs:
            return bad_obs
        if len(bad_obs) < 5:
            return bad_obs

        filter_type = str(self.atpe_params.get("bad_filter_type", "hybrid")).lower()
        threshold = float(self.atpe_params.get("bad_filter_threshold", 3.0))
        filtered: List[Any] = list(bad_obs)

        # Step 1: robust MAD-based clipping of pathological bad outliers.
        if filter_type in {"hybrid", "aggressive", "auto", "zscore", "mad", "mad_z"}:
            losses = np.asarray([o[1] for o in filtered], dtype=float)
            z_scores = self._robust_zscore(losses)
            if threshold > 0:
                mask = z_scores < threshold
            else:
                mask = np.abs(z_scores) >= abs(threshold)
            kept = [filtered[i] for i in range(len(filtered)) if mask[i]]
            if kept:
                filtered = kept

        # Step 2: keep representatives per cluster.
        if filter_type in {"hybrid", "aggressive", "auto", "clustering"}:
            clustered = self._cluster_representatives(filtered)
            if clustered:
                filtered = clustered

        # Step 3: age-decay + EI-weighted keep for the remaining bad set.
        if filter_type in {"hybrid", "aggressive", "auto", "age", "ei", "ei_weighted"}:
            filtered = self._age_ei_weighted_keep(filtered)

        return filtered if filtered else bad_obs

    def _robust_zscore(self, losses: np.ndarray) -> np.ndarray:
        vals = np.asarray(losses, dtype=float)
        if vals.size == 0:
            return np.asarray([], dtype=float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad < 1e-12:
            sigma = float(np.std(vals))
            if sigma < 1e-12:
                return np.zeros(vals.size, dtype=float)
            mu = float(np.mean(vals))
            return (vals - mu) / sigma
        return 0.6745 * (vals - med) / mad

    def _config_feature_vector(self, cfg: Dict[str, Any]) -> np.ndarray:
        vec: List[float] = []
        for p in self.param_names:
            info = self.param_info[p]
            if p not in cfg or not self._is_param_active(cfg, p):
                vec.append(-1.0)
                continue
            if info.typ == "float":
                lo, hi = self._float_bounds_in_model_space(p)
                x = self._transform(float(cfg[p]), p)
                vec.append((x - lo) / max(hi - lo, 1e-12))
            elif info.typ == "int":
                lo, hi = int(info.rng[0]), int(info.rng[1])
                vec.append((float(cfg[p]) - float(lo)) / max(float(hi - lo), 1.0))
            else:
                support = list(info.rng)
                if not support:
                    vec.append(0.0)
                else:
                    idx = support.index(cfg[p]) if cfg[p] in support else 0
                    vec.append(float(idx) / max(1.0, float(len(support) - 1)))
        return np.asarray(vec, dtype=float)

    def _cluster_representatives(self, obs: List[Any]) -> List[Any]:
        if len(obs) < 6:
            return obs
        try:
            from scipy.cluster.vq import kmeans2
        except ImportError:
            return obs

        features = np.asarray(
            [self._config_feature_vector(o[0]) for o in obs], dtype=float
        )
        losses = np.asarray([float(o[1]) for o in obs], dtype=float)
        if features.ndim != 2 or features.shape[0] != len(obs):
            return obs

        loss_z = self._robust_zscore(losses).reshape(-1, 1)
        data = np.concatenate([features, loss_z], axis=1)

        n_obs = len(obs)
        k = max(2, int(math.sqrt(n_obs)))
        k = min(k, n_obs - 1)
        if k < 2:
            return obs

        try:
            centroids, labels = kmeans2(data, k, minit="points")
        except Exception:
            return obs

        representatives: List[Any] = []
        for c in range(k):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            d = np.linalg.norm(data[idxs] - centroids[c], axis=1)
            best_idx = int(idxs[int(np.argmin(d))])
            representatives.append(obs[best_idx])

        if not representatives:
            return obs
        return sorted(representatives, key=lambda o: float(o[1]))

    def _ei_proxy_for_config(self, cfg: Dict[str, Any], best_loss: float) -> float:
        mu, sigma = self._predict_mu_sigma(cfg, k=self.ei_k)
        improvement = max(0.0, float(best_loss) - float(mu))
        if sigma <= 1e-12:
            return improvement
        z = improvement / sigma
        return float(improvement * norm.cdf(z) + sigma * norm.pdf(z))

    def _age_ei_weighted_keep(self, obs: List[Any]) -> List[Any]:
        if len(obs) <= 5:
            return obs

        idx_map = {id(o): i for i, o in enumerate(self.observations)}
        n_total = max(1, len(self.observations))
        age_decay = float(
            self.atpe_params.get(
                "bad_age_decay",
                self.time_weight_decay if self.time_weight_decay is not None else 0.02,
            )
        )
        ei_weight = float(self.atpe_params.get("bad_ei_weight", 1.0))
        keep_frac = float(self.atpe_params.get("bad_keep_frac", 0.65))
        keep_frac = min(max(keep_frac, 0.1), 1.0)
        min_keep = int(
            self.atpe_params.get("bad_min_keep", max(5, int(0.2 * len(obs))))
        )
        n_keep = min(len(obs), max(min_keep, int(math.ceil(keep_frac * len(obs)))))

        best_loss = (
            min(float(o[1]) for o in self.observations)
            if self.observations
            else min(float(o[1]) for o in obs)
        )
        ei_vals = np.asarray(
            [self._ei_proxy_for_config(o[0], best_loss) for o in obs],
            dtype=float,
        )
        ei_max = float(np.max(ei_vals)) if ei_vals.size > 0 else 0.0
        ei_norm = ei_vals / max(ei_max, 1e-12)

        scores = np.zeros(len(obs), dtype=float)
        for i, o in enumerate(obs):
            idx = idx_map.get(id(o), n_total - 1)
            age = max(0, (n_total - 1) - idx)
            age_w = math.exp(-age_decay * age)
            scores[i] = float(age_w * (1.0 + ei_weight * ei_norm[i]))

        top_idx = np.argsort(scores)[-n_keep:]
        kept = [obs[i] for i in top_idx]
        return sorted(kept, key=lambda o: float(o[1]))

    def _predict_mu_sigma(
        self, config: Dict[str, Any], k: int = 10
    ) -> Tuple[float, float]:
        if not self.observations:
            return 0.0, 1.0
        dists = []
        losses = []
        for cfg, loss, _ in self.observations:
            dist = self._config_distance(config, cfg)
            dists.append(dist)
            losses.append(float(loss))
        dists = np.asarray(dists, dtype=float)
        losses = np.asarray(losses, dtype=float)
        k = max(1, min(int(k), len(losses)))
        idx = np.argpartition(dists, k - 1)[:k]
        sel_d = dists[idx]
        sel_l = losses[idx]
        tau = float(np.median(sel_d)) + 1e-8
        w = np.exp(-(sel_d**2) / (tau**2))
        w = safe_normalize(w)
        mu = float(np.sum(w * sel_l))
        var = float(np.sum(w * (sel_l - mu) ** 2))
        sigma = math.sqrt(max(var, 1e-12))
        return mu, sigma

    def _bounds_and_bandwidths(
        self, arr: np.ndarray, cont_params: List[str]
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        bounds: List[Tuple[float, float]] = []
        bws: List[float] = []
        for dim, p in enumerate(cont_params):
            lo, hi = (
                self._float_bounds_in_model_space(p)
                if self.param_info[p].typ == "float"
                else tuple(map(float, self.param_info[p].rng))
            )
            bounds.append((lo, hi))
            bw = self._bandwidth(arr[:, dim], p, (lo, hi))
            bws.append(max(bw * self._get_bandwidth_factor(p), self.min_bandwidth))
        return bounds, np.asarray(bws)

    def _get_max_bandwidth_factor(self, param: str) -> Optional[float]:
        bf = self.max_bandwidth_factor
        if bf is None:
            return None
        if isinstance(bf, dict):
            bf = bf.get(param, None)
        if bf is None:
            return None
        try:
            bf = float(bf)
        except Exception:
            return None
        if bf <= 0:
            return None
        return bf

    def _build_copula_joint(
        self,
        cont_params: List[str],
        vecs: List[List[float]],
        vec_ws: Optional[List[float]],
    ) -> Optional[Dict[str, Any]]:
        if not vecs or len(cont_params) < 2:
            return None

        arr = np.asarray(vecs, dtype=float)
        n_obs, d = arr.shape
        if n_obs < 2:
            return None

        marginal_models = []
        uniform_data = np.zeros_like(arr)

        w = self._normalize_weights(vec_ws)
        for dim, param in enumerate(cont_params):
            values = arr[:, dim]
            bw = self._bandwidth(values, param) * self._get_bandwidth_factor(param)
            bw = max(float(bw), self.min_bandwidth)
            marginal_models.append({"centers": values, "bw": bw, "weights": w})

            for i, x in enumerate(values):
                cdf = self._kde_cdf(x, values, bw, w)
                uniform_data[i, dim] = np.clip(cdf, 1e-6, 1 - 1e-6)

        gaussian_data = norm.ppf(uniform_data)
        corr = np.corrcoef(gaussian_data, rowvar=False)
        corr = self._ensure_psd(corr)

        return {
            "kind": "copula_joint",
            "marginal_models": marginal_models,
            "copula_corr": corr,
            "params": cont_params,
            "prior_w": max(self._prior_mix_for_param(p) for p in cont_params),
        }

    def _kde_cdf(
        self,
        x: float,
        centers: np.ndarray,
        bw: float,
        weights: Optional[np.ndarray],
    ) -> float:
        key = (
            "cdf",
            id(centers),
            float(bw),
            float(x),
            id(weights) if weights is not None else None,
        )
        if self.kde_cache_size > 0 and key in self._kde_cache:
            self._kde_cache.move_to_end(key)
            return float(self._kde_cache[key])
        z = (x - centers) / bw
        cdf_vals = norm.cdf(z)
        if weights is None:
            val = float(np.mean(cdf_vals))
        else:
            w = safe_normalize(weights)
            val = float(np.sum(w * cdf_vals))
        if self.kde_cache_size > 0:
            self._kde_cache[key] = val
            self._kde_cache.move_to_end(key)
            if len(self._kde_cache) > self.kde_cache_size:
                self._kde_cache.popitem(last=False)
        return float(val)

    def _kde_pdf(
        self,
        x: float,
        centers: np.ndarray,
        bw: float,
        weights: Optional[np.ndarray],
    ) -> float:
        key = (
            "pdf",
            id(centers),
            float(bw),
            float(x),
            id(weights) if weights is not None else None,
        )
        if self.kde_cache_size > 0 and key in self._kde_cache:
            self._kde_cache.move_to_end(key)
            return float(self._kde_cache[key])
        z = (x - centers) / bw
        pdf_vals = norm.pdf(z) / max(bw, 1e-12)
        if weights is None:
            val = float(np.mean(pdf_vals))
        else:
            w = safe_normalize(weights)
            val = float(np.sum(w * pdf_vals))
        if self.kde_cache_size > 0:
            self._kde_cache[key] = val
            self._kde_cache.move_to_end(key)
            if len(self._kde_cache) > self.kde_cache_size:
                self._kde_cache.popitem(last=False)
        return float(val)

    def _ensure_psd(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            return np.eye(1)
        try:
            eigvals, eigvecs = np.linalg.eigh(mat)
            eigvals = np.clip(eigvals, 1e-8, None)
            return make_positive_definite(eigvecs @ np.diag(eigvals) @ eigvecs.T)
        except Exception:
            return np.eye(mat.shape[0])

    def _weighted_quantile(
        self, values: np.ndarray, q: float, weights: Optional[np.ndarray]
    ) -> float:
        v = np.asarray(values, dtype=float)
        q = float(min(max(q, 0.0), 1.0))
        if weights is None:
            return float(np.quantile(v, q))
        w = np.asarray(weights, dtype=float)
        if w.size != v.size or np.sum(w) <= 0:
            return float(np.quantile(v, q))
        idx = np.argsort(v)
        v = v[idx]
        w = w[idx]
        w = safe_normalize(w)
        cdf = np.cumsum(w)
        return float(np.interp(q, cdf, v))

    def _maximal_active_subspaces(
        self, configs: List[Dict[str, Any]], params: List[str]
    ) -> List[List[str]]:
        """
        Build maximal active subspaces based on observed configs.
        Each subspace is the set of continuous params that are active together.
        """
        if not params:
            return []
        param_set = set(params)
        active_sets = set()
        for cfg in configs:
            active = frozenset(
                p
                for p in params
                if p in cfg and self._is_param_active(cfg, p) and p in param_set
            )
            if len(active) >= 2:
                active_sets.add(active)

        if not active_sets:
            return []

        # Keep maximal sets (not strict subset of another)
        sorted_sets = sorted(active_sets, key=lambda s: (-len(s), sorted(s)))
        maximal: List[frozenset] = []
        for s in sorted_sets:
            if any(s < m for m in maximal):
                continue
            maximal.append(s)

        return [sorted(list(s)) for s in maximal]

    def _build_multivariate_model(
        self,
        cont_params: List[str],
        configs: List[Dict[str, Any]],
        weights: Optional[np.ndarray],
        cov_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        vecs, vec_ws = self._collect_joint_vectors(configs, cont_params, weights)
        return self._build_multivariate_density(
            cont_params, vecs, vec_ws, cov_key=cov_key
        )

    def _build_multivariate_density(
        self,
        cont_params: List[str],
        vecs: List[List[float]],
        vec_ws: Optional[List[float]],
        cov_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not vecs:
            return None

        arr = np.asarray(vecs)
        n_obs, d = arr.shape
        if self.copula_joint and d >= 2 and n_obs >= 2:
            model = self._build_copula_joint(cont_params, vecs, vec_ws)
            if model is not None:
                return model

        bounds, bw_vec = self._bounds_and_bandwidths(arr, cont_params)
        cov = self._robust_covariance(
            arr,
            bw_vec,
            self.min_bandwidth,
            cov_key=cov_key,
            cont_params=cont_params,
        )
        covs = None
        if self.joint_per_component_bandwidth and n_obs >= 2:
            covs = self._component_covariances(arr, cov, bw_vec)

        w = self._normalize_weights(vec_ws)
        w_joint = max(self._prior_mix_for_param(p) for p in cont_params)

        kind = "joint_single" if n_obs == 1 else "joint_kde"
        data_key = "mu" if kind == "joint_single" else "centers"
        data_value = arr[0] if kind == "joint_single" else arr

        return {
            "kind": kind,
            data_key: data_value,
            "cov": cov,
            "covs": covs,
            "w": w,
            "params": cont_params,
            "bounds": bounds,
            "prior_w": w_joint,
        }

    def _component_covariances(
        self, arr: np.ndarray, base_cov: np.ndarray, bw_vec: np.ndarray
    ) -> Optional[np.ndarray]:
        n_obs, d = arr.shape
        if n_obs < 2:
            return None

        rule = self.joint_component_bw_rule
        if rule == "silverman":
            factor = ((n_obs * (d + 2.0)) / 4.0) ** (-1.0 / (d + 4.0))
        else:
            factor = n_obs ** (-1.0 / (d + 4.0))
        factor = float(max(0.2, min(2.0, factor)))

        if self.joint_component_bw_neighbors is not None:
            k = int(self.joint_component_bw_neighbors)
        else:
            k = max(4, int(math.sqrt(n_obs)))
        k = max(2, min(k, n_obs))

        covs = np.zeros((n_obs, d, d), dtype=float)
        local_floor = np.diag(np.maximum(bw_vec**2, self.min_bandwidth**2))
        for i in range(n_obs):
            dists = np.linalg.norm(arr - arr[i], axis=1)
            idx = np.argpartition(dists, k - 1)[:k]
            local = arr[idx]
            if local.shape[0] > 1:
                local_cov = np.cov(local, rowvar=False)
            else:
                local_cov = local_floor.copy()
            if np.any(~np.isfinite(local_cov)):
                local_cov = local_floor.copy()
            cov_i = (1.0 - factor) * local_cov + factor * base_cov
            cov_i += local_floor * 0.1
            covs[i] = self._ensure_psd(cov_i)
        return covs

    def _bandwidth(
        self,
        values: np.ndarray,
        param: str,
        numeric_bounds: Optional[Tuple[float, float]] = None,
    ) -> float:
        v = np.asarray(values, dtype=float)
        n = int(v.size)
        if n <= 1:
            return 1.0

        scale = _robust_scale_1d(v)
        # Use magnitude of exponent as Scott-like n^{-a}; this keeps behavior
        # stable whether user passes 0.7 or -0.7.
        exp = abs(self.bandwidth_exponent)
        scott = max(n ** (-exp), 1e-3)
        base_bw = scale * scott

        # Per-parameter adjustment (simple MAD-based)
        if n > 5:
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            learned_factor = mad / scale if scale > 0 else 1.0
            learned_factor = max(0.5, min(3.0, learned_factor))
        else:
            learned_factor = 1.0

        bw = base_bw * learned_factor

        # range-based floor
        if numeric_bounds:
            lo, hi = numeric_bounds
        else:
            lo, hi = (
                self._float_bounds_in_model_space(param)
                if self.param_info[param].typ == "float"
                else (min(v), max(v))
            )
        span = max(hi - lo, 1e-12)
        # Magic-clip-like inflation
        magic_clip = 0.0
        if self.consider_magic_clip:
            magic_clip = 0.1 * span / max(n, 1)
            if n < 5:
                magic_clip = max(magic_clip, 0.1 * span)

        bw_floor = self.min_bandwidth_factor * span
        bw = max(bw, magic_clip, bw_floor, self.min_bandwidth)
        max_factor = self._get_max_bandwidth_factor(param)
        if max_factor is not None:
            bw = min(bw, max_factor * span)
        return float(bw)

    def _get_bandwidth_factor(self, param: str) -> float:
        bf = self.bandwidth_factor
        if isinstance(bf, dict):
            bf = bf.get(param, 1.0)
        try:
            bf = float(bf)
        except Exception:
            bf = 1.0
        return float(max(bf, 1e-6))

    def _robust_covariance(
        self,
        arr: np.ndarray,
        bw_vec: np.ndarray,
        min_bw: float,
        cov_key: Optional[str] = None,
        cont_params: Optional[List[str]] = None,
    ) -> np.ndarray:
        cov_emp = None
        if arr.shape[0] > 1:
            cov_emp = np.cov(arr, rowvar=False)
            if np.any(~np.isfinite(cov_emp)):
                cov_emp = None

        n_obs = arr.shape[0]
        d = arr.shape[1]
        if n_obs <= d + 2 or cov_emp is None or np.linalg.matrix_rank(cov_emp) < d:
            diag_var = np.var(arr, axis=0) if n_obs > 1 else np.ones(d)
            diag_var = np.clip(diag_var, max(1e-8, min_bw**2 * 5), None)
            cov = np.diag(diag_var * 1.5)
            cov += np.eye(d) * (min_bw**2 * 5)
        else:
            cov = 0.3 * cov_emp + 0.7 * np.diag(bw_vec**2)
            cov += np.eye(d) * (min_bw**2)
            if self.covariance_shrinkage in {"lw", "oas"}:
                cov = self._shrink_covariance(
                    cov_emp, cov, method=self.covariance_shrinkage
                )
        if self.cma_cov_adaptation and cov_key and cont_params:
            cov = self._cma_adapt_covariance(cov, arr, cov_key, tuple(cont_params))
        return cov

    def _cma_adapt_covariance(
        self,
        cov: np.ndarray,
        arr: np.ndarray,
        cov_key: str,
        cont_params: Tuple[str, ...],
    ) -> np.ndarray:
        n_obs, d = arr.shape
        if n_obs < 2 or d <= 0:
            return cov

        state_key = (cov_key, cont_params)
        state = self._cma_cov_state.get(state_key)
        if state is None:
            p_c = np.zeros(d, dtype=float)
            prev_mean = None
        else:
            p_c = np.asarray(state.get("p_c", np.zeros(d, dtype=float)), dtype=float)
            prev_mean = state.get("mean")

        mean = np.mean(arr, axis=0)
        # Evolution step: mean shift across iterations (CMA-style path driver).
        if prev_mean is None:
            step = np.zeros(d, dtype=float)
        else:
            step = mean - np.asarray(prev_mean, dtype=float)
        step_norm = float(np.linalg.norm(step))
        if step_norm > 1e-12:
            step = step / step_norm
        else:
            step = np.zeros_like(step)

        c_path = 1.0 - self.cma_path_decay
        p_c_new = (1.0 - c_path) * p_c + math.sqrt(
            max(c_path * (2.0 - c_path), 0.0)
        ) * step
        rank1 = np.outer(p_c_new, p_c_new)
        c_cov = self.cma_cov_learning_rate
        cov_new = (1.0 - c_cov) * cov + c_cov * rank1
        cov_new = self._ensure_psd(cov_new)

        self._cma_cov_state[state_key] = {"p_c": p_c_new, "cov": cov_new, "mean": mean}
        return cov_new

    def _shrink_covariance(
        self, emp_cov: np.ndarray, base_cov: np.ndarray, method: str = "lw"
    ) -> np.ndarray:
        s = np.asarray(emp_cov, dtype=float)
        if s.ndim != 2 or s.shape[0] != s.shape[1]:
            return base_cov
        d = s.shape[0]
        mu = float(np.trace(s)) / max(d, 1)
        prior = np.eye(d) * mu
        if method == "oas":
            # Approximate OAS shrinkage
            tr_s = float(np.trace(s))
            tr_s2 = float(np.trace(s @ s))
            denom = (d + 1.0 - 2.0 / max(d, 1)) * (tr_s2 - (tr_s**2) / max(d, 1))
            if denom <= 0:
                shrink = 0.0
            else:
                shrink = min(
                    1.0,
                    max(
                        0.0,
                        (tr_s2 + (tr_s**2)) / denom,
                    ),
                )
        else:
            # Lightweight Ledoit-Wolf-style shrinkage
            tr_s2 = float(np.trace(s @ s))
            if tr_s2 <= 0:
                shrink = 0.0
            else:
                shrink = min(1.0, max(0.0, ((tr_s2 - (tr_s**2) / max(d, 1)) / tr_s2)))
        return (1.0 - shrink) * s + shrink * prior

    def _sample_from_models(self, models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sample one configuration from the product of parameter models (good / l(x)).
        Handles conditional joints plus global/grouped multivariate with proper topological order.
        """
        config: Dict[str, Any] = {}
        joint = models.get("__joint__", {})
        joint_params = joint.get("params", set())

        # 1. Sample unconditional / parent parameters first (independent of joints)
        for param in self.ancestral_order:
            if self.param_info[param].condition_parent is not None:
                continue
            if param in joint_params:
                continue
            m = models.get(param)
            if m:
                config[param] = m.sample()
            else:
                config[param] = self._sample_prior_value(param)

        # 2. Collect and sort joint groups topologically
        groups_dict = joint.get("groups", {})
        if not groups_dict:
            # no joints → go straight to remaining conditionals
            pass
        else:
            # Build a dependency graph: group → its parent parameter (if any)
            group_dependencies = {}
            group_by_name = {}
            for gname, group in groups_dict.items():
                parent = group.get("parent")
                group_by_name[gname] = group
                if parent is not None:
                    group_dependencies[gname] = (
                        parent  # depends on this param being set
                    )
                else:
                    group_dependencies[gname] = None  # root group

            # Get a topological order of groups (parents before children)
            # Reuse your existing topo sort logic as approximation
            group_order = self._topo_sort_groups(
                group_dependencies, list(groups_dict.keys()), groups_dict
            )

            # 3. Sample groups in topological order
            for gname in group_order:
                group = group_by_name[gname]
                parent = group.get("parent")
                params = group["params"]
                if not params:
                    continue

                if parent is None:
                    # Global / grouped multivariate case
                    model = group["values"].get("global")
                    if model is None:
                        for p in params:
                            config[p] = self._sample_prior_value(p)
                        continue
                    vec = self._sample_joint(model)
                    for p, x in zip(params, vec):
                        info = self.param_info[p]
                        if info.typ == "float":
                            lo, hi = self._float_bounds_in_model_space(p)
                            x = _reflect_into_bounds(float(x), lo, hi)
                            config[p] = self._inv_transform(x, p)
                        else:
                            lo, hi = info.rng
                            x = int(
                                round(
                                    _reflect_into_bounds(float(x), float(lo), float(hi))
                                )
                            )
                            config[p] = int(_clamp(x, lo, hi))
                    continue

                # Conditional joint (has parent)
                if parent not in config:
                    continue  # parent not yet sampled → skip (shouldn't happen in topo order)
                if not self._condition_satisfied(config, self.param_info[params[0]]):
                    continue

                parent_val = config[parent]
                model = group["values"].get(parent_val)
                if model is None or model["kind"] == "prior_only":
                    for p in params:
                        config[p] = self._sample_prior_value(p)
                    continue

                vec = self._sample_joint(model)
                for p, x in zip(model["params"], vec):
                    info = self.param_info[p]
                    if info.typ == "float":
                        lo, hi = self._float_bounds_in_model_space(p)
                        x = _reflect_into_bounds(float(x), lo, hi)
                        config[p] = self._inv_transform(x, p)
                    else:
                        lo, hi = info.rng
                        x = int(
                            round(_reflect_into_bounds(float(x), float(lo), float(hi)))
                        )
                        config[p] = int(_clamp(x, lo, hi))

        # 4. Sample remaining conditional parameters independently
        for param in self.ancestral_order:
            if param in config:
                continue
            info = self.param_info[param]
            if not self._is_param_active(config, param):
                continue
            if param in joint_params:
                config[param] = self._sample_prior_value(param)
                continue
            m = models.get(param)
            if m:
                config[param] = m.sample()
            else:
                config[param] = self._sample_prior_value(param)

        return config

    def _topo_sort_groups(
        self,
        dependencies: Dict[str, Optional[str]],
        group_names: List[str],
        groups_by_name: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Topological sort of joint groups based on parent parameter dependencies.
        Uses parameter topo order as tie-breaker / approximation.
        """
        from collections import defaultdict, deque

        # Build graph: group → children groups that depend on it
        graph = defaultdict(list)
        indegree = {g: 0 for g in group_names}

        for g, parent_param in dependencies.items():
            if parent_param is None:
                continue
            # Find which groups have this parent_param in their params
            for other_g in group_names:
                if other_g == g:
                    continue
                if groups_by_name is not None:
                    other_params = groups_by_name.get(other_g, {}).get("params", [])
                else:
                    other_params = self.joint_groups.get(other_g, [])
                if parent_param in other_params:
                    graph[g].append(other_g)
                    indegree[other_g] += 1

        # Kahn's algorithm
        queue = deque([g for g in group_names if indegree[g] == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for child in graph[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        # Add any remaining (cycles or disconnected) in arbitrary order
        remaining = [g for g in group_names if g not in order]
        if remaining:
            # Fall back to parameter topo order as proxy
            param_order = {p: i for i, p in enumerate(self.param_names)}

            def _group_rank(g: str) -> int:
                params = (
                    (groups_by_name.get(g, {}).get("params", []))
                    if groups_by_name is not None
                    else self.joint_groups.get(g, [])
                )
                if not params:
                    return 9999
                return min(param_order.get(p, 9999) for p in params)

            remaining.sort(key=_group_rank)
            order.extend(remaining)

        return order

    def _log_likelihood(
        self, config: Dict[str, Any], models: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Compute log p(x | good models) — product over parameters + joint groups.
        Supports global/grouped multivariate with parent=None.
        """
        lp = 0.0
        joint = models.get("__joint__", {})
        joint_params = joint.get("params", set())

        # ─── Joint groups (conditional + global) ─────────────────────────────────
        for group_name, group in joint.get("groups", {}).items():
            parent = group.get("parent")
            params = group["params"]
            if not params:
                continue

            # Check if group is active
            if parent is not None:
                if parent not in config or not self._condition_satisfied(
                    config, self.param_info[params[0]]
                ):
                    continue
                parent_val = config[parent]
                model = group["values"].get(parent_val)
            else:
                # Global multivariate case
                model = group["values"].get("global")

            if model is None or model["kind"] == "prior_only":
                for p in params:
                    lp += self._log_prior_model_space(p)
                continue

            # Build vector
            vec = []
            missing = False
            for p in params:
                if p not in config:
                    missing = True
                    break
                info = self.param_info[p]
                if info.typ == "float":
                    vec.append(self._transform(float(config[p]), p))
                else:
                    vec.append(float(config[p]))
            if missing:
                lp += -50.0  # strong penalty for incomplete config
                continue

            vec = np.asarray(vec, dtype=float)

            if model["kind"] == "joint_single":
                mu = np.asarray(model["mu"], dtype=float)
                cov = np.asarray(model["cov"], dtype=float)
                log_kde = multivariate_normal.logpdf(
                    vec, mean=mu, cov=cov, allow_singular=True
                )
            elif model["kind"] == "copula_joint":
                marginals = model["marginal_models"]
                corr = np.asarray(model["copula_corr"], dtype=float)
                # marginal CDFs -> gaussian
                u = []
                log_marg = 0.0
                for j, m in enumerate(marginals):
                    xj = vec[j]
                    cdf = self._kde_cdf(xj, m["centers"], m["bw"], m["weights"])
                    u_j = np.clip(cdf, 1e-6, 1 - 1e-6)
                    u.append(u_j)
                    log_marg += safe_log(
                        self._kde_pdf(xj, m["centers"], m["bw"], m["weights"])
                    )
                z = norm.ppf(np.asarray(u))
                try:
                    inv_corr = np.linalg.inv(corr)
                    sign, logdet = np.linalg.slogdet(corr)
                    if sign <= 0:
                        logdet = 0.0
                except Exception:
                    inv_corr = np.linalg.pinv(corr)
                    logdet = 0.0
                quad = float(z.T @ (inv_corr - np.eye(len(z))) @ z)
                log_copula = -0.5 * (logdet + quad)
                log_kde = log_marg + log_copula
            else:
                centers = np.asarray(model["centers"], dtype=float)
                cov = np.asarray(model["cov"], dtype=float)
                covs = model.get("covs", None)
                w = model.get("w", None)
                if covs is not None:
                    covs = np.asarray(covs, dtype=float)
                    log_comp = np.array(
                        [
                            multivariate_normal.logpdf(
                                vec, mean=centers[i], cov=covs[i], allow_singular=True
                            )
                            for i in range(centers.shape[0])
                        ]
                    )
                else:
                    log_comp = np.array(
                        [
                            multivariate_normal.logpdf(
                                vec, mean=centers[i], cov=cov, allow_singular=True
                            )
                            for i in range(centers.shape[0])
                        ]
                    )
                if w is None:
                    log_kde = float(logsumexp(log_comp) - safe_log(len(centers)))
                else:
                    log_kde = float(logsumexp(log_comp, b=w))

            prior_w = float(model.get("prior_w", 0.0))
            if prior_w > 0:
                log_prior = sum(self._log_prior_model_space(p) for p in params)
                log_kde = float(
                    logsumexp(
                        [
                            log_kde + safe_log(1.0 - prior_w),
                            log_prior + safe_log(prior_w),
                        ]
                    )
                )

            lp += log_kde

        # ─── Independent parameters (not in any joint) ───────────────────────────
        for param in self.param_names:
            if param in joint_params:
                continue
            if not self._is_param_active(config, param):
                continue
            if param not in config:
                lp += -50.0
                continue

            m = models.get(param)
            if m is None:
                lp += self._log_prior_model_space(param)
                continue

            lp += m.log_pdf(config[param])

        return float(lp)

    def _sample_joint(self, model: Dict[str, Any]) -> np.ndarray:
        if model["kind"] == "prior_only":
            return np.asarray(
                [self._sample_prior_model_space(p) for p in model["params"]],
                dtype=float,
            )
        prior_w = float(model.get("prior_w", 0.0))
        if prior_w > 0 and self._rng.random() < prior_w:
            return np.asarray(
                [self._sample_prior_model_space(p) for p in model["params"]],
                dtype=float,
            )
        if model["kind"] == "joint_single":
            mu = np.asarray(model["mu"], dtype=float)
            cov = np.asarray(model["cov"], dtype=float)
            return self._rng.multivariate_normal(mu, cov)
        if model["kind"] == "copula_joint":
            marginals = model["marginal_models"]
            corr = np.asarray(model["copula_corr"], dtype=float)
            z = self._rng.multivariate_normal(mean=np.zeros(corr.shape[0]), cov=corr)
            u = norm.cdf(z)
            vec = []
            for j, m in enumerate(marginals):
                q = float(np.clip(u[j], 1e-6, 1 - 1e-6))
                vec.append(
                    self._weighted_quantile(m["centers"], q, m.get("weights", None))
                )
            return np.asarray(vec, dtype=float)
        centers = np.asarray(model["centers"], dtype=float)
        n = int(centers.shape[0])
        if n == 0:
            return np.asarray(
                [self._sample_prior_model_space(p) for p in model["params"]],
                dtype=float,
            )
        w = model.get("w", None)
        if w is None:
            i = int(self._rng.integers(0, n))
        else:
            w = np.asarray(w, dtype=float)
            w = safe_normalize(w)
            i = int(self._rng.choice(n, p=w))
        mu = centers[i]
        covs = model.get("covs", None)
        if covs is not None:
            cov = np.asarray(covs, dtype=float)[i]
        else:
            cov = np.asarray(model["cov"], dtype=float)
        return self._rng.multivariate_normal(mu, cov)

    def _sample_mixture_1d(
        self, centers: np.ndarray, bw: float, weights: Optional[np.ndarray]
    ) -> float:
        """
        Sample from mixture of Gaussians N(center_i, bw^2).
        If weights is provided, it should sum to 1.
        """
        c = np.asarray(centers, dtype=float)
        n = int(c.size)
        if n == 0:
            return float(self._rng.normal(0.0, max(1.0, bw)))

        if weights is None:
            i = int(self._rng.integers(0, n))
        else:
            w = np.asarray(weights, dtype=float)
            w = safe_normalize(w)
            i = int(self._rng.choice(n, p=w))
        return float(self._rng.normal(loc=c[i], scale=float(bw)))

    def _acq_ratio(
        self,
        config: Dict[str, Any],
        good_models: Dict[str, Dict[str, Any]],
        bad_models: Dict[str, Dict[str, Any]],
    ) -> float:
        """
        Acquisition score proportional to l(x) / g(x),
        optionally scaled by a lightweight EI proxy.
        """
        key = None
        if self.acq_cache_size > 0:
            key = _canonical_config_key(config)
            if key in self._acq_cache:
                return self._acq_cache[key]
        score = float(self.acquisition_strategy.score(config, good_models, bad_models))
        if self.acq_cache_size > 0 and key is not None:
            if len(self._acq_cache) >= self.acq_cache_size:
                self._acq_cache.pop(next(iter(self._acq_cache)))
            self._acq_cache[key] = score
        return score

    def _adaptive_gamma_atpe(self, n_obs: int, current_n_good: int) -> int:
        """
        Adjust n_good dynamically for ATPE.
        If we have many observations, we might want to range between 10% and 25%.
        """
        # Simple heuristic: oscillate or scale based on n_obs
        # For now, let's allow it to grow slightly more than fixed gamma if n_obs is large
        # to capture more "good" structure.
        return max(current_n_good, int(0.2 * n_obs))

    def diagnostics(self) -> Dict[str, Any]:
        tr = self.trust_region
        out: Dict[str, Any] = {
            "observations": len(self.observations),
            "trust_region_enabled": bool(self.trust_region_enabled),
            "trust_region_restart_count": int(self._trust_region_restart_count),
        }
        if tr is not None:
            out.update(
                {
                    "trust_region_length": float(tr.length),
                    "trust_region_success_count": int(tr.success_count),
                    "trust_region_failure_count": int(tr.failure_count),
                    "trust_region_min_length": float(tr.min_length),
                }
            )
        return out

    def _atpe_filter_observations(self, obs: List[Any]) -> List[Any]:
        if not self.atpe:
            return obs

        filter_type = self.atpe_params.get("filter_type", "zscore")
        try:
            threshold = float(self.atpe_params.get("filter_threshold", 1.5))
        except (ValueError, TypeError):
            threshold = 1.5

        if len(obs) < 10:
            return obs

        if filter_type == "age":
            n_keep = max(5, int(len(obs) * self.recent_frac))
            return obs[-n_keep:]

        losses = np.array([o[1] for o in obs])

        if filter_type == "zscore":
            z_scores = self._robust_zscore(losses)
            if z_scores.size == 0:
                return obs

            if threshold > 0:
                # Keep low extremes (good)
                mask = z_scores <= -threshold
            else:
                # Keep outliers
                mask = np.abs(z_scores) >= abs(threshold)

            filtered = [obs[i] for i in range(len(obs)) if mask[i]]
            if len(filtered) < 5:
                return sorted(obs, key=lambda x: x[1])[: max(5, int(len(obs) * 0.2))]
            return filtered

        elif filter_type == "clustering":
            try:
                try:
                    from scipy.cluster.vq import kmeans2
                except ImportError:
                    return obs

                cont_params = self._continuous_params()
                vecs = []
                keep_obs = []
                for o in obs:
                    cfg = o[0]
                    vec = (
                        self._config_to_vector(cfg, cont_params) if cont_params else []
                    )
                    if vec is None:
                        continue
                    # Add categorical features (embedding if available, else index)
                    for p in self.param_names:
                        info = self.param_info[p]
                        if info.typ != "choice":
                            continue
                        if p not in cfg or not self._is_param_active(cfg, p):
                            continue
                        val = cfg[p]
                        emb = None
                        if self.categorical_distance in {"embedding", "learned"}:
                            emb = self._categorical_embeddings.get(p, {}).get(val)
                        if emb is not None:
                            vec.extend([float(x) for x in emb])
                        else:
                            support = list(info.rng)
                            if support:
                                idx = support.index(val) if val in support else 0
                                vec.append(float(idx) / max(1.0, len(support) - 1))
                    vecs.append(vec)
                    keep_obs.append(o)

                if not vecs:
                    return obs

                n_obs = len(vecs)
                k = max(2, int(math.sqrt(n_obs)))

                data = np.array(vecs, dtype=float)
                if data.ndim != 2:
                    return obs

                if k >= n_obs:
                    return keep_obs

                centroids, labels = kmeans2(data, k, minit="points")

                # pick best (lowest loss) per cluster
                best_by_cluster: Dict[int, Any] = {}
                for i, label in enumerate(labels):
                    cur = keep_obs[i]
                    loss = cur[1]
                    if label not in best_by_cluster or loss < best_by_cluster[label][1]:
                        best_by_cluster[label] = cur

                reps = list(best_by_cluster.values())

                if len(reps) < 5:
                    return obs
                return reps
            except Exception:
                return obs

        return obs
