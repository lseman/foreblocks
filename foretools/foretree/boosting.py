# ================================================================
# Hybrid GBM with Histogram Trees + Neural Oblivious Trees (ODST)
# Drop-in replacement for your previous hybrid
# ================================================================

import time
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- your project imports (unchanged) --------------------------
from boosting_aux import *
from boosting_bin import *
from boosting_loss import HuberLoss, LogisticLoss, MSELoss, QuantileLoss
from boosting_tree import *

# ---- ODST & utils (expect these in your repo) ------------------
# If your ODST is in a package path, edit these two imports.
from nn import (
    ODST,  # your ODST implementation
    ModuleWithInit,
    sparsemax,  # choice/bin functions
    sparsemoid,
)


# ================================================================
# Configs
# ================================================================
@dataclass
class TreeConfig:
    max_depth: int = 6
    min_samples_split: int = 20
    min_samples_leaf: int = 5
    lambda_: float = 1.0
    gamma: float = 0.0
    alpha: float = 0.0
    max_delta_step: float = 0.0
    max_leaves: int = 31


@dataclass
class SamplingConfig:
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    use_goss: bool = True
    goss_top_rate: float = 0.2
    goss_other_rate: float = 0.1
    # NEW
    score_mode: str = "abs_g"  # {"abs_g", "abs_g_sqrt_h"}
    goss_scale: str = "both"   # {"both", "grad_only"}


# ================================================================
# (REPLACED) NodeConfig with training knobs for ODST
# ================================================================
@dataclass
class NodeConfig:
    num_trees: int = 64
    tree_depth: int = 5
    temperature: float = 1.0
    dropout_rate: float = 0.05
    # Training knobs for ODST
    lr: float = 1e-2
    epochs: int = 50
    batch_size: int = 4096
    weight_decay: float = 5e-5
    temp_min: float = 0.3
    verbose: bool = False


# ================================================================
# (NEW) NeuralObliviousTree: ODST-backed 'tree'
# ================================================================
class NeuralObliviousTree:
    """
    Drop-in 'tree' compatible with BoostRegressor.
    Trains an ODST block on Newton targets r = -g / (h + eps) with weights = h.
    """

    def __init__(self, input_dim: int, config: NodeConfig):
        self.input_dim = int(input_dim)
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_neural = True

        # ODST + small head to 1-d output
        self.odst = ODST(
            in_features=self.input_dim,
            num_trees=self.cfg.num_trees,
            depth=self.cfg.tree_depth,
            tree_dim=1,
            flatten_output=True,
            choice_function=sparsemax,
            bin_function=sparsemoid,
        )
        head_in = self.cfg.num_trees * 1
        self.head = nn.Linear(head_in, 1)
        self.drop = nn.Dropout(self.cfg.dropout_rate) if self.cfg.dropout_rate > 0 else nn.Identity()
        self.temp = nn.Parameter(torch.tensor(float(self.cfg.temperature)), requires_grad=False)
        self.net = nn.Sequential(self.odst, self.drop, self.head).to(self.device)

        self._initialized = False
        self._rescale = 1.0  # set by optional calibration

    # -------- helpers --------
    def _maybe_data_aware_init(self, X_tensor: torch.Tensor):
        if self._initialized:
            return
        with torch.no_grad():
            take = min(8192, X_tensor.shape[0])
            self.odst.initialize(X_tensor[:take])
        self._initialized = True

    def _anneal_temperature(self, ep: int, total: int):
        # cosine anneal: temperature -> cfg.temp_min
        t0, t1 = float(self.cfg.temperature), float(self.cfg.temp_min)
        cos = 0.5 * (1 + np.cos(np.pi * ep / max(1, total)))
        new_t = t1 + (t0 - t1) * cos
        self.temp.data[...] = float(new_t)

    # -------- 'tree' API --------
    def fit(self, X, grad, hess, feature_importance=None, epochs=None, lr=None):
        """
        X: (n, d) float array
        grad, hess: (n,)
        Fits weighted MSE on r = -grad / (hess + 1e-8) with weights = clip(hess, 1e-12, +inf)
        """
        self.net.train()
        X = np.asarray(X)
        g = np.asarray(grad, dtype=np.float64)
        h = np.asarray(hess, dtype=np.float64)
        r = -g / (h + 1e-8)
        w = np.clip(h, 1e-12, None)

        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).view(-1, 1)
        w_t = torch.as_tensor(w, dtype=torch.float32, device=self.device).view(-1, 1)

        self._maybe_data_aware_init(X_t)

        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=lr or self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)

        E = epochs or self.cfg.epochs
        N = X_t.shape[0]
        order = torch.arange(N, device=self.device)

        for ep in range(E):
            self._anneal_temperature(ep, E - 1)
            perm = order[torch.randperm(N, device=self.device)]
            for start in range(0, N, self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                xb, yb, wb = X_t[idx], y_t[idx], w_t[idx]
                opt.zero_grad(set_to_none=True)
                pred = self.net(xb)  # [B,1]
                loss = torch.mean(wb * (pred - yb) ** 2)
                loss.backward()
                opt.step()
            if self.cfg.verbose and (ep % 10 == 0 or ep == E - 1):
                with torch.no_grad():
                    pred_all = self.net(X_t)
                    l = torch.mean(w_t * (pred_all - y_t) ** 2).item()
                print(f"[ODST] ep={ep:03d} temp={self.temp.item():.3f} wMSE={l:.6f}")

        # (Optional) contribute to feature importance from selectors
        # if feature_importance is not None:
        #     with torch.no_grad():
        #         logits = self.odst.feature_selection_logits.detach().abs().sum(dim=(1, 2))  # [in_features]
        #         arr = logits.cpu().numpy()
        #         for j, v in enumerate(arr):
        #             feature_importance.add_gain(j, float(v))

    def post_prune_ccp(self, ccp_alpha=0.0):
        return

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            y = self.net(xt).squeeze(1).detach().cpu().numpy()
        return self._rescale * y  # include calibration scale if applied



class BoostRegressor:
    """
    Gradient boosting with histogram trees, GOSS, DART, and advanced losses.
    Now supports ODST-based NeuralObliviousTree as a drop-in 'tree'.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        # Tree parameters (backward compatible)
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 5,
        lambda_: float = 1.0,
        gamma: float = 0.0,
        alpha: float = 0.0,
        max_delta_step: float = 0.0,
        max_leaves: int = 31,
        # Sampling parameters
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        colsample_bynode: float = 1.0,
        use_goss: bool = True,
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        # Structured configs
        tree_config: Optional[TreeConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        # Advanced features
        objective: str = "reg:squarederror",
        adaptive_lr: bool = False,
        lr_schedule: str = "constant",
        focal_gamma: float = 0.0,
        elastic_net_ratio: float = 0.0,
        # Performance
        early_stopping_rounds: Optional[int] = None,
        n_jobs: int = -1,
        tree_method: str = "binned",
        binned_mode: str = "hist",
        n_bins: int = 256,
        batch_size: int = 1,
        cache_gradients: bool = True,
        # DART
        skip_drop: float = 0.5,
        normalize_type: str = "tree",
        rate_drop: float = 0.1,
        one_drop: bool = False,
        # Tree strategy
        tree_learner: str = "level",
        # Misc
        random_state: Optional[int] = None,
        verbose: bool = True,
        eval_metric: Optional[str] = None,
        monotone_constraints: Optional[Dict] = None,
        interaction_constraints: Optional[List] = None,
        use_gpu: bool = False,
        use_neural: bool = True,          # default: enabled
        enable_interactions: bool = False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.use_neural = use_neural
        self.enable_interactions = enable_interactions

        self.tree_config = tree_config or TreeConfig(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            lambda_=lambda_,
            gamma=gamma,
            alpha=alpha,
            max_delta_step=max_delta_step,
            max_leaves=max_leaves,
        )
        self.sampling_config = sampling_config or SamplingConfig(
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_goss=use_goss,
            goss_top_rate=goss_top_rate,
            goss_other_rate=goss_other_rate,
        )

        # mirror legacy fields
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.use_goss = use_goss
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.max_delta_step = max_delta_step
        self.max_leaves = max_leaves

        self.objective = objective
        self.adaptive_lr = adaptive_lr
        self.lr_schedule = lr_schedule
        self.focal_gamma = focal_gamma
        self.elastic_net_ratio = elastic_net_ratio

        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = None if n_jobs == -1 else n_jobs
        self.tree_method = tree_method
        self.binned_mode = binned_mode
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.cache_gradients = cache_gradients

        # DART
        self.skip_drop = skip_drop
        self.normalize_type = normalize_type
        self.rate_drop = rate_drop
        self.one_drop = one_drop

        self.tree_learner = tree_learner
        self.random_state = random_state
        self._rng = (
            np.random.default_rng(random_state)
            if random_state is not None else np.random.default_rng()
        )
        self.verbose = verbose
        self.eval_metric = eval_metric
        self.monotone_constraints = monotone_constraints or {}
        self.interaction_constraints = interaction_constraints
        if random_state is not None:
            np.random.seed(random_state)

        self.use_gpu = use_gpu
        self.node_frequency = 4                   # <- every ~3 trees
        self.node_config = NodeConfig()           # <- ODST knobs

        self._initialize_components()

    # ----------------------------- init helpers -----------------------------
    def _initialize_components(self):
        self.base_score: Optional[float] = None
        self.trees: List[Tuple[object, np.ndarray]] = []
        self.feature_importance_ = FeatureImportance()
        self.gpu_accelerator = None
        self._setup_loss_function()
        self.train_scores_: List[float] = []
        self.val_scores_: List[float] = []
        self._cached_gradients: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._lr_history: List[float] = []

        # DART state
        self._tree_lr: List[float] = []
        self._y_sum_train: Optional[np.ndarray] = None
        self._y_sum_val: Optional[np.ndarray] = None

        # binning buffers
        self.bin_edges = None
        self._binned = None

    @cached_property
    def _loss_fn(self):
        return self._create_loss_function()

    def _setup_loss_function(self):
        self.loss_fn = self._loss_fn

    def _create_loss_function(self):
        base = self._get_base_loss()
        # optional wrappers (focal/elastic) can be added here
        return base

    def _get_base_loss(self):
        if self.objective == "reg:squarederror":
            return MSELoss()
        elif self.objective == "reg:pseudohubererror":
            return HuberLoss(delta=1.0)
        elif self.objective.startswith("reg:quantileerror"):
            alpha = (
                float(self.objective.split(":")[-1]) if ":" in self.objective else 0.5
            )
            return QuantileLoss(alpha=alpha)
        elif self.objective == "binary:logistic":
            return LogisticLoss()
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")

    def _compute_base_score(self, y: np.ndarray) -> float:
        if self.objective == "reg:squarederror":
            return float(np.mean(y))
        elif self.objective == "reg:pseudohubererror":
            return float(np.median(y))
        elif self.objective.startswith("reg:quantileerror"):
            alpha = (
                float(self.objective.split(":")[-1]) if ":" in self.objective else 0.5
            )
            return float(np.percentile(y, alpha * 100))
        elif self.objective == "binary:logistic":
            pos_ratio = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
            return float(np.log(pos_ratio / (1 - pos_ratio)))
        else:
            return float(np.mean(y))

    def _get_learning_rate(self, iteration: int) -> float:
        lr0 = self.learning_rate
        if not self.adaptive_lr:
            return lr0
        T = max(1, self.n_estimators)
        t = iteration
        if self.lr_schedule == "cosine":
            return lr0 * 0.5 * (1 + np.cos(np.pi * t / T))
        elif self.lr_schedule == "exponential":
            return lr0 * (0.95 ** (t // 10))
        elif self.lr_schedule == "linear":
            return lr0 * (1 - t / T)
        else:
            return lr0

    # ----------------------------- DART helpers -----------------------------
    def _dart_select_drop(self):
        n = len(self.trees)
        if n == 0 or self.rate_drop <= 0.0:
            return None
        if self._rng.random() < float(self.skip_drop):
            return None

        drop_mask = self._rng.random(n) < float(self.rate_drop)
        if self.one_drop and not drop_mask.any():
            drop_mask[self._rng.integers(0, n)] = True
        if drop_mask.all():
            drop_mask[self._rng.integers(0, n)] = False

        k = int(drop_mask.sum())
        if self.normalize_type == "tree":
            tree_scale = float(n) / float(n - k) if k > 0 else 1.0
            forest_newtree_scale = 1.0
        elif self.normalize_type == "forest":
            tree_scale = 1.0
            forest_newtree_scale = 1.0 / max(1e-12, (1.0 - self.rate_drop))
        else:
            tree_scale = 1.0
            forest_newtree_scale = 1.0
        return drop_mask, tree_scale, forest_newtree_scale

    def _dart_iter_pred(self, X, y_sum_running, drop_info):
        if drop_info is None:
            return self.base_score + y_sum_running, np.zeros_like(y_sum_running), 1.0
        drop_mask, tree_scale, _ = drop_info
        dropped = np.zeros_like(y_sum_running)
        for t_idx in np.flatnonzero(drop_mask):
            tree, feats = self.trees[t_idx]
            lr_i = self._tree_lr[t_idx]
            dropped += lr_i * tree.predict(X[:, feats])
        if self.normalize_type == "tree":
            y_iter = self.base_score + tree_scale * (y_sum_running - dropped)
        else:
            y_iter = self.base_score + (y_sum_running - dropped)
        return y_iter, dropped, tree_scale

    # ----------------------------- tree factory -----------------------------
    def _create_tree(self, feature_mask: np.ndarray):
        feature_indices = np.asarray(feature_mask, dtype=np.int32)
        tree_bin_edges_seq = (
            [self.bin_edges[j] for j in feature_indices]
            if self.bin_edges is not None else None
        )
        common_args = {
            "max_depth": self.tree_config.max_depth,
            "min_samples_split": self.tree_config.min_samples_split,
            "min_samples_leaf": self.tree_config.min_samples_leaf,
            "lambda_": self.tree_config.lambda_,
            "gamma": self.tree_config.gamma,
            "alpha": self.tree_config.alpha,
            "feature_indices": feature_mask,
            "n_jobs": self.n_jobs,
            "tree_method": self.tree_method,
            "binned_mode": self.binned_mode,
            "n_bins": self.n_bins,
            "bin_edges": tree_bin_edges_seq,
            "monotone_constraints": self.monotone_constraints,
            "interaction_constraints": self.interaction_constraints,
            "max_delta_step": self.tree_config.max_delta_step,
            "use_gpu": self.use_gpu,
            "feature_importance_": self.feature_importance_,
            "enable_interactions": self.enable_interactions,
        }
        if self.tree_learner == "leaf":
            return UnifiedTree(max_leaves=self.tree_config.max_leaves, **common_args, growth_policy="leaf_wise")
        else:
            return UnifiedTree(max_leaves=self.tree_config.max_leaves, **common_args, growth_policy="level_wise")

    # (ADDED) flip-of-a-coin NODE injection
    def _should_use_node(self):
        if self.node_frequency <= 0:
            return False
        p = 1.0 / float(self.node_frequency)
        return self._rng.random() < p

    # (REPLACED) use NeuralObliviousTree when chosen
    def _create_tree_with_node_option(self, feature_mask: np.ndarray):
        if self.use_neural and self._should_use_node():
            return NeuralObliviousTree(input_dim=len(feature_mask), config=self.node_config)
        return self._create_tree(feature_mask)

    # ----------------------------- GOSS & sampling --------------------------
    def _adapt_goss_rates(self, grad: np.ndarray) -> Tuple[float, float]:
        g = np.abs(grad)
        med = np.median(g)
        mad = np.median(np.abs(g - med)) + 1e-12
        tail = np.mean((g - med) / mad > 3.0)
        a = float(np.clip(0.10 + 0.60 * tail, 0.10, 0.70))
        b = float(np.clip(0.05 + 0.30 * (1.0 - a), 0.05, 0.20))
        return a, b

    def _apply_goss_optimized(self, X, y, grad, hess):
        n = grad.shape[0]
        if getattr(self.sampling_config, "use_goss", True) and getattr(self, "adaptive_goss", True):
            a, b = self._adapt_goss_rates(grad)
            self.sampling_config.goss_top_rate = a
            self.sampling_config.goss_other_rate = b

        a = float(self.sampling_config.goss_top_rate)
        b = float(self.sampling_config.goss_other_rate)

        mode = getattr(self.sampling_config, "score_mode", "abs_g")
        if mode == "abs_g":
            scores = np.abs(grad)
        elif mode == "abs_g_sqrt_h":
            scores = np.abs(grad) * np.sqrt(np.maximum(hess, 0.0) + 1e-12)
        else:
            raise ValueError(f"Invalid score_mode: {mode}")

        top_k = max(0, int(a * n))
        rest_k = max(0, int(b * n))

        if top_k + rest_k <= 0 or top_k + rest_k >= n:
            selected_idx = np.arange(n)
            top_k_effective = 0
        else:
            if top_k > 0:
                kth = n - top_k
                part = np.argpartition(scores, kth)
                top_idx = part[kth:]
            else:
                top_idx = np.empty(0, dtype=np.int64)
            mask = np.ones(n, dtype=bool)
            mask[top_idx] = False
            rest_pool = np.flatnonzero(mask)
            if rest_k > 0 and rest_pool.size > 0:
                rest_k = min(rest_k, rest_pool.size)
                rest_idx = rest_pool[self._rng.choice(rest_pool.size, size=rest_k, replace=False)]
                selected_idx = np.concatenate([top_idx, rest_idx])
            else:
                selected_idx = top_idx
            top_k_effective = top_idx.size

        X_sub = X[selected_idx]
        y_sub = y[selected_idx]
        grad_sub = grad[selected_idx].astype(np.float64, copy=True)
        hess_sub = hess[selected_idx].astype(np.float64, copy=True)

        codes_subset = None
        if hasattr(self, "_precomputed_indices") and self._precomputed_indices is not None:
            codes_subset = self._precomputed_indices[selected_idx]

        if selected_idx.size > top_k_effective:
            if b > 0.0:
                scale = (1.0 - a) / b
                if top_k_effective < selected_idx.size:
                    grad_sub[top_k_effective:] *= scale
                    gscale = getattr(self.sampling_config, "goss_scale", "both")
                    if gscale == "both":
                        hess_sub[top_k_effective:] *= scale
                    elif gscale == "grad_only":
                        pass
                    else:
                        raise ValueError(f"Invalid goss_scale: {gscale}")

        return X_sub, y_sub, grad_sub, hess_sub, np.asarray(selected_idx, dtype=np.int64), codes_subset

    # ----------------------------- Batch build ------------------------------
    def _build_tree_batch(self, X, y, grad, hess) -> List[Tuple]:
        trees: List[Tuple] = []
        n_samples, n_features = X.shape

        for _ in range(self.batch_size):
            # --- rows
            if self.sampling_config.use_goss:
                X_sub, y_sub, grad_sub, hess_sub, selected_idx, codes_subset = \
                    self._apply_goss_optimized(X, y, grad, hess)
            elif self.sampling_config.subsample < 1.0:
                row_mask = (self._rng.random(n_samples) < float(self.sampling_config.subsample))
                selected_idx = np.where(row_mask)[0]
                X_sub, y_sub = X[selected_idx], y[selected_idx]
                grad_sub, hess_sub = grad[selected_idx], hess[selected_idx]
                codes_subset = (
                    self._precomputed_indices[selected_idx]
                    if getattr(self, "_precomputed_indices", None) is not None else None
                )
            else:
                selected_idx = np.arange(n_samples)
                X_sub, y_sub, grad_sub, hess_sub = X, y, grad, hess
                codes_subset = getattr(self, "_precomputed_indices", None)

            # --- cols
            n_features_tree = max(1, int(self.sampling_config.colsample_bytree * n_features))
            feature_mask = self._rng.choice(n_features, size=n_features_tree, replace=False)

            # choose tree (possibly neural)
            tree = self._create_tree_with_node_option(feature_mask)

            # histogram system only for histogram trees
            if not getattr(tree, "is_neural", False):
                tree._histogram_system = self._histogram_system.clone(feature_idx=feature_mask, row_idx=selected_idx)

            # ---- fit
            X_sub_feat = X_sub[:, feature_mask]
            tree.fit(
                X_sub_feat,
                grad_sub,
                hess_sub,
                feature_importance=self.feature_importance_,
            )

            # ---- (ADDED) light magnitude calibration for neural trees
            if getattr(tree, "is_neural", False):
                pred_cal = tree.predict(X_sub_feat)
                med = np.median(np.abs(pred_cal)) + 1e-12
                # scale neural outputs so their typical magnitude matches trees
                tree._rescale = 1.0 / max(med, 1e-6)

            # ---- finalize
            tree.post_prune_ccp(ccp_alpha=1e-4)
            trees.append((tree, feature_mask))

        return trees

    # ----------------------------- fit / predict -----------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: Optional[bool] = None) -> "BoostRegressor":
        if verbose is None:
            verbose = self.verbose
        self.trees.clear(); self._tree_lr.clear(); self.train_scores_.clear(); self.val_scores_.clear(); self._lr_history.clear()

        start_time = time.time()
        self.base_score = self._compute_base_score(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        self._y_sum_train = np.zeros(n_samples, dtype=np.float64)
        if eval_set is not None:
            X_val, y_val = eval_set
            self._y_sum_val = np.zeros(X_val.shape[0], dtype=np.float64)
        else:
            X_val = y_val = None

        # ---- global histogram system (unchanged from your version) ----
        self._histogram_system = None
        self._precomputed_indices = None
        if self.tree_method == "binned":
            config = HistogramConfig(
                method=self.binned_mode,
                max_bins=self.n_bins,
                lambda_reg=self.tree_config.lambda_,
                gamma=self.tree_config.gamma,
                use_parallel=self.sampling_config.use_goss,
                random_state=self.random_state
            )
            self._histogram_system = GradientHistogramSystem(config)
            dummy_pred = np.full_like(y, self.base_score, dtype=np.float64)
            dummy_grad, dummy_hess = self.loss_fn.grad_hess(y, dummy_pred)
            self._histogram_system.fit_bins(X, dummy_grad, dummy_hess)
            self._precomputed_indices = self._histogram_system.prebin_dataset(X, inplace=True)
            if eval_set is not None:
                self._val_precomputed_indices = self._histogram_system.prebin_dataset(X_val, inplace=False)

        if verbose:
            strategy = "Leaf-wise" if self.tree_learner == "leaf" else "Level-wise"
            print(f"🚀 Training with {strategy} trees (batch_size={self.batch_size})")
            print(f"   Objective: {self.objective}, Loss: {self.loss_fn.name}, Base score: {self.base_score:.4f}")
            print(f"   NODE layers {'enabled' if self.use_neural else 'disabled'} (freq={self.node_frequency})")
            if self.sampling_config.use_goss:
                print(f"   GOSS: top_rate={self.sampling_config.goss_top_rate}, other_rate={self.sampling_config.goss_other_rate}")
            if self.rate_drop > 0.0:
                print(f"   DART: rate_drop={self.rate_drop}, skip_drop={self.skip_drop}, normalize_type={self.normalize_type}, one_drop={'yes' if self.one_drop else 'no'}")
            if self.adaptive_lr:
                print(f"   Adaptive LR: {self.lr_schedule}")
            print(f"   Binning: {self.tree_method} ({self.binned_mode}), {self.n_bins} bins")
            print(f"   Training on {n_samples} samples with {n_features} features")

        best_score = np.inf
        best_state = None
        best_iter = -1
        no_improve_rounds = 0

        for iteration in range(0, self.n_estimators, self.batch_size):
            drop_info = self._dart_select_drop() if len(self.trees) > 0 else None

            y_pred_iter, dropped_train, tree_scale = self._dart_iter_pred(X, self._y_sum_train, drop_info)
            grad, hess = self.loss_fn.grad_hess(y, y_pred_iter)

            trees = self._build_tree_batch(X, y, grad, hess)

            current_lr = self._get_learning_rate(iteration)
            self._lr_history.append(current_lr)
            forest_newtree_scale = drop_info[2] if drop_info is not None else 1.0
            effective_lr = current_lr * (forest_newtree_scale if self.normalize_type == "forest" else 1.0)
            for _ in trees:
                self._tree_lr.append(effective_lr)

            delta_train = np.zeros_like(self._y_sum_train)
            for tree, feat_mask in trees:
                delta_train += tree.predict(X[:, feat_mask])
            delta_train *= current_lr * forest_newtree_scale
            self._y_sum_train += delta_train

            if eval_set is not None:
                delta_val = np.zeros_like(self._y_sum_val)
                for tree, feat_mask in trees:
                    delta_val += tree.predict(X_val[:, feat_mask])
                delta_val *= current_lr * forest_newtree_scale
                self._y_sum_val += delta_val

            if drop_info is None:
                y_train_eval = self.base_score + self._y_sum_train
                if eval_set is not None:
                    y_val_eval = self.base_score + self._y_sum_val
            else:
                drop_mask, tree_scale_eval, _ = drop_info
                if self.normalize_type == "tree":
                    y_train_eval = self.base_score + tree_scale_eval * (self._y_sum_train - delta_train - dropped_train) + delta_train
                else:
                    y_train_eval = self.base_score + (self._y_sum_train - dropped_train)

                if eval_set is not None:
                    dropped_val = np.zeros_like(self._y_sum_val)
                    for t_idx in np.flatnonzero(drop_mask):
                        tree, feats = self.trees[t_idx]
                        lr_i = self._tree_lr[t_idx]
                        dropped_val += lr_i * tree.predict(X_val[:, feats])
                    if self.normalize_type == "tree":
                        y_val_eval = self.base_score + tree_scale_eval * (self._y_sum_val - delta_val - dropped_val) + delta_val
                    else:
                        y_val_eval = self.base_score + (self._y_sum_val - dropped_val)

            train_score = self.loss_fn.loss(y, y_train_eval)
            self.train_scores_.append(train_score)

            if eval_set is not None:
                val_score = self.loss_fn.loss(y_val, y_val_eval)
                self.val_scores_.append(val_score)
                if val_score < best_score - 1e-6:
                    best_score = val_score
                    best_iter = iteration
                    no_improve_rounds = 0
                    best_state = (
                        list(self.trees + trees),
                        float(self.base_score),
                        list(self._lr_history),
                        np.copy(self._y_sum_train),
                        np.copy(self._y_sum_val),
                        list(self._tree_lr),
                    )
                else:
                    no_improve_rounds += self.batch_size
                if self.early_stopping_rounds and no_improve_rounds >= self.early_stopping_rounds:
                    if verbose:
                        print(f"⏹️  Early stopping at iteration {iteration}, best {best_iter} val={best_score:.6f}")
                    if best_state is not None:
                        (self.trees, self.base_score, self._lr_history, self._y_sum_train, self._y_sum_val, self._tree_lr) = best_state
                    break

            self.trees.extend(trees)

            if verbose and ((iteration + self.batch_size) % 10 == 0):
                elapsed = time.time() - start_time
                msg = f"[{iteration + self.batch_size:4d}] Train: {train_score:.6f}"
                if eval_set is not None:
                    msg += f", Val: {val_score:.6f}"
                msg += f", Time: {elapsed:.2f}s"
                print(msg)

        if verbose:
            total_time = time.time() - start_time
            print(f"✅ Training completed in {total_time:.2f}s, {len(self.trees)} trees")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model not fitted yet")
        y = np.full(X.shape[0], self.base_score, dtype=np.float64)
        if len(self._tree_lr) != len(self.trees):
            for tree, feats in self.trees:
                y += self.learning_rate * tree.predict(X[:, feats])
        else:
            for lr_i, (tree, feats) in zip(self._tree_lr, self.trees):
                y += lr_i * tree.predict(X[:, feats])
        return y

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.objective != "binary:logistic":
            raise ValueError("predict_proba only available for binary:logistic")
        logits = self.predict(X)
        proba_pos = 1.0 / (1.0 + np.exp(-np.clip(logits, -250, 250)))
        return np.column_stack([1 - proba_pos, proba_pos])

    def feature_importances(self, importance_type: str = "gain", normalize: bool = True) -> np.ndarray:
        importance_dict = self.feature_importance_.get_importance(importance_type)
        arr = np.zeros(self.n_features_, dtype=np.float64)
        for k, v in importance_dict.items():
            if 0 <= k < self.n_features_:
                arr[k] = v
        if normalize and arr.sum() > 0:
            arr /= arr.sum()
        return arr

    def get_training_history(self) -> Dict[str, List[float]]:
        history = {"train_scores": self.train_scores_, "learning_rates": self._lr_history}
        if self.val_scores_:
            history["val_scores"] = self.val_scores_
        return history
