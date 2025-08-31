import time
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from boosting_aux import *
from boosting_loss import HuberLoss, LogisticLoss, MSELoss, QuantileLoss
from boosting_tree import *


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
    score_mode: str = "abs_g"          # {"abs_g", "abs_g_sqrt_h"}
    goss_scale: str = "both"           # {"both", "grad_only"}

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False



@dataclass
class NodeConfig:
    num_trees: int = 64
    tree_depth: int = 4
    temperature: float = 1.0
    dropout_rate: float = 0.1


class NodeTree:
    """Simple NODE layer that can be used as a drop-in replacement for traditional trees."""
    
    def __init__(self, input_dim: int, config: NodeConfig):
        self.input_dim = input_dim
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the neural oblivious decision tree ensemble."""
        model = nn.Sequential(
            # Feature embedding
            nn.Linear(self.input_dim, self.config.num_trees),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            
            # Oblivious decision layer
            ObliviousLayer(self.config.num_trees, self.config.tree_depth, self.config.temperature),
            
            # Output aggregation
            nn.Linear(self.config.num_trees, 1)
        )
        return model
        
    def fit(self, X, grad, hess, feature_importance=None, epochs=30, lr=0.01):
            """Fit NODE to gradient boosting residuals. Compatible with existing tree interface."""
            X_tensor = torch.FloatTensor(X)
            target = torch.FloatTensor(-grad / (hess + 1e-8))  # Newton-Raphson target
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            
            self.model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                pred = self.model(X_tensor).squeeze()
                loss = F.mse_loss(pred, target)
                loss.backward()
                optimizer.step()
            
            # Update feature importance if provided (optional for compatibility)
            if feature_importance is not None:
                # Simple feature importance based on gradient norms
                with torch.no_grad():
                    for i, param in enumerate(self.model.parameters()):
                        if param.grad is not None:
                            importance = torch.sum(torch.abs(param.grad)).item()
                            # Add to feature importance tracker if needed
                            pass
    def post_prune_ccp(self, ccp_alpha=1e-4):
        """Compatibility method - NODE layers don't need post-pruning."""
        pass
    
    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            pred = self.model(X_tensor).squeeze()
        return pred.numpy()

class ObliviousLayer(nn.Module):
    def __init__(self, num_features: int, depth: int, temperature: float):
        super().__init__()
        self.depth = depth
        self.temperature = temperature
        self.num_leaves = 2 ** depth

        self.feature_weights = nn.Parameter(torch.randn(depth, num_features) * 0.01)
        self.thresholds = nn.Parameter(torch.zeros(depth))  # stable start
        self.leaf_values = nn.Parameter(torch.randn(num_features, self.num_leaves) * 0.01)

        # Precompute binary codes for leaves: shape (num_leaves, depth)
        codes = []
        for leaf in range(self.num_leaves):
            bits = [(leaf >> (depth - 1 - i)) & 1 for i in range(depth)]
            codes.append(bits)
        self.register_buffer("leaf_codes", torch.tensor(codes, dtype=torch.float32))  # (L, D)

    def forward(self, x):
        # feature selection per depth
        probs = F.softmax(self.feature_weights, dim=1)         # (D, F)
        sel = x @ probs.t()                                    # (B, D)

        decisions = torch.sigmoid((sel - self.thresholds) / self.temperature)  # (B, D)
        # leaf_probs = Π over depth of decision/1-decision according to leaf code
        p = decisions.unsqueeze(1)                             # (B, 1, D)
        code = self.leaf_codes.unsqueeze(0)                    # (1, L, D)
        leaf_probs = (p * code + (1 - p) * (1 - code)).prod(dim=-1)  # (B, L)

        # aggregate per “tree” (num_features == num_trees)
        out = leaf_probs @ self.leaf_values.t()                # (B, num_features)
        return out

# ================================================================
# BoostRegressor with corrected & fast DART
# ================================================================
class BoostRegressor:
    """
    Gradient boosting with histogram trees, GOSS, DART, and advanced losses.
    DART is applied correctly (drop BEFORE gradient computation) and efficiently
    (running sums + recompute only dropped trees for this round).
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

        # Sampling parameters (backward compatible)
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        colsample_bynode: float = 1.0,
        use_goss: bool = True,
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,

        # New structured configs (optional)
        tree_config: Optional[TreeConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,

        # Advanced features
        objective: str = "reg:squarederror",
        adaptive_lr: bool = False,              # Adaptive learning rate
        lr_schedule: str = "constant",          # "constant", "cosine", "exponential"
        focal_gamma: float = 0.0,               # Focal loss parameter
        elastic_net_ratio: float = 0.0,         # L1/L2 mixing ratio

        # Performance
        early_stopping_rounds: Optional[int] = None,
        n_jobs: int = -1,
        tree_method: str = "binned",
        binned_mode: str = "hist",            # "hist" or "adaptive"
        n_bins: int = 256,
        batch_size: int = 1,
        cache_gradients: bool = True,

        # DART
        dart_rate: float = 0.0,                 # kept for backward compat (alias of rate_drop)
        skip_drop: float = 0.5,
        normalize_type: str = "tree",           # "tree" or "forest"
        rate_drop: float = 0.1,
        one_drop: bool = False,

        # Tree strategy
        tree_learner: str = "level",            # "level" or "leaf"

        # Misc
        random_state: Optional[int] = None,
        verbose: bool = True,
        eval_metric: Optional[str] = None,
        monotone_constraints: Optional[Dict] = None,
        interaction_constraints: Optional[List] = None,
        
        use_gpu: bool = False,
        adaptive_hist: bool = False,
        
        use_neural: bool = False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        self.use_neural = use_neural

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

        # mirror legacy fields for compatibility
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
        self.dart_rate = dart_rate  # kept for compat
        self.skip_drop = skip_drop
        self.normalize_type = normalize_type
        self.rate_drop = rate_drop
        self.one_drop = one_drop

        self.tree_learner = tree_learner

        self.random_state = random_state
        self._rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()

        self.verbose = verbose
        self.eval_metric = eval_metric
        self.monotone_constraints = monotone_constraints or {}
        self.interaction_constraints = interaction_constraints

        if random_state is not None:
            np.random.seed(random_state)

        # Map old alias if provided
        if self.dart_rate and self.rate_drop == 0.0:
            self.rate_drop = self.dart_rate
            
        self.use_gpu = use_gpu
        self.adaptive_hist = adaptive_hist
        self.node_frequency = 2  # Add NODE every 3 trees
        self.node_config = NodeConfig()
        
        self._initialize_components()

    # ----------------------------- init helpers -----------------------------
    def _initialize_components(self):
        self.base_score: Optional[float] = None
        self.trees: List[Tuple[object, np.ndarray]] = []  # (tree, feature_mask)
        self.feature_importance_ = FeatureImportance()
        self.gpu_accelerator = None
        self._setup_loss_function()
        self.train_scores_: List[float] = []
        self.val_scores_: List[float] = []
        self._cached_gradients: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._lr_history: List[float] = []

        # --- DART state ---
        self._tree_lr: List[float] = []        # per-tree LR (supports adaptive LR)
        self._y_sum_train: Optional[np.ndarray] = None  # running sum (no base) for train
        self._y_sum_val: Optional[np.ndarray] = None    # running sum (no base) for val

        # binning buffers
        self.bin_edges = None
        self._binned = None

    @cached_property
    def _loss_fn(self):
        return self._create_loss_function()

    def _setup_loss_function(self):
        self.loss_fn = self._loss_fn

    def _create_loss_function(self):
        base_loss = self._get_base_loss()
        if self.focal_gamma > 0:
            raise NotImplementedError("Focal loss wrapper not implemented yet")
            base_loss = FocalLossWrapper(base_loss, gamma=self.focal_gamma)
        if self.elastic_net_ratio > 0:
            raise NotImplementedError("Elastic net wrapper not implemented yet")
            base_loss = ElasticNetWrapper(
                base_loss, l1_ratio=self.elastic_net_ratio, alpha=self.tree_config.alpha
            )
        return base_loss

    def _get_base_loss(self):
        if self.objective == "reg:squarederror":
            return MSELoss()
        elif self.objective == "reg:pseudohubererror":
            return HuberLoss(delta=1.0)
        elif self.objective.startswith("reg:quantileerror"):
            alpha = float(self.objective.split(":")[-1]) if ":" in self.objective else 0.5
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
            alpha = float(self.objective.split(":")[-1]) if ":" in self.objective else 0.5
            return float(np.percentile(y, alpha * 100))
        elif self.objective == "binary:logistic":
            pos_ratio = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
            return float(np.log(pos_ratio / (1 - pos_ratio)))
        else:
            return float(np.mean(y))

    def _get_learning_rate(self, iteration: int) -> float:
        if not self.adaptive_lr:
            return self.learning_rate
        if self.lr_schedule == "cosine":
            return self.learning_rate * 0.5 * (1 + np.cos(np.pi * iteration / self.n_estimators))
        elif self.lr_schedule == "exponential":
            return self.learning_rate * (0.95 ** (iteration // 10))
        else:
            return self.learning_rate

    # ----------------------------- DART helpers -----------------------------
    def _dart_select_drop(self):
        n = len(self.trees)
        if n == 0 or self.rate_drop <= 0.0:
            return None
        if self._rng.random() < float(self.skip_drop):
            return None

        drop_mask = (self._rng.random(n) < float(self.rate_drop))
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
    

    def _dart_iter_pred(
        self,
        X: np.ndarray,
        y_sum_running: np.ndarray,
        drop_info: Optional[Tuple[np.ndarray, float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Build this round's prediction for gradient computation.
        Returns (y_pred_iter, dropped_contrib, tree_scale).
        """
        if drop_info is None:
            return self.base_score + y_sum_running, np.zeros_like(y_sum_running), 1.0

        drop_mask, tree_scale, _ = drop_info
        dropped = np.zeros_like(y_sum_running)
        # Only predict through DROPPED trees (usually few)
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

        # For hist-only: slice the globally computed edges
        tree_bin_edges_seq = (
            [self.bin_edges[j] for j in feature_indices] if self.bin_edges is not None else None
        )

        common_args = {
            'max_depth': self.tree_config.max_depth,
            'min_samples_split': self.tree_config.min_samples_split,
            'min_samples_leaf': self.tree_config.min_samples_leaf,
            'lambda_': self.tree_config.lambda_,
            'gamma': self.tree_config.gamma,
            'alpha': self.tree_config.alpha,
            'feature_indices': feature_mask,
            'n_jobs': self.n_jobs,
            'tree_method': self.tree_method,
            'binned_mode': self.binned_mode,
            'n_bins': self.n_bins,
            'bin_edges': tree_bin_edges_seq,
            'monotone_constraints': self.monotone_constraints,
            'interaction_constraints': self.interaction_constraints,
            'gpu_accelerator': self.gpu_accelerator,
            'max_delta_step': self.tree_config.max_delta_step,
            'use_gpu': self.use_gpu,
            'adaptive_hist': self.adaptive_hist,
            'feature_importance_': self.feature_importance_,
        }
        if self.tree_learner == "leaf":
            return LeafWiseSingleTree(max_leaves=self.tree_config.max_leaves, **common_args)
        else:
            return SingleTree(**common_args)

    # ----------------------------- sampling + batch build -----------------------------
    def _apply_goss_optimized(self, X, y, grad, hess):
        n = grad.shape[0]
        a = float(self.sampling_config.goss_top_rate)
        b = float(self.sampling_config.goss_other_rate)

        top_k  = max(0, int(a * n))
        rest_k = max(0, int(b * n))  # b is fraction of ALL samples (LightGBM-style)

        if top_k == 0 and rest_k == 0:
            idx = np.arange(n)
            return X, y, grad, hess, idx
        if top_k + rest_k >= n:
            idx = np.arange(n)
            return X, y, grad, hess, idx

        # scoring
        if self.sampling_config.score_mode == "abs_g":
            scores = np.abs(grad)
        elif self.sampling_config.score_mode == "abs_g_sqrt_h":
            scores = np.abs(grad) * np.sqrt(np.maximum(hess, 0.0) + 1e-12)
        else:
            raise ValueError(f"Invalid score_mode: {self.sampling_config.score_mode}")

        # top indices
        if top_k > 0:
            kth = n - top_k
            idx = np.argpartition(scores, kth)
            top_idx = idx[kth:]
        else:
            top_idx = np.empty(0, dtype=np.int64)

        # rest sampling
        mask = np.ones(n, dtype=bool)
        if top_k > 0:
            mask[top_idx] = False
        rest_pool = np.flatnonzero(mask)
        if rest_pool.size == 0 or rest_k == 0:
            sel = top_idx
            Xs, ys = X[sel], y[sel]
            gs, hs = grad[sel].astype(np.float64, copy=True), hess[sel].astype(np.float64, copy=True)
            return Xs, ys, gs, hs, sel

        rest_k = min(rest_k, rest_pool.size)
        rest_idx = rest_pool[self._rng.choice(rest_pool.size, size=rest_k, replace=False)]

        sel = np.concatenate([top_idx, rest_idx])

        Xs, ys = X[sel], y[sel]
        gs = grad[sel].astype(np.float64, copy=True)
        hs = hess[sel].astype(np.float64, copy=True)

        # Reweight small-|g| group for unbiasedness
        if rest_k > 0:
            scale = (1.0 - a) / max(b, 1e-12)
            if top_k < sel.size:
                gs[top_k:] *= scale
                if self.sampling_config.goss_scale == "both":
                    hs[top_k:] *= scale
                elif self.sampling_config.goss_scale == "grad_only":
                    pass
                else:
                    raise ValueError(f"Invalid goss_scale: {self.sampling_config.goss_scale}")

        return Xs, ys, gs, hs, sel

    def _should_use_node(self):
        if self.node_frequency <= 0:
            return False
        p = 1.0 / float(self.node_frequency)  # e.g., 1/3
        return (self._rng.random() < p)


    def _create_tree_with_node_option(self, feature_mask: np.ndarray):
        """
        Modified version of your _create_tree method.
        Replace your existing _create_tree call with this.
        """
        # Decide: NODE or traditional tree
        if self.use_neural and self._should_use_node():
            # Create NODE tree
            return NodeTree(len(feature_mask), self.node_config)
        else:
            # Use your existing tree creation logic
            return self._create_tree(feature_mask)  # Your original method
    
    
    def _build_tree_batch(self, X: np.ndarray, y: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> List[Tuple]:
        trees = []
        n_samples, n_features = X.shape
        for _ in range(self.batch_size):
            # row sampling
            if self.sampling_config.use_goss:
                X_sub, y_sub, grad_sub, hess_sub, selected_idx = self._apply_goss_optimized(X, y, grad, hess)
            elif self.sampling_config.subsample < 1.0:
                row_mask = np.random.rand(n_samples) < self.sampling_config.subsample
                selected_idx = np.where(row_mask)[0]
                X_sub, y_sub = X[selected_idx], y[selected_idx]
                grad_sub, hess_sub = grad[selected_idx], hess[selected_idx]
            else:
                selected_idx = np.arange(n_samples)
                X_sub, y_sub = X, y
                grad_sub, hess_sub = grad, hess

            # feature sampling
            n_features_tree = max(1, int(self.sampling_config.colsample_bytree * n_features))
            feature_mask = self._rng.choice(n_features, size=n_features_tree, replace=False)
            X_sub_feat = X_sub[:, feature_mask]

            if self.use_neural:
                tree = self._create_tree_with_node_option(feature_mask)
            else:
                tree = self._create_tree(feature_mask)
            #t

            tree._row_indexer = selected_idx
            tree._binned = self._binned
            tree._feature_mask = feature_mask

            tree.fit(
                X_sub_feat,
                grad_sub,
                hess_sub,
                feature_importance=self.feature_importance_,
            )
            tree.post_prune_ccp(ccp_alpha=1e-4)

            trees.append((tree, feature_mask))
        return trees

    # ----------------------------- fit / predict -----------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: Optional[bool] = None) -> "BoostRegressor":
        if verbose is None:
            verbose = self.verbose

        # fresh state if re-fitting
        self.trees.clear()
        self._tree_lr.clear()
        self.train_scores_.clear()
        self.val_scores_.clear()
        self._lr_history.clear()

        start_time = time.time()
        self.base_score = self._compute_base_score(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Running sums (no base) for train / val
        self._y_sum_train = np.zeros(n_samples, dtype=np.float64)
        if eval_set is not None:
            X_val, y_val = eval_set
            self._y_sum_val = np.zeros(X_val.shape[0], dtype=np.float64)
        else:
            X_val = y_val = None

        # Pre-binning (kept row-major for compatibility with your trees)
        if self.tree_method == "binned" and self.binned_mode in ("hist", "adaptive"):
            dummy_grad, _ = self.loss_fn.grad_hess(y, np.full_like(y, self.base_score, dtype=np.float64))
            self.bin_edges = create_bin_edges(X, dummy_grad, self.n_bins, self.use_gpu)
            self._binned = prebin_data(X, self.bin_edges, self.n_bins, self.use_gpu)
        else:
            self.bin_edges = None
            self._binned = None

        if verbose:
            strategy = "Leaf-wise" if self.tree_learner == "leaf" else "Level-wise"
            print(f"🚀 Training with {strategy} trees (DART={'on' if self.rate_drop>0 else 'off'}, GOSS={'on' if self.use_goss else 'off'}), batch_size={self.batch_size}")

        best_score = np.inf
        best_state = None  # (trees, base, lr_hist, ysum_tr, ysum_val, tree_lr)
        best_iter = -1
        no_improve_rounds = 0

        # ========================= MAIN LOOP =========================
        for iteration in range(0, self.n_estimators, self.batch_size):
            # 1) DART: choose drop set BEFORE gradient computation
            drop_info = self._dart_select_drop() if len(self.trees) > 0 else None

            # 2) Build prediction for gradients (train)
            y_pred_iter, dropped_train, tree_scale = self._dart_iter_pred(
                X, self._y_sum_train, drop_info
            )
            grad, hess = self.loss_fn.grad_hess(y, y_pred_iter)

            # 3) Build a batch of trees from (X, grad, hess)
            trees = self._build_tree_batch(X, y, grad, hess)

            # 4) LR for this round; store per-tree LRs
            current_lr = self._get_learning_rate(iteration)
            self._lr_history.append(current_lr)
            
            forest_newtree_scale = drop_info[2] if drop_info is not None else 1.0
            effective_lr = current_lr * (forest_newtree_scale if self.normalize_type == "forest" else 1.0)
            for _ in trees:
                self._tree_lr.append(effective_lr)

            # 5) Update running sums with NEW trees (train & val)
            forest_newtree_scale = drop_info[2] if drop_info is not None else 1.0

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

            # 6) Compute losses (with the SAME drop mask of this round)
            if drop_info is None:
                y_train_eval = self.base_score + self._y_sum_train
                if eval_set is not None:
                    y_val_eval = self.base_score + self._y_sum_val
            else:
                drop_mask, tree_scale_eval, _ = drop_info

                # Train evaluation
                if self.normalize_type == "tree":
                    # survivors scaled, new trees not scaled
                    y_train_eval = (
                        self.base_score
                        + tree_scale_eval * (self._y_sum_train - delta_train - dropped_train)
                        + delta_train
                    )
                else:  # forest
                    y_train_eval = self.base_score + (self._y_sum_train - dropped_train)

                # Val evaluation: recompute dropped contributions on X_val
                if eval_set is not None:
                    dropped_val = np.zeros_like(self._y_sum_val)
                    for t_idx in np.flatnonzero(drop_mask):
                        tree, feats = self.trees[t_idx]
                        lr_i = self._tree_lr[t_idx]
                        dropped_val += lr_i * tree.predict(X_val[:, feats])

                    if self.normalize_type == "tree":
                        y_val_eval = (
                            self.base_score
                            + tree_scale_eval * (self._y_sum_val - delta_val - dropped_val)
                            + delta_val
                        )
                    else:
                        y_val_eval = self.base_score + (self._y_sum_val - dropped_val)

            train_score = self.loss_fn.loss(y, y_train_eval)
            self.train_scores_.append(train_score)

            if eval_set is not None:
                val_score = self.loss_fn.loss(y_val, y_val_eval)
                self.val_scores_.append(val_score)

                # early stopping logic
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

                if (self.early_stopping_rounds and
                    no_improve_rounds >= self.early_stopping_rounds):
                    if verbose:
                        print(f"⏹️  Early stopping at iteration {iteration}, "
                              f"best iteration {best_iter} with val={best_score:.6f}")
                    if best_state is not None:
                        (self.trees, self.base_score, self._lr_history,
                         self._y_sum_train, self._y_sum_val, self._tree_lr) = best_state
                    break

            # 7) Commit the new trees
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
        """
        Predict using all trees with their individual learning rates.
        Robust to adaptive LR and re-fits.
        """
        if not self.trees:
            raise ValueError("Model not fitted yet")

        y = np.full(X.shape[0], self.base_score, dtype=np.float64)

        if len(self._tree_lr) != len(self.trees):
            # Defensive fallback: assume constant LR
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
        """
        importance_type ∈ {"gain", "cover", "split", "perm", "input_grad"}
        """
        importance_dict = self.feature_importance_.get_importance(importance_type)
        arr = np.zeros(self.n_features_, dtype=np.float64)
        for k, v in importance_dict.items():
            if 0 <= k < self.n_features_:
                arr[k] = v
        if normalize and arr.sum() > 0:
            arr /= arr.sum()
        return arr

    def get_training_history(self) -> Dict[str, List[float]]:
        history = {
            'train_scores': self.train_scores_,
            'learning_rates': self._lr_history,
        }
        if self.val_scores_:
            history['val_scores'] = self.val_scores_
        return history
