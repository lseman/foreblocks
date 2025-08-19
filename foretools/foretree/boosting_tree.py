from boosting_aux import *


# ============================== Level-wise tree (fallback) ====================
class SingleTree:
    def __init__(
        self,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        lambda_=1.0,
        gamma=0.0,
        alpha=0.0,
        feature_indices=None,
        n_jobs=4,
        tree_method="hist",
        n_bins=256,
        bin_edges=None,
        monotone_constraints=None,
        interaction_constraints=None,
        gpu_accelerator=None,
        max_delta_step=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.feature_indices = feature_indices
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.monotone_constraints = monotone_constraints or {}
        self.interaction_constraints = interaction_constraints
        self.gpu_accelerator = gpu_accelerator or GPUAccelerator()
        self.max_delta_step = max_delta_step

        self.left = self.right = None
        self.feature_index = self.threshold = self.bin_threshold = None
        self.leaf_value = None
        self.n_samples = 0
        self.gain = 0.0

    def _prepare_monotone_constraints(self, n_features):
        if self.monotone_constraints is None:
            return np.zeros(n_features, dtype=np.int8)
        arr = np.zeros(n_features, dtype=np.int8)
        for f, c in self.monotone_constraints.items():
            arr[f] = np.int8(c)  # enforce -1, 0, +1
        return arr

    def _best_split_hist_fast(self, X, g, h):
        if self.bin_edges is None:
            return None, None, -np.inf
        feature_bin_edges = np.array([self.bin_edges[i] for i in self.feature_indices])
        hist_g, hist_h = self.gpu_accelerator.build_histograms(
            X, g, h, feature_bin_edges, self.n_bins
        )
        # Use missing-aware finder for parity
        mono_arr = None
        if self.monotone_constraints:
            mono_arr = self._prepare_monotone_constraints(len(self.feature_indices))

        f_loc, b_idx, gain, _miss_dir = find_best_splits_with_missing(
            hist_g, hist_h, self.lambda_, self.gamma, self.n_bins, 1e-6, mono_arr
        )
        if f_loc == -1:
            return None, None, -np.inf
        actual_feat_idx = self.feature_indices[f_loc]
        threshold = feature_bin_edges[f_loc, b_idx]
        return actual_feat_idx, threshold, gain

    def fit(self, X, g, h, depth=0, feature_importance=None):
        self.n_samples = len(g)
        if (
            (depth >= self.max_depth)
            or (len(g) < self.min_samples_split)
            or (np.sum(np.abs(h)) < 1e-6)
        ):
            self.leaf_value = calc_leaf_value_newton(
                np.sum(g), np.sum(h), self.lambda_, self.alpha, self.max_delta_step
            )
            return self

        feat_idx, threshold, gain = self._best_split_hist_fast(X, g, h)
        if feat_idx is None or gain <= 0:
            self.leaf_value = calc_leaf_value_newton(
                np.sum(g), np.sum(h), self.lambda_, self.alpha, self.max_delta_step
            )
            return self

        local_feat_idx = np.where(self.feature_indices == feat_idx)[0][0]
        self.feature_index = local_feat_idx
        self.threshold = threshold
        self.gain = gain

        if feature_importance:
            feature_importance.update(feat_idx, gain, len(g))

        x_col = X[:, self.feature_index]
        miss = ~np.isfinite(x_col)
        # Simple default: send missing left (could compute both as above)
        mask = (x_col <= self.threshold) | miss

        left_mask = mask
        right_mask = ~mask
        if (
            left_mask.sum() < self.min_samples_leaf
            or right_mask.sum() < self.min_samples_leaf
        ):
            self.leaf_value = calc_leaf_value_newton(
                np.sum(g), np.sum(h), self.lambda_, self.alpha, self.max_delta_step
            )
            return self

        self.left = SingleTree(
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.lambda_,
            self.gamma,
            self.alpha,
            self.feature_indices,
            self.n_jobs,
            self.tree_method,
            self.n_bins,
            self.bin_edges,
            self.monotone_constraints,
            self.interaction_constraints,
            self.gpu_accelerator,
            self.max_delta_step,
        ).fit(X[left_mask], g[left_mask], h[left_mask], depth + 1, feature_importance)

        self.right = SingleTree(
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.lambda_,
            self.gamma,
            self.alpha,
            self.feature_indices,
            self.n_jobs,
            self.tree_method,
            self.n_bins,
            self.bin_edges,
            self.monotone_constraints,
            self.interaction_constraints,
            self.gpu_accelerator,
            self.max_delta_step,
        ).fit(
            X[right_mask], g[right_mask], h[right_mask], depth + 1, feature_importance
        )

        return self

    def predict(self, X):
        if self.leaf_value is not None:
            return np.full(X.shape[0], self.leaf_value)
        x_col = X[:, self.feature_index]
        miss = ~np.isfinite(x_col)
        mask = (x_col <= self.threshold) | miss
        y_pred = np.empty(X.shape[0])
        y_pred[mask] = self.left.predict(X[mask])
        y_pred[~mask] = self.right.predict(X[~mask])
        return y_pred
