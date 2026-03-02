import numpy as np
import foreforest
from foreforest import ForeForest, ForeForestConfig, TreeConfig
from sklearn.metrics import mean_squared_error

print("Testing Monotonic Constraints and Sparsity Handling")

# 1. Monotonic constraints
# Generate dataset where y is strictly positively correlated with X[:, 0]
# and negatively correlated with X[:, 1]
N = 1000
P = 3
np.random.seed(42)
X = np.random.rand(N, P)
y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(N) * 0.1

cfg_c = ForeForestConfig()
cfg_c.n_estimators = 10
cfg_c.learning_rate = 0.1
cfg_c.tree_cfg.max_depth = 5
cfg_c.tree_cfg.monotone_constraints = [1, -1, 0]

cfg_u = ForeForestConfig()
cfg_u.n_estimators = 10
cfg_u.learning_rate = 0.1
cfg_u.tree_cfg.max_depth = 5

model_constrained = ForeForest(cfg_c)
model_unconstrained = ForeForest(cfg_u)

model_constrained.fit_complete(X, y)
model_unconstrained.fit_complete(X, y)

# Small test delta to check monotonicity
X_test = np.copy(X[:10])
pred_base_c = model_constrained.predict(X_test)

X_test_inc_0 = np.copy(X_test)
X_test_inc_0[:, 0] += 0.5
pred_inc_0_c = model_constrained.predict(X_test_inc_0)

X_test_inc_1 = np.copy(X_test)
X_test_inc_1[:, 1] += 0.5
pred_inc_1_c = model_constrained.predict(X_test_inc_1)

# Feature 0 constraint is +1, so increasing X0 MUST increase or maintain prediction
print(f"Monotone Constraint Feature 0 (+1):")
print(f"Base preds: {pred_base_c}")
print(f"Inc  preds: {pred_inc_0_c}")
violations = np.sum(pred_inc_0_c < pred_base_c)
print(f"Violations: {violations}")
assert violations == 0, "Monotonicity violated on feature 0 constraint"

# Feature 1 constraint is -1, so increasing X1 MUST decrease or maintain prediction
print(f"\nMonotone Constraint Feature 1 (-1):")
print(f"Base preds: {pred_base_c}")
print(f"Inc  preds: {pred_inc_1_c}")
violations = np.sum(pred_inc_1_c > pred_base_c)
print(f"Violations: {violations}")
assert violations == 0, "Monotonicity violated on feature 1 constraint"

print("\nAll monotonic constraint checks passed!")

# 2. Sparsity-Aware Splits
# Generate data with lots of missing values explicitly favoring a missing split direction
X_miss = np.random.rand(N, P)
y_miss = np.where(X_miss[:, 0] > 0.5, 1.0, 0.0)

# Simulate NaNs that strongly align with class 1
mask = np.random.rand(N) < 0.3
X_miss[mask, 0] = np.nan
y_miss[mask] = 1.0

c1 = ForeForestConfig()
c1.n_estimators = 10
c1.tree_cfg.max_depth = 3
c1.tree_cfg.missing_policy = foreforest.MissingPolicy.Learn

c2 = ForeForestConfig()
c2.n_estimators = 10
c2.tree_cfg.max_depth = 3
c2.tree_cfg.missing_policy = foreforest.MissingPolicy.AlwaysRight

c3 = ForeForestConfig()
c3.n_estimators = 10
c3.tree_cfg.max_depth = 3
c3.tree_cfg.missing_policy = foreforest.MissingPolicy.AlwaysLeft

m_learn = ForeForest(c1)
m_learn.fit_complete(X_miss, y_miss)
m_right = ForeForest(c2)
m_right.fit_complete(X_miss, y_miss)
m_left = ForeForest(c3)
m_left.fit_complete(X_miss, y_miss)

print("\nMissing Values Training MSE:")
print(f"Policy: Learn        MSE: {mean_squared_error(y_miss, m_learn.predict(X_miss)):.4f}")
print(f"Policy: AlwaysRight  MSE: {mean_squared_error(y_miss, m_right.predict(X_miss)):.4f}")
print(f"Policy: AlwaysLeft   MSE: {mean_squared_error(y_miss, m_left.predict(X_miss)):.4f}")

print("Sparsity tests complete.")
