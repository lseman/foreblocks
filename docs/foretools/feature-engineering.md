# Feature Engineering

`foretools.fengineer` provides a full supervised feature engineering pipeline: transformation, redundancy filtering, and selection. This document covers the mathematical foundations of each stage and the criteria used to evaluate whether the resulting feature set is actually better.

---

## Pipeline overview

```
Raw DataFrame  X ∈ ℝ^{n×d}
       │
       ▼
  MathematicalTransformer      ← monotone transforms, power transforms
  InteractionTransformer        ← pairwise ops, polynomials
  StatisticalTransformer        ← row-wise aggregates
  BinningTransformer            ← quantile bins
  CategoricalTransformer        ← target-encoding, label-encoding
  RandomFourierFeaturesTransformer  ← kernel approximation
       │
       ▼
  CorrelationFilter             ← drop near-redundant columns
       │
       ▼
  FeatureSelector               ← MI / RFECV / Boruta
       │
       ▼
  QuantileTransformer           ← optional final normalisation
       │
       ▼
  X' ∈ ℝ^{n×d'}   (d' ≤ d)
```

---

## 1. Transformation stage

### 1.1 Mathematical transforms

Each numerical column $x_j$ is evaluated against a set of monotone candidates. The candidate that maximises a combined gain over the raw column is kept:

$$
g(t) = \underbrace{\bigl(S_\text{norm}(x_j) - S_\text{norm}(t(x_j))\bigr)}_{\text{normality gain}} + \alpha \underbrace{\bigl(\rho(t(x_j), y) - \rho(x_j, y)\bigr)}_{\text{target-correlation gain}}
$$

where:

- $S_\text{norm}(v) = |\text{skew}(v)| + 0.5\,|\text{excess kurtosis}(v)|$ — a shape score; lower is more Gaussian.
- $\rho(v, y) = |\text{Pearson}(v, y)|$ — absolute linear correlation with the target.
- $\alpha \in [0, 1]$ controls target-awareness (`math_target_weight`, default 0.25).

A transform is kept when $g(t) > \delta$ (default $\delta = 0.1$). The available candidates are:

| Name | Formula | Domain |
|---|---|---|
| `log` | $\ln(1 + x)$ | $x > 0$ |
| `sqrt` | $\sqrt{x}$ | $x \ge 0$ |
| `reciprocal` | $1/x$ | $x \ne 0$ |
| `slog1p` | $\text{sgn}(x)\ln(1+|x|)$ | $\mathbb{R}$ |
| `asinh` | $\sinh^{-1}(x)$ | $\mathbb{R}$ |
| `yeo-johnson` | $\text{YJ}_\lambda(x)$ (fitted) | $\mathbb{R}$ |
| `box-cox` | $\text{BC}_\lambda(x)$ (fitted) | $x > 0$ |

The Yeo–Johnson family unifies Box–Cox for positive and negative inputs. Given optimal $\hat\lambda$ fitted by MLE:

$$
\text{YJ}_\lambda(x) = \begin{cases}
\dfrac{(x+1)^\lambda - 1}{\lambda} & x \ge 0,\; \lambda \ne 0 \\[4pt]
\ln(x + 1) & x \ge 0,\; \lambda = 0 \\[4pt]
-\dfrac{(1-x)^{2-\lambda} - 1}{2 - \lambda} & x < 0,\; \lambda \ne 2 \\[4pt]
-\ln(1 - x) & x < 0,\; \lambda = 2
\end{cases}
$$

### 1.2 Interaction features

For every ordered pair $(j, k)$ from a prescreened column set, interactions of the form $f(x_j, x_k)$ are generated. The pair pre-screen uses a variance–decorrelation heuristic:

$$
h(j, k) = \text{IQR}(x_j)^2 \cdot \text{IQR}(x_k)^2 \cdot \bigl(1 - |\rho_{jk}|\bigr)
$$

This prioritises pairs that are individually informative but not already collinear. Only the top-$K$ pairs (default 800) proceed to feature generation.

Operations applied to each retained pair:

| Operation | Formula | Notes |
|---|---|---|
| `sum` | $x_j + x_k$ | commutative |
| `diff` | $x_j - x_k$ | |
| `prod` | $x_j \cdot x_k$ | commutative |
| `ratio` | $x_j / x_k$ | safe division |
| `norm_ratio` | $(x_j - x_k)/(|x_j|+|x_k|+\varepsilon)$ | bounded in $[-1, 1]$ |
| `zdiff` | $(x_j - \bar x_j) - (x_k - \bar x_k)$ | mean-centered diff |
| `log_ratio` | $\ln(1+|x_j|) - \ln(1+|x_k|)$ | log scale diff |
| `root_prod` | $\text{sgn}(x_j x_k)\sqrt{|x_j x_k|}$ | signed geometric mean |
| `min` / `max` | $\min(x_j, x_k)$ / $\max(x_j, x_k)$ | commutative |

Polynomials (squared, sqrt, cubed, reciprocal, log) are also generated per column before the same scoring/selection step.

### 1.3 Statistical aggregates

For a sample with $d'$ numerical features, row-wise summaries compress the feature vector into a fixed-size descriptor:

$$
\phi_\text{stat}(x) = \bigl[\bar{x},\; \sigma_x,\; x_{(0.5)},\; x_{\min},\; x_{\max},\; x_{\max}-x_{\min},\; \text{skew}(x),\; c_\text{nonull}\bigr]
$$

These are useful when individual feature magnitudes carry relative rather than absolute information (e.g., multi-sensor time-series windows).

### 1.4 Random Fourier Features (RFF)

For a shift-invariant kernel $k(x, z) = k(x - z)$, Bochner's theorem guarantees:

$$
k(x - z) = \mathbb{E}_{\omega \sim p(\omega)}\bigl[e^{i\omega^T(x-z)}\bigr]
$$

The `RandomFourierFeaturesTransformer` approximates this with $D$ random projections:

$$
\hat{\phi}(x) = \sqrt{\tfrac{2}{D}}\,\bigl[\cos(\omega_1^T x + b_1),\, \ldots,\, \cos(\omega_D^T x + b_D)\bigr]
$$

where $\omega_i \sim \mathcal{N}(0, \gamma I)$ (RBF kernel) and $b_i \sim \text{Uniform}(0, 2\pi)$. The inner product $\hat{\phi}(x)^T \hat{\phi}(z) \approx k(x, z)$, approximating a kernel SVM or kernel regression in a linear feature space. $D$ controls the approximation quality; $D = 50$ is the default.

---

## 2. Redundancy filtering

### 2.1 Correlation filter

After transformation, features are deduplicated by Pearson correlation. The upper triangle of the absolute correlation matrix $|R| \in [0,1]^{p \times p}$ is scanned:

$$
|r_{jk}| = \frac{|\text{Cov}(x_j, x_k)|}{\sigma_j \sigma_k}
$$

For any pair with $|r_{jk}| > \tau$ (default 0.95), the feature with lower variance is dropped:

$$
\text{drop} = \arg\min_{j,k}\, \text{Var}(x)
$$

Alternatively (`method="target_corr"`), the feature less correlated with $y$ is dropped:

$$
\text{drop} = \arg\min_{j,k}\, |\text{Corr}(x, y)|
$$

This ensures the remaining feature set spans distinct directions in feature space.

---

## 3. Feature selection — evaluating quality

This is the core question: **is the engineered feature set actually better?** The pipeline offers three nested approaches of increasing rigour.

### 3.1 Mutual Information (default)

Mutual information between feature $X_j$ and target $Y$ measures the reduction in uncertainty about $Y$ given $X_j$:

$$
I(X_j;\, Y) = \int\!\int p(x, y)\, \ln \frac{p(x, y)}{p(x)\,p(y)}\, dx\, dy
$$

For continuous variables this is estimated via $k$-NN density estimation or histogram binning. The pipeline uses `AdaptiveMI`, which applies a multi-scale binning approach over bin counts $k \in \{3, 5, 10\}$:

$$
\hat{I}_k(X_j;\,Y) = \sum_{a,b} \hat{p}_{k}(a, b)\, \ln \frac{\hat{p}_k(a,b)}{\hat{p}_k(a)\,\hat{p}_k(b)}
$$

and aggregates across scales. A Spearman pre-gate ($|\rho| > \tau_s$) filters near-zero-MI features before expensive scoring.

**Stability across folds.** The MI estimate is noisy on small samples. With `selector_stable_mi=True`, scores are computed on $K$ random folds and the median is taken:

$$
\hat{I}_\text{stable}(X_j;\,Y) = \text{median}_{k=1}^{K}\; \hat{I}^{(k)}(X_j;\,Y)
$$

Features with median MI below a threshold or with positive MI in fewer than $\lceil p \cdot K \rceil$ folds are discarded ($p$ = `selector_min_freq`, default 0.5).

**Redundancy pruning.** After MI ranking, a greedy correlation pass (threshold 0.98) removes features that are near-duplicates of a higher-ranked feature, preserving MI ranking order.

**Selection criterion.** Feature $j$ is retained if:

$$
\hat{I}_\text{stable}(X_j;\,Y) > \tau_\text{MI}
$$

with $\tau_\text{MI} = 0.01$ by default.

### 3.2 RFECV — recursive model-based selection

`AdvancedRFECV` wraps a model-based backward elimination with cross-validated scoring. The key idea is to measure **downstream predictive performance** as features are removed, finding the smallest subset $S^*$ such that the CV score does not degrade significantly.

**Algorithm.** Let $S_0 = \{1, \ldots, p\}$ and $m$ be an ensemble of estimators (Random Forest + Ridge/LR by default).

1. Evaluate $\text{CV}(S_t)$ = mean cross-validated score on feature subset $S_t$.
2. Compute ensemble feature importance $I_j^{(t)}$:
   $$
   I_j = \frac{1}{|M|} \sum_{m \in M} w_m\, \hat{I}_j^{(m)}
   $$
   where $\hat{I}_j^{(m)}$ is tree impurity importance or $|\hat{\beta}_j|$ for linear models.
3. Remove the $s$ features with lowest $I_j$ ($s$ = `step`, default 10% of current count).
4. Stop when no improvement exceeds $\delta$ (`improvement_threshold`) for `patience` consecutive rounds.

The best subset is $S^* = \arg\max_{t}\, \text{CV}(S_t)$.

**Stability selection within RFECV.** With `stability_selection=True`, each elimination round runs the full $K$-fold CV independently:

$$
I_j^{\text{stable}} = \frac{1}{K} \sum_{k=1}^{K} I_j^{(k)}
$$

This reduces variance in the importance estimate and gives more reliable elimination order.

**Scoring metrics.** The CV scorer depends on task type:

| Task | Default scorer |
|---|---|
| Regression | $-\text{MSE}$ (neg. mean squared error) |
| Classification | Accuracy |

Any scikit-learn compatible `scoring` string is accepted.

**Reading RFECV results.** After fitting:

```python
eng.plot_rfecv_results()
# shows: CV score vs. number of features
#        feature importance bar chart for selected set
```

The inflection point on the CV-score curve marks $|S^*|$. A flat or decreasing curve after the peak indicates the eliminated features were noise.

### 3.3 Boruta

Boruta is a wrapper method that tests each feature against a randomised shadow copy of itself. For each feature $x_j$ a shadow feature $x_j^\text{shadow}$ is generated by random permutation of the column. A Random Forest is then trained on $[X \| X^\text{shadow}]$ and the importances compared:

$$
Z_j = \frac{\bar{I}_j - \mu_{\text{shadow}}}{\sigma_{\text{shadow}}}
$$

Features with $Z_j$ significantly greater than the maximum shadow importance (MZSA test, Bonferroni-corrected over iterations) are accepted. The test is iterated for `max_iter` rounds (default 20).

**Advantage over MI.** Boruta accounts for joint redundancy; a feature may have high MI individually but add nothing given other features. It is equivalent to testing for conditional importance:

$$
I(X_j;\,Y \mid X_{-j}) > 0
$$

---

## 4. Evaluating the feature set — diagnostics

### 4.1 Per-feature MI score

```python
scores = eng.get_feature_importance()   # pd.Series, sorted descending
```

A good feature set has a monotonically decreasing MI profile with a clear elbow. Features to the right of the elbow are likely noise. A flat profile (all MI near zero) indicates either a weak signal or poor transformation choices.

### 4.2 RFECV CV-score curve

```python
eng.plot_rfecv_results()
```

Interpret the left panel (CV score vs. feature count):

- **Sharp peak then plateau**: good separation between signal and noise.
- **Monotone increase**: all retained features contribute; consider relaxing `min_features_to_select`.
- **Flat or noisy**: model is insensitive to this feature subset — check for target leakage or excessive noise.

The **feature reduction ratio**:

$$
r = \frac{p - |S^*|}{p}
$$

is available via `rfecv_selector_.get_performance_summary()["feature_reduction_ratio"]`. Values $r > 0.5$ indicate the pipeline successfully compressed the expanded feature space.

### 4.3 Correlation audit

After `CorrelationFilter`:

```python
pairs = eng.correlation_filter_.correlation_pairs_  # [(col1, col2, |r|), ...]
```

Inspect `pairs` to verify no informative feature was dropped. Pairs with $|r| \approx 1.0$ and high MI are a sign of duplicate transformations.

### 4.4 Transformation gain audit

`MathematicalTransformer` selects transforms by gain $g(t)$. You can inspect which transforms were kept:

```python
kept = eng.transformers_["mathematical"].valid_transforms_  # {col: [transform_names]}
power_cols = eng.transformers_["mathematical"].valid_cols_power_  # [col_names]
```

If a feature is absent from `valid_transforms_` and `valid_cols_power_`, the transformer found no improvement over identity — the feature was already well-conditioned.

### 4.5 Interaction score audit

```python
scores = eng.transformers_["interactions"].feature_scores_   # {feature_name: MI_score}
```

Top interaction scores reflect pairs whose combined signal exceeds either individual column. A high-scoring interaction $x_j \cdot x_k$ that wasn't in the raw data indicates a nonlinear relationship that linear models would otherwise miss.

### 4.6 Stability diagnostic

Run the pipeline on bootstrap resamples and measure **feature selection stability** (Kuncheva index):

$$
K(S_a, S_b) = \frac{|S_a \cap S_b| - \frac{|S_a||S_b|}{p}}{\min(|S_a|, |S_b|) - \frac{|S_a||S_b|}{p}}
\in [-1, 1]
$$

$K \to 1$ means the same features are selected regardless of which training fold is used. $K < 0.5$ indicates instability — consider increasing sample size, tightening `mi_threshold`, or switching from MI to RFECV.

---

## 5. Usage example

```python
from foretools.fengineer import FeatureEngineer
from foretools.fengineer.transformers.config import FeatureConfig

cfg = FeatureConfig(
    selector_method="rfecv",   # "mi" | "rfecv" | "boruta" | "auto"
    rfecv_cv=5,
    rfecv_step=0.1,
    rfecv_use_ensemble=True,
    rfecv_stability_selection=True,
    corr_threshold=0.95,
    create_rff=False,
    use_quantile_transform=True,
)

eng = FeatureEngineer(config=cfg)
eng.fit(X_train, y_train)

X_train_eng = eng.transform(X_train)
X_test_eng  = eng.transform(X_test)

# Diagnostics
eng.plot_feature_importance(top_k=30)
eng.plot_rfecv_results()

report = eng.get_transformation_report()
print(report["feature_reduction_ratio"])
print(report["top_features"])
```

---

## 6. Selection method comparison

| Method | What it measures | Accounts for redundancy | Cost | Best when |
|---|---|---|---|---|
| **MI** | $I(X_j;\,Y)$ marginal | No (pruned post-hoc) | Low | Large data, fast iteration |
| **Stable MI** | Median $I(X_j;\,Y)$ over folds | No | Medium | Noisy targets, moderate $n$ |
| **RFECV** | Downstream $\text{CV}(S)$ | Yes (via model) | High | Small to medium data, need minimal set |
| **Boruta** | $I(X_j;\,Y \mid X_{-j})$ (approx.) | Yes | High | Rigorous all-relevant selection |

For time-series forecasting use cases with autocorrelated residuals, RFECV with `KFold` (not stratified) is preferred; standard CV underestimates error when folds overlap in time — consider using a time-based split via a custom `cv` splitter.

---

## Related pages

- [Foretools overview](index.md)
- [AutoDA Augmentation](tsaug.md)
- [Repository map](../reference/repository-map.md)
