# 📊 Distribution Analysis

In this module, we analyze the **univariate distribution** of each numeric feature in a dataset using **descriptive statistics**, **entropy-based metrics**, and **normality tests**.

We denote a numeric feature vector by:

$$
X = \{x_1, x_2, \dots, x_n\} \subset \mathbb{R}
$$

with $n = \text{number of non-null observations}$.

---

## 1. **Central Tendency and Dispersion**

### 🔹 Sample Mean

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

### 🔹 Sample Standard Deviation

$$
s = \sqrt{ \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2 }
$$

### 🔹 Minimum and Maximum

$$
\min(X) = \min_i x_i, \quad \max(X) = \max_i x_i
$$

### 🔹 Range

$$
\text{Range}(X) = \max(X) - \min(X)
$$

### 🔹 Coefficient of Variation (CV)

A normalized measure of dispersion:

$$
\text{CV} = \frac{s}{|\bar{x}|}, \quad \text{provided } \bar{x} \ne 0
$$

---

## 2. **Shape Descriptors**

### 🔹 Skewness (3rd standardized moment)

Measures asymmetry:

$$
\text{Skew}(X) = \frac{1}{n} \sum_{i=1}^n \left( \frac{x_i - \bar{x}}{s} \right)^3
$$

* Skew > 0: Right-skewed (long tail to the right)
* Skew < 0: Left-skewed

### 🔹 Kurtosis (4th standardized moment)

Measures tailedness:

$$
\text{Kurtosis}(X) = \frac{1}{n} \sum_{i=1}^n \left( \frac{x_i - \bar{x}}{s} \right)^4
$$

* A Gaussian distribution has kurtosis = 3.
* We compute **excess kurtosis** as:

$$
\text{Excess Kurtosis} = \text{Kurtosis} - 3
$$

---

## 3. **Entropy-Based Metrics**

### 🔹 Histogram-Based Entropy

Let $p_i$ be the estimated probability density in the $i$-th bin (normalized to sum to 1):

$$
\text{Entropy}(X) = - \sum_{i=1}^k p_i \log_2(p_i)
$$

Where:

* $k$: number of bins (typically 30)
* $p_i = \frac{\text{count in bin } i}{n}$

> Entropy captures the **degree of uncertainty** or spread in the distribution:
>
> * Low entropy: concentrated distribution
> * High entropy: dispersed values

---

## 4. **Quantiles and Spread**

Let $q_p = \text{Quantile}(X, p)$

* **First Quartile (Q1)**: $q_{0.25}$
* **Median (Q2)**: $q_{0.5}$
* **Third Quartile (Q3)**: $q_{0.75}$

### 🔹 Interquartile Range (IQR)

$$
\text{IQR} = Q3 - Q1
$$

---

## 5. **Normality Tests**

### 🔹 D’Agostino & Pearson’s Omnibus Test

Tests combined skewness and kurtosis deviation from normality.

* Null Hypothesis $H_0$: data is drawn from a Gaussian distribution.
* p-value > α ⇒ fail to reject normality.

### 🔹 Shapiro-Wilk Test

A powerful test for normality on small samples:

* Null Hypothesis $H_0$: sample is normally distributed.
* For $n > 5000$, not reliable.

---

## 6. **Bimodality and Tails**

### 🔹 Bimodality Coefficient

$$
\text{BC} = \frac{s^2 + 1}{k}, \quad \text{where } s = \text{Skewness},\ k = \text{Kurtosis}
$$

* If BC > 0.555 → possibly bimodal.

### 🔹 Tail Ratio

Measures asymmetry of tails:

$$
\text{Tail Ratio} = \frac{\max(X) - q_{0.95}}{q_{0.05} - \min(X) + \varepsilon}
$$

* High tail ratio → heavy right tail
* $\varepsilon$ is a small constant to avoid division by zero.

---

## 7. **Gaussianity Check (Heuristic)**

We define a feature as **Gaussian** if:

* $p_{\text{normaltest}} > \alpha$ (e.g. α = 0.05)
* and $|\text{Skewness}| < 1$

This acts as a **soft classifier** of approximate normality.

Great — let’s move to a **detailed, mathematically rigorous explanation** of the **Correlation Analysis** module, suitable for students in a Data Science course.

---

# 🔗 Correlation Analysis

Correlation analysis quantifies the **degree of dependence or association** between two variables.

Given two numeric random variables $X = \{x_1, ..., x_n\}$ and $Y = \{y_1, ..., y_n\}$, we analyze their relationship using **linear**, **monotonic**, **nonlinear**, and **information-theoretic** measures.

---

## 1. **Pearson Correlation** (Linear Correlation)

### ✅ When to use:

Both variables are **interval-scaled** and **normally distributed**.

### 🔹 Formula:

$$
\rho_{X,Y} = \frac{\operatorname{cov}(X, Y)}{\sigma_X \sigma_Y}
= \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}
$$

* $\rho \in [-1, 1]$
* $\rho = 1$: perfect positive linear relationship
* $\rho = 0$: no linear relationship
* $\rho = -1$: perfect negative linear relationship

---

## 2. **Spearman’s Rank Correlation** (Monotonic Association)

### ✅ When to use:

Data is **ordinal**, **non-normally distributed**, or contains **nonlinear but monotonic** relationships.

### 🔹 Method:

1. Rank both variables: $R_i = \text{rank}(x_i)$, $S_i = \text{rank}(y_i)$
2. Apply Pearson correlation on the ranks:

$$
\rho_s = \frac{\sum_{i=1}^n (R_i - \bar{R})(S_i - \bar{S})}{\sqrt{\sum_{i=1}^n (R_i - \bar{R})^2} \cdot \sqrt{\sum_{i=1}^n (S_i - \bar{S})^2}}
$$

### 🔸 Simplified for no ties:

$$
\rho_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}, \quad d_i = R_i - S_i
$$

---

## 3. **Mutual Information (MI)**

### ✅ When to use:

You want to measure **nonlinear** or **non-monotonic dependencies**.

### 🔹 Definition (Discrete version):

$$
I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \left( \frac{p(x, y)}{p(x) p(y)} \right)
$$

* $I(X; Y) = 0$ ⇔ $X \perp Y$ (statistical independence)
* MI is **always non-negative**
* High MI → greater shared information

### 🔸 Estimation:

In practice (as in your code), MI is estimated using **k-nearest neighbor** estimators or **entropy estimators** on discretized bins or KDE.

---

## 4. **Distance Correlation** (dCor)

### ✅ When to use:

You want to detect **any kind of dependence** — linear or nonlinear.

### 🔹 Key Property:

$$
\text{dCor}(X, Y) = 0 \iff X \perp Y
$$

Unlike Pearson/Spearman, distance correlation = 0 **only when X and Y are independent**.

### 🔹 Definitions:

Let $A$ and $B$ be **distance matrices**:

$$
A_{ij} = \|x_i - x_j\|, \quad B_{ij} = \|y_i - y_j\|
$$

Let $\tilde{A}, \tilde{B}$ be **double-centered**:

$$
\tilde{A}_{ij} = A_{ij} - \bar{A}_{i\cdot} - \bar{A}_{\cdot j} + \bar{A}_{\cdot \cdot}
$$

Then the **distance covariance** is:

$$
\text{dCov}^2(X, Y) = \frac{1}{n^2} \sum_{i,j} \tilde{A}_{ij} \tilde{B}_{ij}
$$

And the **distance correlation** is:

$$
\text{dCor}(X, Y) = \frac{\text{dCov}(X, Y)}{\sqrt{\text{dCov}(X, X) \cdot \text{dCov}(Y, Y)}}
$$

---

## 5. **PhiK Correlation** (Optional, Categorical-Aware)

### ✅ When to use:

* Mix of **categorical and numerical** data.
* **Nonlinear + non-monotonic** dependencies.

### 🔹 Method:

1. **Discretize** continuous variables into bins.
2. Construct a **contingency table** $O$.
3. Compute expected counts $E$ under independence.
4. Calculate Chi-squared statistic:

   $$
   \chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
   $$
5. Define:

   $$
   \phi_K = \sqrt{ \frac{\chi^2 / n}{\min(k - 1, r - 1)} }
   $$

   where $k, r$ are number of bins or categories in each variable.

* $\phi_K \in [0, 1]$
* $\phi_K = 0$: no association
* $\phi_K = 1$: perfect correlation (of any form)

---

## 📘 Summary Table

| Method         | Detects        | Range         | Zero Implies      |
| -------------- | -------------- | ------------- | ----------------- |
| Pearson        | Linear         | $[-1, 1]$     | No linear rel.    |
| Spearman       | Monotonic      | $[-1, 1]$     | No monotonic rel. |
| Mutual Info    | Nonlinear      | $[0, \infty)$ | Independence      |
| Distance Corr. | Any dependence | $[0, 1]$      | Independence      |
| PhiK           | Mixed types    | $[0, 1]$      | No dependency     |

---

# 🚨 Outlier Detection

Outlier detection aims to identify **observations that deviate significantly** from the majority of the data. These can indicate errors, rare events, or novel phenomena.

Let $X \in \mathbb{R}^{n \times d}$ be a matrix of $n$ observations and $d$ numeric features.

The methods used fall into **four families**:

1. Distance-based (Mahalanobis)
2. Ensemble-based (Isolation Forest)
3. Probabilistic (Elliptic Envelope)
4. Density/Neighborhood-based (LOF)
5. Kernel-based (One-Class SVM)

---

## 1. **Robust Preprocessing**

Before detection, features are optionally scaled using:

* **RobustScaler**: median-centered with IQR normalization.

$$
X_{scaled} = \frac{X - \text{Median}(X)}{\text{IQR}(X)}
$$

This is resilient to outliers and skewed distributions.

---

## 2. **Isolation Forest**

### ✅ Intuition:

* Anomalies are easier to isolate.
* Random trees partition the feature space; outliers tend to be split early.

### 🔹 Algorithm:

* Construct a forest of random binary trees.
* Compute average **path length** $h(x)$ for observation $x$.
* Define anomaly score:

  $$
  s(x) = 2^{-\frac{h(x)}{c(n)}}
  $$

  where $c(n)$ is the average path length in an unsuccessful binary search tree.

### 🔸 Threshold:

Observations with $s(x) > \tau$ are flagged as outliers. Typically, $\tau$ is set to return a user-specified contamination rate $\alpha$.

---

## 3. **Elliptic Envelope** (Minimum Covariance Determinant)

### ✅ Assumption:

* Data is **Gaussian-like** in a multivariate sense.

### 🔹 Method:

* Fit a **robust Gaussian** distribution to the data.
* Use **Minimum Covariance Determinant (MCD)** to estimate $\mu$ and $\Sigma$.
* Compute **Mahalanobis Distance**:

  $$
  D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
  $$
* Flag outliers with:

  $$
  D_M(x)^2 > \chi^2_{d, 1 - \alpha}
  $$

  where $\chi^2_{d}$ is the chi-squared distribution with $d$ degrees of freedom.

---

## 4. **Local Outlier Factor (LOF)**

### ✅ Principle:

Compares the **local density** of a point to its neighbors.

### 🔹 Definitions:

Let $k$-NN be the $k$ nearest neighbors of $x$.

1. **Reachability distance**:

   $$
   \text{reach-dist}_k(x, o) = \max\left\{ \text{k-distance}(o),\ d(x, o) \right\}
   $$

2. **Local Reachability Density**:

   $$
   \text{lrd}_k(x) = \left( \frac{1}{|N_k(x)|} \sum_{o \in N_k(x)} \text{reach-dist}_k(x, o) \right)^{-1}
   $$

3. **LOF score**:

   $$
   \text{LOF}_k(x) = \frac{1}{|N_k(x)|} \sum_{o \in N_k(x)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(x)}
   $$

* If $\text{LOF}_k(x) \gg 1$, $x$ is an outlier (less dense than neighbors).

---

## 5. **One-Class Support Vector Machine (SVM)**

### ✅ Concept:

* Learns a **decision boundary** that encloses the normal data.

### 🔹 Objective:

Solve:

$$
\min_{w, \rho, \xi} \frac{1}{2} \|w\|^2 + \frac{1}{\nu n} \sum_{i=1}^n \xi_i - \rho
$$

Subject to:

$$
(w \cdot \phi(x_i)) \geq \rho - \xi_i, \quad \xi_i \geq 0
$$

* $\phi(\cdot)$: feature map (e.g., RBF kernel)
* $\nu \in (0,1]$: upper bound on outlier fraction

### 🔸 Decision Function:

$$
f(x) = \text{sign}(w \cdot \phi(x) - \rho)
$$

* Negative ⇒ Outlier

---

## 6. **Mahalanobis Distance (Manual)**

If covariance matrix $\Sigma$ is well-conditioned:

### 🔹 Formula:

$$
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$

Used to compute a univariate outlier score. Cutoff is based on quantile of $\chi^2$ distribution:

$$
x \text{ is an outlier if } D_M(x)^2 > \chi^2_{d, 1-\alpha}
$$

---

## 7. **Output**

Each method returns:

* `outliers`: Boolean mask
* `count`: Number of detected outliers
* `percentage`: Outlier rate:

  $$
  \text{Outlier Rate} = \frac{\text{count}}{n} \times 100
  $$

---

## 📘 Summary Table

| Method            | Type           | Strength                        | Weakness                           |
| ----------------- | -------------- | ------------------------------- | ---------------------------------- |
| Isolation Forest  | Ensemble       | Fast, nonlinear, high-dim       | Approximate, random variance       |
| Elliptic Envelope | Statistical    | Multivariate Gaussian detection | Fails on non-Gaussian data         |
| LOF               | Density-based  | Captures local anomalies        | Sensitive to parameter $k$         |
| One-Class SVM     | Kernel-based   | Captures nonlinear boundaries   | Expensive, sensitive to $\nu$      |
| Mahalanobis       | Distance-based | Fast, interpretable             | Sensitive to covariance estimation |

---

## 🧪 Implementation Notes

* **RobustScaler** is preferred if heavy-tailed distributions are suspected.
* **EllipticEnvelope** assumes elliptical (Gaussian) contours.
* **Mahalanobis** is computed manually with `EmpiricalCovariance`.

---

# 🔍 Clustering Analysis

Clustering aims to **partition** a dataset $X \in \mathbb{R}^{n \times d}$ into $k$ groups (clusters) such that data points within the same cluster are **more similar** to each other than to those in other clusters.

This module supports **multiple algorithms**:

* K-Means
* Spectral Clustering
* Gaussian Mixture Models (GMM)
* HDBSCAN (optional)

The analysis includes **cluster labels**, **model parameters**, and **internal validation metrics**.

---

## 1. **K-Means Clustering**

### ✅ Goal:

Minimize the **within-cluster sum of squared distances** (inertia):

$$
\underset{\{C_j\}_{j=1}^k}{\text{argmin}} \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
$$

where:

* $C_j$: cluster $j$
* $\mu_j$: centroid of cluster $C_j$

### 🔹 Algorithm:

1. Initialize $k$ centroids (random or k-means++)
2. Repeat until convergence:

   * Assign each point to its **nearest centroid**
   * Update centroids as the **mean** of their assigned points

### 🔸 Output:

* Labels: $\ell_i \in \{1, ..., k\}$
* Centroids $\mu_1, ..., \mu_k$
* Inertia (total squared distance to centroids)

---

## 2. **Spectral Clustering**

### ✅ Goal:

Use graph theory to cluster data using **eigenvectors of the Laplacian matrix**.

### 🔹 Steps:

1. Construct similarity matrix $W \in \mathbb{R}^{n \times n}$, e.g. Gaussian kernel:

   $$
   W_{ij} = \exp\left( -\frac{\|x_i - x_j\|^2}{2\sigma^2} \right)
   $$

2. Compute **graph Laplacian**:

   $$
   L = D - W, \quad \text{or normalized: } L_{sym} = I - D^{-1/2} W D^{-1/2}
   $$

   where $D$ is the degree matrix.

3. Compute the **first $k$ eigenvectors** of $L$ and stack into matrix $U$.

4. Run K-Means on the rows of $U$.

### 🔸 Strengths:

* Captures **non-convex clusters**
* Sensitive to **manifold structure**

---

## 3. **Gaussian Mixture Models (GMM)**

### ✅ Model:

Data is generated from a **mixture of $k$ Gaussian distributions**:

$$
p(x) = \sum_{j=1}^{k} \pi_j \cdot \mathcal{N}(x \mid \mu_j, \Sigma_j)
$$

* $\pi_j$: mixture weight ($\sum_j \pi_j = 1$)
* $\mathcal{N}$: multivariate normal distribution

### 🔹 Estimation:

Use **Expectation-Maximization (EM)**:

1. **E-Step**: compute responsibilities

   $$
   r_{ij} = \frac{ \pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \Sigma_j) }{ \sum_{l=1}^k \pi_l \cdot \mathcal{N}(x_i \mid \mu_l, \Sigma_l) }
   $$
2. **M-Step**: update parameters $\{\pi_j, \mu_j, \Sigma_j\}$

### 🔸 Model Selection:

Use **Bayesian Information Criterion (BIC)**:

$$
\text{BIC} = -2 \cdot \log \hat{L} + p \cdot \log(n)
$$

* Lower BIC → better model

---

## 4. **HDBSCAN** (Hierarchical Density-Based Clustering)

> Used if installed. Robust to **arbitrary shapes**, **noise**, and **varying densities**.

### 🔹 Key Concepts:

* Uses **mutual reachability distance**:

  $$
  d_{mr}(x_i, x_j) = \max\{ \text{core}_k(x_i), \text{core}_k(x_j), \|x_i - x_j\| \}
  $$
* Builds a **minimum spanning tree** over all pairwise distances
* **Condenses** the hierarchy using a **minimum cluster size** (e.g., 5)

### 🔸 Output:

* Soft clustering labels (including **noise points**: label = -1)
* Cluster stability metrics

---

## 5. **Internal Validation Metrics**

### ✅ Used to evaluate clustering **quality**:

#### a. **Silhouette Score**:

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}, \quad s(i) \in [-1, 1]
$$

* $a(i)$: average intra-cluster distance
* $b(i)$: lowest average inter-cluster distance

#### b. **Calinski-Harabasz Index**:

$$
\text{CH} = \frac{ \text{between-cluster dispersion} }{ \text{within-cluster dispersion} } \cdot \frac{n - k}{k - 1}
$$

#### c. **Davies-Bouldin Index** (lower is better):

$$
\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \ne i} \left( \frac{s_i + s_j}{d_{ij}} \right)
$$

* $s_i$: average intra-cluster distance
* $d_{ij}$: distance between centroids

---

## 📘 Summary Table

| Algorithm | Handles Shapes | Probabilistic | Noise Robust | Parameters         | Best for                  |
| --------- | -------------- | ------------- | ------------ | ------------------ | ------------------------- |
| K-Means   | ✖ (spherical)  | ✖             | ✖            | $k$                | Fast, convex clusters     |
| Spectral  | ✔              | ✖             | ✖            | $k$, similarity    | Manifold structure        |
| GMM       | ✔              | ✔             | ✖            | $k$, init          | Overlapping distributions |
| HDBSCAN   | ✔              | ✔ (soft)      | ✔            | min\_cluster\_size | Noisy, arbitrary clusters |

---

# 📈 Time Series Analysis

The **TimeSeriesAnalyzer** applies multiple techniques to extract **stationarity**, **trends**, **seasonality**, **volatility**, **change points**, **regimes**, and **forecasting readiness** from temporal features.

Let $X_t \in \mathbb{R}$ be a univariate time series indexed over time $t = 1, \dots, T$.

---

## 1. **Stationarity Analysis**

### 🔹 Definitions:

A series is **stationary** if its **statistical properties** (mean, variance, autocorrelation) do not change over time.

---

### a. **ADF Test (Augmented Dickey-Fuller)**

* **Null Hypothesis** $H_0$: The series has a unit root (non-stationary)
* **Test equation**:

$$
\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^p \phi_i \Delta X_{t-i} + \varepsilon_t
$$

* Test statistic: $\text{ADF} = \frac{\hat{\gamma}}{\text{SE}(\hat{\gamma})}$
* Reject $H_0$ if p-value < $\alpha$

---

### b. **KPSS Test (Kwiatkowski–Phillips–Schmidt–Shin)**

* **Null Hypothesis** $H_0$: The series is stationary
* Test statistic:

$$
\text{KPSS} = \frac{1}{T^2} \sum_{t=1}^{T} S_t^2 / \hat{\sigma}^2
$$

Where:

* $S_t = \sum_{i=1}^{t} e_i$ is the cumulative residuals from OLS detrending

---

### c. **Zivot-Andrews Test**

Tests for a **unit root with a structural break**.

* Null: unit root with no break
* Alternative: stationarity with one-time structural break in intercept/trend

---

### d. **Variance Ratio Test**

Tests the **random walk** hypothesis:

$$
\text{VR}(k) = \frac{\operatorname{Var}(X_t - X_{t-k})}{k \cdot \operatorname{Var}(X_t - X_{t-1})}
$$

* VR ≈ 1 ⇒ random walk
* VR < 1 ⇒ mean reversion
* VR > 1 ⇒ trending

---

### Stationarity Classification Logic:

| ADF | KPSS | Conclusion            |
| --- | ---- | --------------------- |
| ✔️  | ✔️   | Stationary            |
| ❌   | ❌    | Non-stationary        |
| ✔️  | ❌    | Trend-stationary      |
| ❌   | ✔️   | Difference-stationary |

---

## 2. **Trend Analysis**

### 🔹 Linear Trend (OLS)

$$
X_t = \beta_0 + \beta_1 t + \varepsilon_t
$$

* Significance of $\beta_1$: via $t$-test (p-value)
* $R^2$: goodness-of-fit

### 🔹 Polynomial Trend (Degree 2)

$$
X_t = \alpha + \beta_1 t + \beta_2 t^2 + \varepsilon_t
$$

### 🔹 HP Filter

Decomposes $X_t$ into:

$$
X_t = \tau_t + c_t
$$

* $\tau_t$: trend
* $c_t$: cyclical component
* Solves:

$$
\min_{\tau_t} \sum_{t=1}^{T} (X_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1} (\tau_{t+1} - 2\tau_t + \tau_{t-1})^2
$$

* λ = 1600 (quarterly), 129600 (monthly), 100 (annual)

---

## 3. **Seasonality Detection**

### a. **STL Decomposition**

$$
X_t = T_t + S_t + R_t
$$

* Seasonal-Trend decomposition using LOESS
* Detects period, seasonal strength:

  $$
  \text{Strength} = \frac{\operatorname{Var}(S_t)}{\operatorname{Var}(X_t)}
  $$

### b. **Periodogram / Spectral Analysis**

Analyzes dominant frequencies $f$ via Fourier transform:

$$
X_t = \sum_{k=1}^{K} A_k \cos(2\pi f_k t + \phi_k)
$$

* Peaks in the periodogram indicate strong periodicities.

### c. **X-13ARIMA-SEATS**

(If available): official seasonal adjustment used by statistical agencies.

---

## 4. **Volatility and Clustering**

### 🔹 Volatility (returns-based):

$$
r_t = \frac{X_t}{X_{t-1}} - 1 \quad \Rightarrow \quad \text{Volatility} = \sigma_r
$$

### 🔹 Rolling Std:

$$
\text{RollingVol}_t = \sqrt{ \frac{1}{w} \sum_{i=t-w+1}^{t} (r_i - \bar{r})^2 }
$$

### 🔹 ARCH Test:

Detects **volatility clustering**:

* Null $H_0$: no ARCH effects (homoscedasticity)
* Based on:

$$
r_t^2 = \alpha_0 + \sum_{i=1}^{p} \alpha_i r_{t-i}^2 + \varepsilon_t
$$

---

## 5. **Change Point Detection**

### 🔹 CUSUM:

Detects shifts in the **mean**:

$$
S_t = \sum_{i=1}^{t} (X_i - \bar{X})
$$

* Significant break if:

$$
\max_t |S_t| > \lambda
$$

### 🔹 Page-Hinkley:

Online change detection:

$$
PH_t = (x_t - \bar{x}_t - \delta) + PH_{t-1}
$$

### 🔹 Binary Segmentation:

Splits time series recursively at the point with the **largest mean difference** using a $t$-test.

---

## 6. **Regime Switching**

### a. **Hidden Markov Models (HMM)**

Assume time series switches between hidden states $z_t \in \{1, 2, ..., K\}$, each modeled as a Gaussian:

$$
X_t \sim \mathcal{N}(\mu_{z_t}, \sigma_{z_t}^2)
$$

* Estimated via EM algorithm
* Output:

  * Transition matrix $A$
  * State probabilities
  * Most likely state sequence

### b. **Threshold Autoregressive (TAR)**

* Splits behavior based on threshold $\theta$:

$$
X_t = \begin{cases}
\phi_1 X_{t-1} + \varepsilon_t, & \text{if } X_{t-d} \le \theta \\
\phi_2 X_{t-1} + \varepsilon_t, & \text{otherwise}
\end{cases}
$$

---

## 7. **Lag Feature Suggestion**

Uses:

* **ACF / PACF** peaks:

  $$
  \rho_k = \frac{\text{Cov}(X_t, X_{t-k})}{\text{Var}(X_t)}
  $$
* **AutoRegressive model selection** (AIC-based)
* **Cross-correlation function (CCF)** for multivariate lags

Output includes:

* Recommended lags
* AR order
* Seasonal lags (e.g., 12, 24, 52)
* Cross-lag suggestions

---

## 8. **Forecast Readiness Score**

Weighted score of:

| Metric             | Weight | Interpretation                           |
| ------------------ | ------ | ---------------------------------------- |
| Data Sufficiency   | 0.20   | ≥ 100 pts preferred                      |
| Missing Values     | 0.15   | Penalizes incomplete features            |
| Stationarity       | 0.15   | Higher if ADF p < 0.05                   |
| Trend Strength     | 0.10   | Moderate trend is helpful                |
| Seasonality        | 0.10   | Strong periodicity is useful             |
| Signal-to-Noise    | 0.15   | Lower noise yields better predictability |
| Autocorrelation    | 0.10   | More significant lags is better          |
| Outlier Robustness | 0.05   | Fewer outliers = higher score            |

### 🔹 Final Readiness Classification:

* **Excellent**: score ≥ 0.80
* **Good**: 0.60 ≤ score < 0.80
* **Fair**: 0.40 ≤ score < 0.60
* **Poor**: score < 0.40

---

## 🧪 Output

The module returns structured dictionaries with:

* DataFrames (e.g., stationarity tests)
* Dictionaries of temporal pattern metadata
* Forecasting recommendations

---

Let's now proceed with a rigorous breakdown of the **Dimensionality Reduction** module — an essential topic in data science, especially for **visualization**, **noise reduction**, and **feature compression**.

---

# 📉 Dimensionality Reduction

Dimensionality Reduction (DR) transforms a high-dimensional dataset $X \in \mathbb{R}^{n \times d}$ into a lower-dimensional representation $Z \in \mathbb{R}^{n \times k}$, where $k \ll d$, while preserving as much **informative structure** as possible.

Your implementation includes the following techniques:

* Principal Component Analysis (PCA)
* Independent Component Analysis (ICA)
* t-SNE
* UMAP (optional)

Let’s go through them with mathematical depth.

---

## 1. **Principal Component Analysis (PCA)**

### ✅ Goal:

Find orthogonal directions $\mathbf{w}_1, \dots, \mathbf{w}_k \in \mathbb{R}^d$ that **maximize variance**:

$$
\mathbf{w}_1 = \underset{\|\mathbf{w}\| = 1}{\text{argmax}} \ \text{Var}(X \mathbf{w})
$$

### 🔹 Procedure:

1. **Center** the data:

$$
\tilde{X} = X - \bar{X}
$$

2. **Compute Covariance Matrix**:

$$
\Sigma = \frac{1}{n-1} \tilde{X}^\top \tilde{X}
$$

3. **Eigen-decomposition**:

$$
\Sigma \mathbf{w}_i = \lambda_i \mathbf{w}_i
$$

* $\lambda_i$: variance along direction $\mathbf{w}_i$

4. **Project**:

$$
Z = \tilde{X} \cdot W_k
$$

### 🔸 Properties:

* Linear method
* Produces **orthogonal axes**
* Captures **global structure**

---

## 2. **Independent Component Analysis (ICA)**

### ✅ Goal:

Separate **independent** sources from observed mixtures, assuming:

$$
X = A S
$$

* $X$: observed mixed signals
* $A$: unknown mixing matrix
* $S$: latent **statistically independent** components

### 🔹 Objective:

Find unmixing matrix $W$ such that:

$$
S = W X
$$

and components of $S$ are **maximally non-Gaussian** (by Central Limit Theorem, mixture of sources tends to Gaussian).

### 🔸 Techniques:

* **Kurtosis maximization**
* **Negentropy**:

$$
J(S) = H(S_{\text{gauss}}) - H(S)
$$

---

## 3. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**

### ✅ Goal:

Preserve **local structure** in a nonlinear embedding.

### 🔹 Step-by-step:

#### 1. Compute pairwise similarities in high-dimensional space:

$$
P_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \ne i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

Define symmetric probability:

$$
P_{ij} = \frac{P_{j|i} + P_{i|j}}{2n}
$$

#### 2. Define low-dimensional similarity using **Student’s t-distribution**:

$$
Q_{ij} = \frac{(1 + \|z_i - z_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|z_k - z_l\|^2)^{-1}}
$$

#### 3. Minimize **Kullback-Leibler divergence**:

$$
\mathcal{L} = \sum_{i \ne j} P_{ij} \log \left( \frac{P_{ij}}{Q_{ij}} \right)
$$

### 🔸 Characteristics:

* **Nonlinear** projection
* Good for **visualizing clusters**
* Sensitive to **perplexity** and local scale

---

## 4. **UMAP (Uniform Manifold Approximation and Projection)**

> Used if `umap-learn` is installed

### ✅ Goal:

Preserve both **local** and **global** structure of data via **manifold learning**.

### 🔹 Mathematical Idea:

* Assume data lies on a **Riemannian manifold**.
* Construct a fuzzy topological graph $G$ in high-dim space.
* Construct an equivalent low-dimensional graph $G'$ and minimize cross-entropy between them.

### 🔸 Optimization:

$$
\mathcal{L}_{\text{UMAP}} = \sum_{(i,j)} -[w_{ij} \log(\hat{w}_{ij}) + (1 - w_{ij}) \log(1 - \hat{w}_{ij})]
$$

Where:

* $w_{ij}$: edge weights in high-dimensional graph
* $\hat{w}_{ij}$: edge weights in low-dimensional embedding

---

## 5. **Dimensionality Reduction Pipeline**

Before applying these methods:

* Data is optionally **sampled** for efficiency (if large)
* Data is **standardized** using:

  $$
  X_{scaled} = \frac{X - \mu}{\sigma}
  $$

Each method is run independently and returns:

* Low-dimensional embedding matrix $Z \in \mathbb{R}^{n \times k}$
* Typically, $k = 2$ for visualization

---

## 📘 Method Summary

| Method | Type      | Captures                 | Pros                          | Cons                                       |
| ------ | --------- | ------------------------ | ----------------------------- | ------------------------------------------ |
| PCA    | Linear    | Global variance          | Fast, interpretable           | Misses nonlinear structures                |
| ICA    | Linear    | Statistical independence | Useful for source separation  | Sensitive to noise                         |
| t-SNE  | Nonlinear | Local neighborhoods      | Excellent visualization       | Non-parametric, slow, perplexity-sensitive |
| UMAP   | Nonlinear | Local + global           | Scalable, preserves structure | Sensitive to parameters                    |

---

Great — let's now cover the **Feature Pattern Detection** stage with mathematical and pedagogical detail. This module classifies features based on **distributional**, **structural**, and **statistical** patterns using advanced techniques.

---

# 🧠 Feature Pattern Detection — Mathematical and Statistical Foundations

This module detects **statistical feature types**, **inter-feature relationships**, and **distributional structures**. It is designed to support automated feature engineering and preprocessing decisions.

---

## 1. **Feature Type Classification**

Each numeric feature is classified based on statistical shape, modality, and normality.

### ✅ Key Metrics:

Let $X = \{x_1, \dots, x_n\} \in \mathbb{R}$

#### a. **Skewness**:

$$
\text{Skew}(X) = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3
$$

#### b. **Kurtosis**:

$$
\text{Kurtosis}(X) = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4
$$

#### c. **Normality Tests**:

* **Anderson-Darling**, **Shapiro-Wilk**, **Jarque-Bera**:
  Each has null hypothesis $H_0$: data is drawn from a normal distribution.

#### d. **Multimodality Detection**:

Fit **Gaussian Mixture Models** (GMMs):

$$
p(x) = \sum_{j=1}^{K} \pi_j \cdot \mathcal{N}(x \mid \mu_j, \sigma_j^2)
$$

Evaluate the **Bayesian Information Criterion (BIC)** to determine the optimal number of components $K$.

---

### 📌 Classification Heuristics:

| Metric Condition        | Inferred Pattern       |
| ----------------------- | ---------------------- |
| Low skew, kurt ≈ 3      | Symmetric, normal-like |
| Skew > 1                | Long tail (right)      |
| Skew < -1               | Long tail (left)       |
| Kurt > 5                | Heavy tails            |
| Multiple GMM components | Multimodal             |
| Shapiro p < 0.05        | Non-normal             |

---

## 2. **Inter-Feature Relationship Detection**

For all feature pairs $(X_i, X_j)$:

### a. **Linear and Nonlinear Dependency**

Use:

* **Pearson Correlation** (linear):

  $$
  \rho_{ij} = \frac{\operatorname{Cov}(X_i, X_j)}{\sigma_i \sigma_j}
  $$
* **Spearman Correlation** (monotonic)
* **Mutual Information** (nonlinear):

  $$
  I(X_i; X_j) = \sum_{x, y} p(x, y) \log \left( \frac{p(x, y)}{p(x)p(y)} \right)
  $$

### b. **Thresholding & Filtering**:

Only retain "strong" relationships:

* $|\rho| > 0.7$
* Mutual Information $> \tau$

---

## 3. **Distributional Fitting**

Fit multiple distributions to each numeric feature:

### Supported Distributions:

* Normal $\mathcal{N}(\mu, \sigma^2)$
* Lognormal
* Exponential
* Gamma
* Beta

### Fit Method:

* Maximum Likelihood Estimation (MLE)
* Compute **AIC (Akaike Information Criterion)**:

  $$
  \text{AIC} = 2k - 2 \log L
  $$

  where:

  * $k$: number of parameters
  * $L$: maximum likelihood

Choose the distribution with the **lowest AIC**.

---

## 4. **Anomaly Pattern Detection**

Detect:

* **Extreme skew**
* **Zero-inflation**
* **Plateaus or constant segments**
* **Power-law tails** (via log-log plots or fitted Pareto)

## Summary

| Pattern Detected      | Use Case                           |
| --------------------- | ---------------------------------- |
| Non-normality         | Apply transformation (e.g., log)   |
| Heavy tail            | Use robust statistics              |
| Multimodality         | Consider mixture models or binning |
| Strong correlation    | Feature selection or interaction   |
| Best-fit distribution | Synthetic data generation, priors  |