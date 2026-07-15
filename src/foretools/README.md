# Foretools - Time Series Analysis, Feature Engineering, and Machine Learning Utilities

Foretools is a comprehensive collection of utility tools and libraries for time series analysis, machine learning, feature engineering, and data processing. It includes signal decomposition methods, feature engineering pipelines, hyperparameter optimization, time series augmentation, and benchmarking frameworks.

## Overview

Foretools provides:

- **Feature Engineering (`fengineer`)**: Comprehensive feature transformation and selection pipeline with statistical, mathematical, categorical, and clustering-based transformers
- **Time Series Analysis (`foreminer`, `arima`)**: Time series mining, ARIMA utilities, and statistical analysis tools
- **Signal Decomposition (`emd_like`, `ewt`)**: Empirical Mode Decomposition (EMD) and Empirical Wavelet Transform (EWT) utilities
- **Hyperparameter Optimization (`bohb`)**: BOHB (Bayesian Optimization with HyperBand) implementation for efficient hyperparameter search
- **Time Series Augmentation & Generation (`tsaug`, `tsgen`)**: Data augmentation and synthetic time series generation utilities
- **Auxiliary Utilities (`aux`)**: Adaptive mutual information, distance correlation, HSIC, and feature binning utilities

## Directory Structure

```
foretools/
├── fengineer/                    # Feature engineering pipeline
│   ├── fengineer.py             # Main feature engineering orchestrator
│   ├── __init__.py
│   ├── transformers/            # Feature transformation modules
│   │   ├── __init__.py
│   │   ├── datetime.py          # DateTime feature transformations
│   │   ├── mathematical.py      # Mathematical transformations (log, sqrt, power, etc.)
│   │   ├── statistical.py       # Statistical transformations (z-score, min-max, etc.)
│   │   ├── categorical.py       # Categorical encoding (one-hot, target, etc.)
│   │   ├── interaction.py       # Feature interaction terms
│   │   ├── polynomial.py        # Polynomial feature expansion
│   │   ├── fourier.py           # Fourier series transformations
│   │   ├── rff.py               # Random Fourier Features
│   │   ├── clustering.py        # Clustering-based feature transformations
│   │   ├── autoencoder.py       # Autoencoder-based feature extraction
│   │   ├── binning.py           # Feature binning strategies
│   │   ├── woe.py               # Weight of Evidence (WOE) transformation
│   │   ├── mdlp.py              # Minimum Description Length Principle binning
│   │   └── aux/                 # Auxiliary transformer utilities
│   │       ├── base.py          # Base transformer classes
│   │       ├── config.py        # Transformer configuration
│   │       ├── binning_strategies.py # Binning strategy implementations
│   │       ├── stats_safe.py    # Safe statistical computations
│   │       └── __init__.py
│   ├── selectors/               # Feature selection modules
│   │   ├── __init__.py
│   │   ├── base.py              # Base feature selector classes
│   │   ├── feature_selector.py  # General feature selector interface
│   │   ├── redundancy.py        # Redundancy-based feature selection
│   │   ├── mi_selector.py       # Mutual Information-based selector
│   │   ├── mrmr_selector.py     # MRMR (Max-Relevance Min-Redundancy) selector
│   │   ├── boruta.py            # Boruta feature selection algorithm
│   │   └── rfecv.py             # Recursive Feature Elimination with CV
│   └── filters/                 # Feature filtering modules
│       ├── __init__.py
│       └── correlation.py       # Correlation-based feature filtering
│
├── foreminer/                    # Time series mining and analysis
│   ├── foreminer.py             # Main foreminer orchestrator
│   ├── core.py                  # Core foreminer functionality
│   ├── report.py                # Reporting utilities
│   └── analyzers/               # Data analyzers
│       ├── __init__.py
│       ├── analyzer_utils.py    # Analyzer utility functions
│       ├── cluster.py           # Clustering analysis
│       └── correl.py            # Correlation analysis
│
├── aux/                          # Auxiliary statistical utilities
│   ├── adaptive_mi.py           # Adaptive Mutual Information computation
│   ├── adaptive_mrmr.py         # Adaptive MRMR feature selection
│   ├── bb_bins.py               # Binning utilities
│   ├── distance_correlation.py  # Distance correlation computation
│   └── hsic.py                  # HSIC (Hilbert-Schmidt Independence Criterion)
│
├── bohb/                         # BOHB (Bayesian Optimization with HyperBand)
│   # BOHB implementation for efficient hyperparameter optimization
│   # Combines Bayesian optimization with HyperBand early stopping
│
├── emd_like/                     # Empirical Mode Decomposition and related methods
│   # EMD, EEMD, CEEMDAN and related signal decomposition utilities
│
├── ewt/                          # Empirical Wavelet Transform
│   # EWT utilities for signal decomposition and frequency analysis
│
├── arima/                        # ARIMA model utilities
│   # ARIMA (AutoRegressive Integrated Moving Average) model utilities
│
├── tsaug/                        # Time Series Augmentation
│   # Data augmentation techniques for time series:
│   # - Noise injection
│   # - Time warping
│   # - Magnitude warping
│   # - Permutation
│   # - Masking
│
├── tsgen/                        # Time Series Generation
│   # Synthetic time series generation utilities
│   # - Statistical generators
│   # - Model-based generators
│   # - GAN-based generators
│
├── benchmarking/                 # Benchmarking frameworks
│   # Tools for benchmarking models and algorithms
│
├── conversor.ipynb               # Conversion utilities notebook
├── __init__.py                   # Package initialization
└── README.md                     # This file

## Core API

### Feature Engineering

```python
from foretools.fengineer import FeatureEngineer
from foretools.fengineer.transformers import (
    # Statistical transformers
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    # Mathematical transformers
    LogTransformer,
    SqrtTransformer,
    PowerTransformer,
    # Categorical transformers
    OneHotEncoder,
    TargetEncoder,
    # Advanced transformers
    PolynomialFeatures,
    FourierFeatures,
    RandomFourierFeatures,
    AutoencoderTransformer,
    # Binning
    MDLPBinner,
    WOETransformer,
)
from foretools.fengineer.selectors import (
    FeatureSelector,
    MutualInformationSelector,
    MRMRSelector,
    BorutaSelector,
    RFECVSelector,
)
```

### Auxiliary Statistical Utilities

```python
from foretools.aux import (
    adaptive_mutual_information,
    adaptive_mrmr,
    distance_correlation,
    hsic,
    binning_utilities,
)
```

### Time Series Augmentation & Generation

```python
from foretools.tsaug import (
    # Augmentation operations
    NoiseInjection,
    TimeWarping,
    MagnitudeWarping,
    Permutation,
    Masking,
)
from foretools.tsgen import (
    # Generation utilities
    StatisticalGenerator,
    ModelBasedGenerator,
)
```

### Signal Decomposition

```python
# EMD and related methods
from foretools.emd_like import (
    emd_decompose,
    eemd_decompose,
    ceemdan_decompose,
)

# EWT
from foretools.ewt import (
    ewt_decompose,
    ewt_reconstruct,
)
```

### Hyperparameter Optimization

```python
from foretools.bohb import (
    BOHBOptimizer,
    # BOHB configuration and utilities
)
```

## Key Features

1. **Comprehensive Feature Engineering**: 15+ transformer types including statistical, mathematical, categorical, polynomial, Fourier, autoencoder, and clustering-based transformations
2. **Advanced Feature Selection**: MI-based, MRMR, Boruta, RFECV, and redundancy-based feature selection methods
3. **Signal Decomposition**: EMD, EEMD, CEEMDAN, and EWT for time series decomposition and frequency analysis
4. **Time Series Augmentation**: Multiple augmentation techniques for improving model robustness and preventing overfitting
5. **BOHB Optimization**: Efficient hyperparameter optimization combining Bayesian optimization with HyperBand early stopping
6. **Statistical Utilities**: Adaptive mutual information, distance correlation, HSIC for feature analysis and selection

## Dependencies

- NumPy, Pandas, SciPy
- Scikit-learn (for feature engineering and selection)
- PyTorch (for autoencoder transformers)
- Wavelet libraries (for EWT)
- BOHB or Optuna (for hyperparameter optimization)
