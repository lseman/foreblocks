# Foretools - Time Series Analysis, Feature Engineering, and Machine Learning Utilities

Foretools is a comprehensive collection of utility tools and libraries for time series analysis, machine learning, feature engineering, and data processing. It includes signal decomposition methods, feature engineering pipelines, hyperparameter optimization, time series augmentation, and benchmarking frameworks.

## Overview

Foretools provides:

- **Feature Engineering (`fengineer`)**: Comprehensive feature transformation and selection pipeline with statistical, mathematical, categorical, and clustering-based transformers
- **Time Series Analysis (`foreminer`, `arima`)**: Time series mining, ARIMA utilities, and statistical analysis tools
- **Signal Decomposition (`decomposition`)**: Empirical Mode Decomposition (EMD family, incl. VMD) and Empirical Wavelet Transform (EWT)
- **Hyperparameter Optimization (`bohb`)**: BOHB (Bayesian Optimization with HyperBand) implementation for efficient hyperparameter search
- **Time Series Augmentation & Generation (`tsaug`, `tsgen`)**: Data augmentation and synthetic time series generation utilities
- **Statistical Utilities (`stats`)**: Adaptive mutual information, distance correlation, HSIC, and Bayesian Blocks binning

## Directory Structure

```
foretools/
├── fengineer/                    # Feature engineering pipeline
│   ├── fengineer.py              # FeatureEngineer orchestrator
│   ├── transformers/              # Feature transformation modules
│   │   ├── datetime.py           # DateTime feature transformations
│   │   ├── mathematical.py       # Mathematical transformations (log, sqrt, power, etc.)
│   │   ├── statistical.py        # Statistical transformations (z-score, min-max, etc.)
│   │   ├── categorical.py        # Categorical encoding (one-hot, target, etc.)
│   │   ├── interaction.py        # Feature interaction terms
│   │   ├── polynomial.py         # Polynomial feature expansion
│   │   ├── fourier.py            # Fourier series transformations
│   │   ├── rff.py                # Random Fourier Features
│   │   ├── clustering.py         # Clustering-based feature transformations
│   │   ├── autoencoder.py        # Autoencoder-based feature extraction
│   │   ├── binning.py            # Feature binning strategies
│   │   ├── woe.py                # Weight of Evidence (WOE) transformation
│   │   ├── mdlp.py               # Minimum Description Length Principle binning
│   │   └── support/              # Shared transformer infrastructure
│   │       ├── base.py           # ABC base class, shared utilities, decorators
│   │       ├── config.py         # FeatureConfig with nested sub-configs
│   │       ├── binning_strategies.py # Standalone binning strategy functions
│   │       └── stats_safe.py     # Safe statistical functions (skew, kurtosis)
│   ├── selectors/                # Feature selection modules
│   │   ├── base.py               # Base feature selector classes
│   │   ├── feature_selector.py   # PipelineSelector (alias: FeatureSelector)
│   │   ├── redundancy.py         # Redundancy-based feature selection
│   │   ├── mi_selector.py        # Mutual Information-based selector
│   │   ├── mrmr_selector.py      # MRMR (Max-Relevance Min-Redundancy) selector
│   │   ├── adaptive_mrmr.py      # AdaptiveMRMR — fengineer-internal MRMR implementation
│   │   ├── boruta.py             # Boruta feature selection algorithm
│   │   └── rfecv.py              # Recursive Feature Elimination with CV
│   └── filters/                  # Feature filtering modules
│       └── correlation.py        # Correlation-based feature filtering
│
├── foreminer/                    # Time series mining and analysis
│   ├── foreminer.py              # DatasetAnalyzer orchestrator
│   ├── core.py                   # Core foreminer functionality
│   ├── report.py                 # Reporting utilities
│   ├── plotting.py               # Plotting utilities
│   └── analyzers/                # Registered analysis strategies
│       ├── cluster.py            # Clustering analysis
│       ├── correlation.py        # Correlation analysis
│       ├── dimension.py          # Dimensionality reduction
│       ├── distribution.py       # Distribution diagnostics
│       ├── features.py           # Feature-engineering analysis (leak-aware suggestions)
│       ├── graph.py              # Correlation-network / graph analysis
│       ├── group.py              # Cohort-level summaries
│       ├── missing.py            # Missingness diagnostics
│       ├── outlier.py            # Outlier diagnostics
│       ├── pattern.py            # Pattern/dependency diagnostics
│       ├── timeseries.py         # Seasonality, trend, temporal-structure diagnostics
│       └── analyzer_utils.py     # Shared analyzer helpers
│
├── stats/                        # Standalone statistical utilities
│   ├── adaptive_mi.py            # Adaptive Mutual Information (shared: fengineer + foreminer)
│   ├── distance_correlation.py   # Distance correlation computation
│   ├── hsic.py                   # HSIC (Hilbert-Schmidt Independence Criterion)
│   └── bb_bins.py                # Bayesian Blocks binning
│
├── decomposition/                # Time-series signal decomposition
│   ├── emd/                      # EMD, EEMD, CEEMDAN, VMD, hierarchical VMD
│   │   ├── common.py             # FFTWManager, BoundaryHandler, SignalAnalyzer, ModeProcessor
│   │   ├── config.py             # VMDOptions, VMDParameters, HierarchicalParameters
│   │   ├── core.py               # VMDCore, CrossModeRefiner, InformerRefiner
│   │   ├── emd.py                # EMDVariants
│   │   ├── pipeline.py           # FastVMD, HierarchicalVMD, VMDOptimizer
│   │   ├── variants.py           # VariationalVariants
│   │   ├── analysis/             # fractal.py, mode_processor.py, signal_analysis.py
│   │   └── support/              # boundary.py, fft.py, utils.py
│   └── ewt/                      # Empirical Wavelet Transform
│       └── ewt_core.py           # EWT1D, EWT_Boundaries_Detect, EWT_Meyer_FilterBank
│
├── bohb/                         # BOHB (Bayesian Optimization with HyperBand)
│   ├── bohb.py                   # BOHB — main optimizer orchestrator
│   ├── hyperband.py              # HyperbandScheduler
│   ├── tpe.py                    # TPEConf, TPEConfig
│   ├── pruning.py                # PruningConfig
│   ├── trial.py                  # Trial, TrialPruned
│   ├── plotter.py                # OptimizationPlotter
│   ├── objectives.py             # Example objectives
│   ├── param_models.py           # Parameter-space models
│   ├── acquisition/               # Acquisition function strategies + factory
│   ├── batch/                    # Batch acquisition strategies (qNEI, Thompson, …)
│   ├── gamma/                    # Gamma (quantile-split) strategies
│   ├── observation/               # Observation store
│   ├── surrogates/                # GPSurrogate, GPEnsemble
│   └── utils/                     # Config and numerics helpers
│
├── arima/                        # ARIMA model utilities
│   └── arima.py
│
├── tsaug/                        # Time Series Augmentation (AutoDA-Timeseries)
│   ├── model.py                  # AutoDATimeseries, AutoDATrainer
│   ├── features.py               # extract_features
│   ├── layers.py                 # Augmentation policy layers
│   ├── losses.py                 # CompositeLoss
│   └── transformations.py        # TRANSFORMATIONS, TRANSFORM_NAMES
│
├── tsgen/                        # Time Series Generation
│   └── ts_gen.py                 # TimeSeriesGenerator
│
├── benchmarking/                 # Benchmarking frameworks
│   └── benchmark.py
│
├── __init__.py                   # Package initialization (lazy public exports)
└── README.md                     # This file
```

## Core API

### Feature Engineering

```python
from foretools.fengineer import FeatureEngineer
from foretools.fengineer.transformers import FeatureConfig
from foretools.fengineer.selectors.feature_selector import FeatureSelector
from foretools.fengineer.filters import CorrelationFilter
```

### Statistical Utilities

```python
from foretools import AdaptiveMI, AdaptiveMRMR, DistanceCorrelation, HSIC
from foretools.stats.bb_bins import BayesianBlocks
```

### Time Series Augmentation & Generation

```python
from foretools.tsaug import AutoDATimeseries, AutoDATrainer, extract_features
from foretools.tsgen import TimeSeriesGenerator
```

### Signal Decomposition

```python
# EMD family + VMD
from foretools.decomposition.emd import FastVMD, VMDOptimizer, FFTWManager

# EWT
from foretools.decomposition.ewt import EWT1D
```

### Hyperparameter Optimization

```python
from foretools.bohb import BOHB, PruningConfig, TPEConf
from foretools.bohb.plotter import OptimizationPlotter
from foretools.bohb.trial import TrialPruned
```

### Exploratory Analysis

```python
from foretools.foreminer.foreminer import DatasetAnalyzer
```

## Key Features

1. **Comprehensive Feature Engineering**: 15+ transformer types including statistical, mathematical, categorical, polynomial, Fourier, autoencoder, and clustering-based transformations
2. **Advanced Feature Selection**: MI-based, MRMR, Boruta, RFECV, and redundancy-based feature selection methods
3. **Signal Decomposition**: EMD, EEMD, CEEMDAN, VMD, and EWT for time series decomposition and frequency analysis
4. **Time Series Augmentation**: Multiple augmentation techniques for improving model robustness and preventing overfitting
5. **BOHB Optimization**: Efficient hyperparameter optimization combining Bayesian optimization with HyperBand early stopping
6. **Statistical Utilities**: Adaptive mutual information, distance correlation, HSIC for feature analysis and selection

## Dependencies

- NumPy, Pandas, SciPy
- Scikit-learn (for feature engineering and selection)
- Numba (for `stats` accelerators)
- PyTorch (for autoencoder and `tsaug` transformers)
- Optuna (for BOHB and VMD auto-parameter search)
