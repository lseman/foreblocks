import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class BorutaSelector(BaseEstimator, TransformerMixin):
    """
    Boruta Feature Selection.
    Uses 'shadow features' (shuffled versions of original features) to 
    determine statistically significant feature importance.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        max_iter: int = 20,
        perc: float = 100,
        alpha: float = 0.05,
        random_state: int = 42,
        verbose: int = 0
    ):
        self.estimator = estimator
        self.max_iter = max_iter
        self.perc = perc
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose
        
        self.support_ = None
        self.ranking_ = None
        self._feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaSelector":
        self._feature_names = X.columns.tolist()
        n_feat = X.shape[1]
        
        # Initialize internal structures
        self.support_ = np.zeros(n_feat, dtype=bool)
        hit_counts = np.zeros(n_feat, dtype=int)
        
        # Default estimator if none provided
        if self.estimator is None:
            # Heuristic for task type
            if y.nunique() < 20:
                self.estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.random_state)
            else:
                self.estimator = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=self.random_state)

        X_array = X.values
        y_array = y.values
        
        rng = np.random.RandomState(self.random_state)
        
        for i in range(self.max_iter):
            # 1. Create shadow features by shuffling
            X_shadow = np.apply_along_axis(rng.permutation, axis=0, arr=X_array)
            X_combined = np.hstack([X_array, X_shadow])
            
            # 2. Fit estimator and get importance
            self.estimator.fit(X_combined, y_array)
            importances = self.estimator.feature_importances_
            
            # 3. Compare real importance to shadow importance
            real_imp = importances[:n_feat]
            shadow_imp = importances[n_feat:]
            
            # Shadow max importance
            m_shadow = np.percentile(shadow_imp, self.perc)
            
            # Count hits (real > shadow_max)
            hits = real_imp > m_shadow
            hit_counts += hits.astype(int)
            
            if self.verbose:
                print(f"Boruta Iter {i+1}: Hits {np.sum(hits)}")

        # 4. Statistical test (Binomial distribution)
        # For simplicity, we use a simple threshold here: more than 50% hits
        # In full Boruta, we'd use p-values from binomial distribution.
        self.support_ = hit_counts >= (self.max_iter // 2)
        
        self.ranking_ = np.zeros(n_feat, dtype=int)
        # Features with more hits get better ranking (lower number)
        self.ranking_ = n_feat - hit_counts + 1
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("BorutaSelector not fitted.")
        return X.iloc[:, self.support_]

    def get_selected_features(self) -> List[str]:
        if not self._feature_names: return []
        return [self._feature_names[i] for i, s in enumerate(self.support_) if s]
