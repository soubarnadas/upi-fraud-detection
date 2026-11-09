"""
LightGBM and XGBoost model wrappers with high-performance pure-NumPy GBDT engine.
Provides fast training, gradient boosting, and path-based feature attributions.
"""

from typing import Optional, List
import numpy as np
from scipy.special import expit
from src.models.base import BaseFraudModel


class PureNumPyDecisionTree:
    """Fast binary regression tree for gradient boosting residuals."""
    def __init__(self, max_depth: int = 4, min_samples_split: int = 8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_idx = None
        self.threshold = None
        self.value = 0.0
        self.left = None
        self.right = None

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int = 0):
        n_samples, n_features = X.shape
        sum_g = np.sum(g)
        sum_h = np.sum(h)
        self.value = -sum_g / (sum_h + 1e-4)

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        # Feature subsampling
        n_sub = max(2, int(np.sqrt(n_features)))
        feature_sub = np.random.choice(n_features, size=n_sub, replace=False)

        for f in feature_sub:
            x_col = X[:, f]
            uniques = np.unique(x_col)
            if len(uniques) <= 1:
                continue
            
            # Percentiles for candidate split thresholds
            if len(uniques) < 10:
                thresholds = uniques[:-1]
            else:
                thresholds = np.percentile(x_col, [15, 30, 50, 70, 85])
                
            for thresh in thresholds:
                left_mask = x_col <= thresh
                right_mask = ~left_mask

                if not (np.any(left_mask) and np.any(right_mask)):
                    continue

                g_L, h_L = np.sum(g[left_mask]), np.sum(h[left_mask])
                g_R, h_R = np.sum(g[right_mask]), np.sum(h[right_mask])

                gain = 0.5 * (
                    (g_L ** 2) / (h_L + 1e-4) +
                    (g_R ** 2) / (h_R + 1e-4) -
                    (sum_g ** 2) / (sum_h + 1e-4)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh

        if best_feat is not None and best_gain > 1e-5:
            self.feature_idx = best_feat
            self.threshold = best_thresh
            left_mask = X[:, best_feat] <= best_thresh
            self.left = PureNumPyDecisionTree(self.max_depth, self.min_samples_split)
            self.right = PureNumPyDecisionTree(self.max_depth, self.min_samples_split)
            self.left.fit(X[left_mask], g[left_mask], h[left_mask], depth + 1)
            self.right.fit(X[~left_mask], g[~left_mask], h[~left_mask], depth + 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature_idx is None:
            return np.full(len(X), self.value)
        left_mask = X[:, self.feature_idx] <= self.threshold
        out = np.empty(len(X), dtype=float)
        if np.any(left_mask):
            out[left_mask] = self.left.predict(X[left_mask])
        if np.any(~left_mask):
            out[~left_mask] = self.right.predict(X[~left_mask])
        return out


class BuiltinGBDT:
    """Ensemble of gradient boosted trees optimizing binary logistic loss with stratified boosting."""
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        scale_pos_weight: float = 1.0,
        subsample: float = 0.8
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.scale_pos_weight = scale_pos_weight
        self.subsample = subsample
        self.trees: List[PureNumPyDecisionTree] = []
        self.base_score = 0.0
        self.importances = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        n_samples, n_features = X.shape
        pos_ratio = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_score = np.log(pos_ratio / (1.0 - pos_ratio))
        
        raw_preds = np.full(n_samples, self.base_score, dtype=float)
        self.trees = []
        self.importances = np.zeros(n_features)

        w = np.ones(n_samples, dtype=float)
        if sample_weight is not None:
            w = sample_weight.copy()
        elif self.scale_pos_weight != 1.0:
            w[y == 1] = self.scale_pos_weight

        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        for _ in range(self.n_estimators):
            p = expit(raw_preds)
            g = w * (p - y)
            h = w * p * (1.0 - p) + 1e-6

            # Stratified subsampling to ensure positive examples are always present
            n_sub_neg = int(len(neg_indices) * self.subsample)
            sub_neg = np.random.choice(neg_indices, size=n_sub_neg, replace=False)
            sub_idx = np.concatenate([pos_indices, sub_neg])
            np.random.shuffle(sub_idx)

            tree = PureNumPyDecisionTree(max_depth=self.max_depth)
            tree.fit(X[sub_idx], g[sub_idx], h[sub_idx])
            
            update = tree.predict(X)
            raw_preds += self.learning_rate * update
            self.trees.append(tree)

            def accumulate_imp(node):
                if node.feature_idx is not None:
                    self.importances[node.feature_idx] += 1
                    if node.left: accumulate_imp(node.left)
                    if node.right: accumulate_imp(node.right)
            accumulate_imp(tree)

        imp_sum = np.sum(self.importances)
        if imp_sum > 0:
            self.importances /= imp_sum

        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        raw = np.full(len(X), self.base_score, dtype=float)
        for tree in self.trees:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return expit(self.predict_raw(X))


class LightGBMModel(BaseFraudModel):
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        scale_pos_weight: float = 1.0
    ):
        super().__init__("LightGBM")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.scale_pos_weight = scale_pos_weight
        self._native_model = None
        self._use_native = False

        try:
            import lightgbm as lgb
            self._use_native = True
        except ImportError:
            self._use_native = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "LightGBMModel":
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        if self._use_native:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                scale_pos_weight=self.scale_pos_weight,
                random_state=42,
                verbosity=-1
            )
            model.fit(X, y, sample_weight=sample_weight)
            self._native_model = model
        else:
            self._native_model = BuiltinGBDT(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                scale_pos_weight=self.scale_pos_weight
            )
            self._native_model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._use_native:
            return self._native_model.predict_proba(X)[:, 1]
        return self._native_model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        if self._use_native:
            return self._native_model.feature_importances_
        return self._native_model.importances


class XGBoostModel(BaseFraudModel):
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        scale_pos_weight: float = 1.0
    ):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.scale_pos_weight = scale_pos_weight
        self._native_model = None
        self._use_native = False

        try:
            import xgboost as xgb
            self._use_native = True
        except ImportError:
            self._use_native = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "XGBoostModel":
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        if self._use_native:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                scale_pos_weight=self.scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X, y, sample_weight=sample_weight)
            self._native_model = model
        else:
            self._native_model = BuiltinGBDT(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                scale_pos_weight=self.scale_pos_weight
            )
            self._native_model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._use_native:
            return self._native_model.predict_proba(X)[:, 1]
        return self._native_model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        if self._use_native:
            return self._native_model.feature_importances_
        return self._native_model.importances
