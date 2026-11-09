"""
Logistic Regression model wrapper with Scikit-Learn and pure SciPy/NumPy fallback.
"""

from typing import Optional, List
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from src.models.base import BaseFraudModel


class FraudLogisticRegression(BaseFraudModel):
    def __init__(self, C: float = 1.0, scale_pos_weight: float = 1.0, max_iter: int = 200):
        super().__init__("LogisticRegression")
        self.C = C
        self.scale_pos_weight = scale_pos_weight
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0.0
        self.feature_means = None
        self.feature_stds = None
        self._use_sklearn = False
        self._sklearn_model = None

        try:
            import sklearn.linear_model
            self._use_sklearn = True
        except ImportError:
            self._use_sklearn = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "FraudLogisticRegression":
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        
        # Standardize features for numerical stability
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.feature_stds[self.feature_stds < 1e-6] = 1.0
        X_scaled = (X - self.feature_means) / self.feature_stds
        
        weights_vec = np.ones(len(y), dtype=float)
        if sample_weight is not None:
            weights_vec = sample_weight.copy()
        elif self.scale_pos_weight != 1.0:
            weights_vec[y == 1] = self.scale_pos_weight

        if self._use_sklearn:
            import sklearn.linear_model
            model = sklearn.linear_model.LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver='lbfgs'
            )
            model.fit(X_scaled, y, sample_weight=weights_vec)
            self._sklearn_model = model
            self.weights = model.coef_[0]
            self.bias = model.intercept_[0]
        else:
            # Pure SciPy L-BFGS-B optimization
            n_features = X.shape[1]
            reg_lambda = 1.0 / self.C

            def loss_and_grad(params):
                w = params[:n_features]
                b = params[n_features]
                z = np.clip(X_scaled @ w + b, -30.0, 30.0)
                p = expit(z)
                
                # Weighted binary cross-entropy with L2 penalty
                eps = 1e-12
                p_safe = np.clip(p, eps, 1.0 - eps)
                bce = -np.sum(weights_vec * (y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe)))
                l2 = 0.5 * reg_lambda * np.sum(w ** 2)
                loss = bce + l2
                
                # Gradients
                diff = weights_vec * (p - y)
                grad_w = X_scaled.T @ diff + reg_lambda * w
                grad_b = np.sum(diff)
                return loss, np.concatenate([grad_w, [grad_b]])

            init_params = np.zeros(n_features + 1)
            res = minimize(
                loss_and_grad,
                init_params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': self.max_iter}
            )
            self.weights = res.x[:n_features]
            self.bias = res.x[n_features]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = (X - self.feature_means) / self.feature_stds
        z = np.clip(X_scaled @ self.weights + self.bias, -30.0, 30.0)
        return expit(z)

    def get_feature_importances(self) -> np.ndarray:
        return np.abs(self.weights)
