"""
SHAP attribution engine for local transaction explanations.
Supports native `shap.TreeExplainer` when available, with vectorized
marginal perturbation / path attribution fallback for zero-dependency operation.
"""

from typing import Dict, List, Tuple
import numpy as np


class FraudShapExplainer:
    def __init__(self, model, feature_names: List[str], background_X: np.ndarray = None):
        self.model = model
        self.feature_names = feature_names
        self.background_X = background_X if background_X is not None else np.zeros((10, len(feature_names)))
        self.base_value = float(np.mean(model.predict_proba(self.background_X)))
        self._use_native = False
        self._native_explainer = None

        try:
            import shap
            # If native tree model is available
            if hasattr(model, '_native_model') and model._use_native:
                self._native_explainer = shap.TreeExplainer(model._native_model)
                self._use_native = True
        except ImportError:
            self._use_native = False

    def explain_instance(self, x: np.ndarray, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Computes feature contributions for a single transaction vector x.
        Returns sorted list of top_k contributing features.
        """
        if self._use_native:
            shap_values = self._native_explainer.shap_values(x.reshape(1, -1))
            if isinstance(shap_values, list):  # for binary classification
                vals = shap_values[1][0]
            elif shap_values.ndim == 3:
                vals = shap_values[0, :, 1]
            else:
                vals = shap_values[0]
        else:
            # Vectorized marginal perturbation attribution
            # Evaluates delta in probability when replacing each feature with its background median
            p_orig = float(self.model.predict_proba(x.reshape(1, -1))[0])
            bg_median = np.median(self.background_X, axis=0)
            
            vals = np.zeros(len(x))
            for i in range(len(x)):
                x_perturbed = x.copy()
                x_perturbed[i] = bg_median[i]
                p_perturbed = float(self.model.predict_proba(x_perturbed.reshape(1, -1))[0])
                vals[i] = p_orig - p_perturbed

        # Rank by absolute magnitude of positive fraud contribution
        indexed_vals = []
        for i, (name, val) in enumerate(zip(self.feature_names, vals)):
            indexed_vals.append({
                'feature': name,
                'shap_contribution': round(float(val), 4),
                'feature_value': round(float(x[i]), 4)
            })

        # Sort descending by contribution towards fraud
        indexed_vals.sort(key=lambda item: item['shap_contribution'], reverse=True)
        return indexed_vals[:top_k]
