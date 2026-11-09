"""
Tests for SHAP explanations and rule engine.
"""

import unittest
import numpy as np
from src.models.tree_models import LightGBMModel
from src.explainability.shap_engine import FraudShapExplainer
from src.explainability.rules import FraudRuleEngine


class TestExplainability(unittest.TestCase):
    def test_shap_factors(self):
        rng = np.random.RandomState(42)
        X = rng.randn(150, 5)
        y = (X[:, 0] > 0.5).astype(int)
        
        feature_names = ['amount', 'velocity', 'novelty', 'drain', 'night']
        model = LightGBMModel(n_estimators=15)
        model.fit(X, y, feature_names=feature_names)
        
        explainer = FraudShapExplainer(model, feature_names=feature_names, background_X=X[:20])
        top_factors = explainer.explain_instance(X[0], top_k=3)
        
        self.assertEqual(len(top_factors), 3)
        self.assertIn('feature', top_factors[0])
        self.assertIn('shap_contribution', top_factors[0])

    def test_rule_engine(self):
        txn = {'amount': 50000.0, 'step': 3}
        features = {
            'new_counterparty': 1.0,
            'is_night_txn': 1.0,
            'log_amount': 10.8,
            'is_account_draining': 1.0,
            'dest_starts_zero': 1.0
        }
        flags = FraudRuleEngine.evaluate_rules(txn, features)
        self.assertIn("HIGH_AMOUNT_NEW_RECIPIENT_NIGHT", flags)
        self.assertIn("ACCOUNT_DRAIN_TO_NEW_MULE", flags)
        
        action = FraudRuleEngine.recommend_action(90, flags)
        self.assertEqual(action, "BLOCK_AND_FREEZE")


if __name__ == '__main__':
    unittest.main()
