"""
Tests for model training, inference, and metric calculations.
"""

import unittest
import numpy as np
from src.models.logistic import FraudLogisticRegression
from src.models.tree_models import LightGBMModel, XGBoostModel
from src.metrics.evaluation import FraudMetrics
from src.imbalance.handlers import ImbalanceHandler


class TestModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(300, 6)
        # Highly imbalanced: minority ~5%
        self.y_train = (self.X_train[:, 0] + self.X_train[:, 1] > 2.0).astype(int)
        
        self.X_test = rng.randn(100, 6)
        self.y_test = (self.X_test[:, 0] + self.X_test[:, 1] > 2.0).astype(int)

    def test_imbalance_smote(self):
        X_res, y_res = ImbalanceHandler.smote(self.X_train, self.y_train, target_ratio=0.3)
        self.assertGreater(len(y_res), len(self.y_train))
        self.assertGreater(np.sum(y_res == 1), np.sum(self.y_train == 1))

    def test_logistic_regression(self):
        model = FraudLogisticRegression(C=1.0)
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_test)
        self.assertEqual(len(probs), len(self.X_test))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        auc = FraudMetrics.roc_auc_score(self.y_test, probs)
        self.assertGreater(auc, 0.6)

    def test_lightgbm_model(self):
        model = LightGBMModel(n_estimators=25, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_test)
        self.assertEqual(len(probs), len(self.X_test))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        auc = FraudMetrics.roc_auc_score(self.y_test, probs)
        self.assertGreater(auc, 0.6)

    def test_xgboost_model(self):
        model = XGBoostModel(n_estimators=25, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_test)
        self.assertEqual(len(probs), len(self.X_test))
        auc = FraudMetrics.roc_auc_score(self.y_test, probs)
        self.assertGreater(auc, 0.6)


if __name__ == '__main__':
    unittest.main()
