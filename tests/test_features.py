"""
Tests for leakage audit and feature extraction modules.
"""

import unittest
import numpy as np
from src.data.synthetic_paysim import generate_synthetic_paysim
from src.audit.leakage_audit import LeakageAuditor
from src.features.behavioural import BehaviouralFeatureExtractor


class TestFeaturesAndAudit(unittest.TestCase):
    def test_leakage_audit(self):
        df = generate_synthetic_paysim(n_samples=200, random_state=42)
        auditor = LeakageAuditor()
        report = auditor.generate_report(auditor.audit_dataframe(df, "PaySim"))
        
        # Verify isFlaggedFraud and isFraud are dropped
        flagged_cols = report[report['Action'] == 'DROP']['Column'].tolist()
        self.assertIn('isFraud', flagged_cols)
        self.assertIn('isFlaggedFraud', flagged_cols)

    def test_behavioural_features(self):
        df = generate_synthetic_paysim(n_samples=500, random_state=42)
        extractor = BehaviouralFeatureExtractor(filter_transfer_cashout=True)
        transformed = extractor.transform(df, is_training=True)
        
        expected_cols = [
            'log_amount', 'amount_to_balance_orig', 'is_account_draining',
            'velocity_1h', 'velocity_6h', 'velocity_24h', 'new_counterparty'
        ]
        for col in expected_cols:
            self.assertIn(col, transformed.columns)
            
        X, y = extractor.get_feature_matrix(transformed)
        self.assertEqual(X.shape[0], len(transformed))
        self.assertEqual(X.shape[1], len(extractor.feature_columns))
        self.assertFalse(np.isnan(X).any())


if __name__ == '__main__':
    unittest.main()
