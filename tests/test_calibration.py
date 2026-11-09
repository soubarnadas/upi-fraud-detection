"""
Tests for probability calibration and threshold selection.
"""

import unittest
import numpy as np
from src.calibration.calibrators import PlattScalingCalibrator, IsotonicCalibrator
from src.calibration.thresholding import ThresholdManager
from src.metrics.evaluation import FraudMetrics


class TestCalibration(unittest.TestCase):
    def test_platt_and_isotonic_calibration(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.2, size=300)
        # Distorted raw probabilities
        raw_probs = np.clip(y_true * 0.7 + rng.uniform(0.0, 0.4, size=300), 0.01, 0.99)
        
        platt = PlattScalingCalibrator().fit(raw_probs, y_true)
        p_platt = platt.predict(raw_probs)
        
        self.assertEqual(len(p_platt), len(raw_probs))
        self.assertTrue((p_platt >= 0.0).all() and (p_platt <= 1.0).all())
        
        iso = IsotonicCalibrator().fit(raw_probs, y_true)
        p_iso = iso.predict(raw_probs)
        self.assertEqual(len(p_iso), len(raw_probs))
        
        # Test threshold manager
        mgr = ThresholdManager(target_review_budget_rate=0.05)
        thresh = mgr.fit_by_budget(p_platt)
        self.assertTrue(0.0 <= thresh <= 1.0)
        
        score = mgr.calculate_risk_score(0.85)
        self.assertEqual(score, 85)
        tier = mgr.get_risk_tier(score)
        self.assertEqual(tier, "CRITICAL")


if __name__ == '__main__':
    unittest.main()
