"""
Operational threshold selection and 0-100 risk scoring.
Ties machine learning probabilities to bank operational review budgets and financial cost matrices.
"""

from typing import Dict, Tuple
import numpy as np


class ThresholdManager:
    def __init__(self, target_review_budget_rate: float = 0.02):
        self.target_review_budget_rate = target_review_budget_rate
        self.operational_threshold = 0.5

    def fit_by_budget(self, y_prob: np.ndarray) -> float:
        """
        Determines the decision threshold that flags exactly the target review budget rate
        (e.g. top 2% of transaction traffic).
        """
        n = len(y_prob)
        k = max(1, int(n * self.target_review_budget_rate))
        sorted_probs = np.sort(y_prob)
        self.operational_threshold = float(sorted_probs[-k])
        return self.operational_threshold

    def fit_by_cost(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fp: float = 10.0,   # Cost of customer friction / manual review
        cost_fn: float = 250.0   # Average cost of unrecovered fraudulent transaction
    ) -> float:
        """
        Finds threshold minimizing financial cost: Cost = Cost_FP * FP + Cost_FN * FN
        """
        candidate_thresholds = np.linspace(0.01, 0.99, 100)
        best_cost = float('inf')
        best_thresh = 0.5

        for thresh in candidate_thresholds:
            preds = (y_prob >= thresh).astype(int)
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            total_cost = cost_fp * fp + cost_fn * fn
            if total_cost < best_cost:
                best_cost = total_cost
                best_thresh = thresh

        self.operational_threshold = float(best_thresh)
        return self.operational_threshold

    @staticmethod
    def calculate_risk_score(p_calibrated: float) -> int:
        """Scales calibrated probability into an intuitive 0 to 100 integer risk score."""
        score = int(round(p_calibrated * 100.0))
        return int(np.clip(score, 0, 100))

    @staticmethod
    def get_risk_tier(risk_score: int) -> str:
        """Maps 0-100 risk score to actionable risk classification."""
        if risk_score < 25:
            return "LOW"
        elif risk_score < 60:
            return "MEDIUM"
        elif risk_score < 85:
            return "HIGH"
        else:
            return "CRITICAL"
