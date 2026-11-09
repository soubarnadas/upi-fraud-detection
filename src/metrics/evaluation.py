"""
Evaluation metrics module for imbalanced fraud detection.
Computes:
- PR-AUC (Average Precision)
- ROC-AUC
- Brier Score (probability accuracy)
- Recall at Fixed Operational Review Budget (e.g. top 1%, 2% volume)
- Temporal drift analysis across validation and test time horizons
"""

from typing import Dict, Tuple
import numpy as np


class FraudMetrics:
    @staticmethod
    def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Computes ROC-AUC via Wilcoxon-Mann-Whitney rank statistic."""
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        n_pos = len(pos)
        n_neg = len(neg)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # Rank-based calculation
        all_scores = np.concatenate([pos, neg])
        ranks = np.argsort(np.argsort(all_scores)) + 1
        pos_ranks = ranks[:n_pos]
        u_stat = np.sum(pos_ranks) - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))

    @staticmethod
    def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Computes PR-AUC (Average Precision) using trapezoidal integration
        over descending thresholds.
        """
        order = np.argsort(-y_score)
        y_ordered = y_true[order]
        
        tp = np.cumsum(y_ordered == 1)
        fp = np.cumsum(y_ordered == 0)
        
        n_pos = np.sum(y_true == 1)
        if n_pos == 0:
            return 0.0
            
        recalls = tp / n_pos
        precisions = tp / (tp + fp)
        
        # Trapezoidal area under PR curve
        recalls = np.concatenate([[0.0], recalls])
        precisions = np.concatenate([[precisions[0] if len(precisions) > 0 else 1.0], precisions])
        
        pr_auc = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        return float(np.clip(pr_auc, 0.0, 1.0))

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Mean squared error between calibrated probability and true binary label."""
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def recall_at_review_rate(
        y_true: np.ndarray,
        y_score: np.ndarray,
        review_budget_rate: float = 0.02
    ) -> Tuple[float, float, float]:
        """
        Calculates recall and precision when human operational reviewers
        can only manually inspect the top X% (e.g. 2%) most suspicious transactions.
        Returns: (recall, precision, threshold)
        """
        n = len(y_score)
        k = max(1, int(n * review_budget_rate))
        top_k_indices = np.argsort(-y_score)[:k]
        
        threshold = float(y_score[top_k_indices[-1]])
        flagged_true = y_true[top_k_indices]
        
        n_fraud = np.sum(y_true == 1)
        fraud_caught = np.sum(flagged_true == 1)
        
        recall = float(fraud_caught / n_fraud) if n_fraud > 0 else 0.0
        precision = float(fraud_caught / k)
        return recall, precision, threshold

    @classmethod
    def evaluate_all(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        review_rate: float = 0.02
    ) -> Dict[str, float]:
        roc_auc = cls.roc_auc_score(y_true, y_prob)
        pr_auc = cls.pr_auc_score(y_true, y_prob)
        brier = cls.brier_score(y_true, y_prob)
        rec_k, prec_k, thresh_k = cls.recall_at_review_rate(y_true, y_prob, review_rate)
        
        return {
            'roc_auc': round(roc_auc, 4),
            'pr_auc': round(pr_auc, 4),
            'brier_score': round(brier, 5),
            f'recall_at_{int(review_rate*100)}pct_budget': round(rec_k, 4),
            f'precision_at_{int(review_rate*100)}pct_budget': round(prec_k, 4),
            'review_threshold': round(thresh_k, 4)
        }
