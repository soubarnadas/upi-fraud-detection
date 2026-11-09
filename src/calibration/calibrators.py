"""
Probability calibration module.
Implements:
1. Platt Scaling (Logistic Sigmoid Calibration)
2. Isotonic Regression via Pool Adjacent Violators Algorithm (PAVA)
3. Calibration Reliability Curve and Expected Calibration Error (ECE)
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


class PlattScalingCalibrator:
    """Parametric sigmoid calibration mapping raw model probabilities to true probabilities."""
    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "PlattScalingCalibrator":
        eps = 1e-6
        clipped_p = np.clip(raw_probs, eps, 1.0 - eps)
        scores = logit(clipped_p)

        def nll(params):
            a, b = params
            z = np.clip(a * scores + b, -30.0, 30.0)
            p = expit(z)
            p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
            loss = -np.sum(y_true * np.log(p_safe) + (1.0 - y_true) * np.log(1.0 - p_safe))
            return loss

        res = minimize(nll, [1.0, 0.0], method='Nelder-Mead')
        self.a, self.b = res.x
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        eps = 1e-6
        clipped_p = np.clip(raw_probs, eps, 1.0 - eps)
        scores = logit(clipped_p)
        z = np.clip(self.a * scores + self.b, -30.0, 30.0)
        return expit(z)


class IsotonicCalibrator:
    """Non-parametric isotonic calibration using Pool Adjacent Violators Algorithm (PAVA)."""
    def __init__(self):
        self.x_thresholds = None
        self.y_values = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        order = np.argsort(raw_probs)
        x = raw_probs[order]
        y = y_true[order].astype(float)

        # PAVA implementation
        y_pava = y.copy()
        weights = np.ones(len(y), dtype=float)
        
        blocks = []  # list of [start, end, weight, avg_y]
        for i in range(len(y)):
            blocks.append([i, i, weights[i], y_pava[i]])
            while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
                # Violates monotonicity: pool the two adjacent blocks
                prev = blocks[-2]
                curr = blocks[-1]
                w_comb = prev[2] + curr[2]
                avg_comb = (prev[2] * prev[3] + curr[2] * curr[3]) / w_comb
                blocks[-2] = [prev[0], curr[1], w_comb, avg_comb]
                blocks.pop()

        for b in blocks:
            y_pava[b[0]:b[1]+1] = b[3]

        self.x_thresholds = x
        self.y_values = y_pava
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        return np.interp(raw_probs, self.x_thresholds, self.y_values)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across n uniform probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)

    for i in range(n_bins):
        bin_mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.any(bin_mask):
            bin_size = np.sum(bin_mask)
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)
