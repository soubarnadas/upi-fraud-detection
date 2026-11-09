"""
Base interface for fraud classification models.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseFraudModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.feature_names = []

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> "BaseFraudModel":
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 1D array of predicted probabilities for positive class (fraud)."""
        pass

    @abstractmethod
    def get_feature_importances(self) -> np.ndarray:
        pass
