"""
Inference serving pipeline orchestrating feature extraction, model prediction,
calibration, SHAP explainability, and rule execution in real-time.
"""

import time
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.features.behavioural import BehaviouralFeatureExtractor
from src.models.tree_models import LightGBMModel
from src.calibration.calibrators import PlattScalingCalibrator
from src.calibration.thresholding import ThresholdManager
from src.explainability.shap_engine import FraudShapExplainer
from src.explainability.rules import FraudRuleEngine
from src.service.schemas import TransactionPayload, FraudScoreResponse, FeatureContribution


class ServingPipeline:
    def __init__(self):
        self.feature_extractor = BehaviouralFeatureExtractor(filter_transfer_cashout=False)
        self.model = None
        self.calibrator = None
        self.threshold_mgr = None
        self.explainer = None
        self.is_ready = False

    def initialize_with_defaults(self):
        """Initializes and pre-trains a baseline model on synthetic data if not pre-loaded."""
        from src.data.synthetic_paysim import generate_synthetic_paysim
        from src.data.temporal_split import TemporalSplitter
        
        df = generate_synthetic_paysim(n_samples=5000, random_state=42)
        splitter = TemporalSplitter(train_ratio=0.7, cal_ratio=0.15, val_ratio=0.05, test_ratio=0.1)
        splits = splitter.split(df)
        
        # Fit features
        train_features = self.feature_extractor.transform(splits['train'], is_training=True)
        X_train, y_train = self.feature_extractor.get_feature_matrix(train_features)
        
        # Train model
        self.model = LightGBMModel(n_estimators=40, learning_rate=0.1, max_depth=4)
        self.model.fit(X_train, y_train, feature_names=self.feature_extractor.feature_columns)
        
        # Calibration
        cal_features = self.feature_extractor.transform(splits['cal'], is_training=False)
        X_cal, y_cal = self.feature_extractor.get_feature_matrix(cal_features)
        raw_cal_probs = self.model.predict_proba(X_cal)
        
        self.calibrator = PlattScalingCalibrator()
        self.calibrator.fit(raw_cal_probs, y_cal)
        
        # Threshold manager
        self.threshold_mgr = ThresholdManager(target_review_budget_rate=0.02)
        calibrated_cal_probs = self.calibrator.predict(raw_cal_probs)
        self.threshold_mgr.fit_by_budget(calibrated_cal_probs)
        
        # SHAP Explainer
        self.explainer = FraudShapExplainer(
            self.model,
            feature_names=self.feature_extractor.feature_columns,
            background_X=X_train[:20]
        )
        self.is_ready = True

    def score(self, payload: TransactionPayload) -> FraudScoreResponse:
        start_time = time.perf_counter()
        
        if not self.is_ready:
            self.initialize_with_defaults()

        # Convert payload to single-row dataframe for feature extraction
        row_dict = {
            'step': payload.step,
            'type': payload.type,
            'amount': payload.amount,
            'nameOrig': payload.nameOrig,
            'oldbalanceOrg': payload.oldbalanceOrg,
            'newbalanceOrig': max(0.0, payload.oldbalanceOrg - payload.amount),
            'nameDest': payload.nameDest,
            'oldbalanceDest': payload.oldbalanceDest,
            'newbalanceDest': payload.oldbalanceDest + payload.amount,
            'isFlaggedFraud': 0
        }
        df_row = pd.DataFrame([row_dict])
        df_feat = self.feature_extractor.transform(df_row, is_training=False)
        
        feature_vals = df_feat[self.feature_extractor.feature_columns].values[0]
        feature_dict = dict(zip(self.feature_extractor.feature_columns, feature_vals))
        
        # Predict & calibrate
        raw_prob = float(self.model.predict_proba(feature_vals.reshape(1, -1))[0])
        p_cal = float(self.calibrator.predict(np.array([raw_prob]))[0])
        p_cal = float(np.clip(p_cal, 0.0, 1.0))
        
        risk_score = self.threshold_mgr.calculate_risk_score(p_cal)
        risk_tier = self.threshold_mgr.get_risk_tier(risk_score)
        
        # Explainability: top 5 SHAP factors
        shap_factors = self.explainer.explain_instance(feature_vals, top_k=5)
        shap_models = [FeatureContribution(**item) for item in shap_factors]
        
        # Heuristic rules
        rule_flags = FraudRuleEngine.evaluate_rules(row_dict, feature_dict)
        recommended_action = FraudRuleEngine.recommend_action(risk_score, rule_flags)
        
        latency = (time.perf_counter() - start_time) * 1000.0
        
        return FraudScoreResponse(
            transaction_ref=str(uuid.uuid4())[:8],
            risk_score=risk_score,
            risk_tier=risk_tier,
            calibrated_probability=round(p_cal, 4),
            action=recommended_action,
            rule_flags=rule_flags,
            top_shap_factors=shap_models,
            latency_ms=round(latency, 2)
        )
