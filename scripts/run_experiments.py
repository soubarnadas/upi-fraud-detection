"""
Comprehensive experimental benchmarking pipeline for PaySim and IEEE-CIS datasets.
Executes the experimental protocol defined in the project report:
1. Leakage audit
2. Feature extraction
3. Strict chronological splitting
4. Imbalance comparison (Class weights vs Undersampling vs SMOTE)
5. Model comparison (LightGBM vs XGBoost vs Logistic Regression)
6. Probability calibration (Platt vs Isotonic vs Raw)
7. Temporal drift estimation on held-out test window
8. Top SHAP explanations
"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.synthetic_paysim import generate_synthetic_paysim
from src.data.synthetic_ieee import generate_synthetic_ieee
from src.data.temporal_split import TemporalSplitter
from src.audit.leakage_audit import LeakageAuditor
from src.features.behavioural import BehaviouralFeatureExtractor
from src.features.contextual import ContextualFeatureExtractor
from src.imbalance.handlers import ImbalanceHandler
from src.models.logistic import FraudLogisticRegression
from src.models.tree_models import LightGBMModel, XGBoostModel
from src.calibration.calibrators import PlattScalingCalibrator, IsotonicCalibrator
from src.calibration.thresholding import ThresholdManager
from src.metrics.evaluation import FraudMetrics
from src.explainability.shap_engine import FraudShapExplainer


def run_all_experiments():
    print("=" * 70)
    print("STARTING UPI FRAUD DETECTION EXPERIMENTAL PROTOCOL")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(results_dir, exist_ok=True)

    # 1. DATA GENERATION & LEAKAGE AUDIT
    print("\n[Stage 1] Generating synthetic PaySim & IEEE-CIS datasets...")
    df_paysim = generate_synthetic_paysim(n_samples=25000, fraud_rate=0.008, random_state=42)
    df_ieee_tx, df_ieee_id = generate_synthetic_ieee(n_samples=10000, fraud_rate=0.035, random_state=42)

    auditor = LeakageAuditor()
    audit_paysim = auditor.generate_report(auditor.audit_dataframe(df_paysim, "PaySim"))
    audit_ieee = auditor.generate_report(auditor.audit_dataframe(df_ieee_tx, "IEEE-CIS"))

    print("PaySim Audit - Flagged Columns:")
    print(audit_paysim[audit_paysim['Status'] != 'SAFE'][['Column', 'Status', 'Action']])

    # 2. FEATURE EXTRACTION
    print("\n[Stage 2] Feature Engineering without temporal lookahead...")
    splitter = TemporalSplitter(time_col='step', train_ratio=0.65, cal_ratio=0.10, val_ratio=0.10, test_ratio=0.15)
    splits_paysim = splitter.split(df_paysim)

    b_extractor = BehaviouralFeatureExtractor(filter_transfer_cashout=True)
    train_paysim = b_extractor.transform(splits_paysim['train'], is_training=True)
    cal_paysim = b_extractor.transform(splits_paysim['cal'], is_training=False)
    val_paysim = b_extractor.transform(splits_paysim['val'], is_training=False)
    test_paysim = b_extractor.transform(splits_paysim['test'], is_training=False)

    n_paysim_features = len(b_extractor.feature_columns)
    print(f"PaySim Behavioural Feature Count (n_paysim): {n_paysim_features}")

    # Contextual features for IEEE-CIS
    ieee_splitter = TemporalSplitter(time_col='TransactionDT', train_ratio=0.65, cal_ratio=0.10, val_ratio=0.10, test_ratio=0.15)
    splits_ieee = ieee_splitter.split(df_ieee_tx)
    c_extractor = ContextualFeatureExtractor()
    train_ieee = c_extractor.transform(splits_ieee['train'], df_ieee_id, is_training=True)
    n_ieee_features = len(c_extractor.feature_columns)
    print(f"IEEE-CIS Contextual Feature Count (n_ieee): {n_ieee_features}")

    # Extract feature matrices for PaySim
    X_train, y_train = b_extractor.get_feature_matrix(train_paysim)
    X_cal, y_cal = b_extractor.get_feature_matrix(cal_paysim)
    X_val, y_val = b_extractor.get_feature_matrix(val_paysim)
    X_test, y_test = b_extractor.get_feature_matrix(test_paysim)

    print(f"Train samples: {len(X_train)} (fraud={y_train.sum()})")
    print(f"Cal samples:   {len(X_cal)} (fraud={y_cal.sum()})")
    print(f"Val samples:   {len(X_val)} (fraud={y_val.sum()})")
    print(f"Test samples:  {len(X_test)} (fraud={y_test.sum()})")

    # 3. IMBALANCE EXPERIMENTS
    print("\n[Stage 3] Running Imbalance Handling Experiments on Validation Fold...")
    imbalance_results = {}

    # Strategy A: Baseline Class Weighting
    _, w_pos = ImbalanceHandler.compute_class_weights(y_train)
    m_weighted = LightGBMModel(n_estimators=40, scale_pos_weight=w_pos)
    m_weighted.fit(X_train, y_train)
    p_val_weighted = m_weighted.predict_proba(X_val)
    imbalance_results['Class_Weighting'] = FraudMetrics.evaluate_all(y_val, p_val_weighted)

    # Strategy B: Random Undersampling
    X_under, y_under = ImbalanceHandler.random_undersample(X_train, y_train, ratio=0.1)
    m_under = LightGBMModel(n_estimators=40)
    m_under.fit(X_under, y_under)
    p_val_under = m_under.predict_proba(X_val)
    imbalance_results['Random_Undersampling'] = FraudMetrics.evaluate_all(y_val, p_val_under)

    # Strategy C: SMOTE
    X_smote, y_smote = ImbalanceHandler.smote(X_train, y_train, target_ratio=0.15)
    m_smote = LightGBMModel(n_estimators=40)
    m_smote.fit(X_smote, y_smote)
    p_val_smote = m_smote.predict_proba(X_val)
    imbalance_results['SMOTE'] = FraudMetrics.evaluate_all(y_val, p_val_smote)

    print("\n--- Imbalance Comparison Table (Validation PR-AUC) ---")
    for strat, res in imbalance_results.items():
        print(f"Strategy: {strat:<22} | PR-AUC: {res['pr_auc']:.4f} | ROC-AUC: {res['roc_auc']:.4f} | Recall@2%: {res['recall_at_2pct_budget']:.4f}")

    best_imbalance_strat = max(imbalance_results.keys(), key=lambda k: imbalance_results[k]['pr_auc'])
    print(f"\nBest Imbalance Handling Strategy: {best_imbalance_strat}")

    # 4. MODEL COMPARISON
    print("\n[Stage 4] Running Model Comparison on Validation Fold...")
    models = {
        'LightGBM': LightGBMModel(n_estimators=45, learning_rate=0.08, scale_pos_weight=w_pos),
        'XGBoost': XGBoostModel(n_estimators=45, learning_rate=0.08, scale_pos_weight=w_pos),
        'LogisticRegression': FraudLogisticRegression(C=1.0, scale_pos_weight=w_pos)
    }

    model_evaluations = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train, feature_names=b_extractor.feature_columns)
        fitted_models[name] = model
        p_val = model.predict_proba(X_val)
        model_evaluations[name] = FraudMetrics.evaluate_all(y_val, p_val)

    print("\n--- Model Comparison Table ---")
    for name, res in model_evaluations.items():
        print(f"Model: {name:<20} | ROC-AUC: {res['roc_auc']:.4f} | PR-AUC: {res['pr_auc']:.4f} | Brier: {res['brier_score']:.5f} | Recall@2%: {res['recall_at_2pct_budget']:.4f}")

    selected_model_name = max(model_evaluations.keys(), key=lambda k: model_evaluations[k]['pr_auc'])
    print(f"\nSelected Primary Model: {selected_model_name}")
    selected_model = fitted_models[selected_model_name]

    # 5. PROBABILITY CALIBRATION & THRESHOLDING
    print("\n[Stage 5] Fitting Probability Calibration on Held-out Calibration Fold...")
    raw_cal_probs = selected_model.predict_proba(X_cal)

    platt_cal = PlattScalingCalibrator().fit(raw_cal_probs, y_cal)
    isotonic_cal = IsotonicCalibrator().fit(raw_cal_probs, y_cal)

    raw_val_probs = selected_model.predict_proba(X_val)
    platt_val_probs = platt_cal.predict(raw_val_probs)
    isotonic_val_probs = isotonic_cal.predict(raw_val_probs)

    brier_raw = FraudMetrics.brier_score(y_val, raw_val_probs)
    brier_platt = FraudMetrics.brier_score(y_val, platt_val_probs)
    brier_iso = FraudMetrics.brier_score(y_val, isotonic_val_probs)

    print(f"Validation Brier Score - Raw: {brier_raw:.5f} | Platt: {brier_platt:.5f} | Isotonic: {brier_iso:.5f}")

    best_calibrator = platt_cal if brier_platt <= brier_iso else isotonic_cal
    calibrator_name = "Platt Scaling" if best_calibrator == platt_cal else "Isotonic Regression"
    print(f"Selected Calibrator: {calibrator_name}")

    # Operational threshold for 2% review budget
    thresh_mgr = ThresholdManager(target_review_budget_rate=0.02)
    calibrated_val_probs = best_calibrator.predict(raw_val_probs)
    review_threshold = thresh_mgr.fit_by_budget(calibrated_val_probs)
    rec_val, prec_val, _ = FraudMetrics.recall_at_review_rate(y_val, calibrated_val_probs, 0.02)

    print(f"Operational 2% Review Budget Threshold: {review_threshold:.4f} (Validation Fraud Recall: {rec_val*100:.1f}%)")

    # 6. TEMPORAL DRIFT & FINAL TEST HORIZON
    print("\n[Stage 6] Evaluating on Held-Out Chronological Test Window...")
    raw_test_probs = selected_model.predict_proba(X_test)
    cal_test_probs = best_calibrator.predict(raw_test_probs)
    test_metrics = FraudMetrics.evaluate_all(y_test, cal_test_probs, review_rate=0.02)

    val_pr_auc = model_evaluations[selected_model_name]['pr_auc']
    test_pr_auc = test_metrics['pr_auc']
    temporal_drift_delta = test_pr_auc - val_pr_auc

    print(f"Validation PR-AUC: {val_pr_auc:.4f}")
    print(f"Held-out Test PR-AUC: {test_pr_auc:.4f}")
    print(f"Temporal Drift (Test - Val): {temporal_drift_delta:+.4f}")
    print(f"Test Recall @ 2% Review Budget: {test_metrics['recall_at_2pct_budget']*100:.1f}%")

    # 7. SHAP EXPLAINABILITY
    print("\n[Stage 7] Generating Sample SHAP Explanations for Flagged Test Transactions...")
    explainer = FraudShapExplainer(selected_model, b_extractor.feature_columns, background_X=X_train[:30])
    flagged_indices = np.where(cal_test_probs >= review_threshold)[0]
    
    sample_explanations = []
    for idx in flagged_indices[:3]:
        x_inst = X_test[idx]
        top_factors = explainer.explain_instance(x_inst, top_k=5)
        p_cal = cal_test_probs[idx]
        sample_explanations.append({
            'test_index': int(idx),
            'true_label': int(y_test[idx]),
            'calibrated_probability': round(float(p_cal), 4),
            'risk_score': thresh_mgr.calculate_risk_score(p_cal),
            'top_factors': top_factors
        })

    print("Sample Top SHAP Explanation:")
    print(json.dumps(sample_explanations[0], indent=2))

    benchmark_payload = {
        'n_paysim': n_paysim_features,
        'n_ieee': n_ieee_features,
        'imbalance_results': imbalance_results,
        'best_imbalance_strat': best_imbalance_strat,
        'model_evaluations': model_evaluations,
        'selected_model': selected_model_name,
        'calibrator_name': calibrator_name,
        'brier_comparison': {'raw': round(brier_raw, 5), 'platt': round(brier_platt, 5), 'isotonic': round(brier_iso, 5)},
        'review_threshold': round(review_threshold, 4),
        'val_metrics': model_evaluations[selected_model_name],
        'test_metrics': test_metrics,
        'temporal_drift_delta': round(temporal_drift_delta, 4),
        'drift_summary': f"PR-AUC: {val_pr_auc:.4f} (Val) vs {test_pr_auc:.4f} (Test), drift delta {temporal_drift_delta:+.4f}",
        'sample_explanations': sample_explanations
    }

    out_path = os.path.join(results_dir, 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(benchmark_payload, f, indent=2)

    print(f"\nBenchmark results successfully saved to: {out_path}")
    print("=" * 70)
    return benchmark_payload


if __name__ == "__main__":
    run_all_experiments()
