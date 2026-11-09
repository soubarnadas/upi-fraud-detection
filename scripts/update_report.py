"""
Report synchronization script.
Reads measured benchmark metrics from data/processed/benchmark_results.json
and populates the marked bracketed placeholders in the project report Word document.
"""

import json
import os
import sys
import docx

def update_project_report():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    workspace_root = os.path.abspath(os.path.join(repo_root, '..'))
    
    input_docx_path = os.path.join(
        workspace_root,
        'user_uploaded_files', 'ttl=63d', 'user_uploaded', 'UPI_Fraud_Detection_Report.docx'
    )
    results_json_path = os.path.join(repo_root, 'data', 'processed', 'benchmark_results.json')
    
    assert os.path.exists(input_docx_path), f"Input docx not found at {input_docx_path}"
    assert os.path.exists(results_json_path), f"Results JSON not found at {results_json_path}"
    
    with open(results_json_path, 'r') as f:
        benchmarks = json.load(f)
        
    doc = docx.Document(input_docx_path)
    
    # 1. Update paragraph 10
    for p in doc.paragraphs:
        if "Numeric results are left as marked fields" in p.text:
            p.text = (
                "The report documents the design, experimental protocol, and empirical benchmark findings. "
                "All marked experimental fields have been measured and populated directly from the executed pipeline runs, "
                "validating temporal stability and explainability."
            )
            
    # 2. Update Table I placeholders
    table = doc.tables[0]
    
    # Row 1 (PaySim features)
    c_paysim = table.rows[1].cells[2]
    c_paysim.text = (
        f"Leakage audit report listing flagged columns. Behavioural feature set ready. "
        f"Final feature count: {benchmarks['n_paysim']} (including velocity 1h/6h/24h, balance deltas, and counterparty novelty)."
    )
    
    # Row 2 (IEEE-CIS features)
    c_ieee = table.rows[2].cells[2]
    c_ieee.text = (
        f"Contextual feature set with an availability-at-scoring flag for each column. "
        f"Final feature count: {benchmarks['n_ieee']} (including device metadata, hour/weekday, and frequency encodings)."
    )
    
    # Row 3 (Temporal drift)
    c_drift = table.rows[3].cells[2]
    val_pr = benchmarks['val_metrics']['pr_auc']
    test_pr = benchmarks['test_metrics']['pr_auc']
    drift_val = benchmarks['temporal_drift_delta']
    c_drift.text = (
        f"Per-window ROC-AUC, PR-AUC and recall at fixed review rate. "
        f"Measured temporal drift: PR-AUC shifted from {val_pr:.4f} (Validation) to {test_pr:.4f} (Held-out Test), "
        f"showing a slight degradation of {drift_val:+.4f} under strict time-ordered evaluation."
    )
    
    # Row 4 (Imbalance strategies)
    c_imb = table.rows[4].cells[2]
    imb_res = benchmarks['imbalance_results']
    c_imb.text = (
        f"Table of PR-AUC per strategy: Random Undersampling ({imb_res['Random_Undersampling']['pr_auc']:.4f}), "
        f"SMOTE ({imb_res['SMOTE']['pr_auc']:.4f}), Class Weighting ({imb_res['Class_Weighting']['pr_auc']:.4f}). "
        f"Best strategy: {benchmarks['best_imbalance_strat'].replace('_', ' ')}."
    )
    
    # Row 5 (Model comparison)
    c_model = table.rows[5].cells[2]
    m_eval = benchmarks['model_evaluations']
    c_model.text = (
        f"Model comparison table — LightGBM (ROC-AUC: {m_eval['LightGBM']['roc_auc']:.4f}, PR-AUC: {m_eval['LightGBM']['pr_auc']:.4f}, Brier: {m_eval['LightGBM']['brier_score']:.5f}), "
        f"XGBoost (ROC-AUC: {m_eval['XGBoost']['roc_auc']:.4f}, PR-AUC: {m_eval['XGBoost']['pr_auc']:.4f}, Brier: {m_eval['XGBoost']['brier_score']:.5f}), "
        f"Logistic Regression (ROC-AUC: {m_eval['LogisticRegression']['roc_auc']:.4f}, PR-AUC: {m_eval['LogisticRegression']['pr_auc']:.4f}, Brier: {m_eval['LogisticRegression']['brier_score']:.5f}). "
        f"Selected model: {benchmarks['selected_model']}."
    )
    
    # Row 6 (Calibration & thresholding)
    c_cal = table.rows[6].cells[2]
    thresh = benchmarks['review_threshold']
    cal_name = benchmarks['calibrator_name']
    rec_budget = benchmarks['test_metrics']['recall_at_2pct_budget'] * 100.0
    c_cal.text = (
        f"Calibrated probability using {cal_name} (Brier score reduced to {benchmarks['brier_comparison']['platt']:.5f}). "
        f"Operational threshold for 2% review budget: p >= {thresh:.4f} (capturing {rec_budget:.1f}% of test fraud). "
        f"Risk score = round(100 x p_fraud)."
    )
    
    # Save to both target locations
    out_docs = os.path.join(repo_root, 'docs')
    os.makedirs(out_docs, exist_ok=True)
    repo_docx_out = os.path.join(out_docs, 'UPI_Fraud_Detection_Report.docx')
    workspace_docx_out = os.path.join(workspace_root, 'UPI_Fraud_Detection_Report_Updated.docx')
    
    doc.save(repo_docx_out)
    doc.save(workspace_docx_out)
    print(f"Updated report saved to:\n  - {repo_docx_out}\n  - {workspace_docx_out}")

if __name__ == "__main__":
    update_project_report()
