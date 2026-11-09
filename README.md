# UPI Fraud Detection System

[![CI Pipeline](https://github.com/soubarna/upi-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/soubarna/upi-fraud-detection/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **A Leakage-Aware, Temporally Validated and Explainable Approach to UPI Fraud Detection**  
> *Author:* Soubarna Das, Department of Information Technology, Meghnad Saha Institute of Technology, Kolkata (2026).

---

## 📌 Overview

Unified Payments Interface (UPI) moves money across bank accounts instantaneously. Once settled, fraudulent transactions are extraordinarily difficult to reverse, making **pre-authorization scoring** essential.

This repository provides an end-to-end production pipeline addressing the three fundamental challenges of real-time payment fraud detection:
1. **Target & Settlement Leakage Avoidance**: Rigorous audit separating pre-authorization signals from post-settlement columns and identifier memorization.
2. **Strict Chronological Evaluation**: Time-ordered Train, Calibration, Validation, and Test partitions without random shuffling, measuring real-world temporal drift.
3. **Calibrated Probabilities & Operational Actionability**: Mapping raw model scores into true probabilities via Platt scaling and Isotonic regression (PAVA), selecting decision thresholds based on actual fraud review capacity (e.g., top 2% traffic budget), and returning 0–100 scaled risk scores.
4. **Explainability & Rule Engine**: Local TreeSHAP attributions isolating the top contributing factors per payment alongside deterministic heuristic rule flags.
5. **Real-Time Serving Endpoint**: FastAPI microservice serving sub-15ms scoring, batch inference, and structured explanations.

---

## 🏛️ System Architecture

```
Raw Transaction Stream (PaySim Behavioural & IEEE-CIS Contextual)
                           │
                           ▼
          [Leakage Audit & Column Availability Tagging]
                           │
                           ▼
         [Feature Store: Rolling Velocity, Deltas, Novelty]
                           │
                           ▼
     [Chronological Splits: Train (65%) | Cal (10%) | Val (10%) | Test (15%)]
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   [Imbalance Strategies]       [Model Comparison]
   • Class Weighting            • LightGBM
   • Random Undersampling       • XGBoost
   • SMOTE                      • Logistic Regression
            │                             │
            └──────────────┬──────────────┘
                           ▼
        [Probability Calibration (Held-Out Cal Fold)]
        • Platt Scaling & Isotonic Regression (PAVA)
                           │
                           ▼
   [Operational Review Budget Thresholding (e.g. Top 2%)]
                           │
                           ▼
      [0–100 Normalized Risk Score & Risk Tiers]
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   [Top 5 SHAP Attributions]     [Deterministic Rule Flags]
            │                             │
            └──────────────┬──────────────┘
                           ▼
      [FastAPI Real-Time Pre-Authorization Microservice]
```

---

## 📊 Experimental Results

All figures are empirically validated using chronological splits across the synthetic PaySim behavioural stream (15 features) and IEEE-CIS contextual stream (32 features):

### 1. Imbalance Handling Comparison (Validation Fold)

| Strategy | Validation PR-AUC | Validation ROC-AUC | Recall @ 2% Review Budget |
| :--- | :---: | :---: | :---: |
| **Random Undersampling** | **0.9526** | **0.9993** | **100.0%** |
| **SMOTE** | 0.9407 | 0.9994 | 100.0% |
| **Class Weighting** | 0.8823 | 0.9976 | 90.0% |

### 2. Model Comparison (Validation Fold)

| Model | ROC-AUC | PR-AUC | Brier Score | Recall @ 2% Review Budget |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.9994** | **0.9556** | **0.00908** | **100.0%** |
| **XGBoost** | 0.9985 | 0.9060 | 0.01303 | 100.0% |
| **Logistic Regression** | 0.9973 | 0.8942 | 0.02796 | 90.0% |

### 3. Probability Calibration & Temporal Drift

* **Calibration Impact**: Platt scaling reduced the validation Brier score from **0.00908** to **0.00080** (>11x improvement in probability reliability).
* **Operational Threshold**: A decision threshold of $p \ge 0.0000$ (top 2% review volume) captures **89.5%** of fraud in the held-out test window.
* **Temporal Drift**: PR-AUC shifted from **0.9556** on the validation window to **0.9295** on the future held-out test window (temporal degradation delta: **-0.0261**).

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/soubarna/upi-fraud-detection.git
cd upi-fraud-detection
pip install -r requirements.txt
```

*(Note: The system contains high-performance pure-NumPy fallbacks for all models, metrics, calibration, and SHAP, allowing zero-dependency execution out of the box).*

### 2. Run the Benchmark Suite

Execute the entire experimental pipeline from data generation to calibration and SHAP generation:

```bash
python scripts/run_experiments.py
```

### 3. Synchronize Findings with Word Report

To automatically populate the marked placeholders inside `UPI_Fraud_Detection_Report.docx`:

```bash
python scripts/update_report.py
```

### 4. Run the Real-Time Scoring API

```bash
python scripts/run_server.py
```

The API will be live at `http://localhost:8000`. Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs`.

---

## 🌐 API Usage & Real-Time Scoring

### Request Payload (`POST /score`)

```json
{
  "step": 14,
  "type": "TRANSFER",
  "amount": 85000.0,
  "nameOrig": "C1049281",
  "oldbalanceOrg": 85500.0,
  "nameDest": "C9928114",
  "oldbalanceDest": 0.0,
  "device_type": "mobile"
}
```

### Response Payload

```json
{
  "transaction_ref": "8f31b2e4",
  "risk_score": 92,
  "risk_tier": "CRITICAL",
  "calibrated_probability": 0.9214,
  "action": "BLOCK_AND_FREEZE",
  "rule_flags": [
    "ACCOUNT_DRAIN_TO_NEW_MULE"
  ],
  "top_shap_factors": [
    {
      "feature": "is_account_draining",
      "shap_contribution": 0.3842,
      "feature_value": 1.0
    },
    {
      "feature": "new_counterparty",
      "shap_contribution": 0.2105,
      "feature_value": 1.0
    },
    {
      "feature": "dest_starts_zero",
      "shap_contribution": 0.1420,
      "feature_value": 1.0
    },
    {
      "feature": "log_amount",
      "shap_contribution": 0.0911,
      "feature_value": 11.3504
    },
    {
      "feature": "amount_to_balance_orig",
      "shap_contribution": 0.0825,
      "feature_value": 0.9942
    }
  ],
  "latency_ms": 7.42
}
```

---

## 🧪 Test Suite

Run all unit tests across data splitters, feature extractors, models, calibration, explainability, and FastAPI endpoints:

```bash
python -m unittest discover tests
```

---

## 📂 Repository Structure

```
upi-fraud-detection/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions continuous integration
├── .gitignore
├── LICENSE                    # MIT License
├── README.md                  # System documentation & benchmarks
├── pyproject.toml             # Packaging metadata
├── requirements.txt           # Dependency specifications
├── configs/                   # Configuration files
├── docs/
│   └── UPI_Fraud_Detection_Report.docx  # Updated project report
├── data/
│   ├── raw/                   # Raw transaction caches
│   └── processed/
│       └── benchmark_results.json       # Experimental run metrics
├── src/
│   ├── data/
│   │   ├── synthetic_paysim.py          # PaySim behavioural generator
│   │   ├── synthetic_ieee.py            # IEEE-CIS contextual generator
│   │   └── temporal_split.py            # Chronological 4-window splitter
│   ├── audit/
│   │   └── leakage_audit.py             # Target & settlement leakage auditor
│   ├── features/
│   │   ├── behavioural.py               # Velocity 1h/6h/24h & balance features
│   │   └── contextual.py                # Device, identity & time-of-day features
│   ├── imbalance/
│   │   └── handlers.py                  # SMOTE, undersampling & class weighting
│   ├── models/
│   │   ├── base.py                      # Model interface
│   │   ├── logistic.py                  # Scikit-learn & SciPy L-BFGS logistic
│   │   └── tree_models.py               # LightGBM, XGBoost & pure-NumPy GBDT
│   ├── calibration/
│   │   ├── calibrators.py               # Platt scaling & Isotonic PAVA
│   │   └── thresholding.py              # Review budget & cost curve thresholds
│   ├── explainability/
│   │   ├── shap_engine.py               # TreeSHAP & feature attribution engine
│   │   └── rules.py                     # Heuristic rules & action recommender
│   ├── metrics/
│   │   └── evaluation.py                # PR-AUC, ROC-AUC, Brier & review recall
│   └── service/
│       ├── api.py                       # FastAPI application endpoints
│       ├── schemas.py                   # Pydantic request & response schemas
│       └── pipeline.py                  # Real-time inference pipeline
├── scripts/
│   ├── run_experiments.py               # Experimental benchmark runner
│   ├── update_report.py                 # Word report synchronization
│   └── run_server.py                    # Uvicorn server launcher
└── tests/
    ├── test_data.py                     # Data & temporal split tests
    ├── test_features.py                 # Leakage & feature extraction tests
    ├── test_models.py                   # Model & imbalance tests
    ├── test_calibration.py              # Calibration & thresholding tests
    ├── test_explainability.py           # SHAP & rule engine tests
    └── test_api.py                      # FastAPI endpoint tests
```

---

## 📜 References

1. E. A. Lopez-Rojas, A. Elmaghraby, and C. Axelsson, *"PaySim: A financial mobile money simulator for fraud detection,"* EMSS 2016.
2. IEEE Computational Intelligence Society and Vesta Corporation, *"IEEE-CIS Fraud Detection,"* Kaggle, 2019.
3. S. Kaufman et al., *"Leakage in data mining: Formulation, detection, and avoidance,"* ACM TKDD, 2012.
4. A. Dal Pozzolo et al., *"Learned lessons in credit card fraud detection from a practitioner perspective,"* Expert Syst. Appl., 2014.
5. A. Dal Pozzolo et al., *"Credit card fraud detection: A realistic modeling and a novel learning strategy,"* IEEE TNNLS, 2018.
6. N. V. Chawla et al., *"SMOTE: Synthetic minority over-sampling technique,"* JAIR, 2002.
7. T. Saito and M. Rehmsmeier, *"The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,"* PLoS ONE, 2015.
8. G. Ke et al., *"LightGBM: A highly efficient gradient boosting decision tree,"* NeurIPS 2017.
9. T. Chen and C. Guestrin, *"XGBoost: A scalable tree boosting system,"* ACM KDD 2016.
10. J. C. Platt, *"Probabilistic outputs for support vector machines,"* MIT Press, 1999.
11. A. Niculescu-Mizil and R. Caruana, *"Predicting good probabilities with supervised learning,"* ICML 2005.
12. S. M. Lundberg and S.-I. Lee, *"A unified approach to interpreting model predictions,"* NeurIPS 2017.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
