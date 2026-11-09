"""
Synthetic IEEE-CIS dataset generator for UPI contextual fraud detection.
Mimics the IEEE-CIS Transaction and Identity tables:
- Transaction table: TransactionID, isFraud, TransactionDT, TransactionAmt,
  ProductCD, card1..card6, P_emaildomain, R_emaildomain, addr1, dist1, C1..C5, D1..D3, V1..V10.
- Identity table: TransactionID, DeviceType, DeviceInfo, id_30 (OS), id_31 (Browser).
- Joined on TransactionID.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_synthetic_ieee(
    n_samples: int = 15000,
    fraud_rate: float = 0.035,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(random_state)
    
    transaction_ids = np.arange(3000000, 3000000 + n_samples)
    
    # TransactionDT: seconds spanning 30 days (0 to 2,592,000)
    dt_intervals = rng.exponential(scale=2592000.0 / n_samples, size=n_samples)
    transaction_dt = np.cumsum(dt_intervals).astype(int)
    
    # Product code
    product_cds = rng.choice(['W', 'C', 'R', 'H', 'S'], size=n_samples, p=[0.70, 0.12, 0.08, 0.06, 0.04])
    
    # Amounts
    amounts = np.exp(rng.normal(loc=4.6, scale=1.1, size=n_samples))
    amounts = np.round(np.clip(amounts, 5.0, 5000.0), 2)
    
    # Card fields
    card1 = rng.randint(1000, 20000, size=n_samples)
    card2 = rng.choice([111.0, 150.0, 226.0, 321.0, 490.0, 555.0, np.nan], size=n_samples)
    card4 = rng.choice(['visa', 'mastercard', 'discover', 'american express'], size=n_samples, p=[0.65, 0.30, 0.03, 0.02])
    card6 = rng.choice(['debit', 'credit'], size=n_samples, p=[0.75, 0.25])
    
    # Email domains
    email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'protonmail.com', 'anonymous.com', 'mail.com']
    p_email_probs = np.array([0.45, 0.25, 0.12, 0.10, 0.04, 0.02, 0.02])
    p_email_probs = p_email_probs / p_email_probs.sum()
    p_email = rng.choice(email_domains, size=n_samples, p=p_email_probs)
    
    r_email_domains = email_domains + [np.nan]
    r_email_probs = np.array([0.40, 0.20, 0.10, 0.08, 0.03, 0.02, 0.02, 0.15])
    r_email_probs = r_email_probs / r_email_probs.sum()
    r_email = rng.choice(r_email_domains, size=n_samples, p=r_email_probs)
    
    addr1 = rng.choice([204.0, 299.0, 315.0, 325.0, 441.0, np.nan], size=n_samples)
    dist1 = np.where(rng.rand(n_samples) < 0.6, np.nan, np.round(rng.exponential(scale=120.0, size=n_samples), 1))
    
    # Count and delta features
    c_features = {f"C{i}": rng.poisson(lam=1.5 * i, size=n_samples) for i in range(1, 6)}
    d_features = {f"D{i}": np.where(rng.rand(n_samples) < 0.2, np.nan, np.round(rng.exponential(scale=15.0 * i, size=n_samples), 1)) for i in range(1, 4)}
    v_features = {f"V{i}": np.round(rng.normal(loc=0.0, scale=1.0, size=n_samples), 3) for i in range(1, 11)}
    
    # Fraud labels
    is_fraud = np.zeros(n_samples, dtype=int)
    n_fraud = int(n_samples * fraud_rate)
    
    # Fraud correlation logic: higher amounts, anonymous emails, rapid counts
    risk_score = (
        (amounts > 250.0).astype(float) * 1.5 +
        np.isin(p_email, ['protonmail.com', 'anonymous.com', 'mail.com']).astype(float) * 2.5 +
        (card6 == 'credit').astype(float) * 0.8 +
        (product_cds == 'C').astype(float) * 1.8 +
        (c_features['C1'] > 5).astype(float) * 1.2
    )
    risk_prob = 1.0 / (1.0 + np.exp(-(risk_score - np.mean(risk_score))))
    risk_prob = risk_prob / risk_prob.sum()
    fraud_indices = rng.choice(np.arange(n_samples), size=n_fraud, replace=False, p=risk_prob)
    is_fraud[fraud_indices] = 1
    
    # Elevate Vesta risk signals for fraud
    for idx in fraud_indices:
        v_features['V1'][idx] += rng.uniform(2.0, 5.0)
        v_features['V3'][idx] += rng.uniform(1.5, 4.0)
    
    df_transaction = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': is_fraud,
        'TransactionDT': transaction_dt,
        'TransactionAmt': amounts,
        'ProductCD': product_cds,
        'card1': card1,
        'card2': card2,
        'card4': card4,
        'card6': card6,
        'P_emaildomain': p_email,
        'R_emaildomain': r_email,
        'addr1': addr1,
        'dist1': dist1,
        **c_features,
        **d_features,
        **v_features
    })
    
    # Identity table: ~35% of transactions have identity information
    identity_indices = np.where(rng.rand(n_samples) < 0.38)[0]
    identity_indices = np.unique(np.concatenate([identity_indices, fraud_indices]))
    
    id_sub = transaction_ids[identity_indices]
    device_types = rng.choice(['mobile', 'desktop'], size=len(id_sub), p=[0.72, 0.28])
    device_infos = rng.choice(
        ['Samsung SM-G998B', 'Apple iPhone14,2', 'Redmi Note 11', 'OnePlus GM1911', 'Windows 10', 'Macintosh'],
        size=len(id_sub),
        p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
    )
    os_versions = rng.choice(
        ['Android 13', 'Android 12', 'iOS 16.5', 'iOS 15.4', 'Windows 10', 'Mac OS X 10_15_7'],
        size=len(id_sub),
        p=[0.30, 0.20, 0.25, 0.10, 0.10, 0.05]
    )
    browsers = rng.choice(
        ['chrome 114.0', 'safari 16.0', 'samsung browser 20.0', 'edge 114.0', 'firefox 113.0'],
        size=len(id_sub),
        p=[0.45, 0.25, 0.15, 0.10, 0.05]
    )
    
    df_identity = pd.DataFrame({
        'TransactionID': id_sub,
        'DeviceType': device_types,
        'DeviceInfo': device_infos,
        'id_30': os_versions,
        'id_31': browsers
    })
    
    return df_transaction, df_identity
