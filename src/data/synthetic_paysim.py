"""
Synthetic PaySim dataset generator for UPI behavioural fraud detection.
Mimics the PaySim mobile money transfer logs schema and temporal dynamics:
- step: hour index (1 to 720, representing 30 days)
- type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
- amount: transaction amount
- nameOrig, oldbalanceOrg, newbalanceOrig
- nameDest, oldbalanceDest, newbalanceDest
- isFraud: binary label (fraud concentrated in TRANSFER and CASH_OUT)
- isFlaggedFraud: naive threshold rule (amount > 200,000)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_synthetic_paysim(
    n_samples: int = 25000,
    fraud_rate: float = 0.003,
    random_state: int = 42
) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    
    # Simulate 30 days of hourly steps (1 to 720)
    # Diurnal volume pattern: lower between 1-5 AM, peak at 12 PM and 8 PM
    hourly_weights = 1.0 + 0.8 * np.sin((np.arange(24) - 6) * np.pi / 12)
    hourly_weights = np.clip(hourly_weights, 0.15, 2.0)
    hourly_weights = hourly_weights / hourly_weights.sum()
    
    steps = []
    for day in range(30):
        day_steps = rng.choice(np.arange(1, 25), size=int(n_samples / 30), p=hourly_weights)
        steps.extend(day * 24 + day_steps)
    
    # Adjust length
    if len(steps) < n_samples:
        steps.extend(rng.randint(1, 721, size=n_samples - len(steps)))
    steps = np.array(steps[:n_samples])
    steps.sort()
    
    # Types distribution typical in UPI / mobile money
    types_pool = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
    types_prob = [0.35, 0.25, 0.25, 0.12, 0.03]
    types = rng.choice(types_pool, size=n_samples, p=types_prob)
    
    # Customer and Merchant IDs
    n_customers = int(n_samples * 0.4)
    n_merchants = int(n_samples * 0.1)
    customer_ids = [f"C{1000000 + i}" for i in range(n_customers)]
    merchant_ids = [f"M{5000000 + i}" for i in range(n_merchants)]
    
    nameOrig = rng.choice(customer_ids, size=n_samples)
    
    # Destinations: merchants for PAYMENT, customers for TRANSFER / CASH_OUT / CASH_IN
    nameDest = []
    for t in types:
        if t == 'PAYMENT':
            nameDest.append(rng.choice(merchant_ids))
        else:
            nameDest.append(rng.choice(customer_ids))
    nameDest = np.array(nameDest)
    
    # Amounts: log-normal distribution
    amounts = np.exp(rng.normal(loc=7.5, scale=1.4, size=n_samples))
    amounts = np.round(np.clip(amounts, 10.0, 500000.0), 2)
    
    # Base balances
    oldbalanceOrg = np.exp(rng.normal(loc=8.5, scale=1.6, size=n_samples))
    oldbalanceOrg = np.round(np.clip(oldbalanceOrg, 0.0, 2000000.0), 2)
    
    # Legitimate state transitions
    newbalanceOrig = np.maximum(0.0, oldbalanceOrg - amounts)
    
    oldbalanceDest = np.exp(rng.normal(loc=8.0, scale=1.5, size=n_samples))
    oldbalanceDest = np.round(np.clip(oldbalanceDest, 0.0, 3000000.0), 2)
    newbalanceDest = oldbalanceDest + amounts
    
    # Initialize fraud labels
    isFraud = np.zeros(n_samples, dtype=int)
    
    # Target fraud to TRANSFER and CASH_OUT (matching PaySim characteristics)
    eligible_indices = np.where(np.isin(types, ['TRANSFER', 'CASH_OUT']))[0]
    n_fraud = int(len(eligible_indices) * fraud_rate)
    fraud_indices = rng.choice(eligible_indices, size=n_fraud, replace=False)
    
    for idx in fraud_indices:
        isFraud[idx] = 1
        # Fraud patterns:
        # 1. Account emptying / draining
        oldbalanceOrg[idx] = amounts[idx] * rng.uniform(0.98, 1.05)
        newbalanceOrig[idx] = 0.0  # complete account drain
        # 2. Mule / newly seen destination account often starts with 0 balance
        if rng.rand() < 0.7:
            oldbalanceDest[idx] = 0.0
            newbalanceDest[idx] = 0.0  # mule immediately cashes out off-ledger
        # 3. High amount spike
        if rng.rand() < 0.4:
            amounts[idx] = rng.uniform(150000.0, 480000.0)
            oldbalanceOrg[idx] = amounts[idx]
    
    # Business rule isFlaggedFraud (transfer > 200,000)
    isFlaggedFraud = np.where((types == 'TRANSFER') & (amounts > 200000.0), 1, 0)
    
    df = pd.DataFrame({
        'step': steps,
        'type': types,
        'amount': amounts,
        'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'nameDest': nameDest,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFraud': isFraud,
        'isFlaggedFraud': isFlaggedFraud
    })
    
    return df


if __name__ == "__main__":
    df = generate_synthetic_paysim(5000)
    print(f"Generated PaySim dataset: shape={df.shape}, fraud_count={df['isFraud'].sum()}")
