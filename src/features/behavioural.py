"""
Behavioural feature engineering module (PaySim pipeline).
Transforms raw transaction logs into behavioural signals available at scoring time:
- Log-scaled transaction amounts
- Amount-to-balance ratios and account draining signals
- Temporal rolling velocity features (1h, 6h, 24h) looking backward only
- Counterparty novelty (first-time receiver detection)
- Diurnal time-of-day risk signals
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


class BehaviouralFeatureExtractor:
    def __init__(self, filter_transfer_cashout: bool = True):
        self.filter_transfer_cashout = filter_transfer_cashout
        self.fitted_seen_counterparties: Set[Tuple[str, str]] = set()
        self.feature_columns: List[str] = []

    def fit_counterparties(self, train_df: pd.DataFrame):
        """Pre-populates seen sender-receiver pairs from historical training data."""
        self.fitted_seen_counterparties = set(
            zip(train_df['nameOrig'], train_df['nameDest'])
        )

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Extracts behavioural features without temporal lookahead.
        """
        data = df.copy()
        
        # Ensure chronological ordering
        data = data.sort_values(by='step').reset_index(drop=True)
        
        # Filter for relevant types if specified
        if self.filter_transfer_cashout:
            data = data[data['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
            data = data.reset_index(drop=True)
        
        # 1. Amount transformations
        data['log_amount'] = np.log1p(data['amount'])
        data['log_oldbalanceOrg'] = np.log1p(data['oldbalanceOrg'])
        data['log_oldbalanceDest'] = np.log1p(data['oldbalanceDest'])
        
        # 2. Balance ratios and account depletion
        data['amount_to_balance_orig'] = data['amount'] / (data['oldbalanceOrg'] + 1.0)
        data['is_account_draining'] = (
            (data['amount'] >= data['oldbalanceOrg'] * 0.95) & (data['oldbalanceOrg'] > 0)
        ).astype(float)
        data['dest_starts_zero'] = (data['oldbalanceDest'] == 0.0).astype(float)
        
        # 3. Time signals
        data['hour_of_day'] = (data['step'] % 24).astype(float)
        data['is_night_txn'] = data['hour_of_day'].isin([0, 1, 2, 3, 4, 5]).astype(float)
        
        # 4. Type one-hot encoding
        data['is_transfer'] = (data['type'] == 'TRANSFER').astype(float)
        data['is_cash_out'] = (data['type'] == 'CASH_OUT').astype(float)
        
        # 5. Temporal rolling velocities (1h, 6h, 24h) and counterparty novelty
        # Computed sequentially to guarantee no future leakage
        n = len(data)
        v_1h = np.zeros(n, dtype=float)
        v_6h = np.zeros(n, dtype=float)
        v_24h = np.zeros(n, dtype=float)
        sum_amt_24h = np.zeros(n, dtype=float)
        new_cp = np.zeros(n, dtype=float)
        
        history_orig_time: Dict[str, List[float]] = {}
        history_orig_amt: Dict[str, List[float]] = {}
        seen_pairs = set(self.fitted_seen_counterparties) if not is_training else set()
        
        steps = data['step'].values
        amounts = data['amount'].values
        origs = data['nameOrig'].values
        dests = data['nameDest'].values
        
        for i in range(n):
            s = steps[i]
            amt = amounts[i]
            orig = origs[i]
            dest = dests[i]
            
            # Check counterparty novelty
            pair = (orig, dest)
            if pair not in seen_pairs:
                new_cp[i] = 1.0
                seen_pairs.add(pair)
            else:
                new_cp[i] = 0.0
            
            # Check rolling window
            if orig in history_orig_time:
                past_times = history_orig_time[orig]
                past_amts = history_orig_amt[orig]
                
                # Count in last 1h, 6h, 24h
                c_1 = 0
                c_6 = 0
                c_24 = 0
                s_24 = 0.0
                for pt, pa in zip(reversed(past_times), reversed(past_amts)):
                    diff = s - pt
                    if diff <= 1:
                        c_1 += 1
                    if diff <= 6:
                        c_6 += 1
                    if diff <= 24:
                        c_24 += 1
                        s_24 += pa
                    else:
                        break  # since list is ordered
                v_1h[i] = c_1
                v_6h[i] = c_6
                v_24h[i] = c_24
                sum_amt_24h[i] = s_24
                
                past_times.append(s)
                past_amts.append(amt)
            else:
                history_orig_time[orig] = [s]
                history_orig_amt[orig] = [amt]
        
        data['velocity_1h'] = v_1h
        data['velocity_6h'] = v_6h
        data['velocity_24h'] = v_24h
        data['amount_sum_24h'] = np.log1p(sum_amt_24h)
        data['new_counterparty'] = new_cp
        
        if is_training:
            self.fitted_seen_counterparties = seen_pairs
            
        feature_cols = [
            'log_amount',
            'log_oldbalanceOrg',
            'log_oldbalanceDest',
            'amount_to_balance_orig',
            'is_account_draining',
            'dest_starts_zero',
            'hour_of_day',
            'is_night_txn',
            'is_transfer',
            'is_cash_out',
            'velocity_1h',
            'velocity_6h',
            'velocity_24h',
            'amount_sum_24h',
            'new_counterparty'
        ]
        self.feature_columns = feature_cols
        
        return data

    def get_feature_matrix(self, df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df_features[self.feature_columns].values.astype(float)
        y = df_features['isFraud'].values.astype(int) if 'isFraud' in df_features else None
        return X, y
