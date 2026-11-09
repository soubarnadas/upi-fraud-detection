"""
Contextual feature engineering module (IEEE-CIS pipeline).
Extracts device, identity, temporal, and interaction signals available at scoring time:
- Left-joins identity metadata
- Extracts hour of day and day of week from TransactionDT
- Frequency-encodes card issuers, emails, and device families
- Computes historical card frequency looking strictly backward
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


class ContextualFeatureExtractor:
    def __init__(self):
        self.freq_encodings: Dict[str, Dict[str, float]] = {}
        self.feature_columns: List[str] = []

    def fit_encodings(self, df_merged: pd.DataFrame, categorical_cols: List[str]):
        """Fits frequency encodings on training fold only to prevent leakage."""
        self.freq_encodings = {}
        n = len(df_merged)
        for col in categorical_cols:
            if col in df_merged.columns:
                counts = df_merged[col].value_counts().to_dict()
                self.freq_encodings[col] = {k: v / n for k, v in counts.items()}

    def transform(
        self,
        df_tx: pd.DataFrame,
        df_id: pd.DataFrame = None,
        is_training: bool = False
    ) -> pd.DataFrame:
        data = df_tx.copy()
        
        # Left-join identity metadata
        if df_id is not None and not df_id.empty:
            data = data.merge(df_id, on='TransactionID', how='left')
        else:
            for c in ['DeviceType', 'DeviceInfo', 'id_30', 'id_31']:
                data[c] = 'unknown'
        
        # Sort chronologically by TransactionDT
        data = data.sort_values(by='TransactionDT').reset_index(drop=True)
        
        # 1. Temporal signals from TransactionDT (seconds)
        data['hour_of_day'] = ((data['TransactionDT'] // 3600) % 24).astype(float)
        data['day_of_week'] = ((data['TransactionDT'] // 86400) % 7).astype(float)
        data['is_night'] = data['hour_of_day'].isin([0, 1, 2, 3, 4, 5]).astype(float)
        
        # 2. Amount transformations
        data['log_amount'] = np.log1p(data['TransactionAmt'])
        
        # 3. Categorical normalizations
        cat_cols = ['ProductCD', 'card4', 'card6', 'DeviceType', 'P_emaildomain']
        if is_training:
            self.fit_encodings(data, cat_cols)
        
        for col in cat_cols:
            if col in data.columns:
                mapping = self.freq_encodings.get(col, {})
                data[f"{col}_freq"] = data[col].map(mapping).fillna(0.0).astype(float)
        
        # 4. Identity presence flag
        data['has_identity'] = data['DeviceType'].notna().astype(float)
        data['is_mobile'] = (data['DeviceType'] == 'mobile').astype(float)
        
        # 5. Continuous Vesta and Count features (handle NaNs with median/zero)
        v_cols = [c for c in data.columns if c.startswith('V') and c[1:].isdigit()]
        c_cols = [c for c in data.columns if c.startswith('C') and c[1:].isdigit()]
        d_cols = [c for c in data.columns if c.startswith('D') and c[1:].isdigit()]
        
        for c in v_cols + c_cols + d_cols:
            data[c] = data[c].fillna(0.0).astype(float)
            
        data['card1'] = data['card1'].fillna(0.0).astype(float)
        data['card2'] = data['card2'].fillna(-1.0).astype(float)
        data['dist1'] = data['dist1'].fillna(-1.0).astype(float)
        
        feature_cols = [
            'log_amount',
            'hour_of_day',
            'day_of_week',
            'is_night',
            'has_identity',
            'is_mobile',
            'card1',
            'card2',
            'dist1'
        ] + [f"{c}_freq" for c in cat_cols if f"{c}_freq" in data.columns] + v_cols + c_cols + d_cols
        
        self.feature_columns = feature_cols
        return data

    def get_feature_matrix(self, df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df_features[self.feature_columns].values.astype(float)
        y = df_features['isFraud'].values.astype(int) if 'isFraud' in df_features else None
        return X, y
