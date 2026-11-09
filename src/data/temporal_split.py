"""
Temporal split module for leakage-free chronological partitioning.
Implements the 4-window time-ordered partition:
1. Train window: model parameter learning
2. Calibration window: probability calibration (Platt scaling / Isotonic)
3. Validation window: hyperparameter tuning & threshold selection
4. Test window: held-out final evaluation
"""

from typing import Dict, Tuple
import pandas as pd


class TemporalSplitter:
    def __init__(
        self,
        time_col: str = 'step',
        train_ratio: float = 0.65,
        cal_ratio: float = 0.10,
        val_ratio: float = 0.10,
        test_ratio: float = 0.15
    ):
        total = train_ratio + cal_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 1e-5, f"Ratios must sum to 1.0, got {total}"
        self.time_col = time_col
        self.train_ratio = train_ratio
        self.cal_ratio = cal_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Sorts df chronologically by time_col and splits into four windows.
        Guarantees no temporal leakage or shuffling.
        """
        df_sorted = df.sort_values(by=self.time_col).reset_index(drop=True)
        n = len(df_sorted)
        
        n_train = int(n * self.train_ratio)
        n_cal = int(n * (self.train_ratio + self.cal_ratio))
        n_val = int(n * (self.train_ratio + self.cal_ratio + self.val_ratio))
        
        train_df = df_sorted.iloc[:n_train].copy()
        cal_df = df_sorted.iloc[n_train:n_cal].copy()
        val_df = df_sorted.iloc[n_cal:n_val].copy()
        test_df = df_sorted.iloc[n_val:].copy()
        
        return {
            'train': train_df,
            'cal': cal_df,
            'val': val_df,
            'test': test_df
        }


def print_split_summary(splits: Dict[str, pd.DataFrame], label_col: str = 'isFraud', time_col: str = 'step'):
    for name, split_df in splits.items():
        n = len(split_df)
        n_pos = split_df[label_col].sum() if label_col in split_df else 0
        rate = (n_pos / n) * 100.0 if n > 0 else 0.0
        t_min = split_df[time_col].min()
        t_max = split_df[time_col].max()
        print(f"[{name.upper()}] Rows: {n:,} | Fraud: {n_pos:,} ({rate:.2f}%) | Time Range: [{t_min} -> {t_max}]")
