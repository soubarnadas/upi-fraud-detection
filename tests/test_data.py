"""
Tests for synthetic data generation and temporal splitting.
"""

import unittest
import numpy as np
from src.data.synthetic_paysim import generate_synthetic_paysim
from src.data.synthetic_ieee import generate_synthetic_ieee
from src.data.temporal_split import TemporalSplitter


class TestDataGeneration(unittest.TestCase):
    def test_paysim_generation(self):
        df = generate_synthetic_paysim(n_samples=500, fraud_rate=0.02, random_state=1)
        self.assertEqual(len(df), 500)
        self.assertIn('isFraud', df.columns)
        self.assertIn('step', df.columns)
        self.assertTrue(df['step'].is_monotonic_increasing)
        self.assertGreater(df['isFraud'].sum(), 0)
        # Verify fraud only in TRANSFER or CASH_OUT
        fraud_types = df[df['isFraud'] == 1]['type'].unique()
        for t in fraud_types:
            self.assertIn(t, ['TRANSFER', 'CASH_OUT'])

    def test_ieee_generation(self):
        df_tx, df_id = generate_synthetic_ieee(n_samples=400, fraud_rate=0.05, random_state=1)
        self.assertEqual(len(df_tx), 400)
        self.assertIn('TransactionDT', df_tx.columns)
        self.assertTrue(df_tx['TransactionDT'].is_monotonic_increasing)
        self.assertGreater(df_tx['isFraud'].sum(), 0)
        self.assertIn('TransactionID', df_id.columns)
        self.assertIn('DeviceType', df_id.columns)

    def test_temporal_splitter_no_leakage(self):
        df = generate_synthetic_paysim(n_samples=1000, random_state=1)
        splitter = TemporalSplitter(train_ratio=0.6, cal_ratio=0.1, val_ratio=0.1, test_ratio=0.2)
        splits = splitter.split(df)
        
        self.assertIn('train', splits)
        self.assertIn('cal', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)
        
        # Verify strictly monotonic time bounds
        train_max = splits['train']['step'].max()
        cal_min = splits['cal']['step'].min()
        cal_max = splits['cal']['step'].max()
        val_min = splits['val']['step'].min()
        val_max = splits['val']['step'].max()
        test_min = splits['test']['step'].min()
        
        self.assertLessEqual(train_max, cal_min)
        self.assertLessEqual(cal_max, val_min)
        self.assertLessEqual(val_max, test_min)


if __name__ == '__main__':
    unittest.main()
