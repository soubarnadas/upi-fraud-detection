"""
Target leakage audit module for real-time UPI fraud detection.
Audits features for scoring-time availability, post-settlement leakage,
future lookahead, and identifier memorization.
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import pandas as pd


@dataclass
class ColumnAuditRecord:
    column_name: str
    available_at_scoring: bool
    status: str  # 'SAFE', 'LEAKAGE_TARGET', 'LEAKAGE_SETTLEMENT', 'MEMORIZATION_RISK', 'RULE_FLAG'
    reason: str
    recommended_action: str  # 'KEEP', 'DROP', 'TRANSFORM'


class LeakageAuditor:
    def __init__(self, target_col: str = 'isFraud'):
        self.target_col = target_col
        
        # Domain knowledge mappings for PaySim & IEEE-CIS & UPI ecosystems
        self.known_rules = {
            'isFraud': ColumnAuditRecord(
                column_name='isFraud',
                available_at_scoring=False,
                status='LEAKAGE_TARGET',
                reason='Target ground truth label. Unknown at scoring time.',
                recommended_action='DROP'
            ),
            'isFlaggedFraud': ColumnAuditRecord(
                column_name='isFlaggedFraud',
                available_at_scoring=False,
                status='RULE_FLAG',
                reason='Static heuristic rule flag evaluated for compliance/reporting. Excluded to avoid circular learning.',
                recommended_action='DROP'
            ),
            'nameOrig': ColumnAuditRecord(
                column_name='nameOrig',
                available_at_scoring=True,
                status='MEMORIZATION_RISK',
                reason='Raw customer account identifier with high cardinality; leads to overfitting.',
                recommended_action='TRANSFORM'
            ),
            'nameDest': ColumnAuditRecord(
                column_name='nameDest',
                available_at_scoring=True,
                status='MEMORIZATION_RISK',
                reason='Raw destination identifier; should be replaced with counterparty novelty and velocity.',
                recommended_action='TRANSFORM'
            ),
            'TransactionID': ColumnAuditRecord(
                column_name='TransactionID',
                available_at_scoring=True,
                status='MEMORIZATION_RISK',
                reason='Arbitrary transaction sequence key.',
                recommended_action='DROP'
            ),
            'newbalanceOrig': ColumnAuditRecord(
                column_name='newbalanceOrig',
                available_at_scoring=False,
                status='LEAKAGE_SETTLEMENT',
                reason='Post-settlement origin balance after completion. In pre-auth scoring, only oldbalanceOrg is known.',
                recommended_action='TRANSFORM'
            ),
            'newbalanceDest': ColumnAuditRecord(
                column_name='newbalanceDest',
                available_at_scoring=False,
                status='LEAKAGE_SETTLEMENT',
                reason='Post-settlement receiver balance. Unavailable prior to transaction settlement.',
                recommended_action='TRANSFORM'
            )
        }

    def audit_dataframe(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, ColumnAuditRecord]:
        results: Dict[str, ColumnAuditRecord] = {}
        for col in df.columns:
            if col in self.known_rules:
                results[col] = self.known_rules[col]
            elif col == self.target_col:
                results[col] = ColumnAuditRecord(
                    column_name=col,
                    available_at_scoring=False,
                    status='LEAKAGE_TARGET',
                    reason='Target variable.',
                    recommended_action='DROP'
                )
            else:
                # Default assume available unless flagged
                results[col] = ColumnAuditRecord(
                    column_name=col,
                    available_at_scoring=True,
                    status='SAFE',
                    reason='Available at pre-authorization scoring time.',
                    recommended_action='KEEP'
                )
        return results

    def generate_report(self, audit_records: Dict[str, ColumnAuditRecord]) -> pd.DataFrame:
        data = []
        for col, rec in audit_records.items():
            data.append({
                'Column': rec.column_name,
                'Available At Scoring': rec.available_at_scoring,
                'Status': rec.status,
                'Action': rec.recommended_action,
                'Rationale': rec.reason
            })
        return pd.DataFrame(data)
