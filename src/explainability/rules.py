"""
Deterministic rule engine and explainable output formatter.
Evaluates heuristic risk rules alongside machine learning models to provide
actionable reasons and suggested investigator next-steps.
"""

from typing import Dict, List, Any


class FraudRuleEngine:
    @staticmethod
    def evaluate_rules(raw_txn: Dict[str, Any], features: Dict[str, float]) -> List[str]:
        """
        Evaluates heuristic domain rules reflecting typical UPI fraud patterns.
        """
        flags = []
        
        # Rule 1: High amount sent to new counterparty at night
        if (
            features.get('new_counterparty', 0.0) == 1.0 and
            features.get('is_night_txn', 0.0) == 1.0 and
            features.get('log_amount', 0.0) > 10.0  # amount > ~22,000 INR
        ):
            flags.append("HIGH_AMOUNT_NEW_RECIPIENT_NIGHT")

        # Rule 2: Complete account draining event
        if features.get('is_account_draining', 0.0) == 1.0 and features.get('dest_starts_zero', 0.0) == 1.0:
            flags.append("ACCOUNT_DRAIN_TO_NEW_MULE")

        # Rule 3: High velocity burst within 1 hour
        if features.get('velocity_1h', 0.0) >= 4.0:
            flags.append("BURST_TRANSACTION_VELOCITY_1H")

        # Rule 4: Overdraft / balance mismatch
        if features.get('amount_to_balance_orig', 0.0) > 1.0:
            flags.append("ORIGIN_BALANCE_EXCEEDED")

        # Rule 5: Contextual device mismatch (IEEE-CIS)
        if features.get('has_identity', 1.0) == 0.0 and features.get('log_amount', 0.0) > 8.0:
            flags.append("UNVERIFIED_DEVICE_HIGH_VALUE")

        return flags

    @staticmethod
    def recommend_action(risk_score: int, rule_flags: List[str]) -> str:
        """Determines automated intervention policy for UPI switch / payment gateway."""
        if risk_score >= 85 or "ACCOUNT_DRAIN_TO_NEW_MULE" in rule_flags:
            return "BLOCK_AND_FREEZE"
        elif risk_score >= 60 or len(rule_flags) >= 2:
            return "TRIGGER_STEP_UP_AUTH_AND_HOLD"
        elif risk_score >= 35 or len(rule_flags) == 1:
            return "QUEUE_FOR_MANUAL_REVIEW"
        else:
            return "APPROVE_TRANSACTION"
