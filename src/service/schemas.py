"""
Pydantic schemas for the real-time UPI Fraud Detection API.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class TransactionPayload(BaseModel):
    step: int = Field(default=1, description="Hour index of transaction (1-720)")
    type: str = Field(default="TRANSFER", description="Transaction type: TRANSFER, CASH_OUT, PAYMENT, etc.")
    amount: float = Field(..., ge=0.01, description="Transaction amount in INR")
    nameOrig: str = Field(..., description="Sender account/VPA identifier")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Pre-transaction sender balance")
    nameDest: str = Field(..., description="Beneficiary account/VPA identifier")
    oldbalanceDest: float = Field(default=0.0, ge=0.0, description="Pre-transaction beneficiary balance")
    device_type: Optional[str] = Field(default="mobile", description="Device type: mobile, desktop, unknown")
    channel: Optional[str] = Field(default="UPI_APP", description="Payment channel")


class BatchTransactionPayload(BaseModel):
    transactions: List[TransactionPayload]


class FeatureContribution(BaseModel):
    feature: str
    shap_contribution: float
    feature_value: float


class FraudScoreResponse(BaseModel):
    transaction_ref: str
    risk_score: int = Field(..., ge=0, le=100, description="0-100 normalized risk score")
    risk_tier: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    calibrated_probability: float
    action: str = Field(..., description="APPROVE_TRANSACTION, QUEUE_FOR_MANUAL_REVIEW, TRIGGER_STEP_UP_AUTH_AND_HOLD, BLOCK_AND_FREEZE")
    rule_flags: List[str]
    top_shap_factors: List[FeatureContribution]
    latency_ms: float
