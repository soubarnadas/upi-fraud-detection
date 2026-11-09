"""
FastAPI application for real-time UPI Fraud Scoring and Explainability.
"""

from fastapi import FastAPI, HTTPException
from typing import List
from src.service.schemas import TransactionPayload, BatchTransactionPayload, FraudScoreResponse
from src.service.pipeline import ServingPipeline

app = FastAPI(
    title="UPI Fraud Detection & Explainability API",
    description="Real-time pre-authorization fraud detection service with probability calibration, 0-100 risk scoring, and SHAP explanations.",
    version="1.0.0"
)

pipeline = ServingPipeline()


@app.on_event("startup")
def startup_event():
    pipeline.initialize_with_defaults()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "upi-fraud-detection",
        "pipeline_ready": pipeline.is_ready,
        "features_loaded": len(pipeline.feature_extractor.feature_columns) if pipeline.is_ready else 0
    }


@app.post("/score", response_model=FraudScoreResponse)
def score_transaction(payload: TransactionPayload):
    try:
        response = pipeline.score(payload)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_score", response_model=List[FraudScoreResponse])
def batch_score(payload: BatchTransactionPayload):
    try:
        results = [pipeline.score(tx) for tx in payload.transactions]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
def get_model_info():
    if not pipeline.is_ready:
        pipeline.initialize_with_defaults()
    return {
        "model_name": pipeline.model.name,
        "calibrator": pipeline.calibrator.__class__.__name__,
        "review_budget_threshold": round(pipeline.threshold_mgr.operational_threshold, 4),
        "feature_count": len(pipeline.feature_extractor.feature_columns),
        "features": pipeline.feature_extractor.feature_columns
    }
