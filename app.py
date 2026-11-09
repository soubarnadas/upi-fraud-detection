"""
Flask Web Application for Real-Time UPI Fraud Detection & Explainability.
Provides interactive fraud scoring, scenario testing, and SHAP visual explanations.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, render_template, request, jsonify
from src.service.pipeline import ServingPipeline
from src.service.schemas import TransactionPayload

app = Flask(__name__, template_folder="templates")
pipeline = ServingPipeline()
pipeline.initialize_with_defaults()

# Dummy test scenarios for quick evaluation
SCENARIOS = {
    "drain": {
        "step": 3,
        "type": "TRANSFER",
        "amount": 95000.0,
        "nameOrig": "C1092841",
        "oldbalanceOrg": 96000.0,
        "nameDest": "C9982144",
        "oldbalanceDest": 0.0,
        "device_type": "mobile",
        "description": "Critical Risk: Complete account balance drained to newly created mule account at 3:00 AM"
    },
    "normal": {
        "step": 14,
        "type": "PAYMENT",
        "amount": 350.0,
        "nameOrig": "C2018472",
        "oldbalanceOrg": 12500.0,
        "nameDest": "M5001294",
        "oldbalanceDest": 45000.0,
        "device_type": "mobile",
        "description": "Low Risk: Small everyday merchant grocery payment during daytime hours"
    },
    "night_burst": {
        "step": 2,
        "type": "TRANSFER",
        "amount": 48000.0,
        "nameOrig": "C3182940",
        "oldbalanceOrg": 52000.0,
        "nameDest": "C8821943",
        "oldbalanceDest": 1200.0,
        "device_type": "unknown",
        "description": "High Risk: High-value payment to unverified counterparty during off-hours from unknown device"
    }
}


@app.route("/", methods=["GET"])
def index():
    scenario_key = "drain"
    if request.args:
        # Check query params for scenario selection
        for k in request.args:
            if k == "scenario":
                val = request.args[k]
                scenario_key = val[0] if isinstance(val, list) else val
                break

    selected_tx = SCENARIOS.get(scenario_key, SCENARIOS["drain"])
    
    payload = TransactionPayload(
        step=selected_tx["step"],
        type=selected_tx["type"],
        amount=selected_tx["amount"],
        nameOrig=selected_tx["nameOrig"],
        oldbalanceOrg=selected_tx["oldbalanceOrg"],
        nameDest=selected_tx["nameDest"],
        oldbalanceDest=selected_tx["oldbalanceDest"],
        device_type=selected_tx["device_type"]
    )
    result = pipeline.score(payload)
    return render_template("index.html", tx=selected_tx, result=result, scenarios=SCENARIOS, current_scenario=scenario_key)


@app.route("/score", methods=["POST"])
def score():
    try:
        if request.json:
            data = request.json
        else:
            data = request.form

        payload = TransactionPayload(
            step=int(data.get("step", 1)),
            type=str(data.get("type", "TRANSFER")),
            amount=float(data.get("amount", 100.0)),
            nameOrig=str(data.get("nameOrig", "C1000000")),
            oldbalanceOrg=float(data.get("oldbalanceOrg", 1000.0)),
            nameDest=str(data.get("nameDest", "C2000000")),
            oldbalanceDest=float(data.get("oldbalanceDest", 0.0)),
            device_type=str(data.get("device_type", "mobile"))
        )

        result = pipeline.score(payload)

        # Return JSON if requested via API, else re-render template
        if request.is_json() or request.environ.get("HTTP_ACCEPT", "").startswith("application/json"):
            return jsonify(result.dict())
        else:
            return render_template("index.html", tx=payload.dict(), result=result, scenarios=SCENARIOS, current_scenario="custom")

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "framework": "Flask (Micro WSGI Engine)",
        "model_loaded": pipeline.is_ready,
        "features": len(pipeline.feature_extractor.feature_columns)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
