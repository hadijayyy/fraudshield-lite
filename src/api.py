"""
FastAPI application for serving FraudShield fraud predictions.

Exposes a prediction endpoint (``POST /predict``) and a health-check
endpoint (``GET /health``). Loads the trained model and config on startup.

Model path: ``models/xgboost_model.json`` (configurable via config.yaml).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb

from .utils import load_config, setup_logging

logger = logging.getLogger(__name__)

# Default model path — relative to the project root (where the app runs).
MODEL_PATH = "models/xgboost_model.json"

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Schema for a single prediction request."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Account / user identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: str = Field(..., description="ISO-8601 transaction timestamp")
    # Additional features can be added as needed
    extra_features: Dict[str, float] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    """Schema for the prediction response."""

    transaction_id: str
    fraud_probability: float
    prediction: int  # 1 = fraud, 0 = legitimate
    threshold: float
    decision: str  # "approve" | "review" | "block"


class HealthResponse(BaseModel):
    """Schema for the health-check response."""

    status: str
    model_version: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FraudShield API",
    description="Anti-fraud & AML scoring engine",
    version="1.0.0",
)

# Global (initialised at startup)
config: Optional[Dict[str, Any]] = None
model: Any = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load config and model on application startup."""
    global config, model
    config = load_config()
    setup_logging()

    # Load XGBoost model from models/xgboost_model.json
    model_path = Path(config.get("model", {}).get("path", MODEL_PATH))
    if model_path.exists():
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            logger.info("Model loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", model_path, exc)
            model = None
    else:
        logger.warning("Model file not found at %s; running without model", model_path)
        model = None

    logger.info("FraudShield API started (model loaded: %s)", model is not None)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return the current health status of the API and its underlying model."""
    model_version = config.get("model", {}).get("version", "unknown") if config else "unknown"
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_version=model_version,
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Run the fraud detection model on a single transaction.

    Parameters
    ----------
    request : PredictRequest
        Transaction features in the request body.

    Returns
    -------
    PredictResponse
        Fraud probability, binary prediction, threshold used, and decision.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a feature vector from the request.
    # The model expects 29 features matching the training schema.
    # We build a DataFrame with the expected feature names (zero-filled)
    # and fill in whatever the request provides via extra_features.
    threshold = config.get("thresholds", {}).get("default", 0.45) if config else 0.45

    try:
        # Retrieve expected feature names from the model dump
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            # Fallback: try to read from model's learner attribute
            booster = model.get_booster()
            fmap = booster.get_score(importance_type="weight")
            # If we can't get feature names, fall through with a simple
            # prediction on the raw amount
            feature_names = list(fmap.keys()) if fmap else None

        if feature_names is not None and len(feature_names) > 0:
            # Build a zero-filled feature vector
            feature_dict = {name: 0.0 for name in feature_names}
            # Fill in provided extra features
            feature_dict.update(request.extra_features)
            # Override amount if it's in the features
            if "amount" in feature_dict:
                feature_dict["amount"] = request.amount

            x_input = pd.DataFrame([feature_dict])[feature_names].values
            fraud_probability = float(model.predict_proba(x_input)[0, 1])
        else:
            # No feature metadata available — use amount as a rough signal
            fraud_probability = 1.0 / (1.0 + np.exp(-request.amount / 1e6))

        prediction = 1 if fraud_probability >= threshold else 0

        if prediction == 1 and fraud_probability >= threshold + 0.15:
            decision = "block"
        elif prediction == 1:
            decision = "review"
        else:
            decision = "approve"

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return PredictResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(fraud_probability, 6),
        prediction=prediction,
        threshold=threshold,
        decision=decision,
    )
