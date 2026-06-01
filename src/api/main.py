"""
main.py
-------
FastAPI application for the Bati Bank credit risk scoring service.

Endpoints:
  GET  /health          - liveness check
  POST /predict         - score a single customer (Task 6 requirement)
  POST /predict/batch   - score multiple customers
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.pydantic_models import CustomerFeatures, HealthResponse, ScoringResponse
from src.predict import load_model, predict_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/credit_risk_model.joblib")

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    try:
        _model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except FileNotFoundError as e:
        logger.warning("Model not found: %s. /predict will return 503.", e)
    yield
    _model = None


app = FastAPI(
    title="Bati Bank Credit Risk Scoring API",
    description=(
        "Scores new loan applicants using a machine learning model trained on "
        "eCommerce transaction data. Returns a risk probability, credit score "
        "(300-850), and loan recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Liveness probe."""
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=ScoringResponse, tags=["Scoring"])
def predict(features: CustomerFeatures):
    """
    Score a single customer and return risk probability, credit score,
    and loan recommendations.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure the model file exists and restart.",
        )
    try:
        result = predict_single(features.model_dump(), model=_model)
        return ScoringResponse(**result)
    except Exception as e:
        logger.exception("Scoring error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[ScoringResponse], tags=["Scoring"])
def predict_batch(customers: list[CustomerFeatures]):
    """Score a batch of customers (max 1000)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000.")
    try:
        results = [predict_single(c.model_dump(), model=_model) for c in customers]
        return [ScoringResponse(**r) for r in results]
    except Exception as e:
        logger.exception("Batch scoring error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
