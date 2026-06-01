"""
predict.py
----------
Inference utilities for the credit risk model.

Converts raw risk probability to credit score (300-850 scale)
and provides loan amount/duration recommendations.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/credit_risk_model.joblib")

SCORE_MIN = 300
SCORE_MAX = 850

LOAN_TIERS = [
    {"max_prob": 0.20, "label": "low",    "max_amount": 50_000, "max_months": 36},
    {"max_prob": 0.50, "label": "medium", "max_amount": 20_000, "max_months": 24},
    {"max_prob": 1.00, "label": "high",   "max_amount":  5_000, "max_months": 12},
]

# Feature columns — must match training order
FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalTransactionAmount",
    "AvgTransactionAmount",
    "TransactionCount",
    "StdTransactionAmount",
    "FraudRate",
    "UniqueProducts",
    "UniqueChannels",
    "NightTxnRatio",
]

# Fallback for old model trained on fewer features
FEATURE_COLS_LEGACY = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgTransactionValue",
    "StdTransactionValue",
    "FraudRate",
    "UniqueProducts",
    "UniqueChannels",
    "NightTxnRatio",
]


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load a serialised sklearn pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    logger.info("Loading model from %s", path)
    return joblib.load(path)


def probability_to_credit_score(probability: float) -> int:
    """
    Convert a default probability to a credit score on a 300-850 scale.
    Linear mapping: p=0 -> 850, p=1 -> 300.
    """
    p = float(np.clip(probability, 0.0, 1.0))
    score = SCORE_MAX - p * (SCORE_MAX - SCORE_MIN)
    return int(np.clip(round(score), SCORE_MIN, SCORE_MAX))


def get_loan_recommendation(probability: float) -> dict:
    """Return recommended loan amount and duration based on risk tier."""
    for tier in LOAN_TIERS:
        if probability <= tier["max_prob"]:
            return {
                "risk_label": tier["label"],
                "recommended_max_amount": tier["max_amount"],
                "recommended_max_months": tier["max_months"],
            }
    return {"risk_label": "high", "recommended_max_amount": 0, "recommended_max_months": 0}


def predict_single(features: dict, model=None) -> dict:
    """
    Score a single customer.

    Parameters
    ----------
    features : dict with keys matching FEATURE_COLS
    model    : loaded sklearn pipeline (loaded from disk if None)
    """
    if model is None:
        model = load_model()

    # Try new feature set first, fall back to legacy
    if all(k in features for k in FEATURE_COLS):
        cols = FEATURE_COLS
    else:
        cols = FEATURE_COLS_LEGACY

    df = pd.DataFrame([features])[cols]
    probability = float(model.predict_proba(df)[0, 1])
    credit_score = probability_to_credit_score(probability)
    loan_rec = get_loan_recommendation(probability)

    return {
        "risk_probability": round(probability, 4),
        "credit_score": credit_score,
        **loan_rec,
    }


def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Score a batch of customers."""
    if model is None:
        model = load_model()

    cols = FEATURE_COLS if all(c in df.columns for c in FEATURE_COLS) else FEATURE_COLS_LEGACY
    probabilities = model.predict_proba(df[cols])[:, 1]
    df = df.copy()
    df["risk_probability"] = probabilities
    df["credit_score"] = df["risk_probability"].apply(probability_to_credit_score)
    loan_recs = df["risk_probability"].apply(get_loan_recommendation).apply(pd.Series)
    df = pd.concat([df, loan_recs], axis=1)
    return df
