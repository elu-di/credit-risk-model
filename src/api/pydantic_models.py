"""
pydantic_models.py
------------------
Request and response schemas for the credit risk scoring API.
"""

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """
    Input features required to score a customer.
    All values are derived from the customer's transaction history.
    """

    Recency: float = Field(..., ge=0, description="Days since last transaction")
    Frequency: float = Field(..., ge=0, description="Total number of transactions")
    Monetary: float = Field(..., description="Total transaction amount (sum of debits)")
    TotalTransactionAmount: float = Field(..., description="Sum of all transaction amounts")
    AvgTransactionAmount: float = Field(..., description="Mean transaction amount")
    TransactionCount: float = Field(..., ge=0, description="Number of transactions")
    StdTransactionAmount: float = Field(..., ge=0, description="Std dev of transaction amounts")
    FraudRate: float = Field(..., ge=0, le=1, description="Fraction of transactions flagged as fraud")
    UniqueProducts: float = Field(..., ge=0, description="Number of distinct products purchased")
    UniqueChannels: float = Field(..., ge=0, description="Number of distinct channels used")
    NightTxnRatio: float = Field(..., ge=0, le=1, description="Fraction of transactions between 22:00-05:00")

    @field_validator("Frequency", "TransactionCount", "UniqueProducts", "UniqueChannels")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Count fields must be non-negative")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "Recency": 14,
                "Frequency": 23,
                "Monetary": 4500.0,
                "TotalTransactionAmount": 4500.0,
                "AvgTransactionAmount": 195.65,
                "TransactionCount": 23,
                "StdTransactionAmount": 88.3,
                "FraudRate": 0.0,
                "UniqueProducts": 7,
                "UniqueChannels": 2,
                "NightTxnRatio": 0.04,
            }
        }
    }


class ScoringResponse(BaseModel):
    """Credit risk scoring result."""

    risk_probability: float = Field(..., description="Probability of default (0-1)")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    risk_label: str = Field(..., description="Risk tier: low | medium | high")
    recommended_max_amount: float = Field(..., description="Recommended maximum loan amount")
    recommended_max_months: int = Field(..., description="Recommended maximum loan duration in months")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
