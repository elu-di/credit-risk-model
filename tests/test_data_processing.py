"""
test_data_processing.py
-----------------------
Unit tests for the feature engineering pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    assign_high_risk_label,
    compute_aggregate_features,
    compute_rfm,
    extract_time_features,
)
from src.predict import get_loan_recommendation, probability_to_credit_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Minimal synthetic transaction dataset."""
    np.random.seed(42)
    n = 200
    customers = [f"C{i:03d}" for i in range(20)]

    return pd.DataFrame({
        "TransactionId": [f"T{i}" for i in range(n)],
        "CustomerId": np.random.choice(customers, n),
        "ProductId": np.random.choice(["P1", "P2", "P3", "P4"], n),
        "ChannelId": np.random.choice(["web", "android", "ios"], n),
        "Amount": np.random.uniform(10, 500, n),
        "FraudResult": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "TransactionStartTime": pd.date_range("2023-01-01", periods=n, freq="6h"),
    })


# ---------------------------------------------------------------------------
# Time feature tests
# ---------------------------------------------------------------------------

class TestExtractTimeFeatures:
    def test_returns_dataframe(self, sample_transactions):
        result = extract_time_features(sample_transactions)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_transactions):
        result = extract_time_features(sample_transactions)
        expected = {"TransactionHour", "TransactionDay", "TransactionMonth", "TransactionYear", "IsNight"}
        assert expected.issubset(result.columns)

    def test_hour_range(self, sample_transactions):
        result = extract_time_features(sample_transactions)
        assert result["TransactionHour"].between(0, 23).all()

    def test_month_range(self, sample_transactions):
        result = extract_time_features(sample_transactions)
        assert result["TransactionMonth"].between(1, 12).all()

    def test_is_night_binary(self, sample_transactions):
        result = extract_time_features(sample_transactions)
        assert set(result["IsNight"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# RFM tests
# ---------------------------------------------------------------------------

class TestComputeRFM:
    def test_returns_dataframe(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        assert isinstance(rfm, pd.DataFrame)

    def test_has_required_columns(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        assert {"Recency", "Frequency", "Monetary"}.issubset(rfm.columns)

    def test_indexed_by_customer(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        assert rfm.index.name == "CustomerId"

    def test_recency_non_negative(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        assert (rfm["Recency"] >= 0).all()

    def test_frequency_positive(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        assert (rfm["Frequency"] > 0).all()

    def test_monetary_positive(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        assert (rfm["Monetary"] > 0).all()

    def test_customer_count(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        expected = sample_transactions["CustomerId"].nunique()
        assert len(rfm) == expected


# ---------------------------------------------------------------------------
# Aggregate feature tests
# ---------------------------------------------------------------------------

class TestComputeAggregateFeatures:
    def test_returns_dataframe(self, sample_transactions):
        agg = compute_aggregate_features(sample_transactions)
        assert isinstance(agg, pd.DataFrame)

    def test_has_required_columns(self, sample_transactions):
        agg = compute_aggregate_features(sample_transactions)
        expected = {
            "TotalTransactionAmount",
            "AvgTransactionAmount",
            "TransactionCount",
            "StdTransactionAmount",
            "FraudRate",
            "UniqueProducts",
            "UniqueChannels",
            "NightTxnRatio",
        }
        assert expected.issubset(agg.columns)

    def test_fraud_rate_bounded(self, sample_transactions):
        agg = compute_aggregate_features(sample_transactions)
        assert (agg["FraudRate"] >= 0).all()
        assert (agg["FraudRate"] <= 1).all()

    def test_night_ratio_bounded(self, sample_transactions):
        agg = compute_aggregate_features(sample_transactions)
        assert (agg["NightTxnRatio"] >= 0).all()
        assert (agg["NightTxnRatio"] <= 1).all()

    def test_std_no_nan(self, sample_transactions):
        agg = compute_aggregate_features(sample_transactions)
        assert not agg["StdTransactionAmount"].isna().any()

    def test_transaction_count_positive(self, sample_transactions):
        agg = compute_aggregate_features(sample_transactions)
        assert (agg["TransactionCount"] > 0).all()


# ---------------------------------------------------------------------------
# Proxy label tests (is_high_risk)
# ---------------------------------------------------------------------------

class TestAssignHighRiskLabel:
    def test_returns_dataframe(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        result = assign_high_risk_label(rfm)
        assert isinstance(result, pd.DataFrame)

    def test_has_is_high_risk_column(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        result = assign_high_risk_label(rfm)
        assert "is_high_risk" in result.columns

    def test_binary_label(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        result = assign_high_risk_label(rfm)
        assert set(result["is_high_risk"].unique()).issubset({0, 1})

    def test_has_both_classes(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        result = assign_high_risk_label(rfm)
        assert result["is_high_risk"].sum() > 0
        assert (result["is_high_risk"] == 0).sum() > 0

    def test_high_risk_rate_reasonable(self, sample_transactions):
        rfm = compute_rfm(sample_transactions)
        result = assign_high_risk_label(rfm)
        rate = result["is_high_risk"].mean()
        assert 0.05 < rate < 0.70


# ---------------------------------------------------------------------------
# Credit score conversion tests
# ---------------------------------------------------------------------------

class TestCreditScore:
    def test_low_risk_high_score(self):
        score = probability_to_credit_score(0.02)
        assert score >= 700

    def test_high_risk_low_score(self):
        score = probability_to_credit_score(0.95)
        assert score <= 450

    def test_score_in_range(self):
        for p in np.linspace(0.01, 0.99, 50):
            score = probability_to_credit_score(p)
            assert 300 <= score <= 850, f"Score {score} out of range for p={p}"

    def test_monotone_decreasing(self):
        probs = np.linspace(0.05, 0.95, 20)
        scores = [probability_to_credit_score(p) for p in probs]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score not monotone at p={probs[i]:.2f}: {scores[i]} < {scores[i+1]}"
            )


# ---------------------------------------------------------------------------
# Loan recommendation tests
# ---------------------------------------------------------------------------

class TestLoanRecommendation:
    def test_low_risk_tier(self):
        rec = get_loan_recommendation(0.10)
        assert rec["risk_label"] == "low"
        assert rec["recommended_max_amount"] == 50_000

    def test_medium_risk_tier(self):
        rec = get_loan_recommendation(0.35)
        assert rec["risk_label"] == "medium"
        assert rec["recommended_max_amount"] == 20_000

    def test_high_risk_tier(self):
        rec = get_loan_recommendation(0.75)
        assert rec["risk_label"] == "high"
        assert rec["recommended_max_amount"] == 5_000

    def test_boundary_low_medium(self):
        rec = get_loan_recommendation(0.20)
        assert rec["risk_label"] == "low"

    def test_boundary_medium_high(self):
        rec = get_loan_recommendation(0.50)
        assert rec["risk_label"] == "medium"
