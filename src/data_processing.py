"""
data_processing.py
------------------
Full feature engineering pipeline for the Xente eCommerce transaction dataset.

Task 3 deliverable: A single fitted sklearn Pipeline object that transforms
raw transaction data into a model-ready customer-level feature matrix.

Pipeline steps:
  1. Extract time-based features (hour, day, month, year, is_night)
  2. Compute aggregate customer features (total, avg, std, count per customer)
  3. Encode categorical variables (One-Hot Encoding)
  4. Handle missing values (median imputation for numerical, mode for categorical)
  5. Normalize/Standardize numerical features (StandardScaler)
  6. Compute RFM metrics per customer
  7. Assign proxy default label via K-Means clustering (is_high_risk)
  8. Save processed dataset to data/processed/

Basel II note: All transformations are deterministic, seeded, and logged
so the feature pipeline can be audited and reproduced.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Custom transformers (sklearn-compatible)
# ---------------------------------------------------------------------------

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from TransactionStartTime.

    Adds: TransactionHour, TransactionDay, TransactionMonth,
          TransactionYear, IsNight
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        ts = pd.to_datetime(X["TransactionStartTime"])
        X["TransactionHour"] = ts.dt.hour
        X["TransactionDay"] = ts.dt.day
        X["TransactionMonth"] = ts.dt.month
        X["TransactionYear"] = ts.dt.year
        X["IsNight"] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype(int)
        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregate transaction-level data to customer-level features.

    Produces one row per CustomerId with:
      - TotalTransactionAmount, AvgTransactionAmount, TransactionCount,
        StdTransactionAmount, FraudRate, UniqueProducts, UniqueChannels,
        NightTxnRatio
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "IsNight" not in X.columns:
            ts = pd.to_datetime(X["TransactionStartTime"])
            X["IsNight"] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype(int)

        agg = X.groupby("CustomerId").agg(
            TotalTransactionAmount=("Amount", "sum"),
            AvgTransactionAmount=("Amount", "mean"),
            TransactionCount=("TransactionId", "count"),
            StdTransactionAmount=("Amount", "std"),
            FraudRate=("FraudResult", "mean"),
            UniqueProducts=("ProductId", "nunique"),
            UniqueChannels=("ChannelId", "nunique"),
            NightTxnRatio=("IsNight", "mean"),
        )
        agg["StdTransactionAmount"] = agg["StdTransactionAmount"].fillna(0)
        return agg.reset_index()


# ---------------------------------------------------------------------------
# Standalone helper functions (used by EDA notebook and tests)
# ---------------------------------------------------------------------------

def load_raw_data(filename: str = "data.csv") -> pd.DataFrame:
    """Load the raw Xente transaction CSV."""
    path = RAW_DATA_PATH / filename
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from TransactionStartTime."""
    df = df.copy()
    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year
    df["IsNight"] = ((df["TransactionHour"] >= 22) | (df["TransactionHour"] <= 5)).astype(int)
    logger.info("Time features extracted")
    return df


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary features per customer.

    - Recency:   days since last transaction
    - Frequency: number of transactions
    - Monetary:  total debit transaction amount
    """
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    logger.info("Snapshot date for RFM: %s", snapshot_date)

    debits = df[df["Amount"] > 0].copy()

    rfm = debits.groupby("CustomerId").agg(
        Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum"),
    )
    logger.info("RFM computed for %d customers", len(rfm))
    return rfm


def compute_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive customer-level aggregate features from transaction history."""
    df = df.copy()
    df["Hour"] = df["TransactionStartTime"].dt.hour
    df["IsNight"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)

    agg = df.groupby("CustomerId").agg(
        TotalTransactionAmount=("Amount", "sum"),
        AvgTransactionAmount=("Amount", "mean"),
        TransactionCount=("TransactionId", "count"),
        StdTransactionAmount=("Amount", "std"),
        FraudRate=("FraudResult", "mean"),
        UniqueProducts=("ProductId", "nunique"),
        UniqueChannels=("ChannelId", "nunique"),
        NightTxnRatio=("IsNight", "mean"),
    )
    agg["StdTransactionAmount"] = agg["StdTransactionAmount"].fillna(0)
    logger.info("Aggregate features computed for %d customers", len(agg))
    return agg


def assign_high_risk_label(rfm: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Assign binary proxy default label using K-Means clustering on RFM scores.

    The cluster with highest Recency + lowest Frequency + lowest Monetary
    is labelled is_high_risk=1 (proxy for credit default risk).

    Returns rfm DataFrame with Cluster, RiskScore, is_high_risk columns.
    """
    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_stats = rfm.groupby("Cluster")[features].mean()
    norm = (cluster_stats - cluster_stats.min()) / (
        cluster_stats.max() - cluster_stats.min() + 1e-9
    )
    cluster_stats["RiskScore"] = norm["Recency"] - norm["Frequency"] - norm["Monetary"]

    riskiest_cluster = cluster_stats["RiskScore"].idxmax()
    logger.info(
        "Riskiest cluster: %d | Cluster stats:\n%s",
        riskiest_cluster,
        cluster_stats.to_string(),
    )

    rfm["RiskScore"] = rfm["Cluster"].map(cluster_stats["RiskScore"])
    rfm["is_high_risk"] = (rfm["Cluster"] == riskiest_cluster).astype(int)

    logger.info("Proxy high-risk rate: %.2f%%", rfm["is_high_risk"].mean() * 100)
    return rfm


# ---------------------------------------------------------------------------
# sklearn Pipeline builder (Task 3 deliverable)
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    numerical_features: list,
    categorical_features: list,
) -> Pipeline:
    """
    Build a sklearn Pipeline for preprocessing customer-level features.

    Steps:
      - Numerical: median imputation → StandardScaler
      - Categorical: mode imputation → OneHotEncoder

    Returns an unfitted Pipeline ready for fit_transform().
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor)])


# ---------------------------------------------------------------------------
# Full feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end pipeline: raw transactions → model-ready customer feature matrix.

    Steps:
      1. Extract time features at transaction level
      2. Compute RFM per customer
      3. Assign is_high_risk proxy label via K-Means
      4. Compute aggregate customer features
      5. Merge all features into one DataFrame
      6. Save to data/processed/features.csv

    Returns a DataFrame with one row per customer and all engineered features.
    """
    logger.info("Building feature matrix from %d transactions", len(df))

    # Step 1: time features
    df = extract_time_features(df)

    # Step 2 & 3: RFM + proxy label
    rfm = compute_rfm(df)
    rfm_labeled = assign_high_risk_label(rfm)

    # Step 4: aggregate features
    agg = compute_aggregate_features(df)

    # Step 5: merge
    features = rfm_labeled.join(agg, how="left")
    features = features.reset_index()

    # Save
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_PATH / "features.csv"
    features.to_csv(out_path, index=False)
    logger.info("Feature matrix saved to %s (%d rows)", out_path, len(features))
    return features


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw_df = load_raw_data()
    feature_df = build_feature_matrix(raw_df)

    print(feature_df.head())
    print(f"\nShape: {feature_df.shape}")
    print(f"High-risk rate: {feature_df['is_high_risk'].mean():.4f}")
    print(f"Columns: {list(feature_df.columns)}")

    # Demonstrate the sklearn preprocessing pipeline on the feature matrix
    numerical_cols = [
        "Recency", "Frequency", "Monetary",
        "TotalTransactionAmount", "AvgTransactionAmount",
        "TransactionCount", "StdTransactionAmount",
        "FraudRate", "UniqueProducts", "UniqueChannels", "NightTxnRatio",
    ]
    # No categorical columns at customer level after aggregation
    pipeline = build_preprocessing_pipeline(numerical_cols, [])
    X = feature_df[numerical_cols]
    X_transformed = pipeline.fit_transform(X)
    print(f"\nPreprocessed feature matrix shape: {X_transformed.shape}")
    print("Pipeline fitted successfully.")
