"""
data_processing.py
------------------
Full feature engineering pipeline for the Xente eCommerce transaction dataset.

Pipeline steps:
  1. Load raw transaction data
  2. Extract time-based features (hour, day, month, year)
  3. Compute aggregate customer features (total, avg, std, count)
  4. Encode categorical variables (One-Hot Encoding)
  5. Handle missing values (median imputation)
  6. Normalize numerical features (StandardScaler)
  7. Compute RFM metrics per customer
  8. Assign proxy default label via K-Means clustering (is_high_risk)
  9. Save processed dataset to data/processed/

Basel II note: All transformations are deterministic and logged so the
feature pipeline can be audited and reproduced.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
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
# 1. Load
# ---------------------------------------------------------------------------

def load_raw_data(filename: str = "data.csv") -> pd.DataFrame:
    """Load the raw Xente transaction CSV."""
    path = RAW_DATA_PATH / filename
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# 2. Time feature extraction
# ---------------------------------------------------------------------------

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from TransactionStartTime.

    Added columns:
      - TransactionHour:  hour of day (0-23)
      - TransactionDay:   day of month (1-31)
      - TransactionMonth: month (1-12)
      - TransactionYear:  year
      - IsNight:          1 if hour between 22-05, else 0
    """
    df = df.copy()
    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year
    df["IsNight"] = ((df["TransactionHour"] >= 22) | (df["TransactionHour"] <= 5)).astype(int)
    logger.info("Time features extracted")
    return df


# ---------------------------------------------------------------------------
# 3. RFM feature engineering
# ---------------------------------------------------------------------------

def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary features per customer.

    - Recency:   days since last transaction (lower = more recent = better)
    - Frequency: number of transactions
    - Monetary:  total transaction amount (positive debits only)

    Returns a DataFrame indexed by CustomerId.
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


# ---------------------------------------------------------------------------
# 4. Aggregate customer features
# ---------------------------------------------------------------------------

def compute_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional customer-level aggregate features.

    Features:
      - TotalTransactionAmount:  sum of all transaction amounts
      - AvgTransactionAmount:    mean transaction amount
      - TransactionCount:        number of transactions
      - StdTransactionAmount:    std dev of transaction amounts
      - FraudRate:               fraction of transactions flagged as fraud
      - UniqueProducts:          number of distinct products purchased
      - UniqueChannels:          number of distinct channels used
      - NightTxnRatio:           fraction of transactions between 22:00-05:00
    """
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


# ---------------------------------------------------------------------------
# 5. Proxy default label via RFM clustering (is_high_risk)
# ---------------------------------------------------------------------------

def assign_high_risk_label(rfm: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Assign a binary proxy default label using K-Means clustering on RFM scores.

    Methodology:
      - Scale RFM features to zero mean / unit variance
      - Cluster customers into n_clusters groups
      - Rank clusters by composite risk score:
          high Recency + low Frequency + low Monetary → high risk
      - The highest-risk cluster is labelled is_high_risk=1, others 0

    Returns rfm DataFrame with added columns: Cluster, RiskScore, is_high_risk
    """
    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_stats = rfm.groupby("Cluster")[features].mean()
    norm = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min() + 1e-9)
    cluster_stats["RiskScore"] = norm["Recency"] - norm["Frequency"] - norm["Monetary"]

    riskiest_cluster = cluster_stats["RiskScore"].idxmax()
    logger.info(
        "Riskiest cluster: %d | Cluster stats:\n%s",
        riskiest_cluster,
        cluster_stats.to_string(),
    )

    rfm["RiskScore"] = rfm["Cluster"].map(cluster_stats["RiskScore"])
    rfm["is_high_risk"] = (rfm["Cluster"] == riskiest_cluster).astype(int)

    default_rate = rfm["is_high_risk"].mean()
    logger.info("Proxy high-risk rate: %.2f%%", default_rate * 100)
    return rfm


# ---------------------------------------------------------------------------
# 6. sklearn Pipeline for numerical + categorical features
# ---------------------------------------------------------------------------

def build_sklearn_pipeline(numerical_features: list, categorical_features: list) -> Pipeline:
    """
    Build a sklearn ColumnTransformer pipeline that:
      - Imputes missing numerical values with median
      - Scales numerical features with StandardScaler
      - Imputes missing categorical values with most frequent
      - One-Hot Encodes categorical features

    Returns a fitted-ready sklearn Pipeline.
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# 7. Build full feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine RFM, aggregate features, and proxy label into a model-ready DataFrame.
    Saves the result to data/processed/features.csv.

    Returns a DataFrame with:
      - Customer-level features (RFM + aggregates)
      - is_high_risk target column
    """
    # Extract time features at transaction level
    df = extract_time_features(df)

    # Compute customer-level features
    rfm = compute_rfm(df)
    rfm_with_labels = assign_high_risk_label(rfm)
    agg = compute_aggregate_features(df)

    # Merge all features
    features = rfm_with_labels.join(agg, how="left")
    features = features.reset_index()

    # Save processed dataset
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
    print("\nHigh-risk rate:", feature_df["is_high_risk"].mean().round(4))
    print("Columns:", list(feature_df.columns))
