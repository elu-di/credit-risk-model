"""
train.py
--------
Train, tune, and compare credit risk classification models.
All experiments are tracked with MLflow.

Models compared:
  - Logistic Regression (interpretable baseline, Basel II compliant)
  - Random Forest
  - XGBoost
  - LightGBM

Hyperparameter tuning via GridSearchCV on each model.
Best model (by ROC-AUC) is registered in the MLflow Model Registry.
"""

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = Path("data/processed/features.csv")
MODEL_OUTPUT_PATH = Path("models")
RANDOM_SEED = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT = "credit-risk-model"

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
TARGET_COL = "is_high_risk"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features() -> tuple[pd.DataFrame, pd.Series]:
    logger.info("Loading features from %s", PROCESSED_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Support both old (is_default) and new (is_high_risk) column names
    if TARGET_COL not in df.columns and "is_default" in df.columns:
        df = df.rename(columns={"is_default": TARGET_COL})

    # Use only columns that exist
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    logger.info("Using features: %s", available_features)

    X = df[available_features]
    y = df[TARGET_COL]
    logger.info("Dataset: %d rows | High-risk rate: %.2f%%", len(df), y.mean() * 100)
    return X, y


# ---------------------------------------------------------------------------
# Model + hyperparameter grid definitions
# ---------------------------------------------------------------------------

def get_models_with_grids() -> dict:
    """Return dict of model name → (Pipeline, param_grid)."""
    return {
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                )),
            ]),
            {"clf__C": [0.01, 0.1, 1.0]},
        ),
        "random_forest": (
            Pipeline([
                ("clf", RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )),
            ]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [6, 8]},
        ),
        "xgboost": (
            Pipeline([
                ("clf", XGBClassifier(
                    scale_pos_weight=5,
                    random_state=RANDOM_SEED,
                    eval_metric="logloss",
                    verbosity=0,
                )),
            ]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [4, 6], "clf__learning_rate": [0.05, 0.1]},
        ),
        "lightgbm": (
            Pipeline([
                ("clf", LGBMClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    verbose=-1,
                )),
            ]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [4, 6], "clf__learning_rate": [0.05, 0.1]},
        ),
    }


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Train with GridSearchCV, log to MLflow, return metrics dict."""
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_name", name)

        # Hyperparameter tuning
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_

        logger.info("%s | Best params: %s | CV AUC: %.4f",
                    name, best_params, grid_search.best_score_)

        # Log parameters
        mlflow.log_params({"model": name, "test_size": TEST_SIZE, **best_params})
        mlflow.log_metric("cv_auc_mean", grid_search.best_score_)

        # Test set evaluation
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        y_pred = best_pipeline.predict(X_test)

        # All required metrics
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_prob),
            "test_avg_precision": average_precision_score(y_test, y_prob),
        }

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        logger.info(
            "%s | Accuracy: %.4f | Precision: %.4f | Recall: %.4f | "
            "F1: %.4f | AUC: %.4f",
            name,
            metrics["test_accuracy"],
            metrics["test_precision"],
            metrics["test_recall"],
            metrics["test_f1"],
            metrics["test_roc_auc"],
        )

        # Log model artifact
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        return {
            "name": name,
            "pipeline": best_pipeline,
            "run_id": mlflow.active_run().info.run_id,
            **metrics,
        }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_all():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    models = get_models_with_grids()
    results = []

    for name, (pipeline, param_grid) in models.items():
        logger.info("--- Training: %s ---", name)
        result = evaluate_model(name, pipeline, param_grid, X_train, X_test, y_train, y_test)
        results.append(result)

    # Select best model by test ROC-AUC
    best = max(results, key=lambda r: r["test_roc_auc"])
    logger.info("Best model: %s (AUC=%.4f)", best["name"], best["test_roc_auc"])

    # Save best model locally
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_OUTPUT_PATH / "credit_risk_model.joblib"
    joblib.dump(best["pipeline"], model_path)
    logger.info("Best model saved to %s", model_path)

    # Register in MLflow Model Registry
    model_uri = f"runs:/{best['run_id']}/model"
    mlflow.register_model(model_uri, "CreditRiskModel")
    logger.info("Model registered in MLflow registry as 'CreditRiskModel'")

    # Print comparison table
    print("\n=== Model Comparison ===")
    print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["test_roc_auc"], reverse=True):
        print(f"{r['name']:<25} {r['test_accuracy']:>9.4f} {r['test_precision']:>10.4f} "
              f"{r['test_recall']:>8.4f} {r['test_f1']:>8.4f} {r['test_roc_auc']:>8.4f}")

    return best


if __name__ == "__main__":
    best_result = train_all()
    print(f"\nBest model: {best_result['name']} | ROC-AUC: {best_result['test_roc_auc']:.4f}")
