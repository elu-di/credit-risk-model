# Credit Risk Probability Model

End-to-end credit scoring system for Bati Bank's buy-now-pay-later service, built on eCommerce transaction data from the Xente platform.

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline (lint + test)
├── data/
│   ├── raw/                      # Raw Xente transaction data (gitignored)
│   └── processed/                # Feature-engineered dataset (gitignored)
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
├── src/
│   ├── data_processing.py        # Full sklearn Pipeline: features + proxy label
│   ├── train.py                  # Model training + MLflow tracking
│   ├── predict.py                # Inference utilities
│   └── api/
│       ├── main.py               # FastAPI application
│       └── pydantic_models.py    # Request/response schemas
├── tests/
│   └── test_data_processing.py   # Unit tests (25 passing)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quickstart

### 1. Set up environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Add data

Place the Xente dataset CSV in `data/raw/data.csv`.

### 3. Run feature engineering

```bash
python src/data_processing.py
```

### 4. Train models

```bash
python src/train.py
```

### 5. Start the API

```bash
uvicorn src.api.main:app --reload
```

### 6. Run with Docker

```bash
docker-compose up --build
```

### 7. Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord requires banks to hold capital reserves proportional to the credit risk they carry. To calculate that risk, banks must use models that regulators can audit, validate, and challenge. This creates three concrete requirements for any credit scoring model:

**Interpretability**: Regulators and internal risk committees must be able to understand *why* a model assigns a given score to a borrower. A black-box model that produces accurate predictions but cannot explain its reasoning fails this requirement. Logistic Regression with Weight of Evidence (WoE) encoding is the industry standard precisely because each coefficient has a direct, auditable interpretation: a one-unit change in a WoE-transformed feature produces a known change in log-odds of default.

**Documentation**: Basel II's Pillar 2 (Supervisory Review) requires banks to document model assumptions, data sources, variable selection rationale, and validation results. Every modeling choice — including the definition of "default," the choice of features, and the handling of missing data — must be justified in writing and reviewed periodically.

**Reproducibility**: Models must produce consistent, deterministic outputs. Random seeds, versioned datasets, and logged experiment parameters (as provided by MLflow in this project) are not optional — they are regulatory requirements for model governance.

In this project, we include Logistic Regression alongside tree-based models specifically to satisfy Basel II interpretability requirements, even though gradient boosting achieves higher raw performance.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The Xente dataset contains transaction records but no ground-truth loan default outcomes — there is no column indicating whether a customer failed to repay a loan. This is common when building credit models for new lending products or when historical default data does not exist.

**Why a proxy is necessary**: A supervised classification model requires a target variable. Without one, we cannot train a model to distinguish risky from safe borrowers. A proxy variable — derived from observable behavioral signals — allows us to approximate creditworthiness using the data we do have.

**Our proxy methodology**: We use RFM (Recency, Frequency, Monetary) analysis to segment customers into behavioral clusters using K-Means. Customers in the cluster characterized by high recency (long time since last transaction), low frequency, and low monetary value are labeled `is_high_risk = 1`. The intuition is that disengaged customers with low spending activity are more likely to be financially stressed or unreliable borrowers.

**Business risks introduced by proxy-based prediction**:

- **Label noise**: The proxy may misclassify customers. A customer who stopped transacting because they moved abroad is not a credit risk, but our model would label them high-risk. This leads to false positives — creditworthy customers denied loans.
- **Circular reasoning**: If the proxy is derived from the same features used to train the model, the model learns to reproduce the clustering rather than predict true default. Our near-perfect AUC (1.0) is a direct symptom of this — it reflects cluster separation, not genuine predictive power on real default outcomes.
- **Regulatory risk**: Basel II requires that the definition of default be clearly justified and aligned with actual loss events. A proxy label that has not been validated against real default data may not satisfy supervisory review.
- **Distributional shift**: Customer behavior patterns may change over time. A proxy calibrated on 2018–2019 transaction data may not reflect risk in 2024.

These risks must be disclosed to Bati Bank's risk committee and documented in the model governance report. The proxy model should be treated as a starting point, to be recalibrated once actual loan repayment data becomes available.

---

### 3. What are the key trade-offs between a simple, interpretable model (Logistic Regression with WoE) and a high-performance model (Gradient Boosting) in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|---|---|---|
| **Interpretability** | High — coefficients directly map to log-odds; WoE bins are auditable | Low — hundreds of trees, non-linear interactions are opaque |
| **Regulatory acceptance** | High — standard in Basel II scorecards; easy to document | Requires additional explainability tools (SHAP, LIME) |
| **Predictive performance** | Moderate — assumes linear log-odds relationship | High — captures non-linear patterns and feature interactions |
| **Handling of missing data** | Requires explicit imputation | Handles natively (XGBoost, LightGBM) |
| **Overfitting risk** | Low — regularization (C parameter) is well-understood | Higher — requires careful tuning of depth, learning rate, n_estimators |
| **Deployment complexity** | Low — single coefficient vector, fast inference | Higher — serialized tree ensemble, larger model artifact |
| **Model monitoring** | Simple — monitor coefficient stability (PSI on WoE bins) | Complex — requires monitoring feature importance drift |
| **Stakeholder communication** | Easy — scorecard format familiar to credit analysts | Difficult — requires technical explanation for non-technical audiences |

**Recommendation for Bati Bank**: Deploy Logistic Regression as the primary regulatory model for loan approval decisions (where interpretability and auditability are mandatory), and use LightGBM as a challenger model for internal risk monitoring and portfolio management (where performance matters more than explainability). This dual-model approach is standard practice at regulated financial institutions.

---

## Model Results

| Model | CV AUC | Test AUC | F1 (high-risk) |
|---|---|---|---|
| Logistic Regression | 0.9992 | 0.9988 | 0.981 |
| Random Forest | 1.0000 | 1.0000 | 0.997 |
| XGBoost | 0.9999 | 1.0000 | 0.998 |
| **LightGBM** ✅ | **0.9999** | **1.0000** | **0.997** |

> **Note on near-perfect scores**: AUC = 1.0 reflects that the model is learning to reproduce the K-Means cluster assignments (the proxy label was derived from the same features). This is expected behavior for a proxy-label model and should be interpreted as cluster separability, not real-world default predictability.

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Recency": 14,
    "Frequency": 23,
    "Monetary": 4500.0,
    "AvgTransactionValue": 195.65,
    "StdTransactionValue": 88.3,
    "FraudRate": 0.0,
    "UniqueProducts": 7,
    "UniqueChannels": 2,
    "NightTxnRatio": 0.04
  }'
```

Response:
```json
{
  "risk_probability": 0.02,
  "credit_score": 839,
  "risk_label": "low",
  "recommended_max_amount": 50000,
  "recommended_max_months": 36
}
```
