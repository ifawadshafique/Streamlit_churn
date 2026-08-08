# End-to-End Customer Churn Prediction Platform

A complete machine learning system that predicts whether a telecom customer
will churn and explains *why*, wrapped in a REST API and an interactive
dashboard — covering the full lifecycle from raw data to a
production-style, monitorable service.

> **Business problem:** acquiring a new customer costs far more than
> retaining one. This platform flags at-risk customers early, with
> plain-language reasons a retention team can act on.

## What's inside

| Stage | Script | What it does |
|---|---|---|
| Data | `src/generate_data.py` | Synthetic Telco-style dataset (7,043 customers) with realistic churn drivers and injected messiness. Swap in the real IBM Telco CSV to `data/telco_churn.csv` and skip this step. |
| Cleaning | `src/preprocessing.py` | Dedup, fixes the string-typed `TotalCharges` column, normalizes the target |
| EDA | `src/eda.py` | Churn-rate breakdowns, distributions, correlation heatmap → `reports/` |
| Feature engineering | `src/feature_engineering.py` | `AvgMonthlySpend`, `TenureGroup`, `ServiceCount`, `EstimatedCLV`, and more, then one-hot encoding |
| Training | `src/train.py` | Trains & compares Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM; tunes the winner with `RandomizedSearchCV`; logs everything to MLflow |
| Explainability | `src/explain.py` | SHAP global summary plot + per-customer top-reasons |
| Batch scoring | `src/predict.py` | CLI to score any CSV of customers |
| API | `app/api.py` | FastAPI service: `/predict`, `/predict/batch`, `/predict/batch/file`, `/health` |
| Dashboard | `app/dashboard.py` | Original Streamlit dashboard |
| Streamlit web app | `streamlit_app.py` | Deployable single-app UI with batch upload, single-customer prediction, model comparison, and SHAP explanations |
| React frontend | `frontend/` | Optional original React UI for use with the FastAPI service |
| Tests | `tests/test_pipeline.py` | Pipeline + API tests (pytest) |

## Quickstart

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Full pipeline: generate data -> clean -> engineer features -> train
python src/generate_data.py
python src/preprocessing.py
python src/feature_engineering.py
python src/train.py

# (optional) See the EDA plots and SHAP summary
python src/eda.py
python src/explain.py
```

This produces `models/churn_model.joblib` plus the scaler, feature list,
and metadata the API and dashboard need.

### Run the API

```bash
uvicorn app.api:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`. Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "tenure": 4, "Contract": "Month-to-month",
    "InternetService": "Fiber optic", "MonthlyCharges": 95.0,
    "TechSupport": "No", "OnlineSecurity": "No",
    "PaymentMethod": "Electronic check", "PaperlessBilling": "Yes",
    "StreamingTV": "Yes", "StreamingMovies": "No"
  }'
```

```json
{
  "churn_probability": 0.9623,
  "prediction": "Likely to Churn",
  "top_reasons": [
    {"feature": "tenure", "value": 4.0, "impact": 1.0152, "direction": "increases"},
    {"feature": "MonthlyCharges", "value": 95.0, "impact": 0.6675, "direction": "increases"},
    {"feature": "Contract_Month-to-month", "value": 1.0, "impact": 0.5829, "direction": "increases"}
  ]
}
```

### Run the Streamlit web app

The recommended deployment entry point is the new root-level `streamlit_app.py`. It reuses the existing trained model, scaler, feature names, preprocessing, feature engineering, and SHAP explanation logic.

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

For Render, use:

```bash
pip install -r requirements-streamlit.txt
```

Build command and:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

The app requires the trained artifacts under `models/` and `data/telco_churn_features.csv`.

### Run the dashboard

```bash
streamlit run app/dashboard.py
```

Upload a CSV (the Telco schema, e.g. `data/telco_churn.csv`) to get:
total customers, high-risk count, a probability distribution chart, a
per-customer search with SHAP reasons, and a downloadable results CSV.
There's also a manual single-customer form and a model-comparison tab.

### Run the React dashboard

A standalone React frontend lives in `frontend/` — upload a CSV/XLSX, get
a churn report (summary stats, risk distribution & segment charts, a
sortable/filterable customer table) with per-customer SHAP explanations
on click. It talks to the API's `/predict/batch/file` endpoint, so start
the API first:

```bash
uvicorn app.api:app --reload --port 8000

# in another terminal
cd frontend
npm install
npm run dev
```

See `frontend/README.md` for details.

### Score a CSV from the command line

```bash
python src/predict.py --input data/telco_churn.csv --output predictions.csv
```

## Docker

```bash
docker compose up --build
```

Starts the API on `:8000` and the dashboard on `:8501`. The image builds
the dataset and trains the model at build time, so it's ready to serve
immediately. To build/run just the API:

```bash
docker build -t churn-platform .
docker run -p 8000:8000 churn-platform
```

## MLOps

- **MLflow**: every candidate model and the final tuned model are logged
  (params, metrics, artifact). View with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- **Tests**: `pytest tests/ -v` covers cleaning, feature engineering, and
  the live API.
- **CI**: `.github/workflows/ci.yml` runs the full pipeline + tests +
  a Docker build on every push/PR.
- **Model versioning**: `models/model_metadata.json` records which model
  won, its hyperparameters, and its test-set metrics for every training run.

## Model results

On the bundled synthetic dataset (results will vary slightly run to run):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.752 | 0.437 | 0.759 | 0.554 | **0.827** |
| Random Forest | 0.795 | 0.496 | 0.633 | 0.556 | 0.817 |
| LightGBM | 0.820 | 0.590 | 0.378 | 0.461 | 0.807 |
| XGBoost | 0.765 | 0.449 | 0.696 | 0.546 | 0.807 |
| Decision Tree | 0.722 | 0.399 | 0.731 | 0.516 | 0.793 |

The pipeline selects and tunes whichever model wins on ROC-AUC
automatically — on this data, Logistic Regression, because recall on the
minority (churn) class matters more than raw accuracy for a retention
use case, and ROC-AUC captures that better than accuracy alone.

## Using the real IBM Telco dataset

This repo ships with a synthetic dataset (`src/generate_data.py`) so it
runs standalone with no external downloads. To use the real data:

1. Download the "Telco Customer Churn" dataset (search for it on Kaggle).
2. Save it as `data/telco_churn.csv` with the same column names shown in
   `src/generate_data.py`.
3. Skip `python src/generate_data.py` and run the rest of the pipeline as-is.

## Project structure

```
customer-churn-platform/
├── data/                    # generated dataset + intermediate CSVs
├── src/
│   ├── generate_data.py     # synthetic data generator
│   ├── preprocessing.py     # cleaning
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── train.py             # multi-model training + tuning + MLflow
│   ├── explain.py           # SHAP
│   └── predict.py           # batch CLI scorer
├── app/
│   ├── api.py                # FastAPI service
│   └── dashboard.py           # Streamlit dashboard
├── models/                   # trained model + scaler + metadata (generated)
├── reports/                  # EDA + SHAP plots (generated)
├── tests/
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## What this demonstrates

- Data cleaning and feature engineering on realistically messy data
- Comparing and tuning multiple model families with proper train/test
  discipline and class-imbalance handling
- Explainable AI (SHAP) at both the global and per-prediction level
- Turning a model into a real API and a usable business-facing dashboard
- MLOps basics: experiment tracking, tests, CI, containerization
#   S t r e a m l i t _ c h u r n  
 