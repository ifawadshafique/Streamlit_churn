"""
Step 8: FastAPI service.

Run with:
    uvicorn app.api:app --reload --port 8000

Endpoints:
    GET  /health
    POST /predict             -> single customer prediction + top SHAP reasons
    POST /predict/batch       -> list of customers -> list of predictions
    POST /predict/batch/file  -> upload a CSV/XLSX -> full churn report (React frontend uses this)
"""

import io
import json
import sys
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from feature_engineering import build_feature_table, tenure_group  # noqa: E402
from explain import explain_instance  # noqa: E402
from preprocessing import clean  # noqa: E402

MODELS_DIR = ROOT / "models"

_model = None
_scaler = None
_feature_names = None
_needs_scaling = None
_background = None


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"] = "Male"
    SeniorCitizen: Literal[0, 1] = 0
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = "Electronic check"
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 4,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0,
            }
        }
    }


class Reason(BaseModel):
    feature: str
    value: object
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
    top_reasons: List[Reason]


def _load_artifacts():
    global _model, _scaler, _feature_names, _needs_scaling, _background
    if _model is not None:
        return
    if not (MODELS_DIR / "churn_model.joblib").exists():
        raise RuntimeError(
            "Model artifacts not found. Run the pipeline first: "
            "python src/generate_data.py && python src/preprocessing.py && "
            "python src/feature_engineering.py && python src/train.py"
        )
    _model = joblib.load(MODELS_DIR / "churn_model.joblib")
    _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    _feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    with open(MODELS_DIR / "model_metadata.json") as f:
        _needs_scaling = json.load(f)["needs_scaling"]

    # Small background sample for SHAP, built once at startup.
    features_path = ROOT / "data" / "telco_churn_features.csv"
    if features_path.exists():
        df = pd.read_csv(features_path)
        _background = df.drop(columns=["Churn"]).sample(n=min(100, len(df)), random_state=1)
    else:
        _background = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts telecom customer churn probability with SHAP-based explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React dev server (and any other origin) to call this API.
# Tighten allow_origins to your deployed frontend's URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


def _customer_to_feature_row(customer: CustomerInput) -> pd.DataFrame:
    row = customer.model_dump()
    if row["TotalCharges"] is None:
        row["TotalCharges"] = row["MonthlyCharges"] * max(row["tenure"], 1)
    row["customerID"] = "TEMP-0000"
    raw_df = pd.DataFrame([row])
    featured = build_feature_table(raw_df)

    # Align columns to the training-time feature set (one-hot columns that
    # didn't appear for this single row need to be added back as 0).
    featured = featured.reindex(columns=_feature_names, fill_value=0)
    return featured


def _predict_row(featured: pd.DataFrame):
    X_input = _scaler.transform(featured) if _needs_scaling else featured.values
    proba = float(_model.predict_proba(X_input)[0, 1])
    label = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"
    return proba, label


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    _load_artifacts()
    try:
        featured = _customer_to_feature_row(customer)
        proba, label = _predict_row(featured)
        reasons = explain_instance(
            _model, _scaler, _feature_names, featured, _needs_scaling,
            top_n=5, background=_background,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(
        churn_probability=round(proba, 4),
        prediction=label,
        top_reasons=reasons,
    )


@app.post("/predict/batch")
def predict_batch(customers: List[CustomerInput]):
    _load_artifacts()
    results = []
    for customer in customers:
        featured = _customer_to_feature_row(customer)
        proba, label = _predict_row(featured)
        results.append({"churn_probability": round(proba, 4), "prediction": label})
    return {"count": len(results), "results": results}


BREAKDOWN_COLUMNS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "SeniorCitizen",
    "gender",
    "TenureGroup",
]


def _risk_level(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.3:
        return "Medium"
    return "Low"


def _read_upload(file: UploadFile, contents: bytes) -> pd.DataFrame:
    name = (file.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(contents))
        return pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read '{file.filename}': {e}")


@app.post("/predict/batch/file")
async def predict_batch_file(file: UploadFile = File(...)):
    """Upload a CSV or XLSX of customers and get back a full churn report:
    per-customer predictions, summary stats, and segment breakdowns, ready
    for the React dashboard to render as charts/tables.
    """
    _load_artifacts()

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    raw_df = _read_upload(file, contents)
    if raw_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file has no rows.")

    work_df = raw_df.drop(columns=["Churn"], errors="ignore")

    required = {"tenure", "MonthlyCharges", "Contract"}
    missing = required - set(work_df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"File is missing expected column(s): {', '.join(sorted(missing))}",
        )

    try:
        cleaned = clean(work_df, verbose=False)
        featured = build_feature_table(cleaned)
        featured = featured.reindex(columns=_feature_names, fill_value=0)
        X_input = _scaler.transform(featured) if _needs_scaling else featured.values
        proba = _model.predict_proba(X_input)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scoring file: {e}")

    result = cleaned.copy()
    result["ChurnProbability"] = proba.round(4)
    result["Prediction"] = ["Likely to Churn" if p >= 0.5 else "Likely to Stay" for p in proba]
    result["RiskLevel"] = [_risk_level(p) for p in proba]
    result["TenureGroup"] = result["tenure"].apply(tenure_group)
    if "customerID" not in result.columns:
        result.insert(0, "customerID", [f"ROW-{i + 1}" for i in range(len(result))])

    total = len(result)
    churn_count = int((proba >= 0.5).sum())
    summary = {
        "total_customers": total,
        "predicted_churn_count": churn_count,
        "predicted_retain_count": total - churn_count,
        "churn_rate": round(churn_count / total, 4),
        "average_risk": round(float(proba.mean()), 4),
        "risk_breakdown": result["RiskLevel"].value_counts().to_dict(),
        "estimated_monthly_revenue_at_risk": round(
            float(result.loc[proba >= 0.5, "MonthlyCharges"].sum()), 2
        ),
    }

    # Risk probability histogram (10 buckets of 10%), for the distribution chart.
    hist_counts, hist_edges = np.histogram(proba, bins=10, range=(0, 1))
    histogram = [
        {"bucket": f"{int(hist_edges[i] * 100)}-{int(hist_edges[i + 1] * 100)}%", "count": int(c)}
        for i, c in enumerate(hist_counts)
    ]

    segment_breakdown = {}
    for col in BREAKDOWN_COLUMNS:
        if col not in result.columns:
            continue
        grp = (
            result.groupby(col)["ChurnProbability"]
            .agg(avg_risk="mean", count="count")
            .reset_index()
            .rename(columns={col: "segment"})
        )
        grp["avg_risk"] = grp["avg_risk"].round(4)
        segment_breakdown[col] = json.loads(grp.to_json(orient="records"))

    # Keep the response light: identifying/raw fields + prediction, not every
    # one-hot engineered column. Reasons are fetched on demand via /predict.
    display_cols = [
        c for c in [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "TenureGroup", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges",
            "ChurnProbability", "Prediction", "RiskLevel",
        ] if c in result.columns
    ]
    customers = json.loads(result[display_cols].to_json(orient="records"))

    return {
        "summary": summary,
        "histogram": histogram,
        "segment_breakdown": segment_breakdown,
        "customers": customers,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
