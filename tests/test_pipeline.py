"""
Tests for the data pipeline, feature engineering, and API.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT))

from preprocessing import clean  # noqa: E402
from feature_engineering import add_features, build_feature_table  # noqa: E402


@pytest.fixture
def raw_sample():
    return pd.DataFrame(
        [
            {
                "customerID": "1001-AAA", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": "285.0",
                "Churn": "Yes",
            },
            {
                "customerID": "1002-BBB", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "Yes", "tenure": 48, "PhoneService": "Yes", "MultipleLines": "Yes",
                "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
                "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check", "MonthlyCharges": 55.0, "TotalCharges": " ",
                "Churn": "No",
            },
            # duplicate of row 1 to test dedup
            {
                "customerID": "1001-AAA", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": "285.0",
                "Churn": "Yes",
            },
        ]
    )


def test_clean_drops_duplicates(raw_sample):
    cleaned = clean(raw_sample)
    assert len(cleaned) == 2
    assert cleaned["customerID"].is_unique


def test_clean_handles_blank_total_charges(raw_sample):
    cleaned = clean(raw_sample)
    assert cleaned["TotalCharges"].isna().sum() == 0
    # blank TotalCharges row (tenure=48, monthly=55) should be imputed to 48*55
    row = cleaned[cleaned["customerID"] == "1002-BBB"].iloc[0]
    assert row["TotalCharges"] == pytest.approx(48 * 55.0)


def test_clean_encodes_target_as_binary(raw_sample):
    cleaned = clean(raw_sample)
    assert set(cleaned["Churn"].unique()) <= {0, 1}


def test_add_features_creates_expected_columns(raw_sample):
    cleaned = clean(raw_sample)
    featured = add_features(cleaned)
    for col in ["AvgMonthlySpend", "TenureGroup", "ServiceCount", "HighValueCustomer", "EstimatedCLV"]:
        assert col in featured.columns


def test_add_features_handles_zero_tenure():
    df = pd.DataFrame([{
        "customerID": "x", "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check", "MonthlyCharges": 40.0,
        "TotalCharges": 0.0, "Churn": 0,
    }])
    # should not raise a divide-by-zero error
    featured = add_features(df)
    assert featured["AvgMonthlySpend"].iloc[0] == 0.0


def test_build_feature_table_output_is_all_numeric(raw_sample):
    cleaned = clean(raw_sample)
    table = build_feature_table(cleaned)
    assert "customerID" not in table.columns
    non_numeric = table.select_dtypes(exclude="number").columns.tolist()
    assert non_numeric == [], f"Non-numeric columns leaked through: {non_numeric}"


def test_model_artifacts_exist_after_training():
    models_dir = ROOT / "models"
    required = ["churn_model.joblib", "scaler.joblib", "feature_names.joblib", "model_metadata.json"]
    missing = [f for f in required if not (models_dir / f).exists()]
    if missing:
        pytest.skip(f"Model not trained yet, skipping: missing {missing}")
    assert True


def test_api_predict_endpoint():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    if not (ROOT / "models" / "churn_model.joblib").exists():
        pytest.skip("Model not trained yet")

    from app.api import app

    # Use as a context manager so FastAPI's startup event (which loads the
    # model artifacts) actually runs.
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is True

        payload = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 100.0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["prediction"] in ("Likely to Churn", "Likely to Stay")
        assert len(body["top_reasons"]) == 5
