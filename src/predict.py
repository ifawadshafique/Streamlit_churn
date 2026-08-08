"""
Standalone CLI to score a CSV of customers with the trained model.

Usage:
    python src/predict.py --input data/telco_churn.csv --output predictions.csv
"""

import argparse
import json

import joblib
import pandas as pd

from feature_engineering import build_feature_table
from preprocessing import clean


def load_model(models_dir="models"):
    model = joblib.load(f"{models_dir}/churn_model.joblib")
    scaler = joblib.load(f"{models_dir}/scaler.joblib")
    feature_names = joblib.load(f"{models_dir}/feature_names.joblib")
    with open(f"{models_dir}/model_metadata.json") as f:
        needs_scaling = json.load(f)["needs_scaling"]
    return model, scaler, feature_names, needs_scaling


def predict_csv(input_path, output_path, models_dir="models"):
    model, scaler, feature_names, needs_scaling = load_model(models_dir)

    raw_df = pd.read_csv(input_path)
    work_df = raw_df.drop(columns=["Churn"], errors="ignore")
    work_df = clean(work_df, verbose=False)

    featured = build_feature_table(work_df)
    featured = featured.reindex(columns=feature_names, fill_value=0)

    X_input = scaler.transform(featured) if needs_scaling else featured.values
    proba = model.predict_proba(X_input)[:, 1]

    result = work_df.copy()
    result["ChurnProbability"] = proba.round(4)
    result["Prediction"] = ["Likely to Churn" if p >= 0.5 else "Likely to Stay" for p in proba]
    result.to_csv(output_path, index=False)
    print(f"Scored {len(result)} customers -> {output_path}")
    print(f"High risk (>=50%): {(proba >= 0.5).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    predict_csv(args.input, args.output, args.models_dir)
