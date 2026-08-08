"""
Step 7: Model Explainability with SHAP.

Loads the trained model and produces:
- a global feature-importance summary plot (reports/shap_summary.png)
- a helper `explain_instance()` used by both the API and the dashboard
  to return the top reasons behind a single prediction.
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/churn_model.joblib")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.joblib")
    return model, scaler, feature_names


def build_explainer(model, background: pd.DataFrame):
    """Pick the right SHAP explainer type for the trained model."""
    model_type = type(model).__name__
    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier", "DecisionTreeClassifier"):
        return shap.TreeExplainer(model)
    if model_type == "LogisticRegression":
        return shap.LinearExplainer(model, background)
    # Fallback: model-agnostic (slower, sampled)
    return shap.Explainer(model.predict_proba, background)


def get_shap_values(explainer, X: pd.DataFrame):
    raw = explainer.shap_values(X)
    if isinstance(raw, list):  # some tree explainers return [class0, class1]
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return arr[:, :, -1]
    return arr


def explain_instance(model, scaler, feature_names, X_row: pd.DataFrame, needs_scaling: bool,
                      top_n=5, background: pd.DataFrame = None):
    """Return the top_n features pushing this single prediction up/down.

    `background` should be a representative sample of training rows (needed
    by LinearExplainer/model-agnostic explainers to measure each feature's
    baseline). Tree explainers ignore it. Falls back to a duplicated single
    row if none is supplied (works, but less accurate for linear models).
    """
    if background is None:
        background = pd.concat([X_row] * 50, ignore_index=True)

    X_bg = scaler.transform(background) if needs_scaling else background.values
    X_input = scaler.transform(X_row) if needs_scaling else X_row.values
    explainer = build_explainer(model, X_bg)
    shap_vals = get_shap_values(explainer, X_input)[0]

    contributions = list(zip(feature_names, shap_vals, X_row.iloc[0].values))
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    reasons = []
    for name, val, raw_val in contributions[:top_n]:
        direction = "increases" if val > 0 else "decreases"
        reasons.append(
            {
                "feature": name,
                "value": float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
                "impact": round(float(val), 4),
                "direction": direction,
            }
        )
    return reasons


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    model, scaler, feature_names = load_artifacts()

    df = pd.read_csv("data/telco_churn_features.csv")
    X = df.drop(columns=["Churn"])
    sample = X.sample(n=min(300, len(X)), random_state=42)

    import json
    with open(f"{MODELS_DIR}/model_metadata.json") as f:
        needs_scaling = json.load(f)["needs_scaling"]

    X_input = scaler.transform(sample) if needs_scaling else sample.values
    explainer = build_explainer(model, X_input)
    shap_values = get_shap_values(explainer, X_input)

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {REPORTS_DIR}/shap_summary.png")

    # Demo: explain one high-risk customer
    demo_idx = df["Churn"].idxmax()
    demo_row = X.iloc[[demo_idx]]
    background_sample = X.sample(n=100, random_state=1)
    reasons = explain_instance(model, scaler, feature_names, demo_row, needs_scaling, background=background_sample)
    print("\nExample explanation for one customer:")
    for r in reasons:
        print(f"  {r['feature']:<30} value={r['value']!r:<10} impact={r['impact']:+.4f} ({r['direction']} churn risk)")


if __name__ == "__main__":
    main()
