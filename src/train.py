"""
Steps 5-6: Train multiple models, evaluate, and tune the winner.

Usage:
    python src/train.py
"""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
REPORTS_DIR = "reports"
TARGET = "Churn"


def load_features(path="data/telco_churn_features.csv"):
    df = pd.read_csv(path)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def get_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=(1),  # set properly below after we see class balance
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1
        ),
    }


def evaluate(model, X_test, y_test, needs_scaling=False, scaler=None):
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    proba = model.predict_proba(X_eval)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("customer-churn-prediction")

    X, y = load_features()
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = get_candidate_models()
    models["XGBoost"].set_params(scale_pos_weight=scale_pos_weight)

    results = {}
    fitted = {}
    print(f"{'Model':<20}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>9}   Time(s)")
    for name, model in models.items():
        t0 = time.time()
        needs_scaling = name == "LogisticRegression"
        model.fit(X_train_scaled if needs_scaling else X_train, y_train)
        metrics = evaluate(model, X_test, y_test, needs_scaling, scaler)
        dt = time.time() - t0
        results[name] = metrics
        fitted[name] = model
        print(
            f"{name:<20}{metrics['accuracy']:>10}{metrics['precision']:>11}"
            f"{metrics['recall']:>9}{metrics['f1']:>8}{metrics['roc_auc']:>9}   {dt:.1f}"
        )

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"candidate-{name}"):
                mlflow.log_param("model_type", name)
                mlflow.log_params({k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None})
                mlflow.log_metrics(metrics)
                mlflow.log_metric("train_time_seconds", dt)

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_name} ({results[best_name]['roc_auc']})")

    # ---- Step 6: Hyperparameter tuning of the winning model ----
    print(f"\nTuning {best_name} with RandomizedSearchCV...")
    if best_name == "XGBoost":
        base = XGBClassifier(eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight)
        param_dist = {
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [200, 300, 400, 600],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        X_fit, y_fit = X_train, y_train
    elif best_name == "LightGBM":
        base = LGBMClassifier(random_state=42, verbosity=-1)
        param_dist = {
            "max_depth": [3, 4, 5, 6, 8, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [200, 300, 400, 600],
            "num_leaves": [15, 31, 63],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        }
        X_fit, y_fit = X_train, y_train
    elif best_name == "RandomForest":
        base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": [200, 300, 400, 600],
            "max_depth": [6, 8, 10, 14, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        X_fit, y_fit = X_train, y_train
    else:
        base = fitted[best_name]
        param_dist = None
        X_fit, y_fit = (X_train_scaled if best_name == "LogisticRegression" else X_train), y_train

    if param_dist:
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=15,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_fit, y_fit)
        tuned_model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    else:
        tuned_model = base

    needs_scaling = best_name == "LogisticRegression"
    tuned_metrics = evaluate(tuned_model, X_test, y_test, needs_scaling, scaler)
    print(f"Tuned {best_name} metrics: {tuned_metrics}")

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"tuned-{best_name}"):
            mlflow.log_param("model_type", f"{best_name}-tuned")
            mlflow.log_metrics(tuned_metrics)
            mlflow.sklearn.log_model(tuned_model, "model")
        print("Logged tuned model run to MLflow (view with: mlflow ui)")

    # ---- Persist everything the API/dashboard need ----
    joblib.dump(tuned_model, f"{MODELS_DIR}/churn_model.joblib")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")
    joblib.dump(feature_names, f"{MODELS_DIR}/feature_names.joblib")

    with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "needs_scaling": needs_scaling,
                "all_model_results": results,
                "tuned_metrics": tuned_metrics,
                "n_features": len(feature_names),
                "train_size": len(X_train),
                "test_size": len(X_test),
            },
            f,
            indent=2,
        )

    print(f"\nSaved model artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
