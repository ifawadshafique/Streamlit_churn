"""
Step 9: Streamlit dashboard.

Run with:
    streamlit run app/dashboard.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from feature_engineering import build_feature_table  # noqa: E402
from explain import explain_instance  # noqa: E402
from preprocessing import clean  # noqa: E402

MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide", page_icon="📉")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "churn_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, feature_names, metadata


@st.cache_data
def load_background():
    path = ROOT / "data" / "telco_churn_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.drop(columns=["Churn"]).sample(n=min(150, len(df)), random_state=1)


def predict_dataframe(raw_df, model, scaler, feature_names, needs_scaling):
    cleaned = clean(raw_df, verbose=False)
    featured = build_feature_table(cleaned)
    featured = featured.reindex(columns=feature_names, fill_value=0)
    X_input = scaler.transform(featured) if needs_scaling else featured.values
    proba = model.predict_proba(X_input)[:, 1]
    return proba, cleaned, featured


def main():
    st.title("📉 Customer Churn Prediction Dashboard")
    st.caption("End-to-end churn platform — upload customers, get risk scores, and see *why*.")

    if not (MODELS_DIR / "churn_model.joblib").exists():
        st.error(
            "No trained model found. Run the pipeline first:\n\n"
            "`python src/generate_data.py && python src/preprocessing.py && "
            "python src/feature_engineering.py && python src/train.py`"
        )
        st.stop()

    model, scaler, feature_names, metadata = load_artifacts()
    background = load_background()
    needs_scaling = metadata["needs_scaling"]

    tab_batch, tab_single, tab_model = st.tabs(["📁 Batch Upload", "🔍 Single Customer", "🧠 Model Info"])

    # ---------------- Batch tab ----------------
    with tab_batch:
        st.subheader("Upload a CSV of customers")
        st.caption(
            "Expected columns match the Telco schema (gender, tenure, Contract, "
            "MonthlyCharges, etc). A sample file is at data/telco_churn.csv."
        )
        uploaded = st.file_uploader("Choose File", type="csv")

        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            work_df = raw_df.drop(columns=["Churn"], errors="ignore")
            proba, cleaned_df, _ = predict_dataframe(work_df, model, scaler, feature_names, needs_scaling)

            result_df = cleaned_df.copy()
            result_df["ChurnProbability"] = proba.round(4)
            result_df["Prediction"] = ["Likely to Churn" if p >= 0.5 else "Likely to Stay" for p in proba]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers", len(result_df))
            c2.metric("High Risk (>=50%)", int((proba >= 0.5).sum()))
            c3.metric("Average Risk", f"{proba.mean():.1%}")

            fig = px.histogram(
                result_df, x="ChurnProbability", nbins=30,
                title="Churn Probability Distribution", color_discrete_sequence=["#c44e52"],
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Customer Search")
            id_col = "customerID" if "customerID" in result_df.columns else result_df.columns[0]
            selected_id = st.selectbox("Select a customer", result_df[id_col].astype(str).tolist())
            row = result_df[result_df[id_col].astype(str) == selected_id].iloc[[0]]
            st.metric("Prediction", row["Prediction"].values[0], f"{row['ChurnProbability'].values[0]:.1%}")

            raw_row = work_df[work_df[id_col].astype(str) == selected_id].iloc[[0]] if id_col in work_df.columns else work_df.iloc[[row.index[0]]]
            featured_row = build_feature_table(clean(raw_row, verbose=False)).reindex(columns=feature_names, fill_value=0)
            reasons = explain_instance(model, scaler, feature_names, featured_row, needs_scaling, background=background)
            st.write("**Top reasons:**")
            for r in reasons:
                icon = "🔺" if r["direction"] == "increases" else "🔻"
                st.write(f"{icon} `{r['feature']}` = {r['value']} (impact {r['impact']:+.3f})")

            st.subheader("Full Results")
            st.dataframe(result_df.sort_values("ChurnProbability", ascending=False), use_container_width=True)
            st.download_button(
                "Download results as CSV",
                result_df.to_csv(index=False).encode(),
                "churn_predictions.csv",
                "text/csv",
            )
        else:
            st.info("Upload a CSV to see predictions, or try the Single Customer tab.")

    # ---------------- Single customer tab ----------------
    with tab_single:
        st.subheader("Score one customer")
        with st.form("single_customer"):
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 4)
            with col2:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                payment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
            with col3:
                monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 95.0)
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

            submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

        if submitted:
            raw_row = pd.DataFrame([{
                "customerID": "manual-entry",
                "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone_service, "MultipleLines": "No",
                "InternetService": internet, "OnlineSecurity": online_security, "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": tech_support, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
                "TotalCharges": monthly_charges * max(tenure, 1),
            }])
            proba, _, featured = predict_dataframe(raw_row, model, scaler, feature_names, needs_scaling)
            p = proba[0]

            st.metric("Churn Probability", f"{p:.1%}", "Likely to Churn" if p >= 0.5 else "Likely to Stay")
            st.progress(min(float(p), 1.0))

            reasons = explain_instance(model, scaler, feature_names, featured, needs_scaling, background=background)
            st.write("**Why:**")
            for r in reasons:
                icon = "🔺" if r["direction"] == "increases" else "🔻"
                st.write(f"{icon} `{r['feature']}` = {r['value']} (impact {r['impact']:+.3f})")

    # ---------------- Model info tab ----------------
    with tab_model:
        st.subheader("Model comparison")
        results_df = pd.DataFrame(metadata["all_model_results"]).T
        st.dataframe(results_df, use_container_width=True)
        fig = px.bar(
            results_df.reset_index().rename(columns={"index": "model"}),
            x="model", y="roc_auc", title="ROC-AUC by model", color="model",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Selected model: {metadata['best_model']}")
        st.json(metadata["tuned_metrics"])

        shap_path = ROOT / "reports" / "shap_summary.png"
        if shap_path.exists():
            st.subheader("Global feature importance (SHAP)")
            st.image(str(shap_path), use_container_width=True)


if __name__ == "__main__":
    main()
