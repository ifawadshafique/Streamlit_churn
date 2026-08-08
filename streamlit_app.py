"""Customer Churn Prediction - Streamlit application.

This UI intentionally reuses the project's existing model, scaler,
preprocessing, feature engineering and SHAP explanation logic.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(SRC_DIR))

from explain import explain_instance  # noqa: E402
from feature_engineering import build_feature_table  # noqa: E402
from preprocessing import clean  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Professional, restrained styling. No icons/emojis are used in the UI.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            width: 100%;
            max-width: 1400px;
            box-sizing: border-box;
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 650;
            line-height: 1.3;
            margin: 1rem 0 .35rem;
        }

        .section-caption {
            color: #64748b;
            font-size: .88rem;
            line-height: 1.5;
            margin-bottom: .8rem;
        }

        .risk-high {
            padding: 14px 16px;
            border-radius: 8px;
            background: #fff1f2;
            border: 1px solid #fecdd3;
            color: #9f1239;
            box-sizing: border-box;
            width: 100%;
        }

        .risk-medium {
            padding: 14px 16px;
            border-radius: 8px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
            box-sizing: border-box;
            width: 100%;
        }

        .risk-low {
            padding: 14px 16px;
            border-radius: 8px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            box-sizing: border-box;
            width: 100%;
        }

        .small-note {
            color: #64748b;
            font-size: .82rem;
            line-height: 1.4;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 14px;
            background: white;
            box-sizing: border-box;
        }

        .report-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 8px;
            background: #fff;
            box-sizing: border-box;
            width: 100%;
        }

        /* Prevent any container from clipping text */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .block-container,
        .stMarkdown {
            overflow: visible !important;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model/data loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    required = [
        MODELS_DIR / "churn_model.joblib",
        MODELS_DIR / "scaler.joblib",
        MODELS_DIR / "feature_names.joblib",
        MODELS_DIR / "model_metadata.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    model = joblib.load(MODELS_DIR / "churn_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    return model, scaler, feature_names, metadata


@st.cache_data(show_spinner=False)
def load_background():
    path = DATA_DIR / "telco_churn_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.drop(columns=["Churn"], errors="ignore")
    return df.sample(n=min(150, len(df)), random_state=1)


@st.cache_data(show_spinner=False)
def load_training_data():
    path = DATA_DIR / "telco_churn.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def predict_dataframe(raw_df, model, scaler, feature_names, needs_scaling):
    """Use the original cleaning, feature engineering and model logic."""
    cleaned = clean(raw_df, verbose=False)
    featured = build_feature_table(cleaned)
    featured = featured.reindex(columns=feature_names, fill_value=0)
    X_input = scaler.transform(featured) if needs_scaling else featured.values
    proba = model.predict_proba(X_input)[:, 1]
    return proba, cleaned, featured


# Cache predictions by file content. Re-running the page will not retrain or
# recompute the same uploaded file unless its contents actually change.
@st.cache_data(show_spinner=False, max_entries=4)
def process_uploaded_bytes(file_bytes: bytes, filename: str, model_name: str):
    del model_name  # cache key only; model itself lives in cache_resource.
    if filename.lower().endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
    return raw_df


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def risk_label(probability: float) -> str:
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.30:
        return "Medium Risk"
    return "Low Risk"


def risk_class(probability: float) -> str:
    if probability >= 0.70:
        return "risk-high"
    if probability >= 0.30:
        return "risk-medium"
    return "risk-low"


def make_results(raw_df, model, scaler, feature_names, needs_scaling):
    work_df = raw_df.drop(columns=["Churn"], errors="ignore")
    proba, cleaned_df, featured = predict_dataframe(
        work_df, model, scaler, feature_names, needs_scaling
    )
    result_df = cleaned_df.copy()
    result_df["ChurnProbability"] = proba
    result_df["RiskLevel"] = [risk_label(float(p)) for p in proba]
    result_df["Prediction"] = [
        "Likely to Churn" if p >= 0.50 else "Likely to Stay" for p in proba
    ]
    return result_df, work_df, featured


def paginate_dataframe(df: pd.DataFrame, key: str, page_size: int = 25):
    """Render only one page to keep large result tables responsive."""
    if df.empty:
        st.info("No records to display.")
        return

    total_pages = max(1, (len(df) + page_size - 1) // page_size)
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=f"{key}_page",
        label_visibility="collapsed",
    )
    start = (int(page) - 1) * page_size
    end = min(start + page_size, len(df))

    st.caption(f"Showing rows {start + 1:,}–{end:,} of {len(df):,}")
    st.dataframe(
        df.iloc[start:end],
        use_container_width=True,
        hide_index=True,
        height=430,
    )


def reasons_chart(reasons):
    if not reasons:
        return
    frame = pd.DataFrame(reasons).sort_values("impact")
    frame["direction"] = frame["impact"].apply(lambda x: "Increases churn" if x > 0 else "Decreases churn")
    fig = px.bar(
        frame,
        x="impact",
        y="feature",
        orientation="h",
        color="direction",
        hover_data=["value"],
        labels={"impact": "SHAP impact", "feature": "Feature", "value": "Value"},
        title="Top factors influencing this prediction",
    )
    fig.update_layout(height=390, legend_title_text="", margin=dict(l=10, r=10, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def download_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()


def show_reasons(model, scaler, feature_names, featured, needs_scaling, background):
    reasons = explain_instance(
        model,
        scaler,
        feature_names,
        featured,
        needs_scaling,
        top_n=8,
        background=background,
    )
    reasons_chart(reasons)
    reason_table = pd.DataFrame(reasons)
    if not reason_table.empty:
        reason_table["direction"] = reason_table["direction"].str.title()
        st.dataframe(reason_table, use_container_width=True, hide_index=True)


def single_customer_form():
    with st.form("single_customer"):
        st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Enter the customer attributes used by the existing model.</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 4)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        with c2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with c3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=95.0, step=0.5)
            total = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=float(monthly * max(tenure, 1)),
                step=1.0,
            )

        submitted = st.form_submit_button("Run Churn Assessment", type="primary", use_container_width=True)

    if not submitted:
        return None

    return pd.DataFrame(
        [{
            "customerID": "manual-entry",
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }]
    )


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def render_overview(metadata, training_df):
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Model performance and the current training dataset.</div>',
        unsafe_allow_html=True,
    )

    metrics = metadata.get("tuned_metrics", {})
    cols = st.columns(5)
    cols[0].metric("Selected model", metadata.get("best_model", "N/A"))
    cols[1].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    cols[2].metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    cols[3].metric("Recall", f"{metrics.get('recall', 0):.1%}")
    cols[4].metric("Features", metadata.get("n_features", "N/A"))

    left, right = st.columns(2)
    with left:
        results = pd.DataFrame(metadata.get("all_model_results", {})).T.reset_index()
        results = results.rename(columns={"index": "Model"})
        fig = px.bar(
            results.sort_values("roc_auc"),
            x="roc_auc",
            y="Model",
            orientation="h",
            text="roc_auc",
            labels={"roc_auc": "ROC-AUC", "Model": ""},
            title="Model comparison",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=410, margin=dict(l=10, r=25, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if training_df is not None and "Churn" in training_df.columns:
            churn_counts = training_df["Churn"].value_counts().rename_axis("Churn").reset_index(name="Customers")
            fig = px.bar(
                churn_counts,
                x="Churn",
                y="Customers",
                text="Customers",
                labels={"Churn": "Churn status", "Customers": "Customers"},
                title="Training dataset: churn distribution",
            )
            fig.update_layout(height=410, margin=dict(l=10, r=10, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)

    if training_df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Training rows", f"{len(training_df):,}")
        c2.metric("Churn rate", f"{(training_df['Churn'].eq('Yes').mean() if training_df['Churn'].dtype == 'object' else training_df['Churn'].mean()):.1%}")
        c3.metric("Test rows", f"{metadata.get('test_size', 0):,}")


def render_batch(model, scaler, feature_names, metadata, background):
    st.markdown('<div class="section-title">Batch prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Upload a CSV or Excel file. The model uses the existing preprocessing and feature-engineering pipeline.</div>',
        unsafe_allow_html=True,
    )

    template_path = DATA_DIR / "telco_churn.csv"
    if template_path.exists():
        sample = pd.read_csv(template_path).head(10)
        st.download_button(
            "Download input template",
            sample.to_csv(index=False).encode("utf-8"),
            "customer_churn_input_template.csv",
            "text/csv",
        )

    uploaded = st.file_uploader(
        "Customer file",
        type=["csv", "xlsx", "xls"],
        help="For large files, results are displayed using pagination so the browser does not render every row at once.",
    )

    if uploaded is None:
        st.info("Choose a file to begin. Prediction starts only after you click Process file.")
        return

    size_mb = uploaded.size / (1024 * 1024)
    st.caption(f"Selected file: {uploaded.name} ({size_mb:.2f} MB)")

    if st.button("Process file", type="primary", key="process_batch"):
        try:
            with st.spinner("Reading and scoring customers..."):
                raw_df = process_uploaded_bytes(uploaded.getvalue(), uploaded.name, metadata.get("best_model", "model"))
                result_df, work_df, _ = make_results(
                    raw_df, model, scaler, feature_names, metadata["needs_scaling"]
                )
            st.session_state["batch_result"] = result_df
            st.session_state["batch_work"] = work_df
            st.session_state["batch_name"] = uploaded.name
        except Exception as exc:
            st.error(f"Could not process the uploaded file: {exc}")
            return

    result_df = st.session_state.get("batch_result")
    work_df = st.session_state.get("batch_work")
    if result_df is None:
        return

    proba = result_df["ChurnProbability"].to_numpy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(result_df):,}")
    c2.metric("High risk", f"{(proba >= 0.70).sum():,}")
    c3.metric("Medium risk", f"{((proba >= 0.30) & (proba < 0.70)).sum():,}")
    c4.metric("Average risk", f"{proba.mean():.1%}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            result_df,
            x="ChurnProbability",
            nbins=30,
            labels={"ChurnProbability": "Churn probability"},
            title="Churn probability distribution",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        risk_counts = result_df["RiskLevel"].value_counts().reindex(
            ["High Risk", "Medium Risk", "Low Risk"], fill_value=0
        ).rename_axis("Risk").reset_index(name="Customers")
        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Customers",
            text="Customers",
            title="Risk segmentation",
            labels={"Risk": "Risk level", "Customers": "Customers"},
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Customer results</div>', unsafe_allow_html=True)
    id_col = "customerID" if "customerID" in result_df.columns else result_df.columns[0]
    search = st.text_input("Search customer ID", key="batch_search")
    filtered = result_df.copy()
    if search:
        filtered = filtered[filtered[id_col].astype(str).str.contains(search, case=False, na=False)]

    # Paginated table: only 25 rows are rendered at a time.
    paginate_dataframe(filtered.sort_values("ChurnProbability", ascending=False), "batch_results", page_size=25)

    st.markdown('<div class="section-title">Customer explanation</div>', unsafe_allow_html=True)
    if len(filtered) == 0:
        st.info("No customer matches the search.")
    else:
        selected_id = st.selectbox(
            "Customer",
            filtered[id_col].astype(str).head(500).tolist(),
            key="batch_customer_select",
        )
        selected_positions = result_df.index[result_df[id_col].astype(str) == selected_id].tolist()
        if selected_positions:
            position = selected_positions[0]
            row = result_df.loc[[position]]
            p = float(row["ChurnProbability"].iloc[0])
            st.markdown(
                f'<div class="{risk_class(p)}"><strong>{risk_label(p)}</strong> — churn probability {p:.1%}. '
                f'Prediction: {row["Prediction"].iloc[0]}.</div>',
                unsafe_allow_html=True,
            )

            raw_row = work_df.loc[[position]]
            featured_row = build_feature_table(clean(raw_row, verbose=False)).reindex(
                columns=feature_names, fill_value=0
            )
            show_reasons(model, scaler, feature_names, featured_row, metadata["needs_scaling"], background)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download predictions as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            "churn_predictions.csv",
            "text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download predictions as Excel",
            download_excel(result_df),
            "churn_predictions.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def render_single(model, scaler, feature_names, metadata, background):
    st.markdown('<div class="section-title">Single customer assessment</div>', unsafe_allow_html=True)
    raw_row = single_customer_form()
    if raw_row is None:
        return

    try:
        proba, _, featured = predict_dataframe(
            raw_row, model, scaler, feature_names, metadata["needs_scaling"]
        )
        p = float(proba[0])
        st.markdown(
            f'<div class="{risk_class(p)}"><strong>{risk_label(p)}</strong> — churn probability {p:.1%}. '
            f'Prediction: {"Likely to Churn" if p >= 0.5 else "Likely to Stay"}.</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(p, 0.0), 1.0))
        show_reasons(model, scaler, feature_names, featured, metadata["needs_scaling"], background)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


def render_reports(training_df):
    st.markdown('<div class="section-title">Reports and analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Existing project reports generated by the EDA and explainability pipeline.</div>',
        unsafe_allow_html=True,
    )

    report_files = [
        ("Churn by category", "churn_by_category.png"),
        ("Correlation heatmap", "correlation_heatmap.png"),
        ("Tenure and charges distribution", "tenure_charges_distribution.png"),
        ("SHAP global feature importance", "shap_summary.png"),
    ]

    for start in range(0, len(report_files), 2):
        cols = st.columns(2)
        for col, (title, filename) in zip(cols, report_files[start:start + 2]):
            path = REPORTS_DIR / filename
            with col:
                st.markdown(f"**{title}**")
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.warning(f"Report not found: {filename}")

    if training_df is not None:
        st.markdown('<div class="section-title">Additional dataset analysis</div>', unsafe_allow_html=True)
        if "MonthlyCharges" in training_df.columns and "tenure" in training_df.columns:
            fig = px.scatter(
                training_df,
                x="tenure",
                y="MonthlyCharges",
                color="Churn" if "Churn" in training_df.columns else None,
                opacity=0.55,
                labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges"},
                title="Tenure versus monthly charges",
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)


def render_model(metadata):
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    results = pd.DataFrame(metadata.get("all_model_results", {})).T
    results.index.name = "Model"
    st.dataframe(results.style.format({c: "{:.3f}" for c in results.columns}), use_container_width=True)

    melted = results.reset_index().melt(
        id_vars="Model",
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model performance by metric",
        labels={"Score": "Score"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)

    selected = metadata.get("best_model", "N/A")
    tuned = metadata.get("tuned_metrics", {})
    st.markdown(f"**Selected model:** {selected}")
    st.json(tuned)

    st.markdown('<div class="section-title">Training configuration</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", f"{metadata.get('train_size', 0):,}")
    c2.metric("Test rows", f"{metadata.get('test_size', 0):,}")
    c3.metric("Model features", metadata.get("n_features", "N/A"))


def main():
    try:
        model, scaler, feature_names, metadata = load_artifacts()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    background = load_background()
    training_df = load_training_data()

    st.markdown('<div class="app-title">Customer Churn Analytics</div>', unsafe_allow_html=True)
    st.markdown(
    """
    <h1 style="
        font-size: clamp(1.5rem, 4vw, 2rem);
        font-weight: 700;
        line-height: 1.25;
        letter-spacing: -0.02em;
        margin: 0 0 0.15rem 0;
        padding: 0;
        overflow: visible !important;
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        white-space: normal;
    ">Customer Churn Analytics</h1>
    <p style="
        color: #64748b;
        font-size: 0.96rem;
        line-height: 1.5;
        margin: 0 0 1.4rem 0;
        padding: 0;
        overflow: visible !important;
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
    ">Customer risk scoring, model diagnostics, explainability and project reports.</p>
    """,
    unsafe_allow_html=True,
)

    with st.sidebar:
        st.markdown("### Customer Churn")
        st.caption("Machine learning analytics dashboard")
        st.divider()
        st.markdown(f"**Selected model**  \\n{metadata.get('best_model', 'N/A')}")
        st.markdown(f"**ROC-AUC**  \\n{metadata.get('tuned_metrics', {}).get('roc_auc', 0):.3f}")
        st.divider()
        st.caption("The dashboard uses the existing trained artifacts and preprocessing pipeline.")

    tabs = st.tabs([
        "Overview",
        "Batch Prediction",
        "Single Customer",
        "Reports",
        "Model Performance",
    ])

    with tabs[0]:
        render_overview(metadata, training_df)
    with tabs[1]:
        render_batch(model, scaler, feature_names, metadata, background)
    with tabs[2]:
        render_single(model, scaler, feature_names, metadata, background)
    with tabs[3]:
        render_reports(training_df)
    with tabs[4]:
        render_model(metadata)


if __name__ == "__main__":
    main()