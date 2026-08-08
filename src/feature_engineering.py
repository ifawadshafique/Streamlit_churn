"""
Step 4: Feature Engineering

Adds derived features on top of the cleaned data, then encodes
categorical columns for modeling.
"""

import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def tenure_group(t: int) -> str:
    if t <= 6:
        return "new (0-6mo)"
    if t <= 24:
        return "established (6-24mo)"
    if t <= 48:
        return "loyal (24-48mo)"
    return "veteran (48mo+)"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avoid divide-by-zero for brand new customers (tenure=0)
    safe_tenure = df["tenure"].replace(0, 1)

    df["AvgMonthlySpend"] = (df["TotalCharges"] / safe_tenure).round(2)
    df["TenureGroup"] = df["tenure"].apply(tenure_group)

    df["ServiceCount"] = df[SERVICE_COLUMNS].apply(
        lambda row: sum(v not in ("No", "No internet service", "No phone service") for v in row),
        axis=1,
    )

    df["HighValueCustomer"] = (
        (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75))
    ).astype(int)

    # Simple CLV estimate: what they've paid + expected remaining value
    # based on contract length (used purely as a modeling feature, not a
    # finance-grade CLV calculation)
    contract_months_remaining = df["Contract"].map(
        {"Month-to-month": 1, "One year": 6, "Two year": 12}
    )
    df["EstimatedCLV"] = (
        df["TotalCharges"] + df["MonthlyCharges"] * contract_months_remaining
    ).round(2)

    df["HasNoAddOns"] = (df["ServiceCount"] <= 1).astype(int)

    df["IsFiberNoSupport"] = (
        (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
    ).astype(int)

    return df


def encode(df: pd.DataFrame, categorical_columns=None) -> pd.DataFrame:
    """One-hot encode categorical columns (including the new TenureGroup)."""
    df = df.copy()
    cat_cols = list(categorical_columns or CATEGORICAL_COLUMNS) + ["TenureGroup"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    drop_cols = [c for c in ["customerID"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = encode(df)
    # bool -> int for model libraries that dislike bool dtype
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/telco_churn_clean.csv")
    featured = build_feature_table(df)
    print(f"Feature table shape: {featured.shape}")
    print(f"Columns added beyond one-hot: AvgMonthlySpend, ServiceCount, "
          f"HighValueCustomer, EstimatedCLV, HasNoAddOns, IsFiberNoSupport, TenureGroup(*)")
    featured.to_csv("data/telco_churn_features.csv", index=False)
    print("Wrote data/telco_churn_features.csv")
