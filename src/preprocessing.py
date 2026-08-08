"""
Step 2: Data Cleaning

Turns the raw Telco CSV into a clean, well-typed DataFrame:
- drops duplicate customer records
- fixes TotalCharges (stored as text in the raw file, with blanks for
  brand-new customers)
- fills/handles missing values
- normalizes the target column to 0/1
"""

import pandas as pd


def load_raw(path: str = "data/telco_churn.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    log = print if verbose else (lambda *a, **k: None)

    # 1. Duplicates
    before = len(df)
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset="customerID", keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    log(f"Dropped {before - len(df)} duplicate rows")

    # 2. TotalCharges arrives as a string with blank entries for
    #    customers with 0 tenure -> coerce to numeric, then fill.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_total = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )
    log(f"Filled {missing_total} missing TotalCharges values (MonthlyCharges x tenure)")

    # 3. Normalize target (only present for training data, not inference input)
    if "Churn" in df.columns and not pd.api.types.is_integer_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # 4. Type hygiene
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # 5. Drop the ID column for modeling (kept separately if needed)
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")
    cleaned = clean(raw)
    print(f"Clean shape: {cleaned.shape}")
    print(cleaned.isna().sum().sum(), "remaining nulls")
    cleaned.to_csv("data/telco_churn_clean.csv", index=False)
    print("Wrote data/telco_churn_clean.csv")
