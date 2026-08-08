"""
Generates a synthetic dataset that mirrors the structure and statistical
patterns of the IBM Telco Customer Churn dataset.

Why synthetic? This environment has no access to Kaggle. The generator below
encodes the same well-known churn drivers (month-to-month contracts, high
monthly charges, low tenure, no tech support/online security, fiber optic
service) so every downstream step (EDA, feature engineering, modeling,
SHAP) behaves the way it would on the real dataset.

If you have the real IBM Telco CSV, just drop it at
data/telco_churn.csv with the same column names and skip this script.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 7043  # same size as the real IBM Telco dataset


def generate(n=N, seed=42):
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, n)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def dependent_service(base_p_yes):
        out = np.empty(n, dtype=object)
        has_internet = internet_service != "No"
        out[~has_internet] = "No internet service"
        yes_mask = has_internet & (rng.random(n) < base_p_yes)
        out[has_internet & yes_mask] = "Yes"
        out[has_internet & ~yes_mask] = "No"
        return out

    online_security = dependent_service(0.35)
    online_backup = dependent_service(0.40)
    device_protection = dependent_service(0.40)
    tech_support = dependent_service(0.35)
    streaming_tv = dependent_service(0.45)
    streaming_movies = dependent_service(0.45)

    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58]),
    )

    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Base monthly charge driven by services subscribed
    base = 18.0
    base += np.where(phone_service == "Yes", 5, 0)
    base += np.where(multiple_lines == "Yes", 5, 0)
    base += np.where(internet_service == "DSL", 25, 0)
    base += np.where(internet_service == "Fiber optic", 45, 0)
    for svc in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
        base += np.where(svc == "Yes", 6, 0)
    monthly_charges = np.round(base + rng.normal(0, 5, n), 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 20, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    # ---- Churn probability model (encodes realistic churn drivers) ----
    logit = -1.6
    logit += np.where(contract == "Month-to-month", 1.05, 0)
    logit += np.where(contract == "One year", -0.35, 0)
    logit += np.where(contract == "Two year", -1.4, 0)
    logit += np.where(internet_service == "Fiber optic", 0.55, 0)
    logit += np.where(internet_service == "No", -0.6, 0)
    logit += -0.045 * tenure
    logit += 0.012 * (monthly_charges - 65)
    logit += np.where(online_security == "No", 0.35, 0)
    logit += np.where(tech_support == "No", 0.35, 0)
    logit += np.where(paperless_billing == "Yes", 0.25, 0)
    logit += np.where(payment_method == "Electronic check", 0.35, 0)
    logit += np.where(senior_citizen == 1, 0.25, 0)
    logit += np.where(partner == "No", 0.15, 0)
    logit += np.where(dependents == "No", 0.10, 0)
    logit += rng.normal(0, 0.6, n)  # noise

    prob_churn = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob_churn, "Yes", "No")

    customer_id = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}" for _ in range(n)]

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    # Inject a small amount of real-world messiness so the cleaning step
    # in preprocessing.py has actual work to do.
    dup_idx = rng.choice(df.index, 25, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Mirrors the real IBM dataset, where TotalCharges is stored as a string
    # column containing blank entries for brand-new customers.
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    missing_idx = rng.choice(df.index, 40, replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan

    blank_idx = rng.choice(df.index, 11, replace=False)
    df.loc[blank_idx, "TotalCharges"] = " "

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/telco_churn.csv", index=False)
    print(f"Wrote data/telco_churn.csv with {len(df)} rows, {df['Churn'].eq('Yes').mean():.1%} churn rate")
