"""
Step 3: Exploratory Data Analysis

Produces a handful of the most decision-relevant plots and saves them to
reports/. Run after preprocessing.py.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT = "reports"


def churn_by(df, col, ax, rotate=0):
    rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=ax, palette="rocket")
    ax.set_ylabel("Churn rate")
    ax.set_title(f"Churn rate by {col}")
    ax.tick_params(axis="x", rotation=rotate)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("data/telco_churn_clean.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    churn_by(df, "Contract", axes[0, 0])
    churn_by(df, "InternetService", axes[0, 1])
    churn_by(df, "PaymentMethod", axes[1, 0], rotate=25)
    churn_by(df, "TechSupport", axes[1, 1])
    fig.tight_layout()
    fig.savefig(f"{OUT}/churn_by_category.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, ax=axes[0], palette=["#4c72b0", "#c44e52"])
    axes[0].set_title("Tenure distribution by churn")
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", multiple="stack", bins=30, ax=axes[1], palette=["#4c72b0", "#c44e52"])
    axes[1].set_title("Monthly charges distribution by churn")
    fig.tight_layout()
    fig.savefig(f"{OUT}/tenure_charges_distribution.png", dpi=140)
    plt.close(fig)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(f"{OUT}/correlation_heatmap.png", dpi=140)
    plt.close(fig)

    print("Saved plots to reports/:")
    print(" - churn_by_category.png")
    print(" - tenure_charges_distribution.png")
    print(" - correlation_heatmap.png")

    print(f"\nOverall churn rate: {df['Churn'].mean():.1%}")
    print(f"Month-to-month churn rate: {df[df.Contract=='Month-to-month']['Churn'].mean():.1%}")
    print(f"Two-year churn rate: {df[df.Contract=='Two year']['Churn'].mean():.1%}")


if __name__ == "__main__":
    main()
