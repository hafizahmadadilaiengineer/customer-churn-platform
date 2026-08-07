import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering required by the trained model.
    """

    df = df.copy()

    df["Avg Monthly Spend"] = df.apply(
        lambda row: 0
        if row["Tenure Months"] == 0
        else row["Total Charges"] / row["Tenure Months"],
        axis=1,
    )

    df["Customer Age Group"] = pd.cut(
        df["Tenure Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "New",
            "Growing",
            "Loyal",
            "Very Loyal",
        ],
    )

    return df