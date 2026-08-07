"""Load sample customer data used to exercise the prediction pipeline."""

import pandas as pd

from src.config.settings import DATA_DIR


def load_sample() -> pd.DataFrame:
    """Load a single sample customer row for a quick prediction.

    Reads the cleaned Telco customer churn dataset and returns its first row
    with the target columns (``Churn Label`` and ``Churn Value``) removed so
    the sample matches the feature set expected by the model.

    Returns:
        A DataFrame containing one row of sample customer features.
    """
    df = pd.read_csv(
        DATA_DIR / "processed" / "cleaned_telco_customer_churn.csv"
    )

    sample = df.drop(
        columns=[
            "Churn Label",
            "Churn Value",
        ]
    ).head(1)

    return sample
