"""Business-level prediction service that adds a risk category."""

import pandas as pd

from src.ml.predictor import ChurnPredictor


class PredictionService:
    """Provide churn predictions together with a human-readable risk level."""

    def __init__(self) -> None:
        """Create the underlying churn predictor used by this service."""
        self.predictor = ChurnPredictor()

    def predict(self, customer: pd.DataFrame) -> dict[str, int | float | str]:
        """Predict churn for a customer and assign a risk category.

        The risk category is derived from the churn probability:
            - ``Low`` for probabilities below 0.30.
            - ``Medium`` for probabilities between 0.30 and 0.60.
            - ``High`` for probabilities of 0.60 or above.

        Args:
            customer: A DataFrame containing one row of customer features in
                the same format expected by the predictor.

        Returns:
            The predictor's result (``prediction`` and ``probability``) plus a
            ``risk`` key describing the churn risk level.
        """
        result = self.predictor.predict(customer)

        probability = result["probability"]

        if probability < 0.30:
            risk = "Low"
        elif probability < 0.60:
            risk = "Medium"
        else:
            risk = "High"

        result["risk"] = risk

        return result

    def predict_batch(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Predict churn for multiple customers and assign risk categories.

        The risk categories use the same thresholds as ``predict``:
            - ``Low`` for probabilities below 0.30.
            - ``Medium`` for probabilities between 0.30 and 0.60.
            - ``High`` for probabilities of 0.60 or above.

        Args:
            customers: A DataFrame with one customer per row, in the same
                feature format expected by the predictor.

        Returns:
            A DataFrame with one row per input customer and the columns
            ``prediction``, ``probability`` and ``risk``.
        """
        results = self.predictor.predict_batch(customers)

        results["risk"] = pd.cut(
            results["probability"],
            bins=[float("-inf"), 0.30, 0.60, float("inf")],
            labels=["Low", "Medium", "High"],
            right=False,
        ).astype(str)

        return results
