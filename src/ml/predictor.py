"""Predict customer churn using the trained churn model and preprocessor."""

import pandas as pd

from src.utils.model_loader import ModelLoader


class ChurnPredictor:
    """Wrap the trained churn model and preprocessor into a simple interface.

    The preprocessor applies the same transformations used during model
    training, and the wrapped model produces both a class prediction and a
    churn probability.
    """

    def __init__(self) -> None:
        """Load the trained model and its matching preprocessor once."""
        self.model = ModelLoader.get_model()
        self.preprocessor = ModelLoader.get_preprocessor()

    def predict(self, customer: pd.DataFrame) -> dict[str, int | float]:
        """Predict whether a single customer will churn.

        Args:
            customer: A DataFrame containing one row of customer features in
                the same format expected by the preprocessor.

        Returns:
            A dictionary with the predicted churn class and the probability
            that the customer will churn:
                - ``prediction``: 1 if the customer is predicted to churn,
                  0 otherwise.
                - ``probability``: the model's estimated churn probability.
        """
        transformed = self.preprocessor.transform(customer)

        prediction = self.model.predict(transformed)[0]
        probability = self.model.predict_proba(transformed)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
        }

    def predict_batch(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Predict churn for multiple customers in a single pass.

        Args:
            customers: A DataFrame with one customer per row, in the same
                feature format expected by the preprocessor.

        Returns:
            A DataFrame with one row per input customer and two columns:
                - ``prediction``: 1 if the customer is predicted to churn,
                  0 otherwise.
                - ``probability``: the model's estimated churn probability.
        """
        transformed = self.preprocessor.transform(customers)

        return pd.DataFrame(
            {
                "prediction": self.model.predict(transformed),
                "probability": self.model.predict_proba(transformed)[:, 1],
            }
        )
