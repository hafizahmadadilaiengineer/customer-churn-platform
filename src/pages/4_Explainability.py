"""Streamlit page for explaining model predictions with SHAP."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.utils.feature_engineering import engineer_features
from src.utils.model_loader import ModelLoader


@st.cache_data(show_spinner=False)
def _read_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read the uploaded customer CSV file."""
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def _compute_shap_values(transformed: np.ndarray) -> tuple[Any, Any]:
    """Compute SHAP values for the whole dataset.

    Cached so that changing the selected customer does not recompute the
    (expensive) SHAP values on every rerun.

    Args:
        transformed: The preprocessed feature matrix for all customers.

    Returns:
        A tuple of ``(shap_values, expected_value)`` as returned by the
        SHAP ``TreeExplainer``.
    """
    model = ModelLoader.get_model()

    explainer = shap.TreeExplainer(model)

    return explainer.shap_values(transformed), explainer.expected_value


def _render_metrics(prediction: int, probability: float) -> None:
    """Display the prediction metrics for the selected customer."""
    if probability >= 0.60:
        risk = "High"
    elif probability >= 0.30:
        risk = "Medium"
    else:
        risk = "Low"

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction", "Churn" if prediction else "No Churn")
    col2.metric("Probability", f"{probability:.2%}")
    col3.metric("Risk", risk)


def _render_global_importance(shap_values: Any, transformed: np.ndarray) -> None:
    """Display the global feature importance summary plot."""
    st.subheader("Global Feature Importance")

    plt.figure()
    shap.summary_plot(shap_values[:, :, 1], transformed, show=False)
    st.pyplot(plt.gcf())
    plt.clf()


def _render_customer_explanation(
    expected_value: Any,
    shap_values: Any,
    transformed: np.ndarray,
    customer: int,
    feature_names: list[str],
) -> None:
    """Display the waterfall explanation for the selected customer."""
    st.subheader("Customer Explanation")

    explanation = shap.Explanation(
        values=shap_values[customer, :, 1],
        base_values=expected_value[1],
        data=transformed[customer],
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(plt.gcf())
    plt.clf()


def _render_top_features(
    shap_values: Any,
    customer: int,
    feature_names: list[str],
) -> None:
    """Display and download the top features for the selected customer."""
    st.subheader("Top 15 Important Features")

    report = pd.DataFrame(
        {
            "Feature": feature_names,
            "Impact": shap_values[customer, :, 1],
        }
    )

    report = report.reindex(
        report["Impact"].abs().sort_values(ascending=False).index
    )

    st.dataframe(report.head(15), use_container_width=True)

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Explanation",
        csv,
        "customer_explanation.csv",
        "text/csv",
    )


st.set_page_config(
    page_title="Explainability",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Model Explainability")

uploaded_file = st.file_uploader("Upload Customer CSV", type="csv")

if uploaded_file:
    df = _read_csv(uploaded_file)
    df = engineer_features(df)

    customer: int = st.selectbox("Customer Index", df.index)

    preprocessor = ModelLoader.get_preprocessor()
    transformed = preprocessor.transform(df)

    model = ModelLoader.get_model()
    prediction = model.predict(transformed[customer : customer + 1])[0]
    probability = model.predict_proba(transformed[customer : customer + 1])[0][1]

    _render_metrics(prediction, probability)

    feature_names = list(preprocessor.get_feature_names_out())

    with st.spinner("Computing SHAP values..."):
        shap_values, expected_value = _compute_shap_values(transformed)

    _render_global_importance(shap_values, transformed)
    _render_customer_explanation(
        expected_value,
        shap_values,
        transformed,
        customer,
        feature_names,
    )
    _render_top_features(shap_values, customer, feature_names)
else:
    st.info("Upload a CSV file to explore model explanations.")
