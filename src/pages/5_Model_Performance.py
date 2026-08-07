"""Streamlit page for reviewing the trained model's performance."""

from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_auc_score,
)

from src.config.settings import DATA_DIR, MODEL_PATH


@st.cache_resource(show_spinner=False)
def _load_model() -> Any:
    """Load the trained churn prediction model."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def _load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test features and target labels."""
    X_test = pd.read_csv(DATA_DIR / "processed" / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "processed" / "y_test.csv").squeeze()
    return X_test, y_test


def _render_figure(fig: Figure) -> None:
    """Render a matplotlib figure and release its resources."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Display the confusion matrix for the test predictions."""
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)

    _render_figure(fig)


def _render_roc_and_precision_recall(
    y_test: pd.Series,
    y_prob: np.ndarray,
) -> None:
    """Display the ROC and precision-recall curves side by side."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")

        fig, ax = plt.subplots(figsize=(6, 5))

        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)

        _render_figure(fig)

    with col2:
        st.subheader("Precision-Recall Curve")

        fig, ax = plt.subplots(figsize=(6, 5))

        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)

        _render_figure(fig)


def _render_probability_chart(y_prob: np.ndarray) -> None:
    """Display the predicted churn probabilities as a line chart."""
    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({"Probability": y_prob})

    st.line_chart(probability_df)


st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Performance")
st.caption("Performance of the churn model on the held-out test set.")

model = _load_model()
X_test, y_test = _load_test_data()

y_pred: np.ndarray = model.predict(X_test)
y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]

roc_auc: float = roc_auc_score(y_test, y_prob)

col1, col2, col3 = st.columns(3)

col1.metric("Test Samples", len(X_test))
col2.metric("Features", X_test.shape[1])
col3.metric("ROC-AUC", f"{roc_auc:.4f}")

st.markdown("---")

_render_confusion_matrix(y_test, y_pred)
_render_roc_and_precision_recall(y_test, y_prob)
_render_probability_chart(y_prob)
