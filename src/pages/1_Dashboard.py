"""Streamlit dashboard giving an overview of the churn dataset and model."""

from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.config.constants import APP_NAME, POSITIVE_CLASS, TARGET_COLUMN
from src.config.settings import DATA_DIR, MODEL_PATH


@st.cache_data(show_spinner=False)
def _load_dataset() -> pd.DataFrame:
    """Load the cleaned Telco customer churn dataset."""
    return pd.read_csv(
        DATA_DIR / "processed" / "cleaned_telco_customer_churn.csv"
    )


@st.cache_resource(show_spinner=False)
def _load_model() -> Any:
    """Load the trained churn prediction model."""
    return joblib.load(MODEL_PATH)


def _render_kpi_row(df: pd.DataFrame, model: Any) -> None:
    """Display the top-level KPI metrics in a row of four columns."""
    n_features: int = df.shape[1] - 2

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Features", n_features)
    col3.metric("Target", POSITIVE_CLASS)
    col4.metric("Model", type(model).__name__)


def _render_dataset_overview(df: pd.DataFrame) -> None:
    """Display key dataset statistics and data quality indicators."""
    st.subheader("Dataset Overview")
    st.caption("Key statistics and data quality indicators.")

    overview = pd.DataFrame(
        {
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
            "Value": [
                df.shape[0],
                df.shape[1],
                int(df.isnull().sum().sum()),
                int(df.duplicated().sum()),
            ],
        }
    )

    st.dataframe(overview, use_container_width=True, hide_index=True)


def _render_target_distribution(df: pd.DataFrame) -> None:
    """Display the distribution of the churn target column."""
    st.subheader("Target Distribution")
    st.caption("Number of customers per churn class.")

    target = df[TARGET_COLUMN].value_counts()

    st.bar_chart(target)


def _render_dataset_preview(df: pd.DataFrame) -> None:
    """Display the first rows of the dataset."""
    st.subheader("Dataset Preview")
    st.caption("First five rows of the cleaned dataset.")

    st.dataframe(df.head(), use_container_width=True)


def _render_sidebar() -> None:
    """Display the application status in the sidebar."""
    st.sidebar.success("Application Status")

    st.sidebar.write("Backend Loaded")
    st.sidebar.write("Model Loaded")
    st.sidebar.write("Preprocessor Loaded")


st.set_page_config(
    page_title="Dashboard",
    page_icon="🏠",
    layout="wide",
)

_render_sidebar()

st.title("🏠 Dashboard")
st.caption(APP_NAME)

st.markdown("---")

with st.spinner("Loading dataset and model..."):
    df: pd.DataFrame = _load_dataset()
    model: Any = _load_model()

_render_kpi_row(df, model)

st.markdown("---")

_render_dataset_overview(df)

st.markdown("---")

_render_target_distribution(df)

st.markdown("---")

_render_dataset_preview(df)
