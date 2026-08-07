"""About page with project overview, dataset, and technology details."""

from textwrap import dedent

import streamlit as st

from src.config.constants import APP_NAME

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide",
)

st.title("ℹ️ About")

st.caption(APP_NAME)

st.markdown("---")

st.header("Project Overview")

st.write(
    dedent(
        """\
        Customer Churn Intelligence Platform is an end-to-end Machine Learning
        application that predicts customer churn and explains the predictions
        using Explainable AI.

        The application helps businesses identify high-risk customers and
        supports retention strategies.
        """
    )
)

st.header("Dataset")

st.markdown(
    dedent(
        """\
        - IBM Telco Customer Churn Dataset
        - 7,043 Customers
        - Binary Classification
        - Target: Churn Label
        """
    )
)

st.header("Machine Learning Workflow")

workflow_steps: list[str] = [
    "Project Setup",
    "Exploratory Data Analysis",
    "Data Preprocessing",
    "Feature Engineering",
    "Model Training",
    "Hyperparameter Tuning",
    "Explainable AI",
    "Streamlit Deployment",
]

for step in workflow_steps:
    st.write(f"✅ {step}")

st.header("Technologies")

technologies: list[str] = [
    "Python",
    "Pandas",
    "NumPy",
    "Scikit-Learn",
    "SHAP",
    "Matplotlib",
    "Streamlit",
    "Joblib",
]

col1, col2 = st.columns(2)

for index, technology in enumerate(technologies):
    column = col1 if index % 2 == 0 else col2
    column.write(f"• {technology}")

st.header("Project Structure")

st.code(
    dedent(
        """\
        data/
        models/
        notebooks/
        reports/
        src/
        tests/
        """
    )
)

st.header("Future Enhancements")

future_enhancements: list[str] = [
    "FastAPI",
    "Docker",
    "Cloud Deployment",
    "CI/CD",
    "Model Monitoring",
    "Database Integration",
]

for enhancement in future_enhancements:
    st.write(f"🚀 {enhancement}")

st.markdown("---")

st.success(APP_NAME)

st.caption("Developed by Hafiz Ahmad Adil")
