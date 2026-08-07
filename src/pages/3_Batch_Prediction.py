"""Streamlit page for running churn predictions on a batch of customers."""

import pandas as pd
import streamlit as st

from src.services.prediction_service import PredictionService
from src.utils.feature_engineering import engineer_features

REQUIRED_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]


st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide",
)

st.title("📂 Batch Prediction")
st.caption("Upload a CSV file containing customer records.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head(), use_container_width=True)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing Columns: {missing_columns}")
        st.stop()

    df = engineer_features(df)

    service = PredictionService()

    with st.spinner("Running churn predictions..."):
        results = service.predict_batch(df)

    df["Prediction"] = results["prediction"].map({0: "No Churn", 1: "Churn"})
    df["Churn Probability (%)"] = (results["probability"] * 100).round(2)
    df["Risk"] = results["risk"]

    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results",
        data=csv,
        file_name="batch_predictions.csv",
        mime="text/csv",
    )
