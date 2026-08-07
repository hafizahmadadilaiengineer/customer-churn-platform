"""Streamlit page for predicting churn for a single customer."""

import pandas as pd
import streamlit as st

from src.services.prediction_service import PredictionService
from src.utils.feature_engineering import engineer_features


def _display_prediction(result: dict[str, int | float | str]) -> None:
    """Display the prediction outcome as three KPI metrics."""
    st.success("Prediction Completed")

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction", result["prediction"])
    col2.metric("Probability", f"{result['probability']:.2%}")
    col3.metric("Risk", result["risk"])


st.set_page_config(
    page_title="Single Prediction",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Single Customer Prediction")
st.caption("Enter customer details below and run a churn prediction.")

with st.form("prediction_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        gender: str = st.selectbox("Gender", ["Male", "Female"])
        senior: str = st.selectbox("Senior Citizen", ["Yes", "No"])

    with col2:
        partner: str = st.selectbox("Partner", ["Yes", "No"])
        dependents: str = st.selectbox("Dependents", ["Yes", "No"])

    st.subheader("Tenure & Charges")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure: int = st.number_input("Tenure Months", 0, 100, 12)

    with col2:
        monthly: float = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

    with col3:
        total: float = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    st.subheader("Services")

    col1, col2 = st.columns(2)

    with col1:
        phone_service: str = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines: str = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
        )
        online_security: str = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )
        device_protection: str = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
        )
        streaming_tv: str = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )

    with col2:
        internet_service: str = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )
        online_backup: str = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
        )
        tech_support: str = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )
        streaming_movies: str = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
        )

    st.subheader("Contract & Billing")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract: str = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

    with col2:
        paperless: str = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col3:
        payment: str = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    service = PredictionService()

    customer = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Senior Citizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure Months": tenure,
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet_service,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection": device_protection,
                "Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies,
                "Contract": contract,
                "Paperless Billing": paperless,
                "Payment Method": payment,
                "Monthly Charges": monthly,
                "Total Charges": total,
            }
        ]
    )

    customer = engineer_features(customer)

    result = service.predict(customer)

    _display_prediction(result)
