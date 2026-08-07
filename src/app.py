import streamlit as st

from src.config.constants import APP_NAME

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(APP_NAME)

st.markdown(
    """
    ### AI-Powered Customer Churn Prediction System

    Use the sidebar to navigate through the application.
    """
)

st.info("Select a page from the left sidebar.")