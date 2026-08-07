import pandas as pd

from fastapi import APIRouter

from api.dependencies import prediction_service
from api.schemas import PredictionResponse

router = APIRouter()

@router.get("/health")
def health():

    return {
        "status": "ok"
    }


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(customer: dict):

    df = pd.DataFrame([customer])

    result = prediction_service.predict(df)

    return {
        "prediction": (
            "Churn"
            if result["prediction"]
            else "No Churn"
        ),
        "probability": result["probability"],
        "risk": result["risk"]
    }

