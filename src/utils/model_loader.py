import joblib

from src.config.settings import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
)


class ModelLoader:

    _model = None
    _preprocessor = None

    @classmethod
    def get_model(cls):

        if cls._model is None:
            cls._model = joblib.load(MODEL_PATH)

        return cls._model

    @classmethod
    def get_preprocessor(cls):

        if cls._preprocessor is None:
            cls._preprocessor = joblib.load(PREPROCESSOR_PATH)

        return cls._preprocessor