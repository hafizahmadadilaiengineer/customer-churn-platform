from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
ASSETS_DIR = ROOT_DIR / "src" / "assets"

MODEL_PATH = MODELS_DIR / "final_churn_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

RANDOM_STATE = 42