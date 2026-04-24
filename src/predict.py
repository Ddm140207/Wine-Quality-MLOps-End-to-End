from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "type",
]

_model: XGBClassifier | None = None


def get_model() -> XGBClassifier:
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(features: dict) -> int:
    """Return 1 (good) or 0 (bad) for given wine features."""
    model = get_model()
    df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    return int(model.predict(df)[0])
