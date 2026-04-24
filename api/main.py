import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI
from pydantic import BaseModel, Field
from predict import predict

app = FastAPI(title="Wine Quality API", version="1.0.0")


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., alias="fixed acidity")
    volatile_acidity: float = Field(..., alias="volatile acidity")
    citric_acid: float = Field(..., alias="citric acid")
    residual_sugar: float = Field(..., alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float
    wine_type: int = Field(..., alias="type", description="0=red, 1=white")

    model_config = {"populate_by_name": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict_quality(features: WineFeatures) -> dict:
    feature_dict = {
        "fixed acidity": features.fixed_acidity,
        "volatile acidity": features.volatile_acidity,
        "citric acid": features.citric_acid,
        "residual sugar": features.residual_sugar,
        "chlorides": features.chlorides,
        "free sulfur dioxide": features.free_sulfur_dioxide,
        "total sulfur dioxide": features.total_sulfur_dioxide,
        "density": features.density,
        "pH": features.pH,
        "sulphates": features.sulphates,
        "alcohol": features.alcohol,
        "type": features.wine_type,
    }
    result = predict(feature_dict)
    return {"quality": result, "label": "good" if result == 1 else "bad"}
