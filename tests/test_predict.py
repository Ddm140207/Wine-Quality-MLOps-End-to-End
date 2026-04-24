import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from main import app  # noqa: E402

client = TestClient(app)

RED_WINE_SAMPLE = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "type": 0,
}

WHITE_WINE_SAMPLE = {
    "fixed acidity": 7.0,
    "volatile acidity": 0.27,
    "citric acid": 0.36,
    "residual sugar": 20.7,
    "chlorides": 0.045,
    "free sulfur dioxide": 45.0,
    "total sulfur dioxide": 170.0,
    "density": 1.001,
    "pH": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8,
    "type": 1,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_red_wine_returns_200():
    response = client.post("/predict", json=RED_WINE_SAMPLE)
    assert response.status_code == 200


def test_predict_white_wine_returns_200():
    response = client.post("/predict", json=WHITE_WINE_SAMPLE)
    assert response.status_code == 200


def test_predict_response_schema():
    response = client.post("/predict", json=RED_WINE_SAMPLE)
    body = response.json()
    assert "quality" in body
    assert "label" in body
    assert body["quality"] in (0, 1)
    assert body["label"] in ("good", "bad")


def test_predict_white_wine_good():
    response = client.post("/predict", json=WHITE_WINE_SAMPLE)
    body = response.json()
    assert body["quality"] == 1
    assert body["label"] == "good"


def test_predict_missing_field_returns_422():
    incomplete = {k: v for k, v in RED_WINE_SAMPLE.items() if k != "alcohol"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
