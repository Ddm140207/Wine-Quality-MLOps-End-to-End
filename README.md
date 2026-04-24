# Wine Quality Classifier

![CI](https://github.com/Ddm140207/Wine-Quality-MLOps-End-to-End/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/Ddm140207/Wine-Quality-MLOps-End-to-End/actions/workflows/cd.yml/badge.svg)

Binary classifier (good/bad) for red and white wine using XGBoost. Trained on the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

## Architecture

```
┌─────────────┐     PR      ┌──────────────┐
│  Developer  │────────────▶│  CI (pytest) │
└─────────────┘             └──────┬───────┘
                                   │ merge to main
                            ┌──────▼───────┐
                            │  CD: build   │
                            │  Docker img  │
                            └──────┬───────┘
                                   │ push
                     ┌─────────────▼──────────────┐
                     │         GHCR               │
                     │  ghcr.io/user/wine-quality  │
                     └─────────────┬──────────────┘
                                   │ deploy
                            ┌──────▼───────┐
                            │   Server     │
                            │  :8000/predict│
                            └──────────────┘

Data & Model versioning:
  CSV files ──▶ DVC ──▶ S3 remote
  model.pkl ──▶ DVC ──▶ S3 remote
```

## Endpoints

| Method | Path       | Description                     |
|--------|------------|---------------------------------|
| GET    | `/health`  | Health check                    |
| POST   | `/predict` | Predict wine quality (good/bad) |

### POST /predict

```json
{
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
  "type": 0
}
```

Response:
```json
{"quality": 1, "label": "good"}
```

`type`: `0` = red wine, `1` = white wine.

## Local Development

```bash
# Install
pip install ".[dev]"

# Pull model artifact
dvc pull models/model.pkl

# Run API
uvicorn api.main:app --reload

# Run tests
pytest
```

## Retraining

```bash
python src/retrain.py --min-improvement 0.001
```

Trains a candidate model, compares weighted F1 against production, promotes only if better.

## Project Structure

```
├── src/
│   ├── train.py       # Training script
│   ├── predict.py     # Inference logic
│   └── retrain.py     # Automatic retraining pipeline
├── api/
│   └── main.py        # FastAPI app
├── tests/
│   └── test_predict.py
├── data/
│   └── dataset.csv.dvc
├── models/
│   └── model.pkl.dvc
├── .dvc/config        # S3 remote configured
├── .env.example
├── Dockerfile
└── pyproject.toml
```
