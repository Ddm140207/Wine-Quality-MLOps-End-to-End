"""Automatic retraining pipeline.

Compares weighted F1 of a freshly trained model against the current production
model. Replaces production model only when the new one scores higher.
"""
import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from train import load_data, train

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"
CANDIDATE_PATH = Path(__file__).parent.parent / "models" / "model_candidate.pkl"


def score_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    return f1_score(y_test, model.predict(X_test), average="weighted")


def main(min_improvement: float = 0.001) -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )

    candidate = train(X_train, y_train)
    candidate_score = score_model(candidate, X_test, y_test)
    print(f"Candidate F1: {candidate_score:.4f}")

    if MODEL_PATH.exists():
        current = joblib.load(MODEL_PATH)
        current_score = score_model(current, X_test, y_test)
        print(f"Current   F1: {current_score:.4f}")
        if candidate_score <= current_score + min_improvement:
            print("No improvement — keeping current model.")
            return

    joblib.dump(candidate, CANDIDATE_PATH)
    shutil.move(str(CANDIDATE_PATH), str(MODEL_PATH))
    print(f"Model promoted to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-improvement", type=float, default=0.001)
    args = parser.parse_args()
    main(min_improvement=args.min_improvement)
