import argparse
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier


DATA_DIR = Path(__file__).parent.parent / "wine+quality"
MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red["type"] = 0
    white["type"] = 1
    df = pd.concat([red, white], ignore_index=True)
    X = df.drop("quality", axis=1)
    y = (df["quality"] >= 5).astype(int)
    return X, y


def train(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(eval_metric="logloss", random_state=40)
    model.fit(X, y)
    return model


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return f1_score(y_test, y_pred, average="weighted")


def main(test_size: float = 0.2) -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=40
    )
    model = train(X_train, y_train)
    score = evaluate(model, X_test, y_test)
    print(f"Weighted F1: {score:.4f}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    main(test_size=args.test_size)
