"""Run a validated prediction from a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict heart-disease probability.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/heart_disease_model.joblib"),
        help="Path to trained model .joblib file.",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Thirteen comma-separated values in documented order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.",
    )
    return parser.parse_args()


def parse_features(raw_features: str) -> pd.DataFrame:
    """Parse and validate comma-separated numeric features into a single-row DataFrame."""
    try:
        values = [float(value.strip()) for value in raw_features.split(",")]
    except ValueError as error:
        raise ValueError("--features must contain numeric comma-separated values.") from error
    if len(values) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} feature values, got {len(values)}."
        )
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def main() -> None:
    args = parse_args()
    model_path = args.model
    if not model_path.is_file():
        candidate = Path(__file__).resolve().parents[2] / args.model
        if candidate.is_file():
            model_path = candidate
        else:
            raise FileNotFoundError(
                f"Model file not found at '{args.model}' or '{candidate}'. Run training first."
            )

    model = joblib.load(model_path)
    features = parse_features(args.features)
    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= 0.5)
    print(
        json.dumps(
            {
                "prediction": prediction,
                "diagnosis": "Diagnosed Heart Disease (High Probability)" if prediction == 1 else "No Diagnosed Disease (Low Probability)",
                "probability_of_diagnosed_disease": round(probability, 4),
                "threshold": 0.5,
                "notice": "Educational model only; not medical advice or a diagnostic tool.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
