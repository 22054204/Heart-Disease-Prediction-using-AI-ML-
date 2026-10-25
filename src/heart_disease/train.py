"""Train and persist the reproducible heart-disease classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from .data import FEATURE_COLUMNS, TARGET_COLUMN, load_cleveland_data
from .model import build_model, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a heart-disease classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("processed.cleveland.data"),
        help="Path to processed.cleveland.data dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to save the trained model and evaluation metrics.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset to reserve for test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
        help="Classifier architecture to train (default: logistic_regression).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")

    data = load_cleveland_data(args.data)
    features, target = data[list(FEATURE_COLUMNS)], data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    model = build_model(
        random_state=args.random_state,
        model_type=args.model_type,
    )
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    metrics.update(
        {
            "model_type": args.model_type,
            "n_training": int(len(y_train)),
            "n_missing_feature_values": int(features.isna().sum().sum()),
            "random_state": args.random_state,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "heart_disease_model.joblib"
    metrics_path = args.output_dir / "metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
