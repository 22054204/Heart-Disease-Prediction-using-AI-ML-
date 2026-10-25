"""Model construction, training and evaluation functions."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS


def build_model(
    random_state: int = 42,
    model_type: str = "logistic_regression",
) -> Pipeline:
    """Return a leakage-safe classification pipeline.

    Supported model types:
      - 'logistic_regression': Interpretable linear model with balanced class weights.
      - 'random_forest': Non-linear ensemble model with balanced class weights.
    """
    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2_000,
            random_state=random_state,
            class_weight="balanced",
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Expected 'logistic_regression' or 'random_forest'."
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """Calculate classification metrics from an untouched evaluation split."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else None
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_evaluated": int(len(target)),
        "features": list(FEATURE_COLUMNS),
    }

    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(target, probabilities))
        metrics["average_precision"] = float(average_precision_score(target, probabilities))

    return metrics
