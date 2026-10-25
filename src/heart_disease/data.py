"""Dataset loading and validation for the UCI processed Cleveland data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)
TARGET_COLUMN = "target"
ALL_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)


def load_cleveland_data(path: str | Path) -> pd.DataFrame:
    """Load UCI's processed Cleveland dataset and make a binary target.

    A target value of 0 means no diagnosed disease; values 1--4 are combined
    into 1 (diagnosed disease). Missing feature values are intentionally kept
    as ``NaN`` so the training pipeline can impute using training-fold values
    only. This avoids dropping patients and avoids data leakage.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        candidate = Path(__file__).resolve().parents[2] / path
        if candidate.is_file():
            dataset_path = candidate
        else:
            raise FileNotFoundError(
                f"Dataset not found: '{dataset_path}'. (Also checked: '{candidate}')"
            )

    frame = pd.read_csv(
        dataset_path,
        header=None,
        names=ALL_COLUMNS,
        na_values="?",
        dtype=str,
    )
    if frame.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} columns in {dataset_path}, "
            f"found {frame.shape[1]}."
        )

    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing or invalid values.")

    frame[TARGET_COLUMN] = (frame[TARGET_COLUMN] > 0).astype(int)
    return frame
