from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from heart_disease.data import FEATURE_COLUMNS
from heart_disease.model import build_model, evaluate_model


def test_build_model_supported_types() -> None:
    lr = build_model(model_type="logistic_regression")
    assert "classifier" in lr.named_steps

    rf = build_model(model_type="random_forest")
    assert "classifier" in rf.named_steps


def test_build_model_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_model(model_type="invalid_type")


def test_model_pipeline_train_and_evaluate(tmp_path: Path) -> None:
    rng = np.random.RandomState(42)
    n_samples = 40

    features = pd.DataFrame(
        rng.randn(n_samples, len(FEATURE_COLUMNS)),
        columns=FEATURE_COLUMNS,
    )
    # Inject a few missing values to test imputer inside pipeline
    features.iloc[0, 2] = np.nan
    features.iloc[5, 8] = np.nan

    target = pd.Series(rng.choice([0, 1], size=n_samples))

    pipeline = build_model(random_state=42, model_type="logistic_regression")
    pipeline.fit(features, target)

    metrics = evaluate_model(pipeline, features, target)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # Test persistence and re-loading
    model_file = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_file)
    loaded = joblib.load(model_file)

    sample = features.iloc[[0]]
    pred_orig = pipeline.predict(sample)
    pred_loaded = loaded.predict(sample)
    assert np.array_equal(pred_orig, pred_loaded)
