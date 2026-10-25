from pathlib import Path

from heart_disease.data import FEATURE_COLUMNS, TARGET_COLUMN, load_cleveland_data


ROOT = Path(__file__).resolve().parents[1]


def test_cleveland_data_has_expected_schema_and_binary_target() -> None:
    data = load_cleveland_data(ROOT / "processed.cleveland.data")

    assert list(data.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert len(data) == 303
    assert set(data[TARGET_COLUMN].unique()) == {0, 1}
    assert data[list(FEATURE_COLUMNS)].isna().sum().sum() > 0
