import pytest

from heart_disease.predict import parse_features


def test_parse_features_requires_all_thirteen_values() -> None:
    with pytest.raises(ValueError, match="Expected 13"):
        parse_features("1,2,3")


def test_parse_features_assigns_columns() -> None:
    frame = parse_features("63,1,3,145,233,1,0,150,0,2.3,0,0,1")
    assert frame.shape == (1, 13)
    assert frame.iloc[0]["age"] == 63.0
