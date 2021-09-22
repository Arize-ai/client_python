import pytest
from arize.pandas.logger import Schema


def test_schema_raise_no_exception():
    try:
        Schema(prediction_id_column_name="prediction_id")
    except TypeError:
        assert (
            False
        ), "only prediction_id_column_name is required. everything else should be optional"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
