"""Unit tests for arize.datasets.validation."""

from __future__ import annotations

import pandas as pd
import pytest

from arize.datasets.errors import BinaryColumnError
from arize.datasets.validation import validate_dataset_df


@pytest.mark.unit
class TestValidateDatasetDfBinaryColumns:
    """validate_dataset_df must reject columns that contain Python bytes."""

    def _base_df(self) -> pd.DataFrame:
        """Return a minimal valid DataFrame with the three required columns."""
        return pd.DataFrame(
            {
                "id": ["row-1", "row-2"],
                "created_at": [1_000_000, 2_000_000],
                "updated_at": [1_000_000, 2_000_000],
            }
        )

    def test_bytes_column_raises_before_upload(self) -> None:
        """A DataFrame with a bytes column must return a BinaryColumnError."""
        df = self._base_df()
        df["markdown"] = [b"hello", b"world"]

        errors = validate_dataset_df(df)

        assert len(errors) == 1
        assert isinstance(errors[0], BinaryColumnError)
        assert "markdown" in errors[0].error_message()
        assert "str" in errors[0].error_message()

    def test_bytes_column_error_message_contains_column_name(self) -> None:
        """The error message must include the offending column name."""
        df = self._base_df()
        df["payload"] = [b"\x00\x01", b"\x02\x03"]

        errors = validate_dataset_df(df)

        assert len(errors) == 1
        msg = errors[0].error_message()
        assert "payload" in msg

    def test_bytes_error_fired_before_required_columns_check(self) -> None:
        """Binary check runs even when required columns are absent."""
        df = pd.DataFrame(
            {
                "content": [b"hello", b"world"],
            }
        )

        errors = validate_dataset_df(df)

        assert len(errors) == 1
        assert isinstance(errors[0], BinaryColumnError)

    def test_str_column_accepted(self) -> None:
        """A DataFrame with only str/numeric columns must pass validation."""
        df = self._base_df()
        df["response"] = ["hello", "world"]
        df["score"] = [1.0, 2.0]

        errors = validate_dataset_df(df)

        assert errors == []

    def test_all_null_column_is_not_flagged(self) -> None:
        """A column containing only NaN values must not trigger the check."""
        df = self._base_df()
        df["optional"] = [None, None]

        errors = validate_dataset_df(df)

        assert errors == []

    def test_mixed_str_and_bytes_column_flagged(self) -> None:
        """A column with the first non-null value as bytes must be rejected."""
        df = self._base_df()
        df["mixed"] = [b"some bytes", "some string"]

        errors = validate_dataset_df(df)

        assert len(errors) == 1
        assert isinstance(errors[0], BinaryColumnError)
        assert "mixed" in errors[0].error_message()

    def test_str_first_then_bytes_column_flagged(self) -> None:
        """A column whose first value is str but later values are bytes must be
        rejected: pyarrow infers Arrow binary for the whole column, so checking
        only the first value would let it through and fail server-side.
        """
        df = self._base_df()
        df["mixed"] = ["some string", b"some bytes"]

        errors = validate_dataset_df(df)

        assert len(errors) == 1
        assert isinstance(errors[0], BinaryColumnError)
        assert "mixed" in errors[0].error_message()

    def test_multiple_bytes_columns_all_reported_in_one_error(self) -> None:
        """Every bytes column is collected so the user sees them all at once."""
        df = self._base_df()
        df["markdown"] = [b"a", b"b"]
        df["thumbnail"] = [b"\x00", b"\x01"]
        df["caption"] = ["fine", "also fine"]  # str column must not be flagged

        errors = validate_dataset_df(df)

        assert len(errors) == 1
        assert isinstance(errors[0], BinaryColumnError)
        assert errors[0].column_names == ["markdown", "thumbnail"]
        msg = errors[0].error_message()
        assert "markdown" in msg
        assert "thumbnail" in msg
        assert "caption" not in msg


@pytest.mark.unit
class TestBinaryColumnError:
    """Unit tests for the BinaryColumnError class itself."""

    def test_error_message(self) -> None:
        """error_message() should include the column names."""
        err = BinaryColumnError(["my_col", "other_col"])
        assert "my_col" in err.error_message()
        assert "other_col" in err.error_message()
        assert "bytes" in err.error_message()
        assert "str" in err.error_message()

    def test_str_returns_error_message(self) -> None:
        """str() should delegate to error_message()."""
        err = BinaryColumnError(["col"])
        assert str(err) == err.error_message()

    def test_repr(self) -> None:
        """repr() should include the class name and column names."""
        err = BinaryColumnError(["col"])
        assert "BinaryColumnError" in repr(err)
        assert "col" in repr(err)
