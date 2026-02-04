"""Tests for arize.embeddings.errors module."""

import pytest

from arize.embeddings.errors import (
    HuggingFaceRepositoryNotFound,
    InvalidIndexError,
)


@pytest.mark.unit
class TestInvalidIndexError:
    """Test InvalidIndexError exception class."""

    def test_stores_field_name(self) -> None:
        """Should store field_name attribute."""
        error = InvalidIndexError(field_name="my_column")
        assert error.field_name == "my_column"

    def test_different_messages_for_dataframe_vs_series(self) -> None:
        """Should generate different error messages for DataFrame vs Series."""
        df_error = InvalidIndexError(field_name="DataFrame")
        series_error = InvalidIndexError(field_name="my_column")

        df_message = df_error.error_message()
        series_message = series_error.error_message()

        # Verify they're different
        assert df_message != series_message
        # Verify DataFrame message mentions DataFrame
        assert "DataFrame" in df_message
        # Verify Series message includes the column name
        assert "my_column" in series_message

    def test_can_be_raised_and_caught(self) -> None:
        """Should be raisable and catchable as exception."""
        with pytest.raises(InvalidIndexError) as exc_info:
            raise InvalidIndexError(field_name="test_field")
        assert exc_info.value.field_name == "test_field"


@pytest.mark.unit
class TestHuggingFaceRepositoryNotFound:
    """Test HuggingFaceRepositoryNotFound exception class."""

    def test_stores_model_name(self) -> None:
        """Should store model_name attribute."""
        error = HuggingFaceRepositoryNotFound(model_name="my-model")
        assert error.model_name == "my-model"

    def test_error_message_includes_model_name(self) -> None:
        """Should include the invalid model name in error message."""
        error = HuggingFaceRepositoryNotFound(model_name="invalid-model-123")
        message = error.error_message()
        assert "invalid-model-123" in message

    def test_can_be_raised_and_caught(self) -> None:
        """Should be raisable and catchable as exception."""
        with pytest.raises(HuggingFaceRepositoryNotFound) as exc_info:
            raise HuggingFaceRepositoryNotFound(model_name="test-model")
        assert exc_info.value.model_name == "test-model"
