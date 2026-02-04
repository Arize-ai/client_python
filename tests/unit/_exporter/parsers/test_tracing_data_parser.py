"""Unit tests for arize._exporter.parsers.tracing_data_parser module."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from arize._exporter.parsers.tracing_data_parser import (
    OtelTracingDataTransformer,
)
from arize.spans.columns import (
    SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL,
    SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL,
    SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL,
    SPAN_ATTRIBUTES_METADATA,
    SPAN_END_TIME_COL,
    SPAN_START_TIME_COL,
)


@pytest.fixture
def transformer() -> OtelTracingDataTransformer:
    """Create a transformer instance."""
    return OtelTracingDataTransformer()


@pytest.fixture
def sample_df_with_json_columns() -> pd.DataFrame:
    """Create a DataFrame with JSON string columns."""
    return pd.DataFrame(
        {
            SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL.name: [
                json.dumps({"role": "user", "content": "hello"}),
                json.dumps({"role": "assistant", "content": "hi"}),
            ],
            SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL.name: [
                json.dumps({"role": "assistant", "content": "response"}),
                "",
            ],
            SPAN_ATTRIBUTES_METADATA.name: [
                json.dumps({"key": "value"}),
                None,
            ],
        }
    )


@pytest.fixture
def sample_df_with_timestamps() -> pd.DataFrame:
    """Create a DataFrame with timestamp columns."""
    return pd.DataFrame(
        {
            SPAN_START_TIME_COL.name: [
                1000000000000,
                2000000000000,
                3000000000000,
            ],
            SPAN_END_TIME_COL.name: [
                1100000000000,
                2100000000000,
                3100000000000,
            ],
            "other_col": ["a", "b", "c"],
        }
    )


@pytest.mark.unit
class TestTransform:
    """Test transform method."""

    def test_transforms_all_column_types(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that transform processes all column types."""
        df = pd.DataFrame(
            {
                SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL.name: [
                    json.dumps({"role": "user"})
                ],
                SPAN_ATTRIBUTES_METADATA.name: [json.dumps({"key": "value"})],
                SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.name: [
                    json.dumps({"temp": 0.7})
                ],
                SPAN_START_TIME_COL.name: [1000000000000],
            }
        )
        result = transformer.transform(df)

        # Check list of dict column
        assert isinstance(
            result[SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL.name].iloc[0], list
        )
        # Check dict column
        assert isinstance(result[SPAN_ATTRIBUTES_METADATA.name].iloc[0], dict)
        # Check cleaned string column
        assert (
            result[SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.name].iloc[0]
            is not None
        )
        # Check timestamp column
        assert isinstance(
            result[SPAN_START_TIME_COL.name].iloc[0], pd.Timestamp
        )

    def test_handles_missing_columns_gracefully(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that transform handles missing columns without error."""
        df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        result = transformer.transform(df)
        assert result.equals(df)

    def test_returns_same_dataframe_when_no_columns_match(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that DataFrame is unchanged when no columns match."""
        df = pd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]})
        result = transformer.transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_collects_errors_without_raising(
        self,
        transformer: OtelTracingDataTransformer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that transformation errors are logged but don't raise."""
        # Create a DataFrame with invalid JSON that will cause errors during transformation
        df = pd.DataFrame(
            {
                SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL.name: ["invalid json {"],
            }
        )

        # Transform should not raise
        result = transformer.transform(df)

        # Should have a result (transformation attempted)
        assert result is not None
        # Warning should be logged
        assert any(
            "Unable to transform" in record.message for record in caplog.records
        )

    def test_empty_dataframe_returns_empty(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that empty DataFrame returns empty."""
        df = pd.DataFrame()
        result = transformer.transform(df)
        assert result.empty


@pytest.mark.unit
class TestTransformValueToListOfDict:
    """Test _transform_value_to_list_of_dict method."""

    def test_json_string_array_to_list_of_dicts(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test JSON string converted to list of dicts."""
        value = json.dumps({"role": "user", "content": "hello"})
        result = transformer._transform_value_to_list_of_dict(value)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hello"}

    def test_already_list_of_dicts_unchanged(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that existing list is processed correctly."""
        value = [json.dumps({"a": 1}), json.dumps({"b": 2})]
        result = transformer._transform_value_to_list_of_dict(value)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}

    def test_empty_string_returns_none(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that empty string returns None."""
        result = transformer._transform_value_to_list_of_dict("")
        assert result is None

    def test_none_returns_none(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that None returns None."""
        result = transformer._transform_value_to_list_of_dict(None)
        assert result is None

    def test_invalid_json_raises_value_error(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON string"):
            transformer._transform_value_to_list_of_dict("invalid json {")

    def test_numpy_array_processed(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that numpy array is processed correctly."""
        value = np.array([json.dumps({"a": 1}), json.dumps({"b": 2})])
        result = transformer._transform_value_to_list_of_dict(value)
        assert isinstance(result, list)
        assert len(result) == 2


@pytest.mark.unit
class TestTransformJsonToDict:
    """Test _transform_json_to_dict method."""

    def test_json_string_to_dict(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test JSON string converted to dict."""
        value = json.dumps({"key": "value", "number": 42})
        result = transformer._transform_json_to_dict(value)
        assert isinstance(result, dict)
        assert result == {"key": "value", "number": 42}

    def test_already_dict_unchanged(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that non-string values return None."""
        result = transformer._transform_json_to_dict({"already": "dict"})
        assert result is None

    def test_empty_string_returns_none(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that empty string returns None."""
        result = transformer._transform_json_to_dict("")
        assert result is None

    def test_invalid_json_raises_value_error(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON string"):
            transformer._transform_json_to_dict("not valid json")


@pytest.mark.unit
class TestConvertTimestampToDatetime:
    """Test _convert_timestamp_to_datetime method."""

    def test_nanosecond_int_to_timestamp(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test nanosecond integer converted to Timestamp."""
        value = 1609459200000000000  # 2021-01-01 00:00:00
        result = transformer._convert_timestamp_to_datetime(value)
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2021

    def test_numpy_int64_to_timestamp(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test numpy int64 converted to Timestamp."""
        value = np.int64(1609459200000000000)
        result = transformer._convert_timestamp_to_datetime(value)
        assert isinstance(result, pd.Timestamp)

    def test_float_timestamp(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test float timestamp converted correctly."""
        value = 1609459200000000000.0
        result = transformer._convert_timestamp_to_datetime(value)
        assert isinstance(result, pd.Timestamp)

    def test_non_numeric_returns_original(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that non-numeric values return unchanged."""
        value = "not a number"
        result = transformer._convert_timestamp_to_datetime(value)
        assert result == value

    def test_none_returns_none(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that None returns None."""
        result = transformer._convert_timestamp_to_datetime(None)
        assert result is None

    def test_zero_returns_original(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test that zero returns unchanged."""
        result = transformer._convert_timestamp_to_datetime(0)
        assert result == 0


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions."""

    def test_is_non_empty_string_true(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test is_non_empty_string returns True for valid strings."""
        assert transformer._is_non_empty_string("hello") is True

    def test_is_non_empty_string_false_for_empty(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test is_non_empty_string returns False for empty string."""
        assert transformer._is_non_empty_string("") is False

    def test_is_non_empty_string_false_for_none(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test is_non_empty_string returns False for None."""
        assert transformer._is_non_empty_string(None) is False

    def test_is_non_empty_string_false_for_non_string(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test is_non_empty_string returns False for non-strings."""
        assert transformer._is_non_empty_string(123) is False
        assert transformer._is_non_empty_string([]) is False

    def test_clean_json_string_empty_to_none(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test clean_json_string converts empty to None."""
        assert transformer._clean_json_string("") is None

    def test_clean_json_string_valid_unchanged(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test clean_json_string keeps valid strings."""
        assert (
            transformer._clean_json_string('{"key": "value"}')
            == '{"key": "value"}'
        )

    def test_clean_json_string_none_to_none(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test clean_json_string converts None to None."""
        assert transformer._clean_json_string(None) is None

    def test_deserialize_json_string_to_dict(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test deserialize_json_string_to_dict parses JSON."""
        json_str = '{"key": "value", "num": 42}'
        result = transformer._deserialize_json_string_to_dict(json_str)
        assert result == {"key": "value", "num": 42}

    def test_deserialize_json_string_invalid_raises(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test deserialize_json_string_to_dict raises on invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON string"):
            transformer._deserialize_json_string_to_dict("invalid")


@pytest.mark.unit
class TestIntegrationScenarios:
    """Test end-to-end transformation scenarios."""

    def test_full_tracing_dataframe_transformation(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test transformation of a complete tracing DataFrame."""
        df = pd.DataFrame(
            {
                SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL.name: [
                    json.dumps({"role": "user", "content": "hello"}),
                ],
                SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL.name: [
                    json.dumps({"role": "assistant", "content": "hi"}),
                ],
                SPAN_ATTRIBUTES_METADATA.name: [
                    json.dumps({"session_id": "123"}),
                ],
                SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.name: [
                    json.dumps({"temperature": 0.7}),
                ],
                SPAN_START_TIME_COL.name: [1609459200000000000],
                SPAN_END_TIME_COL.name: [1609459201000000000],
            }
        )

        result = transformer.transform(df)

        # Verify transformations
        assert isinstance(
            result[SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL.name].iloc[0], list
        )
        assert isinstance(result[SPAN_ATTRIBUTES_METADATA.name].iloc[0], dict)
        assert isinstance(
            result[SPAN_START_TIME_COL.name].iloc[0], pd.Timestamp
        )

    def test_handles_mixed_valid_and_empty_values(
        self, transformer: OtelTracingDataTransformer
    ) -> None:
        """Test handling of mixed valid and empty values."""
        df = pd.DataFrame(
            {
                SPAN_ATTRIBUTES_METADATA.name: [
                    json.dumps({"key": "value"}),
                    "",
                    None,
                ],
                SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.name: [
                    json.dumps({"temp": 0.7}),
                    "",
                    json.dumps({"temp": 0.9}),
                ],
            }
        )

        result = transformer.transform(df)

        # First row should be dict
        assert isinstance(result[SPAN_ATTRIBUTES_METADATA.name].iloc[0], dict)
        # Second and third should be None
        assert result[SPAN_ATTRIBUTES_METADATA.name].iloc[1] is None
        assert result[SPAN_ATTRIBUTES_METADATA.name].iloc[2] is None

        # Check cleaned parameters
        assert (
            result[SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.name].iloc[0]
            is not None
        )
        assert (
            result[SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.name].iloc[1]
            is None
        )
