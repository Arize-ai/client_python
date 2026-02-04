"""Tests for OpenInference data conversion utilities."""

from unittest.mock import patch

import pandas as pd
import pytest

from arize.utils.openinference_conversion import (
    _should_convert_json,
    convert_boolean_columns_to_str,
    convert_datetime_columns_to_int,
    convert_default_columns_to_json_str,
    convert_json_str_to_dict,
)


@pytest.fixture
def sample_datetime_df() -> pd.DataFrame:
    """Create a sample DataFrame with datetime columns."""
    return pd.DataFrame(
        {
            "dt": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "dt_utc": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            "str_col": ["a", "b"],
        }
    )


@pytest.fixture
def sample_dict_df() -> pd.DataFrame:
    """Create a sample DataFrame with dict columns."""
    return pd.DataFrame(
        {
            "eval.test.metadata": [{"key": "value"}, {"foo": "bar"}],
            "other_col": ["a", "b"],
        }
    )


@pytest.mark.unit
class TestConvertDatetimeColumnsToInt:
    """Test convert_datetime_columns_to_int function."""

    def test_converts_datetime64_ns_to_milliseconds(self) -> None:
        """Should convert datetime64[ns] columns to milliseconds since epoch."""
        df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01 00:00:00"])})
        result = convert_datetime_columns_to_int(df)

        # 2020-01-01 00:00:00 UTC = 1577836800000 milliseconds since epoch
        assert result["dt"].iloc[0] == 1577836800000
        assert result["dt"].dtype == "int64"

    def test_converts_datetime64_ns_utc_to_milliseconds(self) -> None:
        """Should convert datetime64[ns, UTC] columns to milliseconds."""
        df = pd.DataFrame(
            {"dt_utc": pd.to_datetime(["2020-01-01 00:00:00"], utc=True)}
        )
        result = convert_datetime_columns_to_int(df)

        assert result["dt_utc"].iloc[0] == 1577836800000
        assert result["dt_utc"].dtype == "int64"

    def test_handles_nat_values(self) -> None:
        """Should handle NaT (Not a Time) values."""
        df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", pd.NaT])})
        result = convert_datetime_columns_to_int(df)

        # NaT becomes NaN in int64, which is represented as a specific value
        assert result["dt"].iloc[0] == 1577836800000
        assert pd.isna(result["dt"].iloc[1]) or result["dt"].iloc[1] < 0

    def test_returns_unchanged_for_no_datetime_columns(self) -> None:
        """Should return DataFrame unchanged when no datetime columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = convert_datetime_columns_to_int(df)

        pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe(self) -> None:
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        result = convert_datetime_columns_to_int(df)

        assert len(result) == 0
        pd.testing.assert_frame_equal(result, df)

    def test_mixed_datetime_types(
        self, sample_datetime_df: pd.DataFrame
    ) -> None:
        """Should convert both datetime64[ns] and datetime64[ns, UTC]."""
        result = convert_datetime_columns_to_int(sample_datetime_df)

        # Both datetime columns should be converted to int64
        assert result["dt"].dtype == "int64"
        assert result["dt_utc"].dtype == "int64"
        # String column should remain unchanged
        assert result["str_col"].dtype == "object"


@pytest.mark.unit
class TestConvertBooleanColumnsToStr:
    """Test convert_boolean_columns_to_str function."""

    def test_converts_bool_to_string(self) -> None:
        """Should convert boolean columns to string type."""
        df = pd.DataFrame({"bool_col": [True, False, True]})
        result = convert_boolean_columns_to_str(df)

        assert result["bool_col"].dtype == "string"
        assert result["bool_col"].iloc[0] == "True"
        assert result["bool_col"].iloc[1] == "False"

    def test_returns_unchanged_for_no_bool_columns(self) -> None:
        """Should return DataFrame unchanged when no boolean columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = convert_boolean_columns_to_str(df)

        pd.testing.assert_frame_equal(result, df)

    def test_handles_none_values_in_bool_column(self) -> None:
        """Should handle None/NA values in boolean columns."""
        df = pd.DataFrame({"bool_col": [True, None, False]})
        result = convert_boolean_columns_to_str(df)

        # Conversion should handle nullable booleans (dtype becomes object when None is present)
        # The function only converts pure bool dtype columns
        assert result["bool_col"].dtype in ["string", "object"]

    def test_empty_dataframe(self) -> None:
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        result = convert_boolean_columns_to_str(df)

        assert len(result) == 0
        pd.testing.assert_frame_equal(result, df)


@pytest.mark.unit
class TestConvertDefaultColumnsToJsonStr:
    """Test convert_default_columns_to_json_str function."""

    def test_converts_dict_to_json_string(self) -> None:
        """Should convert dict values to JSON strings in eligible columns."""
        df = pd.DataFrame(
            {"eval.test.metadata": [{"key": "value"}, {"foo": "bar"}]}
        )
        result = convert_default_columns_to_json_str(df)

        assert result["eval.test.metadata"].iloc[0] == '{"key": "value"}'
        assert result["eval.test.metadata"].iloc[1] == '{"foo": "bar"}'

    def test_leaves_non_dict_values_unchanged(self) -> None:
        """Should leave non-dict values unchanged."""
        df = pd.DataFrame(
            {"eval.test.metadata": [{"key": "value"}, "already a string", None]}
        )
        result = convert_default_columns_to_json_str(df)

        assert result["eval.test.metadata"].iloc[0] == '{"key": "value"}'
        assert result["eval.test.metadata"].iloc[1] == "already a string"
        assert result["eval.test.metadata"].iloc[2] is None

    def test_handles_nested_dicts(self) -> None:
        """Should handle nested dictionaries."""
        df = pd.DataFrame(
            {"eval.test.metadata": [{"outer": {"inner": "value"}}]}
        )
        result = convert_default_columns_to_json_str(df)

        assert (
            result["eval.test.metadata"].iloc[0]
            == '{"outer": {"inner": "value"}}'
        )

    def test_handles_empty_dicts(self) -> None:
        """Should convert empty dicts to empty JSON objects."""
        df = pd.DataFrame({"eval.test.metadata": [{}]})
        result = convert_default_columns_to_json_str(df)

        assert result["eval.test.metadata"].iloc[0] == "{}"

    def test_only_converts_eligible_columns(self) -> None:
        """Should only convert columns that match conversion patterns."""
        df = pd.DataFrame(
            {
                "eval.test.metadata": [{"key": "value"}],
                "other_dict_col": [{"key": "value"}],
                "result": [{"key": "value"}],
            }
        )
        result = convert_default_columns_to_json_str(df)

        # eval.test.metadata and result should be converted
        assert isinstance(result["eval.test.metadata"].iloc[0], str)
        assert isinstance(result["result"].iloc[0], str)
        # other_dict_col should remain a dict
        assert isinstance(result["other_dict_col"].iloc[0], dict)

    def test_logs_exception_and_continues(self) -> None:
        """Should log exception and continue when conversion fails."""
        # Create a DataFrame with a value that will trigger json.dumps error
        # We mock json.dumps to raise an exception
        df = pd.DataFrame({"eval.test.metadata": [{"key": "value"}]})

        with (
            patch(
                "arize.utils.openinference_conversion.logger.debug"
            ) as mock_debug,
            patch(
                "arize.utils.openinference_conversion.json.dumps"
            ) as mock_dumps,
        ):
            mock_dumps.side_effect = Exception("Serialization error")
            result = convert_default_columns_to_json_str(df)

            # Should have logged the error
            mock_debug.assert_called_once()
            assert "Failed to convert column" in str(mock_debug.call_args)

            # DataFrame should be returned (even if conversion failed)
            assert "eval.test.metadata" in result.columns


@pytest.mark.unit
class TestConvertJsonStrToDict:
    """Test convert_json_str_to_dict function."""

    def test_converts_json_string_to_dict(self) -> None:
        """Should convert JSON string values to dicts in eligible columns."""
        df = pd.DataFrame(
            {"eval.test.metadata": ['{"key": "value"}', '{"foo": "bar"}']}
        )
        result = convert_json_str_to_dict(df)

        assert result["eval.test.metadata"].iloc[0] == {"key": "value"}
        assert result["eval.test.metadata"].iloc[1] == {"foo": "bar"}

    def test_leaves_non_string_values_unchanged(self) -> None:
        """Should leave non-string values unchanged."""
        df = pd.DataFrame(
            {
                "eval.test.metadata": [
                    '{"key": "value"}',
                    {"already": "dict"},
                    None,
                ]
            }
        )
        result = convert_json_str_to_dict(df)

        assert result["eval.test.metadata"].iloc[0] == {"key": "value"}
        assert result["eval.test.metadata"].iloc[1] == {"already": "dict"}
        assert result["eval.test.metadata"].iloc[2] is None

    def test_handles_invalid_json(self) -> None:
        """Should log exception and continue when JSON is invalid."""
        df = pd.DataFrame({"eval.test.metadata": ["not valid json"]})

        with patch(
            "arize.utils.openinference_conversion.logger.debug"
        ) as mock_debug:
            result = convert_json_str_to_dict(df)

            # Should have logged the error
            mock_debug.assert_called_once()
            assert "Failed to parse column" in str(mock_debug.call_args)

            # Original value should remain
            assert result["eval.test.metadata"].iloc[0] == "not valid json"

    def test_only_converts_eligible_columns(self) -> None:
        """Should only convert columns that match conversion patterns."""
        df = pd.DataFrame(
            {
                "eval.test.metadata": ['{"key": "value"}'],
                "other_str_col": ['{"key": "value"}'],
                "result": ['{"key": "value"}'],
            }
        )
        result = convert_json_str_to_dict(df)

        # eval.test.metadata and result should be converted
        assert isinstance(result["eval.test.metadata"].iloc[0], dict)
        assert isinstance(result["result"].iloc[0], dict)
        # other_str_col should remain a string
        assert isinstance(result["other_str_col"].iloc[0], str)

    def test_handles_empty_json_string(self) -> None:
        """Should convert empty JSON string to empty dict."""
        df = pd.DataFrame({"eval.test.metadata": ["{}"]})
        result = convert_json_str_to_dict(df)

        assert result["eval.test.metadata"].iloc[0] == {}


@pytest.mark.unit
class TestShouldConvertJson:
    """Test _should_convert_json function."""

    def test_matches_eval_metadata_pattern(self) -> None:
        """Should return True for eval.*.metadata pattern."""
        assert _should_convert_json("eval.test.metadata") is True
        assert _should_convert_json("eval.another.metadata") is True
        assert _should_convert_json("eval.foo.bar.metadata") is True

    def test_matches_open_inference_types(self) -> None:
        """Should return True for columns in OPEN_INFERENCE_JSON_STR_TYPES."""
        # These are from the constant set - we test with known values
        assert _should_convert_json("document.metadata") is True
        assert _should_convert_json("llm.function_call") is True
        assert _should_convert_json("metadata") is True

    def test_matches_result_column(self) -> None:
        """Should return True for column named 'result'."""
        assert _should_convert_json("result") is True

    def test_returns_false_for_other_columns(self) -> None:
        """Should return False for columns that don't match any pattern."""
        assert _should_convert_json("random_column") is False
        assert _should_convert_json("eval_metadata") is False  # Missing dots
        assert _should_convert_json("eval.test.notmetadata") is False
        assert _should_convert_json("results") is False  # Not exactly "result"
        assert (
            _should_convert_json("metadata_col") is False
        )  # metadata not alone
