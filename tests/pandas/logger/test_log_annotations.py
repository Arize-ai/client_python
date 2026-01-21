import datetime as dt
import json
import sys
import uuid
from datetime import timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from arize.pandas.logger import Client
from arize.pandas.proto.flight_pb2 import WriteSpanAnnotationResponse
from arize.utils.errors import AuthError

# Conditionally import tracing/validation components only for Python 3.8+
if sys.version_info >= (3, 8):
    from arize.pandas.tracing.columns import (
        ANNOTATION_LABEL_SUFFIX,
        ANNOTATION_NOTES_COLUMN_NAME,
        ANNOTATION_SCORE_SUFFIX,
        ANNOTATION_UPDATED_AT_SUFFIX,
        ANNOTATION_UPDATED_BY_SUFFIX,
        SPAN_SPAN_ID_COL,
    )
    from arize.pandas.validation.errors import (
        InvalidProjectName,
        InvalidTypeColumns,
        MissingColumns,
        ValidationFailure,
    )
else:
    # Define dummy classes/variables for Python < 3.8
    class ValidationFailure(Exception):
        pass

    class MissingColumns(Exception):
        pass

    class InvalidTypeColumns(Exception):
        pass

    class InvalidProjectName(Exception):
        pass

    # Dummy constants/objects needed for parsing even if tests are skipped
    SPAN_SPAN_ID_COL = type("SpanColumn", (), {"name": "context.span_id"})
    ANNOTATION_LABEL_SUFFIX = ".label"
    ANNOTATION_SCORE_SUFFIX = ".score"
    ANNOTATION_UPDATED_AT_SUFFIX = ".updated_at"
    ANNOTATION_UPDATED_BY_SUFFIX = ".updated_by"
    ANNOTATION_NOTES_COLUMN_NAME = "annotation.notes"


class MockFlightSession:
    """Mocks the pyarrow.flight.FlightClient interaction for testing."""

    def __init__(self, *args, **kwargs):
        self.call_options = None

    def connect(self):
        mock_client = MagicMock()
        mock_client.do_put.return_value = (MagicMock(), MagicMock())
        mock_response = WriteSpanAnnotationResponse()
        mock_response.records_updated = 0
        mock_client.do_put.return_value[1].read.return_value = MagicMock(
            to_pybytes=lambda: mock_response.SerializeToString()
        )
        return mock_client

    def close(self):
        pass


# Use a fixed recent date for generating timestamps
FIXED_TIMESTAMP_MS = int(
    dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
)


def get_annotation_df(num_rows=3, include_metadata=True):
    """Generates a sample pandas DataFrame for annotation logging tests."""
    span_ids = [str(uuid.uuid4()) for _ in range(num_rows)]
    data = {
        SPAN_SPAN_ID_COL.name: span_ids,
        f"annotation.quality{ANNOTATION_LABEL_SUFFIX}": ["good", "bad", "good"],
        f"annotation.quality{ANNOTATION_SCORE_SUFFIX}": [0.9, 0.1, 0.8],
        f"annotation.safety{ANNOTATION_LABEL_SUFFIX}": [
            "safe",
            "safe",
            "unsafe",
        ],
        ANNOTATION_NOTES_COLUMN_NAME: ["Note 1", None, "Note 3"],
        "extra_column_to_be_ignored": [1, 2, 3],
    }
    if include_metadata:
        data[f"annotation.safety{ANNOTATION_UPDATED_BY_SUFFIX}"] = [
            "user1",
            "user2",
            "user1",
        ]
        data[f"annotation.safety{ANNOTATION_UPDATED_AT_SUFFIX}"] = [
            FIXED_TIMESTAMP_MS - 10000,
            FIXED_TIMESTAMP_MS - 5000,
            FIXED_TIMESTAMP_MS,
        ]
    return pd.DataFrame(data)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
@patch(
    "arize.pandas.tracing.validation.common.value_validation._check_value_timestamp",
    return_value=[],
)
def test_log_annotations_success(_mock_check_ts):
    """Tests successful logging of annotations with valid inputs."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_annotation_df(include_metadata=True)
    project_name = "test_project"

    try:
        response = client.log_annotations(
            dataframe=test_df,
            project_name=project_name,
            validate=True,
            verbose=False,
        )
        assert isinstance(response, WriteSpanAnnotationResponse)
        assert response.records_updated == 0
    except Exception as e:
        pytest.fail(f"log_annotations raised an unexpected exception: {e}")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_log_annotations_missing_auth():
    """Tests that AuthError is raised if developer_key or space_id is missing."""
    test_df = get_annotation_df()
    project_name = "test_project"

    client_no_dev_key = Client(space_id="space_id")
    with pytest.raises(AuthError) as excinfo:
        client_no_dev_key.log_annotations(
            dataframe=test_df, project_name=project_name
        )
    assert excinfo.value.missing_developer_key is True
    assert excinfo.value.missing_space_id is False
    assert excinfo.value.method_name == "log_annotations"

    client_no_space_id = Client(developer_key="dev_key")
    with pytest.raises(AuthError) as excinfo:
        client_no_space_id.log_annotations(
            dataframe=test_df, project_name=project_name
        )
    assert excinfo.value.missing_space_id is True
    assert excinfo.value.missing_developer_key is False
    assert excinfo.value.method_name == "log_annotations"

    client_no_auth = Client()
    with pytest.raises(AuthError) as excinfo:
        client_no_auth.log_annotations(
            dataframe=test_df, project_name=project_name
        )
    assert excinfo.value.missing_space_id is True
    assert excinfo.value.missing_developer_key is True
    assert excinfo.value.method_name == "log_annotations"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_annotations_missing_project_name():
    """Tests that ValidationFailure is raised if project_name is missing."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_annotation_df()
    with pytest.raises(ValidationFailure) as excinfo:
        client.log_annotations(dataframe=test_df, project_name=None)
    assert any(isinstance(e, InvalidProjectName) for e in excinfo.value.errors)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_annotations_missing_span_id():
    """Tests that ValidationFailure is raised if context.span_id is missing."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_annotation_df()
    test_df_no_span_id = test_df.drop(columns=[SPAN_SPAN_ID_COL.name])
    project_name = "test_project_missing_span_id"
    with pytest.raises(ValidationFailure) as excinfo:
        client.log_annotations(
            dataframe=test_df_no_span_id, project_name=project_name
        )
    assert SPAN_SPAN_ID_COL.name in str(excinfo.value.errors[0])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_annotations_invalid_column_name():
    """Tests that ValidationFailure is raised for invalid annotation column names."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_annotation_df()
    test_df["annotation.quality"] = "invalid"
    project_name = "test_project_invalid_col"
    with pytest.raises(ValidationFailure) as excinfo:
        client.log_annotations(dataframe=test_df, project_name=project_name)
    error_messages = [str(e) for e in excinfo.value.errors]
    assert any("annotation.quality" in msg for msg in error_messages)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.Client._log_arrow_flight")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
@patch(
    "arize.pandas.tracing.validation.common.value_validation._check_value_timestamp",
    return_value=[],
)
def test_log_annotations_autogenerates_metadata(
    _mock_check_ts, mock_log_arrow_flight
):
    """Tests that missing updated_by and updated_at columns are autogenerated."""
    mock_log_arrow_flight.return_value = WriteSpanAnnotationResponse(
        records_updated=0
    )

    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_annotation_df(include_metadata=False)
    project_name = "test_project_metadata_gen"

    try:
        response = client.log_annotations(
            dataframe=test_df, project_name=project_name
        )
        assert isinstance(response, WriteSpanAnnotationResponse)
        assert response.records_updated == 0
    except Exception as e:
        pytest.fail(f"log_annotations raised an unexpected exception: {e}")

    mock_log_arrow_flight.assert_called_once()
    call_args, call_kwargs = mock_log_arrow_flight.call_args
    pa_table = call_kwargs.get("pa_table", call_args[0] if call_args else None)
    assert pa_table is not None
    assert isinstance(pa_table, pa.Table)
    result_df = pa_table.to_pandas()

    expected_updated_by_col = (
        f"annotation.quality{ANNOTATION_UPDATED_BY_SUFFIX}"
    )
    expected_updated_at_col = (
        f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"
    )
    assert expected_updated_by_col in result_df.columns
    assert expected_updated_at_col in result_df.columns
    assert (result_df[expected_updated_by_col] == "SDK").all()

    if ANNOTATION_NOTES_COLUMN_NAME in test_df.columns:
        assert ANNOTATION_NOTES_COLUMN_NAME in result_df.columns
        try:
            notes_cell_content = result_df.loc[0, ANNOTATION_NOTES_COLUMN_NAME]
            first_element = notes_cell_content[0]
            assert isinstance(first_element, str)
            first_note_obj = json.loads(first_element)
            assert isinstance(first_note_obj, dict)
            assert "text" in first_note_obj
            assert first_note_obj["updated_by"] == "SDK"
            assert "updated_at" in first_note_obj
        except (IndexError, TypeError, json.JSONDecodeError) as e:
            pytest.fail(f"Notes column content check failed: {e}")
        assert pd.isna(result_df.loc[1, ANNOTATION_NOTES_COLUMN_NAME])

    assert "extra_column_to_be_ignored" not in result_df.columns


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_annotations_invalid_data_types():
    """Tests that ValidationFailure is raised for invalid data types."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_annotation_df(include_metadata=True)
    project_name = "test_project_invalid_types"

    score_col = f"annotation.quality{ANNOTATION_SCORE_SUFFIX}"
    label_col = f"annotation.safety{ANNOTATION_LABEL_SUFFIX}"
    updated_at_col = f"annotation.safety{ANNOTATION_UPDATED_AT_SUFFIX}"

    test_df_invalid = test_df.copy()
    test_df_invalid.loc[0, score_col] = "not_a_number"
    test_df_invalid.loc[1, label_col] = 123
    test_df_invalid.loc[2, updated_at_col] = "not_a_timestamp"
    test_df_invalid.loc[0, ANNOTATION_NOTES_COLUMN_NAME] = 123

    with pytest.raises(ValidationFailure) as excinfo:
        client.log_annotations(
            dataframe=test_df_invalid, project_name=project_name
        )

    error_messages = [str(e) for e in excinfo.value.errors]
    assert any(score_col in msg for msg in error_messages), (
        f"Error for {score_col}"
    )
    assert any(label_col in msg for msg in error_messages), (
        f"Error for {label_col}"
    )
    assert any(updated_at_col in msg for msg in error_messages), (
        f"Error for {updated_at_col}"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.Client._log_arrow_flight")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_annotations_empty_dataframe(mock_log_arrow_flight):
    """Tests that logging an empty DataFrame raises ValidationFailure."""
    mock_log_arrow_flight.return_value = WriteSpanAnnotationResponse(
        records_updated=0
    )
    client = Client(developer_key="dev_key", space_id="space_id")
    empty_df = pd.DataFrame(
        columns=[
            SPAN_SPAN_ID_COL.name,
            f"annotation.quality{ANNOTATION_LABEL_SUFFIX}",
        ]
    )
    project_name = "test_project_empty_df"

    with pytest.raises(ValidationFailure):
        client.log_annotations(dataframe=empty_df, project_name=project_name)

    mock_log_arrow_flight.assert_not_called()
