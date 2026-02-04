import json
import logging
from collections.abc import Callable
from typing import Any, TypeGuard

import numpy as np
import pandas as pd

from arize.spans.columns import (
    SPAN_ATTRIBUTES_EMBEDDING_EMBEDDINGS_COL,
    SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL,
    SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL,
    SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL,
    SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VARIABLES_COL,
    SPAN_ATTRIBUTES_LLM_TOOLS_COL,
    SPAN_ATTRIBUTES_METADATA,
    SPAN_ATTRIBUTES_RETRIEVAL_DOCUMENTS_COL,
    SPAN_ATTRIBUTES_TOOL_PARAMETERS_COL,
    SPAN_END_TIME_COL,
    SPAN_START_TIME_COL,
)

logger = logging.getLogger(__name__)


# Data transformer for Otel tracing data into types and values that are more ergonomic for users needing
# to interact with the data in Python; This class is intended to be used by Arize and not by users
# Any errors encountered are unexpected since Arize also controls the data types returned from the platform
# but the resulting error messages provide clarity on what the effect
# of the error is on the data; It should not prevent a user from continuing to use the data
class OtelTracingDataTransformer:
    def _apply_column_transformation(
        self,
        df: pd.DataFrame,
        col_name: str,
        transform_func: Callable[[Any], Any],
    ) -> str | None:
        """Apply a transformation to a column and return error message if it fails."""
        try:
            df[col_name] = df[col_name].apply(transform_func)
        except Exception as e:
            return (
                f"Unable to transform json string data to a Python dict in column '{col_name}'; "
                f"May encounter issues when importing data back into Arize; Error: {e}"
            )
        else:
            return None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        errors: list[str] = []

        # Convert list of json serializable strings columns to list of dictionaries for more
        # conveinent data processing in Python
        list_of_json_string_column_names: list[str] = [
            col.name
            for col in [
                SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL,
                SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL,
                SPAN_ATTRIBUTES_EMBEDDING_EMBEDDINGS_COL,
                SPAN_ATTRIBUTES_RETRIEVAL_DOCUMENTS_COL,
                SPAN_ATTRIBUTES_LLM_TOOLS_COL,
            ]
            if col.name in df.columns
        ]
        for col_name in list_of_json_string_column_names:
            error = self._apply_column_transformation(
                df, col_name, self._transform_value_to_list_of_dict
            )
            if error:
                errors.append(error)

        json_string_column_names: list[str] = [
            col.name
            for col in [
                SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VARIABLES_COL,
                SPAN_ATTRIBUTES_METADATA,
            ]
            if col.name in df.columns
        ]
        for col_name in json_string_column_names:
            error = self._apply_column_transformation(
                df, col_name, self._transform_json_to_dict
            )
            if error:
                errors.append(error)

        # Clean json string columns since empty strings are equivalent here to None but are not valid json
        dirty_string_column_names: list[str] = [
            col.name
            for col in [
                SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL,
                SPAN_ATTRIBUTES_TOOL_PARAMETERS_COL,
            ]
            if col.name in df.columns
        ]
        for col_name in dirty_string_column_names:
            df[col_name] = df[col_name].apply(self._clean_json_string)  # type: ignore[arg-type]

        # Convert timestamp columns to datetime objects
        timestamp_column_names: list[str] = [
            col.name
            for col in [
                SPAN_START_TIME_COL,
                SPAN_END_TIME_COL,
            ]
            if col.name in df.columns
        ]
        for col_name in timestamp_column_names:
            df[col_name] = df[col_name].apply(
                self._convert_timestamp_to_datetime  # type: ignore[arg-type]
            )

        for err in errors:
            logger.warning(err)

        return df

    def _transform_value_to_list_of_dict(
        self, value: object
    ) -> list[object] | None:
        if value is None:
            return None

        if isinstance(value, (list, np.ndarray)):
            return [
                self._deserialize_json_string_to_dict(i)
                for i in value
                if self._is_non_empty_string(i)
            ]
        if self._is_non_empty_string(value):
            return [self._deserialize_json_string_to_dict(value)]
        return None

    def _transform_json_to_dict(self, value: object) -> object | None:
        if value is None:
            return None

        if self._is_non_empty_string(value):
            return self._deserialize_json_string_to_dict(value)

        if isinstance(value, str) and value == "":
            # transform empty string to None
            return None
        return None

    def _is_non_empty_string(self, value: object) -> TypeGuard[str]:
        return isinstance(value, str) and value != ""

    def _deserialize_json_string_to_dict(self, value: str) -> object:
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {value}") from e

    def _clean_json_string(self, value: object) -> object | None:
        return value if self._is_non_empty_string(value) else None

    def _convert_timestamp_to_datetime(self, value: object) -> object:
        return (
            pd.Timestamp(value, unit="ns")
            if value and isinstance(value, (int, float, np.int64))
            else value
        )
