from typing import Any, List

import arize.pandas.tracing.constants as tracing_constants
from arize.pandas.validation.errors import ValidationError
from arize.utils.constants import (
    MAX_EMBEDDING_DIMENSIONALITY,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
)
from arize.utils.logging import log_a_list

# -------------------
# Direct Argument Checks
# -------------------


class InvalidTypeArgument(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_Argument"

    def __init__(self, arg_name: str, arg_type: str, wrong_arg: Any) -> None:
        self.arg_name = arg_name
        self.arg_type = arg_type
        self.wrong_arg = wrong_arg

    def error_message(self) -> str:
        return (
            f"The {self.arg_name} must be a {self.arg_type}. ",
            f"Found {type(self.wrong_arg)}",
        )


class InvalidDateTimeFormatType(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_DateTime_Format_Type"

    def __init__(self, wrong_input: Any) -> None:
        self.wrong_input = wrong_input

    def error_message(self) -> str:
        return (
            "The date time format must be a string. ",
            f"Found {type(self.wrong_input)}",
        )


# ---------------------
# DataFrame Form Checks
# ---------------------


class InvalidDataFrameDuplicateColumns(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_DataFrame_Duplicate_Columns"

    def __init__(self, duplicate_cols: List[str]) -> None:
        self.duplicate_cols = duplicate_cols

    def error_message(self) -> str:
        return (
            f"The following columns have duplicates in the dataframe: "
            f"{log_a_list(self.duplicate_cols, 'and')}"
        )


class InvalidDataFrameMissingColumns(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_DataFrame_Missing_Columns"

    def __init__(self, missing_cols: List[str]) -> None:
        self.missing_cols = missing_cols

    def error_message(self) -> str:
        return (
            f"The following columns are missing in the dataframe and are required: "
            f"{log_a_list(self.missing_cols, 'and')}"
        )


class InvalidDataFrameColumnContentTypes(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_DataFrame_Column_Content_Types"

    def __init__(self, invalid_type_cols: List[str], expected_type: str) -> None:
        self.invalid_type_cols = invalid_type_cols
        self.expected_type = expected_type

    def error_message(self) -> str:
        return (
            "Found dataframe columns containing the wrong data type. "
            f"The following columns should contain {self.expected_type}: "
            f"{log_a_list(self.invalid_type_cols, 'and')}"
        )


# -----------------------
# DataFrame Values Checks
# -----------------------


class InvalidMissingValueInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Missin_Value_In_Column"

    def __init__(self, col_name: str) -> None:
        self.col_name = col_name

    def error_message(self) -> str:
        return (
            f"The column '{self.col_name}' has at least one missing value. "
            "This column must not have missing values"
        )


class InvalidStringLengthInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_String_Length_In_Column"

    def __init__(self, col_name: str, min_length: int, max_length: int) -> None:
        self.col_name = col_name
        self.min_length = min_length
        self.max_length = max_length

    def error_message(self) -> str:
        return (
            f"The column '{self.col_name}' contains invalid string values, "
            f"their length must be between {self.min_length} and {self.max_length}."
        )


class InvalidJsonStringInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Json_String_In_Column"

    def __init__(self, col_name: str) -> None:
        self.col_name = col_name

    def error_message(self) -> str:
        return f"The column '{self.col_name}' contains invalid JSON string values."


class InvalidStringValueNotAllowedInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_String_Value_Not_Allowed_In_Column"

    def __init__(self, col_name: str, allowed_values: List[str]) -> None:
        self.col_name = col_name
        self.allowed_values = allowed_values

    def error_message(self) -> str:
        return (
            f"The column '{self.col_name}' contains invalid string values. "
            f"Allowed values are {log_a_list(self.allowed_values,'and')}"
        )


class InvalidTimestampValueInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Timestamp_Value_In_Column"

    def __init__(self, timestamp_col_name: str) -> None:
        self.timestamp_col_name = timestamp_col_name

    def error_message(self) -> str:
        return (
            f"At least one timestamp in the column '{self.timestamp_col_name}' is out of range. "
            f"Timestamps must be within {MAX_FUTURE_YEARS_FROM_CURRENT_TIME} year "
            f"in the future and {MAX_PAST_YEARS_FROM_CURRENT_TIME} years in the past from "
            "the current time."
        )


class InvalidStartAndEndTimeValuesInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Start_And_End_Time_Values_In_Column"

    def __init__(self, greater_col_name: str, less_col_name: str) -> None:
        self.greater_col_name = greater_col_name
        self.less_col_name = less_col_name

    def error_message(self) -> str:
        return (
            f"Invalid span times. Values in column '{self.greater_col_name}' "
            f"should be greater than values in column '{self.less_col_name}'"
        )


class InvalidEventValueInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Event_Value_In_Column"

    def __init__(
        self, col_name: str, wrong_name: bool, wrong_time: bool, wrong_attrs: bool
    ) -> None:
        self.col_name = col_name
        self.wrong_name = wrong_name
        self.wrong_time = wrong_time
        self.wrong_attrs = wrong_attrs

    def error_message(self) -> str:
        msg = f"Found at least one invalid event in column '{self.col_name}'. "
        if self.wrong_name:
            msg += (
                "Detected invalid names. Names must contain a maximum "
                f"of {tracing_constants.SPAN_EVENT_NAME_MAX_STR_LENGTH} characters."
            )
        if self.wrong_time:
            msg += (
                "Detected invalid times. Timestamps must contain a positive "
                "value in nanoseconds."
            )
        if self.wrong_attrs:
            msg += (
                "Detected invalid attributes. Attributes must be dictionaries "
                "with string keys and serializable to JSON string."
            )
        return msg


class InvalidLLMMessageValueInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_LLM_Message_Value_In_Column"

    def __init__(
        self,
        col_name: str,
        wrong_role: bool,
        wrong_content: bool,
        wrong_tool_calls: bool,
    ) -> None:
        self.col_name = col_name
        self.wrong_role = wrong_role
        self.wrong_content = wrong_content
        self.wrong_tool_calls = wrong_tool_calls

    def error_message(self) -> str:
        msg = f"Found at least one invalid LLM message in column '{self.col_name}'. "
        if self.wrong_role:
            msg += (
                "Detected invalid roles. Roles must contain a maximum "
                f"of {tracing_constants.SPAN_LLM_MESSAGE_ROLE_MAX_STR_LENGTH} characters."
            )
        if self.wrong_content:
            msg += (
                "Detected invalid contents. Contents must contain a maximum"
                f"of {tracing_constants.SPAN_LLM_MESSAGE_CONTENT_MAX_STR_LENGTH} characters."
            )
        if self.wrong_tool_calls:
            msg += (
                "Detected invalid tool calls. Each tool call must contain a maximum "
                f"of {tracing_constants.JSON_STRING_MAX_STR_LENGTH} "
                "characters and be a valid JSON strings."
            )
        return msg


class InvalidEmbeddingValueInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Embedding_Value_In_Column"

    def __init__(self, col_name: str, wrong_vector: bool, wrong_text: bool) -> None:
        self.col_name = col_name
        self.wrong_vector = wrong_vector
        self.wrong_text = wrong_text

    def error_message(self) -> str:
        msg = f"Found at least one invalid embedding object in column '{self.col_name}'. "
        if self.wrong_vector:
            msg += (
                "Detected invalid vectors. Vectors must contain a maximum "
                f"of {MAX_EMBEDDING_DIMENSIONALITY} components and minimum of 1 "
                "character."
            )
        if self.wrong_text:
            msg += (
                "Detected invalid texts. Texts must contain a maximum "
                f"of {tracing_constants.SPAN_EMBEDDING_TEXT_MAX_STR_LENGTH} characters."
            )
        return msg


class InvalidDocumentValueInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Document_Value_In_Column"

    def __init__(
        self, col_name: str, wrong_id: bool, wrong_content: bool, wrong_metadata: bool
    ) -> None:
        self.col_name = col_name
        self.wrong_id = wrong_id
        self.wrong_content = wrong_content
        self.wrong_metadata = wrong_metadata

    def error_message(self) -> str:
        msg = f"Found at least one invalid document in column '{self.col_name}'. "
        if self.wrong_id:
            msg += (
                "Detected invalid ids. Document ids must contain a maximum "
                f"of {tracing_constants.SPAN_DOCUMENT_ID_MAX_STR_LENGTH} characters."
            )
        if self.wrong_content:
            msg += (
                "Detected invalid contents. Document contents must contain a maximum"
                f"of {tracing_constants.SPAN_DOCUMENT_CONTENT_MAX_STR_LENGTH} characters."
            )
        if self.wrong_metadata:
            msg += (
                "Detected invalid document metadata. Document metadata must contain "
                f"a maximum of {tracing_constants.JSON_STRING_MAX_STR_LENGTH} "
                "characters and be a valid JSON strings."
            )
        return msg


class InvalidFloatValueInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Float_Value_In_Column"

    def __init__(self, col_name: str) -> None:
        self.col_name = col_name

    def error_message(self) -> str:
        return (
            f"The column '{self.col_name}' contains invalid float values. "
            f"Invalid values are +/- infinite values."
        )


class InvalidNullEvalLabelAndScore(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Null_Eval_Label_And_Score"

    def __init__(self, eval_names: List[str]) -> None:
        self.eval_names = eval_names

    def error_message(self) -> str:
        return (
            f"There is at least one row without a label and score for the following evals: "
            f"{log_a_list(self.eval_names, 'and')}"
        )
