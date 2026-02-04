"""Common validation error classes for spans."""

from arize.constants.ml import (
    MAX_EMBEDDING_DIMENSIONALITY,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
)
from arize.constants.spans import (
    JSON_STRING_MAX_STR_LENGTH,
    SPAN_DOCUMENT_CONTENT_MAX_STR_LENGTH,
    SPAN_DOCUMENT_ID_MAX_STR_LENGTH,
    SPAN_EMBEDDING_TEXT_MAX_STR_LENGTH,
    SPAN_EVENT_NAME_MAX_STR_LENGTH,
    SPAN_LLM_MESSAGE_CONTENT_MAX_STR_LENGTH,
    SPAN_LLM_MESSAGE_ROLE_MAX_STR_LENGTH,
)
from arize.exceptions.base import ValidationError
from arize.logging import log_a_list

# -------------------
# Direct Argument Checks
# -------------------


class InvalidTypeArgument(ValidationError):
    """Raised when an argument has an invalid type."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Type_Argument"

    def __init__(self, arg_name: str, arg_type: str, wrong_arg: object) -> None:
        """Initialize the exception with argument type validation context.

        Args:
            arg_name: Name of the argument with invalid type.
            arg_type: Expected type for the argument.
            wrong_arg: Actual argument value that was invalid.
        """
        self.arg_name = arg_name
        self.arg_type = arg_type
        self.wrong_arg = wrong_arg

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"The {self.arg_name} must be a {self.arg_type}. Found {type(self.wrong_arg)}"


class InvalidDateTimeFormatType(ValidationError):
    """Raised when datetime format type is invalid or not supported."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_DateTime_Format_Type"

    def __init__(self, wrong_input: object) -> None:
        """Initialize the exception with datetime format validation context.

        Args:
            wrong_input: Invalid input that was provided for datetime format.
        """
        self.wrong_input = wrong_input

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"The date time format must be a string. Found {type(self.wrong_input)}"


# ---------------------
# DataFrame Form Checks
# ---------------------


class InvalidDataFrameDuplicateColumns(ValidationError):
    """Raised when dataframe contains duplicate column names."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_DataFrame_Duplicate_Columns"

    def __init__(self, duplicate_cols: list[str]) -> None:
        """Initialize the exception with duplicate columns context.

        Args:
            duplicate_cols: List of column names that have duplicates in the dataframe.
        """
        self.duplicate_cols = duplicate_cols

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The following columns have duplicates in the dataframe: "
            f"{log_a_list(self.duplicate_cols, 'and')}"
        )


class InvalidDataFrameMissingColumns(ValidationError):
    """Raised when required columns are missing from dataframe."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_DataFrame_Missing_Columns"

    def __init__(self, missing_cols: list[str]) -> None:
        """Initialize the exception with missing columns context.

        Args:
            missing_cols: List of required columns that are missing from the dataframe.
        """
        self.missing_cols = missing_cols

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The following columns are missing in the dataframe and are required: "
            f"{log_a_list(self.missing_cols, 'and')}"
        )


class InvalidDataFrameColumnContentTypes(ValidationError):
    """Raised when dataframe column content types are invalid."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_DataFrame_Column_Content_Types"

    def __init__(
        self, invalid_type_cols: list[str], expected_type: str
    ) -> None:
        """Initialize the exception with column content type validation context.

        Args:
            invalid_type_cols: List of columns with incorrect content types.
            expected_type: Expected content type for the columns.
        """
        self.invalid_type_cols = invalid_type_cols
        self.expected_type = expected_type

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "Found dataframe columns containing the wrong data type. "
            f"The following columns should contain {self.expected_type}: "
            f"{log_a_list(self.invalid_type_cols, 'and')}"
        )


# -----------------------
# DataFrame Values Checks
# -----------------------


class InvalidMissingValueInColumn(ValidationError):
    """Raised when column contains null or missing values."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Missin_Value_In_Column"

    def __init__(self, col_name: str) -> None:
        """Initialize the exception with missing value context.

        Args:
            col_name: Name of the column containing missing values.
        """
        self.col_name = col_name

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The column '{self.col_name}' has at least one missing value. "
            "This column must not have missing values"
        )


class InvalidStringLengthInColumn(ValidationError):
    """Raised when string values in column exceed length limits."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_String_Length_In_Column"

    def __init__(self, col_name: str, min_length: int, max_length: int) -> None:
        """Initialize the exception with string length validation context.

        Args:
            col_name: Name of the column with invalid string lengths.
            min_length: Minimum acceptable string length.
            max_length: Maximum acceptable string length.
        """
        self.col_name = col_name
        self.min_length = min_length
        self.max_length = max_length

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The column '{self.col_name}' contains invalid string values, "
            f"their length must be between {self.min_length} and {self.max_length}."
        )


class InvalidJsonStringInColumn(ValidationError):
    """Raised when JSON string in column is invalid or malformed."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Json_String_In_Column"

    def __init__(self, col_name: str) -> None:
        """Initialize the exception with JSON string validation context.

        Args:
            col_name: Name of the column containing invalid JSON strings.
        """
        self.col_name = col_name

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The column '{self.col_name}' contains invalid JSON string values."
        )


class InvalidStringValueNotAllowedInColumn(ValidationError):
    """Raised when column contains disallowed string values."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_String_Value_Not_Allowed_In_Column"

    def __init__(self, col_name: str, allowed_values: list[str]) -> None:
        """Initialize the exception with allowed string values validation context.

        Args:
            col_name: Name of the column containing disallowed values.
            allowed_values: List of values that are allowed in the column.
        """
        self.col_name = col_name
        self.allowed_values = allowed_values

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The column '{self.col_name}' contains invalid string values. "
            f"Allowed values are {log_a_list(self.allowed_values, 'and')}"
        )


class InvalidTimestampValueInColumn(ValidationError):
    """Raised when timestamp values in column are outside acceptable range."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Timestamp_Value_In_Column"

    def __init__(self, timestamp_col_name: str) -> None:
        """Initialize the exception with timestamp validation context.

        Args:
            timestamp_col_name: Name of the column containing invalid timestamp values.
        """
        self.timestamp_col_name = timestamp_col_name

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"At least one timestamp in the column '{self.timestamp_col_name}' is out of range. "
            f"Timestamps must be within {MAX_FUTURE_YEARS_FROM_CURRENT_TIME} year "
            f"in the future and {MAX_PAST_YEARS_FROM_CURRENT_TIME} years in the past from "
            "the current time."
        )


class InvalidStartAndEndTimeValuesInColumn(ValidationError):
    """Raised when start time is not before end time in span records."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Start_And_End_Time_Values_In_Column"

    def __init__(self, greater_col_name: str, less_col_name: str) -> None:
        """Initialize the exception with span time validation context.

        Args:
            greater_col_name: Name of the column that should have greater values (end time).
            less_col_name: Name of the column that should have lesser values (start time).
        """
        self.greater_col_name = greater_col_name
        self.less_col_name = less_col_name

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"Invalid span times. Values in column '{self.greater_col_name}' "
            f"should be greater than values in column '{self.less_col_name}'"
        )


class InvalidEventValueInColumn(ValidationError):
    """Raised when event values in column are invalid or malformed."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Event_Value_In_Column"

    def __init__(
        self,
        col_name: str,
        wrong_name: bool,
        wrong_time: bool,
        wrong_attrs: bool,
    ) -> None:
        """Initialize the exception with event value validation context.

        Args:
            col_name: Name of the column containing invalid events.
            wrong_name: Whether event names are invalid.
            wrong_time: Whether event timestamps are invalid.
            wrong_attrs: Whether event attributes are invalid.
        """
        self.col_name = col_name
        self.wrong_name = wrong_name
        self.wrong_time = wrong_time
        self.wrong_attrs = wrong_attrs

    def error_message(self) -> str:
        """Return the error message for this exception."""
        msg = f"Found at least one invalid event in column '{self.col_name}'. "
        if self.wrong_name:
            msg += (
                "Detected invalid names. Names must contain a maximum "
                f"of {SPAN_EVENT_NAME_MAX_STR_LENGTH} characters."
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
    """Raised when LLM message values in column are invalid or malformed."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_LLM_Message_Value_In_Column"

    def __init__(
        self,
        col_name: str,
        wrong_role: bool,
        wrong_content: bool,
        wrong_tool_calls: bool,
    ) -> None:
        """Initialize the exception with LLM message validation context.

        Args:
            col_name: Name of the column containing invalid LLM messages.
            wrong_role: Whether message roles are invalid.
            wrong_content: Whether message contents are invalid.
            wrong_tool_calls: Whether tool calls are invalid.
        """
        self.col_name = col_name
        self.wrong_role = wrong_role
        self.wrong_content = wrong_content
        self.wrong_tool_calls = wrong_tool_calls

    def error_message(self) -> str:
        """Return the error message for this exception."""
        msg = f"Found at least one invalid LLM message in column '{self.col_name}'. "
        if self.wrong_role:
            msg += (
                "Detected invalid roles. Roles must contain a maximum "
                f"of {SPAN_LLM_MESSAGE_ROLE_MAX_STR_LENGTH} characters."
            )
        if self.wrong_content:
            msg += (
                "Detected invalid contents. Contents must contain a maximum"
                f"of {SPAN_LLM_MESSAGE_CONTENT_MAX_STR_LENGTH} characters."
            )
        if self.wrong_tool_calls:
            msg += (
                "Detected invalid tool calls. Each tool call must contain a maximum "
                f"of {JSON_STRING_MAX_STR_LENGTH} "
                "characters and be a valid JSON strings."
            )
        return msg


class InvalidEmbeddingValueInColumn(ValidationError):
    """Raised when embedding values in column are invalid or malformed."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Embedding_Value_In_Column"

    def __init__(
        self, col_name: str, wrong_vector: bool, wrong_text: bool
    ) -> None:
        """Initialize the exception with embedding value validation context.

        Args:
            col_name: Name of the column containing invalid embeddings.
            wrong_vector: Whether embedding vectors are invalid.
            wrong_text: Whether embedding texts are invalid.
        """
        self.col_name = col_name
        self.wrong_vector = wrong_vector
        self.wrong_text = wrong_text

    def error_message(self) -> str:
        """Return the error message for this exception."""
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
                f"of {SPAN_EMBEDDING_TEXT_MAX_STR_LENGTH} characters."
            )
        return msg


class InvalidDocumentValueInColumn(ValidationError):
    """Raised when document values in column are invalid or malformed."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Document_Value_In_Column"

    def __init__(
        self,
        col_name: str,
        wrong_id: bool,
        wrong_content: bool,
        wrong_metadata: bool,
    ) -> None:
        """Initialize the exception with document value validation context.

        Args:
            col_name: Name of the column containing invalid documents.
            wrong_id: Whether document IDs are invalid.
            wrong_content: Whether document contents are invalid.
            wrong_metadata: Whether document metadata is invalid.
        """
        self.col_name = col_name
        self.wrong_id = wrong_id
        self.wrong_content = wrong_content
        self.wrong_metadata = wrong_metadata

    def error_message(self) -> str:
        """Return the error message for this exception."""
        msg = (
            f"Found at least one invalid document in column '{self.col_name}'. "
        )
        if self.wrong_id:
            msg += (
                "Detected invalid ids. Document ids must contain a maximum "
                f"of {SPAN_DOCUMENT_ID_MAX_STR_LENGTH} characters."
            )
        if self.wrong_content:
            msg += (
                "Detected invalid contents. Document contents must contain a maximum"
                f"of {SPAN_DOCUMENT_CONTENT_MAX_STR_LENGTH} characters."
            )
        if self.wrong_metadata:
            msg += (
                "Detected invalid document metadata. Document metadata must contain "
                f"a maximum of {JSON_STRING_MAX_STR_LENGTH} "
                "characters and be a valid JSON strings."
            )
        return msg


class InvalidFloatValueInColumn(ValidationError):
    """Raised when float values in column are invalid or out of range."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Float_Value_In_Column"

    def __init__(self, col_name: str) -> None:
        """Initialize the exception with float value validation context.

        Args:
            col_name: Name of the column containing invalid float values.
        """
        self.col_name = col_name

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The column '{self.col_name}' contains invalid float values. "
            f"Invalid values are +/- infinite values."
        )


class InvalidNullEvalLabelAndScore(ValidationError):
    """Raised when both eval label and score are null in a record."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Null_Eval_Label_And_Score"

    def __init__(self, eval_names: list[str]) -> None:
        """Initialize the exception with eval label and score validation context.

        Args:
            eval_names: List of eval names missing both label and score.
        """
        self.eval_names = eval_names

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"There is at least one row without a label and score for the following evals: "
            f"{log_a_list(self.eval_names, 'and')}"
        )


class DuplicateAnnotationNameInSpan(ValidationError):
    """Raised when a span contains duplicate annotation names."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Duplicate_Annotation_Name_In_Span"

    def __init__(self, span_id: str, duplicate_names: list[str]) -> None:
        """Initialize the exception with duplicate annotation names context.

        Args:
            span_id: ID of the span containing duplicate annotations.
            duplicate_names: List of annotation names that are duplicated.
        """
        self.span_id = span_id
        self.duplicate_names = duplicate_names

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"Found duplicate annotation/eval names within the same span_id '{self.span_id}'. "
            f"Duplicate names: {log_a_list(self.duplicate_names, 'and')}. "
            "Each annotation/eval name (e.g., 'quality' in 'annotation.quality.label') "
            "must be unique per span."
        )


class InvalidNullAnnotationLabelAndScore(ValidationError):
    """Raised when both annotation label and score are null in a record."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Null_Annotation_Label_And_Score"

    def __init__(self, annotation_names: list[str]) -> None:
        """Initialize the exception with annotation validation context.

        Args:
            annotation_names: List of annotation names missing both label and score.
        """
        self.annotation_names = annotation_names

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "There is at least one row where both label and score are missing for the "
            f"following annotations: {log_a_list(self.annotation_names, 'and')}. "
            "Each annotation must have at least a label or a score defined."
        )


class InvalidAnnotationColumnFormat(ValidationError):
    """Raised when annotation column format is invalid or malformed."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Annotation_Column_Format"

    def __init__(
        self, invalid_format_cols: list[str], expected_format: str
    ) -> None:
        """Initialize the exception with annotation column format validation context.

        Args:
            invalid_format_cols: List of columns with invalid annotation format.
            expected_format: Expected format for annotation columns.
        """
        self.invalid_format_cols = invalid_format_cols
        self.expected_format = expected_format

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The following columns have an invalid annotation column format: "
            f"{log_a_list(self.invalid_format_cols, 'and')}. "
            f"Annotation columns must follow the format: {self.expected_format}"
        )
