"""Type validation exception classes."""

from collections.abc import Iterable

from arize.constants.ml import (
    MAX_EMBEDDING_DIMENSIONALITY,
    MAX_RAW_DATA_CHARACTERS,
)
from arize.exceptions.base import ValidationError


class InvalidType(ValidationError):
    """Raised when a field has an invalid type compared to expected types."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Type"

    def __init__(
        self, name: str, expected_types: list[str], found_data_type: str
    ) -> None:
        """Initialize the exception with type validation context.

        Args:
            name: Name of the field with invalid type.
            expected_types: List of expected data types.
            found_data_type: Actual data type found.
        """
        self.name = name
        self.expected_types = expected_types
        self.found_data_type = found_data_type

    def error_message(self) -> str:
        """Return the error message for this exception."""
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(map(str, self.expected_types[:-1]))} or {self.expected_types[-1]}"
        )
        return (
            f"{self.name} must be of type {type_list} but found {self.found_data_type}. "
            "Warning: if you are sending a column with integers, presence of a null "
            "value can convert the data type of the entire column to float."
        )


class InvalidTypeColumns(ValidationError):
    """Raised when columns have invalid types compared to expected types."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Type_Columns"

    def __init__(
        self, wrong_type_columns: list[str], expected_types: list[str]
    ) -> None:
        """Initialize the exception with column type validation context.

        Args:
            wrong_type_columns: Columns with incorrect data types.
            expected_types: List of expected data types for the columns.
        """
        self.wrong_type_columns = wrong_type_columns
        self.expected_types = expected_types

    def error_message(self) -> str:
        """Return the error message for this exception."""
        col_list = (
            self.wrong_type_columns[0]
            if len(self.wrong_type_columns) == 1
            else f"{', '.join(self.wrong_type_columns[:-1])}, and {self.wrong_type_columns[-1]}"
        )
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(map(str, self.expected_types[:-1]))} or {self.expected_types[-1]}"
        )
        return f"The column(s) {col_list}; must be of type {type_list}."


class InvalidTypeFeatures(ValidationError):
    """Raised when feature columns have invalid types."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Type_Features"

    def __init__(self, cols: Iterable, expected_types: list[str]) -> None:
        """Initialize the exception with feature type validation context.

        Args:
            cols: Feature columns with unrecognized data types.
            expected_types: List of expected data types for features.
        """
        self.wrong_type_columns = cols
        self.expected_types = expected_types

    def error_message(self) -> str:
        """Return the error message for this exception."""
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(map(str, self.expected_types[:-1]))} or {self.expected_types[-1]}"
        )
        return (
            f"Features must be of type {type_list}. "
            "The following feature columns have unrecognized data types: "
            f"{', '.join(map(str, self.wrong_type_columns))}."
        )


class InvalidFieldTypePromptTemplates(ValidationError):
    """Raised when prompt template field has invalid type."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Input_Type_Prompt_Templates"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "prompt_template_column_names must be of type PromptTemplateColumnNames"


class InvalidFieldTypeLlmConfig(ValidationError):
    """Raised when LLM config field has invalid type."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Input_Type_LLM_Config"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "llm_config_column_names must be of type LLMConfigColumnNames"


class InvalidTypeTags(ValidationError):
    """Raised when tag columns have invalid types."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Type_Tags"

    def __init__(self, cols: Iterable, expected_types: list[str]) -> None:
        """Initialize the exception with tag type validation context.

        Args:
            cols: Tag columns with unrecognized data types.
            expected_types: List of expected data types for tags.
        """
        self.wrong_type_columns = cols
        self.expected_types = expected_types

    def error_message(self) -> str:
        """Return the error message for this exception."""
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(map(str, self.expected_types[:-1]))} or {self.expected_types[-1]}"
        )
        return (
            f"Tags must be of type {type_list}. "
            "The following tag columns have unrecognized data types: "
            f"{', '.join(map(str, self.wrong_type_columns))}."
        )


class InvalidValueEmbeddingVectorDimensionality(ValidationError):
    """Raised when embedding vectors have invalid dimensionality."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Value_Embedding_Vector_Dimensionality"

    def __init__(self, dim_1_cols: list[str], high_dim_cols: list[str]) -> None:
        """Initialize the exception with embedding dimensionality context.

        Args:
            dim_1_cols: Columns with dimensionality of 1.
            high_dim_cols: Columns with dimensionality exceeding the maximum.
        """
        self.dim_1_cols = dim_1_cols
        self.high_dim_cols = high_dim_cols

    def error_message(self) -> str:
        """Return the error message for this exception."""
        msg = (
            "Embedding vectors cannot have length (dimensionality) of 1 or higher "
            f"than {MAX_EMBEDDING_DIMENSIONALITY}. "
        )
        if self.dim_1_cols:
            msg += f"The following columns have dimensionality of 1: {','.join(self.dim_1_cols)}. "
        if self.high_dim_cols:
            msg += (
                f"The following columns have dimensionality greater than {MAX_EMBEDDING_DIMENSIONALITY}: "
                f"{','.join(self.high_dim_cols)}. "
            )

        return msg


class InvalidValueEmbeddingRawDataTooLong(ValidationError):
    """Raised when embedding raw data exceeds maximum character limit."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Value_Embedding_Raw_Data_Too_Long"

    def __init__(self, cols: Iterable) -> None:
        """Initialize the exception with raw data length validation context.

        Args:
            cols: Columns with embedding raw data exceeding maximum characters.
        """
        self.invalid_cols = cols

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"Embedding raw data cannot have more than {MAX_RAW_DATA_CHARACTERS} characters. "
            "The following columns do not satisfy this condition: "
            f"{', '.join(map(str, self.invalid_cols))}."
        )


class InvalidTypeShapValues(ValidationError):
    """Raised when SHAP value columns have invalid types."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Type_SHAP_Values"

    def __init__(self, cols: Iterable, expected_types: list[str]) -> None:
        """Initialize the exception with SHAP value type validation context.

        Args:
            cols: SHAP value columns with unrecognized data types.
            expected_types: List of expected data types for SHAP values.
        """
        self.wrong_type_columns = cols
        self.expected_types = expected_types

    def error_message(self) -> str:
        """Return the error message for this exception."""
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(map(str, self.expected_types[:-1]))} or {self.expected_types[-1]}"
        )
        return (
            f"SHAP values must be of type {type_list}. "
            "The following SHAP columns have unrecognized data types: "
            f"{', '.join(map(str, self.wrong_type_columns))}."
        )
