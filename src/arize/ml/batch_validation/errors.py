"""Batch validation error classes for ML model data.

This module contains exceptions that are specific to batch validation logic.
For common validation exceptions, use:
- arize.exceptions.base (ValidationError, InvalidDataFrameIndex, etc.)
- arize.exceptions.types (InvalidType, InvalidTypeColumns, etc.)
- arize.exceptions.values (InvalidValueTimestamp, InvalidStringLengthInColumn, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arize.constants.ml import MAX_NUMBER_OF_EMBEDDINGS
from arize.exceptions.base import ValidationError
from arize.logging import log_a_list
from arize.ml.types import Environments, ModelTypes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from arize.ml.types import Metrics


# ----------------------
# Minimum required checks
# ----------------------
class InvalidColumnNameEmptyString(ValidationError):
    """Raised when a schema contains an empty string as a column name."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Column_Name_Empty_String"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "Empty column name found: ''. The schema cannot point to columns in the "
            "dataframe denoted by an empty string. You can see the columns used in the "
            "schema by running schema.get_used_columns()"
        )


class InvalidFieldTypeEmbeddingFeatures(ValidationError):
    """Raised when embedding feature column names are not properly formatted."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Input_Type_Embedding_Features"

    def __init__(self) -> None:
        """Initialize the exception."""

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "schema.embedding_feature_column_names should be a dictionary mapping strings "
            "to EmbeddingColumnNames objects"
        )


class InvalidFieldTypePromptResponse(ValidationError):
    """Raised when prompt response field is not of correct type."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Input_Type_Prompt_Response"

    def __init__(self, name: str) -> None:
        """Initialize the exception with field name context.

        Args:
            name: Name of the field with invalid prompt response type.
        """
        self.name = name

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"'{self.name}' must be of type str or EmbeddingColumnNames"


class InvalidSchemaType(ValidationError):
    """Raised when schema type is incompatible with the model environment."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Schema_Type"

    def __init__(self, schema_type: str, environment: Environments) -> None:
        """Initialize the exception with schema type and environment context.

        Args:
            schema_type: Type of schema that is invalid.
            environment: Model environment where schema is being used.
        """
        self.schema_type = schema_type
        self.environment = environment

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"Cannot use a {self.schema_type} for a model with environment: {self.environment}"


# ----------------
# Parameter checks
# ----------------


class MissingPredictionIdColumnForDelayedRecords(ValidationError):
    """Raised when prediction ID is missing for delayed actuals or feature importance."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Prediction_Id_Column_For_Delayed_Records"

    def __init__(
        self, has_actual_info: bool, has_feature_importance_info: bool
    ) -> None:
        """Initialize the exception with delayed record context.

        Args:
            has_actual_info: Whether actual information is present.
            has_feature_importance_info: Whether feature importance information is present.
        """
        self.has_actual_info = has_actual_info
        self.has_feature_importance_info = has_feature_importance_info

    def error_message(self) -> str:
        """Return the error message for this exception."""
        actual = "actual" if self.has_actual_info else ""
        feat_imp = (
            "feature importance" if self.has_feature_importance_info else ""
        )
        if self.has_actual_info and self.has_feature_importance_info:
            msg = " and ".join([actual, feat_imp])
        else:
            msg = "".join([actual, feat_imp])

        return (
            "Missing 'prediction_id_column_name'. While prediction id is optional for most cases, "
            "it is required when sending delayed actuals, i.e. when sending actual or feature importances "
            f"without predictions. In this case, {msg} information was found (without predictions). "
            "To learn more about delayed joins, please see the docs at "
            "https://docs.arize.com/arize/sending-data-guides/how-to-send-delayed-actuals"
        )


class MissingColumns(ValidationError):
    """Raised when columns declared in schema are not found in dataframe."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Columns"

    def __init__(self, cols: Iterable) -> None:
        """Initialize the exception with missing columns context.

        Args:
            cols: Columns declared in schema but not found in dataframe.
        """
        self.missing_cols = set(cols)

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "The following columns are declared in the schema "
            "but are not found in the dataframe: "
            f"{', '.join(map(str, self.missing_cols))}."
        )


class MissingRequiredColumnsMetricsValidation(ValidationError):
    """This error is used only for model mapping validations."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Columns_Required_By_Metrics_Validation"

    def __init__(
        self, model_type: ModelTypes, metrics: list[Metrics], cols: Iterable
    ) -> None:
        """Initialize the exception with model metrics validation context.

        Args:
            model_type: Type of model being validated.
            metrics: List of metrics requiring validation.
            cols: Required columns that are missing.
        """
        self.model_type = model_type
        self.metrics = metrics
        self.missing_cols = cols

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"For logging data for a {self.model_type.name} model with support for metrics "
            f"{', '.join(m.name for m in self.metrics)}, "
            f"schema must include: {', '.join(map(str, self.missing_cols))}."
        )


class ReservedColumns(ValidationError):
    """Raised when reserved column names are used in schema fields."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Reserved_Columns"

    def __init__(self, cols: Iterable) -> None:
        """Initialize the exception with reserved columns context.

        Args:
            cols: Reserved columns that cannot be used in schema fields.
        """
        self.reserved_columns = cols

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "The following columns are reserved and can only be specified "
            "in the proper fields of the schema: "
            f"{', '.join(map(str, self.reserved_columns))}."
        )


class InvalidModelTypeAndMetricsCombination(ValidationError):
    """This error is used only for model mapping validations."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_ModelType_And_Metrics_Combination"

    def __init__(
        self,
        model_type: ModelTypes,
        metrics: list[Metrics],
        suggested_model_metric_combinations: list[list[str]],
    ) -> None:
        """Initialize the exception with model type and metrics combination context.

        Args:
            model_type: Type of model being validated.
            metrics: List of metrics that form invalid combination with model type.
            suggested_model_metric_combinations: Valid metric combinations for the model type.
        """
        self.model_type = model_type
        self.metrics = metrics
        self.suggested_combinations = suggested_model_metric_combinations

    def error_message(self) -> str:
        """Return the error message for this exception."""
        valid_combos = ", or \n".join(
            "[" + ", ".join(combo) + "]"
            for combo in self.suggested_combinations
        )
        return (
            f"Invalid combination of model type {self.model_type.name} and metrics: "
            f"{', '.join(m.name for m in self.metrics)}. "
            f"Valid Metric combinations for this model type:\n{valid_combos}.\n"
        )


class InvalidShapSuffix(ValidationError):
    """Raised when feature or tag names use the reserved '_shap' suffix."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_SHAP_Suffix"

    def __init__(self, cols: Iterable) -> None:
        """Initialize the exception with invalid SHAP suffix columns.

        Args:
            cols: Feature or tag columns using the reserved '_shap' suffix.
        """
        self.invalid_column_names = cols

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "The following features or tags must not be named with a `_shap` suffix: "
            f"{', '.join(map(str, self.invalid_column_names))}."
        )


class InvalidModelType(ValidationError):
    """Raised when an invalid model type is specified."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Model_Type"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "Model type not valid. Choose one of the following: "
            f"{', '.join('ModelTypes.' + mt.name for mt in ModelTypes)}. "
        )


class InvalidEnvironment(ValidationError):
    """Raised when an invalid environment is specified."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Environment"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "Environment not valid. Choose one of the following: "
            f"{', '.join('Environments.' + env.name for env in Environments)}. "
        )


class InvalidBatchId(ValidationError):
    """Raised when batch ID is missing or invalid for validation environment."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Batch_ID"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "Batch ID must be a nonempty string if logging to validation environment."


class InvalidModelVersion(ValidationError):
    """Raised when model version is empty or invalid."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Model_Version"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "Model version must be a nonempty string."


class InvalidModelId(ValidationError):
    """Raised when model ID is empty or invalid."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Model_ID"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "Model ID must be a nonempty string."


class InvalidProjectName(ValidationError):
    """Raised when project name is empty or invalid."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Project_Name"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "Project Name must be a nonempty string. "
            "If Model ID was used instead of Project Name, "
            "it must be a nonempty string."
        )


class MissingPredActShap(ValidationError):
    """Raised when schema is missing prediction, actual, or SHAP values."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Pred_or_Act_or_SHAP"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "The schema must specify at least one of the following: "
            "prediction label, actual label, or SHAP value column names"
        )


class MissingPreprodPredAct(ValidationError):
    """Raised when pre-production data is missing both prediction and actual labels."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Preproduction_Pred_and_Act"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "For logging pre-production data, the schema must specify both "
            "prediction and actual label columns."
        )


class MissingPreprodAct(ValidationError):
    """Raised when pre-production data is missing actual label column."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Preproduction_Act"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "For logging pre-production data, the schema must specify actual label column."


class MissingPreprodPredActNumericAndCategorical(ValidationError):
    """Raised when pre-production numeric/categorical model is missing prediction or actual columns."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Preproduction_Pred_and_Act_Numeric_and_Categorical"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "For logging pre-production data for a numeric or a categorical model, "
            "the schema must specify both prediction and actual label or score columns."
        )


class MissingRequiredColumnsForRankingModel(ValidationError):
    """Raised when ranking model is missing required group ID or rank columns."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Required_Columns_For_Ranking_Model"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "For logging data for a ranking model, schema must specify: "
            "prediction_group_id_column_name and rank_column_name"
        )


class MissingCVPredAct(ValidationError):
    """Raised when computer vision model is missing prediction or actual columns."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_CV_Prediction_or_Actual"

    def __init__(self, environment: Environments) -> None:
        """Initialize the exception with environment context.

        Args:
            environment: Model environment (training, validation, or production).
        """
        self.environment = environment

    def error_message(self) -> str:
        """Return the error message for this exception."""
        if self.environment in (Environments.TRAINING, Environments.VALIDATION):
            env = "pre-production"
            opt = "and"
        elif self.environment == Environments.PRODUCTION:
            env = "production"
            opt = "or"
        else:
            raise TypeError("Invalid environment")
        return (
            f"For logging {env} data for an Object Detection model, "
            "the schema must specify one of: "
            f"('object_detection_prediction_column_names' {opt} "
            f"'object_detection_actual_column_names') "
            f"or ('semantic_segmentation_prediction_column_names' {opt} "
            f"'semantic_segmentation_actual_column_names') "
            f"or ('instance_segmentation_prediction_column_names' {opt} "
            f"'instance_segmentation_actual_column_names')"
        )


class MultipleCVPredAct(ValidationError):
    """Raised when multiple computer vision prediction/actual types are specified."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Multiple_CV_Prediction_or_Actual"

    def __init__(self, environment: Environments) -> None:
        """Initialize the exception with environment context.

        Args:
            environment: Model environment where multiple CV types were specified.
        """
        self.environment = environment

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "The schema must only specify one of the following: "
            "'object_detection_prediction_column_names'/'object_detection_actual_column_names', "
            "'semantic_segmentation_prediction_column_names'/'semantic_segmentation_actual_column_names', "
            "'instance_segmentation_prediction_column_names'/'instance_segmentation_actual_column_names'."
        )


class InvalidPredActCVColumnNamesForModelType(ValidationError):
    """Raised when CV columns are used for non-OBJECT_DETECTION model types."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_CV_Prediction_or_Actual_Column_Names_for_Model_Type"

    def __init__(
        self,
        invalid_model_type: ModelTypes,
    ) -> None:
        """Initialize the exception with model type context.

        Args:
            invalid_model_type: Model type that cannot use CV columns.
        """
        self.invalid_model_type = invalid_model_type

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"Cannot use 'object_detection_prediction_column_names' or "
            f"'object_detection_actual_column_names' or "
            f"'semantic_segmentation_prediction_column_names' or "
            f"'semantic_segmentation_actual_column_names' or "
            f"'instance_segmentation_prediction_column_names' or "
            f"'instance_segmentation_actual_column_names' for {self.invalid_model_type} model "
            f"type. They are only allowed for ModelTypes.OBJECT_DETECTION models"
        )


class MissingReqPredActColumnNamesForMultiClass(ValidationError):
    """Raised when multi-class model is missing required score columns."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Missing_Required_Prediction_or_Actual_Column_Names_for_Multi_Class_Model_Type"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "For logging data for a multi class model, schema must specify: "
            "prediction_scores_column_name and/or actual_score_column_name. "
            "Optionally, you may include multi_class_threshold_scores_column_name "
            "(must include prediction_scores_column_name)"
        )


class InvalidPredActColumnNamesForModelType(ValidationError):
    """Raised when prediction/actual columns are invalid for the model type."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Prediction_or_Actual_Column_Names_for_Model_Type"

    def __init__(
        self,
        invalid_model_type: ModelTypes,
        allowed_fields: list[str],
        wrong_columns: list[str],
    ) -> None:
        """Initialize the exception with model type and column validation context.

        Args:
            invalid_model_type: Model type with invalid columns.
            allowed_fields: List of allowed schema fields for the model type.
            wrong_columns: Columns that are invalid for the model type.
        """
        self.invalid_model_type = invalid_model_type
        self.allowed_fields = allowed_fields
        self.wrong_columns = wrong_columns

    def error_message(self) -> str:
        """Return the error message for this exception."""
        allowed_col_msg = ""
        if self.allowed_fields is not None:
            allowed_col_msg = f" Allowed Schema fields are {log_a_list(self.allowed_fields, 'and')}"
        return (
            f"Invalid Schema fields for {self.invalid_model_type} model type. {allowed_col_msg}. "
            "The following columns of your dataframe are sent as an invalid schema field: "
            f"{log_a_list(self.wrong_columns, 'and')}"
        )


class DuplicateColumnsInDataframe(ValidationError):
    """Raised when dataframe contains duplicate column names used in schema."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Duplicate_Columns_In_Dataframe"

    def __init__(self, cols: Iterable) -> None:
        """Initialize the exception with duplicate columns context.

        Args:
            cols: Columns that have duplicates in the dataframe.
        """
        self.duplicate_cols = cols

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "The following columns are present in the schema and have duplicates in the dataframe: "
            f"{self.duplicate_cols}. "
        )


class InvalidNumberOfEmbeddings(ValidationError):
    """Raised when the number of embeddings exceeds the maximum allowed."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Number_Of_Embeddings"

    def __init__(self, number_of_embeddings: int) -> None:
        """Initialize the exception with embedding count context.

        Args:
            number_of_embeddings: Number of embeddings found in the schema.
        """
        self.number_of_embeddings = number_of_embeddings

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The schema contains {self.number_of_embeddings} different embeddings when a maximum of "
            f"{MAX_NUMBER_OF_EMBEDDINGS} is allowed."
        )
