from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List

from arize.utils.types import Environments, Metrics, ModelTypes


class ValidationError(Exception, ABC):
    def __str__(self) -> str:
        return self.error_message()

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def error_message(self) -> str:
        pass


class ValidationFailure(Exception):
    def __init__(self, errors: List[ValidationError]) -> None:
        self.errors = errors


# ----------------------
# Minimum required checks
# ----------------------
class InvalidFieldTypeConversion(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Input_Type_Conversion"

    def __init__(self, fields: Iterable, type: str) -> None:
        self.fields = fields
        self.type = type

    def error_message(self) -> str:
        return (
            f"The following fields must be convertible to {self.type}: "
            f"{', '.join(map(str, self.fields))}."
        )


class InvalidFieldTypeEmbeddingFeatures(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Input_Type_Embedding_Features"

    def __init__(self) -> None:
        pass

    def error_message(self) -> str:
        return (
            "schema.embedding_feature_column_names should be a dictionary mapping strings "
            "to EmbeddingColumnNames objects"
        )


class InvalidFieldTypePromptResponse(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Input_Type_Prompt_Response"

    def __init__(self, name: str) -> None:
        self.name = name

    def error_message(self) -> str:
        return f"{self.name} must be of type EmbeddingColumnNames"


class InvalidIndex(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Index"

    def error_message(self) -> str:
        return (
            "The index of the dataframe is invalid; "
            "reset the index by using df.reset_index(drop=True, inplace=True)"
        )


# ----------------
# Parameter checks
# ----------------


class MissingColumns(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Columns"

    def __init__(self, cols: Iterable) -> None:
        self.missing_cols = cols

    def error_message(self) -> str:
        return (
            "The following columns are declared in the schema "
            "but are not found in the dataframe: "
            f"{', '.join(map(str, self.missing_cols))}."
        )


class MissingRequiredColumnsMetricsValidation(ValidationError):
    """
    This error is used only for model mapping validations.
    """

    def __repr__(self) -> str:
        return "Missing_Columns_Required_By_Metrics_Validation"

    def __init__(self, model_type: ModelTypes, metrics: List[Metrics], cols: Iterable) -> None:
        self.model_type = model_type
        self.metrics = metrics
        self.missing_cols = cols

    def error_message(self) -> str:
        return (
            f"For logging data for a {self.model_type.name} model with support for metrics "
            f"{', '.join(m.name for m in self.metrics)}, "
            f"schema must include: {', '.join(map(str, self.missing_cols))}."
        )


class InvalidModelTypeAndMetricsCombination(ValidationError):
    """
    This error is used only for model mapping validations.
    """

    def __repr__(self) -> str:
        return "Invalid_ModelType_And_Metrics_Combination"

    def __init__(
        self,
        model_type: ModelTypes,
        metrics: List[Metrics],
        suggested_model_metric_combinations: List[List[str]],
    ) -> None:
        self.model_type = model_type
        self.metrics = metrics
        self.suggested_combinations = suggested_model_metric_combinations

    def error_message(self) -> str:
        valid_combos = ", or \n".join(
            "[" + ", ".join(combo) + "]" for combo in self.suggested_combinations
        )
        return (
            f"Invalid combination of model type {self.model_type.name} and metrics: "
            f"{', '.join(m.name for m in self.metrics)}. "
            f"Valid Metric combinations for this model type:\n{valid_combos}.\n"
        )


class InvalidShapSuffix(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_SHAP_Suffix"

    def __init__(self, cols: Iterable) -> None:
        self.invalid_column_names = cols

    def error_message(self) -> str:
        return (
            "The following features or tags must not be named with a `_shap` suffix: "
            f"{', '.join(map(str, self.invalid_column_names))}."
        )


class InvalidModelType(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Model_Type"

    def error_message(self) -> str:
        return (
            "Model type not valid. Choose one of the following: "
            f"{', '.join('ModelTypes.' + mt.name for mt in ModelTypes)}. "
        )


class InvalidEnvironment(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Environment"

    def error_message(self) -> str:
        return (
            "Environment not valid. Choose one of the following: "
            f"{', '.join('Environments.' + env.name for env in Environments)}. "
        )


class InvalidBatchId(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Batch_ID"

    def error_message(self) -> str:
        return "Batch ID must be a nonempty string if logging to validation environment."


class InvalidModelVersion(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Model_Version"

    def error_message(self) -> str:
        return "Model version must be a nonempty string."


class InvalidModelId(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Model_ID"

    def error_message(self) -> str:
        return "Model ID must be a nonempty string."


class MissingPredActShap(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Pred_or_Act_or_SHAP"

    def error_message(self) -> str:
        return (
            "The schema must specify at least one of the following: "
            "prediction label, actual label, or SHAP value column names"
        )


class MissingPromptResponseGenerativeLLM(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Prompt_Response_Generative_LLM"

    def error_message(self) -> str:
        return (
            "The schema must specify prompt_column_names and response_column_names for "
            "ModelTypes.GENERATIVE_LLM models"
        )


class MissingPredActShapNumeric(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Pred_or_Act_or_SHAP_Numeric"

    def error_message(self) -> str:
        return (
            "For a numeric model, the schema must specify at least one of the following: "
            "prediction label, prediction score, actual label, actual score, or SHAP value column "
            "names"
        )


class MissingPredLabelScoreCategorical(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Pred_Label_Score_Categorical"

    def error_message(self) -> str:
        return (
            "When sending a prediction confidence score and an actual label, the schema must also include "
            "a prediction label."
        )


class MissingPreprodPredAct(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Preproduction_Pred_and_Act"

    def error_message(self) -> str:
        return "For logging pre-production data, the schema must specify both "
        "prediction and actual label columns."


class MissingPreprodAct(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Preproduction_Act"

    def error_message(self) -> str:
        return "For logging pre-production data, the schema must specify actual label column."


class MissingPreprodPredActNumeric(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Preproduction_Pred_and_Act_Numeric"

    def error_message(self) -> str:
        return (
            "For logging pre-production data for a numeric model, "
            "the schema must specify both prediction and actual label or confidence score columns."
        )


class MissingRequiredColumnsForRankingModel(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Required_Columns_For_Ranking_Model"

    def error_message(self) -> str:
        return (
            "For logging data for a ranking model, schema must specify: "
            "prediction_group_id_column_name and rank_column_name"
        )


class MissingObjectDetectionPredAct(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Object_Detection_Prediction_or_Actual"

    def __init__(self, environment: Environments):
        self.environment = environment

    def error_message(self) -> str:
        if self.environment in (Environments.TRAINING, Environments.VALIDATION):
            env = "pre-production"
            opt = "and"
        elif self.environment == Environments.PRODUCTION:
            env = "production"
            opt = "or"
        else:
            raise TypeError("Invalid environment")
        return (
            f"For logging {env} data for an Object Detection model,"
            "the schema must specify 'object_detection_prediction_column_names'"
            f"{opt} 'object_detection_actual_column_names'"
        )


class InvalidPredActColumnNamesForObjectDetectionModelType(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Prediction_or_Actual_Column_Names_for_Object_Detection_Model_Type"

    def __init__(
        self,
        wrong_cols: List[str],
    ) -> None:
        self.wrong_cols = wrong_cols

    def error_message(self) -> str:
        return (
            "Only 'object_detection_prediction_column_names' and "
            "'object_detection_actual_column_names' are allowed for ModelTypes.OBJECT_DETECTION "
            "in order to send predictions and actuals. The following column names "
            f"were declared in the schema and are not allowed: {', '.join(self.wrong_cols)}"
        )


class InvalidPredActObjectDetectionColumnNamesForModelType(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Object_Detection_Prediction_or_Actual_Column_Names_for_Model_Type"

    def __init__(
        self,
        invalid_model_type: ModelTypes,
    ) -> None:
        self.invalid_model_type = invalid_model_type

    def error_message(self) -> str:
        return (
            f"Cannot use 'object_detection_prediction_column_names' or "
            f"'object_detection_actual_column_names' for {self.invalid_model_type} model "
            f"type. They are only allowed for ModelTypes.OBJECT_DETECTION models"
        )


# -----------
# Type checks
# -----------


class InvalidType(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type"

    def __init__(self, name: str, expected_types: List[str]) -> None:
        self.name = name
        self.expected_types = expected_types

    def error_message(self) -> str:
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(map(str, self.expected_types[:-1]))} or {self.expected_types[-1]}"
        )
        return f"{self.name} must be of type {type_list}."


class InvalidTypeColumns(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_Columns"

    def __init__(self, wrong_type_columns: List[str], expected_types: List[str]) -> None:
        self.wrong_type_columns = wrong_type_columns
        self.expected_types = expected_types

    def error_message(self) -> str:
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(map(str, self.expected_types[:-1]))} or {self.expected_types[-1]}"
        )
        return (
            f"The columns {', '.join(self.wrong_type_columns[:-1])}, and {self.wrong_type_columns[-1]}; "
            f"must be of type {type_list}."
        )


class InvalidTypeFeatures(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_Features"

    def __init__(self, cols: Iterable, expected_types: List[str]) -> None:
        self.wrong_type_columns = cols
        self.expected_types = expected_types

    def error_message(self) -> str:
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


class InvalidTypePromptResponse(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_Prompt_Response_Column_Names"

    def __init__(self, wrong_type_columns: List[str], expected_types: List[str]) -> None:
        self.wrong_type_columns = wrong_type_columns
        self.expected_types = expected_types

    def error_message(self) -> str:
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(map(str, self.expected_types[:-1]))} or {self.expected_types[-1]}"
        )
        return (
            f"prompt_column_names and response_column_names must be of type {type_list}. "
            "The following columns have unrecognized data types: "
            f"{', '.join(map(str, self.wrong_type_columns))}."
        )


class InvalidTypeTags(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_Tags"

    def __init__(self, cols: Iterable, expected_types: List[str]) -> None:
        self.wrong_type_columns = cols
        self.expected_types = expected_types

    def error_message(self) -> str:
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


class InvalidValueLowEmbeddingVectorDimensionality(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Value_Low_Embedding_Vector_Dimensionality"

    def __init__(self, cols: Iterable) -> None:
        self.invalid_cols = cols

    def error_message(self) -> str:
        return (
            "Embedding vectors cannot have length (dimensionality) == 1. "
            "The following columns do not satisfy this condition: "
            f"{', '.join(map(str, self.invalid_cols))}."
        )


class InvalidTypeShapValues(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_SHAP_Values"

    def __init__(self, cols: Iterable, expected_types: List[str]) -> None:
        self.wrong_type_columns = cols
        self.expected_types = expected_types

    def error_message(self) -> str:
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


# -----------
# Value checks
# -----------


class InvalidValueTimestamp(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Timestamp_Value"

    def __init__(self, name: str, acceptable_range: str) -> None:
        self.name = name
        self.acceptable_range = acceptable_range

    def error_message(self) -> str:
        return (
            f"{self.name} is out of range. "
            f"Only values within {self.acceptable_range} from the current time are accepted. "
            "If this is your pre-production data, you could also just remove the timestamp column "
            "from the Schema."
        )


class InvalidValueMissingValue(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Missing_Value"

    def __init__(self, name: str, wrong_values: str) -> None:
        self.name = name
        self.wrong_values = wrong_values

    def error_message(self) -> str:
        return f"{self.name} must not contain {self.wrong_values} values."


class InvalidRankValue(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Rank_Value"

    def __init__(self, name: str, acceptable_range: str) -> None:
        self.name = name
        self.acceptable_range = acceptable_range

    def error_message(self) -> str:
        return (
            f"ranking column {self.name} is out of range. "
            f"Only values within {self.acceptable_range}  are accepted. "
        )


class InvalidStringLength(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_String_Length"

    def __init__(self, schema_name: str, col_name: str, min_length: int, max_length: int) -> None:
        self.schema_name = schema_name
        self.col_name = col_name
        self.min_length = min_length
        self.max_length = max_length

    def error_message(self) -> str:
        return (
            f"{self.schema_name} column {self.col_name} contains invalid values. "
            f"Only string values of length within {self.min_length} - {self.max_length} are accepted."
        )


class InvalidRankingCategoryValue(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Ranking_Relevance_Labels_Value"

    def __init__(self, name: str) -> None:
        self.name = name

    def error_message(self) -> str:
        return (
            f"ranking relevance labels {self.name} column contains invalid value"
            f"make sure empty string is not present"
        )


class InvalidBoundingBoxesCoordinates(ValidationError, Exception):
    def __repr__(self) -> str:
        return "Invalid_Bounding_Boxes_Coordinates"

    def __init__(self, reason) -> None:
        self._check_valid_reason(reason)
        self.reason = reason

    @staticmethod
    def _check_valid_reason(reason):
        possible_reasons = (
            "none_boxes",
            "none_or_empty_box",
            "boxes_coordinates_wrong_format",
        )
        if reason not in possible_reasons:
            raise ValueError(
                f"Invalid reason {reason}. Possible reasons are: " f"{', '.join(possible_reasons)}."
            )

    def error_message(self) -> str:
        msg = "Invalid bounding boxes coordinates found. "
        if self.reason == "none_boxes":
            msg += (
                "Found at least one list of bounding boxes coordinates with NoneType. List of "
                "bounding boxes coordinates cannot be None, if you'd like to send no boxes, "
                "send an empty list"
            )
        elif self.reason == "none_or_empty_box":
            msg += (
                "Found at least one bounding box with None value or without coordinates. All "
                "bounding boxes in the list must contain its 4 coordinates"
            )
        elif self.reason == "boxes_coordinates_wrong_format":
            msg += (
                "Found at least one bound box's coordinates incorrectly formatted. Each "
                "bounding box's coordinates must be a collection of 4 positive floats "
                "representing the top-left & bottom-right corners of the box, in pixels"
            )
        return msg


class InvalidBoundingBoxesCategories(ValidationError, Exception):
    def __repr__(self) -> str:
        return "Invalid_Bounding_Boxes_Categories"

    def __init__(self, reason) -> None:
        self._check_valid_reason(reason)
        self.reason = reason

    @staticmethod
    def _check_valid_reason(reason):
        possible_reasons = (
            "none_category_list",
            "none_category",
        )
        if reason not in possible_reasons:
            raise ValueError(
                f"Invalid reason {reason}. Possible reasons are: " f"{', '.join(possible_reasons)}."
            )

    def error_message(self) -> str:
        msg = "Invalid bounding boxes categories found. "
        if self.reason == "none_category_list":
            msg += (
                "Found at least one list of bounding box categories with None value. Must send a "
                "list of categories, one category per bounding box."
            )
        elif self.reason == "none_category":
            msg += (
                "Found at least one category label with None value. Each bounding box category "
                "must be string. Empty strings are allowed"
            )
        return msg


class InvalidBoundingBoxesScores(ValidationError, Exception):
    def __repr__(self) -> str:
        return "Invalid_Bounding_Boxes_Scores"

    def __init__(self, reason) -> None:
        self._check_valid_reason(reason)
        self.reason = reason

    @staticmethod
    def _check_valid_reason(reason):
        possible_reasons = (
            "none_score_list",
            "scores_out_of_bounds",
        )
        if reason not in possible_reasons:
            raise ValueError(
                f"Invalid reason {reason}. Possible reasons are: " f"{', '.join(possible_reasons)}."
            )

    def error_message(self) -> str:
        msg = "Invalid bounding boxes scores found. "
        if self.reason == "none_score_list":
            msg += (
                "Found at least one list of bounding box scores with None value. This field is "
                "optional. If sent, you must send a confidence score per bounding box"
            )
        elif self.reason == "scores_out_of_bounds":
            msg += (
                "Found at least one confidence score out of bounds. "
                "Confidence scores must be between 0 and 1"
            )
        return msg
