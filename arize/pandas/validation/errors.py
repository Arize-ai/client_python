from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Dict, List, Optional

from arize.utils.constants import (
    MAX_EMBEDDING_DIMENSIONALITY,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_MULTI_CLASS_NAME_LENGTH,
    MAX_NUMBER_OF_EMBEDDINGS,
    MAX_NUMBER_OF_MULTI_CLASS_CLASSES,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
    MAX_RAW_DATA_CHARACTERS,
    MAX_TAG_LENGTH,
)
from arize.utils.logging import log_a_list
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
class InvalidColumnNameEmptyString(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Column_Name_Empty_String"

    @staticmethod
    def error_message() -> str:
        return (
            "Empty column name found: ''. The schema cannot point to columns in the "
            "dataframe denoted by an empty string. You can see the columns used in the "
            "schema by running schema.get_used_columns()"
        )


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
        return f"'{self.name}' must be of type str or EmbeddingColumnNames"


class InvalidDataFrameIndex(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Index"

    def error_message(self) -> str:
        return (
            "The index of the dataframe is invalid; "
            "reset the index by using df.reset_index(drop=True, inplace=True)"
        )


class InvalidSchemaType(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Schema_Type"

    def __init__(self, schema_type: str, environment: Environments) -> None:
        self.schema_type = schema_type
        self.environment = environment

    def error_message(self) -> str:
        return f"Cannot use a {self.schema_type} for a model with environment: {self.environment}"


# ----------------
# Parameter checks
# ----------------


class MissingPredictionIdColumnForDelayedRecords(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Prediction_Id_Column_For_Delayed_Records"

    def __init__(self, has_actual_info, has_feature_importance_info) -> None:
        self.has_actual_info = has_actual_info
        self.has_feature_importance_info = has_feature_importance_info

    def error_message(self) -> str:
        actual = "actual" if self.has_actual_info else ""
        feat_imp = "feature importance" if self.has_feature_importance_info else ""
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
    def __repr__(self) -> str:
        return "Missing_Columns"

    def __init__(self, cols: Iterable) -> None:
        self.missing_cols = set(cols)

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


class ReservedColumns(ValidationError):
    def __repr__(self) -> str:
        return "Reserved_Columns"

    def __init__(self, cols: Iterable) -> None:
        self.reserved_columns = cols

    def error_message(self) -> str:
        return (
            "The following columns are reserved and can only be specified "
            "in the proper fields of the schema: "
            f"{', '.join(map(str, self.reserved_columns))}."
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


class InvalidProjectName(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Project_Name"

    def error_message(self) -> str:
        return (
            "Project Name must be a nonempty string. "
            "If Model ID was used instead of Project Name, "
            "it must be a nonempty string."
        )


class MissingPredActShap(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Pred_or_Act_or_SHAP"

    def error_message(self) -> str:
        return (
            "The schema must specify at least one of the following: "
            "prediction label, actual label, or SHAP value column names"
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


class MissingPreprodPredActNumericAndCategorical(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Preproduction_Pred_and_Act_Numeric_and_Categorical"

    def error_message(self) -> str:
        return (
            "For logging pre-production data for a numeric or a categorical model, "
            "the schema must specify both prediction and actual label or score columns."
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


class MissingReqPredActColumnNamesForMultiClass(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Required_Prediction_or_Actual_Column_Names_for_Multi_Class_Model_Type"

    def error_message(self) -> str:
        return (
            "For logging data for a multi class model, schema must specify: "
            "prediction_scores_column_name and/or actual_score_column_name. "
            "Optionally, you may include multi_class_threshold_scores_column_name"
            " (must include prediction_scores_column_name)"
        )


class InvalidPredActColumnNamesForModelType(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Prediction_or_Actual_Column_Names_for_Model_Type"

    def __init__(
        self,
        invalid_model_type: ModelTypes,
        allowed_fields: List[str],
        wrong_columns: List[str],
    ) -> None:
        self.invalid_model_type = invalid_model_type
        self.allowed_fields = allowed_fields
        self.wrong_columns = wrong_columns

    def error_message(self) -> str:
        allowed_col_msg = ""
        if self.allowed_fields is not None:
            allowed_col_msg = f" Allowed Schema fields are {log_a_list(self.allowed_fields, 'and')}"
        return (
            f"Invalid Schema fields for {self.invalid_model_type} model type. {allowed_col_msg}"
            "The following columns of your dataframe are sent as an invalid schema field: "
            f"{log_a_list(self.wrong_columns, 'and')}"
        )


class DuplicateColumnsInDataframe(ValidationError):
    def __repr__(self) -> str:
        return "Duplicate_Columns_In_Dataframe"

    def __init__(self, cols: Iterable) -> None:
        self.duplicate_cols = cols

    def error_message(self) -> str:
        return (
            "The following columns are present in the schema and have duplicates in the dataframe: "
            f"{self.duplicate_cols}. "
        )


class InvalidNumberOfEmbeddings(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Number_Of_Embeddings"

    def __init__(self, number_of_embeddings: int) -> None:
        self.number_of_embeddings = number_of_embeddings

    def error_message(self) -> str:
        return (
            f"The schema contains {self.number_of_embeddings} different embeddings when a maximum of "
            f"{MAX_NUMBER_OF_EMBEDDINGS} is allowed."
        )


# -----------
# Type checks
# -----------


class InvalidType(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type"

    def __init__(self, name: str, expected_types: List[str], found_data_type: str) -> None:
        self.name = name
        self.expected_types = expected_types
        self.found_data_type = found_data_type

    def error_message(self) -> str:
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
    def __repr__(self) -> str:
        return "Invalid_Type_Columns"

    def __init__(self, wrong_type_columns: List[str], expected_types: List[str]) -> None:
        self.wrong_type_columns = wrong_type_columns
        self.expected_types = expected_types

    def error_message(self) -> str:
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


class InvalidFieldTypePromptTemplates(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Input_Type_Prompt_Templates"

    def error_message(self) -> str:
        return "prompt_template_column_names must be of type PromptTemplateColumnNames"


class InvalidFieldTypeLlmConfig(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Input_Type_LLM_Config"

    def error_message(self) -> str:
        return "llm_config_column_names must be of type LLMConfigColumnNames"


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


class InvalidValueEmbeddingVectorDimensionality(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Value_Embedding_Vector_Dimensionality"

    def __init__(self, dim_1_cols: List[str], high_dim_cols: List[str]) -> None:
        self.dim_1_cols = dim_1_cols
        self.high_dim_cols = high_dim_cols

    def error_message(self) -> str:
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
    def __repr__(self) -> str:
        return "Invalid_Value_Embedding_Raw_Data_Too_Long"

    def __init__(self, cols: Iterable) -> None:
        self.invalid_cols = cols

    def error_message(self) -> str:
        return (
            f"Embedding raw data cannot have more than {MAX_RAW_DATA_CHARACTERS} characters. "
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

    def __init__(self, timestamp_col_name: str) -> None:
        self.timestamp_col_name = timestamp_col_name

    def error_message(self) -> str:
        return (
            f"Prediction timestamp in {self.timestamp_col_name} is out of range. "
            f"Prediction timestamps must be within {MAX_FUTURE_YEARS_FROM_CURRENT_TIME} year "
            f"in the future and {MAX_PAST_YEARS_FROM_CURRENT_TIME} years in the past from "
            "the current time. If this is your pre-production data, you could also just "
            "remove the timestamp column from the Schema."
        )


class InvalidValueMissingValue(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Missing_Value"

    def __init__(self, name: str, wrong_values: str, column: Optional[str] = None) -> None:
        self.name = name
        self.wrong_values = wrong_values
        self.column = column

    def error_message(self) -> str:
        if self.name in ["Prediction ID", "Prediction Group ID", "Rank"]:
            return (
                f"{self.name} column '{self.column}' must not contain {self.wrong_values} values."
            )
        else:
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


class InvalidStringLengthInColumn(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_String_Length_In_Column"

    def __init__(self, schema_name: str, col_name: str, min_length: int, max_length: int) -> None:
        self.schema_name = schema_name
        self.col_name = col_name
        self.min_length = min_length
        self.max_length = max_length

    def error_message(self) -> str:
        return (
            f"{self.schema_name} column '{self.col_name}' contains invalid values. "
            f"Only string values of length between {self.min_length} and {self.max_length} are accepted."
        )


class InvalidTagLength(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Tag_Length"

    def __init__(self, cols: Iterable) -> None:
        self.wrong_value_columns = cols

    def error_message(self) -> str:
        return (
            f"Only tag values with less than or equal to {MAX_TAG_LENGTH} characters are supported. "
            f"The following tag columns have more than {MAX_TAG_LENGTH} characters: "
            f"{', '.join(map(str, self.wrong_value_columns))}."
        )


class InvalidRankingCategoryValue(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Ranking_Relevance_Labels_Value"

    def __init__(self, name: str) -> None:
        self.name = name

    def error_message(self) -> str:
        return (
            f"ranking relevance labels '{self.name}' column contains invalid value"
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


class InvalidNumClassesMultiClassMap(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Num_classes_Multi_Class_Map"

    def __init__(self, dict_col_to_list_of_invalid_num_classes: Dict[str, List[str]]) -> None:
        self.invalid_col_num_classes = dict_col_to_list_of_invalid_num_classes

    def error_message(self) -> str:
        err_msg = ""
        for col, list_invalid_num_classes in self.invalid_col_num_classes.items():
            num_invalid_num_classes = len(list_invalid_num_classes)
            set_invalid_num_classes = set(list_invalid_num_classes)  # to de-duplicate
            err_msg += (
                f"Multi-Class dictionary for the following column: {col} had {num_invalid_num_classes} rows"
                f"containing an invalid number of classes. The dictionary must contain at least 1 class"
                f"and at most {MAX_NUMBER_OF_MULTI_CLASS_CLASSES} classes. Found rows with the following "
                f"invalid number of classes: {log_a_list(list(set_invalid_num_classes), 'and')}\n"
            )
        return err_msg


class InvalidMultiClassClassNameLength(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Multi_Class_Class_Name_Length"

    def __init__(self, invalid_col_class_name: Dict[str, set]) -> None:
        self.invalid_col_class_name = invalid_col_class_name

    def error_message(self) -> str:
        err_msg = ""
        for col, class_names in self.invalid_col_class_name.items():
            if len(class_names) > 10:
                # limit to 10
                class_names = list(class_names)[:10]
            else:
                class_names = list(class_names)
            err_msg += (
                f"Found some invalid class names: {log_a_list(class_names, 'and')} in the {col} column. Class"
                f" names must have at least one character and less than {MAX_MULTI_CLASS_NAME_LENGTH}.\n"
            )
        return err_msg


class InvalidMultiClassPredScoreValue(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Multi_Class_Pred_Score_Value"

    def __init__(self, invalid_col_class_scores: Dict[str, set]) -> None:
        self.invalid_col_class_scores = invalid_col_class_scores

    def error_message(self) -> str:
        err_msg = ""
        for col, scores in self.invalid_col_class_scores.items():
            if len(scores) > 10:
                # limit to 10
                scores = list(scores)[:10]
            else:
                scores = list(scores)
            err_msg += (
                f"Found some invalid scores: {log_a_list(scores, 'and')} in the {col} column that was "
                "invalid. All scores (values in dictionary) must be between 0 and 1, inclusive. \n"
            )
        return err_msg


class InvalidMultiClassActScoreValue(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Multi_Class_Act_Score_Value"

    def __init__(self, name: str) -> None:
        self.name = name

    def error_message(self) -> str:
        return (
            f"Found at least one score in the '{self.name}' column that was invalid. "
            f"All scores (values) must be either 0 or 1."
        )


class InvalidMultiClassThresholdClasses(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Multi_Class_Threshold_Classes"

    def __init__(self, name: str, prediction_class_set: set, threshold_class_set: set) -> None:
        self.name = name
        self.prediction_class_set = prediction_class_set
        self.threshold_class_set = threshold_class_set

    def error_message(self) -> str:
        return (
            "Multi-Class Prediction Scores and Threshold Scores Dictionaries must contain the same "
            f"classes. The following classes of the Prediction Scores Dictionary are not in the Threshold "
            f"Scores Dictionary: {self.prediction_class_set.difference(self.threshold_class_set)}"
            "\nThe following classes of the Threshold Scores Dictionary are not in the Prediction Scores "
            f"Dictionary: {self.threshold_class_set.difference(self.prediction_class_set)}\n"
        )


class InvalidAdditionalHeaders(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Additional_Headers"

    def __init__(self, invalid_headers: Iterable) -> None:
        self.invalid_header_names = invalid_headers

    def error_message(self) -> str:
        return (
            "Found invalid additional header, cannot use reserved headers named: "
            f"{', '.join(map(str, self.invalid_header_names))}."
        )


class InvalidRecord(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Record"

    def __init__(self, columns: List[str], indexes: List[int]) -> None:
        self.columns = columns
        self.indexes = indexes

    def error_message(self) -> str:
        return (
            f"Invalid column set full of null values in one or more rows.\n"
            f"\nProblematic Column Set:\n{log_a_list(self.columns, 'and')}\n"
            f"\nProblematic Rows:\n{log_a_list(self.indexes, join_word='and')}\n"
            "\nThis violates one of the following requirements:\n"
            " - If training environment: Prediction or actual columns cannot be all null.\n"
            " - If production environment: Prediction and actual columns cannot be all null.\n"
            " - If you are sending SHAP values, make sure not all your SHAP values are null "
            "in any given row.\n"
        )
