from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List

from arize.utils.types import Environments, Metrics, ModelTypes


class ValidationError(ABC):
    def __str__(self) -> str:
        return self.error_message()

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def error_message(self) -> str:
        pass


class ValidationFailure(Exception):
    def __init__(self, errors: List[ValidationError]):
        self.errors = errors


# ----------------------
# Minimum requred checks
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
        return "schema.embedding_feature_column_names should be a dictionary"


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

    def __init__(
        self, model_type: ModelTypes, metrics: List[Metrics], cols: Iterable
    ) -> None:
        self.model_type = model_type
        self.metrics = metrics
        self.missing_cols = cols

    def error_message(self) -> str:
        return (
            f"For logging data to a {self.model_type.name} model with support for metrics "
            f"{', '.join(m.name for m in self.metrics)}, schema must include: {', '.join(map(str, self.missing_cols))}."
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
        valid_combos = ', or \n'.join('[' + ', '.join(combo) + ']' for combo in self.suggested_combinations)
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
        return (
            "Batch ID must be a nonempty string if logging to validation environment."
        )


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


class MissingPredActShapNumeric(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Pred_or_Act_or_SHAP_Numeric"

    def error_message(self) -> str:
        return (
            "For a numeric model, the schema must specify at least one of the following: "
            "prediction label, prediction score, actual label, actual score, or SHAP value column names"
        )


class MissingPredLabelScoreCategorical(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Pred_Label_Score_Categorical"

    def error_message(self) -> str:
        return (
            "When sending a prediction score and an actual label, the schema must also include "
            "a prediction label."
        )


class MissingPreprodPredAct(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Preproduction_Pred_and_Act"

    def error_message(self) -> str:
        return (
            "For logging pre-production data, "
            "the schema must specify both prediction and actual label columns."
        )


class MissingPreprodPredActNumeric(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Preproduction_Pred_and_Act_Numeric"

    def error_message(self) -> str:
        return (
            "For logging pre-production data to a numeric model, "
            "the schema must specify both prediction and actual label or score columns."
        )


class MissingRequiredColumnsForRankingModel(ValidationError):
    def __repr__(self) -> str:
        return "Missing_Required_Columns_For_Ranking_Model"

    def error_message(self) -> str:
        return (
            "For logging data to a ranking model, schema must specify: "
            "prediction_group_id_column_name and rank_column_name"
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


class InvalidTypeFeatures(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_Features"

    def __init__(self, cols: Iterable, expected_types: List[str]) -> None:
        self.mistyped_cols = cols
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
            f"{', '.join(map(str, self.mistyped_cols))}."
        )


class InvalidTypeTags(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_Tags"

    def __init__(self, cols: Iterable, expected_types: List[str]) -> None:
        self.mistyped_cols = cols
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
            f"{', '.join(map(str, self.mistyped_cols))}."
        )


class InvalidTypeShapValues(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Type_SHAP_Values"

    def __init__(self, cols: Iterable, expected_types: List[str]) -> None:
        self.mistyped_cols = cols
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
            f"{', '.join(map(str, self.mistyped_cols))}."
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
            "If this is your pre-prodution data, you could also just remove the timestamp column "
            "from the Schema."
        )


class InvalidValueMissingValue(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Missing_Value"

    def __init__(self, name: str, missingness: str) -> None:
        self.name = name
        self.missingness = missingness

    def error_message(self) -> str:
        return f"{self.name} must not contain {self.missingness} values."


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


class InvalidPredictionGroupIDLength(ValidationError):
    def __repr__(self) -> str:
        return "Invalid_Prediction_Group_ID_Length"

    def __init__(self, name: str, acceptable_range: str) -> None:
        self.name = name
        self.acceptable_range = acceptable_range

    def error_message(self) -> str:
        return (
            f"prediction group id {self.name} column contains invalid values"
            f"Only string values of length within {self.acceptable_range}  are accepted. "
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
