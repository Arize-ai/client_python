from abc import ABC, abstractmethod
from typing import List

from arize.utils.types import ModelTypes, Environments


class ValidationError(ABC):
    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.error_message()

    @abstractmethod
    def error_message(self) -> str:
        pass


class ValidationFailure(Exception):
    def __init__(self, errors: List[ValidationError]):
        self.errors = errors


# ----------------
# Parameter checks
# ----------------


class MissingColumns(ValidationError):
    def __init__(self, cols: List[str]) -> None:
        self.missing_cols = cols

    def error_message(self) -> str:
        return (
            "The following columns are declared in the schema "
            "but are not found in the dataframe: "
            f"{', '.join(self.missing_cols)}."
        )


class InvalidModelType(ValidationError):
    def error_message(self) -> str:
        return (
            "Model type not valid. Choose one of the following: "
            f"{', '.join('ModelTypes.' + mt.name for mt in ModelTypes)}. "
            "See https://docs.arize.com/arize/concepts-and-terminology/model-types"
        )


class InvalidEnvironment(ValidationError):
    def error_message(self) -> str:
        return (
            "Environment not valid. Choose one of the following: "
            f"{', '.join('Environments.' + env.name for env in Environments)}. "
            "See https://docs.arize.com/arize/concepts-and-terminology/model-environments"
        )


class InvalidBatchId(ValidationError):
    def error_message(self) -> str:
        return (
            "Batch ID must be a nonempty string if logging to validation environment."
        )


class InvalidModelVersion(ValidationError):
    def error_message(self) -> str:
        return "Model version must be a nonempty string."


class InvalidModelId(ValidationError):
    def error_message(self) -> str:
        return "Model ID must be a nonempty string."


class MissingPredActShap(ValidationError):
    def error_message(self) -> str:
        return (
            "The schema must specify at least one of the following: "
            "prediction label, actual label, or SHAP value column names"
        )


class MissingPreprodPredAct(ValidationError):
    def error_message(self) -> str:
        return (
            "For logging pre-production data, "
            "the schema must specify both prediction and actual label columns."
        )


# -----------
# Type checks
# -----------


class InvalidType(ValidationError):
    def __init__(self, name: str, expected_types: List[str]) -> None:
        self.name = name
        self.expected_types = expected_types

    def error_message(self) -> str:
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(self.expected_types[:-1])} or {self.expected_types[-1]}"
        )
        return f"{self.name} must be of type {type_list}."


class InvalidTypeFeatures(ValidationError):
    def __init__(self, cols: List[str], expected_types: List[str]) -> None:
        self.mistyped_cols = cols
        self.expected_types = expected_types

    def error_message(self) -> str:
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(self.expected_types[:-1])} or {self.expected_types[-1]}"
        )
        return (
            f"Features must be of type {type_list}. "
            "The following feature columns have unrecognized data types: "
            f"{', '.join(self.mistyped_cols)}."
        )


class InvalidTypeShapValues(ValidationError):
    def __init__(self, cols: List[str], expected_types: List[str]) -> None:
        self.mistyped_cols = cols
        self.expected_types = expected_types

    def error_message(self) -> str:
        type_list = (
            self.expected_types[0]
            if len(self.expected_types) == 1
            else f"{', '.join(self.expected_types[:-1])} or {self.expected_types[-1]}"
        )
        return (
            f"SHAP values must be of type {type_list}. "
            "The following SHAP columns have unrecognized data types: "
            f"{', '.join(self.mistyped_cols)}."
        )


# -----------
# Value checks
# -----------


class InvalidValueTimestamp(ValidationError):
    def __init__(self, name: str, acceptible_range: str) -> None:
        self.name = name
        self.acceptible_range = acceptible_range

    def error_message(self) -> str:
        return (
            f"{self.name} is out of range. "
            f"Only values within {self.acceptible_range} from the current time are accepted. "
            "If this is your pre-prodution data, you could also just remove the timestamp column from the Schema."
        )


class InvalidValueMissingValue(ValidationError):
    def __init__(self, name: str, missingness: str) -> None:
        self.name = name
        self.missingness = missingness

    def error_message(self) -> str:
        return f"{self.name} must not contain {self.missingness} values."
