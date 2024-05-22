import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import Enum, unique
from itertools import chain
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from arize.utils.constants import (
    MAX_MULTI_CLASS_NAME_LENGTH,
    MAX_NUMBER_OF_MULTI_CLASS_CLASSES,
    MAX_NUMBER_OF_SIMILARITY_REFERENCES,
    MAX_RAW_DATA_CHARACTERS,
    MAX_RAW_DATA_CHARACTERS_TRUNCATION,
)
from arize.utils.errors import InvalidValueType
from arize.utils.logging import get_truncation_warning_message, logger


@unique
class ModelTypes(Enum):
    NUMERIC = 1
    SCORE_CATEGORICAL = 2
    RANKING = 3
    BINARY_CLASSIFICATION = 4
    REGRESSION = 5
    OBJECT_DETECTION = 6
    GENERATIVE_LLM = 7
    MULTI_CLASS = 8

    @classmethod
    def list_types(cls):
        return [t.name for t in cls]


NUMERIC_MODEL_TYPES = [ModelTypes.NUMERIC, ModelTypes.REGRESSION]
CATEGORICAL_MODEL_TYPES = [
    ModelTypes.SCORE_CATEGORICAL,
    ModelTypes.BINARY_CLASSIFICATION,
]


class DocEnum(Enum):
    def __new__(cls, value, doc=None):
        self = object.__new__(cls)  # calling super().__new__(value) here would fail
        self._value_ = value
        if doc is not None:
            self.__doc__ = doc
        return self

    def __repr__(self) -> str:
        return f"{self.name} metrics include: {self.__doc__}"


@unique
class Metrics(DocEnum):
    """
    Metric groupings, used for validation of schema columns in log() call.

    See docstring descriptions of the Enum with __doc__ or __repr__(), e.g.:
    Metrics.RANKING.__doc__
    repr(Metrics.RANKING)
    """

    REGRESSION = 1, "MAPE, MAE, RMSE, MSE, R-Squared, Mean Error"
    CLASSIFICATION = (
        2,
        "Accuracy, Recall, Precision, FPR, FNR, F1, Sensitivity, Specificity",
    )
    RANKING = 3, "NDCG"
    AUC_LOG_LOSS = 4, "AUC, PR-AUC, Log Loss"
    RANKING_LABEL = 5, "GroupAUC, MAP, MRR (soon)"


@unique
class Environments(Enum):
    TRAINING = 1
    VALIDATION = 2
    PRODUCTION = 3
    CORPUS = 4
    TRACING = 5


@dataclass
class EmbeddingColumnNames:
    vector_column_name: str = ""
    data_column_name: Optional[str] = None
    link_to_data_column_name: Optional[str] = None

    def __post_init__(self):
        if not self.vector_column_name:
            raise ValueError(
                "embedding_features require a vector to be specified. You can utilize Arize's "
                "EmbeddingGenerator (from arize.pandas.embeddings) to create embeddings "
                "if you do not have them"
            )

    def __iter__(self):
        return iter(
            (
                self.vector_column_name,
                self.data_column_name,
                self.link_to_data_column_name,
            )
        )


class Embedding(NamedTuple):
    vector: List[float]
    data: Optional[Union[str, List[str]]] = None
    link_to_data: Optional[str] = None

    def validate(self, emb_name: Union[str, int, float]) -> None:
        """
        Validates that the embedding object passed is of the correct format. That is, validations must
        be passed for vector, data & link_to_data.

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to

        Raises:
        -------
            TypeError: If the embedding fields are of the wrong type
        """

        if self.vector is not None:
            self._validate_embedding_vector(emb_name)

        # Validate embedding raw data, if present
        if self.data is not None:
            self._validate_embedding_data(emb_name, self.data)

        # Validate embedding link to data, if present
        if self.link_to_data is not None:
            self._validate_embedding_link_to_data(emb_name, self.link_to_data)

        return None

    def _validate_embedding_vector(
        self,
        emb_name: Union[str, int, float],
    ) -> None:
        """
        Validates that the embedding vector passed is of the correct format. That is:
            1. Type must be list or convertible to list (like numpy arrays, pandas Series)
            2. List must not be empty
            3. Elements in list must be floats

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """

        if not Embedding._is_valid_iterable(self.vector):
            raise TypeError(
                f'Embedding feature "{emb_name}" has vector type {type(self.vector)}. Must be '
                f"list, "
                f"np.ndarray or pd.Series"
            )
        # Fail if not all elements in list are floats
        allowed_types = (int, float, np.int16, np.int32, np.float16, np.float32)
        if not all(isinstance(val, allowed_types) for val in self.vector):  # type: ignore
            raise TypeError(
                f"Embedding vector must be a vector of integers and/or floats. Got "
                f"{emb_name}.vector = {self.vector}"
            )
        # Fail if the length of the vector is 1
        if len(self.vector) == 1:
            raise ValueError("Embedding vector must not have a size of 1")

    @staticmethod
    def _validate_embedding_data(
        emb_name: Union[str, int, float], data: Union[str, List[str]]
    ) -> None:
        """
        Validates that the embedding raw data field is of the correct format. That is:
            1. Must be string or list of strings (NLP case)

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            data (str, int, float): Raw data associated with the embedding feature. Typically raw text.

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """
        # Validate that data is a string or iterable of strings
        is_string = isinstance(data, str)
        is_allowed_iterable = not is_string and Embedding._is_valid_iterable(data)
        if not (is_string or is_allowed_iterable):
            raise TypeError(
                f'Embedding feature "{emb_name}" data field must be str, list, np.ndarray or '
                f"pd.Series"
            )

        if is_allowed_iterable:
            # Fail if not all elements in iterable are strings
            if not all(isinstance(val, str) for val in data):
                raise TypeError("Embedding data field must contain strings")

        character_count = count_characters_raw_data(data)
        if character_count > MAX_RAW_DATA_CHARACTERS:
            raise ValueError(
                f"Embedding data field must not contain more than {MAX_RAW_DATA_CHARACTERS} characters. "
                f"Found {character_count}."
            )
        elif character_count > MAX_RAW_DATA_CHARACTERS_TRUNCATION:
            logger.warning(
                get_truncation_warning_message(
                    "Embedding raw data fields", MAX_RAW_DATA_CHARACTERS_TRUNCATION
                )
            )

    @staticmethod
    def _validate_embedding_link_to_data(
        emb_name: Union[str, int, float], link_to_data: str
    ) -> None:
        """
        Validates that the embedding link to data field is of the correct format. That is:
            1. Must be string

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            link_to_data (str): Link to source data of embedding feature, typically an image file on
                cloud storage

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """
        if not isinstance(link_to_data, str):
            raise TypeError(
                f'Embedding feature "{emb_name}" link_to_data field must be str and got '
                f"{type(link_to_data)}"
            )

    @staticmethod
    def _is_valid_iterable(data: Union[str, List[str], List[float], np.ndarray, pd.Series]) -> bool:
        """
        Validates that the input data field is of the correct iterable type. That is:
            1. List or
            2. numpy array or
            3. pandas Series

        Arguments:
        ----------
            data: input iterable

        Returns:
        --------
            True if the data type is one of the accepted iterable types, false otherwise
        """
        return any(isinstance(data, t) for t in (list, np.ndarray, pd.Series))


@dataclass
class _PromptOrResponseText:
    data: str

    def validate(self, name: str) -> None:
        # Validate that data is a string
        if not isinstance(self.data, str):
            raise TypeError(f"'{name}' must be a str")

        character_count = len(self.data)
        if character_count > MAX_RAW_DATA_CHARACTERS:
            raise ValueError(
                f"'{name}' field must not contain more than {MAX_RAW_DATA_CHARACTERS} characters. "
                f"Found {character_count}."
            )
        elif character_count > MAX_RAW_DATA_CHARACTERS_TRUNCATION:
            logger.warning(
                get_truncation_warning_message(f"'{name}'", MAX_RAW_DATA_CHARACTERS_TRUNCATION)
            )
        return None


class LLMRunMetadata(NamedTuple):
    total_token_count: Optional[int] = None
    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    response_latency_ms: Optional[Union[int, float]] = None

    def validate(self) -> None:
        allowed_types = (int, float, np.int16, np.int32, np.float16, np.float32)
        if not isinstance(self.total_token_count, allowed_types):
            raise InvalidValueType(
                "total_token_count",
                self.total_token_count,
                "one of: int, float",
            )
        if not isinstance(self.prompt_token_count, allowed_types):
            raise InvalidValueType(
                "prompt_token_count",
                self.prompt_token_count,
                "one of: int, float",
            )
        if not isinstance(self.response_token_count, allowed_types):
            raise InvalidValueType(
                "response_token_count",
                self.response_token_count,
                "one of: int, float",
            )
        if not isinstance(self.response_latency_ms, allowed_types):
            raise InvalidValueType(
                "response_latency_ms",
                self.response_latency_ms,
                "one of: int, float",
            )


class ObjectDetectionColumnNames(NamedTuple):
    """
    Used to log object detection prediction and actual values that are assigned to the prediction or
    actual schema parameter.

    Arguments:
    ----------
        bounding_boxes_coordinates_column_name (str): Column name containing the coordinates of the
            rectangular outline that locates an object within an image or video. Pascal VOC format
            required. The contents of this column must be a List[List[float]].
        categories_column_name (str): Column name containing the predefined classes or labels used
            by the model to classify the detected objects. The contents of this column must be List[str].
        scores_column_names (str, optional): Column name containing the confidence scores that the
            model assigns to it's predictions, indicating how certain the model is that the predicted
            class is contained within the bounding box. This argument is only applicable for prediction
            values. The contents of this column must be List[float].
    """

    bounding_boxes_coordinates_column_name: str
    categories_column_name: str
    scores_column_name: Optional[str] = None


class ObjectDetectionLabel(NamedTuple):
    bounding_boxes_coordinates: List[List[float]]
    categories: List[str]
    scores: Optional[List[float]] = None  # Actual Object Detection Labels won't have scores

    def validate(self, prediction_or_actual: str):
        # Validate bounding boxes
        self._validate_bounding_boxes_coordinates()
        # Validate categories
        self._validate_categories()
        # Validate scores
        self._validate_scores(prediction_or_actual)
        # Validate we have the same number of bounding boxes, categories and scores
        self._validate_count_match()

    def _validate_bounding_boxes_coordinates(self):
        if not is_list_of(self.bounding_boxes_coordinates, list):
            raise TypeError(
                "Object Detection Label bounding boxes must be a list of lists of floats"
            )
        for coordinates in self.bounding_boxes_coordinates:
            if not is_list_of(coordinates, float):
                raise TypeError("Each bounding box's coordinates must be a lists of floats")
            # Format must be (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
            if len(coordinates) != 4:
                raise ValueError(
                    "Each bounding box's coordinates must be a collection of 4 floats. Found "
                    f"{coordinates}"
                )
            if any(coord < 0 for coord in coordinates):
                raise ValueError(
                    f"Bounding box's coordinates cannot be negative. Found {coordinates}"
                )
            if not (coordinates[2] > coordinates[0]):
                raise ValueError(
                    "Each bounding box bottom-right X coordinate should be larger than the "
                    f"top-left. Found {coordinates}"
                )
            if not (coordinates[3] > coordinates[1]):
                raise ValueError(
                    "Each bounding box bottom-right Y coordinate should be larger than the "
                    f"top-left. Found {coordinates}"
                )

    def _validate_categories(self):
        # Allows for categories as empty strings
        if not is_list_of(self.categories, str):
            raise TypeError("Object Detection Label categories must be a list of strings")

    def _validate_scores(self, prediction_or_actual: str):
        if self.scores is None:
            if prediction_or_actual == "prediction":
                raise ValueError("Bounding box confidence scores must not be None for predictions")
        else:
            if prediction_or_actual == "actual":
                raise ValueError("Bounding box confidence scores must be None for actuals")

            if not is_list_of(self.scores, float):
                raise TypeError("Object Detection Label scores must be a list of floats")
            if any(score > 1 or score < 0 for score in self.scores):
                raise ValueError(
                    f"Bounding box confidence scores must be between 0 and 1, inclusive. Found "
                    f"{self.scores}"
                )

    def _validate_count_match(self):
        n_bounding_boxes = len(self.bounding_boxes_coordinates)
        if n_bounding_boxes == 0:
            raise ValueError(
                f"Object Detection Labels must contain at least 1 bounding box. Found"
                f" {n_bounding_boxes}."
            )

        n_categories = len(self.categories)
        if n_bounding_boxes != n_categories:
            raise ValueError(
                "Object Detection Labels must contain the same number of bounding boxes and "
                f"categories. Found {n_bounding_boxes} bounding boxes and {n_categories} "
                "categories."
            )

        if self.scores is not None:
            n_scores = len(self.scores)
            if n_bounding_boxes != n_scores:
                raise ValueError(
                    "Object Detection Labels must contain the same number of bounding boxes and "
                    f"confidence scores. Found {n_bounding_boxes} bounding boxes and {n_scores} "
                    "scores."
                )


class MultiClassPredictionLabel(NamedTuple):
    """
    Used to log multi class prediction label
    Arguments:
    ----------
    MultiClassPredictionLabel
        prediction_scores (Dict[str, Union[float, int]]): the prediction scores of the classes.
        threshold_scores (Optional[Dict[str, Union[float, int]]]): the threshold scores of the classes.
            Only Multi Label will have threshold scores.
    """

    prediction_scores: Dict[str, Union[float, int]]
    threshold_scores: Dict[str, Union[float, int]] = None

    def validate(self):
        # Validate scores
        self._validate_prediction_scores()
        self._validate_threshold_scores()

    def _validate_prediction_scores(self):
        # prediction dictionary validations
        if not is_dict_of(
            self.prediction_scores,
            key_allowed_types=str,
            value_allowed_types=(int, float),
        ):
            raise ValueError(
                "Multi-Class Prediction Scores must be a dictionary with keys of type str "
                "and values must be a numeric type (int or float)."
            )
        # validate length of prediction scores
        n_prediction_scores = len(self.prediction_scores)
        if n_prediction_scores == 0 or n_prediction_scores > MAX_NUMBER_OF_MULTI_CLASS_CLASSES:
            raise ValueError(
                f"Multi-Class Prediction Scores dictionary must contain at least 1 class and "
                f"can contain at most {MAX_NUMBER_OF_MULTI_CLASS_CLASSES} classes. "
                f"Found {n_prediction_scores} classes."
            )

        for class_name, score in self.prediction_scores.items():
            if class_name == "":
                raise ValueError(
                    "Found at least one class name as an empty string in the Multi-Class Prediction Scores "
                    "dictionary. All class names (keys in dictionary) must be non-empty strings."
                )
            if len(class_name) > MAX_MULTI_CLASS_NAME_LENGTH:
                raise ValueError(
                    f"Found at least one class name with more characters than the limit allowed: "
                    f"{MAX_MULTI_CLASS_NAME_LENGTH} characters. "
                    f"The class name '{class_name}' has {len(class_name)} characters."
                )
            if score > 1 or score < 0:
                raise ValueError(
                    "Found at least one score in the Multi-Class Prediction Scores dictionary that was "
                    "invalid. All scores (values in dictionary) must be between 0 and 1, inclusive."
                )

    def _validate_threshold_scores(self):
        if self.threshold_scores is None or len(self.threshold_scores) == 0:
            return
        if not is_dict_of(
            self.threshold_scores,
            key_allowed_types=str,
            value_allowed_types=(int, float),
        ):
            raise ValueError(
                "Multi-Class Threshold Scores must be a dictionary with keys of type str "
                "and values must be a numeric type (int or float)."
            )

        # validate there are the same number of thresholds as predictions
        if len(self.threshold_scores) != len(self.prediction_scores):
            raise ValueError(
                "Multi-Class Prediction Scores and Threshold Scores Dictionaries must contain the same number"
                f" of number of classes. Found Prediction Scores Dictionary contains "
                f"{len(self.prediction_scores)} classes and Threshold Scores Dictionary contains "
                f"{len(self.threshold_scores)} classes."
            )

        # validate prediction scores and threshold scores dictionaries contain same classes
        prediction_class_set = set(self.prediction_scores.keys())
        threshold_class_set = set(self.threshold_scores.keys())
        if prediction_class_set != threshold_class_set:
            raise ValueError(
                "Multi-Class Prediction Scores and Threshold Scores Dictionaries must contain the same "
                f"classes. The following classes of the Prediction Scores Dictionary are not in the "
                f"Threshold Scores Dictionary: {prediction_class_set.difference(threshold_class_set)} \n"
                "The following classes of the Threshold Scores Dictionary are not in the Prediction Scores "
                f"Dictionary: {threshold_class_set.difference(prediction_class_set)}"
            )

        for class_name, t_score in self.threshold_scores.items():
            if math.isnan(t_score) or t_score > 1 or t_score < 0:
                raise ValueError(
                    "Found at least one score in the Multi-Class Threshold Scores dictionary that was "
                    "invalid. All scores (values) must be between 0 and 1, inclusive. "
                    f"Found class '{class_name}' has score {t_score}"
                )


class MultiClassActualLabel(NamedTuple):
    """
    Used to log multi class actual label
    Arguments:
    ----------
    MultiClassActualLabel
        actual_scores (Dict[str, Union[float, int]]): the actual scores of the classes.
        Any class in actual_scores with a score of 1 will be sent to arize
    """

    actual_scores: Dict[str, Union[float, int]]

    def validate(self):
        # Validate scores
        self._validate_actual_scores()

    def _validate_actual_scores(self):
        if not is_dict_of(
            self.actual_scores, key_allowed_types=str, value_allowed_types=(int, float)
        ):
            raise ValueError(
                "Multi-Class Actual Scores must be a dictionary with keys of type str "
                "and values must be a numeric type (int or float)."
            )
        n_actual_scores = len(self.actual_scores)
        if n_actual_scores == 0 or n_actual_scores > MAX_NUMBER_OF_MULTI_CLASS_CLASSES:
            raise ValueError(
                f"Multi-Class Actual Scores dictionary must contain at least 1 class and "
                f"can contain at most {MAX_NUMBER_OF_MULTI_CLASS_CLASSES} classes. "
                f"Found {n_actual_scores} classes."
            )
        for class_name, score in self.actual_scores.items():
            if class_name == "":
                raise ValueError(
                    "Found at least one class name as an empty string in the Multi-Class Actual Scores "
                    "dictionary. All class names (keys) must be non-empty strings."
                )
            if len(class_name) > MAX_MULTI_CLASS_NAME_LENGTH:
                raise ValueError(
                    f"Found at least one class name with more characters than the limit allowed: "
                    f"{MAX_MULTI_CLASS_NAME_LENGTH} characters. "
                    f"The class name '{class_name}' has {len(class_name)} characters."
                )
            if score != 1 and score != 0:
                raise ValueError(
                    "Found at least one score in the Multi-Class Actual Scores dictionary that was invalid. "
                    f"All scores (values) must be either 0 or 1. Found class '{class_name}' has score {score}"
                )


class RankingPredictionLabel(NamedTuple):
    group_id: str
    rank: int
    score: Optional[float] = None
    label: Optional[str] = None

    def validate(self):
        # Validate existence of required fields: prediction_group_id and rank
        if self.group_id is None or self.rank is None:
            raise ValueError("RankingPredictionLabel must contain: group_id and rank")
        # Validate prediction_group_id
        self._validate_group_id()
        # Validate rank
        self._validate_rank()
        # Validate label type
        if self.label is not None:
            self._validate_label()
        # Validate score type
        if self.score is not None:
            self._validate_score()

    def _validate_group_id(self):
        if not isinstance(self.group_id, str):
            raise TypeError("Prediction Group ID must be a string")
        if not (1 <= len(self.group_id) <= 36):
            raise ValueError(
                f"Prediction Group ID must have length between 1 and 36. Found {len(self.group_id)}"
            )

    def _validate_rank(self):
        if not isinstance(self.rank, int):
            raise TypeError("Prediction Rank must be an int")
        if not (1 <= self.rank <= 100):
            raise ValueError(
                f"Prediction Rank must be between 1 and 100, inclusive. Found {self.rank}"
            )

    def _validate_label(self):
        if not isinstance(self.label, str):
            raise TypeError("Prediction Label must be a str")
        if self.label == "":
            raise ValueError("Prediction Label must not be an empty string.")

    def _validate_score(self):
        if not isinstance(self.score, (float, int)):
            raise TypeError("Prediction Score must be a float or an int")


class RankingActualLabel(NamedTuple):
    relevance_labels: Optional[List[str]] = None
    relevance_score: Optional[float] = None

    def validate(self):
        # Validate relevance_labels type
        if self.relevance_labels is not None:
            self._validate_relevance_labels(self.relevance_labels)
        # Validate relevance score type
        if self.relevance_score is not None:
            self._validate_relevance_score(self.relevance_score)

    @staticmethod
    def _validate_relevance_labels(relevance_labels: List[str]):
        if not is_list_of(relevance_labels, str):
            raise TypeError("Actual Relevance Labels must be a list of strings")
        if any(label == "" for label in relevance_labels):
            raise ValueError("Actual Relevance Labels must be not contain empty strings")

    @staticmethod
    def _validate_relevance_score(relevance_score: float):
        if not isinstance(relevance_score, (float, int)):
            raise TypeError("Actual Relevance score must be a float or an int")


@dataclass
class PromptTemplateColumnNames:
    template_column_name: Optional[str] = None
    template_version_column_name: Optional[str] = None

    def __iter__(self):
        return iter((self.template_column_name, self.template_version_column_name))


@dataclass
class LLMConfigColumnNames:
    model_column_name: Optional[str] = None
    params_column_name: Optional[str] = None

    def __iter__(self):
        return iter((self.model_column_name, self.params_column_name))


@dataclass
class LLMRunMetadataColumnNames:
    total_token_count_column_name: Optional[str] = None
    prompt_token_count_column_name: Optional[str] = None
    response_token_count_column_name: Optional[str] = None
    response_latency_ms_column_name: Optional[str] = None

    def __iter__(self):
        return iter(
            (
                self.total_token_count_column_name,
                self.prompt_token_count_column_name,
                self.response_token_count_column_name,
                self.response_latency_ms_column_name,
            )
        )


@dataclass
class DocumentColumnNames:
    id_column_name: Optional[str] = None
    version_column_name: Optional[str] = None
    text_embedding_column_names: Optional[EmbeddingColumnNames] = None

    def __iter__(self):
        return iter(
            (
                self.id_column_name,
                self.version_column_name,
                self.text_embedding_column_names,
            )
        )


@dataclass
class SimilarityReference:
    prediction_id: str
    reference_column_name: str
    prediction_timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.prediction_id == "":
            raise ValueError("prediction id cannot be empty")
        if self.reference_column_name == "":
            raise ValueError("Reference column name cannot be empty")
        if self.prediction_timestamp and not isinstance(self.prediction_timestamp, datetime):
            raise TypeError("prediction_timestamp must be a datetime object")


@dataclass
class SimilaritySearchParams:
    references: List[SimilarityReference]
    search_column_name: str
    threshold: float = 0

    def __post_init__(self):
        if (
            not self.references
            or len(self.references) <= 0
            or len(self.references) > MAX_NUMBER_OF_SIMILARITY_REFERENCES
        ):
            raise ValueError(
                f"must have at least 1 and no more than {MAX_NUMBER_OF_SIMILARITY_REFERENCES} references"
            )
        if self.search_column_name == "":
            raise ValueError("search column name cannot be empty")
        if self.threshold > 1 or self.threshold < -1:
            raise ValueError("threshold cannot be outside of range -1, 1")
        for reference in self.references:
            if not isinstance(reference, SimilarityReference):
                raise TypeError("all references must be instances of SimilarityReference")


@dataclass(frozen=True)
class BaseSchema:
    def replace(self, **changes):
        return replace(self, **changes)

    def asdict(self) -> Dict[str, str]:
        return asdict(self)

    def get_used_columns(self) -> Set[str]:
        return set(self.get_used_columns_counts().keys())

    def get_used_columns_counts(self) -> Dict[str, int]:
        raise NotImplementedError()


@dataclass(frozen=True)
class TypedColumns:
    """
    Optional class used for explicit type enforcement of feature and tag columns in the dataframe.

    Usage:
    ------
        When initializing a Schema, use TypedColumns in place of a list of string column names.
        e.g. feature_column_names=TypedColumns(
                inferred=["feature_1", "feature_2"],
                to_str=["feature_3"],
                to_int=["feature_4"]
            )

    Fields:
    -------
        inferred (Optional[List[str]]): List of columns that will not be altered at all.
            The values in these columns will have their type inferred as Arize validates and ingests the data.
            There's no difference between passing in all column names as inferred
            vs. not using TypedColumns at all.
        to_str (Optional[List[str]]): List of columns that should be cast to pandas StringDType.
        to_int (Optional[List[str]]): List of columns that should be cast to pandas Int64DType.
        to_float (Optional[List[str]]): List of columns that should be cast to pandas Float64DType.

    Notes:
    ------
        - If a TypedColumns object is included in a Schema, pandas version 1.0.0 or higher is required.
        - Pandas StringDType is still considered an experimental field.
        - Columns not present in any field will not be captured in the Schema.
        - StringDType, Int64DType, and Float64DType are all nullable column types.
        Null values will be ingested and represented in Arize as empty values.
    """

    inferred: Optional[List[str]] = None
    to_str: Optional[List[str]] = None
    to_int: Optional[List[str]] = None
    to_float: Optional[List[str]] = None

    def get_all_column_names(self) -> List[str]:
        return list(chain.from_iterable(filter(None, self.__dict__.values())))

    def has_duplicate_columns(self) -> Tuple[bool, Set[str]]:
        # True if there are duplicates within a field's list or across fields.
        # Return a set of the duplicate column names.
        cols = self.get_all_column_names()
        duplicates = set([x for x in cols if cols.count(x) > 1])
        return len(duplicates) > 0, duplicates

    def is_empty(self) -> bool:
        return not self.get_all_column_names()


@dataclass(frozen=True)
class Schema(BaseSchema):
    """
    Used to organize and map column names containing model data within your Pandas dataframe to
    Arize.

    Arguments:
    ----------
        prediction_id_column_name (str, optional): Column name for the predictions unique identifier.
            Unique IDs are used to match a prediction to delayed actuals or feature importances in Arize.
            If prediction ids are not provided, it will default to an empty string "" and, when possible,
            Arize will create a random prediction id on the server side. Prediction id must be a string column
            with each row indicating a unique prediction event.
        feature_column_names (Union[List[str], TypedColumns], optional): Column names for features.
            The content of feature columns can be int, float, string. If TypedColumns is used,
            the columns will be cast to the provided types prior to logging.
        tag_column_names (Union[List[str], TypedColumns], optional): Column names for tags. The content of tag
            columns can be int, float, string. If TypedColumns is used,
            the columns will be cast to the provided types prior to logging.
        timestamp_column_name (str, optional): Column name for timestamps. The content of this
            column must be int Unix Timestamps in seconds.
        prediction_label_column_name (str, optional): Column name for categorical prediction values.
            The content of this column must be convertible to string.
        prediction_score_column_name (str, optional): Column name for numeric prediction values. The
            content of this column must be int/float or list of dictionaries mapping class names to
            int/float scores in the case of MULTI_CLASS model types.
        actual_label_column_name (str, optional): Column name for categorical ground truth values.
            The content of this column must be convertible to string.
        actual_score_column_name (str, optional): Column name for numeric ground truth values. The
            content of this column must be int/float or list of dictionaries mapping class names to
            int/float scores in the case of MULTI_CLASS model types.
        shap_values_column_names (Dict[str, str], optional): Dictionary mapping feature column name
            and corresponding SHAP feature importance column name. e.g.
            {{"feat_A": "feat_A_shap", "feat_B": "feat_B_shap"}}
        embedding_feature_column_names (Dict[str, EmbeddingColumnNames], optional): Dictionary
            mapping embedding display names to EmbeddingColumnNames objects.
        prediction_group_id_column_name (str, optional): Column name for ranking groups or lists in
            ranking models. The content of this column must be string and is limited to 128 characters.
        rank_column_name (str, optional): Column name for rank of each element on the its group or
            list. The content of this column must be integer between 1-100.
        relevance_score_column_name (str, optional): Column name for ranking model type numeric
            ground truth values. The content of this column must be int/float.
        relevance_labels_column_name (str, optional): Column name for ranking model type categorical
            ground truth values. The content of this column must be a string.
        object_detection_prediction_column_names (ObjectDetectionColumnNames, optional):
            ObjectDetectionColumnNames object containing information defining the predicted bounding
            boxes' coordinates, categories, and scores.
        object_detection_actual_column_names (ObjectDetectionColumnNames, optional):
            ObjectDetectionColumnNames object containing information defining the actual bounding
            boxes' coordinates, categories, and scores.
        prompt_column_names (str or EmbeddingColumnNames, optional): column names for text that is passed
            to the GENERATIVE_LLM model. It accepts a string (if sending only a text column) or
            EmbeddingColumnNames object containing the embedding vector data (required) and raw text
            (optional) for the input text your model acts on.
        response_column_names (str or EmbeddingColumnNames, optional): column names for text generated by
            the GENERATIVE_LLM model. It accepts a string (if sending only a text column) or
            EmbeddingColumnNames object containing the embedding vector data (required) and raw text
            (optional) for the text your model generates.
        prompt_template_column_names (PromptTemplateColumnNames, optional): PromptTemplateColumnNames object
            containing the prompt template and the prompt template version.
        llm_config_column_names (LLMConfigColumnNames, optional): LLMConfigColumnNames object containing
            the LLM's model name and its hyper parameters used at inference.
        llm_run_metadata_column_names (LLMRunMetadataColumnNames, optional): LLMRunMetadataColumnNames
            object containing token counts and latency metrics
        retrieved_document_ids_column_name (str, optional): Column name for retrieved document ids.
            The content of this column must be lists with entries convertible to strings.
        multi_class_threshold_scores_column_name (str, optional):
            Column name for dictionary that maps class names to threshold values. The
            content of this column must be dictionary of str -> int/float.

    Methods:
    --------
        replace(**changes):
            Replaces fields of the schema
        asdict():
            Returns the schema as a dictionary. Warning: the types are not maintained, fields are
            converted to strings.
        get_used_columns():
            Returns a set with the unique collection of columns to be used from the dataframe.
    """

    prediction_id_column_name: Optional[str] = None
    feature_column_names: Optional[Union[List[str], TypedColumns]] = None
    tag_column_names: Optional[Union[List[str], TypedColumns]] = None
    timestamp_column_name: Optional[str] = None
    prediction_label_column_name: Optional[str] = None
    prediction_score_column_name: Optional[str] = None
    actual_label_column_name: Optional[str] = None
    actual_score_column_name: Optional[str] = None
    shap_values_column_names: Optional[Dict[str, str]] = None
    embedding_feature_column_names: Optional[Dict[str, EmbeddingColumnNames]] = None  # type:ignore
    prediction_group_id_column_name: Optional[str] = None
    rank_column_name: Optional[str] = None
    attributions_column_name: Optional[str] = None
    relevance_score_column_name: Optional[str] = None
    relevance_labels_column_name: Optional[str] = None
    object_detection_prediction_column_names: Optional[ObjectDetectionColumnNames] = None
    object_detection_actual_column_names: Optional[ObjectDetectionColumnNames] = None
    prompt_column_names: Optional[Union[str, EmbeddingColumnNames]] = None
    response_column_names: Optional[Union[str, EmbeddingColumnNames]] = None
    prompt_template_column_names: Optional[PromptTemplateColumnNames] = None
    llm_config_column_names: Optional[LLMConfigColumnNames] = None
    llm_run_metadata_column_names: Optional[LLMRunMetadataColumnNames] = None
    retrieved_document_ids_column_name: Optional[List[str]] = None
    multi_class_threshold_scores_column_name: Optional[str] = None

    def get_used_columns_counts(self) -> Dict[str, int]:
        columns_used_counts = {}

        for field in self.__dataclass_fields__:
            if field.endswith("column_name"):
                col = getattr(self, field)
                if col is not None:
                    add_to_column_count_dictionary(columns_used_counts, col)

        if self.feature_column_names is not None:
            for col in self.feature_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.embedding_feature_column_names is not None:
            for emb_col_names in self.embedding_feature_column_names.values():
                add_to_column_count_dictionary(
                    columns_used_counts, emb_col_names.vector_column_name
                )
                if emb_col_names.data_column_name is not None:
                    add_to_column_count_dictionary(
                        columns_used_counts, emb_col_names.data_column_name
                    )
                if emb_col_names.link_to_data_column_name is not None:
                    add_to_column_count_dictionary(
                        columns_used_counts, emb_col_names.link_to_data_column_name
                    )

        if self.tag_column_names is not None:
            for col in self.tag_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.shap_values_column_names is not None:
            for col in self.shap_values_column_names.values():
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.object_detection_prediction_column_names is not None:
            for col in self.object_detection_prediction_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.object_detection_actual_column_names is not None:
            for col in self.object_detection_actual_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.prompt_column_names is not None:
            if isinstance(self.prompt_column_names, str):
                add_to_column_count_dictionary(columns_used_counts, self.prompt_column_names)
            elif isinstance(self.prompt_column_names, EmbeddingColumnNames):
                add_to_column_count_dictionary(
                    columns_used_counts, self.prompt_column_names.vector_column_name
                )
                if self.prompt_column_names.data_column_name is not None:
                    add_to_column_count_dictionary(
                        columns_used_counts, self.prompt_column_names.data_column_name
                    )

        if self.response_column_names is not None:
            if isinstance(self.response_column_names, str):
                add_to_column_count_dictionary(columns_used_counts, self.response_column_names)
            elif isinstance(self.response_column_names, EmbeddingColumnNames):
                add_to_column_count_dictionary(
                    columns_used_counts, self.response_column_names.vector_column_name
                )
                if self.response_column_names.data_column_name is not None:
                    add_to_column_count_dictionary(
                        columns_used_counts, self.response_column_names.data_column_name
                    )

        if self.prompt_template_column_names is not None:
            for col in self.prompt_template_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.llm_config_column_names is not None:
            for col in self.llm_config_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.llm_run_metadata_column_names is not None:
            for col in self.llm_run_metadata_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        return columns_used_counts

    def has_prediction_columns(self) -> bool:
        prediction_cols = (
            self.prediction_label_column_name,
            self.prediction_score_column_name,
            self.rank_column_name,
            self.prediction_group_id_column_name,
            self.object_detection_prediction_column_names,
            self.multi_class_threshold_scores_column_name,
        )
        return any(col is not None for col in prediction_cols)

    def has_actual_columns(self) -> bool:
        actual_cols = (
            self.actual_label_column_name,
            self.actual_score_column_name,
            self.relevance_labels_column_name,
            self.relevance_score_column_name,
            self.object_detection_actual_column_names,
        )
        return any(col is not None for col in actual_cols)

    def has_feature_importance_columns(self) -> bool:
        feature_importance_cols = (self.shap_values_column_names,)
        return any(col is not None for col in feature_importance_cols)

    def has_typed_columns(self) -> bool:
        return any(self.typed_column_fields())

    def typed_column_fields(self) -> Set[str]:
        return {
            field
            for field in self.__dataclass_fields__
            if isinstance(getattr(self, field), TypedColumns)
        }


@dataclass(frozen=True)
class CorpusSchema(BaseSchema):
    document_id_column_name: Optional[str] = None
    document_version_column_name: Optional[str] = None
    document_text_embedding_column_names: Optional[EmbeddingColumnNames] = None

    def get_used_columns_counts(self) -> Dict[str, int]:
        columns_used_counts = {}

        if self.document_id_column_name is not None:
            add_to_column_count_dictionary(columns_used_counts, self.document_id_column_name)
        if self.document_version_column_name is not None:
            add_to_column_count_dictionary(columns_used_counts, self.document_version_column_name)
        if self.document_text_embedding_column_names is not None:
            add_to_column_count_dictionary(
                columns_used_counts,
                self.document_text_embedding_column_names.vector_column_name,
            )
            if self.document_text_embedding_column_names.data_column_name is not None:
                add_to_column_count_dictionary(
                    columns_used_counts,
                    self.document_text_embedding_column_names.data_column_name,
                )
            if self.document_text_embedding_column_names.link_to_data_column_name is not None:
                add_to_column_count_dictionary(
                    columns_used_counts,
                    self.document_text_embedding_column_names.link_to_data_column_name,
                )
        return columns_used_counts


@unique
class ArizeTypes(Enum):
    STR = 0
    FLOAT = 1
    INT = 2


@dataclass(frozen=True)
class TypedValue:
    type: ArizeTypes
    value: Union[str, bool, float, int]


def is_json_str(s: str) -> bool:
    try:
        json.loads(s)
    except ValueError:
        return False
    except TypeError:
        return False
    return True


T = TypeVar("T", bound=type)


def is_array_of(arr: Sequence[object], tp: T) -> bool:
    return isinstance(arr, np.ndarray) and all(isinstance(x, tp) for x in arr)


def is_list_of(lst: Sequence[object], tp: T) -> bool:
    return isinstance(lst, list) and all(isinstance(x, tp) for x in lst)


def is_iterable_of(lst: Sequence[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)


def is_dict_of(
    d: Dict[object, object],
    key_allowed_types: T,
    value_allowed_types: T = (),
    value_list_allowed_types: T = (),
) -> bool:
    """
    Method to check types are valid for dictionary.

    Arguments:
    ----------
        d (Dict[object, object]): dictionary itself
        key_allowed_types (T): all allowed types for keys of dictionary
        value_allowed_types (T): all allowed types for values of dictionary
        value_list_allowed_types (T): if value is a list, these are the allowed types for value list

    Returns:
    --------
        True if the data types of dictionary match the types specified by the arguments, false otherwise
    """
    if value_list_allowed_types and not isinstance(value_list_allowed_types, tuple):
        value_list_allowed_types = (value_list_allowed_types,)

    return (
        isinstance(d, dict)
        and all(isinstance(k, key_allowed_types) for k in d.keys())
        and all(
            isinstance(v, value_allowed_types)
            or any(is_list_of(v, t) for t in value_list_allowed_types)
            for v in d.values()
            if value_allowed_types or value_list_allowed_types
        )
    )


def count_characters_raw_data(data: Union[str, List[str]]) -> int:
    character_count = 0
    if isinstance(data, str):
        character_count = len(data)
    elif is_iterable_of(data, str):
        for string in data:
            character_count += len(string)
    else:
        raise TypeError(
            f"Cannot count characters for raw data. Expecting strings or "
            f"list of strings but another type was found: {type(data)}."
        )
    return character_count


def add_to_column_count_dictionary(column_dictionary: Dict[str, int], col: Optional[str]):
    if col:
        if col in column_dictionary:
            column_dictionary[col] += 1
        else:
            column_dictionary[col] = 1
