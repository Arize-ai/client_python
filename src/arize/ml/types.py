"""Common type definitions and data models used across the ML Client."""

import logging
import math
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum, unique
from itertools import chain
from typing import NamedTuple

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np

from arize.constants.ml import (
    MAX_MULTI_CLASS_NAME_LENGTH,
    MAX_NUMBER_OF_MULTI_CLASS_CLASSES,
    MAX_NUMBER_OF_SIMILARITY_REFERENCES,
    MAX_RAW_DATA_CHARACTERS,
    MAX_RAW_DATA_CHARACTERS_TRUNCATION,
)
from arize.exceptions.parameters import InvalidValueType
from arize.logging import get_truncation_warning_message
from arize.utils.types import is_dict_of, is_iterable_of, is_list_of

logger = logging.getLogger(__name__)


def _normalize_column_names(
    col_names: "list[str] | TypedColumns | None",
) -> list[str]:
    """Convert TypedColumns or list to a flat list of column names."""
    if col_names is None:
        return []
    if isinstance(col_names, list):
        return col_names
    return col_names.get_all_column_names()


@unique
class ModelTypes(Enum):
    """Enum representing supported model types in Arize."""

    NUMERIC = 1
    SCORE_CATEGORICAL = 2
    RANKING = 3
    BINARY_CLASSIFICATION = 4
    REGRESSION = 5
    OBJECT_DETECTION = 6
    GENERATIVE_LLM = 7
    MULTI_CLASS = 8

    @classmethod
    def list_types(cls) -> list[str]:
        """Return a list of all type names in this enum."""
        return [t.name for t in cls]


NUMERIC_MODEL_TYPES = [ModelTypes.NUMERIC, ModelTypes.REGRESSION]
CATEGORICAL_MODEL_TYPES = [
    ModelTypes.SCORE_CATEGORICAL,
    ModelTypes.BINARY_CLASSIFICATION,
]


class DocEnum(Enum):
    """Enum subclass supporting inline documentation for enum members."""

    def __new__(cls, value: object, doc: str | None = None) -> Self:
        """Create a new enum instance with optional documentation."""
        self = object.__new__(
            cls
        )  # calling super().__new__(value) here would fail
        self._value_ = value
        if doc is not None:
            self.__doc__ = doc
        return self

    def __repr__(self) -> str:
        """Return a string representation including documentation."""
        return f"{self.name} metrics include: {self.__doc__}"


@unique
class Metrics(DocEnum):
    """Metric groupings, used for validation of schema columns in log() call.

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
    """Enum representing deployment environments for models."""

    TRAINING = 1
    VALIDATION = 2
    PRODUCTION = 3
    CORPUS = 4
    TRACING = 5


@dataclass
class EmbeddingColumnNames:
    """Column names for embedding feature data."""

    vector_column_name: str = ""
    data_column_name: str | None = None
    link_to_data_column_name: str | None = None

    def __post_init__(self) -> None:
        """Validate that vector column name is specified.

        Raises:
            ValueError: If vector_column_name is empty.
        """
        if not self.vector_column_name:
            raise ValueError(
                "embedding_features require a vector to be specified. You can "
                "utilize Arize's EmbeddingGenerator to create embeddings "
                "(from arize.pandas.embeddings) if you do not have them"
            )

    def __iter__(self) -> Iterator[str | None]:
        """Iterate over the embedding column names."""
        return iter(
            (
                self.vector_column_name,
                self.data_column_name,
                self.link_to_data_column_name,
            )
        )


class Embedding(NamedTuple):
    """Container for embedding vector data with optional raw data and links."""

    vector: list[float]
    data: str | list[str] | None = None
    link_to_data: str | None = None

    def validate(self, emb_name: str | int | float) -> None:
        """Validates that the embedding object passed is of the correct format.

        Ensures validations are passed for vector, data, and link_to_data fields.

        Args:
            emb_name: Name of the embedding feature the
                vector belongs to.

        Raises:
            TypeError: If the embedding fields are of the wrong type.
        """
        if self.vector is not None:
            self._validate_embedding_vector(emb_name)

        # Validate embedding raw data, if present
        if self.data is not None:
            self._validate_embedding_data(emb_name, self.data)

        # Validate embedding link to data, if present
        if self.link_to_data is not None:
            self._validate_embedding_link_to_data(emb_name, self.link_to_data)

        return

    def _validate_embedding_vector(
        self,
        emb_name: str | int | float,
    ) -> None:
        """Validates that the embedding vector passed is of the correct format.

        Requirements: 1) Type must be list or convertible to list (like numpy arrays,
        pandas Series), 2) List must not be empty, 3) Elements in list must be floats.

        Args:
            emb_name: Name of the embedding feature the vector
                belongs to.

        Raises:
            TypeError: If the embedding does not satisfy requirements above.
        """
        if not Embedding._is_valid_iterable(self.vector):
            raise TypeError(
                f"Embedding feature '{emb_name}' has vector type "
                f"{type(self.vector)}. Must be list or np.ndarray"
            )
        # Fail if not all elements in list are floats
        allowed_types = (int, float, np.int16, np.int32, np.float16, np.float32)
        if not all(isinstance(val, allowed_types) for val in self.vector):
            raise TypeError(
                f"Embedding vector must be a vector of integers and/or floats. Got "
                f"{emb_name}.vector = {self.vector}"
            )
        # Fail if the length of the vector is 1
        if len(self.vector) == 1:
            raise ValueError("Embedding vector must not have a size of 1")

    @staticmethod
    def _validate_embedding_data(
        emb_name: str | int | float, data: str | list[str]
    ) -> None:
        """Validates that the embedding raw data field is of the correct format.

        Requirement: Must be string or list of strings (NLP case).

        Args:
            emb_name: Name of the embedding feature the vector belongs to.
            data: Raw data associated with the embedding feature.
                Typically raw text.

        Raises:
            TypeError: If the embedding does not satisfy requirements above.
        """
        # Validate that data is a string or iterable of strings
        is_string = isinstance(data, str)
        is_allowed_iterable = not is_string and Embedding._is_valid_iterable(
            data
        )
        if not (is_string or is_allowed_iterable):
            raise TypeError(
                f'Embedding feature "{emb_name}" data field must be str, list, or np.ndarray'
            )

        # Fail if not all elements in iterable are strings
        if is_allowed_iterable and not all(
            isinstance(val, str) for val in data
        ):
            raise TypeError("Embedding data field must contain strings")

        character_count = _count_characters_raw_data(data)
        if character_count > MAX_RAW_DATA_CHARACTERS:
            raise ValueError(
                f"Embedding data field must not contain more than {MAX_RAW_DATA_CHARACTERS} characters. "
                f"Found {character_count}."
            )
        if character_count > MAX_RAW_DATA_CHARACTERS_TRUNCATION:
            logger.warning(
                get_truncation_warning_message(
                    "Embedding raw data fields",
                    MAX_RAW_DATA_CHARACTERS_TRUNCATION,
                )
            )

    @staticmethod
    def _validate_embedding_link_to_data(
        emb_name: str | int | float, link_to_data: str
    ) -> None:
        """Validates that the embedding link to data field is of the correct format.

        Requirement: Must be string.

        Args:
            emb_name: Name of the embedding feature the vector belongs to.
            link_to_data: Link to source data of embedding feature, typically an
                image file on cloud storage.

        Raises:
            TypeError: If the embedding does not satisfy requirements above.
        """
        if not isinstance(link_to_data, str):
            raise TypeError(
                f'Embedding feature "{emb_name}" link_to_data field must be str and got '
                f"{type(link_to_data)}"
            )

    @staticmethod
    def _is_valid_iterable(
        data: object,
    ) -> bool:
        """Validates that the input data field is of the correct iterable type.

        Accepted types: 1) List, 2) numpy array, or 3) pandas Series.

        Args:
            data: Input iterable.

        Returns:
            True if the data type is one of the accepted iterable types,
                false otherwise.
        """
        return any(isinstance(data, t) for t in (list, np.ndarray))


class LLMRunMetadata(NamedTuple):
    """Metadata for LLM execution including token counts and latency."""

    total_token_count: int | None = None
    prompt_token_count: int | None = None
    response_token_count: int | None = None
    response_latency_ms: int | float | None = None

    def validate(self) -> None:
        """Validate the field values and constraints."""
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
    """Used to log object detection prediction and actual values.

    These values are assigned to the prediction or actual schema parameter.

    Args:
        bounding_boxes_coordinates_column_name: Column name containing the coordinates of the
            rectangular outline that locates an object within an image or video. Pascal VOC format
            required. The contents of this column must be a List[List[float]].
        categories_column_name: Column name containing the predefined classes or labels used
            by the model to classify the detected objects. The contents of this column must be List[str].
        scores_column_names: Column name containing the confidence scores that the
            model assigns to it's predictions, indicating how certain the model is that the predicted
            class is contained within the bounding box. This argument is only applicable for prediction
            values. The contents of this column must be List[float].
    """

    bounding_boxes_coordinates_column_name: str
    categories_column_name: str
    scores_column_name: str | None = None


class SemanticSegmentationColumnNames(NamedTuple):
    """Used to log semantic segmentation prediction and actual values.

    These values are assigned to the prediction or actual schema parameter.

    Args:
        polygon_coordinates_column_name: Column name containing the coordinates of the vertices
            of the polygon mask within an image or video. The first sublist contains the
            coordinates of the outline of the polygon. The subsequent sublists contain the coordinates
            of any cutouts within the polygon. The contents of this column must be a List[List[float]].
        categories_column_name: Column name containing the predefined classes or labels used
            by the model to classify the detected objects. The contents of this column must be List[str].
    """

    polygon_coordinates_column_name: str
    categories_column_name: str


class InstanceSegmentationPredictionColumnNames(NamedTuple):
    """Used to log instance segmentation prediction values for the prediction schema parameter.

    Args:
        polygon_coordinates_column_name: Column name containing the coordinates of the vertices
            of the polygon mask within an image or video. The first sublist contains the
            coordinates of the outline of the polygon. The subsequent sublists contain the coordinates
            of any cutouts within the polygon. The contents of this column must be a List[List[float]].
        categories_column_name: Column name containing the predefined classes or labels used
            by the model to classify the detected objects. The contents of this column must be List[str].
        scores_column_name: Column name containing the confidence scores that the
            model assigns to it's predictions, indicating how certain the model is that the predicted
            class is contained within the bounding box. This argument is only applicable for prediction
            values. The contents of this column must be List[float].
        bounding_boxes_coordinates_column_name: Column name containing the coordinates of the
            rectangular outline that locates an object within an image or video. Pascal VOC format
            required. The contents of this column must be a List[List[float]].
    """

    polygon_coordinates_column_name: str
    categories_column_name: str
    scores_column_name: str | None = None
    bounding_boxes_coordinates_column_name: str | None = None


class InstanceSegmentationActualColumnNames(NamedTuple):
    """Used to log instance segmentation actual values that are assigned to the actual schema parameter.

    Args:
        polygon_coordinates_column_name: Column name containing the coordinates of the
            polygon that locates an object within an image or video. The contents of this column
            must be a List[List[float]].
        categories_column_name: Column name containing the predefined classes or labels used
            by the model to classify the detected objects. The contents of this column must be List[str].
        bounding_boxes_coordinates_column_name: Column name containing the coordinates of the
            rectangular outline that locates an object within an image or video. Pascal VOC format
            required. The contents of this column must be a List[List[float]].
    """

    polygon_coordinates_column_name: str
    categories_column_name: str
    bounding_boxes_coordinates_column_name: str | None = None


class ObjectDetectionLabel(NamedTuple):
    """Label data for object detection tasks with bounding boxes and categories."""

    bounding_boxes_coordinates: list[list[float]]
    categories: list[str]
    # Actual Object Detection Labels won't have scores
    scores: list[float] | None = None

    def validate(self, prediction_or_actual: str) -> None:
        """Validate the object detection label fields and constraints."""
        # Validate bounding boxes
        self._validate_bounding_boxes_coordinates()
        # Validate categories
        self._validate_categories()
        # Validate scores
        self._validate_scores(prediction_or_actual)
        # Validate we have the same number of bounding boxes, categories and scores
        self._validate_count_match()

    def _validate_bounding_boxes_coordinates(self) -> None:
        if not is_list_of(self.bounding_boxes_coordinates, list):
            raise TypeError(
                "Object Detection Label bounding boxes must be a list of lists of floats"
            )
        for coordinates in self.bounding_boxes_coordinates:
            _validate_bounding_box_coordinates(coordinates)

    def _validate_categories(self) -> None:
        # Allows for categories as empty strings
        if not is_list_of(self.categories, str):
            raise TypeError(
                "Object Detection Label categories must be a list of strings"
            )

    def _validate_scores(self, prediction_or_actual: str) -> None:
        if self.scores is None:
            if prediction_or_actual == "prediction":
                raise ValueError(
                    "Bounding box confidence scores must not be None for predictions"
                )
        else:
            if prediction_or_actual == "actual":
                raise ValueError(
                    "Bounding box confidence scores must be None for actuals"
                )

            if not is_list_of(self.scores, float):
                raise TypeError(
                    "Object Detection Label scores must be a list of floats"
                )
            if any(score > 1 or score < 0 for score in self.scores):
                raise ValueError(
                    f"Bounding box confidence scores must be between 0 and 1, inclusive. Found "
                    f"{self.scores}"
                )

    def _validate_count_match(self) -> None:
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


class SemanticSegmentationLabel(NamedTuple):
    """Label data for semantic segmentation with polygon coordinates and categories."""

    polygon_coordinates: list[list[float]]
    categories: list[str]

    def validate(self) -> None:
        """Validate the field values and constraints."""
        # Validate polygon coordinates
        self._validate_polygon_coordinates()
        # Validate categories
        self._validate_categories()
        # Validate we have the same number of polygon coordinates and categories
        self._validate_count_match()

    def _validate_polygon_coordinates(self) -> None:
        _validate_polygon_coordinates(self.polygon_coordinates)

    def _validate_categories(self) -> None:
        # Allows for categories as empty strings
        if not is_list_of(self.categories, str):
            raise TypeError(
                "Semantic Segmentation Label categories must be a list of strings"
            )

    def _validate_count_match(self) -> None:
        n_polygon_coordinates = len(self.polygon_coordinates)
        if n_polygon_coordinates == 0:
            raise ValueError(
                f"Semantic Segmentation Labels must contain at least 1 polygon. Found"
                f" {n_polygon_coordinates}."
            )

        n_categories = len(self.categories)
        if n_polygon_coordinates != n_categories:
            raise ValueError(
                "Semantic Segmentation Labels must contain the same number of polygons and "
                f"categories. Found {n_polygon_coordinates} polygons and {n_categories} "
                "categories."
            )


class InstanceSegmentationPredictionLabel(NamedTuple):
    """Prediction label for instance segmentation with polygons and category information."""

    polygon_coordinates: list[list[float]]
    categories: list[str]
    scores: list[float] | None = None
    bounding_boxes_coordinates: list[list[float]] | None = None

    def validate(self) -> None:
        """Validate the field values and constraints."""
        # Validate polygon coordinates
        self._validate_polygon_coordinates()
        # Validate categories
        self._validate_categories()
        # Validate scores
        self._validate_scores()
        # Validate bounding boxes
        self._validate_bounding_boxes()
        # Validate we have the same number of polygon coordinates and categories
        self._validate_count_match()

    def _validate_polygon_coordinates(self) -> None:
        _validate_polygon_coordinates(self.polygon_coordinates)

    def _validate_categories(self) -> None:
        # Allows for categories as empty strings
        if not is_list_of(self.categories, str):
            raise TypeError(
                "Instance Segmentation Prediction Label categories must be a list of strings"
            )

    def _validate_scores(self) -> None:
        if self.scores is not None:
            if not is_list_of(self.scores, float):
                raise TypeError(
                    "Instance Segmentation Prediction Label confidence scores must be a list of floats"
                )
            if any(score > 1 or score < 0 for score in self.scores):
                raise ValueError(
                    "Instance Segmentation Prediction Label confidence scores must "
                    "be between 0 and 1, inclusive. Found "
                    f"{self.scores}"
                )

    def _validate_bounding_boxes(self) -> None:
        if self.bounding_boxes_coordinates is not None:
            if not is_list_of(self.bounding_boxes_coordinates, list):
                raise TypeError(
                    "Instance Segmentation Prediction Label bounding boxes must be a list of lists of floats"
                )
            for coordinates in self.bounding_boxes_coordinates:
                _validate_bounding_box_coordinates(coordinates)

    def _validate_count_match(self) -> None:
        n_polygon_coordinates = len(self.polygon_coordinates)
        if n_polygon_coordinates == 0:
            raise ValueError(
                f"Instance Segmentation Prediction Labels must contain at least 1 polygon. Found"
                f" {n_polygon_coordinates}."
            )

        n_categories = len(self.categories)
        if n_polygon_coordinates != n_categories:
            raise ValueError(
                "Instance Segmentation Prediction Labels must contain the same number "
                f"of polygons and categories. Found {n_polygon_coordinates} polygons "
                f"and {n_categories} categories."
            )
        if self.scores is not None:
            n_scores = len(self.scores)
            if n_polygon_coordinates != n_scores:
                raise ValueError(
                    "Instance Segmentation Prediction Labels must contain the same "
                    f"number of scores and polygons. Found {n_polygon_coordinates} "
                    f"polygons and {n_scores} scores."
                )

        if self.bounding_boxes_coordinates is not None:
            n_bounding_boxes = len(self.bounding_boxes_coordinates)
            if n_polygon_coordinates != n_bounding_boxes:
                raise ValueError(
                    "Instance Segmentation Prediction Labels must contain the same number "
                    f"of bounding boxes and polygons. Found {n_polygon_coordinates} polygons "
                    f"and {n_bounding_boxes} bounding boxes."
                )


class InstanceSegmentationActualLabel(NamedTuple):
    """Actual label for instance segmentation with polygon coordinates and categories."""

    polygon_coordinates: list[list[float]]
    categories: list[str]
    bounding_boxes_coordinates: list[list[float]] | None = None

    def validate(self) -> None:
        """Validate the field values and constraints."""
        # Validate polygon coordinates
        self._validate_polygon_coordinates()
        # Validate categories
        self._validate_categories()
        # Validate bounding boxes
        self._validate_bounding_boxes()
        # Validate we have the same number of polygon coordinates and categories
        self._validate_count_match()

    def _validate_polygon_coordinates(self) -> None:
        _validate_polygon_coordinates(self.polygon_coordinates)

    def _validate_categories(self) -> None:
        # Allows for categories as empty strings
        if not is_list_of(self.categories, str):
            raise TypeError(
                "Instance Segmentation Actual Label categories must be a list of strings"
            )

    def _validate_bounding_boxes(self) -> None:
        if self.bounding_boxes_coordinates is not None:
            if not is_list_of(self.bounding_boxes_coordinates, list):
                raise TypeError(
                    "Instance Segmentation Actual Label bounding boxes must be a list of lists of floats"
                )
            for coordinates in self.bounding_boxes_coordinates:
                _validate_bounding_box_coordinates(coordinates)

    def _validate_count_match(self) -> None:
        n_polygon_coordinates = len(self.polygon_coordinates)
        if n_polygon_coordinates == 0:
            raise ValueError(
                f"Instance Segmentation Actual Labels must contain at least 1 polygon. Found"
                f" {n_polygon_coordinates}."
            )

        n_categories = len(self.categories)
        if n_polygon_coordinates != n_categories:
            raise ValueError(
                "Instance Segmentation Actual Labels must contain the same number of polygons and "
                f"categories. Found {n_polygon_coordinates} polygons and {n_categories} "
                "categories."
            )

        if self.bounding_boxes_coordinates is not None:
            n_bounding_boxes = len(self.bounding_boxes_coordinates)
            if n_polygon_coordinates != n_bounding_boxes:
                raise ValueError(
                    "Instance Segmentation Actual Labels must contain the same number of bounding boxes and "
                    f"polygons. Found {n_polygon_coordinates} polygons and {n_bounding_boxes} "
                    "bounding boxes."
                )


class MultiClassPredictionLabel(NamedTuple):
    """Used to log multi class prediction label.

    Args:
        prediction_scores: The prediction scores of the classes.
        threshold_scores: The threshold scores of the classes.
            Only Multi Label will have threshold scores.
    """

    prediction_scores: dict[str, float | int]
    threshold_scores: dict[str, float | int] | None = None

    def validate(self) -> None:
        """Validate the field values and constraints."""
        # Validate scores
        self._validate_prediction_scores()
        self._validate_threshold_scores()

    def _validate_prediction_scores(self) -> None:
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
        if (
            n_prediction_scores == 0
            or n_prediction_scores > MAX_NUMBER_OF_MULTI_CLASS_CLASSES
        ):
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

    def _validate_threshold_scores(self) -> None:
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
    """Used to log multi class actual label.

    Args:
        actual_scores: The actual scores of the classes.
            Any class in actual_scores with a score of 1 will be sent to arize.
    """

    actual_scores: dict[str, float | int]

    def validate(self) -> None:
        """Validate the field values and constraints."""
        # Validate scores
        self._validate_actual_scores()

    def _validate_actual_scores(self) -> None:
        if not is_dict_of(
            self.actual_scores,
            key_allowed_types=str,
            value_allowed_types=(int, float),
        ):
            raise ValueError(
                "Multi-Class Actual Scores must be a dictionary with keys of type str "
                "and values must be a numeric type (int or float)."
            )
        n_actual_scores = len(self.actual_scores)
        if (
            n_actual_scores == 0
            or n_actual_scores > MAX_NUMBER_OF_MULTI_CLASS_CLASSES
        ):
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
    """Prediction label for ranking tasks with group and rank information."""

    group_id: str
    rank: int
    score: float | None = None
    label: str | None = None

    def validate(self) -> None:
        """Validate the field values and constraints."""
        # Validate existence of required fields: prediction_group_id and rank
        if self.group_id is None or self.rank is None:
            raise ValueError(
                "RankingPredictionLabel must contain: group_id and rank"
            )
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

    def _validate_group_id(self) -> None:
        if not isinstance(self.group_id, str):
            raise TypeError("Prediction Group ID must be a string")
        if not (1 <= len(self.group_id) <= 36):
            raise ValueError(
                f"Prediction Group ID must have length between 1 and 36. Found {len(self.group_id)}"
            )

    def _validate_rank(self) -> None:
        if not isinstance(self.rank, int):
            raise TypeError("Prediction Rank must be an int")
        if not (1 <= self.rank <= 100):
            raise ValueError(
                f"Prediction Rank must be between 1 and 100, inclusive. Found {self.rank}"
            )

    def _validate_label(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("Prediction Label must be a str")
        if self.label == "":
            raise ValueError("Prediction Label must not be an empty string.")

    def _validate_score(self) -> None:
        if not isinstance(self.score, (float, int)):
            raise TypeError("Prediction Score must be a float or an int")


class RankingActualLabel(NamedTuple):
    """Actual label for ranking tasks with relevance information."""

    relevance_labels: list[str] | None = None
    relevance_score: float | None = None

    def validate(self) -> None:
        """Validate the field values and constraints."""
        # Validate relevance_labels type
        if self.relevance_labels is not None:
            self._validate_relevance_labels(self.relevance_labels)
        # Validate relevance score type
        if self.relevance_score is not None:
            self._validate_relevance_score(self.relevance_score)

    @staticmethod
    def _validate_relevance_labels(relevance_labels: list[str]) -> None:
        """Validate relevance labels.

        Args:
            relevance_labels: List of relevance labels to validate.

        Raises:
            TypeError: If relevance_labels is not a list of strings.
            ValueError: If any label is an empty string.
        """
        if not is_list_of(relevance_labels, str):
            raise TypeError("Actual Relevance Labels must be a list of strings")
        if any(label == "" for label in relevance_labels):
            raise ValueError(
                "Actual Relevance Labels must be not contain empty strings"
            )

    @staticmethod
    def _validate_relevance_score(relevance_score: float) -> None:
        """Validate relevance score.

        Args:
            relevance_score: Relevance score to validate.

        Raises:
            TypeError: If relevance_score is not a float or int.
        """
        if not isinstance(relevance_score, (float, int)):
            raise TypeError("Actual Relevance score must be a float or an int")


@dataclass
class PromptTemplateColumnNames:
    """Column names for prompt template configuration in LLM schemas."""

    template_column_name: str | None = None
    template_version_column_name: str | None = None

    def __iter__(self) -> Iterator[str | None]:
        """Iterate over the prompt template column names."""
        return iter(
            (self.template_column_name, self.template_version_column_name)
        )


@dataclass
class LLMConfigColumnNames:
    """Column names for LLM configuration parameters in schemas."""

    model_column_name: str | None = None
    params_column_name: str | None = None

    def __iter__(self) -> Iterator[str | None]:
        """Iterate over the LLM config column names."""
        return iter((self.model_column_name, self.params_column_name))


@dataclass
class LLMRunMetadataColumnNames:
    """Column names for LLM run metadata fields in schemas."""

    total_token_count_column_name: str | None = None
    prompt_token_count_column_name: str | None = None
    response_token_count_column_name: str | None = None
    response_latency_ms_column_name: str | None = None

    def __iter__(self) -> Iterator[str | None]:
        """Iterate over the LLM run metadata column names."""
        return iter(
            (
                self.total_token_count_column_name,
                self.prompt_token_count_column_name,
                self.response_token_count_column_name,
                self.response_latency_ms_column_name,
            )
        )


@dataclass
class SimilarityReference:
    """Reference to a prediction for similarity search operations."""

    prediction_id: str
    reference_column_name: str
    prediction_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Validate similarity reference fields after initialization.

        Raises:
            ValueError: If prediction_id or reference_column_name is empty.
            TypeError: If prediction_timestamp is not a datetime object.
        """
        if self.prediction_id == "":
            raise ValueError("prediction id cannot be empty")
        if self.reference_column_name == "":
            raise ValueError("Reference column name cannot be empty")
        if self.prediction_timestamp and not isinstance(
            self.prediction_timestamp, datetime
        ):
            raise TypeError("prediction_timestamp must be a datetime object")


@dataclass
class SimilaritySearchParams:
    """Parameters for configuring similarity search operations."""

    references: list[SimilarityReference]
    search_column_name: str
    threshold: float = 0

    def __post_init__(self) -> None:
        """Validate similarity search parameters after initialization.

        Raises:
            ValueError: If references list is invalid, search_column_name is
                empty, or threshold is out of range.
            TypeError: If any reference is not a SimilarityReference instance.
        """
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
                raise TypeError(
                    "all references must be instances of SimilarityReference"
                )


@dataclass(frozen=True)
class BaseSchema:
    """Base class for all schema definitions with immutable fields."""

    def replace(self, **changes: object) -> Self:
        """Return a new instance with specified fields replaced."""
        return replace(self, **changes)

    def asdict(self) -> dict[str, str]:
        """Convert the schema to a dictionary."""
        return asdict(self)

    def get_used_columns(self) -> set[str]:
        """Return the set of column names used in this schema."""
        return set(self.get_used_columns_counts().keys())

    def get_used_columns_counts(self) -> dict[str, int]:
        """Return a dict mapping column names to their usage count."""
        raise NotImplementedError()


@dataclass(frozen=True)
class TypedColumns:
    """Optional class used for explicit type enforcement of feature and tag columns in the dataframe.

    When initializing a Schema, use TypedColumns in place of a list of string column names::

        feature_column_names = TypedColumns(
            inferred=["feature_1", "feature_2"],
            to_str=["feature_3"],
            to_int=["feature_4"],
        )

    Notes:
        - If a TypedColumns object is included in a Schema, pandas version 1.0.0 or higher is required.
        - Pandas StringDType is still considered an experimental field.
        - Columns not present in any field will not be captured in the Schema.
        - StringDType, Int64DType, and Float64DType are all nullable column types.
          Null values will be ingested and represented in Arize as empty values.
    """

    inferred: list[str] | None = None
    to_str: list[str] | None = None
    to_int: list[str] | None = None
    to_float: list[str] | None = None

    def get_all_column_names(self) -> list[str]:
        """Return all column names across all conversion lists."""
        return list(chain.from_iterable(filter(None, self.__dict__.values())))

    def has_duplicate_columns(self) -> tuple[bool, set[str]]:
        """Check for duplicate columns and return (has_duplicates, duplicate_set)."""
        # True if there are duplicates within a field's list or across fields.
        # Return a set of the duplicate column names.
        cols = self.get_all_column_names()
        duplicates = {x for x in cols if cols.count(x) > 1}
        return len(duplicates) > 0, duplicates

    def is_empty(self) -> bool:
        """Return True if no columns are configured for conversion."""
        return not self.get_all_column_names()


@dataclass(frozen=True)
class Schema(BaseSchema):
    """Used to organize and map column names containing model data within your Pandas dataframe to Arize.

    Args:
        prediction_id_column_name: Column name for the predictions unique identifier.
            Unique IDs are used to match a prediction to delayed actuals or feature importances in Arize.
            If prediction ids are not provided, it will default to an empty string "" and, when possible,
            Arize will create a random prediction id on the server side. Prediction id must be a string column
            with each row indicating a unique prediction event.
        feature_column_names: Column names for features.
            The content of feature columns can be int, float, string. If TypedColumns is used,
            the columns will be cast to the provided types prior to logging.
        tag_column_names: Column names for tags. The content of tag
            columns can be int, float, string. If TypedColumns is used,
            the columns will be cast to the provided types prior to logging.
        timestamp_column_name: Column name for timestamps. The content of this
            column must be int Unix Timestamps in seconds.
        prediction_label_column_name: Column name for categorical prediction values.
            The content of this column must be convertible to string.
        prediction_score_column_name: Column name for numeric prediction values. The
            content of this column must be int/float or list of dictionaries mapping class names to
            int/float scores in the case of MULTI_CLASS model types.
        actual_label_column_name: Column name for categorical ground truth values.
            The content of this column must be convertible to string.
        actual_score_column_name: Column name for numeric ground truth values. The
            content of this column must be int/float or list of dictionaries mapping class names to
            int/float scores in the case of MULTI_CLASS model types.
        shap_values_column_names: Dictionary mapping feature column name
            and corresponding SHAP feature importance column name. e.g.
            {{"feat_A": "feat_A_shap", "feat_B": "feat_B_shap"}}
        embedding_feature_column_names: Dictionary
            mapping embedding display names to EmbeddingColumnNames objects.
        prediction_group_id_column_name: Column name for ranking groups or lists in
            ranking models. The content of this column must be string and is limited to 128 characters.
        rank_column_name: Column name for rank of each element on the its group or
            list. The content of this column must be integer between 1-100.
        relevance_score_column_name: Column name for ranking model type numeric
            ground truth values. The content of this column must be int/float.
        relevance_labels_column_name: Column name for ranking model type categorical
            ground truth values. The content of this column must be a string.
        object_detection_prediction_column_names:
            ObjectDetectionColumnNames object containing information defining the predicted bounding
            boxes' coordinates, categories, and scores.
        object_detection_actual_column_names:
            ObjectDetectionColumnNames object containing information defining the actual bounding
            boxes' coordinates, categories, and scores.
        prompt_column_names: column names for text that is passed
            to the GENERATIVE_LLM model. It accepts a string (if sending only a text column) or
            EmbeddingColumnNames object containing the embedding vector data (required) and raw text
            (optional) for the input text your model acts on.
        response_column_names: column names for text generated by
            the GENERATIVE_LLM model. It accepts a string (if sending only a text column) or
            EmbeddingColumnNames object containing the embedding vector data (required) and raw text
            (optional) for the text your model generates.
        prompt_template_column_names: PromptTemplateColumnNames object
            containing the prompt template and the prompt template version.
        llm_config_column_names: LLMConfigColumnNames object containing
            the LLM's model name and its hyper parameters used at inference.
        llm_run_metadata_column_names: LLMRunMetadataColumnNames
            object containing token counts and latency metrics
        retrieved_document_ids_column_name: Column name for retrieved document ids.
            The content of this column must be lists with entries convertible to strings.
        multi_class_threshold_scores_column_name:
            Column name for dictionary that maps class names to threshold values. The
            content of this column must be dictionary of str -> int/float.
        semantic_segmentation_prediction_column_names:
            SemanticSegmentationColumnNames object containing information defining the predicted
            polygon coordinates and categories.
        semantic_segmentation_actual_column_names:
            SemanticSegmentationColumnNames object containing information defining the actual
            polygon coordinates and categories.
        instance_segmentation_prediction_column_names:
            InstanceSegmentationPredictionColumnNames object containing information defining the predicted
            polygon coordinates, categories, scores, and bounding box coordinates.
        instance_segmentation_actual_column_names:
            InstanceSegmentationActualColumnNames object containing information defining the actual
            polygon coordinates, categories, scores, and bounding box coordinates.
    """

    prediction_id_column_name: str | None = None
    feature_column_names: list[str] | TypedColumns | None = None
    tag_column_names: list[str] | TypedColumns | None = None
    timestamp_column_name: str | None = None
    prediction_label_column_name: str | None = None
    prediction_score_column_name: str | None = None
    actual_label_column_name: str | None = None
    actual_score_column_name: str | None = None
    shap_values_column_names: dict[str, str] | None = None
    embedding_feature_column_names: dict[str, EmbeddingColumnNames] | None = (
        None
    )
    prediction_group_id_column_name: str | None = None
    rank_column_name: str | None = None
    attributions_column_name: str | None = None
    relevance_score_column_name: str | None = None
    relevance_labels_column_name: str | None = None
    object_detection_prediction_column_names: (
        ObjectDetectionColumnNames | None
    ) = None
    object_detection_actual_column_names: ObjectDetectionColumnNames | None = (
        None
    )
    prompt_column_names: str | EmbeddingColumnNames | None = None
    response_column_names: str | EmbeddingColumnNames | None = None
    prompt_template_column_names: PromptTemplateColumnNames | None = None
    llm_config_column_names: LLMConfigColumnNames | None = None
    llm_run_metadata_column_names: LLMRunMetadataColumnNames | None = None
    retrieved_document_ids_column_name: str | None = None
    multi_class_threshold_scores_column_name: str | None = None
    semantic_segmentation_prediction_column_names: (
        SemanticSegmentationColumnNames | None
    ) = None
    semantic_segmentation_actual_column_names: (
        SemanticSegmentationColumnNames | None
    ) = None
    instance_segmentation_prediction_column_names: (
        InstanceSegmentationPredictionColumnNames | None
    ) = None
    instance_segmentation_actual_column_names: (
        InstanceSegmentationActualColumnNames | None
    ) = None

    def get_used_columns_counts(self) -> dict[str, int]:
        """Return a dict mapping column names to their usage count."""
        columns_used_counts: dict[str, int] = {}

        for field in self.__dataclass_fields__:
            if field.endswith("column_name"):
                col = getattr(self, field)
                if col is not None:
                    add_to_column_count_dictionary(columns_used_counts, col)

        if self.feature_column_names is not None:
            for col in _normalize_column_names(self.feature_column_names):
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
                        columns_used_counts,
                        emb_col_names.link_to_data_column_name,
                    )

        if self.tag_column_names is not None:
            for col in _normalize_column_names(self.tag_column_names):
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
                add_to_column_count_dictionary(
                    columns_used_counts, self.prompt_column_names
                )
            elif isinstance(self.prompt_column_names, EmbeddingColumnNames):
                add_to_column_count_dictionary(
                    columns_used_counts,
                    self.prompt_column_names.vector_column_name,
                )
                if self.prompt_column_names.data_column_name is not None:
                    add_to_column_count_dictionary(
                        columns_used_counts,
                        self.prompt_column_names.data_column_name,
                    )

        if self.response_column_names is not None:
            if isinstance(self.response_column_names, str):
                add_to_column_count_dictionary(
                    columns_used_counts, self.response_column_names
                )
            elif isinstance(self.response_column_names, EmbeddingColumnNames):
                add_to_column_count_dictionary(
                    columns_used_counts,
                    self.response_column_names.vector_column_name,
                )
                if self.response_column_names.data_column_name is not None:
                    add_to_column_count_dictionary(
                        columns_used_counts,
                        self.response_column_names.data_column_name,
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

        if self.semantic_segmentation_prediction_column_names is not None:
            for col in self.semantic_segmentation_prediction_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.semantic_segmentation_actual_column_names is not None:
            for col in self.semantic_segmentation_actual_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.instance_segmentation_prediction_column_names is not None:
            for col in self.instance_segmentation_prediction_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        if self.instance_segmentation_actual_column_names is not None:
            for col in self.instance_segmentation_actual_column_names:
                add_to_column_count_dictionary(columns_used_counts, col)

        return columns_used_counts

    def has_prediction_columns(self) -> bool:
        """Return True if prediction columns are configured."""
        prediction_cols = (
            self.prediction_label_column_name,
            self.prediction_score_column_name,
            self.rank_column_name,
            self.prediction_group_id_column_name,
            self.object_detection_prediction_column_names,
            self.semantic_segmentation_prediction_column_names,
            self.instance_segmentation_prediction_column_names,
            self.multi_class_threshold_scores_column_name,
        )
        return any(col is not None for col in prediction_cols)

    def has_actual_columns(self) -> bool:
        """Return True if actual label columns are configured."""
        actual_cols = (
            self.actual_label_column_name,
            self.actual_score_column_name,
            self.relevance_labels_column_name,
            self.relevance_score_column_name,
            self.object_detection_actual_column_names,
            self.semantic_segmentation_actual_column_names,
            self.instance_segmentation_actual_column_names,
        )
        return any(col is not None for col in actual_cols)

    def has_feature_importance_columns(self) -> bool:
        """Return True if feature importance columns are configured."""
        feature_importance_cols = (self.shap_values_column_names,)
        return any(col is not None for col in feature_importance_cols)

    def has_typed_columns(self) -> bool:
        """Return True if typed columns are configured."""
        return any(self.typed_column_fields())

    def typed_column_fields(self) -> set[str]:
        """Return the set of field names with typed columns."""
        return {
            field
            for field in self.__dataclass_fields__
            if isinstance(getattr(self, field), TypedColumns)
        }

    def is_delayed(self) -> bool:
        """Check if the schema has inherently latent information.

        Determines this based on the columns provided by the user.

        Returns:
            bool: True if the schema is "delayed", i.e., does not possess prediction
            columns and has actual or feature importance columns.
        """
        return (
            self.has_actual_columns() or self.has_feature_importance_columns()
        ) and not self.has_prediction_columns()


@dataclass(frozen=True)
class CorpusSchema(BaseSchema):
    """Schema for corpus data with document identification and content columns."""

    document_id_column_name: str | None = None
    document_version_column_name: str | None = None
    document_text_embedding_column_names: EmbeddingColumnNames | None = None

    def get_used_columns_counts(self) -> dict[str, int]:
        """Return a dict mapping column names to their usage count."""
        columns_used_counts: dict[str, int] = {}

        if self.document_id_column_name is not None:
            add_to_column_count_dictionary(
                columns_used_counts, self.document_id_column_name
            )
        if self.document_version_column_name is not None:
            add_to_column_count_dictionary(
                columns_used_counts, self.document_version_column_name
            )
        if self.document_text_embedding_column_names is not None:
            add_to_column_count_dictionary(
                columns_used_counts,
                self.document_text_embedding_column_names.vector_column_name,
            )
            if (
                self.document_text_embedding_column_names.data_column_name
                is not None
            ):
                add_to_column_count_dictionary(
                    columns_used_counts,
                    self.document_text_embedding_column_names.data_column_name,
                )
            if (
                self.document_text_embedding_column_names.link_to_data_column_name
                is not None
            ):
                add_to_column_count_dictionary(
                    columns_used_counts,
                    self.document_text_embedding_column_names.link_to_data_column_name,
                )
        return columns_used_counts


@unique
class ArizeTypes(Enum):
    """Enum representing supported data types in Arize platform."""

    STR = 0
    FLOAT = 1
    INT = 2


@dataclass(frozen=True)
class TypedValue:
    """Container for a value with its associated Arize type."""

    type: ArizeTypes
    value: str | bool | float | int


def _count_characters_raw_data(data: str | list[str]) -> int:
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


def add_to_column_count_dictionary(
    column_dictionary: dict[str, int], col: str | None
) -> None:
    """Increment the count for a column name in a dictionary.

    Args:
        column_dictionary: Dictionary mapping column names to counts.
        col: The column name to increment, or :obj:`None` to skip.
    """
    if col:
        if col in column_dictionary:
            column_dictionary[col] += 1
        else:
            column_dictionary[col] = 1


def _validate_bounding_box_coordinates(
    bounding_box_coordinates: list[float],
) -> None:
    if not is_list_of(bounding_box_coordinates, float):
        raise TypeError(
            "Each bounding box's coordinates must be a lists of floats"
        )
    # Format must be (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
    if len(bounding_box_coordinates) != 4:
        raise ValueError(
            "Each bounding box's coordinates must be a collection of 4 floats. Found "
            f"{bounding_box_coordinates}"
        )
    if any(coord < 0 for coord in bounding_box_coordinates):
        raise ValueError(
            f"Bounding box's coordinates cannot be negative. Found {bounding_box_coordinates}"
        )
    if not (bounding_box_coordinates[2] > bounding_box_coordinates[0]):
        raise ValueError(
            "Each bounding box bottom-right X coordinate should be larger than the "
            f"top-left. Found {bounding_box_coordinates}"
        )
    if not (bounding_box_coordinates[3] > bounding_box_coordinates[1]):
        raise ValueError(
            "Each bounding box bottom-right Y coordinate should be larger than the "
            f"top-left. Found {bounding_box_coordinates}"
        )

    return


def _validate_polygon_coordinates(
    polygon_coordinates: list[list[float]],
) -> None:
    if not is_list_of(polygon_coordinates, list):
        raise TypeError("Polygon coordinates must be a list of lists of floats")
    for coordinates in polygon_coordinates:
        if not is_list_of(coordinates, float):
            raise TypeError(
                "Each polygon's coordinates must be a lists of floats"
            )
        if len(coordinates) < 6:
            raise ValueError(
                "Each polygon's coordinates must be a collection of at least 6 "
                "floats (3 pairs of x, y coordinates). Received coordinates: "
                f"{coordinates}"
            )
        if len(coordinates) % 2 != 0:
            raise ValueError(
                "Each polygon's coordinates must be a collection of an even number "
                "of floats (2 * n pairs of x, y coordinates). Received coordinates: "
                f"{coordinates}"
            )
        if any(coord < 0 for coord in coordinates):
            raise ValueError(
                "Polygon's coordinates cannot be negative. Received coordinates: "
                f"{coordinates}"
            )

        # Validate polygon is well-formed (no repeated vertices, no self-intersections)
        points = [
            (coordinates[i], coordinates[i + 1])
            for i in range(0, len(coordinates), 2)
        ]

        # Check for repeated vertices. Also, create edges for later intersection checks
        edges = []
        for i in range(len(points)):
            if any(
                points[i] == points[j]
                for i in range(len(points))
                for j in range(i + 1, len(points))
            ):
                raise ValueError(
                    "Polygon's coordinates cannot have repeated vertices. Received coordinates: "
                    f"{coordinates}"
                )
            edges.append((points[i], points[(i + 1) % len(points)]))

        # Check for self-intersections
        for i in range(len(edges)):
            for j in range(i + 2, len(edges)):
                # Skip adjacent edges
                if i == 0 and j == len(edges) - 1:
                    continue

                # Check if edges intersect
                if segments_intersect(
                    edges[i][0], edges[i][1], edges[j][0], edges[j][1]
                ):
                    raise ValueError(
                        "Polygon's coordinates cannot have self-intersections. Received coordinates: "
                        f"{coordinates}"
                    )

    return


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Check if two line segments intersect.

    Args:
        p1: First endpoint of the first line segment (x,y)
        p2: Second endpoint of the first line segment (x,y)
        p3: First endpoint of the second line segment (x,y)
        p4: Second endpoint of the second line segment (x,y)

    Returns:
        True if the line segments intersect, False otherwise
    """

    # Function to calculate direction
    def orientation(
        p: tuple[float, float],
        q: tuple[float, float],
        r: tuple[float, float],
    ) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    # Function to check if point q is on segment pr
    def on_segment(
        p: tuple[float, float],
        q: tuple[float, float],
        r: tuple[float, float],
    ) -> bool:
        return (
            q[0] <= max(p[0], r[0])
            and q[0] >= min(p[0], r[0])
            and q[1] <= max(p[1], r[1])
            and q[1] >= min(p[1], r[1])
        )

    # Calculate directions
    o1 = orientation(p3, p4, p1)
    o2 = orientation(p3, p4, p2)
    o3 = orientation(p1, p2, p3)
    o4 = orientation(p1, p2, p4)

    # Check for general intersection
    if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and (
        (o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)
    ):
        return True

    # Check for special cases where points are collinear
    if o1 == 0 and on_segment(p3, p1, p4):
        return True
    if o2 == 0 and on_segment(p3, p2, p4):
        return True
    if o3 == 0 and on_segment(p1, p3, p2):
        return True
    return o4 == 0 and on_segment(p1, p4, p2)


@unique
class StatusCodes(Enum):
    """Enum representing status codes for operations and responses."""

    UNSET = 0
    OK = 1
    ERROR = 2

    @classmethod
    def list_codes(cls) -> list[str]:
        """Return a list of all status code names."""
        return [t.name for t in cls]


def convert_element(value: object) -> object:
    """Converts scalar or array to python native."""
    val = getattr(value, "tolist", lambda: value)()
    # Check if it's a list since elements from pd indices are converted to a
    # scalar whereas pd series/dataframe elements are converted to list of 1
    # with the native value
    if isinstance(val, list):
        val = val[0] if val else None
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Decimal) and value.is_nan():
        return None
    return val


PredictionLabelTypes = (
    str
    | bool
    | int
    | float
    | tuple[str, float]
    | ObjectDetectionLabel
    | RankingPredictionLabel
    | MultiClassPredictionLabel
)

ActualLabelTypes = (
    str
    | bool
    | int
    | float
    | tuple[str, float]
    | ObjectDetectionLabel
    | RankingActualLabel
    | MultiClassActualLabel
)

PredictionIDType = str | int | float
