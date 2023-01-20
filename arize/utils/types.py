from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd


@unique
class ModelTypes(Enum):
    NUMERIC = 1
    SCORE_CATEGORICAL = 2
    RANKING = 3
    BINARY_CLASSIFICATION = 4
    REGRESSION = 5


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


class EmbeddingColumnNames(NamedTuple):
    vector_column_name: str
    data_column_name: Optional[str] = None
    link_to_data_column_name: Optional[str] = None


class Embedding(NamedTuple):
    vector: List[float]
    data: Optional[Union[str, List[str]]] = None
    link_to_data: Optional[str] = None

    @staticmethod
    def validate_embedding_object(emb_name: Union[str, int, float], embedding) -> None:
        """Validates that the embedding object passed is of the correct format. That is,
        validations must be passed for vector, data & link_to_data.

        Args:
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            Embedding (Embedding): Embedding object

        Raises:
            TypeError: If the embedding fields are of the wrong type
        """

        if embedding.vector is not None:
            Embedding._validate_embedding_vector(emb_name, embedding.vector)

        # Validate embedding raw data, if present
        if embedding.data is not None:
            Embedding._validate_embedding_data(emb_name, embedding.data)

        # Validate embedding link to data, if present
        if embedding.link_to_data is not None:
            Embedding._validate_embedding_link_to_data(emb_name, embedding.link_to_data)

        return None

    @staticmethod
    def _validate_embedding_vector(
        emb_name: Union[str, int, float],
        vector: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        """Validates that the embedding vector passed is of the correct format. That is:
        1. Type must be list or convertible to list (like numpy arrays, pandas Series)
        2. List must not be empty
        3. Elements in list must be floats

        Args:
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            vector (List[float], np.ndarray, pd.Series): Embedding vector

        Raises:
            TypeError: If the embedding does not satisfy requirements above
        """

        if not Embedding.is_valid_iterable(vector):
            raise TypeError(
                f'Embedding feature "{emb_name}" has vector type {type(vector)}. Must be list, '
                f"np.ndarray or pd.Series"
            )
        # Fail if not all elements in list are floats
        allowed_types = (int, float, np.int16, np.int32, np.float16, np.float32)
        if not all(isinstance(val, allowed_types) for val in vector):
            raise TypeError(
                f"Embedding vector must be a vector of integers and/or floats. Got "
                f"{emb_name}.vector = {vector}"
            )

    @staticmethod
    def _validate_embedding_data(
        emb_name: Union[str, int, float],
        data: Union[str, List[str], np.ndarray, pd.Series],
    ) -> None:
        """Validates that the embedding raw data field is of the correct format. That is:
        1. Must be string or list of strings (NLP case)

        Args:
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            data (str, List[str]): Piece of text represented by a string or a token array
            represented by a list of strings

        Raises:
            TypeError: If the embedding does not satisfy requirements above
        """
        # Validate that data is a string or iterable of strings
        is_string = isinstance(data, str)
        is_allowed_iterable = not is_string and Embedding.is_valid_iterable(data)
        if not (is_string or is_allowed_iterable):
            raise TypeError(
                f'Embedding feature "{emb_name}" data field must be str, list, np.ndarray or '
                f"pd.Series"
            )

        if is_allowed_iterable:
            # Fail if not all elements in iterable are strings
            if not all(isinstance(val, str) for val in data):
                raise TypeError(f"Embedding data field must contain strings")

    @staticmethod
    def _validate_embedding_link_to_data(
        emb_name: Union[str, int, float], link_to_data: str
    ) -> None:
        """Validates that the embedding link to data field is of the correct format. That is:
        1. Must be string

        Args:
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            link_to_data (str): Link to raw data file in a blob store

        Raises:
            TypeError: If the embedding does not satisfy requirements above
        """
        if not isinstance(link_to_data, str):
            raise TypeError(
                f'Embedding feature "{emb_name}" link_to_data field must be str and got '
                f"{type(link_to_data)}"
            )

    @staticmethod
    def is_valid_iterable(data: Union[str, List[str], np.ndarray, pd.Series]) -> bool:
        """Validates that the input data field is of the correct iterable type. That is:
        1. List or
        2. numpy array or
        3. pandas Series

        Args:
            data: input iterable

        Returns:
            True if the data type is one of the accepted iterable types, false otherwise
        """
        return any(isinstance(data, t) for t in (list, np.ndarray, pd.Series))


@dataclass(frozen=True)
class Schema:
    prediction_id_column_name: str
    feature_column_names: Optional[List[str]] = None
    tag_column_names: Optional[List[str]] = None
    timestamp_column_name: Optional[str] = None
    prediction_label_column_name: Optional[str] = None
    prediction_score_column_name: Optional[str] = None
    actual_label_column_name: Optional[str] = None
    actual_score_column_name: Optional[str] = None
    shap_values_column_names: Optional[Dict[str, str]] = None
    actual_numeric_sequence_column_name: Optional[str] = None
    embedding_feature_column_names: Optional[Dict[str, EmbeddingColumnNames]] = None
    prediction_group_id_column_name: Optional[str] = None
    rank_column_name: Optional[str] = None
    attributions_column_name: Optional[str] = None
    relevance_score_column_name: Optional[str] = None
    relevance_labels_column_name: Optional[str] = None
