from enum import Enum, unique
from typing import List, Dict, Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


@unique
class ModelTypes(Enum):
    NUMERIC = 1
    SCORE_CATEGORICAL = 2
    RANKING = 3


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
                f'Embedding feature "{emb_name}" has vector type {type(vector)}. Must be list, np.ndarray or pd.Series'
            )
        # Fail if not all elements in list are floats
        allowed_types = (int, float, np.int16, np.int32, np.float16, np.float32)
        if not all(isinstance(val, allowed_types) for val in vector):
            raise TypeError(
                f"Embedding vector must be a vector of integers and/or floats. Got {emb_name}.vector = {vector}"
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
            data (str, List[str]): Piece of text represented by a string or a token array represented by a list of strings

        Raises:
            TypeError: If the embedding does not satisfy requirements above
        """
        # Validate that data is a string or iterable of strings
        is_string = isinstance(data, str)
        is_allowed_iterable = not is_string and Embedding.is_valid_iterable(data)
        if not (is_string or is_allowed_iterable):
            raise TypeError(
                f'Embedding feature "{emb_name}" data field must be str, list, np.ndarray or pd.Series'
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
                f'Embedding feature "{emb_name}" link_to_data field must be str and got {type(link_to_data)}'
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
    embedding_feature_column_names: Optional[List[EmbeddingColumnNames]] = None
    prediction_group_id_column_name: Optional[str] = None
    rank_column_name: Optional[str] = None
