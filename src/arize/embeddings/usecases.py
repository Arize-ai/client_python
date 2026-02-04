"""Use case definitions and enums for embedding generation."""

from dataclasses import dataclass
from enum import Enum, auto, unique


@unique
class NLPUseCases(Enum):
    """Enum representing supported NLP use cases for embedding generation."""

    SEQUENCE_CLASSIFICATION = auto()
    SUMMARIZATION = auto()


@unique
class CVUseCases(Enum):
    """Enum representing supported computer vision use cases for embedding generation."""

    IMAGE_CLASSIFICATION = auto()
    OBJECT_DETECTION = auto()


@unique
class TabularUseCases(Enum):
    """Enum representing supported tabular/structured data use cases for embedding generation."""

    TABULAR_EMBEDDINGS = auto()


@dataclass
class UseCases:
    """Container grouping all use case enums for embedding generators."""

    NLP = NLPUseCases
    CV = CVUseCases
    STRUCTURED = TabularUseCases
