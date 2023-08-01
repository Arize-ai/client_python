from dataclasses import dataclass
from enum import Enum, auto, unique


@unique
class NLPUseCases(Enum):
    SEQUENCE_CLASSIFICATION = auto()
    SUMMARIZATION = auto()


@unique
class CVUseCases(Enum):
    IMAGE_CLASSIFICATION = auto()
    OBJECT_DETECTION = auto()


@unique
class TabularUsecases(Enum):
    TABULAR_EMBEDDINGS = auto()


@dataclass
class UseCases:
    NLP = NLPUseCases
    CV = CVUseCases
    STRUCTURED = TabularUsecases
