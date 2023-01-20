from dataclasses import dataclass
from enum import Enum, auto, unique


@unique
class NLPUseCases(Enum):
    SEQUENCE_CLASSIFICATION = auto()


@unique
class CVUseCases(Enum):
    IMAGE_CLASSIFICATION = auto()


@dataclass
class UseCases:
    NLP = NLPUseCases
    CV = CVUseCases
