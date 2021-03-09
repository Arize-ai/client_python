from enum import Enum, unique


@unique
class ModelTypes(Enum):
    BINARY = 1
    NUMERIC = 2
    CATEGORICAL = 3
    SCORE_CATEGORICAL = 4
