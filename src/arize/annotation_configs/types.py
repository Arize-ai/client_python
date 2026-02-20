"""Types for annotation configs."""

from enum import Enum


class AnnotationConfigType(Enum):
    """Supported annotation config types."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    FREEFORM = "freeform"
