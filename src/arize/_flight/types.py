from enum import Enum


class FlightRequestType(str, Enum):
    EVALUATION = "evaluation"
    ANNOTATION = "annotation"
    METADATA = "metadata"
    LOG_EXPERIMENT_DATA = "log_experiment_data"
