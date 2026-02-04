"""Stream validation logic for ML model predictions."""

from arize.constants.ml import MAX_PREDICTION_ID_LEN, MIN_PREDICTION_ID_LEN
from arize.exceptions.parameters import (
    InvalidValueType,
)
from arize.ml.types import (
    CATEGORICAL_MODEL_TYPES,
    NUMERIC_MODEL_TYPES,
    ActualLabelTypes,
    Embedding,
    Environments,
    InstanceSegmentationActualLabel,
    InstanceSegmentationPredictionLabel,
    ModelTypes,
    MultiClassActualLabel,
    MultiClassPredictionLabel,
    ObjectDetectionLabel,
    PredictionIDType,
    PredictionLabelTypes,
    RankingActualLabel,
    RankingPredictionLabel,
    SemanticSegmentationLabel,
)


def validate_label(
    prediction_or_actual: str,
    model_type: ModelTypes,
    label: object,
    embedding_features: dict[str, Embedding] | None,
) -> None:
    """Validate a label value against the specified model type.

    Args:
        prediction_or_actual: Whether this is a "prediction" or "actual" label.
        model_type: The type of model (numeric, categorical, etc.).
        label: The label value to validate.
        embedding_features: Dictionary of embedding features for validation.

    Raises:
        ValueError: If label is invalid for the given model type.
        TypeError: If label type is incorrect.
    """
    if model_type in NUMERIC_MODEL_TYPES:
        _validate_numeric_label(model_type, label)
    elif model_type in CATEGORICAL_MODEL_TYPES:
        _validate_categorical_label(model_type, label)
    elif model_type == ModelTypes.OBJECT_DETECTION:
        _validate_cv_label(prediction_or_actual, label, embedding_features)
    elif model_type == ModelTypes.RANKING:
        _validate_ranking_label(label)
    elif model_type == ModelTypes.GENERATIVE_LLM:
        _validate_generative_llm_label(label)
    elif model_type == ModelTypes.MULTI_CLASS:
        _validate_multi_class_label(label)
    else:
        raise InvalidValueType(
            "model_type", model_type, "arize.utils.ModelTypes"
        )


def _validate_numeric_label(
    model_type: ModelTypes,
    label: object,
) -> None:
    """Validate that a label is numeric (int or float) for numeric model types.

    Args:
        model_type: The model type being validated.
        label: The label value to validate.

    Raises:
        InvalidValueType: If the label is not an int or float.
    """
    if not isinstance(label, (float, int)):
        raise InvalidValueType(
            f"label {label}",
            label,
            f"either float or int for model_type {model_type}",
        )


def _validate_categorical_label(
    model_type: ModelTypes,
    label: object,
) -> None:
    """Validate that a label is categorical (scalar or tuple with confidence) for categorical model types.

    Args:
        model_type: The model type being validated.
        label: The label value to validate.

    Raises:
        InvalidValueType: If the label is not a valid categorical type (bool, int, float, str,
            or tuple of [str/bool, float]).
    """
    is_valid = isinstance(label, (str, bool, int, float)) or (
        isinstance(label, tuple)
        and isinstance(label[0], (str, bool))
        and isinstance(label[1], float)
    )
    if not is_valid:
        raise InvalidValueType(
            f"label {label}",
            label,
            f"one of: bool, int, float, str or Tuple[str, float] for model type {model_type}",
        )


def _validate_cv_label(
    prediction_or_actual: str,
    label: object,
    embedding_features: dict[str, Embedding] | None,
) -> None:
    """Validate a computer vision label for object detection or segmentation tasks.

    Args:
        prediction_or_actual: Either 'prediction' or 'actual' to indicate label context.
        label: The CV label to validate.
        embedding_features: Dictionary of embedding features that must contain exactly one entry.

    Raises:
        InvalidValueType: If the label is not a valid CV label type.
        ValueError: If embedding_features is None or doesn't contain exactly one feature.
    """
    if (
        not isinstance(label, ObjectDetectionLabel)
        and not isinstance(label, SemanticSegmentationLabel)
        and not isinstance(label, InstanceSegmentationPredictionLabel)
        and not isinstance(label, InstanceSegmentationActualLabel)
    ):
        raise InvalidValueType(
            f"label {label}",
            label,
            "one of: ObjectDetectionLabel, SemanticSegmentationLabel, InstanceSegmentationPredictionLabel, "
            f"or InstanceSegmentationActualLabel for model type {ModelTypes.OBJECT_DETECTION}",
        )
    if embedding_features is None:
        raise ValueError(
            f"Cannot use {type(label)} without an embedding feature"
        )
    if len(embedding_features.keys()) != 1:
        raise ValueError(
            f"{type(label)} must be sent with exactly one embedding feature"
        )
    if isinstance(label, ObjectDetectionLabel):
        label.validate(prediction_or_actual=prediction_or_actual)
    else:
        label.validate()


def _validate_ranking_label(
    label: object,
) -> None:
    """Validate a ranking label for ranking model types.

    Args:
        label: The ranking label to validate.

    Raises:
        InvalidValueType: If the label is not a RankingPredictionLabel or RankingActualLabel.
    """
    if not isinstance(label, (RankingPredictionLabel, RankingActualLabel)):
        raise InvalidValueType(
            f"label {label}",
            label,
            f"RankingPredictionLabel or RankingActualLabel for model type {ModelTypes.RANKING}",
        )
    label.validate()


def _validate_generative_llm_label(
    label: object,
) -> None:
    """Validate a label for generative LLM model types.

    Args:
        label: The label value to validate.

    Raises:
        InvalidValueType: If the label is not a bool, int, float, or str.
    """
    is_valid = isinstance(label, (str, bool, int, float))
    if not is_valid:
        raise InvalidValueType(
            f"label {label}",
            label,
            f"one of: bool, int, float, str for model type {ModelTypes.GENERATIVE_LLM}",
        )


def _validate_multi_class_label(
    label: object,
) -> None:
    """Validate a multi-class label for multi-class model types.

    Args:
        label: The multi-class label to validate.

    Raises:
        InvalidValueType: If the label is not a MultiClassPredictionLabel or MultiClassActualLabel.
    """
    if not isinstance(
        label, (MultiClassPredictionLabel, MultiClassActualLabel)
    ):
        raise InvalidValueType(
            f"label {label}",
            label,
            f"MultiClassPredictionLabel or MultiClassActualLabel for model type {ModelTypes.MULTI_CLASS}",
        )
    label.validate()


def validate_and_convert_prediction_id(
    prediction_id: PredictionIDType | None,
    environment: Environments,
    prediction_label: PredictionLabelTypes | None = None,
    actual_label: ActualLabelTypes | None = None,
    shap_values: dict[str, float] | None = None,
) -> str:
    """Validate and convert a prediction ID to string format, or generate one if absent.

    Args:
        prediction_id: The prediction ID to validate/convert, or :obj:`None`.
        environment: The environment context (training, validation, production).
        prediction_label: Optional prediction label for delayed record detection.
        actual_label: Optional actual label for delayed record detection.
        shap_values: Optional SHAP values for delayed record detection.

    Returns:
        A validated prediction ID string.

    Raises:
        ValueError: If prediction ID is invalid.
    """
    # If the user does not provide prediction id
    if prediction_id:
        # If prediction id is given by user, convert it to string and validate length
        return _convert_prediction_id(prediction_id)

    # delayed records have actual information but not prediction information
    is_delayed_record = prediction_label is None and (
        actual_label is not None or shap_values is not None
    )
    # Pre-production environment does not need prediction id
    # Production environment needs prediction id for delayed record, since joins are needed
    if is_delayed_record and environment == Environments.PRODUCTION:
        raise ValueError(
            "prediction_id value cannot be None for delayed records, i.e., records ",
            "without prediction_label and with either actual_label or shap_values",
        )
    # Prediction ids are optional for: pre-production records and
    # production records that are not delayed records, they are generated
    # server-side
    return ""


def _convert_prediction_id(
    prediction_id: PredictionIDType,
) -> str:
    if not isinstance(prediction_id, str):
        try:
            prediction_id = str(
                prediction_id
            ).strip()  # strip ensures we don't receive whitespaces as part of the prediction id
        except Exception as e:
            raise ValueError(
                f"prediction_id value {prediction_id} must be convertible to a string"
            ) from e

    if len(prediction_id) not in range(
        MIN_PREDICTION_ID_LEN, MAX_PREDICTION_ID_LEN + 1
    ):
        raise ValueError(
            f"The string length of prediction_id {prediction_id} must be between {MIN_PREDICTION_ID_LEN} "
            f"and {MAX_PREDICTION_ID_LEN}"
        )
    return prediction_id
