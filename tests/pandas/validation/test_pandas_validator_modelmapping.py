from collections import ChainMap

import pandas as pd
import pytest

from arize.pandas.validation.errors import (
    InvalidModelTypeAndMetricsCombination,
    MissingRequiredColumnsMetricsValidation,
)
from arize.pandas.validation.validator import Validator
from arize.utils.types import (
    Environments,
    Metrics,
    ModelTypes,
    ObjectDetectionColumnNames,
    Schema,
)


def test__check_model_type_metrics_no_metrics_selected():
    # This would otherwise raise a validation error because rank_column_name is required for
    # the combination of ranking model & ranking metrics.
    errors = Validator.validate_params(
        **ChainMap(kwargs),
    )
    assert len(errors) == 0


def test__check_model_type_metrics_old_model_type():
    # This would otherwise raise a validation error because the combination of metrics is not valid.
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.NUMERIC,
                "metric_families": [Metrics.RANKING, Metrics.CLASSIFICATION],
            },
            kwargs,
        ),
    )
    assert len(errors) == 0


def test__check_model_type_metrics_bad_metric_combo():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.REGRESSION,
                "metric_families": [Metrics.RANKING, Metrics.CLASSIFICATION],
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is InvalidModelTypeAndMetricsCombination

    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.BINARY_CLASSIFICATION,
                "metric_families": [Metrics.RANKING, Metrics.CLASSIFICATION],
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is InvalidModelTypeAndMetricsCombination


def test__check_model_type_metrics_bin_classif_with_classification_metrics():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.BINARY_CLASSIFICATION,
                "metric_families": [Metrics.CLASSIFICATION],
            },
            classification_no_score,
            kwargs,
        ),
    )
    assert len(errors) == 0


def test__check_model_type_metrics_bin_classif_with_classification_auc_logloss_metrics():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.BINARY_CLASSIFICATION,
                "metric_families": [
                    Metrics.CLASSIFICATION,
                    Metrics.AUC_LOG_LOSS,
                ],
            },
            classification_with_score,
            kwargs,
        ),
    )
    assert len(errors) == 0

    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.BINARY_CLASSIFICATION,
                "metric_families": [
                    Metrics.CLASSIFICATION,
                    Metrics.AUC_LOG_LOSS,
                ],
            },
            classification_no_score,
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingRequiredColumnsMetricsValidation


def test__check_model_type_metrics_bin_classif_with_classification_regression_metrics():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.BINARY_CLASSIFICATION,
                "metric_families": [Metrics.CLASSIFICATION, Metrics.REGRESSION],
            },
            classification_with_score,
            kwargs,
        ),
    )
    assert len(errors) == 0

    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.BINARY_CLASSIFICATION,
                "metric_families": [Metrics.CLASSIFICATION, Metrics.REGRESSION],
            },
            classification_no_score,
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingRequiredColumnsMetricsValidation


def test__check_model_type_metrics_bin_classif_with_classification_auc_logloss_regression_metrics():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.BINARY_CLASSIFICATION,
                "metric_families": [
                    Metrics.CLASSIFICATION,
                    Metrics.AUC_LOG_LOSS,
                    Metrics.REGRESSION,
                ],
            },
            classification_with_score,
            kwargs,
        ),
    )
    assert len(errors) == 0

    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.BINARY_CLASSIFICATION,
                "metric_families": [
                    Metrics.CLASSIFICATION,
                    Metrics.AUC_LOG_LOSS,
                    Metrics.REGRESSION,
                ],
            },
            classification_no_score,
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingRequiredColumnsMetricsValidation


def test__check_model_type_metrics_regression_with_regression_metrics():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.REGRESSION,
                "metric_families": [Metrics.REGRESSION],
            },
            regression,
            kwargs,
        ),
    )
    assert len(errors) == 0


def test__check_model_type_metrics_ranking_with_ranking_metrics():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.RANKING,
                "metric_families": [Metrics.RANKING],
            },
            ranking,
            kwargs,
        ),
    )
    assert len(errors) == 0


def test__check_model_type_metrics_ranking_with_ranking_auc_logloss_metrics():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.RANKING,
                "metric_families": [Metrics.RANKING, Metrics.AUC_LOG_LOSS],
            },
            ranking_score,
            kwargs,
        ),
    )
    assert len(errors) == 0

    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.RANKING,
                "metric_families": [Metrics.RANKING, Metrics.AUC_LOG_LOSS],
            },
            ranking,
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingRequiredColumnsMetricsValidation


def test__check_model_type_metrics_object_detection_with_classification_metrics():
    kwargs = get_object_detection_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            kwargs,
        ),
    )
    assert len(errors) == 0

    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_id",
                    object_detection_actual_column_names=ObjectDetectionColumnNames(
                        bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                        categories_column_name="bounding_boxes_categories",
                        scores_column_name="bounding_boxes_scores",
                    ),
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 2
    assert MissingRequiredColumnsMetricsValidation in (type(e) for e in errors)


# Helpers #

classification_with_score = {
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0"]),
            "prediction_label": pd.Series(["fraud"]),
            "prediction_score": pd.Series([1]),
        }
    ),
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_label_column_name="prediction_label",
        prediction_score_column_name="prediction_score",
    ),
}

classification_no_score = {
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0"]),
            "prediction_label": pd.Series(["fraud"]),
        }
    ),
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_label_column_name="prediction_label",
    ),
}

# prediction_score or prediction_label can be used for the prediction value.
regression = {
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0"]),
            "prediction_score": pd.Series([3]),
        }
    ),
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_score_column_name="prediction_score",
    ),
}

ranking = {
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0"]),
            "group_prediction_id": pd.Series(["5"]),
            "prediction_label": pd.Series(["fraud"]),
            "rank": pd.Series([1]),
        }
    ),
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_group_id_column_name="group_prediction_id",
        prediction_label_column_name="prediction_label",
        rank_column_name="rank",
    ),
}

ranking_score = {
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0"]),
            "group_prediction_id": pd.Series(["5"]),
            "prediction_label": pd.Series(["fraud"]),
            "rank": pd.Series([1]),
            "prediction_score": pd.Series([1]),
        }
    ),
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_group_id_column_name="group_prediction_id",
        prediction_label_column_name="prediction_label",
        rank_column_name="rank",
        prediction_score_column_name="prediction_score",
    ),
}

kwargs = {
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0"]),
            "prediction_label": pd.Series(["3"]),
        }
    ),
    "model_id": "fraud",
    "model_version": "v1.0",
    "environment": Environments.PRODUCTION,
    "model_type": ModelTypes.REGRESSION,
    "metric_families": None,
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_label_column_name="prediction_label",
    ),
}


def get_object_detection_kwargs():
    return {
        "model_id": "object_detection",
        "model_type": ModelTypes.OBJECT_DETECTION,
        "environment": Environments.PRODUCTION,
        "metric_families": [Metrics.CLASSIFICATION],
        "dataframe": pd.DataFrame(
            {
                "prediction_id": pd.Series(["0"]),
                "bounding_boxes_coordinates": pd.Series(
                    [
                        [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
                    ]
                ),
                "bounding_boxes_categories": pd.Series(
                    [
                        ["dog", "cat"],
                    ]
                ),
                "bounding_boxes_scores": pd.Series(
                    [
                        [0.18, 0.33],
                    ]
                ),
            }
        ),
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            object_detection_prediction_column_names=ObjectDetectionColumnNames(
                bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                categories_column_name="bounding_boxes_categories",
                scores_column_name="bounding_boxes_scores",
            ),
            object_detection_actual_column_names=ObjectDetectionColumnNames(
                bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                categories_column_name="bounding_boxes_categories",
                scores_column_name="bounding_boxes_scores",
            ),
        ),
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
