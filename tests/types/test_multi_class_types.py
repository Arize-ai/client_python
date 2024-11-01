import pytest

from arize.utils.constants import (
    MAX_MULTI_CLASS_NAME_LENGTH,
    MAX_NUMBER_OF_MULTI_CLASS_CLASSES,
)
from arize.utils.types import MultiClassActualLabel, MultiClassPredictionLabel

overMaxClasses = {"class": 0.2}
for i in range(MAX_NUMBER_OF_MULTI_CLASS_CLASSES):
    overMaxClasses[f"class_{i}"] = 0.2
overMaxClassLen = "a"
for _ in range(MAX_MULTI_CLASS_NAME_LENGTH):
    overMaxClassLen += "a"

input_labels = {
    "correct:prediction_scores": MultiClassPredictionLabel(
        prediction_scores={"class1": 0.1, "class2": 0.2},
    ),
    "correct:threshold_scores": MultiClassPredictionLabel(
        prediction_scores={"class1": 0.1, "class2": 0.2},
        threshold_scores={"class1": 0.1, "class2": 0.2},
    ),
    "invalid:wrong_pred_dictionary_type": MultiClassPredictionLabel(
        prediction_scores={"class1": "score", "class2": "score2"},
    ),
    "invalid:no_prediction_scores": MultiClassPredictionLabel(
        prediction_scores={},
    ),
    "invalid:too many_prediction_scores": MultiClassPredictionLabel(
        prediction_scores=overMaxClasses,
    ),
    "invalid:pred_empty_class_name": MultiClassPredictionLabel(
        prediction_scores={"": 1.1, "class2": 0.2},
    ),
    "invalid:pred_class_name_too_long": MultiClassPredictionLabel(
        prediction_scores={overMaxClassLen: 1.1, "class2": 0.2},
    ),
    "invalid:pred_score_over_1": MultiClassPredictionLabel(
        prediction_scores={"class1": 1.1, "class2": 0.2},
    ),
    "invalid:wrong_thresh_dictionary_type": MultiClassPredictionLabel(
        prediction_scores={"class1": 0.1, "class2": 0.2},
        threshold_scores={"class1": "score", "class2": 0.2},
    ),
    "invalid:pred_thresh_not_same_num_scores": MultiClassPredictionLabel(
        prediction_scores={"class1": 0.1, "class2": 0.2},
        threshold_scores={"class1": 0.1},
    ),
    "invalid:pred_thresh_not_same_classes": MultiClassPredictionLabel(
        prediction_scores={"class1": 0.1, "class2": 0.2},
        threshold_scores={"class1": 0.1, "class3": 0.1},
    ),
    "invalid:thresh_score_under_0": MultiClassPredictionLabel(
        prediction_scores={"class1": 0.1, "class2": 0.2},
        threshold_scores={"class1": -1, "class2": 0.2},
    ),
    "correct:actual_scores": MultiClassActualLabel(
        actual_scores={"class1": 0, "class2": 1},
    ),
    "correct:actual_scores_multi_1": MultiClassActualLabel(
        actual_scores={"class1": 1, "class2": 1},
    ),
    "invalid:wrong_actual_dictionary_type": MultiClassActualLabel(
        actual_scores={"class1": "score", "class2": 0},
    ),
    "invalid:no_actual_scores": MultiClassActualLabel(
        actual_scores={},
    ),
    "invalid:too_many_actual_scores": MultiClassActualLabel(
        actual_scores=overMaxClasses,
    ),
    "invalid:actual_score_empty_class_name": MultiClassActualLabel(
        actual_scores={"": 1, "class2": 0},
    ),
    "invalid:act_class_name_too_long": MultiClassActualLabel(
        actual_scores={overMaxClassLen: 1.1, "class2": 0.2},
    ),
    "invalid:actual_score_not_0_or_1": MultiClassActualLabel(
        actual_scores={"class1": 0.7, "class2": 0.2},
    ),
}


def test_correct_multi_class_label():
    keys = [key for key in input_labels if "correct:" in key]
    assert len(keys) > 0, "Test configuration error: keys must not be empty"

    for key in keys:
        multi_class_label = input_labels[key]
        try:
            multi_class_label.validate()
        except Exception as err:
            raise AssertionError(
                f"Correct mutli class prediction label should give no errors. Failing key = {key:s}. "
                f"Error = {err}"
            ) from None


def test_invalid_scores():
    keys = [key for key in input_labels if "invalid:" in key]
    assert len(keys) > 0, "Test configuration error: keys must not be empty"

    for key in keys:
        multi_class_label = input_labels[key]
        with pytest.raises(ValueError) as e:
            multi_class_label.validate()
            assert isinstance(
                e, ValueError
            ), "Invalid values should raise value errors"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
