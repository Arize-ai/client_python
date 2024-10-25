import sys

import arize.pandas.tracing.constants as tracing_constants
import numpy as np
import pandas as pd
import pytest

if sys.version_info >= (3, 8):
    from arize.pandas.tracing.validation.evals import evals_validation

valid_spans_dataframe = pd.DataFrame(
    {
        "context.span_id": ["span_id_11111111", "span_id_22222222"],
        "context.trace_id": ["trace_id_11111111", "trace_id_22222222"],
        "name": ["name_1", "name_2"],
        "start_time": [
            1710881086000000000,
            1710881087000000000,
        ],
        "end_time": [
            1710881088000000000,
            1710881089000000000,
        ],
    }
)
valid_project_name = "project-name"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_labels_and_explanations():
    evals_dataframe = pd.DataFrame(
        {
            "eval.eval_1.label": ["relevant", "irrelevant"],
            "eval.eval_1.explanation": [
                "explanation for relevant",
                "explanation for irrelevant",
            ],
        }
    )
    errors = evals_validation.validate_values(
        evals_dataframe=evals_dataframe,
        project_name=valid_project_name,
    )
    assert len(errors) == 0, "Expected no validation errors for valid labels and explanations"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_label_not_empty_string():
    evals_dataframe = pd.DataFrame(
        {
            "eval.eval_1.label": [
                "",
            ],
            "eval.eval_1.explanation": [
                "explanation",
            ],
        }
    )
    errors = evals_validation.validate_values(
        evals_dataframe=evals_dataframe,
        project_name=valid_project_name,
    )
    assert len(errors) > 0, "Expected validation errors for empty string labels"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_label_exceeds_max_length():
    evals_dataframe = pd.DataFrame(
        {
            "eval.eval_1.label": ["r" * (tracing_constants.EVAL_EXPLANATION_MAX_STR_LENGTH + 1)],
            "eval.eval_1.explanation": [
                "explanation",
            ],
        }
    )
    errors = evals_validation.validate_values(
        evals_dataframe=evals_dataframe,
        project_name=valid_project_name,
    )
    assert len(errors) > 0, "Expected validation errors for labels exceeding max length"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_explanation_exceeds_max_length():
    evals_dataframe = pd.DataFrame(
        {
            "eval.eval_1.label": [
                "relevant",
            ],
            "eval.eval_1.explanation": [
                "e" * (tracing_constants.EVAL_EXPLANATION_MAX_STR_LENGTH + 1),
            ],
        }
    )
    errors = evals_validation.validate_values(
        evals_dataframe=evals_dataframe,
        project_name=valid_project_name,
    )
    assert len(errors) > 0, "Expected validation errors for explanations exceeding max length"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_float_values():
    evals_dataframe = pd.DataFrame(
        {
            "eval.eval_1.label": ["relevant", "irrelevant"],
            "eval.eval_1.score": [1.0, None],
            "eval.eval_1.explanation": [
                "explanation for relevant",
                "explanation for irrelevant",
            ],
        }
    )
    errors = evals_validation.validate_values(
        evals_dataframe=evals_dataframe,
        project_name=valid_project_name,
    )
    assert len(errors) == 0, "Expected no validation errors for valid float values"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_infinite_values_present():
    evals_dataframe = pd.DataFrame(
        {
            "eval.eval_1.label": ["relevant", "irrelevant"],
            "eval.eval_1.score": [1.0, np.inf],
            "eval.eval_1.explanation": [
                "explanation for relevant",
                "explanation for irrelevant",
            ],
        }
    )
    errors = evals_validation.validate_values(
        evals_dataframe=evals_dataframe,
        project_name=valid_project_name,
    )
    assert len(errors) > 0, "Expected validation errors for infinite values present"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_null_values():
    # Test all columns present
    evals_dataframe = pd.DataFrame(
        {
            "eval.eval_1.label": ["pass", None, None],
            "eval.eval_1.score": [None, 95, None],
            "eval.eval_1.explanation": ["explanation for pass", None, None],
            "eval.eval_2.label": ["pass", None, None],
            "eval.eval_2.score": [None, 95, None],
            "eval.eval_3.label": ["pass", None, None],
            "eval.eval_4.score": [None, 95, None],
        }
    )
    errors = evals_validation.validate_values(
        evals_dataframe=evals_dataframe,
        project_name=valid_project_name,
    )
    assert (
        len(errors) == 0
    ), "Expected no validation errors for evals with a label and/or score present with explanations"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_null_values():
    # Labels and scores are both null, explanations are not null -> invalid since we require
    # at least one of labels or scores when explanations are present
    evals_dataframe = pd.DataFrame(
        {
            "eval.eval_1.label": [None, None],
            "eval.eval_1.score": [None, None],
            "eval.eval_1.explanation": ["explantion 1", "explanation 2"],
        }
    )
    errors = evals_validation.validate_values(
        evals_dataframe=evals_dataframe,
        project_name=valid_project_name,
    )
    assert (
        len(errors) > 0
    ), "Expected validation errors for invalid labels and scores all null with explanations present"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
