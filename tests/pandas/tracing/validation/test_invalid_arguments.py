import sys

import pandas as pd
import pytest

if sys.version_info >= (3, 8):
    from arize.pandas.tracing.validation import validate_dataframe_form

valid_spans_dataframe = pd.DataFrame(
    {
        "context.span_id": ["span_id_11111111", "span_id_22222222"],
        "context.trace_id": ["trace_id_11111111", "trace_id_22222222"],
        "name": ["name_1", "name_2"],
        "start_time": [
            "2024-01-18T18:28:27.429383+00:00",
            "2024-01-18T18:28:27.429383+00:00",
        ],
        "end_time": [
            "2024-01-18T18:28:27.429383+00:00",
            "2024-01-18T18:28:27.429383+00:00",
        ],
    }
)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_eval_column_types():
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            "eval.eval_1.label": ["relevant", "irrelevant"],
            "eval.eval_1.score": [1.0, None],
            "eval.eval_1.explanation": [
                "explanation for relevant",
                "explanation for irrelevant",
            ],
        }
    )
    errors = validate_dataframe_form(
        spans_dataframe=valid_spans_dataframe, evals_dataframe=evals_dataframe
    )
    assert len(errors) == 0, "Expected no validation errors for all columns"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_label_columns_type():
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            "eval.eval_1.label": [
                1,
                2,
            ],
        }
    )
    errors = validate_dataframe_form(
        spans_dataframe=valid_spans_dataframe, evals_dataframe=evals_dataframe
    )
    assert len(errors) > 0, "Expected validation errors for label columns with incorrect type"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_score_columns_type():
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            "eval.eval_1.score": [
                "1.0",
                "None",
            ],
        }
    )
    errors = validate_dataframe_form(
        spans_dataframe=valid_spans_dataframe, evals_dataframe=evals_dataframe
    )
    assert len(errors) > 0, "Expected validation errors for score columns with incorrect type"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_explanation_columns_type():
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            "eval.eval_1.explanation": [
                1,
                2,
            ],
        }
    )
    errors = validate_dataframe_form(
        spans_dataframe=valid_spans_dataframe, evals_dataframe=evals_dataframe
    )
    assert len(errors) > 0, "Expected validation errors for explanation columns with incorrect type"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
