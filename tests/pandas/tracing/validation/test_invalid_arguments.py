import sys

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
            "2024-01-18T18:28:27.429383+00:00",
            "2024-01-18T18:28:27.429383+00:00",
        ],
        "end_time": [
            "2024-01-18T18:28:27.429383+00:00",
            "2024-01-18T18:28:27.429383+00:00",
        ],
    }
)

EVAL_PREFIXES = ["eval", "session_eval", "trace_eval"]


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@pytest.mark.parametrize("prefix", EVAL_PREFIXES)
def test_valid_eval_column_types(prefix):
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            f"{prefix}.eval_1.label": ["relevant", "irrelevant"],
            f"{prefix}.eval_1.score": [1.0, None],
            f"{prefix}.eval_1.explanation": [
                "explanation for relevant",
                "explanation for irrelevant",
            ],
        }
    )
    errors = evals_validation.validate_dataframe_form(
        evals_dataframe=evals_dataframe
    )
    assert (
        len(errors) == 0
    ), f"Expected no validation errors for all columns with prefix {prefix}"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@pytest.mark.parametrize("prefix", EVAL_PREFIXES)
def test_invalid_label_columns_type(prefix):
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            f"{prefix}.eval_1.label": [1, 2],
        }
    )
    errors = evals_validation.validate_dataframe_form(
        evals_dataframe=evals_dataframe
    )
    assert (
        len(errors) > 0
    ), f"Expected validation errors for label columns with incorrect type with prefix {prefix}"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@pytest.mark.parametrize("prefix", EVAL_PREFIXES)
def test_invalid_score_columns_type(prefix):
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            f"{prefix}.eval_1.score": ["1.0", "None"],
        }
    )
    errors = evals_validation.validate_dataframe_form(
        evals_dataframe=evals_dataframe
    )
    assert (
        len(errors) > 0
    ), f"Expected validation errors for score columns with incorrect type with prefix {prefix}"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@pytest.mark.parametrize("prefix", EVAL_PREFIXES)
def test_invalid_explanation_columns_type(prefix):
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            f"{prefix}.eval_1.explanation": [1, 2],
        }
    )
    errors = evals_validation.validate_dataframe_form(
        evals_dataframe=evals_dataframe
    )
    assert (
        len(errors) > 0
    ), f"Expected validation errors for explanation columns with incorrect type with prefix {prefix}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
