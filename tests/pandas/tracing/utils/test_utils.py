import sys

import pandas as pd
import pytest

if sys.version_info >= (3, 8):
    from arize.pandas.tracing.utils import sanitize_dataframe_column_names


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_replace_spaces_with_underscores_eval_columns():
    evals_dataframe = pd.DataFrame(
        {
            "context.span_id": ["span_id_1", "span_id_2"],
            "eval.eval 1.label": [
                1,
                2,
            ],
        }
    )
    evals_dataframe_with_underscores = sanitize_dataframe_column_names(evals_dataframe)
    assert " " not in evals_dataframe_with_underscores.columns, "Expected no spaces in column names"
    assert (
        evals_dataframe_with_underscores.columns[1] == "eval.eval_1.label"
    ), "Expected column name to be replaced with underscores"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
