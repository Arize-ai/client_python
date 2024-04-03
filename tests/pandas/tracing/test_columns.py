import re
import sys

import pytest

if sys.version_info >= (3, 8):
    from arize.pandas.tracing.columns import (
        EVAL_COLUMN_PATTERN,
        EVAL_EXPLANATION_PATTERN,
        EVAL_LABEL_PATTERN,
        EVAL_NAME_PATTERN,
        EVAL_SCORE_PATTERN,
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_eval_column_pattern():
    assert re.match(EVAL_COLUMN_PATTERN, "eval.name.label")
    assert re.match(EVAL_COLUMN_PATTERN, "eval.name with spaces.label")
    assert re.match(EVAL_COLUMN_PATTERN, "eval.name.score")
    assert re.match(EVAL_COLUMN_PATTERN, "eval.name.explanation")
    assert not re.match(EVAL_COLUMN_PATTERN, "eval.name")
    assert not re.match(EVAL_COLUMN_PATTERN, "name.label")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_eval_label_pattern():
    assert re.match(EVAL_LABEL_PATTERN, "eval.name.label")
    assert re.match(EVAL_LABEL_PATTERN, "eval.name with spaces.label")
    assert not re.match(EVAL_LABEL_PATTERN, "eval.name.score")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_eval_score_pattern():
    assert re.match(EVAL_SCORE_PATTERN, "eval.name.score")
    assert re.match(EVAL_SCORE_PATTERN, "eval.name with spaces.score")
    assert not re.match(EVAL_SCORE_PATTERN, "eval.name.label")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_eval_explanation_pattern():
    assert re.match(EVAL_EXPLANATION_PATTERN, "eval.name.explanation")
    assert re.match(EVAL_EXPLANATION_PATTERN, "eval.name with spaces.explanation")
    assert not re.match(EVAL_EXPLANATION_PATTERN, "eval_name.label")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_eval_name_capture():
    matches = [
        ("eval.name_part.label", "name_part"),
        ("eval.name with spaces.label", "name with spaces"),
    ]
    for test_str, expected_name in matches:
        match = re.match(EVAL_NAME_PATTERN, test_str)
        assert match is not None, f"Failed to match '{test_str}'"
        assert match.group(1) == expected_name, f"Incorrect name captured for '{test_str}'"

    non_matches = [
        "evalname.",  # Missing dot
        "name_part.",  # Missing prefix
        "eval.name",  # Missing suffix
    ]
    for test_str in non_matches:
        assert re.match(EVAL_NAME_PATTERN, test_str) is None, f"Incorrectly matched '{test_str}'"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
