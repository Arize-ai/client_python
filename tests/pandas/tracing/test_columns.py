import re
import sys

import pytest

if sys.version_info >= (3, 8):
    from arize.pandas.tracing.columns import (
        ANNOTATION_COLUMN_PATTERN,
        ANNOTATION_NAME_PATTERN,
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
    assert re.match(
        EVAL_EXPLANATION_PATTERN, "eval.name with spaces.explanation"
    )
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
        assert (
            match.group(1) == expected_name
        ), f"Incorrect name captured for '{test_str}'"

    non_matches = [
        "evalname.",  # Missing dot
        "name_part.",  # Missing prefix
        "eval.name",  # Missing suffix
    ]
    for test_str in non_matches:
        assert (
            re.match(EVAL_NAME_PATTERN, test_str) is None
        ), f"Incorrectly matched '{test_str}'"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_annotation_column_pattern():
    """Test the regex pattern for matching valid annotation column names."""
    valid_patterns = [
        "annotation.quality.label",
        "annotation.toxicity_score.score",
        "annotation.needs review.updated_by",
        "annotation.timestamp_col.updated_at",
        "annotation.numeric_name123.label",
    ]
    invalid_patterns = [
        "annotation.quality",  # Missing suffix
        "annotations.quality.label",  # Wrong prefix
        "annotation.quality.Score",  # Incorrect suffix case
        "annotation..label",  # Empty name part
        "quality.label",  # Missing prefix
        "annotation.notes",  # Specific reserved name (tested separately if needed)
        "eval.quality.label",  # Different prefix
    ]

    for pattern in valid_patterns:
        assert re.match(
            ANNOTATION_COLUMN_PATTERN, pattern
        ), f"Should match: {pattern}"
    for pattern in invalid_patterns:
        assert not re.match(
            ANNOTATION_COLUMN_PATTERN, pattern
        ), f"Should NOT match: {pattern}"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_annotation_name_capture():
    """Test capturing the annotation name from column strings."""
    matches = [
        ("annotation.quality.label", "quality"),
        ("annotation.name with spaces.score", "name with spaces"),
        ("annotation.name_123.updated_by", "name_123"),
    ]
    for test_str, expected_name in matches:
        match = re.match(ANNOTATION_NAME_PATTERN, test_str)
        assert match is not None, f"Failed to match '{test_str}'"
        assert (
            match.group(1) == expected_name
        ), f"Incorrect name captured for '{test_str}'"

    non_matches = [
        "annotationname.label",  # Missing dot after prefix
        "annotation..score",  # Empty name part
        "annotation.name",  # No suffix
        "quality.label",  # Missing prefix
    ]
    for test_str in non_matches:
        assert (
            re.match(ANNOTATION_NAME_PATTERN, test_str) is None
        ), f"Incorrectly matched '{test_str}'"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
