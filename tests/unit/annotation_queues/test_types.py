"""Tests for arize.annotation_queues.types public re-exports."""

from __future__ import annotations

import pytest

import arize.annotation_queues.types as types_module
from arize.annotation_queues.types import (
    AnnotationInput,
    AnnotationQueue,
    AnnotationQueueExampleRecordInput,
    AnnotationQueueRecordAnnotateResult,
    AnnotationQueueRecordAssignResult,
    AnnotationQueueRecordInput,
    AnnotationQueueRecordsList200Response,
    AnnotationQueuesList200Response,
    AnnotationQueueSpanRecordInput,
    AnnotationQueuesRecordsCreate200Response,
    AssignmentMethod,
)


@pytest.mark.unit
class TestAnnotationQueuesTypes:
    """Tests for the annotation_queues types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        expected = {
            "AnnotationInput",
            "AnnotationQueue",
            "AnnotationQueueExampleRecordInput",
            "AnnotationQueueRecordAnnotateResult",
            "AnnotationQueueRecordAssignResult",
            "AnnotationQueueRecordInput",
            "AnnotationQueueRecordsList200Response",
            "AnnotationQueueSpanRecordInput",
            "AnnotationQueuesList200Response",
            "AnnotationQueuesRecordsCreate200Response",
            "AssignmentMethod",
        }
        assert expected.issubset(set(types_module.__all__))

    def test_assignment_method_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(AssignmentMethod, Enum)

    @pytest.mark.parametrize(
        "cls",
        [
            AnnotationInput,
            AnnotationQueue,
            AnnotationQueueExampleRecordInput,
            AnnotationQueueRecordAnnotateResult,
            AnnotationQueueRecordAssignResult,
            AnnotationQueueRecordInput,
            AnnotationQueueRecordsList200Response,
            AnnotationQueueSpanRecordInput,
            AnnotationQueuesList200Response,
            AnnotationQueuesRecordsCreate200Response,
        ],
    )
    def test_type_is_class(self, cls: type) -> None:
        assert isinstance(cls, type)
