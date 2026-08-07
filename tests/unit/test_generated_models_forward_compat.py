"""Regression tests for forward-compatible generated model deserialization."""

from __future__ import annotations

import pytest

from arize._generated.api_client.models.create_project_request import (
    CreateProjectRequest,
)
from arize._generated.api_client.models.project import Project
from arize._generated.api_client.models.task_run import TaskRun

_TASK_RUN = {
    "id": "run_1",
    "task_id": "task_1",
    "experiment_id": None,
    "status": "COMPLETED",
    "run_started_at": None,
    "run_finished_at": None,
    "data_start_time": None,
    "data_end_time": None,
    "num_successes": 5,
    "num_errors": 0,
    "num_skipped": 0,
    "created_at": "2026-07-23T00:00:00Z",
    "created_by_user_id": None,
    "failure_reason": None,
}


@pytest.mark.unit
class TestTaskRunForwardCompatibility:
    """Guard the response deserialization failure reported in #80452."""

    def test_tolerates_unknown_fields(self) -> None:
        run = TaskRun.from_dict({**_TASK_RUN, "future_field_xyz": "new_value"})

        assert run is not None
        assert run.additional_properties["future_field_xyz"] == "new_value"

    def test_failure_reason_deserializes(self) -> None:
        run = TaskRun.from_dict(
            {
                **_TASK_RUN,
                "status": "CANCELLED",
                "failure_reason": "all data already has evaluation labels",
            }
        )

        assert run is not None
        assert run.status == "CANCELLED"
        assert run.failure_reason == "all data already has evaluation labels"


@pytest.mark.unit
class TestResponseGenerationPolicy:
    """Distinguish response tolerance from strict request validation."""

    def test_response_model_ignores_unknown_fields(self) -> None:
        project = Project.from_dict(
            {
                "id": "project_1",
                "name": "example",
                "space_id": "space_1",
                "created_at": "2026-07-23T00:00:00Z",
                "updated_at": "2026-07-23T00:00:00Z",
                "future_field_xyz": "new_value",
            }
        )

        assert project is not None
        assert not hasattr(project, "additional_properties")

    def test_request_model_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValueError, match="future_field_xyz"):
            CreateProjectRequest.from_dict(
                {
                    "name": "example",
                    "space_id": "space_1",
                    "future_field_xyz": "new_value",
                }
            )
