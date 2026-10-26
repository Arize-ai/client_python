"""Regression tests for partial-update model deserialization."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol, cast

import pytest

from arize._generated.api_client.models.update_agent_integration_request import (
    UpdateAgentIntegrationRequest,
)
from arize._generated.api_client.models.update_ai_integration_request import (
    UpdateAiIntegrationRequest,
)
from arize._generated.api_client.models.update_annotation_queue_request import (
    UpdateAnnotationQueueRequest,
)
from arize._generated.api_client.models.update_evaluation_task_request import (
    UpdateEvaluationTaskRequest,
)
from arize._generated.api_client.models.update_evaluator_integration_request import (
    UpdateEvaluatorIntegrationRequest,
)
from arize._generated.api_client.models.update_evaluator_request import (
    UpdateEvaluatorRequest,
)
from arize._generated.api_client.models.update_organization_request import (
    UpdateOrganizationRequest,
)
from arize._generated.api_client.models.update_prompt_request import (
    UpdatePromptRequest,
)
from arize._generated.api_client.models.update_role_request import (
    UpdateRoleRequest,
)
from arize._generated.api_client.models.update_space_request import (
    UpdateSpaceRequest,
)
from arize._generated.api_client.models.update_tag_request import (
    UpdateTagRequest,
)
from arize._generated.api_client.models.update_webhook_request import (
    UpdateWebhookRequest,
)


class PartialUpdateModel(Protocol):
    model_fields_set: set[str]

    def to_dict(self) -> dict[str, Any]: ...


Deserializer = Callable[[type[Any], dict[str, Any]], PartialUpdateModel]


def _from_dict(model: type[Any], payload: dict[str, Any]) -> PartialUpdateModel:
    result = model.from_dict(payload)
    assert result is not None
    return cast("PartialUpdateModel", result)


def _from_json(model: type[Any], payload: dict[str, Any]) -> PartialUpdateModel:
    result = model.from_json(json.dumps(payload))
    assert result is not None
    return cast("PartialUpdateModel", result)


_DESERIALIZERS = [
    pytest.param(_from_dict, id="from_dict"),
    pytest.param(_from_json, id="from_json"),
]

_UPDATE_MODELS = [
    pytest.param(
        UpdateAiIntegrationRequest,
        {"name": "renamed"},
        "api_key",
        id="UpdateAiIntegrationRequest",
    ),
    pytest.param(
        UpdateAgentIntegrationRequest,
        {"type": "AGENT"},
        "description",
        id="UpdateAgentIntegrationRequest",
    ),
    pytest.param(
        UpdateAnnotationQueueRequest,
        {"name": "renamed"},
        "instructions",
        id="UpdateAnnotationQueueRequest",
    ),
    pytest.param(
        UpdateEvaluationTaskRequest,
        {"name": "renamed"},
        "query_filter",
        id="UpdateEvaluationTaskRequest",
    ),
    pytest.param(
        UpdateEvaluatorRequest,
        {"name": "renamed"},
        "description",
        id="UpdateEvaluatorRequest",
    ),
    pytest.param(
        UpdateEvaluatorIntegrationRequest,
        {"type": "EVALUATOR"},
        "description",
        id="UpdateEvaluatorIntegrationRequest",
    ),
    pytest.param(
        UpdateOrganizationRequest,
        {"name": "renamed"},
        "description",
        id="UpdateOrganizationRequest",
    ),
    pytest.param(
        UpdatePromptRequest,
        {},
        "description",
        id="UpdatePromptRequest",
    ),
    pytest.param(
        UpdateRoleRequest,
        {"name": "renamed"},
        "description",
        id="UpdateRoleRequest",
    ),
    pytest.param(
        UpdateSpaceRequest,
        {"name": "renamed"},
        "description",
        id="UpdateSpaceRequest",
    ),
    pytest.param(
        UpdateTagRequest,
        {"name": "renamed"},
        "description",
        id="UpdateTagRequest",
    ),
    pytest.param(
        UpdateWebhookRequest,
        {"name": "renamed"},
        "description",
        id="UpdateWebhookRequest",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("deserialize", _DESERIALIZERS)
@pytest.mark.parametrize("model,payload,nullable_field", _UPDATE_MODELS)
def test_omitted_partial_update_fields_stay_absent(
    deserialize: Deserializer,
    model: type[Any],
    payload: dict[str, Any],
    nullable_field: str,
) -> None:
    update = deserialize(model, payload)

    assert nullable_field not in update.model_fields_set
    assert update.to_dict() == payload


@pytest.mark.unit
@pytest.mark.parametrize("deserialize", _DESERIALIZERS)
@pytest.mark.parametrize("model,payload,nullable_field", _UPDATE_MODELS)
def test_explicit_null_partial_update_fields_stay_present(
    deserialize: Deserializer,
    model: type[Any],
    payload: dict[str, Any],
    nullable_field: str,
) -> None:
    payload_with_null = {**payload, nullable_field: None}
    update = deserialize(model, payload_with_null)

    assert nullable_field in update.model_fields_set
    assert update.to_dict() == payload_with_null
