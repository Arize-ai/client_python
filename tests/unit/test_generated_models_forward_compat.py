"""Forward-compatibility tests for generated OpenAPI model deserialization.

These tests guard against generated from_dict methods in the Python SDK
raising ValueError on any unknown response field, turning every additive backend change
into a client-breaking change until a new SDK was published.

Fix: a three-tier schema taxonomy is enforced in the OpenAPI spec and applied via a
post-gen patch in recompile_openapi.sh:

  Type 1 — strict (request schemas):
    additionalProperties: false, no x-forward-compatible extension.
    from_dict raises ValueError on unknown fields.

  Type 2 — ignore (response/entity schemas):
    additionalProperties: false + x-forward-compatible: true.
    Post-gen patch removes the strict check; unknown fields are silently dropped.

  Type 3 — store (intentionally open-ended schemas):
    additionalProperties: true in the spec.
    Generator emits a from_dict that stores unknown fields in additional_properties.

Test structure:
  - TestForwardCompatModels: parametrized over all Type 3 models; each must accept
    an unknown field without raising and store it in additional_properties.
  - TestTaskRunFailureReason: regression test for the exact field that triggered #80452.
  - TestIgnoreModelsDropUnknownFields: spot-check that Type 2 entity models silently
    drop unknown fields (no ValueError, no additional_properties attribute).
  - TestStrictModelsUnchanged: spot-check that Type 1 request schemas still raise on
    unknown fields (we didn't accidentally loosen everything).
"""

from __future__ import annotations

import pytest

from arize._generated.api_client.models.agent_call_run_config import (
    AgentCallRunConfig,
)
from arize._generated.api_client.models.agent_call_run_config_request import (
    AgentCallRunConfigRequest,
)
from arize._generated.api_client.models.agent_config import AgentConfig
from arize._generated.api_client.models.agent_request_preset import (
    AgentRequestPreset,
)
from arize._generated.api_client.models.anthropic_config import AnthropicConfig
from arize._generated.api_client.models.aws_bedrock_config import (
    AwsBedrockConfig,
)
from arize._generated.api_client.models.create_project_request import (
    CreateProjectRequest,
)
from arize._generated.api_client.models.custom_config import CustomConfig
from arize._generated.api_client.models.gemini_config import GeminiConfig
from arize._generated.api_client.models.llm_generation_run_config import (
    LlmGenerationRunConfig,
)
from arize._generated.api_client.models.llm_generation_run_config_request import (
    LlmGenerationRunConfigRequest,
)
from arize._generated.api_client.models.llm_message import LLMMessage
from arize._generated.api_client.models.llm_message_request import (
    LLMMessageRequest,
)
from arize._generated.api_client.models.nvidia_nim_config import NvidiaNimConfig
from arize._generated.api_client.models.open_ai_config import OpenAiConfig
from arize._generated.api_client.models.project import Project
from arize._generated.api_client.models.task_run import TaskRun
from arize._generated.api_client.models.template_evaluation_run_config import (
    TemplateEvaluationRunConfig,
)
from arize._generated.api_client.models.template_evaluation_run_config_request import (
    TemplateEvaluationRunConfigRequest,
)
from arize._generated.api_client.models.tool_config import ToolConfig
from arize._generated.api_client.models.vertex_ai_config import VertexAiConfig

# ---------------------------------------------------------------------------
# Minimal valid payloads for each targeted model
# ---------------------------------------------------------------------------

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

_LLM_GENERATION_RUN_CONFIG = {
    "experiment_type": "LLM_GENERATION",
    "ai_integration_id": "ai_1",
    "messages": [{"role": "USER", "content": "hello"}],
    "input_variable_format": "MUSTACHE",
}

_TEMPLATE_EVALUATION_RUN_CONFIG = {
    "experiment_type": "TEMPLATE_EVALUATION",
    "ai_integration_id": "ai_1",
    "template": "evaluate {{input}}",
    "provide_explanation": False,
}

_AGENT_CALL_RUN_CONFIG = {
    "experiment_type": "AGENT_CALL",
    "integration_id": "integ_1",
    "input_template": {},
}

_AGENT_CONFIG = {
    "endpoint": "https://example.com/agent",
    "has_headers": False,
    "input_schema": {},
    "request_presets": [],
}

_AGENT_REQUEST_PRESET = {
    "name": "preset_1",
    "config": {"key": "value"},
}

_OPEN_AI_CONFIG = {
    "is_function_calling_enabled": True,
    "provider": "OPEN_AI",
    "has_api_key": True,
}

_ANTHROPIC_CONFIG = {
    "is_function_calling_enabled": True,
    "provider": "ANTHROPIC",
    "has_api_key": True,
}

_GEMINI_CONFIG = {
    "is_function_calling_enabled": True,
    "provider": "GEMINI",
    "has_api_key": True,
}

_AWS_BEDROCK_CONFIG = {
    "provider": "AWS_BEDROCK",
    "is_default_models_enabled": True,
    "model_names": [],
    "auth": {
        "auth_type": "DEFAULT",
        "role_arn": "arn:aws:iam::123:role/R",
        "external_id": None,
        "base_url": None,
    },
}

_CUSTOM_CONFIG = {
    "is_function_calling_enabled": False,
    "provider": "CUSTOM",
    "has_api_key": False,
    "base_url": "https://api.example.com",
    "header_names": [],
    "is_default_models_enabled": False,
    "model_names": [],
}

_NVIDIA_NIM_CONFIG = {
    "is_function_calling_enabled": False,
    "provider": "NVIDIA_NIM",
    "has_api_key": False,
    "base_url": None,
    "header_names": [],
    "is_default_models_enabled": False,
    "model_names": [],
}

_VERTEX_AI_CONFIG = {
    "provider": "VERTEX_AI",
    "project_id": "my-project",
    "location": "us-central1",
    "project_access_label": "label",
}

_LLM_MESSAGE = {
    "role": "USER",
    "content": "hello",
}

_TOOL_CONFIG: dict = {}


# ---------------------------------------------------------------------------
# Parametrized forward-compat tests
# ---------------------------------------------------------------------------

_FORWARD_COMPAT_CASES = [
    pytest.param(TaskRun, _TASK_RUN, id="TaskRun"),
    pytest.param(
        LlmGenerationRunConfig,
        _LLM_GENERATION_RUN_CONFIG,
        id="LlmGenerationRunConfig",
    ),
    pytest.param(
        TemplateEvaluationRunConfig,
        _TEMPLATE_EVALUATION_RUN_CONFIG,
        id="TemplateEvaluationRunConfig",
    ),
    pytest.param(
        AgentCallRunConfig, _AGENT_CALL_RUN_CONFIG, id="AgentCallRunConfig"
    ),
    pytest.param(AgentConfig, _AGENT_CONFIG, id="AgentConfig"),
    pytest.param(
        AgentRequestPreset, _AGENT_REQUEST_PRESET, id="AgentRequestPreset"
    ),
    pytest.param(OpenAiConfig, _OPEN_AI_CONFIG, id="OpenAiConfig"),
    pytest.param(AnthropicConfig, _ANTHROPIC_CONFIG, id="AnthropicConfig"),
    pytest.param(GeminiConfig, _GEMINI_CONFIG, id="GeminiConfig"),
    pytest.param(AwsBedrockConfig, _AWS_BEDROCK_CONFIG, id="AwsBedrockConfig"),
    pytest.param(CustomConfig, _CUSTOM_CONFIG, id="CustomConfig"),
    pytest.param(NvidiaNimConfig, _NVIDIA_NIM_CONFIG, id="NvidiaNimConfig"),
    pytest.param(VertexAiConfig, _VERTEX_AI_CONFIG, id="VertexAiConfig"),
    pytest.param(LLMMessage, _LLM_MESSAGE, id="LLMMessage"),
    pytest.param(ToolConfig, _TOOL_CONFIG, id="ToolConfig"),
]


@pytest.mark.unit
class TestForwardCompatModels:
    """All models marked additionalProperties: true must accept unknown response fields.

    Each case verifies:
    1. from_dict does not raise when the response contains an unrecognized key.
    2. The unknown field is stored in additional_properties (round-trippable), not dropped.
    3. Known fields are still populated correctly.
    """

    @pytest.mark.parametrize("model_cls,base_payload", _FORWARD_COMPAT_CASES)
    def test_tolerates_unknown_fields(
        self, model_cls: type, base_payload: dict
    ) -> None:
        """from_dict must not raise on an unknown field."""
        payload = {**base_payload, "future_field_xyz": "new_value"}
        obj = model_cls.from_dict(payload)
        assert obj is not None

    @pytest.mark.parametrize("model_cls,base_payload", _FORWARD_COMPAT_CASES)
    def test_unknown_fields_stored_in_additional_properties(
        self, model_cls: type, base_payload: dict
    ) -> None:
        """Unknown fields should be accessible via additional_properties, not silently dropped."""
        payload = {**base_payload, "future_field_xyz": "new_value"}
        obj = model_cls.from_dict(payload)
        assert obj is not None
        assert hasattr(obj, "additional_properties"), (
            f"{model_cls.__name__} has no additional_properties attribute — "
            "was it generated with additionalProperties: true?"
        )
        assert obj.additional_properties.get("future_field_xyz") == "new_value"

    @pytest.mark.parametrize("model_cls,base_payload", _FORWARD_COMPAT_CASES)
    def test_known_fields_still_populated(
        self, model_cls: type, base_payload: dict
    ) -> None:
        """Adding an unknown field must not corrupt known field deserialization."""
        payload = {**base_payload, "future_field_xyz": "new_value"}
        clean = model_cls.from_dict(base_payload)
        with_extra = model_cls.from_dict(payload)
        assert clean is not None and with_extra is not None
        for field_name in base_payload:
            clean_val = getattr(clean, field_name, None)
            extra_val = getattr(with_extra, field_name, None)
            assert clean_val == extra_val, (
                f"{model_cls.__name__}.{field_name}: "
                f"got {extra_val!r} with extra field, expected {clean_val!r}"
            )


# ---------------------------------------------------------------------------
# Regression test for issue #80452 — TaskRun.failure_reason
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskRunFailureReason:
    """Regression tests for issue #80452.

    The backend added failure_reason to TaskRun responses before a matching SDK
    was published. Strict from_dict raised ValueError for any unknown field, making
    this a 100% reproducible client-side failure.
    """

    def test_failure_reason_null(self) -> None:
        """failure_reason: null (omitted or explicitly null) deserializes to None."""
        run = TaskRun.from_dict(_TASK_RUN)
        assert run is not None
        assert run.failure_reason is None

    def test_failure_reason_string(self) -> None:
        """failure_reason with a value is populated on the model."""
        payload = {
            **_TASK_RUN,
            "status": "CANCELLED",
            "failure_reason": "all data already has evaluation labels",
        }
        run = TaskRun.from_dict(payload)
        assert run is not None
        assert run.status == "CANCELLED"
        assert run.failure_reason == "all data already has evaluation labels"

    def test_unknown_field_before_failure_reason_fix(self) -> None:
        """Simulates what would have failed in 8.41.0: a new field in the response.

        Before the fix, this raised:
          ValueError: Error due to additional fields (not defined in TaskRun) in the input: failure_reason
        """
        payload = {**_TASK_RUN, "failure_reason": "cancelled — no new data"}
        run = TaskRun.from_dict(payload)
        assert run is not None
        assert run.failure_reason == "cancelled — no new data"


# ---------------------------------------------------------------------------
# Type 2 models — silently drop unknown fields (no error, no additional_properties)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIgnoreModelsDropUnknownFields:
    """Entity/response models marked x-forward-compatible: true silently drop unknown fields.

    These are Type 2 schemas: they neither raise ValueError nor store extra fields in
    additional_properties — they simply ignore them.
    """

    def test_project_silently_drops_unknown_field(self) -> None:
        """Project is a Type 2 entity schema — unknown fields are silently dropped."""
        payload = {
            "id": "proj_1",
            "name": "my-project",
            "space_id": "space_1",
            "created_at": "2026-01-01T00:00:00Z",
            "unknown_future_field": "should_be_dropped",
        }
        project = Project.from_dict(payload)
        assert project is not None
        assert project.id == "proj_1"
        assert not hasattr(project, "unknown_future_field")
        assert not getattr(project, "additional_properties", {}).get(
            "unknown_future_field"
        )

    def test_project_known_fields_unaffected(self) -> None:
        """Known fields on a Type 2 model are still populated correctly."""
        payload = {
            "id": "proj_1",
            "name": "my-project",
            "space_id": "space_1",
            "created_at": "2026-01-01T00:00:00Z",
        }
        project = Project.from_dict(payload)
        assert project is not None
        assert project.id == "proj_1"
        assert project.name == "my-project"
        assert project.space_id == "space_1"


# ---------------------------------------------------------------------------
# Strict models must remain strict
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrictModelsUnchanged:
    """Request schemas must still reject unknown fields (Type 1 — strict).

    This ensures we only loosened the targeted response/entity models and didn't
    accidentally drop the strict check from request schemas.
    """

    def test_create_project_request_rejects_unknown_field(self) -> None:
        """CreateProjectRequest is a request schema — unknown fields must raise ValueError."""
        payload = {
            "name": "my-project",
            "space_id": "space_1",
            "unknown_future_field": "should_fail",
        }
        with pytest.raises(ValueError, match="additional fields"):
            CreateProjectRequest.from_dict(payload)

    def test_llm_message_request_rejects_unknown_field(self) -> None:
        """Nested message request objects must remain strict."""
        with pytest.raises(ValueError, match="additional fields"):
            LLMMessageRequest.from_dict(
                {**_LLM_MESSAGE, "unknown_future_field": "should_fail"}
            )

    @pytest.mark.parametrize(
        "model_cls,payload",
        [
            pytest.param(
                LlmGenerationRunConfigRequest,
                _LLM_GENERATION_RUN_CONFIG,
                id="LlmGenerationRunConfigRequest",
            ),
            pytest.param(
                TemplateEvaluationRunConfigRequest,
                _TEMPLATE_EVALUATION_RUN_CONFIG,
                id="TemplateEvaluationRunConfigRequest",
            ),
            pytest.param(
                AgentCallRunConfigRequest,
                _AGENT_CALL_RUN_CONFIG,
                id="AgentCallRunConfigRequest",
            ),
        ],
    )
    def test_run_configuration_requests_reject_unknown_fields(
        self, model_cls: type, payload: dict
    ) -> None:
        """Request variants stay strict while response variants remain extensible."""
        with pytest.raises(ValueError, match="additional fields"):
            model_cls.from_dict(
                {**payload, "unknown_future_field": "should_fail"}
            )
