"""Integration tests for IntegrationsClient end-to-end flows.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Integrations are polymorphic and owned at the account level (a name is unique
per ``(account, type)``), so these flows do not need a space; ``space`` is only
an optional visibility filter on ``get``/``list``/``delete``.

External-dependency notes:
    - Agent creates supply a public HTTPS ``endpoint``; the server validates it
      for SSRF (must resolve to a public address) but does not require the
      endpoint to be reachable at create time.
    - LLM creates supply a provider ``api_key``. The key is stored write-only
      and is not validated against the provider until it is actually used, so a
      placeholder key is sufficient for CRUD lifecycle testing.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_integrations_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from arize.integrations.types import (
    CreateAgentRequestPresetInput,
    CreateOpenAiConfig,
    IntegrationType,
)
from arize.utils.resolve import is_resource_id

API_KEY = os.environ.get("ARIZE_API_KEY", "")
SPACE_NAME = os.environ.get("ARIZE_TEST_SPACE_NAME", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not SPACE_NAME,
        reason="ARIZE_API_KEY and ARIZE_TEST_SPACE_NAME must be set",
    ),
]

_AGENT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}
_AGENT_ENDPOINT = "https://example.com/agent-replay"


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def integrations_client(arize_client) -> Any:
    return arize_client.integrations


class TestAgentIntegrationCRUD:
    """End-to-end CRUD flows for agent integrations."""

    def test_create_get_delete_by_id(self, integrations_client) -> None:
        """Create an agent integration, retrieve by ID, delete by ID.

        By-ID operations need no ``integration_type``.
        """
        name = _unique("sdk-test-agent")
        created = integrations_client.create_agent(
            name=name,
            endpoint=_AGENT_ENDPOINT,
            input_schema=_AGENT_INPUT_SCHEMA,
            description="created by SDK integration test",
        )
        try:
            assert created.name == name
            assert created.type == IntegrationType.AGENT.value
            assert is_resource_id(created.id)

            fetched = integrations_client.get(integration=created.id)
            assert fetched.id == created.id
            assert fetched.name == name
        finally:
            integrations_client.delete(integration=created.id)

    def test_create_get_delete_by_name(self, integrations_client) -> None:
        """Resolve an agent integration by name (no space) for get and delete."""
        name = _unique("sdk-test-agent")
        created = integrations_client.create_agent(
            name=name,
            endpoint=_AGENT_ENDPOINT,
            input_schema=_AGENT_INPUT_SCHEMA,
        )
        try:
            fetched = integrations_client.get(
                integration=name,
                integration_type=IntegrationType.AGENT,
            )
            assert fetched.id == created.id
        finally:
            integrations_client.delete(
                integration=name,
                integration_type=IntegrationType.AGENT,
            )

    def test_create_with_presets(self, integrations_client) -> None:
        """Create an agent integration with an initial request preset."""
        name = _unique("sdk-test-agent")
        created = integrations_client.create_agent(
            name=name,
            endpoint=_AGENT_ENDPOINT,
            input_schema=_AGENT_INPUT_SCHEMA,
            request_presets=[
                CreateAgentRequestPresetInput(
                    name="default",
                    config={"input": "hello"},
                    description="the default preset",
                )
            ],
        )
        try:
            assert is_resource_id(created.id)
            preset_names = [p.name for p in created.config.request_presets]
            assert "default" in preset_names
        finally:
            integrations_client.delete(
                integration=created.id,
                integration_type=IntegrationType.AGENT,
            )

    def test_create_appears_in_list(self, integrations_client) -> None:
        """A newly created agent integration appears in list() results."""
        name = _unique("sdk-test-agent")
        created = integrations_client.create_agent(
            name=name,
            endpoint=_AGENT_ENDPOINT,
            input_schema=_AGENT_INPUT_SCHEMA,
        )
        try:
            resp = integrations_client.list(
                integration_type=IntegrationType.AGENT, limit=100
            )
            ids = [item.id for item in resp.integrations]
            assert created.id in ids
        finally:
            integrations_client.delete(
                integration=created.id,
                integration_type=IntegrationType.AGENT,
            )

    def test_list_filter_by_name(self, integrations_client) -> None:
        """list() name filter returns the matching agent integration."""
        name = _unique("sdk-test-agent")
        created = integrations_client.create_agent(
            name=name,
            endpoint=_AGENT_ENDPOINT,
            input_schema=_AGENT_INPUT_SCHEMA,
        )
        try:
            resp = integrations_client.list(
                integration_type=IntegrationType.AGENT, name=name, limit=100
            )
            names = [item.name for item in resp.integrations]
            assert name in names
        finally:
            integrations_client.delete(
                integration=created.id,
                integration_type=IntegrationType.AGENT,
            )

    def test_update_description_then_clear(self, integrations_client) -> None:
        """update_agent sets a description, then clears it with None."""
        name = _unique("sdk-test-agent")
        created = integrations_client.create_agent(
            name=name,
            endpoint=_AGENT_ENDPOINT,
            input_schema=_AGENT_INPUT_SCHEMA,
            description="initial",
        )
        try:
            updated = integrations_client.update_agent(
                integration=created.id,
                description="updated description",
            )
            assert updated.description == "updated description"

            cleared = integrations_client.update_agent(
                integration=created.id,
                description=None,
            )
            assert cleared.description is None
        finally:
            integrations_client.delete(
                integration=created.id,
                integration_type=IntegrationType.AGENT,
            )


class TestLlmIntegrationCRUD:
    """End-to-end CRUD flows for LLM integrations."""

    def test_create_get_delete_by_id(self, integrations_client) -> None:
        """Create an OpenAI LLM integration, retrieve by ID, delete by ID.

        By-ID operations need no ``integration_type``.
        """
        name = _unique("sdk-test-llm")
        created = integrations_client.create_llm(
            name=name,
            config=CreateOpenAiConfig(
                provider="OPEN_AI", api_key="sk-placeholder-not-validated"
            ),
        )
        try:
            assert created.name == name
            assert created.type == IntegrationType.LLM.value
            assert is_resource_id(created.id)

            fetched = integrations_client.get(integration=created.id)
            assert fetched.id == created.id
            assert fetched.name == name
        finally:
            integrations_client.delete(integration=created.id)

    def test_create_get_delete_by_name(self, integrations_client) -> None:
        """Resolve an LLM integration by name (no space) for get and delete."""
        name = _unique("sdk-test-llm")
        created = integrations_client.create_llm(
            name=name,
            config=CreateOpenAiConfig(
                provider="OPEN_AI", api_key="sk-placeholder-not-validated"
            ),
        )
        try:
            fetched = integrations_client.get(
                integration=name,
                integration_type=IntegrationType.LLM,
            )
            assert fetched.id == created.id
        finally:
            integrations_client.delete(
                integration=name,
                integration_type=IntegrationType.LLM,
            )

    def test_update_rename_and_toggle_function_calling(
        self, integrations_client
    ) -> None:
        """update_llm renames the integration and toggles function calling."""
        name = _unique("sdk-test-llm")
        new_name = _unique("sdk-test-llm-renamed")
        created = integrations_client.create_llm(
            name=name,
            config=CreateOpenAiConfig(
                provider="OPEN_AI", api_key="sk-placeholder-not-validated"
            ),
        )
        try:
            updated = integrations_client.update_llm(
                integration=created.id,
                name=new_name,
                function_calling_enabled=False,
            )
            assert updated.name == new_name

            fetched = integrations_client.get(
                integration=created.id,
                integration_type=IntegrationType.LLM,
            )
            assert fetched.name == new_name
        finally:
            integrations_client.delete(
                integration=created.id,
                integration_type=IntegrationType.LLM,
            )


class TestPolymorphicList:
    """The untyped list merges every integration type."""

    def test_untyped_list_returns_all_types(self, integrations_client) -> None:
        """list() without a type returns LLM and agent integrations together."""
        agent_name = _unique("sdk-test-agent")
        llm_name = _unique("sdk-test-llm")
        created_agent = integrations_client.create_agent(
            name=agent_name,
            endpoint=_AGENT_ENDPOINT,
            input_schema=_AGENT_INPUT_SCHEMA,
        )
        try:
            created_llm = integrations_client.create_llm(
                name=llm_name,
                config=CreateOpenAiConfig(
                    provider="OPEN_AI", api_key="sk-placeholder-not-validated"
                ),
            )
            try:
                resp = integrations_client.list(limit=100)
                by_id = {item.id: item for item in resp.integrations}
                assert created_agent.id in by_id
                assert created_llm.id in by_id
                assert (
                    by_id[created_agent.id].type == IntegrationType.AGENT.value
                )
                assert by_id[created_llm.id].type == IntegrationType.LLM.value
            finally:
                integrations_client.delete(integration=created_llm.id)
        finally:
            integrations_client.delete(integration=created_agent.id)
