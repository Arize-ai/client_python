"""Integration tests for PromptsClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_prompts_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

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


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _make_prompt_args(name: str) -> dict:
    """Return a minimal set of kwargs to create a prompt."""
    from arize._generated import api_client as gen

    return {
        "space": SPACE_NAME,
        "name": name,
        "commit_message": "initial version",
        "input_variable_format": gen.InputVariableFormat.F_STRING,
        "provider": gen.LlmProvider.OPEN_AI,
        "model": "gpt-4o-mini",
        "messages": [
            gen.LLMMessage(
                role="user",
                content="Hello {name}, please answer: {question}",
            )
        ],
    }


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def prompts_client(arize_client) -> Any:
    return arize_client.prompts


class TestPromptsCRUD:
    """End-to-end CRUD flows for PromptsClient."""

    def test_create_get_delete_by_id(self, prompts_client) -> None:
        """Create a prompt, retrieve it by ID, then delete it."""
        name = _unique("sdk-test-prompt")
        prompt = prompts_client.create(**_make_prompt_args(name))
        try:
            assert prompt.name == name
            assert is_resource_id(prompt.id)

            fetched = prompts_client.get(prompt=prompt.id)
            assert fetched.id == prompt.id
            assert fetched.name == name
        finally:
            prompts_client.delete(prompt=prompt.id)

    def test_create_get_delete_by_name(self, prompts_client) -> None:
        """Create a prompt, retrieve and delete it by name."""
        name = _unique("sdk-test-prompt")
        prompt = prompts_client.create(**_make_prompt_args(name))
        try:
            fetched = prompts_client.get(prompt=name, space=SPACE_NAME)
            assert fetched.id == prompt.id
        finally:
            prompts_client.delete(prompt=name, space=SPACE_NAME)

    def test_create_update_description(self, prompts_client) -> None:
        """Create a prompt then update its description."""
        name = _unique("sdk-test-prompt")
        prompt = prompts_client.create(**_make_prompt_args(name))
        try:
            updated = prompts_client.update(
                prompt=prompt.id,
                description="Updated by SDK integration test",
            )
            assert updated.id == prompt.id
        finally:
            prompts_client.delete(prompt=prompt.id)

    def test_create_version_and_list_versions(self, prompts_client) -> None:
        """Create a prompt, add a second version, list versions."""
        from arize._generated import api_client as gen

        name = _unique("sdk-test-prompt")
        prompt = prompts_client.create(**_make_prompt_args(name))
        try:
            prompts_client.create_version(
                prompt=prompt.id,
                commit_message="second version",
                input_variable_format=gen.InputVariableFormat.F_STRING,
                provider=gen.LlmProvider.OPEN_AI,
                model="gpt-4o-mini",
                messages=[
                    gen.LLMMessage(
                        role="user",
                        content="Updated content for {name}",
                    )
                ],
            )
            versions_resp = prompts_client.list_versions(prompt=prompt.id)
            assert len(versions_resp.prompt_versions) >= 2
        finally:
            prompts_client.delete(prompt=prompt.id)

    def test_set_and_get_label(self, prompts_client) -> None:
        """Set a label on a prompt version and resolve it back."""
        name = _unique("sdk-test-prompt")
        prompt = prompts_client.create(**_make_prompt_args(name))
        try:
            version_id = prompt.version.id
            prompts_client.set_labels(version_id=version_id, labels=["staging"])

            resolved = prompts_client.get_label(
                prompt=prompt.id, label_name="staging"
            )
            assert resolved.id == version_id
        finally:
            prompts_client.delete(prompt=prompt.id)

    def test_create_appears_in_list(self, prompts_client) -> None:
        """Newly created prompt appears in list() results."""
        name = _unique("sdk-test-prompt")
        prompt = prompts_client.create(**_make_prompt_args(name))
        try:
            resp = prompts_client.list(space=SPACE_NAME, limit=100)
            prompt_ids = [p.id for p in resp.prompts]
            assert prompt.id in prompt_ids
        finally:
            prompts_client.delete(prompt=prompt.id)
