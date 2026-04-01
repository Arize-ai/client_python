r"""Integration tests for EvaluatorsClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

**AI integration and provider keys:** Evaluator templates attach an
``EvaluatorLlmConfig`` to an existing **AI integration** in the test space (the
tests use the first integration from ``ai_integrations.list``). Configure that
integration in Arize with **valid provider credentials**—for the default model in
these tests, an OpenAI-compatible integration needs a real **OpenAI API key** (or
equivalent) on the integration. Without a working integration and key, create
calls can fail at runtime even when ``ARIZE_API_KEY`` is valid.

**Template text:** The API requires at least one **f-string-style** placeholder
(e.g. ``{output}``), not doubled braces like ``{{output}}``.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_evaluators_flows.py -m integration -v
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


def _make_template_config(eval_name: str, ai_integration_id: str) -> Any:
    from arize._generated import api_client as gen

    return gen.TemplateConfig(
        name=eval_name,
        template=(
            "Rate the following response on a scale of 0-1.\n"
            "Response: {output}\n"
            "Score (0-1):"
        ),
        include_explanations=False,
        use_function_calling_if_available=False,
        llm_config=gen.EvaluatorLlmConfig(
            ai_integration_id=ai_integration_id,
            model_name="gpt-4o-mini",
            invocation_parameters={"temperature": 0.0},
            provider_parameters={},
        ),
    )


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def evaluators_client(arize_client) -> Any:
    return arize_client.evaluators


@pytest.fixture(scope="module")
def ai_integration_id(arize_client) -> str:
    """Evaluator LLM config requires a space AI integration ID."""
    resp = arize_client.ai_integrations.list(space=SPACE_NAME, limit=1)
    if not resp.ai_integrations:
        pytest.skip(
            "No AI integrations in test space (required for EvaluatorLlmConfig)"
        )
    return resp.ai_integrations[0].id


class TestEvaluatorsCRUD:
    """End-to-end CRUD flows for EvaluatorsClient."""

    def test_create_get_delete_by_id(
        self, evaluators_client, ai_integration_id
    ) -> None:
        """Create an evaluator, retrieve it by ID, then delete it."""
        name = _unique("sdk-test-eval")
        evaluator = evaluators_client.create(
            name=name,
            space=SPACE_NAME,
            commit_message="initial version",
            template_config=_make_template_config(name, ai_integration_id),
        )
        try:
            assert evaluator.name == name
            assert is_resource_id(evaluator.id)

            fetched = evaluators_client.get(evaluator=evaluator.id)
            assert fetched.id == evaluator.id
            assert fetched.name == name
        finally:
            evaluators_client.delete(evaluator=evaluator.id)

    def test_create_get_delete_by_name(
        self, evaluators_client, ai_integration_id
    ) -> None:
        """Create an evaluator, retrieve and delete it by name."""
        name = _unique("sdk-test-eval")
        evaluator = evaluators_client.create(
            name=name,
            space=SPACE_NAME,
            commit_message="initial version",
            template_config=_make_template_config(name, ai_integration_id),
        )
        try:
            fetched = evaluators_client.get(evaluator=name, space=SPACE_NAME)
            assert fetched.id == evaluator.id
        finally:
            evaluators_client.delete(evaluator=name, space=SPACE_NAME)

    def test_create_update_delete(
        self, evaluators_client, ai_integration_id
    ) -> None:
        """Create an evaluator, update its description, then delete it."""
        name = _unique("sdk-test-eval")
        evaluator = evaluators_client.create(
            name=name,
            space=SPACE_NAME,
            commit_message="initial version",
            template_config=_make_template_config(name, ai_integration_id),
            description="Original description",
        )
        try:
            updated = evaluators_client.update(
                evaluator=evaluator.id,
                description="Updated by SDK integration test",
            )
            assert updated.id == evaluator.id
        finally:
            evaluators_client.delete(evaluator=evaluator.id)

    def test_create_version_and_list_versions(
        self, evaluators_client, ai_integration_id
    ) -> None:
        """Create an evaluator, add a second version, list versions."""
        name = _unique("sdk-test-eval")
        evaluator = evaluators_client.create(
            name=name,
            space=SPACE_NAME,
            commit_message="initial version",
            template_config=_make_template_config(name, ai_integration_id),
        )
        try:
            evaluators_client.create_version(
                evaluator=evaluator.id,
                commit_message="second version",
                template_config=_make_template_config(name, ai_integration_id),
            )
            versions_resp = evaluators_client.list_versions(
                evaluator=evaluator.id
            )
            assert len(versions_resp.evaluator_versions) >= 2
        finally:
            evaluators_client.delete(evaluator=evaluator.id)

    def test_create_appears_in_list(
        self, evaluators_client, ai_integration_id
    ) -> None:
        """Newly created evaluator appears in list() results."""
        name = _unique("sdk-test-eval")
        evaluator = evaluators_client.create(
            name=name,
            space=SPACE_NAME,
            commit_message="initial version",
            template_config=_make_template_config(name, ai_integration_id),
        )
        try:
            resp = evaluators_client.list(space=SPACE_NAME, limit=100)
            evaluator_ids = [e.id for e in resp.evaluators]
            assert evaluator.id in evaluator_ids
        finally:
            evaluators_client.delete(evaluator=evaluator.id)
