"""Integration tests for DatasetsClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_datasets_flows.py -m integration -v
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


_EXAMPLES = [
    {"input": "What is 2+2?", "output": "4"},
    {"input": "What is the capital of France?", "output": "Paris"},
]


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def datasets_client(arize_client) -> Any:
    return arize_client.datasets


class TestDatasetsCRUD:
    """End-to-end CRUD flows for DatasetsClient."""

    def test_create_get_delete_by_id(self, datasets_client) -> None:
        """Create a dataset, retrieve it by ID, then delete it."""
        name = _unique("sdk-test-ds")
        dataset = datasets_client.create(
            name=name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            assert dataset.name == name
            assert is_resource_id(dataset.id)

            fetched = datasets_client.get(dataset=dataset.id)
            assert fetched.id == dataset.id
            assert fetched.name == name
        finally:
            datasets_client.delete(dataset=dataset.id)

    def test_create_get_delete_by_name(self, datasets_client) -> None:
        """Create a dataset, retrieve and delete it by name."""
        name = _unique("sdk-test-ds")
        dataset = datasets_client.create(
            name=name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            fetched = datasets_client.get(dataset=name, space=SPACE_NAME)
            assert fetched.id == dataset.id
        finally:
            datasets_client.delete(dataset=name, space=SPACE_NAME)

    def test_create_appears_in_list(self, datasets_client) -> None:
        """Newly created dataset appears in list() results."""
        name = _unique("sdk-test-ds")
        dataset = datasets_client.create(
            name=name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            resp = datasets_client.list(space=SPACE_NAME, limit=100)
            dataset_ids = [d.id for d in resp.datasets]
            assert dataset.id in dataset_ids
        finally:
            datasets_client.delete(dataset=dataset.id)

    def test_list_filter_by_name(self, datasets_client) -> None:
        """list() name filter returns only matching datasets."""
        name = _unique("sdk-test-ds")
        dataset = datasets_client.create(
            name=name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            resp = datasets_client.list(space=SPACE_NAME, name=name, limit=100)
            assert len(resp.datasets) >= 1
            assert any(d.id == dataset.id for d in resp.datasets)
        finally:
            datasets_client.delete(dataset=dataset.id)

    def test_list_examples(self, datasets_client) -> None:
        """list_examples() returns the examples that were uploaded."""
        name = _unique("sdk-test-ds")
        dataset = datasets_client.create(
            name=name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            resp = datasets_client.list_examples(dataset=dataset.id, limit=100)
            assert len(resp.examples) == len(_EXAMPLES)
        finally:
            datasets_client.delete(dataset=dataset.id)

    def test_append_examples(self, datasets_client) -> None:
        """append_examples() adds rows to an existing dataset."""
        name = _unique("sdk-test-ds")
        dataset = datasets_client.create(
            name=name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            new_examples = [{"input": "New question", "output": "New answer"}]
            datasets_client.append_examples(
                dataset=dataset.id,
                examples=new_examples,
            )
            resp = datasets_client.list_examples(dataset=dataset.id, limit=100)
            assert len(resp.examples) >= len(_EXAMPLES) + len(new_examples)
        finally:
            datasets_client.delete(dataset=dataset.id)
