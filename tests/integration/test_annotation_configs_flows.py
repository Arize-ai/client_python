"""Integration tests for AnnotationConfigsClient end-to-end flows.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_annotation_configs_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from arize.annotation_configs.types import AnnotationConfigType
from arize.utils.resolve import _find_space_id, is_resource_id

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


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def annotation_configs_client(arize_client) -> Any:
    return arize_client.annotation_configs


@pytest.fixture(scope="module")
def space_id(arize_client) -> str:
    from arize._generated import api_client as gen
    from arize._generated.api_client.api_client import ApiClient
    from arize._generated.api_client.configuration import Configuration
    from arize.config import SDKConfiguration

    config = SDKConfiguration(api_key=API_KEY)
    api_config = Configuration(host=config.api_url, access_token=API_KEY)
    client = ApiClient(api_config)
    spaces_api = gen.SpacesApi(client)
    return _find_space_id(spaces_api, SPACE_NAME)


class TestAnnotationConfigsCRUD:
    """End-to-end CRUD flows for AnnotationConfigsClient."""

    def test_create_get_delete_continuous_by_id(
        self, annotation_configs_client, space_id
    ) -> None:
        """Create a continuous annotation config, retrieve by ID, delete."""
        name = _unique("sdk-test-ac")
        ac = annotation_configs_client.create(
            name=name,
            space=space_id,
            config_type=AnnotationConfigType.CONTINUOUS,
            minimum_score=0.0,
            maximum_score=1.0,
        )
        inner = ac.actual_instance
        try:
            assert inner is not None
            assert inner.name == name
            assert is_resource_id(inner.id)

            fetched = annotation_configs_client.get(annotation_config=inner.id)
            fetched_inner = fetched.actual_instance
            assert fetched_inner is not None
            assert fetched_inner.id == inner.id
            assert fetched_inner.name == name
        finally:
            annotation_configs_client.delete(annotation_config=inner.id)

    def test_create_get_delete_continuous_by_name(
        self, annotation_configs_client, space_id
    ) -> None:
        """Create a continuous annotation config, retrieve and delete by name."""
        name = _unique("sdk-test-ac")
        ac = annotation_configs_client.create(
            name=name,
            space=space_id,
            config_type=AnnotationConfigType.CONTINUOUS,
            minimum_score=0.0,
            maximum_score=5.0,
        )
        inner = ac.actual_instance
        try:
            fetched = annotation_configs_client.get(
                annotation_config=name, space=space_id
            )
            fetched_inner = fetched.actual_instance
            assert fetched_inner is not None
            assert fetched_inner.id == inner.id
        finally:
            annotation_configs_client.delete(
                annotation_config=name, space=space_id
            )

    def test_create_get_delete_categorical(
        self, annotation_configs_client, space_id
    ) -> None:
        """Create a categorical annotation config, retrieve it, then delete."""
        from arize._generated import api_client as gen

        name = _unique("sdk-test-ac")
        values = [
            gen.CategoricalAnnotationValue(label="good", score=1.0),
            gen.CategoricalAnnotationValue(label="bad", score=0.0),
        ]
        ac = annotation_configs_client.create(
            name=name,
            space=space_id,
            config_type=AnnotationConfigType.CATEGORICAL,
            values=values,
        )
        inner = ac.actual_instance
        try:
            assert inner is not None
            assert inner.name == name

            fetched = annotation_configs_client.get(annotation_config=inner.id)
            fetched_inner = fetched.actual_instance
            assert fetched_inner is not None
            assert fetched_inner.id == inner.id
        finally:
            annotation_configs_client.delete(annotation_config=inner.id)

    def test_create_get_delete_freeform(
        self, annotation_configs_client, space_id
    ) -> None:
        """Create a freeform annotation config, retrieve it, then delete."""
        name = _unique("sdk-test-ac")
        ac = annotation_configs_client.create(
            name=name,
            space=space_id,
            config_type=AnnotationConfigType.FREEFORM,
        )
        inner = ac.actual_instance
        try:
            assert inner is not None
            assert inner.name == name

            fetched = annotation_configs_client.get(annotation_config=inner.id)
            fetched_inner = fetched.actual_instance
            assert fetched_inner is not None
            assert fetched_inner.id == inner.id
        finally:
            annotation_configs_client.delete(annotation_config=inner.id)

    def test_create_appears_in_list(
        self, annotation_configs_client, space_id
    ) -> None:
        """Newly created annotation config appears in list() results."""
        name = _unique("sdk-test-ac")
        ac = annotation_configs_client.create(
            name=name,
            space=space_id,
            config_type=AnnotationConfigType.FREEFORM,
        )
        inner = ac.actual_instance
        try:
            resp = annotation_configs_client.list(space=space_id, limit=100)
            ids = [
                item.actual_instance.id
                for item in resp.annotation_configs
                if item.actual_instance is not None
            ]
            assert inner.id in ids
        finally:
            annotation_configs_client.delete(annotation_config=inner.id)

    def test_list_filter_by_name(
        self, annotation_configs_client, space_id
    ) -> None:
        """list() name filter returns only matching configs."""
        name = _unique("sdk-test-ac")
        ac = annotation_configs_client.create(
            name=name,
            space=space_id,
            config_type=AnnotationConfigType.FREEFORM,
        )
        inner = ac.actual_instance
        try:
            resp = annotation_configs_client.list(
                space=space_id, name=name, limit=100
            )
            assert len(resp.annotation_configs) >= 1
            names = [
                item.actual_instance.name
                for item in resp.annotation_configs
                if item.actual_instance is not None
            ]
            assert name in names
        finally:
            annotation_configs_client.delete(annotation_config=inner.id)
