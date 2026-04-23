r"""Integration tests for dataset example annotation flows.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with::

    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_dataset_annotations_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

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
def datasets_client(arize_client: Any) -> Any:
    return arize_client.datasets


@pytest.fixture(scope="module")
def annotation_configs_client(arize_client: Any) -> Any:
    return arize_client.annotation_configs


@pytest.fixture(scope="module")
def space_id(arize_client: Any) -> str:
    from arize._generated import api_client as gen
    from arize._generated.api_client.api_client import ApiClient
    from arize._generated.api_client.configuration import Configuration
    from arize.config import SDKConfiguration
    from arize.utils.resolve import _find_space_id

    config = SDKConfiguration(api_key=API_KEY)
    api_config = Configuration(host=config.api_url, access_token=API_KEY)
    client = ApiClient(api_config)
    spaces_api = gen.SpacesApi(client)
    return _find_space_id(spaces_api, SPACE_NAME)


class TestDatasetExamplesAnnotate:
    """End-to-end annotation flows for dataset examples."""

    def test_annotate_continuous(
        self,
        datasets_client: Any,
        annotation_configs_client: Any,
        space_id: str,
    ) -> None:
        """Create dataset + continuous annotation config, annotate an example, verify."""
        from arize._generated import api_client as gen
        from arize.annotation_configs.types import AnnotationConfigType

        ds_name = _unique("sdk-test-annot-ds")
        ac_name = _unique("sdk-test-annot-ac-cont")

        dataset = datasets_client.create(
            name=ds_name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        ac = annotation_configs_client.create(
            name=ac_name,
            space=space_id,
            config_type=AnnotationConfigType.CONTINUOUS,
            minimum_score=0.0,
            maximum_score=1.0,
        )
        ac_id = ac.actual_instance.id
        try:
            examples_resp = datasets_client.list_examples(
                dataset=dataset.id, limit=10
            )
            example_id = examples_resp.examples[0].id

            result = datasets_client.annotate_examples(
                dataset=dataset.id,
                annotations=[
                    gen.AnnotateRecordInput(
                        record_id=example_id,
                        values=[gen.AnnotationInput(name=ac_name, score=0.85)],
                    )
                ],
            )

            assert len(result.results) == 1
            assert result.results[0].record_id == example_id
            assert len(result.results[0].annotations) >= 1
            result_annotation = next(
                a for a in result.results[0].annotations if a.name == ac_name
            )
            assert result_annotation.score == 0.85
        finally:
            datasets_client.delete(dataset=dataset.id)
            annotation_configs_client.delete(annotation_config=ac_id)

    def test_annotate_categorical(
        self,
        datasets_client: Any,
        annotation_configs_client: Any,
        space_id: str,
    ) -> None:
        """Create dataset + categorical annotation config, annotate an example, verify."""
        from arize._generated import api_client as gen
        from arize.annotation_configs.types import AnnotationConfigType

        ds_name = _unique("sdk-test-annot-ds")
        ac_name = _unique("sdk-test-annot-ac-cat")

        dataset = datasets_client.create(
            name=ds_name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        ac = annotation_configs_client.create(
            name=ac_name,
            space=space_id,
            config_type=AnnotationConfigType.CATEGORICAL,
            values=["correct", "incorrect", "partial"],
        )
        ac_id = ac.actual_instance.id
        try:
            examples_resp = datasets_client.list_examples(
                dataset=dataset.id, limit=10
            )
            example_id = examples_resp.examples[0].id

            result = datasets_client.annotate_examples(
                dataset=dataset.id,
                annotations=[
                    gen.AnnotateRecordInput(
                        record_id=example_id,
                        values=[
                            gen.AnnotationInput(name=ac_name, label="correct")
                        ],
                    )
                ],
            )

            assert len(result.results) == 1
            assert result.results[0].record_id == example_id
            result_annotation = next(
                a for a in result.results[0].annotations if a.name == ac_name
            )
            assert result_annotation.label == "correct"
        finally:
            datasets_client.delete(dataset=dataset.id)
            annotation_configs_client.delete(annotation_config=ac_id)

    def test_annotate_multiple_examples(
        self,
        datasets_client: Any,
        annotation_configs_client: Any,
        space_id: str,
    ) -> None:
        """Annotate multiple examples in a single batch request."""
        from arize._generated import api_client as gen
        from arize.annotation_configs.types import AnnotationConfigType

        ds_name = _unique("sdk-test-annot-ds")
        ac_name = _unique("sdk-test-annot-ac-cont")

        dataset = datasets_client.create(
            name=ds_name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        ac = annotation_configs_client.create(
            name=ac_name,
            space=space_id,
            config_type=AnnotationConfigType.CONTINUOUS,
            minimum_score=0.0,
            maximum_score=1.0,
        )
        ac_id = ac.actual_instance.id
        try:
            examples_resp = datasets_client.list_examples(
                dataset=dataset.id, limit=10
            )
            example_ids = [e.id for e in examples_resp.examples]
            assert len(example_ids) >= 2

            result = datasets_client.annotate_examples(
                dataset=dataset.id,
                annotations=[
                    gen.AnnotateRecordInput(
                        record_id=example_ids[0],
                        values=[gen.AnnotationInput(name=ac_name, score=1.0)],
                    ),
                    gen.AnnotateRecordInput(
                        record_id=example_ids[1],
                        values=[gen.AnnotationInput(name=ac_name, score=0.5)],
                    ),
                ],
            )

            assert len(result.results) == 2
            result_ids = {r.record_id for r in result.results}
            assert result_ids == set(example_ids[:2])
        finally:
            datasets_client.delete(dataset=dataset.id)
            annotation_configs_client.delete(annotation_config=ac_id)
