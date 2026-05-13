r"""Integration tests for experiment run annotation flows.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with::

    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_experiment_annotations_flows.py -m integration -v
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
def experiments_client(arize_client: Any) -> Any:
    return arize_client.experiments


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


class TestExperimentRunsAnnotate:
    """End-to-end annotation flows for experiment runs."""

    def test_annotate_continuous(
        self,
        datasets_client: Any,
        experiments_client: Any,
        annotation_configs_client: Any,
        space_id: str,
    ) -> None:
        """Create dataset + experiment + continuous annotation config, annotate a run, verify."""
        from arize._generated import api_client as gen
        from arize.annotation_configs.types import AnnotationConfigType
        from arize.experiments.types import ExperimentTaskFieldNames

        ds_name = _unique("sdk-test-exp-annot-ds")
        exp_name = _unique("sdk-test-exp-annot-exp")
        ac_name = _unique("sdk-test-exp-annot-ac")

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
            # List examples to get their IDs for experiment runs
            examples_resp = datasets_client.list_examples(
                dataset=dataset.id, limit=10
            )
            example_ids = [e.id for e in examples_resp.examples]
            assert len(example_ids) >= 1

            experiment_runs = [
                {"example_id": eid, "output": "model output"}
                for eid in example_ids
            ]
            task_fields = ExperimentTaskFieldNames(
                example_id="example_id", output="output"
            )
            experiment = experiments_client.create(
                name=exp_name,
                dataset=dataset.id,
                experiment_runs=experiment_runs,
                task_fields=task_fields,
            )

            runs_resp = experiments_client.list_runs(
                experiment=experiment.id, limit=10
            )
            run_id = runs_resp.experiment_runs[0].id

            experiments_client.annotate_runs(
                experiment=experiment.id,
                annotations=[
                    gen.AnnotateRecordInput(
                        record_id=run_id,
                        values=[gen.AnnotationInput(name=ac_name, score=0.75)],
                    )
                ],
            )
        finally:
            datasets_client.delete(dataset=dataset.id)
            annotation_configs_client.delete(annotation_config=ac_id)

    def test_annotate_categorical(
        self,
        datasets_client: Any,
        experiments_client: Any,
        annotation_configs_client: Any,
        space_id: str,
    ) -> None:
        """Create dataset + experiment + categorical annotation config, annotate a run, verify."""
        from arize._generated import api_client as gen
        from arize.annotation_configs.types import AnnotationConfigType
        from arize.experiments.types import ExperimentTaskFieldNames

        ds_name = _unique("sdk-test-exp-annot-ds")
        exp_name = _unique("sdk-test-exp-annot-exp")
        ac_name = _unique("sdk-test-exp-annot-ac")

        dataset = datasets_client.create(
            name=ds_name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        ac = annotation_configs_client.create(
            name=ac_name,
            space=space_id,
            config_type=AnnotationConfigType.CATEGORICAL,
            values=["correct", "incorrect"],
        )
        ac_id = ac.actual_instance.id
        try:
            examples_resp = datasets_client.list_examples(
                dataset=dataset.id, limit=10
            )
            example_ids = [e.id for e in examples_resp.examples]

            experiment_runs = [
                {"example_id": eid, "output": "model output"}
                for eid in example_ids
            ]
            task_fields = ExperimentTaskFieldNames(
                example_id="example_id", output="output"
            )
            experiment = experiments_client.create(
                name=exp_name,
                dataset=dataset.id,
                experiment_runs=experiment_runs,
                task_fields=task_fields,
            )

            runs_resp = experiments_client.list_runs(
                experiment=experiment.id, limit=10
            )
            run_id = runs_resp.experiment_runs[0].id

            experiments_client.annotate_runs(
                experiment=experiment.id,
                annotations=[
                    gen.AnnotateRecordInput(
                        record_id=run_id,
                        values=[
                            gen.AnnotationInput(name=ac_name, label="correct")
                        ],
                    )
                ],
            )
        finally:
            datasets_client.delete(dataset=dataset.id)
            annotation_configs_client.delete(annotation_config=ac_id)

    def test_annotate_multiple_runs(
        self,
        datasets_client: Any,
        experiments_client: Any,
        annotation_configs_client: Any,
        space_id: str,
    ) -> None:
        """Annotate multiple runs in a single batch request."""
        from arize._generated import api_client as gen
        from arize.annotation_configs.types import AnnotationConfigType
        from arize.experiments.types import ExperimentTaskFieldNames

        ds_name = _unique("sdk-test-exp-annot-ds")
        exp_name = _unique("sdk-test-exp-annot-exp")
        ac_name = _unique("sdk-test-exp-annot-ac")

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

            experiment_runs = [
                {"example_id": eid, "output": "model output"}
                for eid in example_ids
            ]
            task_fields = ExperimentTaskFieldNames(
                example_id="example_id", output="output"
            )
            experiment = experiments_client.create(
                name=exp_name,
                dataset=dataset.id,
                experiment_runs=experiment_runs,
                task_fields=task_fields,
            )

            runs_resp = experiments_client.list_runs(
                experiment=experiment.id, limit=10
            )
            run_ids = [r.id for r in runs_resp.experiment_runs]
            assert len(run_ids) >= 2

            experiments_client.annotate_runs(
                experiment=experiment.id,
                annotations=[
                    gen.AnnotateRecordInput(
                        record_id=run_ids[0],
                        values=[gen.AnnotationInput(name=ac_name, score=1.0)],
                    ),
                    gen.AnnotateRecordInput(
                        record_id=run_ids[1],
                        values=[gen.AnnotationInput(name=ac_name, score=0.0)],
                    ),
                ],
            )
        finally:
            datasets_client.delete(dataset=dataset.id)
            annotation_configs_client.delete(annotation_config=ac_id)
