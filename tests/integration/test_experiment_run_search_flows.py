r"""Integration tests for experiment run search flows.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with::

    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_experiment_run_search_flows.py -m integration -v
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


class TestExperimentRunsSearch:
    """End-to-end search flows for experiment runs, via list_runs()."""

    def test_list_runs_no_filter_returns_all_runs(
        self,
        datasets_client: Any,
        experiments_client: Any,
    ) -> None:
        """Create dataset + experiment, list with no filter, verify all runs."""
        from arize.experiments.types import ExperimentTaskFieldNames

        ds_name = _unique("sdk-test-exp-search-ds")
        exp_name = _unique("sdk-test-exp-search-exp")

        dataset = datasets_client.create(
            name=ds_name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            examples_resp = datasets_client.list_examples(
                dataset=dataset.id, limit=10
            )
            example_ids = [e.id for e in examples_resp.examples]
            assert len(example_ids) >= 1

            experiment_runs = [
                {"example_id": eid, "output": ex["output"]}
                for eid, ex in zip(example_ids, _EXAMPLES, strict=False)
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

            list_resp = experiments_client.list_runs(
                experiment=experiment.id, limit=10
            )
            assert len({r.id for r in list_resp.experiment_runs}) == len(
                experiment_runs
            )
        finally:
            datasets_client.delete(dataset=dataset.id)

    def test_list_runs_with_filter_narrows_results(
        self,
        datasets_client: Any,
        experiments_client: Any,
    ) -> None:
        """Create dataset + experiment, list with a filter, verify narrowing."""
        from arize.experiments.types import ExperimentTaskFieldNames

        ds_name = _unique("sdk-test-exp-search-ds")
        exp_name = _unique("sdk-test-exp-search-exp")

        dataset = datasets_client.create(
            name=ds_name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            examples_resp = datasets_client.list_examples(
                dataset=dataset.id, limit=10
            )
            example_ids = [e.id for e in examples_resp.examples]
            assert len(example_ids) >= 2

            experiment_runs = [
                {"example_id": eid, "output": ex["output"]}
                for eid, ex in zip(example_ids, _EXAMPLES, strict=False)
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

            filtered_resp = experiments_client.list_runs(
                experiment=experiment.id,
                filter="output = '4'",
                limit=10,
            )
            assert len(filtered_resp.experiment_runs) == 1
            assert filtered_resp.experiment_runs[0].output == "4"
        finally:
            datasets_client.delete(dataset=dataset.id)

    def test_list_runs_pagination_with_cursor(
        self,
        datasets_client: Any,
        experiments_client: Any,
    ) -> None:
        """List with limit=1 and page via cursor without duplicates or gaps."""
        from arize.experiments.types import ExperimentTaskFieldNames

        ds_name = _unique("sdk-test-exp-search-ds")
        exp_name = _unique("sdk-test-exp-search-exp")

        dataset = datasets_client.create(
            name=ds_name,
            space=SPACE_NAME,
            examples=_EXAMPLES,
        )
        try:
            examples_resp = datasets_client.list_examples(
                dataset=dataset.id, limit=10
            )
            example_ids = [e.id for e in examples_resp.examples]
            assert len(example_ids) >= 2

            experiment_runs = [
                {"example_id": eid, "output": ex["output"]}
                for eid, ex in zip(example_ids, _EXAMPLES, strict=False)
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

            list_resp = experiments_client.list_runs(
                experiment=experiment.id, limit=10
            )
            expected_ids = {r.id for r in list_resp.experiment_runs}
            assert len(expected_ids) >= 2

            page1 = experiments_client.list_runs(
                experiment=experiment.id, limit=1
            )
            assert len(page1.experiment_runs) == 1
            assert page1.pagination.has_more
            assert page1.pagination.next_cursor is not None

            page2 = experiments_client.list_runs(
                experiment=experiment.id,
                limit=1,
                cursor=page1.pagination.next_cursor,
            )
            assert len(page2.experiment_runs) == 1

            page1_ids = {r.id for r in page1.experiment_runs}
            page2_ids = {r.id for r in page2.experiment_runs}
            assert page1_ids.isdisjoint(page2_ids)
            assert page1_ids | page2_ids == expected_ids
        finally:
            datasets_client.delete(dataset=dataset.id)
