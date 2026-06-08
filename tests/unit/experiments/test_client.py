"""Unit tests for src/arize/experiments/client.py."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from arize.experiments.client import ExperimentsClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock ExperimentsApi instance."""
    return Mock()


@pytest.fixture
def experiments_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> ExperimentsClient:
    """Provide an ExperimentsClient with mocked internals."""
    with (
        patch(
            "arize._generated.api_client.ExperimentsApi", return_value=mock_api
        ),
        patch("arize._generated.api_client.DatasetsApi", return_value=Mock()),
    ):
        return ExperimentsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.fixture
def run_experiment_df() -> pd.DataFrame:
    """Dataframe shaped like the output of run_experiment() in functions.py."""
    df = pd.DataFrame(
        {
            "id": ["run-1"],
            "example_id": ["ex-abc"],
            "output": ["pong"],
            "error": [None],
            "result.trace.id": ["trace-1"],
            "result.trace.timestamp": [1700000000000],
        }
    )
    df.set_index("id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@pytest.mark.unit
class TestAppendRuns:
    """Tests for ExperimentsClient.append_runs."""

    def test_calls_experiments_runs_insert_with_correct_body(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """append_runs must forward runs to experiments_runs_insert."""
        mock_api.experiments_runs_insert.return_value = Mock()

        from arize._generated import api_client as gen

        runs = [
            gen.ExperimentRunCreate(example_id="ex-1", output="result-1"),
            gen.ExperimentRunCreate(example_id="ex-2", output="result-2"),
        ]
        with patch(
            "arize.experiments.client._find_experiment_id",
            return_value="exp-id-123",
        ):
            experiments_client.append_runs(
                experiment="my-experiment",
                experiment_runs=runs,
            )

        mock_api.experiments_runs_insert.assert_called_once()
        call_kwargs = mock_api.experiments_runs_insert.call_args.kwargs
        assert call_kwargs["experiment_id"] == "exp-id-123"
        body = call_kwargs["insert_experiment_runs_body"]
        assert len(body.experiment_runs) == 2
        assert body.experiment_runs[0].example_id == "ex-1"
        assert body.experiment_runs[1].example_id == "ex-2"

    def test_converts_dataframe_to_run_records(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """append_runs must convert a DataFrame to ExperimentRunCreate records."""
        import pandas as pd

        mock_api.experiments_runs_insert.return_value = Mock()

        df = pd.DataFrame(
            {"example_id": ["ex-a", "ex-b"], "output": ["out-a", "out-b"]}
        )
        with patch(
            "arize.experiments.client._find_experiment_id",
            return_value="exp-id-456",
        ):
            experiments_client.append_runs(
                experiment="exp-id-456",
                experiment_runs=df,
            )

        mock_api.experiments_runs_insert.assert_called_once()
        body = mock_api.experiments_runs_insert.call_args.kwargs[
            "insert_experiment_runs_body"
        ]
        assert len(body.experiment_runs) == 2
        assert body.experiment_runs[0].output == "out-a"
        assert body.experiment_runs[1].output == "out-b"


@pytest.mark.unit
class TestPostExperimentRunsViaHttp:
    """Tests for ExperimentsClient._post_experiment_runs_via_http."""

    def test_forwards_output_column_to_request(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
        run_experiment_df: pd.DataFrame,
    ) -> None:
        """HTTP path must forward the `output` column so ExperimentRunCreate validates."""
        mock_api.experiments_create.return_value = Mock()

        experiments_client._post_experiment_runs_via_http(
            name="repro-exp",
            dataset_id="ds-123",
            experiment_df=run_experiment_df,
        )

        mock_api.experiments_create.assert_called_once()
        call_kwargs = mock_api.experiments_create.call_args.kwargs
        body = call_kwargs["experiments_create_request"]
        assert len(body.experiment_runs) == 1
        assert body.experiment_runs[0].output == "pong"
        assert body.experiment_runs[0].example_id == "ex-abc"
