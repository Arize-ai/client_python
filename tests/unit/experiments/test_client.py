"""Unit tests for src/arize/experiments/client.py."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, create_autospec, patch

import pandas as pd
import pytest

from arize._generated.api_client import ExperimentsApi
from arize.experiments.client import ExperimentsClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock ExperimentsApi instance."""
    return create_autospec(ExperimentsApi, instance=True)


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
        mock_api.insert_experiment_runs.return_value = Mock()

        from arize._generated import api_client as gen

        runs = [
            gen.ExperimentRunInput(example_id="ex-1", output="result-1"),
            gen.ExperimentRunInput(example_id="ex-2", output="result-2"),
        ]
        with patch(
            "arize.experiments.client._find_experiment_id",
            return_value="exp-id-123",
        ):
            experiments_client.append_runs(
                experiment="my-experiment",
                experiment_runs=runs,
            )

        mock_api.insert_experiment_runs.assert_called_once()
        call_kwargs = mock_api.insert_experiment_runs.call_args.kwargs
        assert call_kwargs["experiment_id"] == "exp-id-123"
        body = call_kwargs["insert_experiment_runs_request"]
        assert len(body.experiment_runs) == 2
        assert body.experiment_runs[0].example_id == "ex-1"
        assert body.experiment_runs[1].example_id == "ex-2"

    def test_converts_dataframe_to_run_records(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """append_runs must convert a DataFrame to ExperimentRunInput records."""
        import pandas as pd

        mock_api.insert_experiment_runs.return_value = Mock()

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

        mock_api.insert_experiment_runs.assert_called_once()
        body = mock_api.insert_experiment_runs.call_args.kwargs[
            "insert_experiment_runs_request"
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
        """HTTP path must forward the `output` column so ExperimentRunInput validates."""
        mock_api.create_experiment.return_value = Mock()

        experiments_client._post_experiment_runs_via_http(
            name="repro-exp",
            dataset_id="ds-123",
            experiment_df=run_experiment_df,
        )

        mock_api.create_experiment.assert_called_once()
        call_kwargs = mock_api.create_experiment.call_args.kwargs
        body = call_kwargs["create_experiment_request"]
        assert len(body.experiment_runs) == 1
        assert body.experiment_runs[0].output == "pong"
        assert body.experiment_runs[0].example_id == "ex-abc"


@pytest.mark.unit
class TestListRunsCaching:
    """Tests for ExperimentsClient.list_runs() caching behaviour."""

    def _make_client(
        self, mock_sdk_config: Mock, enable_caching: bool
    ) -> ExperimentsClient:
        mock_sdk_config.enable_caching = enable_caching
        with (
            patch(
                "arize._generated.api_client.ExperimentsApi",
                return_value=Mock(),
            ),
            patch(
                "arize._generated.api_client.DatasetsApi", return_value=Mock()
            ),
        ):
            return ExperimentsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )

    def test_cache_write_skipped_when_caching_disabled(
        self, mock_sdk_config: Mock
    ) -> None:
        """list_runs(all=True) must not write to cache when enable_caching=False."""
        client = self._make_client(mock_sdk_config, enable_caching=False)

        experiment_obj = Mock()
        experiment_obj.updated_at = "2024-01-01T00:00:00Z"
        experiment_obj.dataset_id = "RGF0YXNldDoxMjM6YWJj"

        dataset_obj = Mock()
        dataset_obj.space_id = "space-123"

        empty_df = pd.DataFrame(columns=["id", "example_id", "output"])

        with (
            patch.object(client, "get", return_value=experiment_obj),
            patch.object(
                client._datasets_api, "datasets_get", return_value=dataset_obj
            ),
            patch(
                "arize.experiments.client.load_cached_resource",
                return_value=None,
            ),
            patch(
                "arize.experiments.client.cache_resource"
            ) as mock_cache_write,
            patch(
                "arize.experiments.client.ArizeFlightClient"
            ) as mock_flight_cls,
        ):
            mock_flight_instance = MagicMock()
            mock_flight_instance.__enter__ = Mock(
                return_value=mock_flight_instance
            )
            mock_flight_instance.__exit__ = Mock(return_value=False)
            mock_flight_instance.get_experiment_runs.return_value = empty_df
            mock_flight_cls.return_value = mock_flight_instance

            # Use a base64-encoded ID so _find_experiment_id treats it as a
            # direct resource ID and skips the name-lookup API call.
            client.list_runs(experiment="RXhwZXJpbWVudDoxMjM6YWJj", all=True)

        mock_cache_write.assert_not_called()

    def test_cache_write_called_when_caching_enabled(
        self, mock_sdk_config: Mock
    ) -> None:
        """list_runs(all=True) must write to cache when enable_caching=True."""
        client = self._make_client(mock_sdk_config, enable_caching=True)

        experiment_obj = Mock()
        experiment_obj.updated_at = "2024-01-01T00:00:00Z"
        experiment_obj.dataset_id = "RGF0YXNldDoxMjM6YWJj"

        dataset_obj = Mock()
        dataset_obj.space_id = "space-123"

        empty_df = pd.DataFrame(columns=["id", "example_id", "output"])

        with (
            patch.object(client, "get", return_value=experiment_obj),
            patch.object(
                client._datasets_api, "datasets_get", return_value=dataset_obj
            ),
            patch(
                "arize.experiments.client.load_cached_resource",
                return_value=None,
            ),
            patch(
                "arize.experiments.client.cache_resource"
            ) as mock_cache_write,
            patch(
                "arize.experiments.client.ArizeFlightClient"
            ) as mock_flight_cls,
        ):
            mock_flight_instance = MagicMock()
            mock_flight_instance.__enter__ = Mock(
                return_value=mock_flight_instance
            )
            mock_flight_instance.__exit__ = Mock(return_value=False)
            mock_flight_instance.get_experiment_runs.return_value = empty_df
            mock_flight_cls.return_value = mock_flight_instance

            # Use a base64-encoded ID so _find_experiment_id treats it as a
            # direct resource ID and skips the name-lookup API call.
            client.list_runs(experiment="RXhwZXJpbWVudDoxMjM6YWJj", all=True)

        mock_cache_write.assert_called_once()
