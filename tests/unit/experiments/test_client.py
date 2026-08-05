"""Unit tests for src/arize/experiments/client.py."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, create_autospec, patch

import pandas as pd
import pytest

from arize._generated.api_client import ExperimentsApi
from arize.experiments.client import ExperimentsClient
from arize.experiments.types import ExperimentTaskFieldNames


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
class TestList:
    """Tests for ExperimentsClient.list scoping."""

    def test_space_only_sends_space_id(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """A space without a dataset must filter by space_id — the only way to
        list experiments that aren't associated with a dataset.
        """
        with patch(
            "arize.experiments.client._find_space_id",
            return_value="space-id-123",
        ) as mock_find_space:
            experiments_client.list(space="my-space")

        mock_find_space.assert_called_once()
        kwargs = mock_api.list_experiments.call_args.kwargs
        assert kwargs["space_id"] == "space-id-123"
        assert kwargs["dataset_id"] is None

    def test_dataset_wins_when_both_given(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """The endpoint rejects both scopes, so dataset — the narrower one —
        must win and space must only resolve the dataset name.
        """
        with (
            patch(
                "arize.experiments.client._find_dataset_id",
                return_value="dataset-id-456",
            ),
            patch("arize.experiments.client._find_space_id") as mock_find_space,
        ):
            experiments_client.list(dataset="my-dataset", space="my-space")

        mock_find_space.assert_not_called()
        kwargs = mock_api.list_experiments.call_args.kwargs
        assert kwargs["dataset_id"] == "dataset-id-456"
        assert kwargs["space_id"] is None

    def test_neither_scope_sends_no_filter(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """No scope still means every experiment the caller can read."""
        experiments_client.list()

        kwargs = mock_api.list_experiments.call_args.kwargs
        assert kwargs["dataset_id"] is None
        assert kwargs["space_id"] is None

    def test_forwards_pagination_arguments(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """Limit and cursor must reach the generated client unchanged."""
        experiments_client.list(cursor="opaque-cursor", limit=25)

        kwargs = mock_api.list_experiments.call_args.kwargs
        assert kwargs["limit"] == 25
        assert kwargs["cursor"] == "opaque-cursor"


@pytest.mark.unit
class TestCreate:
    """Tests for ExperimentsClient.create."""

    def test_standalone_uses_space_id_and_skips_example_id(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
        mock_sdk_config: Mock,
    ) -> None:
        """A standalone (space-only) create must send space_id, not dataset_id,
        and must not require example_id on runs.
        """
        mock_sdk_config.max_http_payload_size_mb = 100
        mock_api.create_experiment.return_value = Mock()

        with patch(
            "arize.experiments.client._find_space_id",
            return_value="space-id-123",
        ) as mock_find_space:
            experiments_client.create(
                name="standalone-exp",
                space="my-space",
                experiment_runs=[{"output": "4"}],
                task_fields=ExperimentTaskFieldNames(output="output"),
            )

        mock_find_space.assert_called_once()
        mock_api.create_experiment.assert_called_once()
        body = mock_api.create_experiment.call_args.kwargs[
            "create_experiment_request"
        ]
        assert body.dataset_id is None
        assert body.space_id == "space-id-123"
        assert len(body.experiment_runs) == 1
        assert body.experiment_runs[0].example_id is None
        assert body.experiment_runs[0].output == "4"

    def test_dataset_backed_sets_dataset_id_not_space_id(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
        mock_sdk_config: Mock,
    ) -> None:
        """A dataset-backed create must be unaffected: dataset_id set, space_id
        absent, example_id still required and forwarded.
        """
        mock_sdk_config.max_http_payload_size_mb = 100
        mock_api.create_experiment.return_value = Mock()

        with patch(
            "arize.experiments.client._find_dataset_id",
            return_value="dataset-id-456",
        ):
            experiments_client.create(
                name="dataset-exp",
                dataset="my-dataset",
                experiment_runs=[{"example_id": "ex-1", "output": "out"}],
                task_fields=ExperimentTaskFieldNames(
                    example_id="example_id", output="output"
                ),
            )

        body = mock_api.create_experiment.call_args.kwargs[
            "create_experiment_request"
        ]
        assert body.dataset_id == "dataset-id-456"
        assert body.space_id is None
        assert body.experiment_runs[0].example_id == "ex-1"

    def test_raises_value_error_without_dataset_or_space(
        self,
        experiments_client: ExperimentsClient,
    ) -> None:
        """Neither dataset nor space is a validation error, raised before any
        API call.
        """
        with pytest.raises(ValueError, match="Either 'dataset' or 'space'"):
            experiments_client.create(
                name="no-target",
                experiment_runs=[{"output": "x"}],
                task_fields=ExperimentTaskFieldNames(output="output"),
            )


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
        experiment_obj.space_id = "space-123"

        experiment_df = pd.DataFrame(
            {
                "id": ["run-1"],
                "example_id": ["example-1"],
                "output": ['{"ok": true}'],
            }
        )

        with (
            patch.object(client, "get", return_value=experiment_obj),
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
            mock_flight_instance.get_experiment_runs.return_value = (
                experiment_df
            )
            mock_flight_cls.return_value = mock_flight_instance

            # Use a base64-encoded ID so _find_experiment_id treats it as a
            # direct resource ID and skips the name-lookup API call.
            response = client.list_runs(
                experiment="RXhwZXJpbWVudDoxMjM6YWJj", all=True
            )

        assert response.experiment_runs[0].output == '{"ok": true}'
        mock_cache_write.assert_not_called()
        mock_flight_instance.get_experiment_runs.assert_called_once_with(
            space_id="space-123",
            experiment_id="RXhwZXJpbWVudDoxMjM6YWJj",
        )

    def test_cache_write_called_when_caching_enabled(
        self, mock_sdk_config: Mock
    ) -> None:
        """list_runs(all=True) must write to cache when enable_caching=True."""
        client = self._make_client(mock_sdk_config, enable_caching=True)

        experiment_obj = Mock()
        experiment_obj.updated_at = "2024-01-01T00:00:00Z"
        experiment_obj.space_id = "space-123"

        empty_df = pd.DataFrame(columns=["id", "example_id", "output"])

        with (
            patch.object(client, "get", return_value=experiment_obj),
            patch(
                "arize.experiments.client.load_cached_resource",
                return_value=None,
            ) as mock_cache_read,
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

        mock_cache_read.assert_called_once()
        assert mock_cache_read.call_args.kwargs["resource"] == "experiment_runs"
        mock_cache_write.assert_called_once()
        assert (
            mock_cache_write.call_args.kwargs["resource"] == "experiment_runs"
        )


@pytest.mark.unit
class TestListRunsStandalone:
    """Tests for ExperimentsClient.list_runs(all=True) on standalone
    (dataset-less) experiments.
    """

    def test_uses_experiment_space_id_without_calling_get_dataset(
        self, mock_sdk_config: Mock
    ) -> None:
        """list_runs(all=True) must succeed for a standalone experiment,
        using experiment.space_id directly rather than resolving a dataset.
        """
        with (
            patch(
                "arize._generated.api_client.ExperimentsApi",
                return_value=Mock(),
            ),
            patch(
                "arize._generated.api_client.DatasetsApi", return_value=Mock()
            ),
        ):
            client = ExperimentsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        mock_sdk_config.enable_caching = False

        experiment_obj = Mock()
        experiment_obj.dataset_id = None
        experiment_obj.space_id = "space-456"
        experiment_obj.updated_at = "2024-01-01T00:00:00Z"

        empty_df = pd.DataFrame(columns=["id", "example_id", "output"])

        with (
            patch.object(client, "get", return_value=experiment_obj),
            patch.object(
                client._datasets_api, "get_dataset"
            ) as mock_get_dataset,
            patch(
                "arize.experiments.client.load_cached_resource",
                return_value=None,
            ),
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

        mock_get_dataset.assert_not_called()
        mock_flight_instance.get_experiment_runs.assert_called_once_with(
            space_id="space-456",
            experiment_id="RXhwZXJpbWVudDoxMjM6YWJj",
        )
