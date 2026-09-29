"""Unit tests for src/arize/experiments/client.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, create_autospec, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arize._generated.api_client import ExperimentsApi
from arize.experiments.client import ExperimentsClient
from arize.experiments.types import ExperimentTaskFieldNames

if TYPE_CHECKING:
    from pathlib import Path


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


_EXPERIMENT_ID = "RXhwZXJpbWVudDoxMjM6YWJj"


@pytest.mark.unit
class TestListRuns:
    """Tests for ExperimentsClient.list_runs()."""

    def test_list_runs_with_filter_calls_search_experiment_runs(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """Filtered list_runs must POST search_experiment_runs with the request body."""
        from arize._generated import api_client as gen

        expected = Mock()
        mock_api.search_experiment_runs.return_value = expected

        result = experiments_client.list_runs(
            experiment=_EXPERIMENT_ID,
            filter="output = 'pong'",
            limit=25,
            cursor="cursor-abc",
        )

        mock_api.search_experiment_runs.assert_called_once_with(
            experiment_id=_EXPERIMENT_ID,
            search_experiment_runs_request=gen.SearchExperimentRunsRequest(
                filter="output = 'pong'",
                limit=25,
                cursor="cursor-abc",
            ),
        )
        assert result is expected

    def test_list_runs_filter_with_all_raises_value_error(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
    ) -> None:
        """all=True with filter must fail before calling the API."""
        with pytest.raises(
            ValueError, match="filter is not supported with all=True"
        ):
            experiments_client.list_runs(
                experiment=_EXPERIMENT_ID,
                filter="output = 'pong'",
                all=True,
            )

        mock_api.search_experiment_runs.assert_not_called()


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


@pytest.mark.unit
class TestCreateFromFiles:
    """Tests for ExperimentsClient.create() with file-path runs."""

    TASK = ExperimentTaskFieldNames(example_id="example_id", output="output")

    @pytest.fixture(autouse=True)
    def _configure(
        self, mock_sdk_config: Mock, experiments_client: ExperimentsClient
    ) -> None:
        mock_sdk_config.pyarrow_max_chunksize = 10
        experiments_client._datasets_api.get_dataset.return_value = Mock(
            space_id="space-1"
        )

    @pytest.fixture
    def flight_client(self) -> MagicMock:
        client = MagicMock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.init_experiment.return_value = ("exp-1", "trace-project")
        client.written = []

        def log_arrow_table(**kwargs: object) -> Mock:
            client.written.extend(kwargs["reader"])  # type: ignore[attr-defined]
            return Mock(experiment_id="exp-1")

        client.log_arrow_table.side_effect = log_arrow_table
        return client

    @pytest.fixture
    def runs_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "runs.parquet"
        pq.write_table(
            pa.table(
                {
                    "example_id": ["e1", "e2", "e3"],
                    "output": [{"answer": 1}, {"answer": 2}, {"answer": 3}],
                }
            ),
            path,
        )
        return path

    def test_file_input_inits_then_streams(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
        flight_client: MagicMock,
        runs_file: Path,
    ) -> None:
        with (
            patch(
                "arize.experiments.client._find_dataset_id",
                return_value="dataset-1",
            ),
            patch(
                "arize.experiments.client.ArizeFlightClient",
                return_value=flight_client,
            ),
            patch.object(ExperimentsClient, "get", return_value=Mock()) as get,
        ):
            experiments_client.create(
                name="exp",
                dataset="ds",
                experiment_runs=runs_file,
                task_fields=self.TASK,
            )
        mock_api.create_experiment.assert_not_called()
        flight_client.init_experiment.assert_called_once_with(
            space_id="space-1", dataset_id="dataset-1", experiment_name="exp"
        )
        kwargs = flight_client.log_arrow_table.call_args.kwargs
        assert kwargs["dataset_id"] == "dataset-1"
        table = pa.Table.from_batches(flight_client.written)
        assert table.schema.equals(kwargs["reader"].schema)
        assert table.column("example_id").to_pylist() == ["e1", "e2", "e3"]
        assert table.column("output").to_pylist() == [
            '{"answer": 1}',
            '{"answer": 2}',
            '{"answer": 3}',
        ]
        get.assert_called_once_with(experiment="exp-1")

    def test_force_http_with_path_raises_before_resolution(
        self, experiments_client: ExperimentsClient, runs_file: Path
    ) -> None:
        with (
            patch("arize.experiments.client._find_dataset_id") as find,
            pytest.raises(ValueError, match="force_http"),
        ):
            experiments_client.create(
                name="exp",
                dataset="ds",
                experiment_runs=runs_file,
                task_fields=self.TASK,
                force_http=True,
            )
        find.assert_not_called()

    def test_get_keeps_prerelease_decorator(self) -> None:
        assert hasattr(ExperimentsClient.get, "__wrapped__")

    def test_path_without_dataset_raises(
        self, experiments_client: ExperimentsClient, runs_file: Path
    ) -> None:
        with (
            patch("arize.experiments.client._find_space_id") as find,
            pytest.raises(ValueError, match="require `dataset`"),
        ):
            experiments_client.create(
                name="exp",
                space="sp",
                experiment_runs=str(runs_file),
                task_fields=ExperimentTaskFieldNames(output="output"),
            )
        find.assert_not_called()

    @pytest.mark.parametrize(
        ("table", "match"),
        [
            (
                pa.table(
                    {
                        "example_id": pa.array([], pa.string()),
                        "output": pa.array([], pa.string()),
                    }
                ),
                "no rows",
            ),
            (pa.table({"example_id": ["e1"]}), "Missing required columns"),
        ],
        ids=["empty", "missing-output"],
    )
    def test_pre_stream_checks_fail_before_init(
        self,
        experiments_client: ExperimentsClient,
        tmp_path: Path,
        table: pa.Table,
        match: str,
    ) -> None:
        path = tmp_path / "bad.parquet"
        pq.write_table(table, path)
        with (
            patch(
                "arize.experiments.client._find_dataset_id",
                return_value="dataset-1",
            ),
            patch("arize.experiments.client.ArizeFlightClient") as flight_cls,
            pytest.raises(ValueError, match=match),
        ):
            experiments_client.create(
                name="exp",
                dataset="ds",
                experiment_runs=path,
                task_fields=self.TASK,
            )
        flight_cls.assert_not_called()


@pytest.mark.unit
class TestRunFlightUpload:
    """run() uploads its results as record batches over Flight."""

    def test_run_streams_output_batches(
        self,
        experiments_client: ExperimentsClient,
        mock_api: Mock,
        mock_sdk_config: Mock,
        run_experiment_df: pd.DataFrame,
    ) -> None:
        mock_sdk_config.enable_caching = False
        mock_sdk_config.pyarrow_max_chunksize = 10
        experiments_client._datasets_api.get_dataset.return_value = Mock(
            space_id="space-1", updated_at=None
        )
        mock_api.get_experiment.return_value = Mock()
        flight_client = MagicMock()
        flight_client.__enter__ = Mock(return_value=flight_client)
        flight_client.__exit__ = Mock(return_value=False)
        written: list[pa.RecordBatch] = []

        def log_arrow_table(**kwargs: object) -> Mock:
            written.extend(kwargs["reader"])  # type: ignore[arg-type]
            return Mock(experiment_id="exp-1")

        flight_client.log_arrow_table.side_effect = log_arrow_table
        output_df = pd.concat([run_experiment_df] * 25, ignore_index=True)

        with (
            patch(
                "arize.experiments.client._find_dataset_id",
                return_value="dataset-1",
            ),
            patch.object(
                ExperimentsClient,
                "_init_experiment_via_flight",
                return_value=("exp-1", "trace-project"),
            ),
            patch.object(
                ExperimentsClient,
                "_get_dataset_examples_via_flight",
                return_value=pd.DataFrame({"id": ["ex-abc"], "q": ["ping"]}),
            ),
            patch(
                "arize.experiments.client._get_tracer_resource",
                return_value=(Mock(), Mock(), Mock()),
            ),
            patch(
                "arize.experiments.client.run_experiment",
                return_value=output_df,
            ),
            patch(
                "arize.experiments.client.ArizeFlightClient",
                return_value=flight_client,
            ),
            patch.object(ExperimentsClient, "get", return_value=Mock()) as get,
        ):
            _, result_df = experiments_client.run(
                name="exp", dataset="ds", task=lambda example: "pong"
            )

        assert result_df is output_df
        kwargs = flight_client.log_arrow_table.call_args.kwargs
        assert kwargs["dataset_id"] == "dataset-1"
        assert kwargs["space_id"] == "space-1"
        assert [b.num_rows for b in written] == [10, 10, 5]
        assert all(b.schema.equals(kwargs["reader"].schema) for b in written)
        assert (
            pa.Table.from_batches(written).column("output").to_pylist()
            == ["pong"] * 25
        )
        get.assert_called_once_with(experiment="exp-1")
