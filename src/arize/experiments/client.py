"""Client implementation for managing experiments in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import opentelemetry.sdk.trace as trace_sdk
import pandas as pd
import pyarrow as pa
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from arize._flight.client import ArizeFlightClient
from arize._flight.types import FlightRequestType
from arize._generated.api_client import models
from arize.exceptions.base import INVALID_ARROW_CONVERSION_MSG
from arize.experiments.functions import (
    run_experiment,
    transform_to_experiment_format,
)
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.cache import cache_resource, load_cached_resource
from arize.utils.openinference_conversion import (
    convert_boolean_columns_to_str,
    convert_default_columns_to_json_str,
)
from arize.utils.size import get_payload_size_mb

if TYPE_CHECKING:
    # builtins is needed to use builtins.list in type annotations because
    # the class has a list() method that shadows the built-in list type
    import builtins

    from opentelemetry.trace import Tracer

    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.experiments.evaluators.base import Evaluators
    from arize.experiments.evaluators.types import EvaluationResultFieldNames
    from arize.experiments.types import (
        ExperimentTask,
        ExperimentTaskFieldNames,
    )

logger = logging.getLogger(__name__)


class ExperimentsClient:
    """Client for managing experiments including creation, execution, and result tracking.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The experiments client is a thin wrapper around the generated REST API client,
    using the shared generated API client owned by
    :class:`arize.config.SDKConfiguration`.
    """

    def __init__(
        self, *, sdk_config: SDKConfiguration, generated_client: ApiClient
    ) -> None:
        """
        Args:
            sdk_config: Resolved SDK configuration.
            generated_client: Shared generated API client instance.
        """  # noqa: D205, D212
        self._sdk_config = sdk_config
        from arize._generated import api_client as gen

        # Use the provided client directly for both APIs
        self._api = gen.ExperimentsApi(generated_client)
        # TODO(Kiko): Space ID should not be needed,
        # should work on server tech debt to remove this
        self._datasets_api = gen.DatasetsApi(generated_client)

    @prerelease_endpoint(key="experiments.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        dataset_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.ExperimentsList200Response:
        """List experiments the user has access to.

        To filter experiments by the dataset they were run on, provide `dataset_id`.

        Args:
            dataset_id: Optional dataset ID to filter experiments.
            limit: Maximum number of experiments to return. The server enforces an
                upper bound.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the experiments and pagination information.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/429).
        """
        return self._api.experiments_list(
            dataset_id=dataset_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="experiments.create", stage=ReleaseStage.BETA)
    def create(
        self,
        *,
        name: str,
        dataset_id: str,
        experiment_runs: builtins.list[dict[str, object]] | pd.DataFrame,
        task_fields: ExperimentTaskFieldNames,
        evaluator_columns: dict[str, EvaluationResultFieldNames] | None = None,
        force_http: bool = False,
    ) -> models.Experiment:
        """Create an experiment with one or more experiment runs.

        Experiments are composed of runs. Each run must include:
            - `example_id`: ID of an existing example in the dataset/version
            - `output`: Model/task output for the matching example

        You may include any additional user-defined fields per run (e.g. `model`,
        `latency_ms`, `temperature`, `prompt`, `tool_calls`, etc.) that can be used
        for analysis or filtering.

        This method transforms the input runs into the server's expected experiment
        format using `task_fields` and optional `evaluator_columns`.

        Transport selection:
            - If the payload is below the configured REST payload threshold (or
              `force_http=True`), this method uploads via REST.
            - Otherwise, it attempts a more efficient upload path via gRPC + Flight.

        Args:
            name: Experiment name. Must be unique within the target dataset.
            dataset_id: Dataset ID to attach the experiment to.
            experiment_runs: Experiment runs either as:
                - a list of JSON-like dicts, or
                - a :class:`pandas.DataFrame`.
            task_fields: Mapping that identifies the columns/fields containing the
                task results (e.g. `example_id`, output fields).
            evaluator_columns: Optional mapping describing evaluator result columns.
            force_http: If True, force REST upload even if the payload exceeds the
                configured REST payload threshold.

        Returns:
            The created experiment object.

        Raises:
            TypeError: If `experiment_runs` is not a list of dicts or a DataFrame.
            RuntimeError: If the Flight upload path is selected and the Flight request
                fails.
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/429).
        """
        if not isinstance(experiment_runs, list | pd.DataFrame):
            raise TypeError(
                "Experiment runs must be a list of dicts or a pandas DataFrame"
            )
        # transform experiment data to experiment format
        experiment_df = transform_to_experiment_format(
            experiment_runs, task_fields, evaluator_columns
        )

        below_threshold = (
            get_payload_size_mb(experiment_runs)
            <= self._sdk_config.max_http_payload_size_mb
        )
        if below_threshold or force_http:
            from arize._generated import api_client as gen

            data = experiment_df.to_dict(orient="records")
            body = gen.ExperimentsCreateRequest(
                name=name,
                dataset_id=dataset_id,
                experiment_runs=cast("list[gen.ExperimentRunCreate]", data),
            )
            return self._api.experiments_create(experiments_create_request=body)

        # If we have too many examples, try to convert to a dataframe
        # and log via gRPC + flight
        logger.info(
            f"Uploading {len(experiment_df)} experiment runs via REST may be slow. "
            "Trying for more efficient upload via gRPC + Flight."
        )

        # TODO(Kiko): Space ID should not be needed,
        # should work on server tech debt to remove this
        dataset = self._datasets_api.datasets_get(dataset_id=dataset_id)
        space_id = dataset.space_id

        return self._create_experiment_via_flight(
            name=name,
            dataset_id=dataset_id,
            space_id=space_id,
            experiment_df=experiment_df,
        )

    @prerelease_endpoint(key="experiments.get", stage=ReleaseStage.BETA)
    def get(self, *, experiment_id: str) -> models.Experiment:
        """Get an experiment by ID.

        The response does not include the experiment's runs. Use `list_runs()` to
        retrieve runs for an experiment.

        Args:
            experiment_id: Experiment ID to retrieve.

        Returns:
            The experiment object.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        return self._api.experiments_get(experiment_id=experiment_id)

    @prerelease_endpoint(key="experiments.delete", stage=ReleaseStage.BETA)
    def delete(self, *, experiment_id: str) -> None:
        """Delete an experiment by ID.

        This operation is irreversible.

        Args:
            experiment_id: Experiment ID to delete.

        Returns:
            This method returns None on success (common empty 204 response).

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        return self._api.experiments_delete(
            experiment_id=experiment_id,
        )

    @prerelease_endpoint(key="experiments.list_runs", stage=ReleaseStage.BETA)
    def list_runs(
        self,
        *,
        experiment_id: str,
        limit: int = 100,
        all: bool = False,
    ) -> models.ExperimentsRunsList200Response:
        """List runs for an experiment.

        Runs are returned in insertion order.

        Pagination notes:
            - The response includes `pagination` for forward compatibility.
            - Cursor pagination may not be fully implemented by the server yet.
            - If `all=True`, this method retrieves all runs via the Flight path and
              returns them in a single response with `has_more=False`.

        Args:
            experiment_id: Experiment ID to list runs for.
            limit: Maximum number of runs to return when `all=False`. The server
                enforces an upper bound.
            all: If True, fetch all runs (ignores `limit`) via Flight and return a
                single response.

        Returns:
            A response object containing `experiment_runs` and `pagination` metadata.

        Raises:
            RuntimeError: If the Flight request fails or returns no response when
                `all=True`.
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response when `all=False` (e.g. 401/403/404/429).
        """
        if not all:
            return self._api.experiments_runs_list(
                experiment_id=experiment_id,
                limit=limit,
            )

        experiment = self.get(experiment_id=experiment_id)
        experiment_updated_at = getattr(experiment, "updated_at", None)
        # TODO(Kiko): Space ID should not be needed,
        # should work on server tech debt to remove this
        dataset = self._datasets_api.datasets_get(
            dataset_id=experiment.dataset_id
        )
        space_id = dataset.space_id

        experiment_df = None
        # try to load dataset from cache
        if self._sdk_config.enable_caching:
            experiment_df = load_cached_resource(
                cache_dir=self._sdk_config.cache_dir,
                resource="experiment",
                resource_id=experiment_id,
                resource_updated_at=experiment_updated_at,
            )
        if experiment_df is not None:
            return models.ExperimentsRunsList200Response(
                experiment_runs=cast(
                    "list[models.ExperimentRun]",
                    experiment_df.to_dict(orient="records"),
                ),
                pagination=models.PaginationMetadata(
                    has_more=False,  # Note that all=True
                ),
            )

        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            try:
                experiment_df = flight_client.get_experiment_runs(
                    space_id=space_id,
                    experiment_id=experiment_id,
                )
            except Exception as e:
                msg = f"Error during request: {e!s}"
                logger.exception(msg)
                raise RuntimeError(msg) from e
        if experiment_df is None:
            # This should not happen with proper Flight client implementation,
            # but we handle it defensively
            msg = "No response received from flight server during request"
            logger.error(msg)
            raise RuntimeError(msg)

        # cache experiment for future use
        cache_resource(
            cache_dir=self._sdk_config.cache_dir,
            resource="experiment",
            resource_id=experiment_id,
            resource_updated_at=experiment_updated_at,
            resource_data=experiment_df,
        )

        return models.ExperimentsRunsList200Response(
            experiment_runs=cast(
                "list[models.ExperimentRun]",
                experiment_df.to_dict(orient="records"),
            ),
            pagination=models.PaginationMetadata(
                has_more=False,  # Note that all=True
            ),
        )

    def run(
        self,
        *,
        name: str,
        dataset_id: str,
        task: ExperimentTask,
        evaluators: Evaluators | None = None,
        dry_run: bool = False,
        dry_run_count: int = 10,
        concurrency: int = 3,
        set_global_tracer_provider: bool = False,
        exit_on_error: bool = False,
        timeout: int = 120,
    ) -> tuple[models.Experiment | None, pd.DataFrame]:
        """Run an experiment on a dataset and optionally upload results.

        This method executes a task against dataset examples, optionally evaluates
        outputs, and (when `dry_run=False`) uploads results to Arize.

        High-level flow:
            1) Resolve the dataset and `space_id`.
            2) Download dataset examples (or load from cache if enabled).
            3) Run the task and evaluators with configurable concurrency.
            4) If not a dry run, upload experiment runs and return the created
                experiment plus the results dataframe.

        Notes:
            - If `dry_run=True`, no data is uploaded and the returned experiment is
              `None`.
            - When `enable_caching=True`, dataset examples may be cached and reused.

        Args:
            name: Experiment name.
            dataset_id: Dataset ID to run the experiment against.
            task: The task to execute for each dataset example.
            evaluators: Optional evaluators used to score outputs.
            dry_run: If True, do not upload results to Arize.
            dry_run_count: Number of dataset rows to use when `dry_run=True`.
            concurrency: Number of concurrent tasks to run.
            set_global_tracer_provider: If True, sets the global OpenTelemetry tracer
                provider for the experiment run.
            exit_on_error: If True, stop on the first error encountered during
                execution.
            timeout: The timeout in seconds for each task execution. Defaults to 120.

        Returns:
            If `dry_run=True`, returns `(None, results_df)`.
            If `dry_run=False`, returns `(experiment, results_df)`.

        Raises:
            RuntimeError: If Flight operations (init/download/upload) fail or return
                no response.
            pa.ArrowInvalid: If converting results to Arrow fails.
            Exception: For unexpected errors during Arrow conversion.
        """
        # TODO(Kiko): Space ID should not be needed,
        # should work on server tech debt to remove this
        dataset = self._datasets_api.datasets_get(dataset_id=dataset_id)
        space_id = dataset.space_id
        dataset_updated_at = getattr(dataset, "updated_at", None)

        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            # set up initial experiment and trace model
            if dry_run:
                trace_model_name = "traces_for_dry_run"
                experiment_id = "experiment_id_for_dry_run"
            else:
                response = None
                try:
                    response = flight_client.init_experiment(
                        space_id=space_id,
                        dataset_id=dataset_id,
                        experiment_name=name,
                    )
                except Exception as e:
                    msg = f"Error during request: {e!s}"
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

                if response is None:
                    # This should not happen with proper Flight client implementation,
                    # but we handle it defensively
                    msg = (
                        "No response received from flight server during request"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
                experiment_id, trace_model_name = response

            dataset_df = None
            # try to load dataset from cache
            if self._sdk_config.enable_caching:
                dataset_df = load_cached_resource(
                    cache_dir=self._sdk_config.cache_dir,
                    resource="dataset",
                    resource_id=dataset_id,
                    resource_updated_at=dataset_updated_at,
                )

            if dataset_df is None:
                # download dataset
                try:
                    dataset_df = flight_client.get_dataset_examples(
                        space_id=space_id,
                        dataset_id=dataset_id,
                    )
                except Exception as e:
                    msg = f"Error during request: {e!s}"
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                if dataset_df is None:
                    # This should not happen with proper Flight client implementation,
                    # but we handle it defensively
                    msg = (
                        "No response received from flight server during request"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

            if dataset_df.empty:
                raise ValueError(f"Dataset {dataset_id} is empty")

            # cache dataset for future use
            cache_resource(
                cache_dir=self._sdk_config.cache_dir,
                resource="dataset",
                resource_id=dataset_id,
                resource_updated_at=dataset_updated_at,
                resource_data=dataset_df,
            )

            if dry_run:
                # only dry_run experiment on a subset (first N rows) of the dataset
                dataset_df = dataset_df.head(dry_run_count)

            # trace model and resource for the experiment
            tracer, resource = _get_tracer_resource(
                project_name=trace_model_name,
                space_id=space_id,
                api_key=self._sdk_config.api_key,
                endpoint=self._sdk_config.otlp_url,
                dry_run=dry_run,
                set_global_tracer_provider=set_global_tracer_provider,
            )

            output_df = run_experiment(
                experiment_name=name,
                experiment_id=experiment_id,
                dataset=dataset_df,
                task=task,
                tracer=tracer,
                resource=resource,
                evaluators=evaluators,
                concurrency=concurrency,
                exit_on_error=exit_on_error,
                timeout=timeout,
            )
            output_df = convert_default_columns_to_json_str(output_df)
            output_df = convert_boolean_columns_to_str(output_df)
            if dry_run:
                return None, output_df

            # Convert to Arrow table
            try:
                logger.debug("Converting data to Arrow format")
                pa_table = pa.Table.from_pandas(output_df, preserve_index=False)
            except pa.ArrowInvalid as e:
                logger.exception(INVALID_ARROW_CONVERSION_MSG)
                raise pa.ArrowInvalid(
                    f"Error converting to Arrow format: {e!s}"
                ) from e
            except Exception:
                logger.exception("Unexpected error creating Arrow table")
                raise

            request_type = FlightRequestType.LOG_EXPERIMENT_DATA
            post_resp = None
            try:
                post_resp = flight_client.log_arrow_table(
                    space_id=space_id,
                    pa_table=pa_table,
                    dataset_id=dataset_id,
                    experiment_name=name,
                    request_type=request_type,
                )
            except Exception as e:
                msg = f"Error during update request: {e!s}"
                logger.exception(msg)
                raise RuntimeError(msg) from e

            if post_resp is None:
                # This should not happen with proper Flight client implementation,
                # but we handle it defensively
                msg = "No response received from flight server during request"
                logger.error(msg)
                raise RuntimeError(msg)

            experiment = self.get(experiment_id=str(post_resp.experiment_id))
            return experiment, output_df

    def _create_experiment_via_flight(
        self,
        name: str,
        dataset_id: str,
        space_id: str,
        experiment_df: pd.DataFrame,
    ) -> models.Experiment:
        """Internal method to create an experiment using Flight protocol for large datasets."""
        # Convert to Arrow table
        try:
            logger.debug("Converting data to Arrow format")
            pa_table = pa.Table.from_pandas(experiment_df, preserve_index=False)
        except pa.ArrowInvalid as e:
            logger.exception(INVALID_ARROW_CONVERSION_MSG)
            raise pa.ArrowInvalid(
                f"Error converting to Arrow format: {e!s}"
            ) from e
        except Exception:
            logger.exception("Unexpected error creating Arrow table")
            raise

        experiment_id = ""
        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            # set up initial experiment and trace model
            response = None
            try:
                response = flight_client.init_experiment(
                    space_id=space_id,
                    dataset_id=dataset_id,
                    experiment_name=name,
                )
            except Exception as e:
                msg = f"Error during request: {e!s}"
                logger.exception(msg)
                raise RuntimeError(msg) from e

            if response is None:
                # This should not happen with proper Flight client implementation,
                # but we handle it defensively
                msg = "No response received from flight server during request"
                logger.error(msg)
                raise RuntimeError(msg)

            experiment_id, _ = response
            if not experiment_id:
                msg = "No experiment ID received from flight server during request"
                logger.error(msg)
                raise RuntimeError(msg)

            request_type = FlightRequestType.LOG_EXPERIMENT_DATA
            post_resp = None
            try:
                post_resp = flight_client.log_arrow_table(
                    space_id=space_id,
                    pa_table=pa_table,
                    dataset_id=dataset_id,
                    experiment_name=name,
                    request_type=request_type,
                )
            except Exception as e:
                msg = f"Error during update request: {e!s}"
                logger.exception(msg)
                raise RuntimeError(msg) from e

            if post_resp is None:
                # This should not happen with proper Flight client implementation,
                # but we handle it defensively
                msg = "No response received from flight server during request"
                logger.error(msg)
                raise RuntimeError(msg)

        return self.get(experiment_id=str(post_resp.experiment_id))


def _get_tracer_resource(
    project_name: str,
    space_id: str,
    api_key: str,
    endpoint: str,
    dry_run: bool = False,
    set_global_tracer_provider: bool = False,
) -> tuple[Tracer, Resource]:
    """Initialize and return an OpenTelemetry tracer and resource for experiment tracing."""
    resource = Resource(
        {
            ResourceAttributes.PROJECT_NAME: project_name,
        }
    )
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    headers = {
        "authorization": api_key,
        "arize-space-id": space_id,
        "arize-interface": "otel",
    }
    use_tls = any(endpoint.startswith(v) for v in ["https://", "grpc+tls://"])
    insecure = not use_tls
    exporter = (
        ConsoleSpanExporter()
        if dry_run
        else GrpcSpanExporter(
            endpoint=endpoint, insecure=insecure, headers=headers
        )
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    if set_global_tracer_provider:
        trace.set_tracer_provider(tracer_provider)

    return tracer_provider.get_tracer(__name__), resource
