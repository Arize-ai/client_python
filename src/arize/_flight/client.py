from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from google.protobuf import json_format
from pyarrow import flight

from arize._flight.types import FlightRequestType
from arize._generated.protocol.flight import flight_pb2
from arize.config import PYTHON_VERSION
from arize.logging import log_a_list
from arize.utils.openinference_conversion import convert_json_str_to_dict
from arize.utils.proto import get_pb_schema_tracing
from arize.version import __version__

if TYPE_CHECKING:
    import types
    from collections.abc import Iterator

    import pandas as pd
    import pyarrow as pa


BytesPair: TypeAlias = tuple[bytes, bytes]
Headers: TypeAlias = list[BytesPair]
FlightPostArrowFileResponse: TypeAlias = (
    flight_pb2.WriteSpanEvaluationResponse
    | flight_pb2.WriteSpanAnnotationResponse
    | flight_pb2.WriteSpanAttributesMetadataResponse
    | flight_pb2.PostExperimentDataResponse
)

logger = logging.getLogger(__name__)


class FlightActionKey(Enum):
    CREATE_EXPERIMENT_DB_ENTRY = "create_experiment_db_entry"
    # GET_DATASET_VERSION = "get_dataset_version"
    # LIST_DATASETS = "list_datasets"
    # DELETE_DATASET = "delete_dataset"
    # DELETE_EXPERIMENT = "delete_experiment"


@dataclass(frozen=True)
class ArizeFlightClient:
    """Low-level Apache Arrow Flight client for Arize data operations.

    This client provides the underlying Flight RPC transport for uploading and
    downloading data to/from Arize using Apache Arrow format. It's used internally
    by public-facing clients (SpansClient, ExperimentsClient, DatasetsClient, etc.)
    and is not intended for direct use by end users.

    The client manages connections, authentication, and protocol buffer serialization
    for Flight operations.
    """

    api_key: str = field(repr=False)
    host: str
    port: int
    scheme: str
    max_chunksize: int
    request_verify: bool

    # internal cache for the underlying FlightClient
    _client: flight.FlightClient | None = field(
        default=None, init=False, repr=False
    )

    # ---------- Properties ----------

    @property
    def headers(self) -> Headers:
        return [
            (b"origin", b"arize-logging-client"),
            (b"auth-token-bin", str(self.api_key).encode("utf-8")),
            (b"sdk-language", b"python"),
            (b"language-version", PYTHON_VERSION.encode("utf-8")),
            (b"sdk-version", __version__.encode("utf-8")),
        ]

    @property
    def call_options(self) -> flight.FlightCallOptions:
        return flight.FlightCallOptions(headers=self.headers)

    # ---------- Connection management ----------

    def _ensure_client(self) -> flight.FlightClient:
        """Lazily initialize and return the underlying Flight client connection.

        Returns:
            flight.FlightClient: The initialized Apache Arrow Flight client.
        """
        client = object.__getattribute__(self, "_client")
        if client is not None:
            return client

        # disable TLS verification for local dev on localhost, or if user opts out
        disable_cert = (
            self.request_verify is False or self.host.lower() == "localhost"
        )

        new_client = flight.FlightClient(
            location=f"{self.scheme}://{self.host}:{self.port}",
            disable_server_verification=disable_cert,
        )
        object.__setattr__(self, "_client", new_client)
        return new_client

    def close(self) -> None:
        """Close the Flight client connection and clean up resources."""
        client = object.__getattribute__(self, "_client")
        if client is not None:
            client.close()
            object.__setattr__(self, "_client", None)

    # ---------- Context manager ----------

    def __enter__(self) -> ArizeFlightClient:
        """Context manager entry point. Ensures the Flight client is initialized."""
        self._ensure_client()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _: types.TracebackType | None,
    ) -> None:
        """Context manager exit point. Closes the Flight client connection."""
        if exc_type:
            logger.error(f"An exception occurred: {exc_val}")
        self.close()

    # ---------- methods simple passthrough wrappers ----------

    def get_flight_info(self, *args: object, **kwargs: object) -> object:
        """Get flight information. Passthrough to underlying Flight client with auth options.

        Returns:
            object: FlightInfo object containing metadata about the requested data stream.
        """
        client = self._ensure_client()
        kwargs.setdefault("options", self.call_options)
        return client.get_flight_info(*args, **kwargs)

    def do_get(
        self, *args: object, **kwargs: object
    ) -> flight.FlightStreamReader:
        """Retrieve data stream via Flight DoGet.

        Passthrough to underlying Flight client with auth options.

        Returns:
            flight.FlightStreamReader: A stream reader for retrieving Arrow record batches.
        """
        client = self._ensure_client()
        kwargs.setdefault("options", self.call_options)
        return client.do_get(*args, **kwargs)

    def do_put(
        self, *args: object, **kwargs: object
    ) -> tuple[flight.FlightStreamWriter, flight.FlightMetadataReader]:
        """Upload data stream via Flight DoPut.

        Passthrough to underlying Flight client with auth options.

        Returns:
            tuple[flight.FlightStreamWriter, flight.FlightMetadataReader]: A tuple containing
                a stream writer for uploading Arrow record batches and a metadata reader for
                receiving server responses.
        """
        client = self._ensure_client()
        kwargs.setdefault("options", self.call_options)
        return client.do_put(*args, **kwargs)

    def do_action(
        self, *args: object, **kwargs: object
    ) -> Iterator[flight.Result]:
        """Execute an action via Flight DoAction.

        Passthrough to underlying Flight client with auth options.

        Returns:
            Iterable[flight.Result]: An iterable of Result objects from the action execution.
        """
        client = self._ensure_client()
        kwargs.setdefault("options", self.call_options)
        return client.do_action(*args, **kwargs)

    # ---------- logging methods ----------

    def log_arrow_table(
        self,
        space_id: str,
        request_type: FlightRequestType,
        pa_table: pa.Table,
        project_name: str | None = None,
        dataset_id: str | None = None,
        experiment_name: str | None = None,
    ) -> FlightPostArrowFileResponse:
        """Upload an Arrow table via Flight DoPut.

        Args:
            space_id: Space ID for the request.
            request_type: Type of Flight request (EVALUATION, ANNOTATION, METADATA, or
                LOG_EXPERIMENT_DATA).
            pa_table: PyArrow Table containing the data to upload.
            project_name: Project/model name (required for tracing request types).
            dataset_id: Dataset ID (required for experiment request types).
            experiment_name: Experiment name (required for experiment request types).

        Returns:
            FlightPostArrowFileResponse containing the server response, including counts
            of processed records and any errors.

        Raises:
            ValueError: If request_type is unsupported or required parameters are missing
                for the request type.
            RuntimeError: If the Flight request fails.
        """
        pa_schema = pa_table.schema
        if request_type in (
            FlightRequestType.EVALUATION,
            FlightRequestType.ANNOTATION,
            FlightRequestType.METADATA,
        ):
            if project_name is None:
                raise ValueError(
                    f"project_name is required for {request_type.name} request type"
                )
            proto_schema = get_pb_schema_tracing(project_name=project_name)
            base64_schema = base64.b64encode(proto_schema.SerializeToString())
            pa_schema = append_to_pyarrow_metadata(
                pa_table.schema, {"arize-schema": base64_schema}
            )

        doput_request = _get_pb_flight_doput_request(
            space_id=space_id,
            request_type=request_type,
            model_id=project_name,
            dataset_id=dataset_id,
            experiment_name=experiment_name,
        )

        descriptor = flight.FlightDescriptor.for_command(
            json_format.MessageToJson(doput_request).encode("utf-8")
        )
        try:
            flight_writer, flight_metadata_reader = self.do_put(
                descriptor, pa_schema, options=self.call_options
            )
            with flight_writer:
                # write table as stream to flight server
                flight_writer.write_table(pa_table, self.max_chunksize)
                # indicate that client has flushed all contents to stream
                flight_writer.done_writing()
                # read response from flight server
                flight_response = flight_metadata_reader.read()
                if flight_response is None:
                    return None

        except Exception as e:
            logger.exception("Error logging arrow table to Arize")
            raise RuntimeError(
                f"Error logging arrow table to Arize: {e}"
            ) from e

        match request_type:
            case FlightRequestType.EVALUATION:
                res = flight_pb2.WriteSpanEvaluationResponse()
                res.ParseFromString(flight_response.to_pybytes())
            case FlightRequestType.ANNOTATION:
                res = flight_pb2.WriteSpanAnnotationResponse()
                res.ParseFromString(flight_response.to_pybytes())
            case FlightRequestType.METADATA:
                res = flight_pb2.WriteSpanAttributesMetadataResponse()
                res.ParseFromString(flight_response.to_pybytes())
            case FlightRequestType.LOG_EXPERIMENT_DATA:
                res = flight_pb2.PostExperimentDataResponse()
                res.ParseFromString(flight_response.to_pybytes())
        return res

    # ---------- dataset methods ----------

    def create_dataset(
        self,
        space_id: str,
        dataset_name: str,
        pa_table: pa.Table,
    ) -> str | None:
        """Create a new dataset via Flight DoPut.

        Args:
            space_id: Space ID where the dataset will be created.
            dataset_name: Name for the new dataset.
            pa_table: PyArrow Table containing the dataset examples.

        Returns:
            The created dataset ID as a string.

        Raises:
            RuntimeError: If the Flight request fails.
        """
        doput_request = flight_pb2.DoPutRequest(
            create_dataset=flight_pb2.CreateDatasetRequest(
                space_id=space_id,
                dataset_name=dataset_name,
                dataset_type=flight_pb2.GENERATIVE,
            )
        )
        descriptor = flight.FlightDescriptor.for_command(
            json_format.MessageToJson(doput_request).encode("utf-8")
        )
        try:
            flight_writer, flight_metadata_reader = self.do_put(
                descriptor, pa_table.schema, options=self.call_options
            )
            with flight_writer:
                # write table as stream to flight server
                flight_writer.write_table(pa_table, self.max_chunksize)
                # indicate that client has flushed all contents to stream
                flight_writer.done_writing()
                # read response from flight server
                flight_response = flight_metadata_reader.read()
                if flight_response is None:
                    return None

                res = flight_pb2.CreateDatasetResponse()
                res.ParseFromString(flight_response.to_pybytes())
                return str(res.dataset_id) if res else None
        except Exception as e:
            logger.exception("Error logging arrow table to Arize")
            raise RuntimeError(
                f"Error logging arrow table to Arize: {e}"
            ) from e

    def get_dataset_examples(
        self,
        space_id: str,
        dataset_id: str,
        dataset_version_id: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve dataset examples via Flight DoGet.

        Args:
            space_id: Space ID containing the dataset.
            dataset_id: Dataset ID to retrieve examples from.
            dataset_version_id: Optional specific version ID. If None, retrieves the
                latest version.

        Returns:
            :class:`pandas.DataFrame`: A pandas DataFrame containing the dataset examples
                with JSON string columns converted to dict objects.

        Raises:
            RuntimeError: If the Flight request fails.
        """
        # TODO(Kiko): Space ID should not be needed,
        # should work on server tech debt to remove this
        doget_request = flight_pb2.DoGetRequest(
            get_dataset=flight_pb2.GetDatasetRequest(
                space_id=space_id,
                dataset_id=dataset_id,
                dataset_version=dataset_version_id,
            )
        )
        descriptor = flight.Ticket(
            json_format.MessageToJson(doget_request).encode("utf-8")
        )
        try:
            reader = self.do_get(descriptor, options=self.call_options)
            # read all data into pandas dataframe
            df = reader.read_all().to_pandas()
            return convert_json_str_to_dict(df)
        except Exception as e:
            logger.exception(f"Failed to get dataset id={dataset_id}")
            raise RuntimeError(f"Failed to get dataset id={dataset_id}") from e

    # ---------- experiment methods ----------

    def get_experiment_runs(
        self,
        space_id: str,
        experiment_id: str,
    ) -> pd.DataFrame:
        """Retrieve experiment runs via Flight DoGet.

        Args:
            space_id: Space ID containing the experiment.
            experiment_id: Experiment ID to retrieve runs from.

        Returns:
            :class:`pandas.DataFrame`: A pandas DataFrame containing the experiment runs
                with JSON string columns converted to dict objects.

        Raises:
            RuntimeError: If the Flight request fails.
        """
        # TODO(Kiko): Space ID should not be needed,
        # should work on server tech debt to remove this
        doget_request = flight_pb2.DoGetRequest(
            get_experiment=flight_pb2.GetExperimentRequest(
                space_id=space_id,
                experiment_id=experiment_id,
            )
        )
        descriptor = flight.Ticket(
            json_format.MessageToJson(doget_request).encode("utf-8")
        )
        try:
            reader = self.do_get(descriptor, options=self.call_options)
            # read all data into pandas dataframe
            df = reader.read_all().to_pandas()
            return convert_json_str_to_dict(df)
        except Exception as e:
            logger.exception(f"Failed to get experiment id={experiment_id}")
            raise RuntimeError(
                f"Failed to get experiment id={experiment_id}"
            ) from e

    def init_experiment(
        self,
        space_id: str,
        dataset_id: str,
        experiment_name: str,
    ) -> tuple[str, str] | None:
        """Initialize a new experiment via Flight DoAction.

        Creates database entries for a new experiment and allocates a trace model for
        capturing experiment traces.

        Args:
            space_id: Space ID where the experiment will be created.
            dataset_id: Dataset ID the experiment will run against.
            experiment_name: Name for the new experiment.

        Returns:
            A tuple of (experiment_id, trace_model_name) if successful, None if the
            action response is empty.

        Raises:
            RuntimeError: If the Flight action fails.
        """
        request = flight_pb2.DoActionRequest(
            create_experiment_db_entry=flight_pb2.CreateExperimentDBEntryRequest(
                space_id=space_id,
                dataset_id=dataset_id,
                experiment_name=experiment_name,
            )
        )
        action = flight.Action(
            FlightActionKey.CREATE_EXPERIMENT_DB_ENTRY.value,
            json_format.MessageToJson(request).encode("utf-8"),
        )
        try:
            response = self.do_action(action, options=self.call_options)
        except Exception as e:
            logger.exception(f"Failed to init experiment {experiment_name}")
            raise RuntimeError(
                f"Failed to init experiment {experiment_name}"
            ) from e

        res = next(response, None)
        if res is None:
            return None
        resp_pb = flight_pb2.CreateExperimentDBEntryResponse()
        resp_pb.ParseFromString(res.body.to_pybytes())
        return (
            resp_pb.experiment_id,
            resp_pb.trace_model_name,
        )


def append_to_pyarrow_metadata(
    pa_schema: pa.Schema, new_metadata: dict[str, object]
) -> object:
    """Append metadata to a PyArrow schema without overwriting existing metadata.

    Args:
        pa_schema: The PyArrow schema to add metadata to.
        new_metadata: Dictionary of metadata key-value pairs to append.

    Returns:
        A new PyArrow schema with the combined metadata.

    Raises:
        KeyError: If any keys in new_metadata already exist in the schema's metadata.
    """
    # Ensure metadata is handled correctly, even if initially None.
    metadata = pa_schema.metadata
    if metadata is None:
        # Initialize an empty dict if schema metadata was None
        metadata = {}

    conflicting_keys = metadata.keys() & new_metadata.keys()
    if conflicting_keys:
        raise KeyError(
            "Cannot append metadata to pyarrow schema. "
            f"There are conflicting keys: {log_a_list(conflicting_keys, join_word='and')}"
        )

    updated_metadata = metadata.copy()
    updated_metadata.update(new_metadata)
    return pa_schema.with_metadata(updated_metadata)


def _get_pb_flight_doput_request(
    space_id: str,
    request_type: FlightRequestType,
    model_id: str | None = None,
    dataset_id: str | None = None,
    experiment_name: str | None = None,
) -> flight_pb2.DoPutRequest:
    """Construct a Flight DoPut protocol buffer request for the given request type.

    Args:
        space_id: Space ID for the request.
        request_type: FlightRequestType enum value.
        model_id: Optional model/project ID (required for tracing request types).
        dataset_id: Optional dataset ID (required for experiment request types).
        experiment_name: Optional experiment name (required for experiment request types).

    Returns:
        A DoPutRequest protocol buffer configured for the specified request type.

    Raises:
        ValueError: If the request_type is unsupported or required parameters are
            missing for the given request_type.
    """
    common_args = {"space_id": space_id}

    if model_id:
        common_args["external_model_id"] = model_id
    if dataset_id:
        common_args["dataset_id"] = dataset_id
    if experiment_name:
        common_args["experiment_name"] = experiment_name

    if model_id:
        # model-based request types
        match request_type:
            case FlightRequestType.EVALUATION:
                return flight_pb2.DoPutRequest(
                    write_span_evaluation_request=flight_pb2.WriteSpanEvaluationRequest(
                        **common_args
                    )
                )
            case FlightRequestType.ANNOTATION:
                return flight_pb2.DoPutRequest(
                    write_span_annotation_request=flight_pb2.WriteSpanAnnotationRequest(
                        **common_args
                    )
                )
            case FlightRequestType.METADATA:
                return flight_pb2.DoPutRequest(
                    write_span_attributes_metadata_request=flight_pb2.WriteSpanAttributesMetadataRequest(
                        **common_args
                    )
                )
            case _:
                raise ValueError(f"Unsupported request_type: {request_type}")

    if dataset_id and experiment_name:
        # dataset-based request types
        match request_type:
            case FlightRequestType.LOG_EXPERIMENT_DATA:
                return flight_pb2.DoPutRequest(
                    post_experiment_data=flight_pb2.PostExperimentDataRequest(
                        **common_args
                    )
                )
            case _:
                raise ValueError(f"Unsupported request_type: {request_type}")

    raise ValueError(
        f"Unsupported combination: {request_type=} with provided arguments."
    )
