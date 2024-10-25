import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import opentelemetry.sdk.trace as trace_sdk
import pandas as pd
import pyarrow as pa
from google.protobuf import json_format, message
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Tracer
from pyarrow import flight

from .. import requests_pb2 as request_pb
from ..experiments.evaluators.base import Evaluators
from ..experiments.functions import run_experiment
from ..experiments.types import ExperimentTask
from ..utils.constants import (
    DEFAULT_ARIZE_FLIGHT_HOST,
    DEFAULT_ARIZE_FLIGHT_PORT,
    DEFAULT_ARIZE_OTLP_ENDPOINT,
    DEFAULT_TRANSPORT_SCHEME,
    OPEN_INFERENCE_JSON_STR_TYPES,
    FlightActionKey,
)
from ..validation.validator import Validator
from .session import Session


@dataclass
class ArizeDatasetsClient:
    """
    ArizeDatasetsClient is a client for interacting with the Arize Datasets API.

    Args:
        developer_key (str, required): Arize provided developer key associated with your user profile,
            located on the space settings page.
        api_key (str, required): Arize provided API key associated with your user profile,
            located on the space settings page.
        host (str, optional): URI endpoint host to send your export request to Arize AI. Defaults to
            "{DEFAULT_ARIZE_FLIGHT_HOST}".
        port (int, optional): URI endpoint port to send your export request to Arize AI. Defaults to
            {DEFAULT_ARIZE_FLIGHT_PORT}.
        scheme (str, optional): Transport scheme to use for the connection. Defaults to
            "{DEFAULT_TRANSPORT_SCHEME}".
        otlp_endpoint (str, optional): OTLP endpoint to send experiment traces to Arize. Defaults to
            "{DEFAULT_ARIZE_OTLP_ENDPOINT}".
    """

    developer_key: str
    api_key: str
    host: str = DEFAULT_ARIZE_FLIGHT_HOST
    port: int = DEFAULT_ARIZE_FLIGHT_PORT
    scheme: str = DEFAULT_TRANSPORT_SCHEME
    otlp_endpoint: str = DEFAULT_ARIZE_OTLP_ENDPOINT

    def __post_init__(self) -> None:
        """
        Initializes the Arize Dataset Client.
        """
        self.__session = Session(
            self.developer_key,
            self.host,
            self.port,
            self.scheme,
        )

    @property
    def session(self) -> Session:
        return self.__session

    def run_experiment(
        self,
        space_id: str,
        experiment_name: str,
        task: ExperimentTask,
        dataset_df: Optional[pd.DataFrame] = None,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        evaluators: Optional[Evaluators] = None,
        dry_run: bool = False,
        concurrency: int = 3,
        set_global_tracer_provider: bool = False,
        exit_on_error: bool = False,
    ) -> Union[Tuple[str, pd.DataFrame], None]:
        """
        Run an experiment on a dataset and upload the results.

        This function initializes an experiment, retrieves or uses a provided dataset,
        runs the experiment with specified tasks and evaluators, and uploads the results.

        Args:
            space_id (str): The ID of the space where the experiment will be run.
            experiment_name (str): The name of the experiment.
            task (ExperimentTask): The task to be performed in the experiment.
            dataset_df (Optional[pd.DataFrame], optional): The dataset as a pandas DataFrame.
                If not provided, the dataset will be downloaded using dataset_id or dataset_name.
                Defaults to None.
            dataset_id (Optional[str], optional): The ID of the dataset to use.
                Required if dataset_df and dataset_name are not provided. Defaults to None.
            dataset_name (Optional[str], optional): The name of the dataset to use.
                Used if dataset_df and dataset_id are not provided. Defaults to None.
            evaluators (Optional[Evaluators], optional): The evaluators to use in the experiment.
                Defaults to None.
            dry_run (bool): If True, the experiment result will not be uploaded to Arize.
                Defaults to False.
            concurrency (int): The number of concurrent tasks to run. Defaults to 3.
            set_global_tracer_provider (bool): If True, sets the global tracer provider for the experiment.
                Defaults to False.
            exit_on_error (bool): If True, the experiment will stop running on first occurrence of an error.

        Returns:
            Tuple[str, pd.DataFrame]:
                A tuple of experiment ID and experiment result DataFrame.
                If dry_run is True, the experiment ID will be an empty string.

        Raises:
            ValueError: If dataset_id and dataset_name are both not provided, or if the dataset is empty.
            RuntimeError: If experiment initialization, dataset download, or result upload fails.
        """

        if dataset_id is None and dataset_name is None:
            raise ValueError("must provide dataset_id or dataset_name")
        dataset_identifier = dataset_id or dataset_name
        # this is the trace model in the platform storing the traces for the experiment
        trace_model_id = f"{experiment_name}_{_get_hex_hash(dataset_identifier)}"

        # set up initial experiment and trace model
        if not dry_run:
            try:
                init_result = self._init_experiment(
                    space_id=space_id,
                    dataset_id=dataset_id,
                    trace_model_name=trace_model_id,
                    dataset_name=dataset_name,
                    experiment_name=experiment_name,
                )
                if init_result is None:
                    raise RuntimeError(f"Failed to initialize experiment {experiment_name}")
                _, dataset_id = init_result
            except BaseException as exc:
                raise RuntimeError(f"Failed to initialize experiment {experiment_name}") from exc

        # download dataset if not provided
        if dataset_df is None:
            try:
                # one of dataset_id or dataset_name is required
                if dataset_id:
                    dataset_df = self.get_dataset(space_id=space_id, dataset_id=dataset_id)
                else:
                    dataset_df = self.get_dataset(space_id=space_id, dataset_name=dataset_name)
            except BaseException as exc:
                raise RuntimeError(
                    f"Failed to download dataset {dataset_id} to run experiment"
                ) from exc
        if dataset_df is None or dataset_df.empty:
            raise ValueError(f"Dataset {dataset_id} is empty")

        input_df = dataset_df.copy()
        if dry_run:
            # only dry_run experiment on a subset (first 10 rows) of the dataset
            input_df = input_df.head(10)

        # trace model and resource for the experiment
        tracer, resource = _get_tracer_resource(
            model_id=trace_model_id,
            space_id=space_id,
            api_key=self.api_key,
            endpoint=self.otlp_endpoint,
            dry_run=dry_run,
            insecure=(self.host == "localhost"),
            set_global_tracer_provider=set_global_tracer_provider,
        )

        output_df = run_experiment(
            dataset=input_df,
            task=task,
            evaluators=evaluators,
            experiment_name=experiment_name,
            tracer=tracer,
            resource=resource,
            concurrency=concurrency,
            exit_on_error=exit_on_error,
        )
        output_df = _convert_default_columns_to_json_str(output_df)
        if dry_run:
            return "", output_df
        try:
            experiment_id = self._post_experiment_data(
                experiment_name=experiment_name,
                experiment_df=output_df,
                space_id=space_id,
                dataset_id=dataset_id,
            )
        except BaseException as exc:
            raise RuntimeError(f"Failed to upload experiment data for {experiment_name}") from exc
        else:
            return experiment_id, output_df

    def _init_experiment(
        self,
        experiment_name: str,
        space_id: str,
        trace_model_name: str,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        request = request_pb.DoActionRequest(
            create_experiment_db_entry=request_pb.CreateExperimentDBEntryRequest(
                space_id=space_id,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                experiment_name=experiment_name,
                trace_model_name=trace_model_name,
            )
        )
        action = _action_for_request(FlightActionKey.CREATE_EXPERIMENT_DB_ENTRY, request)
        flight_client = self.session.connect()
        try:
            response = flight_client.do_action(action, self.session.call_options)
            res = next(response, None)
        except BaseException as exc:
            raise RuntimeError(f"Failed to create experiment {experiment_name}") from exc
        else:
            if res is None:
                return None
            resp_pb = request_pb.CreateExperimentDBEntryResponse()
            resp_pb.ParseFromString(res.body.to_pybytes())
            return resp_pb.experiment_id, resp_pb.dataset_id
        finally:
            flight_client.close()

    def _post_experiment_data(
        self,
        experiment_name: str,
        experiment_df: pd.DataFrame,
        space_id: str,
        dataset_id: str,
    ) -> Optional[str]:
        if experiment_df.empty:
            raise ValueError("experiment result DataFrame cannot be empty")
        tbl = pa.Table.from_pandas(experiment_df)
        request = request_pb.DoPutRequest(
            post_experiment_data=request_pb.PostExperimentDataRequest(
                space_id=space_id,
                dataset_id=dataset_id,
                experiment_name=experiment_name,
            )
        )
        descriptor = _descriptor_for_request(request)
        flight_client = self.session.connect()
        try:
            writer, metadata_reader = flight_client.do_put(
                descriptor, tbl.schema, self.session.call_options
            )
            with writer:
                writer.write_table(tbl, max_chunksize=10_000)
                writer.done_writing()
                response = metadata_reader.read()
                if response is not None:
                    res = request_pb.PostExperimentDataResponse()
                    res.ParseFromString(response.to_pybytes())
                    if res:
                        return str(res.experiment_id)
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to upload experiment run result to Arize for dataset_id={dataset_id},"
                f" experiment_name={experiment_name}"
            ) from exc
        finally:
            flight_client.close()

    def create_dataset(
        self,
        space_id: str,
        dataset_name: str,
        dataset_type: request_pb.DatasetType,
        data: pd.DataFrame,
        convert_dict_to_json: bool = True,
    ) -> Optional[str]:
        """
        Create a new dataset.

        Args:
            space_id (str): The ID of the space where the dataset will be created.
            dataset_name (str): The name of the dataset.
            dataset_type (DatasetType): The type of the dataset.
            data (pd.DataFrame): The data to be included in the dataset.
            convert_dict_to_json (bool, optional): Convert dictionary columns to JSON strings
                for default JSON str columns per Open Inference. Defaults to True.
        Returns:
            str: The ID of the created dataset, or None if the creation failed.
        """

        df = self._set_default_columns_for_dataset(data)
        if convert_dict_to_json:
            df = _convert_default_columns_to_json_str(df)
        validation_errors = Validator.validate(df)
        if validation_errors:
            raise RuntimeError([e.error_message() for e in validation_errors])

        pa_schema = pa.Schema.from_pandas(df)
        new_schema = pa.schema([field for field in pa_schema])
        tbl = pa.Table.from_pandas(df, schema=new_schema)

        request = request_pb.DoPutRequest(
            create_dataset=request_pb.CreateDatasetRequest(
                space_id=space_id,
                dataset_name=dataset_name,
                dataset_type=dataset_type,
            )
        )
        descriptor = _descriptor_for_request(request)
        flight_client = self.session.connect()
        try:
            writer, metadata_reader = flight_client.do_put(
                descriptor, tbl.schema, self.session.call_options
            )
            with writer:
                writer.write_table(tbl, max_chunksize=10_000)
                writer.done_writing()
                response = metadata_reader.read()
                if response is not None:
                    res = request_pb.CreateDatasetResponse()
                    res.ParseFromString(response.to_pybytes())
                    if res:
                        return str(res.dataset_id)
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to create dataset: name={dataset_name}, type={dataset_type} for space_id={space_id}"
            ) from exc
        finally:
            flight_client.close()

    def update_dataset(self, space_id: str, dataset_id: str, data: pd.DataFrame) -> Optional[str]:
        """
        Update an existing dataset by creating a new version.

        Args:
            space_id (str): The ID of the space where the dataset is located.
            dataset_id (str): The ID of the dataset to update.
            data (pd.DataFrame): The updated data to be included in the dataset.

        Returns:
            str: The ID of the updated dataset, or None if the update failed.
        """
        df = self._set_default_columns_for_dataset(data)
        validation_errors = Validator.validate(df)
        if validation_errors:
            raise RuntimeError([e.error_message() for e in validation_errors])
        pa_schema = pa.Schema.from_pandas(df)
        new_schema = pa.schema([field for field in pa_schema])
        tbl = pa.Table.from_pandas(df, schema=new_schema)

        request = request_pb.DoPutRequest(
            update_dataset=request_pb.UpdateDatasetRequest(
                space_id=space_id,
                dataset_id=dataset_id,
            )
        )
        descriptor = _descriptor_for_request(request)
        flight_client = self.session.connect()
        try:
            writer, metadata_reader = flight_client.do_put(
                descriptor, tbl.schema, self.session.call_options
            )
            with writer:
                writer.write_table(tbl, max_chunksize=10_000)
                writer.done_writing()
                response = metadata_reader.read()
                if response is not None:
                    res = request_pb.UpdateDatasetResponse()
                    res.ParseFromString(response.to_pybytes())
                    return str(res.dataset_id)
        except Exception as e:
            raise RuntimeError("Failed to update dataset") from e
        finally:
            flight_client.close()

    def get_dataset(
        self,
        space_id: str,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_version: Optional[str] = None,
        convert_json_str_to_dict: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get the data of a dataset.

        Args:
            space_id (str): The ID of the space where the dataset is located.
            dataset_id (str): Dataset id. Required if dataset_name is not provided.
            dataset_name (str): Dataset name. Required if dataset_id is not provided.
            dataset_version (str, optional): version name of the dataset
                Defaults to "" and gets the latest version by based on creation time
            convert_json_str_to_dict (bool, optional): Convert JSON strings to Python dictionaries.
                For default JSON str columns per Open Inference. Defaults to True.
        Returns:
            pd.DataFrame: The data of the dataset.
        """
        only_one_of_id_or_name_specified = (dataset_id is None) ^ (dataset_name is None)
        if not only_one_of_id_or_name_specified:
            raise ValueError(
                f"one and only one of dataset_id={dataset_id} or dataset_name={dataset_name} is required"
            )

        request = request_pb.DoGetRequest(
            get_dataset=request_pb.GetDatasetRequest(
                space_id=space_id,
                dataset_version=dataset_version,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
            )
        )

        ticket = _ticket_for_request(request)
        flight_client = self.session.connect()
        try:
            reader = flight_client.do_get(ticket, self.session.call_options)
            df = reader.read_all().to_pandas()
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to get dataset name={dataset_name}, id={dataset_id} for space_id={space_id}"
            ) from exc
        else:
            if convert_json_str_to_dict is True:
                df = _convert_json_str_to_dict(df)
            return df
        finally:
            flight_client.close()

    def get_experiment(
        self,
        space_id: str,
        experiment_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve experiment data from Arize.

        Args:
            space_id (str): The ID of the space where the experiment is located.
            experiment_name (Optional[str]): The name of the experiment.
                Required if experiment_id is not provided.
            dataset_name (Optional[str]): The name of the dataset associated with the experiment.
                Required if experiment_id is not provided.
            experiment_id (Optional[str]): The ID of the experiment.
                Required if experiment_name and dataset_name are not provided.

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the experiment data,
                or None if the retrieval fails.

        Raises:
            ValueError: If neither experiment_id nor both experiment_name and dataset_name are provided.
            RuntimeError: If the experiment retrieval fails.

        Note:
            You must provide either the experiment_id or both the experiment_name and dataset_name.
        """
        experiment_id_invalid = experiment_id is None
        experiment_name_invalid = experiment_name is None or dataset_name is None
        if experiment_id_invalid and experiment_name_invalid:
            raise ValueError("must provide experiment_id or both experiment_name and dataset_name")

        request = request_pb.DoGetRequest(
            get_experiment=request_pb.GetExperimentRequest(
                space_id=space_id,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                experiment_id=experiment_id,
            )
        )
        ticket = _ticket_for_request(request)
        flight_client = self.session.connect()
        try:
            reader = flight_client.do_get(ticket, self.session.call_options)
            df = reader.read_all().to_pandas()
        except BaseException as exc:
            raise RuntimeError(f"Failed to get experiment {experiment_name}") from exc
        else:
            return df
        finally:
            flight_client.close()

    def get_dataset_versions(self, space_id: str, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Get the versions of a dataset.

        Args:
            space_id (str): The ID of the space where the dataset is located.
            dataset_id (str): The ID of the dataset to get versions info for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the versions of the dataset.
        """
        request = request_pb.DoActionRequest(
            get_dataset_versions=request_pb.GetDatasetVersionsRequest(
                space_id=space_id, dataset_id=dataset_id
            )
        )
        action = _action_for_request(FlightActionKey.GET_DATASET_VERSION, request)
        flight_client = self.session.connect()
        try:
            response = flight_client.do_action(action, self.session.call_options)
            res = next(response, None)
        except BaseException as exc:
            raise RuntimeError(f"Failed to get versions info for dataset id={dataset_id}") from exc
        else:
            if res is None:
                return None
            resp_pb = request_pb.GetDatasetVersionsResponse()
            resp_pb.ParseFromString(res.body.to_pybytes())
            out = []
            for v in resp_pb.versions:
                out.append(
                    {
                        "dataset_version": v.version_name,
                        "created_at": v.created_at.ToJsonString(),
                        "updated_at": v.updated_at.ToJsonString(),
                    }
                )
            return pd.DataFrame(out)
        finally:
            flight_client.close()

    def list_datasets(self, space_id: str) -> Optional[pd.DataFrame]:
        """
        List all datasets in a space.

        Args:
            space_id (str): The ID of the space to list datasets for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the datasets in the space.
        """
        request = request_pb.DoActionRequest(
            list_datasets=request_pb.ListDatasetsRequest(space_id=space_id)
        )
        action = _action_for_request(FlightActionKey.LIST_DATASETS, request)
        flight_client = self.session.connect()
        try:
            response = flight_client.do_action(action, self.session.call_options)
            res = next(response, None)
        except BaseException as exc:
            raise RuntimeError(f"Failed to list datasets in space={space_id}") from exc
        else:
            if res is None:
                return None
            resp_pb = request_pb.ListDatasetsResponse()
            resp_pb.ParseFromString(res.body.to_pybytes())
            out = []
            for dataset in resp_pb.datasets:
                out.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "dataset_name": dataset.dataset_name,
                        "dataset_type": request_pb.DatasetType.Name(dataset.dataset_type),
                        "created_at": dataset.created_at.ToJsonString(),
                        "updated_at": dataset.updated_at.ToJsonString(),
                    }
                )
            return pd.DataFrame(out)
        finally:
            flight_client.close()

    def delete_dataset(self, space_id: str, dataset_id: str) -> Optional[bool]:
        """
        Delete a dataset.

        Args:
            space_id (str): The ID of the space where the dataset is located.
            dataset_id (str): The ID of the dataset to delete.

        Returns:
            bool: True if the dataset was successfully deleted, False otherwise.
        """
        request = request_pb.DoActionRequest(
            delete_dataset=request_pb.DeleteDatasetRequest(space_id=space_id, dataset_id=dataset_id)
        )
        action = _action_for_request(FlightActionKey.DELETE_DATASET, request)
        flight_client = self.session.connect()
        try:
            response = flight_client.do_action(action, self.session.call_options)
            res = next(response, None)
        except BaseException as exc:
            raise RuntimeError(f"Failed to delete dataset {dataset_id}") from exc
        else:
            if res is None:
                return False
            resp_pb = request_pb.DeleteDatasetResponse()
            resp_pb.ParseFromString(res.body.to_pybytes())
            return resp_pb.success
        finally:
            flight_client.close()

    @staticmethod
    def _set_default_columns_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
        current_time = int(time.time() * 1000)
        if "created_at" in df.columns:
            if df["created_at"].isnull().values.any():
                df["created_at"].fillna(current_time, inplace=True)
        else:
            df["created_at"] = current_time

        if "updated_at" in df.columns:
            if df["updated_at"].isnull().values.any():
                df["updated_at"].fillna(current_time, inplace=True)
        else:
            df["updated_at"] = current_time

        if "id" in df.columns:
            if df["id"].isnull().values.any():
                df["id"] = df["id"].apply(lambda x: str(uuid.uuid4()) if pd.isnull(x) else x)
        else:
            df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        return df


def _convert_default_columns_to_json_str(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if _should_convert(col):
            try:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            except Exception:
                continue
    return df


def _convert_json_str_to_dict(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if _should_convert(col):
            try:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            except Exception:
                continue
    return df


def _should_convert(col_name: str) -> bool:
    """
    Check if a column should be converted to/from a JSON string/PythonDictionary.
    """
    is_eval_metadata = col_name.startswith("eval.") and col_name.endswith(".metadata")
    is_json_str = col_name in OPEN_INFERENCE_JSON_STR_TYPES
    is_task_result = col_name == "result"
    return is_eval_metadata or is_json_str or is_task_result


def _descriptor_for_request(request: message) -> flight.FlightDescriptor:
    data = json_format.MessageToJson(request).encode("utf-8")
    return flight.FlightDescriptor.for_command(data)


def _ticket_for_request(request: message) -> flight.Ticket:
    data = json_format.MessageToJson(request).encode("utf-8")
    return flight.Ticket(data)


def _action_for_request(action_key: FlightActionKey, request: message) -> flight.Action:
    req_bytes = json_format.MessageToJson(request).encode("utf-8")
    return flight.Action(action_key.value, req_bytes)


def _get_tracer_resource(
    model_id: str,
    space_id: str,
    api_key: str,
    endpoint: str,
    dry_run: bool = False,
    insecure: bool = False,
    set_global_tracer_provider: bool = False,
) -> Tuple[Tracer, Resource]:
    resource = Resource({"model_id": model_id})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    headers = f"space_id={space_id},api_key={api_key}"
    span_processor = SimpleSpanProcessor(
        ConsoleSpanExporter()
        if dry_run
        else GrpcSpanExporter(endpoint=endpoint, insecure=insecure, headers=headers)
    )
    tracer_provider.add_span_processor(span_processor)

    if set_global_tracer_provider:
        trace.set_tracer_provider(tracer_provider)

    return tracer_provider.get_tracer(__name__), resource


def _get_hex_hash(input_string):
    input_bytes = input_string.encode("utf-8")
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_bytes)
    return sha256_hash.hexdigest()
