import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import opentelemetry.sdk.trace as trace_sdk
import pandas as pd
import pyarrow as pa
from google.protobuf import json_format, message
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Tracer
from pyarrow import flight

from arize.pandas.proto import requests_pb2 as pb2

from ..experiments.evaluators.base import Evaluators
from ..experiments.functions import (
    run_experiment,
    transform_to_experiment_format,
)
from ..experiments.types import (
    EvaluationResultColumnNames,
    ExperimentTask,
    ExperimentTaskResultColumnNames,
)
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

        # set up initial experiment and trace model
        if not dry_run:
            try:
                init_result = self._init_experiment(
                    space_id=space_id,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    experiment_name=experiment_name,
                )
                if init_result is None:
                    raise RuntimeError(
                        f"Failed to initialize experiment {experiment_name}"
                    )
                _, dataset_id, trace_model_name = init_result
            except BaseException as exc:
                raise RuntimeError(
                    f"Failed to initialize experiment {experiment_name}"
                ) from exc
        else:
            trace_model_name = "traces_for_dry_run"

        # download dataset if not provided
        if dataset_df is None:
            try:
                # one of dataset_id or dataset_name is required
                if dataset_id:
                    dataset_df = self.get_dataset(
                        space_id=space_id, dataset_id=dataset_id
                    )
                else:
                    dataset_df = self.get_dataset(
                        space_id=space_id, dataset_name=dataset_name
                    )
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
            model_id=trace_model_name,
            space_id=space_id,
            api_key=self.api_key,
            endpoint=self.otlp_endpoint,
            dry_run=dry_run,
            insecure=(self.host == "localhost"),
            set_global_tracer_provider=set_global_tracer_provider,
        )

        output_df = run_experiment(
            experiment_name=experiment_name,
            dataset=input_df,
            task=task,
            tracer=tracer,
            resource=resource,
            evaluators=evaluators,
            concurrency=concurrency,
            exit_on_error=exit_on_error,
        )
        output_df = _convert_default_columns_to_json_str(output_df)
        output_df = _convert_boolean_columns_to_str(output_df)
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
            raise RuntimeError(
                f"Failed to upload experiment data for {experiment_name}"
            ) from exc
        else:
            return experiment_id, output_df

    def _init_experiment(
        self,
        experiment_name: str,
        space_id: str,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Optional[Tuple[str, str, str]]:
        request = pb2.DoActionRequest(
            create_experiment_db_entry=pb2.CreateExperimentDBEntryRequest(
                space_id=space_id,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                experiment_name=experiment_name,
            )
        )
        action = _action_for_request(
            FlightActionKey.CREATE_EXPERIMENT_DB_ENTRY, request
        )
        flight_client = self.session.connect()
        try:
            response = flight_client.do_action(
                action, self.session.call_options
            )
            res = next(response, None)
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to create experiment {experiment_name}"
            ) from exc
        else:
            if res is None:
                return None
            resp_pb = pb2.CreateExperimentDBEntryResponse()
            resp_pb.ParseFromString(res.body.to_pybytes())
            return (
                resp_pb.experiment_id,
                resp_pb.dataset_id,
                resp_pb.trace_model_name,
            )
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
        request = pb2.DoPutRequest(
            post_experiment_data=pb2.PostExperimentDataRequest(
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
                    res = pb2.PostExperimentDataResponse()
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

    def log_experiment(
        self,
        space_id: str,
        experiment_name: str,
        experiment_df: pd.DataFrame,
        task_columns: ExperimentTaskResultColumnNames,
        evaluator_columns: Optional[
            Dict[str, EvaluationResultColumnNames]
        ] = None,
        dataset_id: str = "",
        dataset_name: str = "",
    ) -> Optional[str]:
        """
        Log an experiment to Arize.

        Args:
            space_id (str): The ID of the space where the experiment will be logged.
            experiment_name (str): The name of the experiment.
            experiment_df (pd.DataFrame): The data to be logged.
            task_columns (ExperimentTaskResultColumnNames): The column names for task results.
            evaluator_columns (Optional[Dict[str, EvaluationResultColumnNames]]):
                The column names for evaluator results.
            dataset_id (str, optional): The ID of the dataset associated with the experiment.
                Required if dataset_name is not provided. Defaults to "".
            dataset_name (str, optional): The name of the dataset associated with the experiment.
                Required if dataset_id is not provided. Defaults to "".

        Examples:
            >>> # Example DataFrame:
            >>> df = pd.DataFrame({
            ...     "example_id": ["1", "2"],
            ...     "result": ["success", "failure"],
            ...     "accuracy": [0.95, 0.85],
            ...     "ground_truth": ["A", "B"],
            ...     "explanation_text": ["Good match", "Poor match"],
            ...     "confidence": [0.9, 0.7],
            ...     "model_version": ["v1", "v2"],
            ...     "custom_metric": [0.8, 0.6],
            ...})
            ...
            >>> # Define column mappings for task
            >>> task_cols = ExperimentTaskResultColumnNames(
            ...    example_id="example_id", result="result"
            ...)
            >>> # Define column mappings for evaluator
            >>> evaluator_cols = EvaluationResultColumnNames(
            ...     score="accuracy",
            ...     label="ground_truth",
            ...     explanation="explanation_text",
            ...     metadata={
            ...         "confidence": None,  # Will use "confidence" column
            ...         "version": "model_version",  # Will use "model_version" column
            ...         "custom_metric": None,  # Will use "custom_metric" column
            ...     },
            ... )
            >>> # Use with ArizeDatasetsClient.log_experiment()
            >>> ArizeDatasetsClient.log_experiment(
            ...     space_id="my_space_id",
            ...     experiment_name="my_experiment",
            ...     experiment_df=df,
            ...     task_columns=task_cols,
            ...     evaluator_columns={"my_evaluator": evaluator_cols},
            ...     dataset_name="my_dataset_name",
            ... )

        Returns:
            Optional[str]: The ID of the logged experiment, or None if the logging failed.
        """
        if dataset_id == "" and dataset_name == "":
            raise ValueError("one of dataset_id or dataset_name is required")
        if experiment_df.empty:
            raise ValueError("experiment_df cannot be empty")

        init_result = self._init_experiment(
            space_id=space_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
        )
        if init_result is None:
            raise RuntimeError(
                f"Failed to initialize experiment {experiment_name}"
            )
        _, dataset_id, _ = init_result

        # transform experiment data to experiment format
        experiment_df = transform_to_experiment_format(
            experiment_df, task_columns, evaluator_columns
        )

        # log experiment data using post_experiment_data
        return self._post_experiment_data(
            experiment_name=experiment_name,
            experiment_df=experiment_df,
            space_id=space_id,
            dataset_id=dataset_id,
        )

    def create_dataset(
        self,
        space_id: str,
        dataset_name: str,
        dataset_type: pb2.DatasetType,
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
        df = _convert_boolean_columns_to_str(df)
        validation_errors = Validator.validate(df)
        if validation_errors:
            raise RuntimeError([e.error_message() for e in validation_errors])

        pa_schema = pa.Schema.from_pandas(df)
        new_schema = pa.schema([field for field in pa_schema])
        tbl = pa.Table.from_pandas(df, schema=new_schema)

        request = pb2.DoPutRequest(
            create_dataset=pb2.CreateDatasetRequest(
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
                    res = pb2.CreateDatasetResponse()
                    res.ParseFromString(response.to_pybytes())
                    if res:
                        return str(res.dataset_id)
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to create dataset: name={dataset_name}, type={dataset_type} for space_id={space_id}"
            ) from exc
        finally:
            flight_client.close()

    def update_dataset(
        self,
        space_id: str,
        data: pd.DataFrame,
        dataset_id: str = "",
        dataset_name: str = "",
    ) -> str:
        """
        Update an existing dataset by creating a new version.

        Args:
            space_id (str): The ID of the space where the dataset is located.
            data (pd.DataFrame): The updated data to be included in the dataset.
            dataset_id (str, optional): The ID of the dataset to update.
                Required if dataset_name is not provided.
            dataset_name (str, optional): The name of the dataset to update.
                Required if dataset_id is not provided.

        Returns:
            str: The ID of the updated dataset.

        Raises:
            ValueError: If neither dataset_id nor dataset_name is provided.
            RuntimeError: If validation of the data fails or the update operation fails.
        """
        if dataset_id == "" and dataset_name == "":
            raise ValueError("one of dataset_id or dataset_name is required")

        df = self._set_default_columns_for_dataset(data)
        df = _convert_default_columns_to_json_str(df)
        df = _convert_boolean_columns_to_str(df)
        validation_errors = Validator.validate(df)
        if validation_errors:
            raise RuntimeError([e.error_message() for e in validation_errors])
        pa_schema = pa.Schema.from_pandas(df)
        new_schema = pa.schema([field for field in pa_schema])
        tbl = pa.Table.from_pandas(df, schema=new_schema)

        request = pb2.DoPutRequest(
            update_dataset=pb2.UpdateDatasetRequest(
                space_id=space_id,
                dataset_id=dataset_id if dataset_id else None,
                dataset_name=dataset_name if dataset_name else None,
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
                    res = pb2.UpdateDatasetResponse()
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
        only_one_of_id_or_name_specified = (dataset_id is None) ^ (
            dataset_name is None
        )
        if not only_one_of_id_or_name_specified:
            raise ValueError(
                f"one and only one of dataset_id={dataset_id} or dataset_name={dataset_name} is required"
            )

        request = pb2.DoGetRequest(
            get_dataset=pb2.GetDatasetRequest(
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
        experiment_name_invalid = (
            experiment_name is None or dataset_name is None
        )
        if experiment_id_invalid and experiment_name_invalid:
            raise ValueError(
                "must provide experiment_id or both experiment_name and dataset_name"
            )

        request = pb2.DoGetRequest(
            get_experiment=pb2.GetExperimentRequest(
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
            raise RuntimeError(
                f"Failed to get experiment {experiment_name}"
            ) from exc
        else:
            return df
        finally:
            flight_client.close()

    def get_dataset_versions(
        self,
        space_id: str,
        dataset_id: str = "",
        dataset_name: str = "",
    ) -> pd.DataFrame:
        """
        Get versions information of a dataset.

        Args:
            space_id (str): The ID of the space where the dataset is located.
            dataset_id (str, optional): The dataset ID to get versions info for.
                Required if dataset_name is not provided.
            dataset_name (str, optional): The name of the dataset to get versions info for.
                Required if dataset_id is not provided.

        Returns:
            pd.DataFrame: A DataFrame containing dataset versions info of the dataset

        Raises:
            ValueError: If neither dataset_id nor dataset_name is provided.
            RuntimeError: If the request to get dataset versions fails.
        """
        if dataset_id == "" and dataset_name == "":
            raise ValueError("one of dataset_id or dataset_name is required")
        request = pb2.DoActionRequest(
            get_dataset_versions=pb2.GetDatasetVersionsRequest(
                space_id=space_id,
                dataset_id=dataset_id if dataset_id else None,
                dataset_name=dataset_name if dataset_name else None,
            )
        )
        action = _action_for_request(
            FlightActionKey.GET_DATASET_VERSION, request
        )
        flight_client = self.session.connect()
        try:
            response = flight_client.do_action(
                action, self.session.call_options
            )
            res = next(response, None)
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to get versions info for dataset id={dataset_id}"
            ) from exc
        else:
            if res is None:
                return None
            resp_pb = pb2.GetDatasetVersionsResponse()
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

    def list_datasets(self, space_id: str) -> pd.DataFrame:
        """
        List all datasets in a space.

        Args:
            space_id (str): The ID of the space to list datasets for.

        Returns:
            pd.DataFrame: A table summary of the datasets in the space.
        """
        request = pb2.DoActionRequest(
            list_datasets=pb2.ListDatasetsRequest(space_id=space_id)
        )
        action = _action_for_request(FlightActionKey.LIST_DATASETS, request)
        flight_client = self.session.connect()
        try:
            response = flight_client.do_action(
                action, self.session.call_options
            )
            res = next(response, None)
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to list datasets in space={space_id}"
            ) from exc
        else:
            if res is None:
                return None
            resp_pb = pb2.ListDatasetsResponse()
            resp_pb.ParseFromString(res.body.to_pybytes())
            out = []
            for dataset in resp_pb.datasets:
                out.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "dataset_name": dataset.dataset_name,
                        "dataset_type": pb2.DatasetType.Name(
                            dataset.dataset_type
                        ),
                        "created_at": dataset.created_at.ToJsonString(),
                        "updated_at": dataset.updated_at.ToJsonString(),
                    }
                )
            return pd.DataFrame(out)
        finally:
            flight_client.close()

    def delete_dataset(
        self,
        space_id: str,
        dataset_id: str = "",
        dataset_name: str = "",
    ) -> bool:
        """
        Delete a dataset.

        Args:
            space_id (str): The ID of the space where the dataset is located.
            dataset_id (str, optional): The ID of the dataset to delete.
                Required if dataset_name is not provided.
            dataset_name (str, optional): The name of the dataset to delete.
                Required if dataset_id is not provided.

        Returns:
            bool: True if the dataset was successfully deleted, False otherwise.

        Raises:
            ValueError: If neither dataset_id nor dataset_name is provided.
            RuntimeError: If the request to delete the dataset fails.
        """
        if dataset_id == "" and dataset_name == "":
            raise ValueError("one of dataset_id or dataset_name is required")
        request = pb2.DoActionRequest(
            delete_dataset=pb2.DeleteDatasetRequest(
                space_id=space_id,
                dataset_id=dataset_id if dataset_id else None,
                dataset_name=dataset_name if dataset_name else None,
            )
        )
        action = _action_for_request(FlightActionKey.DELETE_DATASET, request)
        flight_client = self.session.connect()
        try:
            response = flight_client.do_action(
                action, self.session.call_options
            )
            res = next(response, None)
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to delete dataset {dataset_id}"
            ) from exc
        else:
            if res is None:
                return False
            resp_pb = pb2.DeleteDatasetResponse()
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
                df["id"] = df["id"].apply(
                    lambda x: str(uuid.uuid4()) if pd.isnull(x) else x
                )
        else:
            df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        return df


def _convert_default_columns_to_json_str(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if _should_convert_json(col):
            try:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
            except Exception:
                continue
    return df


def _convert_boolean_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype("string")
    return df


def _convert_json_str_to_dict(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if _should_convert_json(col):
            try:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            except Exception:
                continue
    return df


def _should_convert_json(col_name: str) -> bool:
    """
    Check if a column should be converted to/from a JSON string/PythonDictionary.
    """
    is_eval_metadata = col_name.startswith("eval.") and col_name.endswith(
        ".metadata"
    )
    is_json_str = col_name in OPEN_INFERENCE_JSON_STR_TYPES
    is_task_result = col_name == "result"
    return is_eval_metadata or is_json_str or is_task_result


def _descriptor_for_request(request: message) -> flight.FlightDescriptor:
    data = json_format.MessageToJson(request).encode("utf-8")
    return flight.FlightDescriptor.for_command(data)


def _ticket_for_request(request: message) -> flight.Ticket:
    data = json_format.MessageToJson(request).encode("utf-8")
    return flight.Ticket(data)


def _action_for_request(
    action_key: FlightActionKey, request: message
) -> flight.Action:
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
        else GrpcSpanExporter(
            endpoint=endpoint, insecure=insecure, headers=headers
        )
    )
    tracer_provider.add_span_processor(span_processor)

    if set_global_tracer_provider:
        trace.set_tracer_provider(tracer_provider)

    return tracer_provider.get_tracer(__name__), resource
