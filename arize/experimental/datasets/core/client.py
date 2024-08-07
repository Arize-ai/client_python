import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pyarrow as pa
from google.protobuf import json_format, message
from pyarrow import flight

from .. import requests_pb2 as request_pb
from ..experiments.evaluators.base import Evaluators
from ..experiments.functions import run_experiment
from ..experiments.types import ExperimentTask
from ..utils.constants import (
    DEFAULT_ARIZE_FLIGHT_HOST,
    DEFAULT_ARIZE_FLIGHT_PORT,
    DEFAULT_TRANSPORT_SCHEME,
    FLIGHT_ACTION_KEY,
    OPEN_INFERENCE_JSON_STR_TYPES,
)
from ..validation.validator import Validator
from .session import Session


@dataclass
class ArizeDatasetsClient:
    """
    ArizeDatasetsClient is a client for interacting with the Arize Datasets API.

    Args:
        developer_key (str, optional): Arize provided developer key associated with your user profile,
            located on the API Explorer page. API key is required to initiate a new client, it can
            be passed in explicitly, or set up as an environment variable or in profile file.
        host (str, optional): URI endpoint host to send your export request to Arize AI. Defaults to
            "{DEFAULT_ARIZE_FLIGHT_HOST}".
        port (int, optional): URI endpoint port to send your export request to Arize AI. Defaults to
            {DEFAULT_ARIZE_FLIGHT_PORT}.
        scheme (str, optional): Transport scheme to use for the connection. Defaults to
            "{DEFAULT_TRANSPORT_SCHEME}".

    Attributes:
        session (Session): The session object used for making API requests.

    """

    developer_key: Optional[str]
    host: str = DEFAULT_ARIZE_FLIGHT_HOST
    port: int = DEFAULT_ARIZE_FLIGHT_PORT
    scheme: str = DEFAULT_TRANSPORT_SCHEME

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
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        evaluators: Optional[Evaluators] = None,
    ) -> Optional[str]:

        if not (dataset_id or dataset_name):
            raise ValueError("Either dataset_id or dataset_name must be provided")

        dataset = self.get_dataset(
            space_id=space_id, dataset_id=dataset_id, dataset_name=dataset_name
        )
        if dataset is None or dataset.empty:
            raise RuntimeError("Dataset is empty or does not exist")

        exp_df = run_experiment(
            dataset=dataset,
            task=task,
            evaluators=evaluators,
            experiment_name=experiment_name,
        )
        exp_df = self._convert_default_columns_to_json_str(exp_df)
        pa_schema = pa.Schema.from_pandas(exp_df)
        new_schema = pa.schema([field for field in pa_schema])
        tbl = pa.Table.from_pandas(exp_df, schema=new_schema)
        request = request_pb.DoPutRequest(
            create_experiment=request_pb.CreateExperimentRequest(
                space_id=space_id,
                dataset_id=dataset_id,
                experiment_name=experiment_name,
            )
        )

        descriptor = self._descriptor_for_request(request)
        try:
            flight_client = self.session.connect()
            writer, metadata_reader = flight_client.do_put(
                descriptor, tbl.schema, self.session.call_options
            )
            with writer:
                writer.write_table(tbl, max_chunksize=10_000)
                writer.done_writing()
                response = metadata_reader.read()
                if response is not None:
                    res = request_pb.CreateExperimentResponse()
                    res.ParseFromString(response.to_pybytes())
                    if res:
                        return str(res.experiment_id)
        except BaseException as exc:
            raise RuntimeError(
                "Failed to upload experiment run result to Arize for "
                f"dataset_id={dataset_id}, dataset_name={dataset_name}, "
                f"experiment_name={experiment_name}"
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

        Returns:
            str: The ID of the created dataset, or None if the creation failed.
        """
        ## Validate and convert to arrow table
        df = self._set_default_columns_for_dataset(data)
        if convert_dict_to_json:
            df = self._convert_default_columns_to_json_str(df)
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
        descriptor = self._descriptor_for_request(request)
        try:
            flight_client = self.session.connect()
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
        descriptor = self._descriptor_for_request(request)
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
            dataset_name (str): Dataset name. Reqired if dataset_id is not provided.
            dataset_version (str, optional): version name of the dataset
            Defaults to "" and gets the latest version by based on creation time

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

        ticket = self._ticket_for_request(request)
        try:
            flight_client = self.session.connect()
            reader = flight_client.do_get(ticket, self.session.call_options)
            df = reader.read_all().to_pandas()
        except BaseException as exc:
            raise RuntimeError(
                f"Failed to get dataset name={dataset_name}, id={dataset_id} for space_id={space_id}"
            ) from exc
        else:
            if convert_json_str_to_dict is True:
                df = self._convert_json_str_to_dict(df)
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
        action = self._action_for_request(FLIGHT_ACTION_KEY.GET_DATASET_VERSION, request)
        try:
            flight_client = self.session.connect()
            response = flight_client.do_action(action, self.session.call_options)
            res = next(response, None)
            # Close the client here to drain the response stream
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
        action = self._action_for_request(FLIGHT_ACTION_KEY.LIST_DATASETS, request)
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
        action = self._action_for_request(FLIGHT_ACTION_KEY.DELETE_DATASET, request)
        try:
            flight_client = self.session.connect()
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

    def _descriptor_for_request(self, request: message) -> flight.FlightDescriptor:
        data = json_format.MessageToJson(request).encode("utf-8")
        return flight.FlightDescriptor.for_command(data)

    def _ticket_for_request(self, request: message) -> flight.Ticket:
        data = json_format.MessageToJson(request).encode("utf-8")
        return flight.Ticket(data)

    def _action_for_request(self, action_key: FLIGHT_ACTION_KEY, request: message) -> flight.Action:
        req_bytes = json_format.MessageToJson(request).encode("utf-8")
        return flight.Action(action_key.value, req_bytes)

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

    @staticmethod
    def _convert_default_columns_to_json_str(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if _should_convert(col):
                try:
                    df[col] = df[col].apply(lambda x: json.dumps(x))
                    print(f"converted {col} to JSON string for data import")
                except Exception:
                    continue
        return df

    @staticmethod
    def _convert_json_str_to_dict(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if _should_convert(col):
                try:
                    df[col] = df[col].apply(lambda x: json.loads(x))
                    print(f"converted {col} to dict for data export")
                except Exception:
                    continue
        return df


def _should_convert(col_name: str) -> bool:
    """
    Check if a column should be converted to/from a JSON string/PythonDictionary.
    """
    is_eval_metadata = col_name.startswith("eval.") and col_name.endswith(".metadata")
    is_json_str = col_name in OPEN_INFERENCE_JSON_STR_TYPES
    return is_eval_metadata or is_json_str
