# type: ignore[pb2]
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
from arize.utils.types import Environments, SimilaritySearchParams
from google.protobuf.timestamp_pb2 import Timestamp
from pyarrow import flight
from tqdm import tqdm

from .. import publicexporter_pb2 as exp_pb2
from ..utils.constants import (
    ARIZE_PROFILE,
    DEFAULT_ARIZE_FLIGHT_HOST,
    DEFAULT_ARIZE_FLIGHT_PORT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROFILE_NAME,
    DEFAULT_TRANSPORT_SCHEME,
)
from ..utils.tracing import OtelTracingDataTransformer
from ..utils.validation import Validator
from .query import Query
from .session import Session


@dataclass
class ArizeExportClient:
    """
    Arize's Export Client.

    Arguments:
    ----------
        api_key (str, optional): Arize provided personal API key associated with your user profile,
            located on the API Explorer page. API key is required to initiate a new client, it can
            be passed in explicitly, or set up as an environment variable or in profile file.
        arize_profile (str, optional): profile name for ArizeExportClient credentials and endpoint.
        arize_config_path (str, optional): path to the config file that stores ArizeExportClient
            credentials and endpoint. Defaults to '~/.arize'.
        host (str, optional): URI endpoint host to send your export request to Arize AI.
        port (int, optional): URI endpoint port to send your export request to Arize AI.
    """

    api_key: Optional[str] = None
    arize_profile: str = ARIZE_PROFILE or DEFAULT_PROFILE_NAME
    arize_config_path: Optional[str] = DEFAULT_CONFIG_PATH
    host: str = DEFAULT_ARIZE_FLIGHT_HOST
    port: int = DEFAULT_ARIZE_FLIGHT_PORT
    scheme: str = DEFAULT_TRANSPORT_SCHEME

    def __post_init__(
        self,
    ) -> None:
        """
        Initializes the Arize Export Client.
        """
        self.__session = Session(
            self.api_key,
            self.arize_profile,
            self.arize_config_path,
            self.host,
            self.port,
            self.scheme,
        )

    @property
    def session(self) -> Session:
        return self.__session

    def export_model_to_df(
        self,
        space_id: str,
        model_id: str,
        environment: Environments,
        start_time: datetime,
        end_time: datetime,
        include_actuals: bool = False,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
        where: Optional[str] = None,
        similarity_search_params: Optional[SimilaritySearchParams] = None,
    ) -> pd.DataFrame:
        """
        Exports data of a specific model in the Arize platform to a pandas dataframe for a defined
        time interval and model environment, optionally by model version and/or batch id.

        Arguments:
        ----------
            space_id (str): The id for the space where to export models from, can be retrieved from
                the url of the Space Overview page in the Arize UI.
            model_id (str): The name of the model to export, can be found in the Model Overview
                tab in the Arize UI.
            environment (Environments): The environment for the model to export (can be Production,
                Training, or Validation).
            start_time (datetime): The start time for the data to export for the model, start time
                is inclusive. Time interval has hourly granularity.
            end_time (datetime): The end time for the data to export for the model, end time is not
                inclusive. Time interval has hourly granularity.
            include_actuals (bool, optional): An input to indicate whether to include actuals
                / ground truth in the data to export. `include_actuals` only applies to the Production
                environment and defaults to 'False'.
            model_version (str, optional): An input to indicate the version of the model to
                export. Model versions for all model environments can be found in the Datasets tab on
                the model page in the Arize UI. Defaults to None.
            batch_id (str, optional): An input to indicate the batch name of the model to export.
                Batches only apply to the Validation environment, and can be found in the Datasets tab on
                the model page in the Arize UI. Defaults to None.
            where (str, optional): An input to provide sql like where statement to filter a
                subset of records from the model, e.g. "age > 50 And state='CA'". Defaults to None.
            similarity_search_params (SimilaritySearchParams, optional): Parameters for embedding similarity
                search using cosine similarity. It includes 'references', a list of reference embeddings for
                comparison; 'search_column_name', specifying the column that contains the embeddings to search
                within; and 'threshold', which sets the cosine similarity threshold required for embeddings to
                be considered similar.

        Returns:
        --------
            A pandas dataframe
        """
        stream_reader, num_recs = self._get_model_stream_reader(
            space_id=space_id,
            model_id=model_id,
            environment=environment,
            start_time=start_time,
            end_time=end_time,
            include_actuals=include_actuals,
            model_version=model_version,
            batch_id=batch_id,
            where=where,
            similarity_search_params=similarity_search_params,
        )
        if stream_reader is None:
            return pd.DataFrame()
        progress_bar = self.get_progress_bar(num_recs)
        list_of_df = []
        while True:
            try:
                flight_batch = stream_reader.read_chunk()
                batch_df = flight_batch.data.to_pandas()
                list_of_df.append(batch_df)
                progress_bar.update(batch_df.shape[0])
            except StopIteration:
                break
        progress_bar.close()
        df = pd.concat(list_of_df)
        null_columns = df.columns[df.isnull().all()]
        df.drop(null_columns, axis=1, inplace=True)

        if environment == Environments.TRACING:
            try:
                oic_spec = importlib.util.find_spec("openinference.semconv")
            except Exception as e:
                raise ImportError(
                    "An error occurred while trying to find the "
                    "'openinference-semantic-conventions' package, which is a required dependency. "
                    + str(e)
                ) from e
            if oic_spec is None:
                raise ImportError(
                    "Required 'openinference-semantic-conventions' dependency is missing"
                )

            # by default, transform the exported tracing data so that it's
            # easier to work with in Python
            transformer = OtelTracingDataTransformer()
            df = transformer.transform(df)

        df.sort_values(by=["time"], inplace=True)
        return df.reset_index(drop=True)

    def export_model_to_parquet(
        self,
        path: str,
        space_id: str,
        model_id: str,
        environment: Environments,
        start_time: datetime,
        end_time: datetime,
        include_actuals: bool = False,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
        where: Optional[str] = None,
        similarity_search_params: Optional[SimilaritySearchParams] = None,
    ) -> None:
        """
        Exports data of a specific model in the Arize platform to a parquet file for a defined time
        interval and model environment, optionally by model version and/or batch id.

        Arguments:
        ----------
            path (str): path to the file to store exported data. File must be in parquet format and
                has a '.parquet' extension.
            space_id (str): The id for the space where to export models from, can be retrieved from
                the url of the Space Overview page in the Arize UI.
            model_id (str): The name of the model to export, can be found in the Model Overview
                tab in the Arize UI.
            environment (Environments): The environment for the model to export (can be Production,
                Training, or Validation).
            start_time (datetime): The start time for the data to export for the model, start time
                is inclusive. Time interval has hourly granularity.
            end_time (datetime): The end time for the data to export for the model, end time is not
                inclusive. Time interval has hourly granularity.
            include_actuals (bool, optional): An input to indicate whether to include actuals
                / ground truth in the data to export. `include_actuals` only applies to the Production
                environment and defaults to 'False'.
            model_version (str, optional): An input to indicate the version of the model to
                export. Model versions for all model environments can be found in the Datasets tab on
                the model page in the Arize UI. Defaults to None.
            batch_id (str, optional): An input to indicate the batch name of the model to export.
                Batches only apply to the Validation environment, and can be found in the Datasets tab on
                the model page in the Arize UI. Defaults to None.
            where (str, optional): An input to provide sql like where statement to filter a
                subset of records from the model, e.g. "age > 50 And state='CA'". Defaults to None.
            similarity_search_params (SimilaritySearchParams, optional): Parameters for embedding similarity
                search using cosine similarity. It includes 'references', a list of reference embeddings for
                comparison; 'search_column_name', specifying the column that contains the embeddings to search
                within; and 'threshold', which sets the cosine similarity threshold required for embeddings to
                be considered similar.

        Returns:
        --------
            None
        """
        Validator.validate_input_type(path, "path", str)
        stream_reader, num_recs = self._get_model_stream_reader(
            space_id=space_id,
            model_id=model_id,
            environment=environment,
            start_time=start_time,
            end_time=end_time,
            include_actuals=include_actuals,
            model_version=model_version,
            batch_id=batch_id,
            where=where,
            similarity_search_params=similarity_search_params,
        )
        if stream_reader is None:
            return None
        progress_bar = self.get_progress_bar(num_recs)
        with pq.ParquetWriter(path, schema=stream_reader.schema) as writer:
            while True:
                try:
                    flight_batch = stream_reader.read_chunk()
                    record_batch = flight_batch.data
                    writer.write_batch(record_batch)
                    progress_bar.update(record_batch.num_rows)
                except StopIteration:
                    break
        progress_bar.close()

    def _get_model_stream_reader(
        self,
        space_id: str,
        model_id: str,
        environment: Environments,
        start_time: datetime,
        end_time: datetime,
        include_actuals: bool = False,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
        where: Optional[str] = None,
        similarity_search_params: Optional[SimilaritySearchParams] = None,
    ) -> Tuple[flight.FlightStreamReader, int]:
        Validator.validate_input_type(space_id, "space_id", str)
        Validator.validate_input_type(model_id, "model_id", str)
        Validator.validate_input_type(environment, "environment", Environments)
        Validator.validate_input_type(include_actuals, "include_actuals", bool)
        Validator.validate_input_type(start_time, "start_time", datetime)
        Validator.validate_input_type(end_time, "end_time", datetime)
        Validator.validate_input_type(model_version, "model_version", str)
        Validator.validate_input_type(batch_id, "batch_id", str)
        Validator.validate_start_end_time(start_time, end_time)
        Validator.validate_input_type(where, "where", str)

        # Create query descriptor
        query_descriptor = exp_pb2.RecordQueryDescriptor(
            space_id=space_id,
            model_id=model_id,
            environment=environment.name,
            model_version=model_version,
            batch_id=batch_id,
            include_actuals=include_actuals,
            start_time=Timestamp(seconds=int(start_time.timestamp())),
            end_time=Timestamp(seconds=int(end_time.timestamp())),
            filter_expression=where,
            similarity_search_params=(
                self._to_similarity_proto_params(similarity_search_params)
                if similarity_search_params
                else None
            ),
        )

        flight_client = self.session.connect()
        query = Query(query_descriptor)
        reader = query.execute(flight_client, self.session.call_options)
        return reader

    def get_progress_bar(self, num_recs):
        return tqdm(
            total=num_recs,
            desc=f"  exporting {num_recs} rows",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
            ncols=80,
            colour="#008000",
            unit=" row",
        )

    def _to_similarity_proto_params(
        self, similarity_params: SimilaritySearchParams
    ) -> exp_pb2.SimilaritySearchParams:
        proto_params = exp_pb2.SimilaritySearchParams()
        proto_params.search_column_name = similarity_params.search_column_name
        proto_params.threshold = similarity_params.threshold
        for ref in similarity_params.references:
            new_ref = proto_params.references.add()
            new_ref.prediction_id = ref.prediction_id
            new_ref.reference_column_name = ref.reference_column_name
            if ref.prediction_timestamp:
                new_ref.prediction_timestamp.FromDatetime(ref.prediction_timestamp)

        return proto_params
