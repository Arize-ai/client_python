import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from google.protobuf import json_format
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.wrappers_pb2 import Int64Value
from pyarrow import flight
from tqdm import tqdm

from arize._exporter.validation import (
    validate_input_type,
    validate_start_end_time,
)
from arize._generated.protocol.flight import flight_pb2
from arize.logging import CtxAdapter
from arize.ml.types import Environments, SimilaritySearchParams
from arize.utils.dataframe import reset_dataframe_index

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArizeExportClient:
    flight_client: flight.FlightClient

    def export_to_df(
        self,
        space_id: str,
        model_id: str,
        environment: Environments,
        start_time: datetime,
        end_time: datetime,
        where: str = "",
        columns: list | None = None,
        similarity_search_params: SimilaritySearchParams | None = None,
        model_version: str = "",
        batch_id: str = "",
        include_actuals: bool = False,
        stream_chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """Exports data of a specific model in the Arize platform to a pandas dataframe.

        The export covers a defined time interval and model environment, and can
        optionally be filtered by model version and/or batch id.

        Args:
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
            columns (list, optional): Specifies the columns to include from the model data during export. If
                not provided, all columns will be exported.
            stream_chunk_size (int, optional): Optional parameter to explicitly specify the pagination chunk
                size during the export stream. Normally this value is determined dynamically in the backend
                but in extreme cases where individual records are large enough to cause issues that result
                in export stream error, setting this to a very low value (e.g. 10) could help.
                The maximum value accepted by the server is 5000. Defaults to None.

        Returns:
            A pandas dataframe

        """
        stream_reader, num_recs = self._get_stream_reader(
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
            columns=columns,
            stream_chunk_size=stream_chunk_size,
        )
        if stream_reader is None:
            return pd.DataFrame()
        progress_bar = self._get_progress_bar(num_recs)
        list_of_df = []
        try:
            while True:
                flight_batch = stream_reader.read_chunk()
                batch_df = flight_batch.data.to_pandas()
                list_of_df.append(batch_df)
                progress_bar.update(batch_df.shape[0])
        except StopIteration:
            pass
        progress_bar.close()
        df = pd.concat(list_of_df)
        null_columns = df.columns[df.isnull().all()]
        df.drop(null_columns, axis=1, inplace=True)

        if environment == Environments.TRACING:
            from arize._exporter.parsers.tracing_data_parser import (
                OtelTracingDataTransformer,
            )

            # by default, transform the exported tracing data so that it's
            # easier to work with in Python
            df = OtelTracingDataTransformer().transform(df)

        df.sort_values(by=["time"], inplace=True)
        reset_dataframe_index(df)
        return df

    def export_to_parquet(
        self,
        path: str,
        space_id: str,
        model_id: str,
        environment: Environments,
        start_time: datetime,
        end_time: datetime,
        where: str = "",
        columns: list | None = None,
        similarity_search_params: SimilaritySearchParams | None = None,
        model_version: str = "",
        batch_id: str = "",
        include_actuals: bool = False,
        stream_chunk_size: int | None = None,
    ) -> None:
        """Exports data of a specific model in the Arize platform to a parquet file.

        The export covers a defined time interval and model environment, and can
        optionally be filtered by model version and/or batch id.

        Args:
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
            columns (list, optional): Specifies the columns to include from the model data during export. If
                not provided, all columns will be exported.
            stream_chunk_size (int, optional): Optional parameter to explicitly specify the pagination chunk
                size during the export stream. Normally this value is determined dynamically in the backend
                but in extreme cases where individual records are large enough to cause issues that result
                in export stream error, setting this to a very low value (e.g. 10) could help.
                The maximum value accepted by the server is 5000. Defaults to None.


        Returns:
            None

        """
        validate_input_type(path, "path", str)
        stream_reader, num_recs = self._get_stream_reader(
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
            columns=columns,
            stream_chunk_size=stream_chunk_size,
        )
        if stream_reader is None:
            return
        progress_bar = self._get_progress_bar(num_recs)
        with pq.ParquetWriter(path, schema=stream_reader.schema) as writer:
            try:
                while True:
                    flight_batch = stream_reader.read_chunk()
                    record_batch = flight_batch.data
                    writer.write_batch(record_batch)
                    progress_bar.update(record_batch.num_rows)
            except StopIteration:
                pass
        progress_bar.close()

    def _get_stream_reader(
        self,
        space_id: str,
        model_id: str,
        environment: Environments,
        start_time: datetime,
        end_time: datetime,
        include_actuals: bool = False,
        model_version: str = "",
        batch_id: str = "",
        where: str = "",
        similarity_search_params: SimilaritySearchParams | None = None,
        columns: list | None = None,
        stream_chunk_size: int | None = None,
    ) -> tuple[flight.FlightStreamReader | None, int]:
        # Validate inputs first before creating logging context
        validate_input_type(space_id, "space_id", str)
        validate_input_type(model_id, "model_id", str)
        validate_input_type(environment, "environment", Environments)
        validate_input_type(include_actuals, "include_actuals", bool)
        validate_input_type(start_time, "start_time", datetime)
        validate_input_type(end_time, "end_time", datetime)
        validate_input_type(model_version, "model_version", str)
        validate_input_type(batch_id, "batch_id", str)
        validate_input_type(where, "where", str)
        validate_input_type(columns, "columns", list, allow_none=True)
        validate_input_type(
            stream_chunk_size, "stream_chunk_size", int, allow_none=True
        )
        validate_start_end_time(start_time, end_time)

        # Bind common context for this operation
        log = CtxAdapter(
            logger,
            {
                "component": "exporter",
                "operation": "export_to_df",
                "space_id": space_id,
                "model_id": model_id,
                "environment": environment.name,
                "model_version": model_version,
                "batch_id": batch_id,
                "include_actuals": include_actuals,
                "where": where,
                "columns": columns,
                "similarity_search_params": similarity_search_params,
                "stream_chunk_size": stream_chunk_size,
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        log.debug("Getting stream reader...")

        # Create query descriptor
        query_descriptor = flight_pb2.RecordQueryDescriptor(
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
                _get_pb_similarity_search_params(similarity_search_params)
                if similarity_search_params
                else None
            ),
            projected_columns=columns if columns else [],
            stream_chunk_size=(
                Int64Value(value=stream_chunk_size)
                if stream_chunk_size is not None
                else None
            ),
        )

        try:
            flight_info = self.flight_client.get_flight_info(
                flight.FlightDescriptor.for_command(
                    json_format.MessageToJson(query_descriptor)
                ),
            )
            logger.info("Fetching data...")

            if flight_info.total_records == 0:
                logger.warning("Query returns no data")
                return None, 0
            logger.debug("Ticket: %s", flight_info.endpoints[0].ticket)
        except Exception as e:
            msg = f"Error getting flight info or do_get: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        # Retrieve the result set as flight stream reader
        reader = self.flight_client.do_get(flight_info.endpoints[0].ticket)
        return reader, flight_info.total_records

    @staticmethod
    def _get_progress_bar(num_recs: int) -> tqdm:
        """Create a progress bar for export operations.

        Args:
            num_recs: Total number of records to export.

        Returns:
            A tqdm progress bar configured for data export display.
        """
        return tqdm(
            total=num_recs,
            desc=f"  exporting {num_recs} rows",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
            ncols=80,
            colour="#008000",
            unit=" row",
        )


def _get_pb_similarity_search_params(
    similarity_params: SimilaritySearchParams,
) -> flight_pb2.SimilaritySearchParams:
    """Convert SimilaritySearchParams to protocol buffer format.

    Args:
        similarity_params: Similarity search parameters containing search column name,
            threshold, and reference examples.

    Returns:
        A protocol buffer SimilaritySearchParams object for Flight requests.
    """
    proto_params = flight_pb2.SimilaritySearchParams()
    proto_params.search_column_name = similarity_params.search_column_name
    proto_params.threshold = similarity_params.threshold
    for ref in similarity_params.references:
        new_ref = proto_params.references.add()
        new_ref.prediction_id = ref.prediction_id
        new_ref.reference_column_name = ref.reference_column_name
        if ref.prediction_timestamp:
            new_ref.prediction_timestamp.FromDatetime(ref.prediction_timestamp)

    return proto_params
