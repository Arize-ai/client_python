from dataclasses import dataclass
from typing import Tuple

from arize.utils.logging import logger
from google.protobuf import json_format
from pyarrow import flight

from .. import publicexporter_pb2 as exp_pb2


@dataclass(frozen=True)
class Query:
    query_descriptor: exp_pb2.RecordQueryDescriptor

    def execute(
        self, client: flight.FlightClient, call_options: flight.FlightCallOptions
    ) -> Tuple[flight.FlightStreamReader, int]:
        try:
            flight_info = client.get_flight_info(
                flight.FlightDescriptor.for_command(
                    json_format.MessageToJson(self.query_descriptor)  # type: ignore
                ),
                call_options,
            )
            logger.info("Fetching data...")

            if flight_info.total_records == 0:
                logger.info("Query returns no data")
                return None, 0
            logger.debug("Ticket: %s", flight_info.endpoints[0].ticket)

            # Retrieve the result set as flight stream reader
            reader = client.do_get(flight_info.endpoints[0].ticket, call_options)
            logger.info("Starting exporting...")
            return reader, flight_info.total_records

        except Exception:
            logger.error("There was an error trying to get the data from the endpoint")
            raise
