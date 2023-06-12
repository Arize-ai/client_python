import logging

from pyarrow import flight

from .session import Session

logging.basicConfig(level=logging.INFO)


class Query:
    def __init__(
        self,
        query: str,
        client: flight.FlightClient,
        session: Session,
    ) -> None:
        self.query = query
        self.client = client
        self.headers = session.headers

    def execute_query(self) -> flight.FlightStreamReader:
        try:
            options = flight.FlightCallOptions(headers=self.headers)
            flight_info = self.client.get_flight_info(
                flight.FlightDescriptor.for_command(self.query), options
            )
            logging.info("Fetching data...")
            logging.debug("Ticket: %s", flight_info.endpoints[0].ticket)

            # Retrieve the result set as flight stream reader
            reader = self.client.do_get(flight_info.endpoints[0].ticket, options)
            logging.info("Start exporting...")
            return reader

        except Exception:
            logging.exception("There was an error trying to get the data from the endpoint")
            raise
