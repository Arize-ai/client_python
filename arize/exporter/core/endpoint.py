from pyarrow import flight

from .query import Query


class Endpoint:
    def __init__(self, session) -> None:
        self.session = session

    def connect(self) -> flight.FlightClient:
        return self.session.connect()

    def execute_query(
        self, flight_client: flight.FlightClient, query: str
    ) -> flight.FlightStreamReader:
        arize_flight_query = Query(query, flight_client, self.session)
        return arize_flight_query.execute_query()
