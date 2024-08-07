import uuid
from dataclasses import dataclass, field

from arize.__init__ import __version__
from arize.utils.logging import logger
from arize.utils.utils import get_python_version
from pyarrow import flight

from ..utils.constants import DEFAULT_PACKAGE_NAME
from ..validation.errors import InvalidSessionError


@dataclass
class Session:
    developer_key: str
    host: str
    port: int
    scheme: str
    session_name: str = field(init=False)
    call_options: flight.FlightCallOptions = field(init=False)

    def __post_init__(self):
        self.session_name = f"python-sdk-{DEFAULT_PACKAGE_NAME}-{uuid.uuid4()}"
        logger.debug(f"Creating named session as '{self.session_name}'.")
        if self.developer_key is None:
            logger.error(InvalidSessionError.error_message())
            raise InvalidSessionError

        logger.debug(
            f"Created session with Arize Developer Key '{self.developer_key}' at '{self.host}':'{self.port}'"
        )
        self._set_headers()

    def connect(self) -> flight.FlightClient:
        """Connects to Arize Flight server public endpoint with the
        provided api key."""
        try:
            disable_cert = True if self.host.lower() == "localhost" else False
            client = flight.FlightClient(
                location=f"{self.scheme}://{self.host}:{self.port}",
                disable_server_verification=disable_cert,
            )
            self.call_options = flight.FlightCallOptions(headers=self._headers)
            return client
        except Exception:
            logger.error("There was an error trying to connect to the Arize Flight Endpoint")
            raise

    def _set_headers(self) -> None:
        self._headers = [
            (b"origin", b"arize-python-datasets-client"),
            (b"auth-token-bin", f"{self.developer_key}".encode("utf-8")),
            (b"sdk-language", b"python"),
            (b"language-version", get_python_version().encode("utf-8")),
            (b"sdk-version", __version__.encode("utf-8")),
        ]
