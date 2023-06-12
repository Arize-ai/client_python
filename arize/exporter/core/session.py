import configparser
import logging
import os
import uuid
from dataclasses import InitVar, dataclass, field

from pyarrow import flight

from ..utils.constants import (
    ARIZE_API_KEY,
    ARIZE_PROFILE,
    DEFAULT_ARIZE_API_KEY_CONFIG_KEY,
    DEFAULT_ARIZE_FLIGHT_HOST,
    DEFAULT_ARIZE_FLIGHT_PORT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PACKAGE_NAME,
    DEFAULT_PROFILE_NAME,
    PROFILE_FILE_NAME,
)
from ..utils.errors import InvalidConfigFileError, InvalidSessionError

logging.basicConfig(level=logging.INFO)


@dataclass
class Session:
    api_key: InitVar[str] = None
    arize_profile: InitVar[str] = None
    arize_config_path: InitVar[str] = DEFAULT_CONFIG_PATH
    host: InitVar[str] = DEFAULT_ARIZE_FLIGHT_HOST
    port: InitVar[int] = DEFAULT_ARIZE_FLIGHT_PORT
    arize_api_key: str = field(init=False)
    session_name: str = field(init=False)

    def __post_init__(
        self, api_key: str, arize_profile: str, arize_config_path: str, host: str, port: int
    ):
        self.headers = []
        self.call_options = None
        self.session_name = f"python-sdk-{DEFAULT_PACKAGE_NAME}-{uuid.uuid4()}"
        logging.info(f"Creating named session as '{self.session_name}'.")
        api_key = api_key or ARIZE_API_KEY
        if api_key:
            self.arize_api_key = api_key
        else:
            self._read_config(arize_profile, arize_config_path)

        if host:
            self.host = host
        else:
            self.host = DEFAULT_ARIZE_FLIGHT_HOST

        if port:
            self.port = port
        else:
            self.port = DEFAULT_ARIZE_FLIGHT_PORT

        logging.debug(
            f"Created session with Arize API Token '{self.arize_api_key}' at '{self.host}':'{self.port}'"
        )

    def _read_config(self, arize_profile: str, arize_config_path: str):
        arize_profile = arize_profile or ARIZE_PROFILE or DEFAULT_PROFILE_NAME
        arize_config_path = arize_config_path or DEFAULT_CONFIG_PATH

        config_parser = Session._get_config_parser()
        file_path = os.path.join(arize_config_path, PROFILE_FILE_NAME)
        logging.debug(
            f"No provided connection details. Looking up session values from '{arize_profile}' in "
            f"'{file_path}'."
        )
        try:
            config_parser.read(file_path)
            self.arize_api_key = config_parser.get(arize_profile, DEFAULT_ARIZE_API_KEY_CONFIG_KEY)
        except configparser.NoSectionError:
            raise InvalidSessionError(
                "Credentials not provided or invalid. Please pass in the correct api_key when"
                " initiating a new client or set up credentials in profile or as env variables"
            )
        except Exception as err:
            raise InvalidConfigFileError from err

    @staticmethod
    def _get_config_parser() -> configparser.ConfigParser:
        return configparser.ConfigParser()

    def connect(self) -> flight.FlightClient:
        """Connects to Arize Flight server public endpoint with the
        provided api key."""
        try:
            # Default to use an unencrypted TCP connection.
            scheme = "grpc+tls"
            if self.host.lower() == "localhost":
                disable_cert = True
            else:
                disable_cert = False
            if self.arize_api_key:
                client = flight.FlightClient(
                    location=f"{scheme}://{self.host}:{self.port}",
                    disable_server_verification=disable_cert,
                )
                self.headers.append((b"auth-token-bin", f"{self.arize_api_key}".encode("utf-8")))
                self.call_options = flight.FlightCallOptions(timeout=600, headers=self.headers)
                return client
            raise ConnectionError("arize api key must be supplied.")
        except Exception:
            logging.exception("There was an error trying to connect to the Arize Flight Endpoint")
            raise
