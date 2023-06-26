import configparser
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from arize.__init__ import __version__
from arize.utils.logging import logger
from arize.utils.utils import get_python_version
from pyarrow import flight

from ..utils.constants import (
    ARIZE_API_KEY,
    DEFAULT_ARIZE_API_KEY_CONFIG_KEY,
    DEFAULT_PACKAGE_NAME,
    PROFILE_FILE_NAME,
)
from ..utils.errors import InvalidConfigFileError, InvalidSessionError


@dataclass
class Session:
    api_key: Optional[str]
    arize_profile: str
    arize_config_path: str
    host: str
    port: int
    scheme: str
    session_name: str = field(init=False)
    call_options: flight.FlightCallOptions = field(init=False)

    def __post_init__(self):
        self.session_name = f"python-sdk-{DEFAULT_PACKAGE_NAME}-{uuid.uuid4()}"
        logger.info(f"Creating named session as '{self.session_name}'.")
        # If api_key is not passed, try reading from environment variable.
        # If api_key is also not set as environment variable, read from config file
        self.api_key = self.api_key or ARIZE_API_KEY or self._read_config()
        if self.api_key is None:
            logger.error(InvalidSessionError.error_message())
            raise InvalidSessionError

        logger.debug(
            f"Created session with Arize API Token '{self.api_key}' at '{self.host}':'{self.port}'"
        )
        self._set_headers()

    def _read_config(self) -> Optional[str]:
        config_parser = Session._get_config_parser()
        file_path = os.path.join(self.arize_config_path, PROFILE_FILE_NAME)
        logger.debug(
            f"No provided connection details. Looking up session values from '{self.arize_profile}' in "
            f"'{file_path}'."
        )
        try:
            config_parser.read(file_path)
            return config_parser.get(self.arize_profile, DEFAULT_ARIZE_API_KEY_CONFIG_KEY)
        except configparser.NoSectionError as err:
            # Missing api key error is raised in the __post_init__ method
            logger.warning(f"Can't extract API key from config file. {err.message}")
            return None
        except Exception as err:
            logger.error(InvalidConfigFileError.error_message())
            raise InvalidConfigFileError from err

    @staticmethod
    def _get_config_parser() -> configparser.ConfigParser:
        return configparser.ConfigParser()

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
            (b"origin", b"arize-python-exporter"),
            (b"auth-token-bin", f"{self.api_key}".encode("utf-8")),
            (b"sdk-language", b"python"),
            (b"language-version", get_python_version().encode("utf-8")),
            (b"sdk-version", __version__.encode("utf-8")),
        ]
