"""Region definitions and configuration for Arize deployment zones."""

from dataclasses import dataclass
from enum import Enum

from arize.constants.config import DEFAULT_FLIGHT_PORT


class Region(Enum):
    """Enum representing available Arize deployment regions."""

    CA_CENTRAL_1A = "ca-central-1a"
    EU_WEST_1A = "eu-west-1a"
    US_CENTRAL_1A = "us-central-1a"
    US_EAST_1B = "us-east-1b"
    UNSET = ""


@dataclass(frozen=True)
class RegionEndpoints:
    """Container for region-specific API endpoint hostnames and ports."""

    api_host: str
    otlp_host: str
    flight_host: str
    flight_port: int


def _get_region_endpoints(region: Region) -> RegionEndpoints:
    return RegionEndpoints(
        api_host=f"api.{region.value}.arize.com",
        otlp_host=f"otlp.{region.value}.arize.com",
        flight_host=f"flight.{region.value}.arize.com",
        flight_port=DEFAULT_FLIGHT_PORT,
    )


REGION_ENDPOINTS: dict[Region, RegionEndpoints] = {
    r: _get_region_endpoints(r) for r in list(Region) if r != Region.UNSET
}
