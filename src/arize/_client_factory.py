"""Factory for creating and caching the generated OpenAPI client."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration


class GeneratedClientFactory:
    """Factory for creating and caching generated OpenAPI clients.

    This factory is owned by ArizeClient and provides thread-safe lazy
    initialization of the OpenAPI client used by various subclients.
    """

    def __init__(self, sdk_config: SDKConfiguration) -> None:
        """Initialize the factory.

        Args:
            sdk_config: SDK configuration containing API settings.
        """
        self._sdk_config = sdk_config
        self._client: ApiClient | None = None
        self._lock = threading.Lock()

    def get_client(self) -> ApiClient:
        """Get or create the generated OpenAPI client instance.

        Returns:
            The shared generated API client instance.
        """
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            # Import lazily to avoid extra dependencies at config time
            from arize._generated import api_client as gen

            cfg = gen.Configuration(host=self._sdk_config.api_url)
            if self._sdk_config.api_key:
                cfg.access_token = self._sdk_config.api_key
            self._client = gen.ApiClient(cfg)
            return self._client
