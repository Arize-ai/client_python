"""Factory for creating and caching the generated OpenAPI client."""

from __future__ import annotations

import base64
import threading
from typing import TYPE_CHECKING, cast
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from typing import Any

    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration


class GeneratedClientFactory:
    """Factory for creating and caching generated OpenAPI clients.

    This factory is owned by ArizeClient and provides thread-safe lazy
    initialization of the OpenAPI client used by various subclients.
    """

    def __init__(
        self,
        sdk_config: SDKConfiguration,
        connection_pool_maxsize: int | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            sdk_config: SDK configuration containing API settings.
            connection_pool_maxsize: urllib3 connection pool size per client.
                When None, the urllib3 default applies.
        """
        self._sdk_config = sdk_config
        self._connection_pool_maxsize = connection_pool_maxsize
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
            cfg.verify_ssl = self._sdk_config.request_verify
            if self._sdk_config.ssl_ca_cert:
                cfg.ssl_ca_cert = self._sdk_config.ssl_ca_cert
            if self._sdk_config.proxy_url:
                cfg.proxy = self._sdk_config.proxy_url
                proxy = urlsplit(self._sdk_config.proxy_url)
                if (
                    proxy.scheme in ("http", "https")
                    and proxy.username is not None
                ):
                    cfg.proxy = proxy._replace(
                        netloc=proxy.netloc.rsplit("@", 1)[1]
                    ).geturl()
                    username = unquote(proxy.username)
                    password = unquote(proxy.password or "")
                    token = base64.b64encode(
                        f"{username}:{password}".encode()
                    ).decode("ascii")
                    cast("Any", cfg).proxy_headers = {
                        "proxy-authorization": f"Basic {token}"
                    }
            if self._connection_pool_maxsize is not None:
                cfg.connection_pool_maxsize = self._connection_pool_maxsize
            self._client = gen.ApiClient(cfg)

            for key, value in self._sdk_config.headers.items():
                if key != "authorization":
                    self._client.set_default_header(key, value)

            return self._client
