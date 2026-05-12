"""Main Arize SDK client for interacting with Arize AI platform services."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from arize._lazy import LazySubclientsMixin
from arize.config import SDKConfiguration

if TYPE_CHECKING:
    from arize.ai_integrations.client import AiIntegrationsClient
    from arize.annotation_configs.client import AnnotationConfigsClient
    from arize.annotation_queues.client import AnnotationQueuesClient
    from arize.api_keys.client import ApiKeysClient
    from arize.datasets.client import DatasetsClient
    from arize.evaluators.client import EvaluatorsClient
    from arize.experiments.client import ExperimentsClient
    from arize.ml.client import MLModelsClient
    from arize.organizations.client import OrganizationsClient
    from arize.projects.client import ProjectsClient
    from arize.prompts.client import PromptsClient
    from arize.regions import Region
    from arize.resource_restrictions.client import ResourceRestrictionsClient
    from arize.role_bindings.client import RoleBindingsClient
    from arize.roles.client import RolesClient
    from arize.spaces.client import SpacesClient
    from arize.spans.client import SpansClient
    from arize.tasks.client import TasksClient
    from arize.users.client import UsersClient

logger = logging.getLogger(__name__)

# TODO: Go over headers on each logging call
# TODO: InvalidAdditionalHeadersError is unused. Have we handled extra headers?

# TODO: Need to implement 'Update existing examples in a dataset'

# TODO: Protobuf versioning is too old
# TODO: Go through main APIs and add CtxAdapter where missing
# TODO: Search and handle other TODOs


class ArizeClient(LazySubclientsMixin):
    """Root client for the Arize SDK.

    The ArizeClient provides access to all Arize platform services including datasets,
    experiments, ML models, projects, and spans. It uses :class:`arize.config.SDKConfiguration`
    internally to manage configuration settings.

    All parameters are optional (except api_key which must be provided via argument
    or environment variable). For each parameter, values are resolved in this order:

    1. Explicit value passed to constructor (highest priority)
    2. Environment variable value (see SDKConfiguration for variable names)
    3. Built-in default value (lowest priority)

    Examples:
        Initialize with API key only (other settings use defaults):

            >>> client = ArizeClient(api_key="your-api-key")

        Initialize with custom endpoints:

            >>> client = ArizeClient(
            ...     api_key="your-api-key",
            ...     api_host="custom.api.com",
            ...     flight_port=8443,
            ... )

        Initialize with region (overrides host/port settings):

            >>> from arize.regions import Region
            >>> client = ArizeClient(
            ...     api_key="your-api-key", region=Region.EU_WEST
            ... )
    """

    _SUBCLIENTS: ClassVar[dict[str, tuple[str, str]]] = {
        "ai_integrations": (
            "arize.ai_integrations.client",
            "AiIntegrationsClient",
        ),
        "datasets": (
            "arize.datasets.client",
            "DatasetsClient",
        ),
        "experiments": (
            "arize.experiments.client",
            "ExperimentsClient",
        ),
        "projects": (
            "arize.projects.client",
            "ProjectsClient",
        ),
        "ml": (
            "arize.ml.client",
            "MLModelsClient",
        ),
        "organizations": (
            "arize.organizations.client",
            "OrganizationsClient",
        ),
        "spans": (
            "arize.spans.client",
            "SpansClient",
        ),
        "annotation_configs": (
            "arize.annotation_configs.client",
            "AnnotationConfigsClient",
        ),
        "annotation_queues": (
            "arize.annotation_queues.client",
            "AnnotationQueuesClient",
        ),
        "spaces": (
            "arize.spaces.client",
            "SpacesClient",
        ),
        "prompts": (
            "arize.prompts.client",
            "PromptsClient",
        ),
        "api_keys": (
            "arize.api_keys.client",
            "ApiKeysClient",
        ),
        "evaluators": (
            "arize.evaluators.client",
            "EvaluatorsClient",
        ),
        "resource_restrictions": (
            "arize.resource_restrictions.client",
            "ResourceRestrictionsClient",
        ),
        "role_bindings": (
            "arize.role_bindings.client",
            "RoleBindingsClient",
        ),
        "roles": (
            "arize.roles.client",
            "RolesClient",
        ),
        "tasks": (
            "arize.tasks.client",
            "TasksClient",
        ),
        "users": (
            "arize.users.client",
            "UsersClient",
        ),
    }
    # DISABLED: Optional dependency gating system
    # This dict would map subclients to their optional dependencies and extra names.
    # When enabled, it prevents loading subclients if dependencies aren't installed,
    # showing: "Install via: pip install arize[extra-name]"
    #
    # To re-enable, populate with entries like:
    # "subclient_name": ("extra-name", ("package1", "package2", "package3")),
    # "another_subclient": (None, ()),  # No requirements
    #
    _EXTRAS: ClassVar[dict[str, tuple[str | None, tuple[str, ...]]]] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: Region | None = None,
        api_host: str | None = None,
        api_scheme: str | None = None,
        otlp_host: str | None = None,
        otlp_scheme: str | None = None,
        flight_host: str | None = None,
        flight_port: int | None = None,
        flight_scheme: str | None = None,
        pyarrow_max_chunksize: int | None = None,
        request_verify: bool | None = None,
        stream_max_workers: int | None = None,
        stream_max_queue_bound: int | None = None,
        max_http_payload_size_mb: float | None = None,
        arize_directory: str | None = None,
        enable_caching: bool | None = None,
        single_host: str | None = None,
        single_port: int | None = None,
        base_domain: str | None = None,
        max_past_years: int | None = None,
    ) -> None:
        """Initialize the Arize client with configuration parameters.

        All parameters are optional and follow the precedence order documented in the
        class docstring: explicit argument > environment variable > default value.

        Args:
            api_key: Arize API key for authentication. Required - must be provided here
                or via ARIZE_API_KEY environment variable.
                Raises MissingAPIKeyError if not set.
            region: Arize region (e.g., Region.US_CENTRAL, Region.EU_WEST). When specified,
                overrides individual host/port settings.
                ENV: ARIZE_REGION.
                Default: Region.UNSET.
            api_host: Custom API endpoint host.
                ENV: ARIZE_API_HOST.
                Default: "api.arize.com".
            api_scheme: API endpoint scheme (http/https).
                ENV: ARIZE_API_SCHEME.
                Default: "https".
            otlp_host: OTLP endpoint host.
                ENV: ARIZE_OTLP_HOST.
                Default: "otlp.arize.com".
            otlp_scheme: OTLP endpoint scheme (http/https).
                ENV: ARIZE_OTLP_SCHEME.
                Default: "https".
            flight_host: Apache Arrow Flight endpoint host.
                ENV: ARIZE_FLIGHT_HOST.
                Default: "flight.arize.com".
            flight_port: Apache Arrow Flight endpoint port (1-65535).
                ENV: ARIZE_FLIGHT_PORT.
                Default: 443.
            flight_scheme: Apache Arrow Flight endpoint scheme.
                ENV: ARIZE_FLIGHT_SCHEME.
                Default: "grpc+tls".
            pyarrow_max_chunksize: Maximum PyArrow chunk size (1 to MAX_CHUNKSIZE).
                ENV: ARIZE_MAX_CHUNKSIZE.
                Default: 10_000.
            request_verify: Whether to verify SSL certificates.
                ENV: ARIZE_REQUEST_VERIFY.
                Default: True.
            stream_max_workers: Maximum worker threads for streaming (minimum: 1).
                ENV: ARIZE_STREAM_MAX_WORKERS.
                Default: 8.
            stream_max_queue_bound: Maximum queue size for streaming (minimum: 1).
                ENV: ARIZE_STREAM_MAX_QUEUE_BOUND.
                Default: 5000.
            max_http_payload_size_mb: Maximum HTTP payload size in MB (minimum: 1).
                ENV: ARIZE_MAX_HTTP_PAYLOAD_SIZE_MB.
                Default: 100.
            arize_directory: Directory for SDK files (cache, logs, etc.).
                ENV: ARIZE_DIRECTORY.
                Default: "~/.arize".
            enable_caching: Whether to enable local caching.
                ENV: ARIZE_ENABLE_CACHING.
                Default: True.
            single_host: Single host for all endpoints. When specified, overrides
                individual hosts.
                ENV: ARIZE_SINGLE_HOST.
                Default: None.
            single_port: Single port for all endpoints. When specified, overrides
                individual ports.
                ENV: ARIZE_SINGLE_PORT.
                Default: 0 (not set).
            base_domain: Base domain for generating endpoint hosts as api.<base_domain>,
                otlp.<base_domain>, flight.<base_domain>. Intended for Private Connect setups.
                When specified, overrides individual hosts.
                ENV: ARIZE_BASE_DOMAIN.
                Default: None.
            max_past_years: Maximum number of years in the past allowed for prediction timestamps.
                For on-prem deployments with custom retention policies, this can be increased.
                ENV: ARIZE_MAX_PAST_YEARS.
                Default: 5.

        Raises:
            MissingAPIKeyError: If api_key is not provided via argument or environment variable.
            MultipleEndpointOverridesError: If multiple endpoint override options (region,
                single_host/single_port, base_domain) are provided.

        Notes:
            Values provided to this class override environment variables, which in turn
            override default values. See :class:`arize.config.SDKConfiguration`
            for detailed parameter documentation.
        """
        cfg_kwargs: dict = {}
        if api_key is not None:
            cfg_kwargs["api_key"] = api_key
        if region is not None:
            cfg_kwargs["region"] = region
        if api_host is not None:
            cfg_kwargs["api_host"] = api_host
        if api_scheme is not None:
            cfg_kwargs["api_scheme"] = api_scheme
        if otlp_host is not None:
            cfg_kwargs["otlp_host"] = otlp_host
        if otlp_scheme is not None:
            cfg_kwargs["otlp_scheme"] = otlp_scheme
        if flight_host is not None:
            cfg_kwargs["flight_host"] = flight_host
        if flight_port is not None:
            cfg_kwargs["flight_port"] = flight_port
        if flight_scheme is not None:
            cfg_kwargs["flight_scheme"] = flight_scheme
        if pyarrow_max_chunksize is not None:
            cfg_kwargs["pyarrow_max_chunksize"] = pyarrow_max_chunksize
        if request_verify is not None:
            cfg_kwargs["request_verify"] = request_verify
        if stream_max_workers is not None:
            cfg_kwargs["stream_max_workers"] = stream_max_workers
        if stream_max_queue_bound is not None:
            cfg_kwargs["stream_max_queue_bound"] = stream_max_queue_bound
        if max_http_payload_size_mb is not None:
            cfg_kwargs["max_http_payload_size_mb"] = max_http_payload_size_mb
        if arize_directory is not None:
            cfg_kwargs["arize_directory"] = arize_directory
        if enable_caching is not None:
            cfg_kwargs["enable_caching"] = enable_caching
        if single_host is not None:
            cfg_kwargs["single_host"] = single_host
        if single_port is not None:
            cfg_kwargs["single_port"] = single_port
        if base_domain is not None:
            cfg_kwargs["base_domain"] = base_domain
        if max_past_years is not None:
            cfg_kwargs["max_past_years"] = max_past_years

        # Only the explicitly provided fields are passed; the rest use
        # SDKConfiguration's default factories / defaults.
        super().__init__(SDKConfiguration(**cfg_kwargs))

    # typed properties for IDE completion
    @property
    def ai_integrations(self) -> AiIntegrationsClient:
        """Access the AI integrations client for managing LLM provider integrations (lazy-loaded)."""
        return cast("AiIntegrationsClient", self.__getattr__("ai_integrations"))

    @property
    def datasets(self) -> DatasetsClient:
        """Access the datasets client for dataset operations (lazy-loaded)."""
        return cast("DatasetsClient", self.__getattr__("datasets"))

    @property
    def experiments(self) -> ExperimentsClient:
        """Access the experiments client for experiment operations (lazy-loaded)."""
        return cast("ExperimentsClient", self.__getattr__("experiments"))

    @property
    def ml(self) -> MLModelsClient:
        """Access the ML models client for ML model operations (lazy-loaded)."""
        return cast("MLModelsClient", self.__getattr__("ml"))

    @property
    def organizations(self) -> OrganizationsClient:
        """Access the organizations client for organization operations (lazy-loaded)."""
        return cast("OrganizationsClient", self.__getattr__("organizations"))

    @property
    def projects(self) -> ProjectsClient:
        """Access the projects client for project operations (lazy-loaded)."""
        return cast("ProjectsClient", self.__getattr__("projects"))

    @property
    def spans(self) -> SpansClient:
        """Access the spans client for tracing and span operations (lazy-loaded)."""
        return cast("SpansClient", self.__getattr__("spans"))

    @property
    def annotation_configs(self) -> AnnotationConfigsClient:
        """Access the annotation configs client for config operations."""
        return cast(
            "AnnotationConfigsClient",
            self.__getattr__("annotation_configs"),
        )

    @property
    def annotation_queues(self) -> AnnotationQueuesClient:
        """Access the annotation queues client for queue and record operations."""
        return cast(
            "AnnotationQueuesClient",
            self.__getattr__("annotation_queues"),
        )

    @property
    def spaces(self) -> SpacesClient:
        """Access the spaces client for space operations (lazy-loaded)."""
        return cast("SpacesClient", self.__getattr__("spaces"))

    @property
    def prompts(self) -> PromptsClient:
        """Access the prompts client for prompt management operations (lazy-loaded)."""
        return cast("PromptsClient", self.__getattr__("prompts"))

    @property
    def api_keys(self) -> ApiKeysClient:
        """Access the API keys client for API key operations (lazy-loaded)."""
        return cast("ApiKeysClient", self.__getattr__("api_keys"))

    @property
    def evaluators(self) -> EvaluatorsClient:
        """Access the evaluators client for evaluator operations (lazy-loaded)."""
        return cast("EvaluatorsClient", self.__getattr__("evaluators"))

    @property
    def resource_restrictions(self) -> ResourceRestrictionsClient:
        """Access the resource restrictions client for restricting resources (lazy-loaded)."""
        return cast(
            "ResourceRestrictionsClient",
            self.__getattr__("resource_restrictions"),
        )

    @property
    def role_bindings(self) -> RoleBindingsClient:
        """Access the role bindings client for role binding operations (lazy-loaded)."""
        return cast("RoleBindingsClient", self.__getattr__("role_bindings"))

    @property
    def roles(self) -> RolesClient:
        """Access the roles client for role management operations (lazy-loaded)."""
        return cast("RolesClient", self.__getattr__("roles"))

    @property
    def tasks(self) -> TasksClient:
        """Access the tasks client for task and task run operations (lazy-loaded)."""
        return cast("TasksClient", self.__getattr__("tasks"))

    @property
    def users(self) -> UsersClient:
        """Access the users client for user management operations (lazy-loaded)."""
        return cast("UsersClient", self.__getattr__("users"))

    def __repr__(self) -> str:
        """Return a string representation of the Arize client configuration."""
        # The repr looks like:
        #  ArizeClient(
        #   sdk_config=SDKConfiguration(
        #     api_key='cacaca***',
        #     api_host='api.arize.com',
        #     ...
        #     arize_directory='~/.arize',
        #     enable_caching=True,
        #   )
        #   subclients={
        #     'datasets': lazy,
        #     'experiments': lazy,
        #     'spans': lazy,
        #     'ml': lazy,
        #   }
        # )
        lines = [f"{self.__class__.__name__}("]
        # Indent the SDKConfiguration repr
        cfg_repr = repr(self.sdk_config).splitlines()
        lines.append(f"  sdk_config={cfg_repr[0]}")
        lines.extend("  " + line for line in cfg_repr[1:])
        # Add subclient states
        lines.append("  subclients={")
        for name in self._SUBCLIENTS:
            state = "loaded" if name in self._lazy_cache else "lazy"
            lines.append(f"    {name!r}: {state},")
        lines.append("  }")
        lines.append(")")
        return "\n".join(lines)

    def clear_cache(self) -> None:
        """Clear the local cache directory.

        Removes all cached data from the SDK's cache directory. This can be useful
        when troubleshooting caching issues or freeing up disk space. The cache
        directory is automatically recreated on subsequent operations that require
        caching.

        Raises:
            NotADirectoryError: If the cache path exists but is not a directory.

        Notes:
            - This operation permanently deletes all cached data
            - If the cache directory doesn't exist, a warning is logged but no error is raised
            - The cache directory location is configured via sdk_config.cache_dir
        """
        p = Path(self.sdk_config.cache_dir).expanduser().resolve()

        if not p.exists():
            logger.warning(
                f"Cache directory does not exist at {p}, nothing to clear"
            )
        if not p.is_dir():
            logger.error(f"Cache path is not a directory: {p}")
            raise NotADirectoryError(f"Not a directory: {p}")

        logger.info(f"Clearing cache directory at {p}")
        shutil.rmtree(p)
