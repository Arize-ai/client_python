"""Client implementation for managing traces in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import _find_project_id

if TYPE_CHECKING:
    from datetime import datetime

    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.traces.types import ListTracesResponse

logger = logging.getLogger(__name__)


class TracesClient:
    """Client for listing LLM traces from the Arize platform.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The traces client is a thin wrapper around the generated REST API client,
    using the shared generated API client owned by
    :class:`arize.config.SDKConfiguration`.
    """

    def __init__(
        self, *, sdk_config: SDKConfiguration, generated_client: ApiClient
    ) -> None:
        """
        Args:
            sdk_config: Resolved SDK configuration.
            generated_client: Shared generated API client instance.
        """  # noqa: D205, D212
        self._sdk_config = sdk_config

        # Import at runtime so it's still lazy and extras-gated by the parent
        from arize._generated import api_client as gen

        # Use the provided client directly
        self._api = gen.TracesApi(generated_client)
        self._projects_api = gen.ProjectsApi(generated_client)
        self._spaces_api = gen.SpacesApi(generated_client)

    @prerelease_endpoint(key="traces.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        project: str,
        space: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        filter: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListTracesResponse:
        """List traces for a project within a time range.

        Each returned trace carries its full (flat) list of spans plus
        lightweight roll-up metadata. Traces are returned newest-first.

        The ``filter`` uses the same expression syntax as
        :meth:`arize.spans.client.SpansClient.list`, but the semantics differ:
        a ``filter`` selects traces that contain at least one matching span
        (the matching span is usually a child, not the root), rather than only
        traces whose root span matches.

        Args:
            project: Project name or identifier (base64) to list traces for.
                If the value is a name, ``space`` must also be provided.
            space: Optional space name or ID used to disambiguate the project
                lookup. Required when ``project`` is a name.
            start_time: Inclusive lower bound of the time window. Defaults to
                seven days before the request time.
            end_time: Exclusive upper bound of the time window. Defaults to the
                request time.
            filter: Optional filter expression to narrow results. Supports
                equality, comparison, and SQL-style ``AND``/``OR`` operators.
                A trace is returned when **any** of its spans matches the
                filter. Examples::

                    "status_code = 'ERROR'"
                    "span_kind = 'LLM'"
                    "status_code = 'ERROR' AND span_kind = 'LLM'"
            limit: Maximum number of traces to return. The server enforces an
                upper bound. Defaults to 50.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the traces and pagination information.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/429).
        """
        project_id = _find_project_id(
            api=self._projects_api,
            spaces_api=self._spaces_api,
            project=project,
            space=space,
        )
        from arize._generated import api_client as gen

        body = gen.ListTracesRequest(
            project_id=project_id,
            start_time=start_time,
            end_time=end_time,
            filter=filter,
        )
        return self._api.list_traces(
            list_traces_request=body,
            limit=limit,
            cursor=cursor,
        )
