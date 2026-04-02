"""Client implementation for managing annotation queues in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    _find_annotation_queue_id,
    _find_space_id,
    _resolve_resource,
)

if TYPE_CHECKING:
    import builtins

    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
    from arize._generated.api_client.models.assignment_method import (
        AssignmentMethod,
    )
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class AnnotationQueuesClient:
    """Client for managing annotation queues.

    Supports creation, retrieval, update, deletion, and record management.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The annotation queues client is a thin wrapper around the generated REST API client,
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

        self._api = gen.AnnotationQueuesApi(generated_client)
        self._spaces_api = gen.SpacesApi(generated_client)

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    @prerelease_endpoint(key="annotation_queues.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        space: str | None = None,
        name: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.AnnotationQueuesList200Response:
        """List annotation queues the user has access to.

        Annotation queues are returned in descending creation order (most recently
        created first).

        Args:
            space: Optional space filter. If the value is a base64-encoded resource
                ID it is treated as a space ID; otherwise it is used as a
                case-insensitive substring filter on the space name.
            name: Optional name substring to filter results.
            limit: Maximum number of annotation queues to return. The server
                enforces an upper bound.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the annotation queues and pagination information.

        Raises:
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/429).
        """
        resolved_space = _resolve_resource(space)
        return self._api.annotation_queues_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="annotation_queues.get", stage=ReleaseStage.ALPHA)
    def get(
        self, *, annotation_queue: str, space: str | None = None
    ) -> models.AnnotationQueue:
        """Get an annotation queue by ID or name.

        Args:
            annotation_queue: Annotation queue ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_queue* is a
                name so it can be resolved to an ID.

        Returns:
            The annotation queue object.

        Raises:
            NotFoundError: If the annotation queue name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 401/403/404/429).
        """
        annotation_queue_id = _find_annotation_queue_id(
            api=self._api,
            annotation_queue=annotation_queue,
            space=space,
        )
        return self._api.annotation_queues_get(
            annotation_queue_id=annotation_queue_id
        )

    @prerelease_endpoint(
        key="annotation_queues.create", stage=ReleaseStage.ALPHA
    )
    def create(
        self,
        *,
        name: str,
        space: str,
        annotation_config_ids: builtins.list[str],
        annotator_emails: builtins.list[str],
        instructions: str | None = None,
        assignment_method: AssignmentMethod | None = None,
        record_sources: builtins.list[models.AnnotationQueueRecordInput]
        | None = None,
    ) -> models.AnnotationQueue:
        """Create an annotation queue.

        Args:
            name: Annotation queue name (must be unique within the target space,
                max 255 characters).
            space: Space ID or name to create the annotation queue in.
            annotation_config_ids: IDs of annotation configs to associate with
                the queue (at least one required).
            annotator_emails: Emails of users to assign as annotators (at least
                one required). All users must have active accounts with access to
                the space.
            instructions: Optional instructions for annotators (max 5000 characters).
            assignment_method: How records are assigned to annotators. Defaults to
                ``ALL`` (every annotator sees every record).
            record_sources: Optional initial record sources to populate the queue
                (at most 2 sources per request).

        Returns:
            The created annotation queue object as returned by the API.

        Raises:
            NotFoundError: If the space name cannot be resolved to an ID.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/409/429).
        """
        from arize._generated import api_client as gen

        space_id = _find_space_id(self._spaces_api, space)

        body = gen.CreateAnnotationQueueRequestBody(
            name=name,
            space_id=space_id,
            annotation_config_ids=annotation_config_ids,
            annotator_emails=annotator_emails,
            instructions=instructions,
            assignment_method=assignment_method,
            record_sources=record_sources,
        )
        return self._api.annotation_queues_create(
            create_annotation_queue_request_body=body
        )

    @prerelease_endpoint(
        key="annotation_queues.update", stage=ReleaseStage.ALPHA
    )
    def update(
        self,
        *,
        annotation_queue: str,
        space: str | None = None,
        name: str | None = None,
        instructions: str | None = None,
        annotation_config_ids: builtins.list[str] | None = None,
        annotator_emails: builtins.list[str] | None = None,
    ) -> models.AnnotationQueue:
        """Update an annotation queue.

        At least one field must be provided. List fields (``annotation_config_ids``,
        ``annotator_emails``) fully replace the existing values when provided.

        Args:
            annotation_queue: Annotation queue ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_queue* is a
                name so it can be resolved to an ID.
            name: New name for the queue (must remain unique within the space).
            instructions: New instructions for annotators. Pass an empty string
                to clear existing instructions.
            annotation_config_ids: Full replacement list of annotation config IDs.
            annotator_emails: Full replacement list of annotator emails.

        Returns:
            The updated annotation queue object as returned by the API.

        Raises:
            NotFoundError: If the annotation queue name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/409/429).
        """
        from arize._generated import api_client as gen

        annotation_queue_id = _find_annotation_queue_id(
            api=self._api,
            annotation_queue=annotation_queue,
            space=space,
        )
        body = gen.UpdateAnnotationQueueRequestBody(
            name=name,
            instructions=instructions,
            annotation_config_ids=annotation_config_ids,
            annotator_emails=annotator_emails,
        )
        return self._api.annotation_queues_update(
            annotation_queue_id=annotation_queue_id,
            update_annotation_queue_request_body=body,
        )

    @prerelease_endpoint(
        key="annotation_queues.delete", stage=ReleaseStage.ALPHA
    )
    def delete(
        self, *, annotation_queue: str, space: str | None = None
    ) -> None:
        """Delete an annotation queue by ID or name.

        This operation is irreversible.

        Args:
            annotation_queue: Annotation queue ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_queue* is a
                name so it can be resolved to an ID.

        Returns:
            This method returns None on success (HTTP 204 No Content).

        Raises:
            NotFoundError: If the annotation queue name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 401/403/404/429).
        """
        annotation_queue_id = _find_annotation_queue_id(
            api=self._api,
            annotation_queue=annotation_queue,
            space=space,
        )
        return self._api.annotation_queues_delete(
            annotation_queue_id=annotation_queue_id
        )

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------

    @prerelease_endpoint(
        key="annotation_queues.list_records", stage=ReleaseStage.ALPHA
    )
    def list_records(
        self,
        *,
        annotation_queue: str,
        space: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.AnnotationQueueRecordsList200Response:
        """List records in an annotation queue.

        Each record includes its data as flat key-value pairs, any annotations
        that have been added, and the users assigned to annotate it.

        Args:
            annotation_queue: Annotation queue ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_queue* is a
                name so it can be resolved to an ID.
            limit: Maximum number of records to return (server enforces an upper
                bound of 500).
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the records and pagination information.

        Raises:
            NotFoundError: If the annotation queue name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 401/403/404/429).
        """
        annotation_queue_id = _find_annotation_queue_id(
            api=self._api,
            annotation_queue=annotation_queue,
            space=space,
        )
        return self._api.annotation_queue_records_list(
            annotation_queue_id=annotation_queue_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(
        key="annotation_queues.add_records", stage=ReleaseStage.ALPHA
    )
    def add_records(
        self,
        *,
        annotation_queue: str,
        space: str | None = None,
        record_sources: builtins.list[models.AnnotationQueueRecordInput],
    ) -> models.AnnotationQueuesRecordsCreate200Response:
        """Add records to an annotation queue.

        Records may come from spans (a project time range) or dataset examples.
        At most 2 record sources and 500 total records may be added per request.

        Args:
            annotation_queue: Annotation queue ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_queue* is a
                name so it can be resolved to an ID.
            record_sources: List of record sources (1-2 sources). Each source is
                an :class:`~arize._generated.api_client.models.AnnotationQueueRecordInput`
                wrapping either an
                :class:`~arize._generated.api_client.models.AnnotationQueueSpanRecordInput`
                or
                :class:`~arize._generated.api_client.models.AnnotationQueueExampleRecordInput`.

        Returns:
            A response object containing the created record sources.

        Raises:
            NotFoundError: If the annotation queue name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        from arize._generated import api_client as gen

        annotation_queue_id = _find_annotation_queue_id(
            api=self._api,
            annotation_queue=annotation_queue,
            space=space,
        )
        body = gen.AddAnnotationQueueRecordsRequestBody(
            record_sources=record_sources,
        )
        return self._api.annotation_queues_records_create(
            annotation_queue_id=annotation_queue_id,
            add_annotation_queue_records_request_body=body,
        )

    @prerelease_endpoint(
        key="annotation_queues.delete_records", stage=ReleaseStage.ALPHA
    )
    def delete_records(
        self,
        *,
        annotation_queue: str,
        space: str | None = None,
        record_ids: builtins.list[str],
    ) -> None:
        """Delete records from an annotation queue.

        Record IDs that are not found or do not belong to the specified queue are
        silently ignored. A successful response does not guarantee all provided IDs
        were deleted.

        Args:
            annotation_queue: Annotation queue ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_queue* is a
                name so it can be resolved to an ID.
            record_ids: IDs of records to delete (1-100 IDs per request).

        Returns:
            This method returns None on success (HTTP 204 No Content).

        Raises:
            NotFoundError: If the annotation queue name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        from arize._generated import api_client as gen

        annotation_queue_id = _find_annotation_queue_id(
            api=self._api,
            annotation_queue=annotation_queue,
            space=space,
        )
        body = gen.DeleteAnnotationQueueRecordsRequestBody(
            record_ids=record_ids,
        )
        return self._api.annotation_queues_records_delete(
            annotation_queue_id=annotation_queue_id,
            delete_annotation_queue_records_request_body=body,
        )

    @prerelease_endpoint(
        key="annotation_queues.annotate_record", stage=ReleaseStage.ALPHA
    )
    def annotate_record(
        self,
        *,
        annotation_queue: str,
        space: str | None = None,
        record_id: str,
        annotations: builtins.list[models.AnnotationInput],
    ) -> models.AnnotationQueueRecordAnnotateResult:
        """Submit annotations for an annotation queue record.

        Annotations are upserted by annotation config name; omitted configs are
        left unchanged.

        Args:
            annotation_queue: Annotation queue ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_queue* is a
                name so it can be resolved to an ID.
            record_id: ID of the record to annotate.
            annotations: Annotation values to upsert. Each entry targets one
                annotation config by name and may set a ``score``, ``label``,
                and/or ``text``.

        Returns:
            A result object with the submitted annotation values.

        Raises:
            NotFoundError: If the annotation queue name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        from arize._generated import api_client as gen

        annotation_queue_id = _find_annotation_queue_id(
            api=self._api,
            annotation_queue=annotation_queue,
            space=space,
        )
        body = gen.AnnotateAnnotationQueueRecordRequestBody(
            annotations=annotations,
        )
        return self._api.annotation_queues_records_annotate(
            annotation_queue_id=annotation_queue_id,
            annotation_queue_record_id=record_id,
            annotate_annotation_queue_record_request_body=body,
        )

    @prerelease_endpoint(
        key="annotation_queues.assign_record", stage=ReleaseStage.ALPHA
    )
    def assign_record(
        self,
        *,
        annotation_queue: str,
        space: str | None = None,
        record_id: str,
        assigned_user_emails: builtins.list[str],
    ) -> models.AnnotationQueueRecordAssignResult:
        """Assign users to an annotation queue record.

        Fully replaces the current record-level user assignment. Pass an empty
        list to remove all assignments.

        Args:
            annotation_queue: Annotation queue ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_queue* is a
                name so it can be resolved to an ID.
            record_id: ID of the record to assign users to.
            assigned_user_emails: Emails of users to assign (at most 100). Replaces
                all existing record-level assignments.

        Returns:
            A result object with the updated user assignments.

        Raises:
            NotFoundError: If the annotation queue name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        from arize._generated import api_client as gen

        annotation_queue_id = _find_annotation_queue_id(
            api=self._api,
            annotation_queue=annotation_queue,
            space=space,
        )
        body = gen.AssignAnnotationQueueRecordRequestBody(
            assigned_user_emails=assigned_user_emails,
        )
        return self._api.annotation_queues_records_assign(
            annotation_queue_id=annotation_queue_id,
            annotation_queue_record_id=record_id,
            assign_annotation_queue_record_request_body=body,
        )
