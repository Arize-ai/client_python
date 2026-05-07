"""Unit tests for src/arize/annotation_queues/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize._generated.api_client.models.assignment_method import (
    AssignmentMethod,
)
from arize.annotation_queues.client import AnnotationQueuesClient

# Base64 IDs that pass is_resource_id() — decode to "Type:123"
_QUEUE_ID = "QW5ub3RhdGlvblF1ZXVlOjEyMw=="  # AnnotationQueue:123


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock AnnotationQueuesApi instance."""
    return Mock()


@pytest.fixture
def annotation_queues_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> AnnotationQueuesClient:
    """Provide an AnnotationQueuesClient with mocked internals."""
    with (
        patch(
            "arize._generated.api_client.AnnotationQueuesApi",
            return_value=mock_api,
        ),
        patch(
            "arize._generated.api_client.SpacesApi",
        ),
    ):
        return AnnotationQueuesClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestAnnotationQueuesClientInit:
    """Tests for AnnotationQueuesClient initialisation."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor must store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.AnnotationQueuesApi",
            return_value=mock_api,
        ):
            client = AnnotationQueuesClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )

        assert client._sdk_config is mock_sdk_config

    def test_creates_api_with_generated_client(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor must instantiate AnnotationQueuesApi with the provided client."""
        mock_generated_client = Mock()

        with patch(
            "arize._generated.api_client.AnnotationQueuesApi"
        ) as mock_api_cls:
            mock_api_cls.return_value = mock_api
            AnnotationQueuesClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )

        mock_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestAnnotationQueuesClientList:
    """Tests for AnnotationQueuesClient.list()."""

    def test_list_with_space_id(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """list() should resolve a base64 resource ID space value to space_id."""
        annotation_queues_client.list(
            space="U3BhY2U6OTA1MDoxSmtS",
            name="review",
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.annotation_queues_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name="review",
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_with_space_name(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        annotation_queues_client.list(
            space="my-space",
            name="review",
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.annotation_queues_list.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="review",
            limit=50,
            cursor="cursor-abc",
        )

    def test_uses_default_limit(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """list() must default limit to 100 when not provided."""
        annotation_queues_client.list()

        _, kwargs = mock_api.annotation_queues_list.call_args
        assert kwargs["limit"] == 100

    def test_returns_api_response(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """list() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_queues_list.return_value = expected

        result = annotation_queues_client.list()

        assert result is expected

    def test_emits_alpha_prerelease_warning(
        self,
        annotation_queues_client: AnnotationQueuesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to list() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        annotation_queues_client.list()

        assert any(
            "ALPHA" in record.message
            and "annotation_queues.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestAnnotationQueuesClientGet:
    """Tests for AnnotationQueuesClient.get()."""

    def test_forwards_id(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """get() with a base64 resource ID must pass it directly as annotation_queue_id."""
        annotation_queues_client.get(annotation_queue=_QUEUE_ID)

        mock_api.annotation_queues_get.assert_called_once_with(
            annotation_queue_id=_QUEUE_ID
        )

    def test_returns_api_response(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """get() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_queues_get.return_value = expected

        result = annotation_queues_client.get(annotation_queue=_QUEUE_ID)

        assert result is expected


@pytest.mark.unit
class TestAnnotationQueuesClientCreate:
    """Tests for AnnotationQueuesClient.create()."""

    def test_builds_request_body_with_required_fields(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """create() must build CreateAnnotationQueueRequestBody with required args."""
        # Use a base64 resource ID so find_space_id returns it directly without
        # calling the spaces API.
        with patch(
            "arize._generated.api_client.CreateAnnotationQueueRequestBody"
        ) as mock_body_cls:
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            annotation_queues_client.create(
                name="Review Queue",
                space="U3BhY2U6OTA1MDoxSmtS",
                annotation_config_ids=["ac_001"],
                annotator_emails=["reviewer@example.com"],
            )

        mock_body_cls.assert_called_once_with(
            name="Review Queue",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            annotation_config_ids=["ac_001"],
            annotator_emails=["reviewer@example.com"],
            instructions=None,
            assignment_method=None,
            record_sources=None,
        )
        mock_api.annotation_queues_create.assert_called_once_with(
            create_annotation_queue_request_body=mock_body
        )

    def test_converts_assignment_method_enum_to_value(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """create() must pass assignment_method.value (str), not the enum itself."""
        with patch(
            "arize._generated.api_client.CreateAnnotationQueueRequestBody"
        ) as mock_body_cls:
            annotation_queues_client.create(
                name="Q",
                space="U3BhY2U6OTA1MDoxSmtS",
                annotation_config_ids=["ac_001"],
                annotator_emails=["reviewer@example.com"],
                assignment_method=AssignmentMethod.RANDOM,
            )

        _, kwargs = mock_body_cls.call_args
        assert kwargs["assignment_method"] == "random"

    def test_assignment_method_all_value(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """AssignmentMethod.ALL must map to the string 'all'."""
        with patch(
            "arize._generated.api_client.CreateAnnotationQueueRequestBody"
        ) as mock_body_cls:
            annotation_queues_client.create(
                name="Q",
                space="U3BhY2U6OTA1MDoxSmtS",
                annotation_config_ids=["ac_001"],
                annotator_emails=["reviewer@example.com"],
                assignment_method=AssignmentMethod.ALL,
            )

        _, kwargs = mock_body_cls.call_args
        assert kwargs["assignment_method"] == "all"

    def test_returns_api_response(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """create() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_queues_create.return_value = expected

        with patch(
            "arize._generated.api_client.CreateAnnotationQueueRequestBody"
        ):
            result = annotation_queues_client.create(
                name="Q",
                space="U3BhY2U6OTA1MDoxSmtS",
                annotation_config_ids=["ac_001"],
                annotator_emails=["reviewer@example.com"],
            )

        assert result is expected

    def test_emits_alpha_prerelease_warning(
        self,
        annotation_queues_client: AnnotationQueuesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch(
            "arize._generated.api_client.CreateAnnotationQueueRequestBody"
        ):
            annotation_queues_client.create(
                name="Q",
                space="U3BhY2U6OTA1MDoxSmtS",
                annotation_config_ids=["ac_001"],
                annotator_emails=["reviewer@example.com"],
            )

        assert any(
            "ALPHA" in record.message
            and "annotation_queues.create" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestAnnotationQueuesClientUpdate:
    """Tests for AnnotationQueuesClient.update()."""

    def test_builds_request_body(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """update() must only include the fields that were explicitly provided."""
        with patch(
            "arize._generated.api_client.UpdateAnnotationQueueRequestBody"
        ) as mock_body_cls:
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            annotation_queues_client.update(
                annotation_queue=_QUEUE_ID,
                name="New Name",
                instructions="Please review carefully.",
            )

        # annotation_config_ids and annotator_emails were not provided — must be absent
        mock_body_cls.assert_called_once_with(
            name="New Name",
            instructions="Please review carefully.",
        )
        mock_api.annotation_queues_update.assert_called_once_with(
            annotation_queue_id=_QUEUE_ID,
            update_annotation_queue_request_body=mock_body,
        )

    def test_raises_when_no_fields_provided(
        self, annotation_queues_client: AnnotationQueuesClient
    ) -> None:
        """update() must raise ValueError when no fields are provided."""
        with pytest.raises(ValueError, match="At least one of"):
            annotation_queues_client.update(annotation_queue=_QUEUE_ID)

    def test_empty_string_instructions_sends_through(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """update() should send instructions='' as-is to clear it on the server."""
        with patch(
            "arize._generated.api_client.UpdateAnnotationQueueRequestBody"
        ) as mock_body_cls:
            mock_body_cls.return_value = Mock()

            annotation_queues_client.update(
                annotation_queue=_QUEUE_ID,
                instructions="",
            )

        mock_body_cls.assert_called_once_with(instructions="")

    def test_returns_api_response(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """update() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_queues_update.return_value = expected

        with patch(
            "arize._generated.api_client.UpdateAnnotationQueueRequestBody"
        ):
            result = annotation_queues_client.update(
                annotation_queue=_QUEUE_ID, name="New"
            )

        assert result is expected


@pytest.mark.unit
class TestAnnotationQueuesClientDelete:
    """Tests for AnnotationQueuesClient.delete()."""

    def test_forwards_id(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """delete() with a base64 resource ID must pass it directly as annotation_queue_id."""
        annotation_queues_client.delete(annotation_queue=_QUEUE_ID)

        mock_api.annotation_queues_delete.assert_called_once_with(
            annotation_queue_id=_QUEUE_ID
        )

    def test_returns_none(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """delete() must return the API result (None on 204)."""
        mock_api.annotation_queues_delete.return_value = None

        result = annotation_queues_client.delete(annotation_queue=_QUEUE_ID)

        assert result is None


@pytest.mark.unit
class TestAnnotationQueuesClientListRecords:
    """Tests for AnnotationQueuesClient.list_records()."""

    def test_forwards_all_params(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """list_records() must forward all parameters to the generated API."""
        annotation_queues_client.list_records(
            annotation_queue=_QUEUE_ID,
            limit=50,
            cursor="cursor-xyz",
        )

        mock_api.annotation_queue_records_list.assert_called_once_with(
            annotation_queue_id=_QUEUE_ID,
            limit=50,
            cursor="cursor-xyz",
        )

    def test_uses_default_limit(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """list_records() must default limit to 100."""
        annotation_queues_client.list_records(annotation_queue=_QUEUE_ID)

        _, kwargs = mock_api.annotation_queue_records_list.call_args
        assert kwargs["limit"] == 100

    def test_returns_api_response(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """list_records() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_queue_records_list.return_value = expected

        result = annotation_queues_client.list_records(
            annotation_queue=_QUEUE_ID
        )

        assert result is expected


@pytest.mark.unit
class TestAnnotationQueuesClientAddRecords:
    """Tests for AnnotationQueuesClient.add_records()."""

    def test_builds_request_body(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """add_records() must build AddAnnotationQueueRecordsRequestBody."""
        mock_sources = [Mock(), Mock()]

        with patch(
            "arize._generated.api_client.AddAnnotationQueueRecordsRequestBody"
        ) as mock_body_cls:
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            annotation_queues_client.add_records(
                annotation_queue=_QUEUE_ID,
                record_sources=mock_sources,
            )

        mock_body_cls.assert_called_once_with(record_sources=mock_sources)
        mock_api.annotation_queues_records_create.assert_called_once_with(
            annotation_queue_id=_QUEUE_ID,
            add_annotation_queue_records_request_body=mock_body,
        )

    def test_returns_api_response(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """add_records() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_queues_records_create.return_value = expected

        with patch(
            "arize._generated.api_client.AddAnnotationQueueRecordsRequestBody"
        ):
            result = annotation_queues_client.add_records(
                annotation_queue=_QUEUE_ID, record_sources=[Mock()]
            )

        assert result is expected


@pytest.mark.unit
class TestAnnotationQueuesClientDeleteRecords:
    """Tests for AnnotationQueuesClient.delete_records()."""

    def test_builds_request_body(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """delete_records() must build DeleteAnnotationQueueRecordsRequestBody."""
        with patch(
            "arize._generated.api_client.DeleteAnnotationQueueRecordsRequestBody"
        ) as mock_body_cls:
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            annotation_queues_client.delete_records(
                annotation_queue=_QUEUE_ID,
                record_ids=["rec_001", "rec_002"],
            )

        mock_body_cls.assert_called_once_with(record_ids=["rec_001", "rec_002"])
        mock_api.annotation_queues_records_delete.assert_called_once_with(
            annotation_queue_id=_QUEUE_ID,
            delete_annotation_queue_records_request_body=mock_body,
        )

    def test_returns_none(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """delete_records() must return the API result (None on 204)."""
        mock_api.annotation_queues_records_delete.return_value = None

        with patch(
            "arize._generated.api_client.DeleteAnnotationQueueRecordsRequestBody"
        ):
            result = annotation_queues_client.delete_records(
                annotation_queue=_QUEUE_ID, record_ids=["rec_001"]
            )

        assert result is None


@pytest.mark.unit
class TestAnnotationQueuesClientAnnotateRecord:
    """Tests for AnnotationQueuesClient.annotate_record()."""

    def test_builds_request_body(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """annotate_record() must build AnnotateAnnotationQueueRecordRequestBody."""
        mock_annotations = [Mock(), Mock()]

        with patch(
            "arize._generated.api_client.AnnotateAnnotationQueueRecordRequestBody"
        ) as mock_body_cls:
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            annotation_queues_client.annotate_record(
                annotation_queue=_QUEUE_ID,
                record_id="rec_001",
                annotations=mock_annotations,
            )

        mock_body_cls.assert_called_once_with(annotations=mock_annotations)
        mock_api.annotation_queues_records_annotate.assert_called_once_with(
            annotation_queue_id=_QUEUE_ID,
            annotation_queue_record_id="rec_001",
            annotate_annotation_queue_record_request_body=mock_body,
        )

    def test_returns_api_response(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """annotate_record() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_queues_records_annotate.return_value = expected

        with patch(
            "arize._generated.api_client.AnnotateAnnotationQueueRecordRequestBody"
        ):
            result = annotation_queues_client.annotate_record(
                annotation_queue=_QUEUE_ID,
                record_id="rec_001",
                annotations=[Mock()],
            )

        assert result is expected

    def test_emits_alpha_prerelease_warning(
        self,
        annotation_queues_client: AnnotationQueuesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to annotate_record() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch(
            "arize._generated.api_client.AnnotateAnnotationQueueRecordRequestBody"
        ):
            annotation_queues_client.annotate_record(
                annotation_queue=_QUEUE_ID,
                record_id="rec_001",
                annotations=[Mock()],
            )

        assert any(
            "ALPHA" in record.message
            and "annotation_queues.annotate_record" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestAnnotationQueuesClientAssignRecord:
    """Tests for AnnotationQueuesClient.assign_record()."""

    def test_builds_request_body(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """assign_record() must build AssignAnnotationQueueRecordRequestBody."""
        emails = ["alice@example.com", "bob@example.com"]

        with patch(
            "arize._generated.api_client.AssignAnnotationQueueRecordRequestBody"
        ) as mock_body_cls:
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            annotation_queues_client.assign_record(
                annotation_queue=_QUEUE_ID,
                record_id="rec_001",
                assigned_user_emails=emails,
            )

        mock_body_cls.assert_called_once_with(assigned_user_emails=emails)
        mock_api.annotation_queues_records_assign.assert_called_once_with(
            annotation_queue_id=_QUEUE_ID,
            annotation_queue_record_id="rec_001",
            assign_annotation_queue_record_request_body=mock_body,
        )

    def test_supports_empty_email_list(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """assign_record() must accept an empty list to remove all assignments."""
        with patch(
            "arize._generated.api_client.AssignAnnotationQueueRecordRequestBody"
        ) as mock_body_cls:
            annotation_queues_client.assign_record(
                annotation_queue=_QUEUE_ID,
                record_id="rec_001",
                assigned_user_emails=[],
            )

        _, kwargs = mock_body_cls.call_args
        assert kwargs["assigned_user_emails"] == []

    def test_returns_api_response(
        self, annotation_queues_client: AnnotationQueuesClient, mock_api: Mock
    ) -> None:
        """assign_record() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_queues_records_assign.return_value = expected

        with patch(
            "arize._generated.api_client.AssignAnnotationQueueRecordRequestBody"
        ):
            result = annotation_queues_client.assign_record(
                annotation_queue=_QUEUE_ID,
                record_id="rec_001",
                assigned_user_emails=["user@example.com"],
            )

        assert result is expected

    def test_emits_alpha_prerelease_warning(
        self,
        annotation_queues_client: AnnotationQueuesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to assign_record() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch(
            "arize._generated.api_client.AssignAnnotationQueueRecordRequestBody"
        ):
            annotation_queues_client.assign_record(
                annotation_queue=_QUEUE_ID,
                record_id="rec_001",
                assigned_user_emails=["user@example.com"],
            )

        assert any(
            "ALPHA" in record.message
            and "annotation_queues.assign_record" in record.message
            for record in caplog.records
        )
