"""Unit tests for src/arize/evaluators/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.evaluators.client import EvaluatorsClient

# Base64 ID that decodes to "Evaluator:123" — passes _is_resource_id()
_EVALUATOR_ID = "RXZhbHVhdG9yOjEyMw=="


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock EvaluatorsApi instance."""
    return Mock()


@pytest.fixture
def evaluators_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> EvaluatorsClient:
    """Provide an EvaluatorsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.EvaluatorsApi", return_value=mock_api
    ):
        return EvaluatorsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestEvaluatorsClientInit:
    """Tests for EvaluatorsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.EvaluatorsApi", return_value=mock_api
        ):
            client = EvaluatorsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_evaluators_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to EvaluatorsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.EvaluatorsApi"
        ) as mock_evaluators_api_cls:
            EvaluatorsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_evaluators_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestEvaluatorsClientList:
    """Tests for EvaluatorsClient.list()."""

    def test_list_with_space_id(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a base64 resource ID space value to space_id."""
        evaluators_client.list(
            name="my-evaluator",
            space="U3BhY2U6OTA1MDoxSmtS",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.evaluators_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name="my-evaluator",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_with_space_name(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        evaluators_client.list(
            name="my-evaluator",
            space="my-space",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.evaluators_list.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="my-evaluator",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_defaults(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list() should default space/name/cursor to None and limit to 100."""
        evaluators_client.list()

        mock_api.evaluators_list.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from evaluators_list."""
        expected = Mock()
        mock_api.evaluators_list.return_value = expected

        result = evaluators_client.list()

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.list()

        assert any(
            "ALPHA" in record.message and "evaluators.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientGet:
    """Tests for EvaluatorsClient.get()."""

    def test_get_calls_api_with_evaluator_id(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get() should resolve evaluator and forward evaluator_id to evaluators_get."""
        evaluators_client.get(evaluator=_EVALUATOR_ID)

        mock_api.evaluators_get.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            version_id=None,
        )

    def test_get_with_version_id(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get() should forward version_id when provided."""
        evaluators_client.get(evaluator=_EVALUATOR_ID, version_id="ver-456")

        mock_api.evaluators_get.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            version_id="ver-456",
        )

    def test_get_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from evaluators_get."""
        expected = Mock()
        mock_api.evaluators_get.return_value = expected

        result = evaluators_client.get(evaluator=_EVALUATOR_ID)

        assert result is expected

    def test_get_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.get(evaluator=_EVALUATOR_ID)

        assert any(
            "ALPHA" in record.message and "evaluators.get" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientCreate:
    """Tests for EvaluatorsClient.create()."""

    def test_create_builds_request_and_calls_api(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create() should build the request objects and call evaluators_create."""
        mock_template_config = Mock()

        with (
            patch(
                "arize._generated.api_client.EvaluatorsCreateRequestVersion"
            ) as mock_version_cls,
            patch(
                "arize._generated.api_client.EvaluatorsCreateRequest"
            ) as mock_request_cls,
        ):
            mock_version = Mock()
            mock_version_cls.return_value = mock_version
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            evaluators_client.create(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial version",
                template_config=mock_template_config,
            )

        mock_version_cls.assert_called_once_with(
            commit_message="initial version",
            template_config=mock_template_config,
        )
        mock_request_cls.assert_called_once_with(
            name="my-evaluator",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            type="template",
            description=None,
            version=mock_version,
        )
        mock_api.evaluators_create.assert_called_once_with(
            evaluators_create_request=mock_body
        )

    def test_create_explicit_template_type(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create() should accept evaluator_type='template' explicitly."""
        with (
            patch("arize._generated.api_client.EvaluatorsCreateRequestVersion"),
            patch(
                "arize._generated.api_client.EvaluatorsCreateRequest"
            ) as mock_request_cls,
        ):
            mock_request_cls.return_value = Mock()

            evaluators_client.create(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                evaluator_type="template",
                commit_message="initial",
                template_config=Mock(),
            )

        _, kwargs = mock_request_cls.call_args
        assert kwargs["type"] == "template"

    def test_create_with_description(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create() should forward description to EvaluatorsCreateRequest."""
        mock_template_config = Mock()

        with (
            patch("arize._generated.api_client.EvaluatorsCreateRequestVersion"),
            patch(
                "arize._generated.api_client.EvaluatorsCreateRequest"
            ) as mock_request_cls,
        ):
            mock_request_cls.return_value = Mock()

            evaluators_client.create(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=mock_template_config,
                description="An evaluator for relevance",
            )

        _, kwargs = mock_request_cls.call_args
        assert kwargs["description"] == "An evaluator for relevance"

    def test_create_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from evaluators_create."""
        expected = Mock()
        mock_api.evaluators_create.return_value = expected

        with (
            patch("arize._generated.api_client.EvaluatorsCreateRequestVersion"),
            patch("arize._generated.api_client.EvaluatorsCreateRequest"),
        ):
            result = evaluators_client.create(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=Mock(),
            )

        assert result is expected

    def test_create_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.EvaluatorsCreateRequestVersion"),
            patch("arize._generated.api_client.EvaluatorsCreateRequest"),
        ):
            evaluators_client.create(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=Mock(),
            )

        assert any(
            "ALPHA" in record.message and "evaluators.create" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientUpdate:
    """Tests for EvaluatorsClient.update()."""

    def test_update_with_name(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should build EvaluatorsUpdateRequest with only name when only name is given."""
        with patch(
            "arize._generated.api_client.EvaluatorsUpdateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            evaluators_client.update(evaluator=_EVALUATOR_ID, name="new-name")

        mock_request_cls.assert_called_once_with(
            name="new-name", description=None
        )
        mock_api.evaluators_update.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            evaluators_update_request=mock_body,
        )

    def test_update_with_description(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should forward description to EvaluatorsUpdateRequest."""
        with patch(
            "arize._generated.api_client.EvaluatorsUpdateRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            evaluators_client.update(
                evaluator=_EVALUATOR_ID, description="Updated description"
            )

        mock_request_cls.assert_called_once_with(
            name=None, description="Updated description"
        )

    def test_update_with_both_fields(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should forward both name and description."""
        with patch(
            "arize._generated.api_client.EvaluatorsUpdateRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            evaluators_client.update(
                evaluator=_EVALUATOR_ID,
                name="new-name",
                description="new description",
            )

        mock_request_cls.assert_called_once_with(
            name="new-name", description="new description"
        )

    def test_update_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from evaluators_update."""
        expected = Mock()
        mock_api.evaluators_update.return_value = expected

        with patch("arize._generated.api_client.EvaluatorsUpdateRequest"):
            result = evaluators_client.update(evaluator=_EVALUATOR_ID, name="x")

        assert result is expected

    def test_update_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to update() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.EvaluatorsUpdateRequest"):
            evaluators_client.update(evaluator=_EVALUATOR_ID, name="x")

        assert any(
            "ALPHA" in record.message and "evaluators.update" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientDelete:
    """Tests for EvaluatorsClient.delete()."""

    def test_delete_calls_api_with_evaluator_id(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """delete() should resolve evaluator and pass evaluator_id to evaluators_delete."""
        evaluators_client.delete(evaluator=_EVALUATOR_ID)

        mock_api.evaluators_delete.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID
        )

    def test_delete_returns_none(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """delete() should always return None (204 No Content) regardless of API return."""
        mock_api.evaluators_delete.return_value = "unexpected"

        result = evaluators_client.delete(evaluator=_EVALUATOR_ID)

        assert result is None

    def test_delete_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to delete() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.delete(evaluator=_EVALUATOR_ID)

        assert any(
            "ALPHA" in record.message and "evaluators.delete" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientListVersions:
    """Tests for EvaluatorsClient.list_versions()."""

    def test_list_versions_calls_api_with_all_params(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list_versions() should forward all parameters to evaluator_versions_list."""
        evaluators_client.list_versions(
            evaluator=_EVALUATOR_ID,
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.evaluator_versions_list.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_versions_defaults(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list_versions() should default limit to 100 and cursor to None."""
        evaluators_client.list_versions(evaluator=_EVALUATOR_ID)

        mock_api.evaluator_versions_list.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            limit=100,
            cursor=None,
        )

    def test_list_versions_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list_versions() should propagate the return value."""
        expected = Mock()
        mock_api.evaluator_versions_list.return_value = expected

        result = evaluators_client.list_versions(evaluator=_EVALUATOR_ID)

        assert result is expected

    def test_list_versions_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.list_versions(evaluator=_EVALUATOR_ID)

        assert any(
            "ALPHA" in record.message
            and "evaluators.list_versions" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientGetVersion:
    """Tests for EvaluatorsClient.get_version()."""

    def test_get_version_calls_api_with_version_id(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get_version() should pass version_id to evaluator_versions_get."""
        evaluators_client.get_version(version_id="ver-456")

        mock_api.evaluator_versions_get.assert_called_once_with(
            version_id="ver-456"
        )

    def test_get_version_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get_version() should propagate the return value."""
        expected = Mock()
        mock_api.evaluator_versions_get.return_value = expected

        result = evaluators_client.get_version(version_id="ver-456")

        assert result is expected

    def test_get_version_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.get_version(version_id="ver-456")

        assert any(
            "ALPHA" in record.message
            and "evaluators.get_version" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientCreateVersion:
    """Tests for EvaluatorsClient.create_version()."""

    def test_create_version_builds_request_and_calls_api(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_version() should build the request and call evaluator_versions_create."""
        mock_template_config = Mock()

        with patch(
            "arize._generated.api_client.EvaluatorVersionsCreateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            evaluators_client.create_version(
                evaluator=_EVALUATOR_ID,
                commit_message="fix prompt wording",
                template_config=mock_template_config,
            )

        mock_request_cls.assert_called_once_with(
            commit_message="fix prompt wording",
            template_config=mock_template_config,
        )
        mock_api.evaluator_versions_create.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            evaluator_versions_create_request=mock_body,
        )

    def test_create_version_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_version() should propagate the return value."""
        expected = Mock()
        mock_api.evaluator_versions_create.return_value = expected

        with patch(
            "arize._generated.api_client.EvaluatorVersionsCreateRequest"
        ):
            result = evaluators_client.create_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                template_config=Mock(),
            )

        assert result is expected

    def test_create_version_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch(
            "arize._generated.api_client.EvaluatorVersionsCreateRequest"
        ):
            evaluators_client.create_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                template_config=Mock(),
            )

        assert any(
            "ALPHA" in record.message
            and "evaluators.create_version" in record.message
            for record in caplog.records
        )
