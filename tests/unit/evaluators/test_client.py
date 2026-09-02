"""Unit tests for src/arize/evaluators/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, create_autospec, patch

import pytest

from arize._generated.api_client import EvaluatorsApi, UpdateEvaluatorRequest
from arize.evaluators.client import EvaluatorsClient
from arize.evaluators.types import (
    CodeConfigRequest,
    EvaluatorVersionCode,
    EvaluatorWithVersion,
    ListEvaluatorVersionsResponse,
    TemplateConfigInput,
)

# Base64 ID that decodes to "Evaluator:123" — passes _is_resource_id()
_EVALUATOR_ID = "RXZhbHVhdG9yOjEyMw=="


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock EvaluatorsApi instance."""
    return create_autospec(EvaluatorsApi, instance=True)


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

        mock_api.list_evaluators.assert_called_once_with(
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

        mock_api.list_evaluators.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="my-evaluator",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_defaults(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list() should default space/name/cursor to None and limit to 50."""
        evaluators_client.list()

        mock_api.list_evaluators.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=50,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from evaluators_list."""
        expected = Mock()
        mock_api.list_evaluators.return_value = expected

        result = evaluators_client.list()

        assert result is expected

    def test_list_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.list()

        assert any(
            "BETA" in record.message and "evaluators.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientGet:
    """Tests for EvaluatorsClient.get()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            EvaluatorWithVersion,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_get_calls_api_with_evaluator_id(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get() should resolve evaluator and forward evaluator_id to evaluators_get."""
        evaluators_client.get(evaluator=_EVALUATOR_ID)

        mock_api.get_evaluator.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            version_id=None,
        )

    def test_get_with_version_id(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get() should forward version_id when provided."""
        evaluators_client.get(evaluator=_EVALUATOR_ID, version_id="ver-456")

        mock_api.get_evaluator.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            version_id="ver-456",
        )

    def test_get_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from evaluators_get."""
        expected = Mock()
        mock_api.get_evaluator.return_value = expected

        result = evaluators_client.get(evaluator=_EVALUATOR_ID)

        assert result is expected

    def test_get_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.get(evaluator=_EVALUATOR_ID)

        assert any(
            "BETA" in record.message and "evaluators.get" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientCreateTemplate:
    """Tests for EvaluatorsClient.create_template_evaluator()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            EvaluatorWithVersion,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_create_template_builds_template_request(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_evaluator() should build a template-type create request."""
        mock_template_config = Mock(spec=TemplateConfigInput)

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluatorVersionRequest"
            ) as mock_template_cls,
            patch(
                "arize._generated.api_client.CreateEvaluatorVersionRequest"
            ) as mock_version_cls,
            patch(
                "arize._generated.api_client.CreateEvaluatorRequest"
            ) as mock_request_cls,
        ):
            mock_template = Mock()
            mock_template_cls.return_value = mock_template
            mock_version_wrap = Mock()
            mock_version_cls.return_value = mock_version_wrap
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            evaluators_client.create_template_evaluator(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial version",
                template_config=mock_template_config,
            )

        mock_template_cls.assert_called_once_with(
            commit_message="initial version",
            template_config=mock_template_config,
        )
        mock_version_cls.assert_called_once_with(mock_template)
        mock_request_cls.assert_called_once_with(
            name="my-evaluator",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            type="TEMPLATE",
            description=None,
            version=mock_version_wrap,
        )
        mock_api.create_evaluator.assert_called_once_with(
            create_evaluator_request=mock_body
        )

    def test_create_template_with_description(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_evaluator() should forward description to CreateEvaluatorRequest."""
        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
            patch(
                "arize._generated.api_client.CreateEvaluatorRequest"
            ) as mock_request_cls,
        ):
            mock_request_cls.return_value = Mock()

            evaluators_client.create_template_evaluator(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=Mock(spec=TemplateConfigInput),
                description="An evaluator for relevance",
            )

        _, kwargs = mock_request_cls.call_args
        assert kwargs["description"] == "An evaluator for relevance"

    def test_create_template_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_evaluator() should propagate the return value from evaluators_create."""
        expected = Mock()
        mock_api.create_evaluator.return_value = expected

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
            patch("arize._generated.api_client.CreateEvaluatorRequest"),
        ):
            result = evaluators_client.create_template_evaluator(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=Mock(spec=TemplateConfigInput),
            )

        assert result is expected

    def test_create_template_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create_template_evaluator() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
            patch("arize._generated.api_client.CreateEvaluatorRequest"),
        ):
            evaluators_client.create_template_evaluator(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=Mock(spec=TemplateConfigInput),
            )

        assert any(
            "BETA" in record.message
            and "evaluators.create_template" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientCreateCode:
    """Tests for EvaluatorsClient.create_code_evaluator()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            EvaluatorWithVersion,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_create_code_builds_code_request(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_code_evaluator() should build a code-type create request."""
        mock_code_config = Mock(spec=CodeConfigRequest)

        with (
            patch(
                "arize._generated.api_client.CreateCodeEvaluatorVersionRequest"
            ) as mock_code_cls,
            patch(
                "arize._generated.api_client.CreateEvaluatorVersionRequest"
            ) as mock_version_cls,
            patch(
                "arize._generated.api_client.CreateEvaluatorRequest"
            ) as mock_request_cls,
        ):
            mock_code = Mock()
            mock_code_cls.return_value = mock_code
            mock_version_wrap = Mock()
            mock_version_cls.return_value = mock_version_wrap
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            evaluators_client.create_code_evaluator(
                name="code-eval",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                code_config=mock_code_config,
            )

        mock_code_cls.assert_called_once_with(
            commit_message="initial",
            code_config=mock_code_config,
        )
        mock_version_cls.assert_called_once_with(mock_code)
        mock_request_cls.assert_called_once_with(
            name="code-eval",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            type="CODE",
            description=None,
            version=mock_version_wrap,
        )

    def test_create_code_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_code_evaluator() should propagate the return value."""
        expected = Mock()
        mock_api.create_evaluator.return_value = expected

        with (
            patch(
                "arize._generated.api_client.CreateCodeEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
            patch("arize._generated.api_client.CreateEvaluatorRequest"),
        ):
            result = evaluators_client.create_code_evaluator(
                name="code-eval",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                code_config=Mock(spec=CodeConfigRequest),
            )

        assert result is expected

    def test_create_code_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create_code_evaluator() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch(
                "arize._generated.api_client.CreateCodeEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
            patch("arize._generated.api_client.CreateEvaluatorRequest"),
        ):
            evaluators_client.create_code_evaluator(
                name="code-eval",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                code_config=Mock(spec=CodeConfigRequest),
            )

        assert any(
            "BETA" in record.message
            and "evaluators.create_code" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientUpdate:
    """Tests for EvaluatorsClient.update()."""

    def test_update_with_name(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should set only a provided name in its request body."""
        evaluators_client.update(evaluator=_EVALUATOR_ID, name="new-name")

        body = mock_api.update_evaluator.call_args.kwargs[
            "update_evaluator_request"
        ]
        assert isinstance(body, UpdateEvaluatorRequest)
        assert body.model_fields_set == {"name"}
        assert body.to_dict() == {"name": "new-name"}
        mock_api.update_evaluator.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            update_evaluator_request=body,
        )

    def test_update_with_description(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should set only a provided description in its request body."""
        evaluators_client.update(
            evaluator=_EVALUATOR_ID, description="Updated description"
        )

        body = mock_api.update_evaluator.call_args.kwargs[
            "update_evaluator_request"
        ]
        assert body.model_fields_set == {"description"}
        assert body.to_dict() == {"description": "Updated description"}

    def test_update_with_both_fields(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should set both concrete metadata values in its request body."""
        evaluators_client.update(
            evaluator=_EVALUATOR_ID,
            name="new-name",
            description="new description",
        )

        body = mock_api.update_evaluator.call_args.kwargs[
            "update_evaluator_request"
        ]
        assert body.model_fields_set == {"name", "description"}
        assert body.to_dict() == {
            "name": "new-name",
            "description": "new description",
        }

    def test_update_omits_unprovided_fields(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should leave unprovided fields absent from the request."""
        evaluators_client.update(evaluator=_EVALUATOR_ID)

        body = mock_api.update_evaluator.call_args.kwargs[
            "update_evaluator_request"
        ]
        assert body.model_fields_set == set()
        assert body.to_dict() == {}

    def test_update_omits_none_name(
        self,
        evaluators_client: EvaluatorsClient,
        mock_api: Mock,
    ) -> None:
        """update() should leave a ``None`` name absent from its request body."""
        evaluators_client.update(evaluator=_EVALUATOR_ID, name=None)

        body = mock_api.update_evaluator.call_args.kwargs[
            "update_evaluator_request"
        ]
        assert body.model_fields_set == set()
        assert body.to_dict() == {}

    def test_update_includes_explicit_none_to_clear_description(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should send an explicit ``None`` to clear the description."""
        evaluators_client.update(evaluator=_EVALUATOR_ID, description=None)

        body = mock_api.update_evaluator.call_args.kwargs[
            "update_evaluator_request"
        ]
        assert body.model_fields_set == {"description"}
        assert body.to_dict() == {"description": None}

    def test_update_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from evaluators_update."""
        expected = Mock()
        mock_api.update_evaluator.return_value = expected

        result = evaluators_client.update(evaluator=_EVALUATOR_ID, name="x")

        assert result is expected

    def test_update_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to update() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.update(evaluator=_EVALUATOR_ID, name="x")

        assert any(
            "BETA" in record.message and "evaluators.update" in record.message
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

        mock_api.delete_evaluator.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID
        )

    def test_delete_returns_none(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """delete() should always return None (204 No Content) regardless of API return."""
        mock_api.delete_evaluator.return_value = "unexpected"

        result = evaluators_client.delete(evaluator=_EVALUATOR_ID)

        assert result is None

    def test_delete_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to delete() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.delete(evaluator=_EVALUATOR_ID)

        assert any(
            "BETA" in record.message and "evaluators.delete" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientListVersions:
    """Tests for EvaluatorsClient.list_versions()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            ListEvaluatorVersionsResponse,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_list_versions_calls_api_with_all_params(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list_versions() should forward all parameters to evaluator_versions_list."""
        evaluators_client.list_versions(
            evaluator=_EVALUATOR_ID,
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.list_evaluator_versions.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_versions_defaults(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list_versions() should default limit to 50 and cursor to None."""
        evaluators_client.list_versions(evaluator=_EVALUATOR_ID)

        mock_api.list_evaluator_versions.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            limit=50,
            cursor=None,
        )

    def test_list_versions_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """list_versions() should propagate the return value."""
        expected = Mock()
        mock_api.list_evaluator_versions.return_value = expected

        result = evaluators_client.list_versions(evaluator=_EVALUATOR_ID)

        assert result is expected

    def test_list_versions_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.list_versions(evaluator=_EVALUATOR_ID)

        assert any(
            "BETA" in record.message
            and "evaluators.list_versions" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientDeleteVersions:
    """Tests for EvaluatorsClient.delete_versions()."""

    def test_delete_versions_builds_request_and_forwards(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """delete_versions() should build the request body and forward it."""
        result = evaluators_client.delete_versions(
            evaluator=_EVALUATOR_ID,
            version_ids=["v1", "v2"],
        )

        mock_api.delete_evaluator_versions.assert_called_once()
        call = mock_api.delete_evaluator_versions.call_args
        assert call.kwargs["evaluator_id"] == _EVALUATOR_ID
        body = call.kwargs["delete_evaluator_versions_request"]
        assert body.version_ids == ["v1", "v2"]
        assert result is mock_api.delete_evaluator_versions.return_value

    def test_delete_versions_returns_partial_result_unchanged(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """delete_versions() should return partial delete responses unchanged."""
        from arize._generated.api_client.models.delete_evaluator_versions_response import (
            DeleteEvaluatorVersionsResponse,
        )

        expected = DeleteEvaluatorVersionsResponse(
            completed=True,
            deleted_version_ids=["v1"],
            not_deleted_version_ids=["v2"],
        )
        mock_api.delete_evaluator_versions.return_value = expected

        result = evaluators_client.delete_versions(
            evaluator=_EVALUATOR_ID,
            version_ids=["v1", "v2"],
        )

        assert result is expected
        assert result.completed is True
        assert result.deleted_version_ids == ["v1"]
        assert result.not_deleted_version_ids == ["v2"]

    def test_delete_versions_rejects_empty_list(
        self, evaluators_client: EvaluatorsClient
    ) -> None:
        """delete_versions() should reject an empty version list (min_length=1)."""
        with pytest.raises(Exception):
            evaluators_client.delete_versions(
                evaluator=_EVALUATOR_ID,
                version_ids=[],
            )

    def test_delete_versions_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to delete_versions() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.delete_versions(
            evaluator=_EVALUATOR_ID,
            version_ids=["v1"],
        )

        assert any(
            "BETA" in record.message
            and "evaluators.delete_versions" in record.message
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

        mock_api.get_evaluator_version.assert_called_once_with(
            version_id="ver-456"
        )

    def test_get_version_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get_version() should propagate the unwrapped return value."""
        expected = Mock()
        mock_api.get_evaluator_version.return_value.actual_instance = expected

        result = evaluators_client.get_version(version_id="ver-456")

        assert result is expected

    def test_get_version_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        evaluators_client.get_version(version_id="ver-456")

        assert any(
            "BETA" in record.message
            and "evaluators.get_version" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientCreateTemplateVersion:
    """Tests for EvaluatorsClient.create_template_version()."""

    def test_create_template_version_builds_template_version(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_version() should build a template version."""
        mock_template_config = Mock(spec=TemplateConfigInput)

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluatorVersionRequest"
            ) as mock_template_cls,
            patch(
                "arize._generated.api_client.CreateEvaluatorVersionRequest"
            ) as mock_version_cls,
        ):
            mock_template = Mock()
            mock_template_cls.return_value = mock_template
            mock_body = Mock()
            mock_version_cls.return_value = mock_body

            evaluators_client.create_template_version(
                evaluator=_EVALUATOR_ID,
                commit_message="fix prompt wording",
                template_config=mock_template_config,
            )

        mock_template_cls.assert_called_once_with(
            commit_message="fix prompt wording",
            template_config=mock_template_config,
        )
        mock_version_cls.assert_called_once_with(mock_template)
        mock_api.create_evaluator_version.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            create_evaluator_version_request=mock_body,
        )

    def test_create_template_version_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_version() should propagate the unwrapped return value."""
        expected = Mock()
        mock_api.create_evaluator_version.return_value.actual_instance = (
            expected
        )

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
        ):
            result = evaluators_client.create_template_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                template_config=Mock(spec=TemplateConfigInput),
            )

        assert result is expected

    def test_create_template_version_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
        ):
            evaluators_client.create_template_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                template_config=Mock(spec=TemplateConfigInput),
            )

        assert any(
            "BETA" in record.message
            and "evaluators.create_template_version" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestEvaluatorsClientCreateCodeVersion:
    """Tests for EvaluatorsClient.create_code_version()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            EvaluatorVersionCode,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_create_code_version_builds_code_version(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_code_version() should build a code version."""
        mock_code_config = Mock(spec=CodeConfigRequest)

        with (
            patch(
                "arize._generated.api_client.CreateCodeEvaluatorVersionRequest"
            ) as mock_code_cls,
            patch(
                "arize._generated.api_client.CreateEvaluatorVersionRequest"
            ) as mock_version_cls,
        ):
            mock_code = Mock()
            mock_code_cls.return_value = mock_code
            mock_body = Mock()
            mock_version_cls.return_value = mock_body

            evaluators_client.create_code_version(
                evaluator=_EVALUATOR_ID,
                commit_message="tune keywords",
                code_config=mock_code_config,
            )

        mock_code_cls.assert_called_once_with(
            commit_message="tune keywords",
            code_config=mock_code_config,
        )
        mock_version_cls.assert_called_once_with(mock_code)
        mock_api.create_evaluator_version.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            create_evaluator_version_request=mock_body,
        )

    def test_create_code_version_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_code_version() should propagate the unwrapped return value."""
        expected = Mock()
        mock_api.create_evaluator_version.return_value.actual_instance = (
            expected
        )

        with (
            patch(
                "arize._generated.api_client.CreateCodeEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
        ):
            result = evaluators_client.create_code_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                code_config=Mock(spec=CodeConfigRequest),
            )

        assert result is expected

    def test_create_code_version_emits_beta_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch(
                "arize._generated.api_client.CreateCodeEvaluatorVersionRequest"
            ),
            patch("arize._generated.api_client.CreateEvaluatorVersionRequest"),
        ):
            evaluators_client.create_code_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                code_config=Mock(spec=CodeConfigRequest),
            )

        assert any(
            "BETA" in record.message
            and "evaluators.create_code_version" in record.message
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# Real-instance round-trip tests
# ---------------------------------------------------------------------------
# These tests use the actual generated Pydantic classes (no mocking of
# CreateTemplateEvaluatorVersionRequest, CreateCodeEvaluatorVersionRequest, or
# CreateEvaluatorVersionRequest) to verify that the oneOf Pydantic validation is
# exercised.  A refactor that passes type checks but produces an invalid
# payload would fail here.
# ---------------------------------------------------------------------------


def _make_real_llm_config() -> object:
    """Build a real gen.EvaluatorLlmConfigRequest instance."""
    from arize._generated import api_client as gen

    return gen.EvaluatorLlmConfigRequest(
        ai_integration_id="TGxtSW50ZWdyYXRpb246MQ==",
        model_name="gpt-4o",
        invocation_parameters=gen.InvocationParamsRequest(),
        provider_parameters=gen.ProviderParamsRequest(),
    )


def _make_real_template_config() -> object:
    """Build a real gen.TemplateConfigInput instance."""
    from arize._generated import api_client as gen

    return gen.TemplateConfigInput(
        name="relevance",
        template="Is {{output}} relevant?",
        include_explanations=True,
        use_function_calling=False,
        classification_choices={"relevant": 1, "irrelevant": 0},
        llm_config=_make_real_llm_config(),
    )


def _make_real_code_config() -> object:
    """Build a real gen.CodeConfigRequest(ManagedCodeConfigRequest) instance."""
    from arize._generated import api_client as gen

    managed = gen.ManagedCodeConfigRequest(
        type="MANAGED",
        name="json_parseable",
        managed_evaluator=gen.ManagedCodeEvaluator("JSON_PARSEABLE"),
        variables=["output"],
    )
    return gen.CodeConfigRequest(managed)


@pytest.mark.unit
class TestEvaluatorsClientCreateTemplateRealInstance:
    """Round-trip test using a real TemplateConfigInput for create_template_evaluator()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            EvaluatorWithVersion,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_create_template_real_instance_builds_valid_payload(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_evaluator() with a real TemplateConfigInput produces a valid CreateEvaluatorVersionRequest."""
        evaluators_client.create_template_evaluator(
            name="my-evaluator",
            space="U3BhY2U6OTA1MDoxSmtS",
            commit_message="initial",
            template_config=_make_real_template_config(),
        )

        mock_api.create_evaluator.assert_called_once()
        _, kwargs = mock_api.create_evaluator.call_args
        body = kwargs["create_evaluator_request"]
        assert body.type == "TEMPLATE"
        assert body.version is not None
        # The point of this PR: classification_choices is required and must
        # reach the request payload. Pin it so a dropped/renamed field fails.
        template_config = body.version.actual_instance.template_config
        assert template_config.classification_choices == {
            "relevant": 1,
            "irrelevant": 0,
        }


@pytest.mark.unit
class TestEvaluatorsClientCreateCodeRealInstance:
    """Round-trip test using a real CodeConfig for create_code_evaluator()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            EvaluatorWithVersion,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_create_code_real_instance_builds_valid_payload(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_code_evaluator() with a real CodeConfig produces a valid CreateEvaluatorVersionRequest."""
        evaluators_client.create_code_evaluator(
            name="code-eval",
            space="U3BhY2U6OTA1MDoxSmtS",
            commit_message="initial",
            code_config=_make_real_code_config(),
        )

        mock_api.create_evaluator.assert_called_once()
        _, kwargs = mock_api.create_evaluator.call_args
        body = kwargs["create_evaluator_request"]
        assert body.type == "CODE"
        assert body.version is not None


@pytest.mark.unit
class TestEvaluatorsClientCreateTemplateVersionRealInstance:
    """Round-trip test using a real TemplateConfigInput for create_template_version()."""

    def test_create_template_version_real_instance_builds_valid_payload(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_version() with a real TemplateConfigInput produces a valid payload."""
        evaluators_client.create_template_version(
            evaluator=_EVALUATOR_ID,
            commit_message="fix wording",
            template_config=_make_real_template_config(),
        )

        mock_api.create_evaluator_version.assert_called_once()
        _, kwargs = mock_api.create_evaluator_version.call_args
        body = kwargs["create_evaluator_version_request"]
        assert body is not None
        # classification_choices must reach the version request payload.
        template_config = body.actual_instance.template_config
        assert template_config.classification_choices == {
            "relevant": 1,
            "irrelevant": 0,
        }


@pytest.mark.unit
class TestEvaluatorsClientGetVersionCode:
    """Tests for EvaluatorsClient.get_version() code evaluator path."""

    def test_get_version_model_validates_code_evaluator(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """get_version() should model_validate when the version is a code evaluator."""
        from arize._generated.api_client.models.evaluator_version_code import (
            EvaluatorVersionCode as GenEvaluatorVersionCode,
        )

        gen_ver = Mock(spec=GenEvaluatorVersionCode)
        mock_api.get_evaluator_version.return_value = Mock(
            actual_instance=gen_ver
        )
        expected = Mock()
        with patch.object(
            EvaluatorVersionCode,
            "model_validate",
            return_value=expected,
        ):
            result = evaluators_client.get_version(version_id="ver-123")

        assert result is expected


@pytest.mark.unit
class TestCoerceCodeConfig:
    """Tests for EvaluatorsClient._coerce_code_config() branches."""

    def test_coerce_custom_code_config_request(self) -> None:
        """_coerce_code_config wraps CustomCodeConfigRequest in CodeConfigRequest."""
        from arize._generated import api_client as gen

        item = Mock(spec=gen.CustomCodeConfigRequest)
        result = EvaluatorsClient._coerce_code_config(item)
        assert isinstance(result, gen.CodeConfigRequest)

    def test_coerce_managed_code_config_request(self) -> None:
        """_coerce_code_config wraps ManagedCodeConfigRequest in CodeConfigRequest."""
        from arize._generated import api_client as gen

        item = Mock(spec=gen.ManagedCodeConfigRequest)
        result = EvaluatorsClient._coerce_code_config(item)
        assert isinstance(result, gen.CodeConfigRequest)

    def test_coerce_unknown_type_raises(self) -> None:
        """_coerce_code_config raises ValidationError for unrecognized types."""
        from pydantic import ValidationError

        with pytest.raises((TypeError, ValidationError)):
            EvaluatorsClient._coerce_code_config("not-a-config")  # type: ignore[arg-type]


@pytest.mark.unit
class TestEvaluatorsClientCreateCodeVersionRealInstance:
    """Round-trip test using a real CodeConfig for create_code_version()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            EvaluatorVersionCode,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_create_code_version_real_instance_builds_valid_payload(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_code_version() with a real CodeConfig produces a valid payload."""
        evaluators_client.create_code_version(
            evaluator=_EVALUATOR_ID,
            commit_message="update code",
            code_config=_make_real_code_config(),
        )

        mock_api.create_evaluator_version.assert_called_once()
        _, kwargs = mock_api.create_evaluator_version.call_args
        assert kwargs["create_evaluator_version_request"] is not None
