"""Unit tests for src/arize/evaluators/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.evaluators.client import EvaluatorsClient
from arize.evaluators.types import (
    CodeConfig,
    EvaluatorVersionCode,
    EvaluatorVersionsList200Response,
    EvaluatorWithVersion,
)

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
        mock_template_config = Mock()

        with (
            patch(
                "arize._generated.api_client.EvaluatorVersionTemplateCreate"
            ) as mock_template_cls,
            patch(
                "arize._generated.api_client.EvaluatorVersionCreate"
            ) as mock_version_cls,
            patch(
                "arize._generated.api_client.EvaluatorsCreateRequest"
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
            type="template",
            description=None,
            version=mock_version_wrap,
        )
        mock_api.evaluators_create.assert_called_once_with(
            evaluators_create_request=mock_body
        )

    def test_create_template_with_description(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_evaluator() should forward description to EvaluatorsCreateRequest."""
        with (
            patch("arize._generated.api_client.EvaluatorVersionTemplateCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
            patch(
                "arize._generated.api_client.EvaluatorsCreateRequest"
            ) as mock_request_cls,
        ):
            mock_request_cls.return_value = Mock()

            evaluators_client.create_template_evaluator(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=Mock(),
                description="An evaluator for relevance",
            )

        _, kwargs = mock_request_cls.call_args
        assert kwargs["description"] == "An evaluator for relevance"

    def test_create_template_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_evaluator() should propagate the return value from evaluators_create."""
        expected = Mock()
        mock_api.evaluators_create.return_value = expected

        with (
            patch("arize._generated.api_client.EvaluatorVersionTemplateCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
            patch("arize._generated.api_client.EvaluatorsCreateRequest"),
        ):
            result = evaluators_client.create_template_evaluator(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=Mock(),
            )

        assert result is expected

    def test_create_template_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create_template_evaluator() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.EvaluatorVersionTemplateCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
            patch("arize._generated.api_client.EvaluatorsCreateRequest"),
        ):
            evaluators_client.create_template_evaluator(
                name="my-evaluator",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                template_config=Mock(),
            )

        assert any(
            "ALPHA" in record.message
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
        mock_code_config = Mock(spec=CodeConfig)

        with (
            patch(
                "arize._generated.api_client.EvaluatorVersionCodeCreate"
            ) as mock_code_cls,
            patch(
                "arize._generated.api_client.EvaluatorVersionCreate"
            ) as mock_version_cls,
            patch(
                "arize._generated.api_client.EvaluatorsCreateRequest"
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
            type="code",
            description=None,
            version=mock_version_wrap,
        )

    def test_create_code_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_code_evaluator() should propagate the return value."""
        expected = Mock()
        mock_api.evaluators_create.return_value = expected

        with (
            patch("arize._generated.api_client.EvaluatorVersionCodeCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
            patch("arize._generated.api_client.EvaluatorsCreateRequest"),
        ):
            result = evaluators_client.create_code_evaluator(
                name="code-eval",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                code_config=Mock(spec=CodeConfig),
            )

        assert result is expected

    def test_create_code_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create_code_evaluator() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.EvaluatorVersionCodeCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
            patch("arize._generated.api_client.EvaluatorsCreateRequest"),
        ):
            evaluators_client.create_code_evaluator(
                name="code-eval",
                space="U3BhY2U6OTA1MDoxSmtS",
                commit_message="initial",
                code_config=Mock(spec=CodeConfig),
            )

        assert any(
            "ALPHA" in record.message
            and "evaluators.create_code" in record.message
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

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            EvaluatorVersionsList200Response,
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
        """get_version() should propagate the unwrapped return value."""
        expected = Mock()
        mock_api.evaluator_versions_get.return_value.actual_instance = expected

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
class TestEvaluatorsClientCreateTemplateVersion:
    """Tests for EvaluatorsClient.create_template_version()."""

    def test_create_template_version_builds_template_version(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_version() should build a template version."""
        mock_template_config = Mock()

        with (
            patch(
                "arize._generated.api_client.EvaluatorVersionTemplateCreate"
            ) as mock_template_cls,
            patch(
                "arize._generated.api_client.EvaluatorVersionCreate"
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
        mock_api.evaluator_versions_create.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            evaluator_version_create=mock_body,
        )

    def test_create_template_version_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_version() should propagate the unwrapped return value."""
        expected = Mock()
        mock_api.evaluator_versions_create.return_value.actual_instance = (
            expected
        )

        with (
            patch("arize._generated.api_client.EvaluatorVersionTemplateCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
        ):
            result = evaluators_client.create_template_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                template_config=Mock(),
            )

        assert result is expected

    def test_create_template_version_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.EvaluatorVersionTemplateCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
        ):
            evaluators_client.create_template_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                template_config=Mock(),
            )

        assert any(
            "ALPHA" in record.message
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
        mock_code_config = Mock(spec=CodeConfig)

        with (
            patch(
                "arize._generated.api_client.EvaluatorVersionCodeCreate"
            ) as mock_code_cls,
            patch(
                "arize._generated.api_client.EvaluatorVersionCreate"
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
        mock_api.evaluator_versions_create.assert_called_once_with(
            evaluator_id=_EVALUATOR_ID,
            evaluator_version_create=mock_body,
        )

    def test_create_code_version_returns_api_response(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_code_version() should propagate the unwrapped return value."""
        expected = Mock()
        mock_api.evaluator_versions_create.return_value.actual_instance = (
            expected
        )

        with (
            patch("arize._generated.api_client.EvaluatorVersionCodeCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
        ):
            result = evaluators_client.create_code_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                code_config=Mock(spec=CodeConfig),
            )

        assert result is expected

    def test_create_code_version_emits_alpha_prerelease_warning(
        self,
        evaluators_client: EvaluatorsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.EvaluatorVersionCodeCreate"),
            patch("arize._generated.api_client.EvaluatorVersionCreate"),
        ):
            evaluators_client.create_code_version(
                evaluator=_EVALUATOR_ID,
                commit_message="v2",
                code_config=Mock(spec=CodeConfig),
            )

        assert any(
            "ALPHA" in record.message
            and "evaluators.create_code_version" in record.message
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# Real-instance round-trip tests
# ---------------------------------------------------------------------------
# These tests use the actual generated Pydantic classes (no mocking of
# EvaluatorVersionTemplateCreate, EvaluatorVersionCodeCreate, or
# EvaluatorVersionCreate) to verify that the oneOf Pydantic validation is
# exercised.  A refactor that passes type checks but produces an invalid
# payload would fail here.
# ---------------------------------------------------------------------------


def _make_real_llm_config() -> object:
    """Build a real gen.EvaluatorLlmConfig instance."""
    from arize._generated import api_client as gen

    return gen.EvaluatorLlmConfig(
        ai_integration_id="TGxtSW50ZWdyYXRpb246MQ==",
        model_name="gpt-4o",
        invocation_parameters=gen.InvocationParams(),
        provider_parameters=gen.ProviderParams(),
    )


def _make_real_template_config() -> object:
    """Build a real gen.TemplateConfig instance."""
    from arize._generated import api_client as gen

    return gen.TemplateConfig(
        name="relevance",
        template="Is {{output}} relevant?",
        include_explanations=True,
        use_function_calling_if_available=False,
        llm_config=_make_real_llm_config(),
    )


def _make_real_code_config() -> object:
    """Build a real gen.CodeConfig(ManagedCodeConfig) instance."""
    from arize._generated import api_client as gen

    managed = gen.ManagedCodeConfig(
        type="managed",
        name="json_parseable",
        managed_evaluator=gen.ManagedCodeEvaluator("JSONParseable"),
        variables=["output"],
    )
    return gen.CodeConfig(managed)


@pytest.mark.unit
class TestEvaluatorsClientCreateTemplateRealInstance:
    """Round-trip test using a real TemplateConfig for create_template_evaluator()."""

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
        """create_template_evaluator() with a real TemplateConfig produces a valid EvaluatorVersionCreate."""
        evaluators_client.create_template_evaluator(
            name="my-evaluator",
            space="U3BhY2U6OTA1MDoxSmtS",
            commit_message="initial",
            template_config=_make_real_template_config(),
        )

        mock_api.evaluators_create.assert_called_once()
        _, kwargs = mock_api.evaluators_create.call_args
        body = kwargs["evaluators_create_request"]
        assert body.type == "template"
        assert body.version is not None


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
        """create_code_evaluator() with a real CodeConfig produces a valid EvaluatorVersionCreate."""
        evaluators_client.create_code_evaluator(
            name="code-eval",
            space="U3BhY2U6OTA1MDoxSmtS",
            commit_message="initial",
            code_config=_make_real_code_config(),
        )

        mock_api.evaluators_create.assert_called_once()
        _, kwargs = mock_api.evaluators_create.call_args
        body = kwargs["evaluators_create_request"]
        assert body.type == "code"
        assert body.version is not None


@pytest.mark.unit
class TestEvaluatorsClientCreateTemplateVersionRealInstance:
    """Round-trip test using a real TemplateConfig for create_template_version()."""

    def test_create_template_version_real_instance_builds_valid_payload(
        self, evaluators_client: EvaluatorsClient, mock_api: Mock
    ) -> None:
        """create_template_version() with a real TemplateConfig produces a valid payload."""
        evaluators_client.create_template_version(
            evaluator=_EVALUATOR_ID,
            commit_message="fix wording",
            template_config=_make_real_template_config(),
        )

        mock_api.evaluator_versions_create.assert_called_once()
        _, kwargs = mock_api.evaluator_versions_create.call_args
        assert kwargs["evaluator_version_create"] is not None


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

        mock_api.evaluator_versions_create.assert_called_once()
        _, kwargs = mock_api.evaluator_versions_create.call_args
        assert kwargs["evaluator_version_create"] is not None
