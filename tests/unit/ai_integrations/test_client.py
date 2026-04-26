"""Unit tests for src/arize/ai_integrations/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.ai_integrations.client import AiIntegrationsClient

# Base64 ID that decodes to "Integration:123" — passes _is_resource_id()
_INTEGRATION_ID = "SW50ZWdyYXRpb246MTIz"


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock AIIntegrationsApi instance."""
    return Mock()


@pytest.fixture
def ai_integrations_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> AiIntegrationsClient:
    """Provide an AiIntegrationsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.AIIntegrationsApi", return_value=mock_api
    ):
        return AiIntegrationsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestAiIntegrationsClientInit:
    """Tests for AiIntegrationsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.AIIntegrationsApi",
            return_value=mock_api,
        ):
            client = AiIntegrationsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to AIIntegrationsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.AIIntegrationsApi"
        ) as mock_api_cls:
            AiIntegrationsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestAiIntegrationsClientList:
    """Tests for AiIntegrationsClient.list()."""

    def test_list_calls_api_with_space_id(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a base64 resource ID space value to space_id."""
        ai_integrations_client.list(
            space="U3BhY2U6OTA1MDoxSmtS",
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.ai_integrations_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name=None,
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_calls_api_with_space_name(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        ai_integrations_client.list(
            space="my-space",
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.ai_integrations_list.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name=None,
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """list() should default space/name/cursor to None and limit to 100."""
        ai_integrations_client.list()

        mock_api.ai_integrations_list.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from ai_integrations_list."""
        expected = Mock()
        mock_api.ai_integrations_list.return_value = expected

        result = ai_integrations_client.list()

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self,
        ai_integrations_client: AiIntegrationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        ai_integrations_client.list()

        assert any(
            "ALPHA" in record.message
            and "ai_integrations.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestAiIntegrationsClientGet:
    """Tests for AiIntegrationsClient.get()."""

    def test_get_calls_api_with_integration_id(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """get() should resolve integration and pass integration_id to ai_integrations_get."""
        ai_integrations_client.get(integration=_INTEGRATION_ID)

        mock_api.ai_integrations_get.assert_called_once_with(
            integration_id=_INTEGRATION_ID
        )

    def test_get_returns_api_response(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from ai_integrations_get."""
        expected = Mock()
        mock_api.ai_integrations_get.return_value = expected

        result = ai_integrations_client.get(integration=_INTEGRATION_ID)

        assert result is expected


@pytest.mark.unit
class TestAiIntegrationsClientCreate:
    """Tests for AiIntegrationsClient.create()."""

    def test_create_builds_request_and_calls_api(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """create() should build AiIntegrationsCreateRequest and pass it to ai_integrations_create."""
        with patch(
            "arize._generated.api_client.AiIntegrationsCreateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            ai_integrations_client.create(
                name="Production OpenAI",
                provider="openAI",
                api_key="sk-abc123",
                model_names=["gpt-4", "gpt-4o"],
                enable_default_models=True,
            )

        mock_request_cls.assert_called_once_with(
            name="Production OpenAI",
            provider="openAI",
            api_key="sk-abc123",
            base_url=None,
            model_names=["gpt-4", "gpt-4o"],
            headers=None,
            enable_default_models=True,
            function_calling_enabled=None,
            auth_type=None,
            provider_metadata=None,
            scopings=None,
        )
        mock_api.ai_integrations_create.assert_called_once_with(
            ai_integrations_create_request=mock_body
        )

    def test_create_returns_api_response(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from ai_integrations_create."""
        expected = Mock()
        mock_api.ai_integrations_create.return_value = expected

        with patch("arize._generated.api_client.AiIntegrationsCreateRequest"):
            result = ai_integrations_client.create(
                name="My Integration",
                provider="openAI",
            )

        assert result is expected

    def test_create_wraps_typed_provider_metadata(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """create() should wrap typed provider metadata in oneOf wrapper."""
        from arize._generated.api_client.models.aws_provider_metadata import (
            AwsProviderMetadata,
        )

        aws_meta = AwsProviderMetadata(
            kind="aws",
            role_arn="arn:aws:iam::role/x",
            external_id=None,
        )
        mock_wrapped = Mock()
        with (
            patch(
                "arize._generated.api_client.AiIntegrationsCreateRequest"
            ) as mock_request_cls,
            patch(
                "arize._generated.api_client"
                ".AiIntegrationsCreateRequestProviderMetadata"
            ) as mock_meta_cls,
        ):
            mock_meta_cls.return_value = mock_wrapped
            mock_request_cls.return_value = Mock()

            ai_integrations_client.create(
                name="AWS Bedrock",
                provider="awsBedrock",
                provider_metadata=aws_meta,
            )

        mock_meta_cls.assert_called_once_with(actual_instance=aws_meta)
        call_kwargs = mock_request_cls.call_args[1]
        assert call_kwargs["provider_metadata"] is mock_wrapped

    def test_create_emits_alpha_prerelease_warning(
        self,
        ai_integrations_client: AiIntegrationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.AiIntegrationsCreateRequest"):
            ai_integrations_client.create(name="test", provider="openAI")

        assert any(
            "ALPHA" in record.message
            and "ai_integrations.create" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestAiIntegrationsClientUpdate:
    """Tests for AiIntegrationsClient.update()."""

    def test_update_only_sends_provided_fields(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """update() should only pass caller-provided fields to AiIntegrationsUpdateRequest."""
        with patch(
            "arize._generated.api_client.AiIntegrationsUpdateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            ai_integrations_client.update(
                integration=_INTEGRATION_ID,
                name="Updated Integration",
                model_names=["gpt-4"],
            )

        # Only name and model_names should be passed — no api_key, base_url, etc.
        mock_request_cls.assert_called_once_with(
            name="Updated Integration",
            model_names=["gpt-4"],
        )
        mock_api.ai_integrations_update.assert_called_once_with(
            integration_id=_INTEGRATION_ID,
            ai_integrations_update_request=mock_body,
        )

    def test_update_explicit_none_is_forwarded(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """update() should forward explicit None so the server clears the field."""
        with patch(
            "arize._generated.api_client.AiIntegrationsUpdateRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            ai_integrations_client.update(
                integration=_INTEGRATION_ID,
                name="Keep Name",
                api_key=None,
            )

        mock_request_cls.assert_called_once_with(
            name="Keep Name",
            api_key=None,
        )

    def test_update_no_fields_sends_empty_request(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """update() with no optional fields should send an empty request body."""
        with patch(
            "arize._generated.api_client.AiIntegrationsUpdateRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            ai_integrations_client.update(integration=_INTEGRATION_ID)

        mock_request_cls.assert_called_once_with()

    def test_update_accepts_false_boolean_field(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """update() should forward falsy (but non-None) boolean values."""
        with patch(
            "arize._generated.api_client.AiIntegrationsUpdateRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            ai_integrations_client.update(
                integration=_INTEGRATION_ID,
                enable_default_models=False,
            )

        mock_request_cls.assert_called_once_with(
            enable_default_models=False,
        )

    def test_update_wraps_typed_provider_metadata(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """update() should wrap typed provider metadata in oneOf wrapper."""
        from arize._generated.api_client.models.aws_provider_metadata import (
            AwsProviderMetadata,
        )

        aws_meta = AwsProviderMetadata(
            kind="aws",
            role_arn="arn:aws:iam::role/x",
            external_id=None,
        )
        mock_wrapped = Mock()
        with (
            patch(
                "arize._generated.api_client.AiIntegrationsUpdateRequest"
            ) as mock_request_cls,
            patch(
                "arize._generated.api_client"
                ".AiIntegrationsUpdateRequestProviderMetadata"
            ) as mock_meta_cls,
        ):
            mock_meta_cls.return_value = mock_wrapped
            mock_request_cls.return_value = Mock()

            ai_integrations_client.update(
                integration=_INTEGRATION_ID,
                provider_metadata=aws_meta,
            )

        mock_meta_cls.assert_called_once_with(actual_instance=aws_meta)
        call_kwargs = mock_request_cls.call_args[1]
        assert call_kwargs["provider_metadata"] is mock_wrapped

    def test_update_explicit_none_provider_metadata_is_forwarded(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """update() should forward explicit None for provider_metadata without wrapping."""
        with patch(
            "arize._generated.api_client.AiIntegrationsUpdateRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            ai_integrations_client.update(
                integration=_INTEGRATION_ID,
                provider_metadata=None,
            )

        mock_request_cls.assert_called_once_with(
            provider_metadata=None,
        )

    def test_update_returns_api_response(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from ai_integrations_update."""
        expected = Mock()
        mock_api.ai_integrations_update.return_value = expected

        with patch("arize._generated.api_client.AiIntegrationsUpdateRequest"):
            result = ai_integrations_client.update(
                integration=_INTEGRATION_ID,
                name="Updated Integration",
            )

        assert result is expected

    def test_update_emits_alpha_prerelease_warning(
        self,
        ai_integrations_client: AiIntegrationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.AiIntegrationsUpdateRequest"):
            ai_integrations_client.update(
                integration=_INTEGRATION_ID, name="Updated"
            )

        assert any(
            "ALPHA" in record.message
            and "ai_integrations.update" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestAiIntegrationsClientDelete:
    """Tests for AiIntegrationsClient.delete()."""

    def test_delete_calls_api_with_integration_id(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """delete() should resolve integration and pass integration_id to ai_integrations_delete."""
        ai_integrations_client.delete(integration=_INTEGRATION_ID)

        mock_api.ai_integrations_delete.assert_called_once_with(
            integration_id=_INTEGRATION_ID
        )

    def test_delete_returns_none(
        self, ai_integrations_client: AiIntegrationsClient, mock_api: Mock
    ) -> None:
        """delete() should return None."""
        mock_api.ai_integrations_delete.return_value = None

        result = ai_integrations_client.delete(integration=_INTEGRATION_ID)

        assert result is None

    def test_delete_emits_alpha_prerelease_warning(
        self,
        ai_integrations_client: AiIntegrationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        ai_integrations_client.delete(integration=_INTEGRATION_ID)

        assert any(
            "ALPHA" in record.message
            and "ai_integrations.delete" in record.message
            for record in caplog.records
        )
