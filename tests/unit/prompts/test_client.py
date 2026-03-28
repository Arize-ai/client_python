"""Unit tests for src/arize/prompts/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.prompts.client import PromptsClient

# Base64 ID that decodes to "Prompt:123" — passes _is_resource_id()
_PROMPT_ID = "UHJvbXB0OjEyMw=="


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock PromptsApi instance."""
    return Mock()


@pytest.fixture
def prompts_client(mock_sdk_config: Mock, mock_api: Mock) -> PromptsClient:
    """Provide a PromptsClient with mocked internals."""
    with patch("arize._generated.api_client.PromptsApi", return_value=mock_api):
        return PromptsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestPromptsClientInit:
    """Tests for PromptsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.PromptsApi", return_value=mock_api
        ):
            client = PromptsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_prompts_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to PromptsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.PromptsApi"
        ) as mock_prompts_api_cls:
            PromptsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_prompts_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestPromptsClientList:
    """Tests for PromptsClient.list()."""

    def test_list_with_space_id(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a base64 resource ID space value to space_id."""
        prompts_client.list(
            name="my-prompt",
            space="U3BhY2U6OTA1MDoxSmtS",
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.prompts_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name="my-prompt",
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_with_space_name(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        prompts_client.list(
            name="my-prompt",
            space="my-space",
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.prompts_list.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="my-prompt",
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """list() should default space/name/cursor to None and limit to 100."""
        prompts_client.list()

        mock_api.prompts_list.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from prompts_list."""
        expected = Mock()
        mock_api.prompts_list.return_value = expected

        result = prompts_client.list()

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self,
        prompts_client: PromptsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        prompts_client.list()

        assert any(
            "ALPHA" in record.message and "prompts.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestPromptsClientCreate:
    """Tests for PromptsClient.create()."""

    def test_create_builds_request_and_calls_api(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """create() should build PromptsCreateRequest and pass it to prompts_create."""
        mock_messages = [Mock()]
        mock_input_format = Mock()
        mock_provider = Mock()

        with (
            patch(
                "arize._generated.api_client.PromptVersionCreateRequest"
            ) as mock_version_cls,
            patch(
                "arize._generated.api_client.PromptsCreateRequest"
            ) as mock_request_cls,
        ):
            mock_version = Mock()
            mock_version_cls.return_value = mock_version
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            prompts_client.create(
                space="U3BhY2U6OTA1MDoxSmtS",
                name="my-prompt",
                description="a prompt",
                commit_message="initial version",
                input_variable_format=mock_input_format,
                provider=mock_provider,
                messages=mock_messages,
                model="gpt-4",
            )

        mock_version_cls.assert_called_once_with(
            commit_message="initial version",
            input_variable_format=mock_input_format,
            provider=mock_provider,
            model="gpt-4",
            messages=mock_messages,
            invocation_params=None,
            provider_params=None,
        )
        mock_request_cls.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            name="my-prompt",
            description="a prompt",
            version=mock_version,
        )
        mock_api.prompts_create.assert_called_once_with(
            prompts_create_request=mock_body
        )

    def test_create_returns_api_response(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from prompts_create."""
        expected = Mock()
        mock_api.prompts_create.return_value = expected

        with (
            patch("arize._generated.api_client.PromptVersionCreateRequest"),
            patch("arize._generated.api_client.PromptsCreateRequest"),
        ):
            result = prompts_client.create(
                space="U3BhY2U6OTA1MDoxSmtS",
                name="my-prompt",
                commit_message="v1",
                input_variable_format=Mock(),
                provider=Mock(),
                messages=[Mock()],
            )

        assert result is expected


@pytest.mark.unit
class TestPromptsClientGet:
    """Tests for PromptsClient.get()."""

    def test_get_calls_api_with_prompt_id(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """get() should resolve prompt and pass prompt_id to prompts_get."""
        prompts_client.get(prompt=_PROMPT_ID)

        mock_api.prompts_get.assert_called_once_with(
            prompt_id=_PROMPT_ID,
            version_id=None,
            label=None,
        )

    def test_get_with_version_id(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """get() should forward version_id to prompts_get."""
        prompts_client.get(prompt=_PROMPT_ID, version_id="ver-456")

        mock_api.prompts_get.assert_called_once_with(
            prompt_id=_PROMPT_ID,
            version_id="ver-456",
            label=None,
        )

    def test_get_with_label(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """get() should forward label to prompts_get."""
        prompts_client.get(prompt=_PROMPT_ID, label="production")

        mock_api.prompts_get.assert_called_once_with(
            prompt_id=_PROMPT_ID,
            version_id=None,
            label="production",
        )

    def test_get_returns_api_response(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from prompts_get."""
        expected = Mock()
        mock_api.prompts_get.return_value = expected

        result = prompts_client.get(prompt=_PROMPT_ID)

        assert result is expected


@pytest.mark.unit
class TestPromptsClientUpdate:
    """Tests for PromptsClient.update()."""

    def test_update_builds_request_and_calls_api(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """update() should build PromptsUpdateRequest and pass it to prompts_update."""
        with patch(
            "arize._generated.api_client.PromptsUpdateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            prompts_client.update(
                prompt=_PROMPT_ID,
                description="updated description",
            )

        mock_request_cls.assert_called_once_with(
            description="updated description"
        )
        mock_api.prompts_update.assert_called_once_with(
            prompt_id=_PROMPT_ID,
            prompts_update_request=mock_body,
        )

    def test_update_returns_api_response(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from prompts_update."""
        expected = Mock()
        mock_api.prompts_update.return_value = expected

        with patch("arize._generated.api_client.PromptsUpdateRequest"):
            result = prompts_client.update(
                prompt=_PROMPT_ID,
                description="updated",
            )

        assert result is expected


@pytest.mark.unit
class TestPromptsClientDelete:
    """Tests for PromptsClient.delete()."""

    def test_delete_calls_api_with_prompt_id(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """delete() should resolve prompt and pass prompt_id to prompts_delete."""
        prompts_client.delete(prompt=_PROMPT_ID)

        mock_api.prompts_delete.assert_called_once_with(prompt_id=_PROMPT_ID)

    def test_delete_returns_none(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """delete() should return None on success."""
        mock_api.prompts_delete.return_value = None

        result = prompts_client.delete(prompt=_PROMPT_ID)

        assert result is None


@pytest.mark.unit
class TestPromptsClientListVersions:
    """Tests for PromptsClient.list_versions()."""

    def test_list_versions_calls_api_with_all_params(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """list_versions() should pass prompt_id, limit, and cursor to prompt_versions_list."""
        prompts_client.list_versions(
            prompt=_PROMPT_ID,
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.prompt_versions_list.assert_called_once_with(
            prompt_id=_PROMPT_ID,
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_versions_defaults(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """list_versions() should default limit to 100 and cursor to None."""
        prompts_client.list_versions(prompt=_PROMPT_ID)

        mock_api.prompt_versions_list.assert_called_once_with(
            prompt_id=_PROMPT_ID,
            limit=100,
            cursor=None,
        )

    def test_list_versions_returns_api_response(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """list_versions() should propagate the return value from prompt_versions_list."""
        expected = Mock()
        mock_api.prompt_versions_list.return_value = expected

        result = prompts_client.list_versions(prompt=_PROMPT_ID)

        assert result is expected


@pytest.mark.unit
class TestPromptsClientCreateVersion:
    """Tests for PromptsClient.create_version()."""

    def test_create_version_builds_request_and_calls_api(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """create_version() should build PromptVersionsCreateRequest and call the API."""
        mock_messages = [Mock()]
        mock_input_format = Mock()
        mock_provider = Mock()

        with patch(
            "arize._generated.api_client.PromptVersionsCreateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            prompts_client.create_version(
                prompt=_PROMPT_ID,
                commit_message="v2",
                input_variable_format=mock_input_format,
                provider=mock_provider,
                messages=mock_messages,
                model="gpt-4o",
            )

        mock_request_cls.assert_called_once_with(
            commit_message="v2",
            input_variable_format=mock_input_format,
            provider=mock_provider,
            model="gpt-4o",
            messages=mock_messages,
            invocation_params=None,
            provider_params=None,
        )
        mock_api.prompt_versions_create.assert_called_once_with(
            prompt_id=_PROMPT_ID,
            prompt_versions_create_request=mock_body,
        )

    def test_create_version_returns_api_response(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """create_version() should propagate the return value from prompt_versions_create."""
        expected = Mock()
        mock_api.prompt_versions_create.return_value = expected

        with patch("arize._generated.api_client.PromptVersionsCreateRequest"):
            result = prompts_client.create_version(
                prompt=_PROMPT_ID,
                commit_message="v2",
                input_variable_format=Mock(),
                provider=Mock(),
                messages=[Mock()],
            )

        assert result is expected


@pytest.mark.unit
class TestPromptsClientGetLabel:
    """Tests for PromptsClient.get_label()."""

    def test_get_label_calls_api_with_params(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """get_label() should resolve prompt and pass prompt_id and label_name to prompt_labels_get."""
        prompts_client.get_label(prompt=_PROMPT_ID, label_name="production")

        mock_api.prompt_labels_get.assert_called_once_with(
            prompt_id=_PROMPT_ID,
            label_name="production",
        )

    def test_get_label_returns_api_response(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """get_label() should propagate the return value from prompt_labels_get."""
        expected = Mock()
        mock_api.prompt_labels_get.return_value = expected

        result = prompts_client.get_label(
            prompt=_PROMPT_ID, label_name="staging"
        )

        assert result is expected


@pytest.mark.unit
class TestPromptsClientSetLabels:
    """Tests for PromptsClient.set_labels()."""

    def test_set_labels_builds_request_and_calls_api(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """set_labels() should build PromptVersionLabelsSetRequest and call the API."""
        with patch(
            "arize._generated.api_client.PromptVersionLabelsSetRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            prompts_client.set_labels(
                version_id="ver-456",
                labels=["production", "stable"],
            )

        mock_request_cls.assert_called_once_with(
            labels=["production", "stable"]
        )
        mock_api.prompt_version_labels_set.assert_called_once_with(
            version_id="ver-456",
            prompt_version_labels_set_request=mock_body,
        )

    def test_set_labels_returns_api_response(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """set_labels() should propagate the return value from prompt_version_labels_set."""
        expected = Mock()
        mock_api.prompt_version_labels_set.return_value = expected

        with patch("arize._generated.api_client.PromptVersionLabelsSetRequest"):
            result = prompts_client.set_labels(
                version_id="ver-456",
                labels=["production"],
            )

        assert result is expected


@pytest.mark.unit
class TestPromptsClientDeleteLabel:
    """Tests for PromptsClient.delete_label()."""

    def test_delete_label_calls_api_with_params(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """delete_label() should pass version_id and label_name to prompt_version_labels_delete."""
        prompts_client.delete_label(version_id="ver-456", label_name="staging")

        mock_api.prompt_version_labels_delete.assert_called_once_with(
            version_id="ver-456",
            label_name="staging",
        )

    def test_delete_label_returns_none(
        self, prompts_client: PromptsClient, mock_api: Mock
    ) -> None:
        """delete_label() should return None on success."""
        mock_api.prompt_version_labels_delete.return_value = None

        result = prompts_client.delete_label(
            version_id="ver-456", label_name="staging"
        )

        assert result is None
