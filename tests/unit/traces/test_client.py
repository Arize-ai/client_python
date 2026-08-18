"""Unit tests for src/arize/traces/client.py."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import Mock, create_autospec, patch

import pytest

from arize._generated.api_client import TracesApi
from arize._generated.api_client.exceptions import ApiException
from arize.traces.client import TracesClient

# Base64 ID that passes is_resource_id() — decodes to "Project:123"
_PROJECT_ID = "UHJvamVjdDoxMjM="

# Base64 ID that decodes to "Space:9050:1JkR"
_SPACE_ID = "U3BhY2U6OTA1MDoxSmtS"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk_config() -> Mock:
    """Provide a mock SDKConfiguration."""
    from arize.config import SDKConfiguration

    config = Mock(spec=SDKConfiguration)
    config.api_key = "test_api_key"
    return config


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock TracesApi instance."""
    return create_autospec(TracesApi, instance=True)


@pytest.fixture
def traces_client(mock_sdk_config: Mock, mock_api: Mock) -> TracesClient:
    """Provide a TracesClient with mocked internals."""
    with patch("arize._generated.api_client.TracesApi", return_value=mock_api):
        return TracesClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTracesClientInit:
    """Tests for TracesClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.TracesApi", return_value=mock_api
        ):
            client = TracesClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_traces_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to TracesApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.TracesApi"
        ) as mock_traces_api_cls:
            TracesClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_traces_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestTracesClientList:
    """Tests for TracesClient.list()."""

    @pytest.fixture(autouse=True)
    def _clear_warned(self) -> None:
        from arize import pre_releases

        pre_releases._WARNED.clear()

    def test_list_builds_request_with_all_params(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should forward all body parameters into ListTracesRequest."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 8, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.ListTracesRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            traces_client.list(
                project=_PROJECT_ID,
                start_time=start,
                end_time=end,
                filter="status_code = 'ERROR'",
                limit=50,
                cursor="cursor-abc",
            )

        mock_request_cls.assert_called_once_with(
            project_id=_PROJECT_ID,
            start_time=start,
            end_time=end,
            filter="status_code = 'ERROR'",
        )

    def test_list_calls_api_with_request_and_pagination(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should pass the built request, limit, and cursor to traces_list."""
        with patch(
            "arize._generated.api_client.ListTracesRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            traces_client.list(
                project=_PROJECT_ID,
                limit=50,
                cursor="cursor-abc",
            )

        mock_api.list_traces.assert_called_once_with(
            list_traces_request=mock_body,
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should default time/filter to None and limit to 50."""
        with patch(
            "arize._generated.api_client.ListTracesRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            traces_client.list(project=_PROJECT_ID)

        mock_request_cls.assert_called_once_with(
            project_id=_PROJECT_ID,
            start_time=None,
            end_time=None,
            filter=None,
        )
        mock_api.list_traces.assert_called_once_with(
            list_traces_request=mock_request_cls.return_value,
            limit=50,
            cursor=None,
        )

    def test_list_forwards_all_optional_args(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should forward start_time, end_time, filter, and cursor."""
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 2, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.ListTracesRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            traces_client.list(
                project=_PROJECT_ID,
                start_time=start,
                end_time=end,
                filter="span_kind = 'LLM'",
                limit=25,
                cursor="next-page",
            )

        mock_request_cls.assert_called_once_with(
            project_id=_PROJECT_ID,
            start_time=start,
            end_time=end,
            filter="span_kind = 'LLM'",
        )
        mock_api.list_traces.assert_called_once_with(
            list_traces_request=mock_request_cls.return_value,
            limit=25,
            cursor="next-page",
        )

    def test_list_returns_api_response(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from traces_list."""
        expected = Mock()
        mock_api.list_traces.return_value = expected

        with patch("arize._generated.api_client.ListTracesRequest"):
            result = traces_client.list(project=_PROJECT_ID)

        assert result is expected

    def test_list_with_project_name_resolves_id(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should resolve a project name to an ID via ProjectsApi."""
        mock_project = Mock()
        mock_project.id = _PROJECT_ID
        mock_project.name = "my-project"
        mock_projects_api = Mock()
        mock_projects_api.list_projects.return_value = Mock(
            projects=[mock_project],
            pagination=Mock(next_cursor=None),
        )
        traces_client._projects_api = mock_projects_api

        with patch(
            "arize._generated.api_client.ListTracesRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()
            traces_client.list(project="my-project", space=_SPACE_ID)

        mock_request_cls.assert_called_once_with(
            project_id=_PROJECT_ID,
            start_time=None,
            end_time=None,
            filter=None,
        )

    def test_list_project_id_path_no_space_needed(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should accept a project ID without requiring space."""
        with patch(
            "arize._generated.api_client.ListTracesRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()
            traces_client.list(project=_PROJECT_ID)

        mock_request_cls.assert_called_once_with(
            project_id=_PROJECT_ID,
            start_time=None,
            end_time=None,
            filter=None,
        )
        mock_api.list_traces.assert_called_once_with(
            list_traces_request=mock_request_cls.return_value,
            limit=50,
            cursor=None,
        )

    def test_list_pagination_cursor_forwarded(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should forward the cursor to the api call for pagination."""
        with patch(
            "arize._generated.api_client.ListTracesRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()
            traces_client.list(project=_PROJECT_ID, cursor="page-2")

        mock_api.list_traces.assert_called_once_with(
            list_traces_request=mock_request_cls.return_value,
            limit=50,
            cursor="page-2",
        )

    def test_list_propagates_api_exception(
        self, traces_client: TracesClient, mock_api: Mock
    ) -> None:
        """list() should raise when the underlying api raises ApiException."""
        mock_api.list_traces.side_effect = ApiException(
            status=403, reason="Forbidden"
        )

        with (
            patch("arize._generated.api_client.ListTracesRequest"),
            pytest.raises(ApiException),
        ):
            traces_client.list(project=_PROJECT_ID)

    def test_list_emits_beta_prerelease_warning(
        self, traces_client: TracesClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        with patch("arize._generated.api_client.ListTracesRequest"):
            caplog.set_level(logging.WARNING)
            traces_client.list(project=_PROJECT_ID)

        assert any(
            "BETA" in r.message and "traces.list" in r.message
            for r in caplog.records
        )

    def test_list_beta_warning_only_on_first_call(
        self, traces_client: TracesClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The BETA prerelease warning should be emitted only on the first call."""
        with patch("arize._generated.api_client.ListTracesRequest"):
            caplog.set_level(logging.WARNING)

            traces_client.list(project=_PROJECT_ID)
            beta_count_first = sum(
                1 for r in caplog.records if "BETA" in r.message
            )
            caplog.clear()

            traces_client.list(project=_PROJECT_ID)
            beta_count_second = sum(
                1 for r in caplog.records if "BETA" in r.message
            )

        assert beta_count_first == 1
        assert beta_count_second == 0


@pytest.mark.unit
class TestTracesTypes:
    """Tests for the traces types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        import arize.traces.types as types_module

        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        import arize.traces.types as types_module

        assert "ListTracesResponse" in types_module.__all__
        assert "Trace" in types_module.__all__

    def test_public_types_importable(self) -> None:
        """Trace and ListTracesResponse should be importable from arize.traces.types."""
        from arize.traces.types import ListTracesResponse, Trace

        assert isinstance(ListTracesResponse, type)
        assert isinstance(Trace, type)
