"""Unit tests for src/arize/spans/client.py."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest

from arize.spans.client import SpansClient

if TYPE_CHECKING:
    import pandas as pd

# Base64 ID that passes is_resource_id() — decodes to "Project:123"
_PROJECT_ID = "UHJvamVjdDoxMjM="

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
    """Provide a mock SpansApi instance."""
    return Mock()


@pytest.fixture
def spans_client(mock_sdk_config: Mock, mock_api: Mock) -> SpansClient:
    """Provide a SpansClient with mocked internals."""
    with patch("arize._generated.api_client.SpansApi", return_value=mock_api):
        return SpansClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpansClientInit:
    """Tests for SpansClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.SpansApi", return_value=mock_api
        ):
            client = SpansClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_spans_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to SpansApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.SpansApi"
        ) as mock_spans_api_cls:
            SpansClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_spans_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestSpansClientList:
    """Tests for SpansClient.list()."""

    def test_list_builds_request_with_all_params(
        self, spans_client: SpansClient, mock_api: Mock
    ) -> None:
        """list() should forward all parameters into SpansListRequest."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 8, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.SpansListRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            spans_client.list(
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
        self, spans_client: SpansClient, mock_api: Mock
    ) -> None:
        """list() should pass the built request, limit, and cursor to spans_list."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        with patch(
            "arize._generated.api_client.SpansListRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            spans_client.list(
                project=_PROJECT_ID,
                limit=50,
                cursor="cursor-abc",
            )

        mock_api.spans_list.assert_called_once_with(
            spans_list_request=mock_body,
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, spans_client: SpansClient, mock_api: Mock
    ) -> None:
        """list() should default start_time/end_time/filter to None and limit to 100."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        with patch(
            "arize._generated.api_client.SpansListRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            spans_client.list(project=_PROJECT_ID)

        mock_request_cls.assert_called_once_with(
            project_id=_PROJECT_ID,
            start_time=None,
            end_time=None,
            filter=None,
        )
        mock_api.spans_list.assert_called_once_with(
            spans_list_request=mock_request_cls.return_value,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, spans_client: SpansClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from spans_list."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        expected = Mock()
        mock_api.spans_list.return_value = expected

        with patch("arize._generated.api_client.SpansListRequest"):
            result = spans_client.list(project=_PROJECT_ID)

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self, spans_client: SpansClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        with patch("arize._generated.api_client.SpansListRequest"):
            caplog.set_level(logging.WARNING)
            spans_client.list(project=_PROJECT_ID)

        assert any(
            "ALPHA" in r.message and "spans.list" in r.message
            for r in caplog.records
        )

    def test_list_emits_active_development_warning(
        self, spans_client: SpansClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """list() should log a warning that the endpoint is in active development."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        with patch("arize._generated.api_client.SpansListRequest"):
            caplog.set_level(logging.WARNING)
            spans_client.list(project=_PROJECT_ID)

        assert any("active development" in r.message for r in caplog.records)

    def test_list_alpha_warning_only_on_first_call(
        self, spans_client: SpansClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The ALPHA prerelease warning should be emitted only on the first call."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        with patch("arize._generated.api_client.SpansListRequest"):
            caplog.set_level(logging.WARNING)

            spans_client.list(project=_PROJECT_ID)
            alpha_count_first = sum(
                1 for r in caplog.records if "ALPHA" in r.message
            )
            caplog.clear()

            spans_client.list(project=_PROJECT_ID)
            alpha_count_second = sum(
                1 for r in caplog.records if "ALPHA" in r.message
            )

        assert alpha_count_first == 1
        assert alpha_count_second == 0

    def test_list_with_project_name_resolves_id(
        self, spans_client: SpansClient, mock_api: Mock
    ) -> None:
        """list() should resolve a project name to an ID via ProjectsApi."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        mock_project = Mock()
        mock_project.id = _PROJECT_ID
        mock_project.name = "my-project"
        mock_projects_api = Mock()
        mock_projects_api.projects_list.return_value = Mock(
            projects=[mock_project],
            pagination=Mock(next_cursor=None),
        )
        spans_client._projects_api = mock_projects_api

        with patch(
            "arize._generated.api_client.SpansListRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()
            spans_client.list(
                project="my-project", space="U3BhY2U6OTA1MDoxSmtS"
            )

        mock_request_cls.assert_called_once_with(
            project_id=_PROJECT_ID,
            start_time=None,
            end_time=None,
            filter=None,
        )


@pytest.mark.unit
class TestSpansClientExportToDfDeprecated:
    """Tests for the @deprecated decorator applied to SpansClient.export_to_df."""

    @pytest.fixture(autouse=True)
    def _clear_warned(self) -> None:
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

    @pytest.fixture(autouse=True)
    def _mock_flight_stack(self) -> MagicMock:
        """Patch ArizeFlightClient and ArizeExportClient for all tests in this class."""
        import pandas as pd

        mock_flight = MagicMock()
        mock_flight.__enter__ = Mock(return_value=mock_flight)
        mock_flight.__exit__ = Mock(return_value=False)

        mock_exporter = Mock()
        mock_exporter.export_to_df.return_value = pd.DataFrame()

        with (
            patch(
                "arize.spans.client.ArizeFlightClient", return_value=mock_flight
            ),
            patch(
                "arize.spans.client.ArizeExportClient",
                return_value=mock_exporter,
            ),
        ):
            yield

    @pytest.fixture
    def spans_client(self, mock_api: Mock) -> SpansClient:
        """SpansClient with a plain Mock sdk_config that allows all attribute access."""
        sdk_config = Mock()
        sdk_config.api_key = "test_api_key"
        sdk_config.flight_host = "flight.arize.com"
        sdk_config.flight_port = 443
        sdk_config.flight_scheme = "https"
        sdk_config.request_verify = True
        sdk_config.pyarrow_max_chunksize = 1000
        with patch(
            "arize._generated.api_client.SpansApi", return_value=mock_api
        ):
            return SpansClient(sdk_config=sdk_config, generated_client=Mock())

    def _call_export(self, spans_client: SpansClient) -> None:
        spans_client.export_to_df(
            space_id="space-1",
            project_name="my-project",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 8, tzinfo=timezone.utc),
        )


@pytest.mark.unit
class TestSpansClientLogWithEvals:
    """Tests for SpansClient.log() with an evals_dataframe."""

    @pytest.fixture
    def spans_client(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> SpansClient:
        """Provide a SpansClient with mocked internals and required config attrs."""
        mock_sdk_config.files_url = "https://files.arize.com"
        mock_sdk_config.request_verify = True
        mock_sdk_config.pyarrow_max_chunksize = 1000
        mock_sdk_config.headers = {"Authorization": "Bearer test_api_key"}
        with patch(
            "arize._generated.api_client.SpansApi", return_value=mock_api
        ):
            return SpansClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )

    @staticmethod
    def _make_spans_df() -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "context.span_id": "span-000000001",
                    "context.trace_id": "trace-00000001",
                    "name": "llm_call",
                    "span_kind": "LLM",
                    "start_time": "2024-01-15T10:00:00.000000+00:00",
                    "end_time": "2024-01-15T10:00:02.000000+00:00",
                    "attributes.llm.model_name": "gpt-4",
                },
            ]
        )

    @staticmethod
    def _make_evals_df() -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "context.span_id": "span-000000001",
                    "eval.Correctness.label": "correct",
                    "eval.Correctness.score": 1.0,
                },
            ]
        )

    def test_log_with_evals_does_not_raise_value_error(
        self, spans_client: SpansClient
    ) -> None:
        """log() with evals_dataframe should not raise ValueError from DataFrame truth checks."""
        with patch("arize.spans.client.post_arrow_table", return_value=Mock()):
            spans_client.log(
                space_id="space-1",
                project_name="my-project",
                dataframe=self._make_spans_df(),
                evals_dataframe=self._make_evals_df(),
            )
