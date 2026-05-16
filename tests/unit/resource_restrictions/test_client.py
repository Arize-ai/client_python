"""Unit tests for src/arize/resource_restrictions/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.resource_restrictions.client import ResourceRestrictionsClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock ResourceRestrictionsApi instance."""
    return Mock()


@pytest.fixture
def resource_restrictions_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> ResourceRestrictionsClient:
    """Provide a ResourceRestrictionsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.ResourceRestrictionsApi",
        return_value=mock_api,
    ):
        return ResourceRestrictionsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestResourceRestrictionsClientInit:
    """Tests for ResourceRestrictionsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.ResourceRestrictionsApi",
            return_value=mock_api,
        ):
            client = ResourceRestrictionsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_resource_restrictions_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to ResourceRestrictionsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.ResourceRestrictionsApi"
        ) as mock_api_cls:
            ResourceRestrictionsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestResourceRestrictionsClientRestrict:
    """Tests for ResourceRestrictionsClient.restrict()."""

    def test_restrict_builds_request_and_calls_api(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """restrict() should build ResourceRestrictionsCreate and call the API."""
        mock_response = Mock()
        mock_api.resource_restrictions_create.return_value = mock_response

        with patch(
            "arize._generated.api_client.ResourceRestrictionCreate"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            resource_restrictions_client.restrict(resource_id="project-abc")

        mock_request_cls.assert_called_once_with(resource_id="project-abc")
        mock_api.resource_restrictions_create.assert_called_once_with(mock_body)

    def test_restrict_returns_resource_restriction(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """restrict() should return the resource_restriction field from the API response."""
        mock_restriction = Mock()
        mock_response = Mock()
        mock_response.resource_restriction = mock_restriction
        mock_api.resource_restrictions_create.return_value = mock_response

        with patch("arize._generated.api_client.ResourceRestrictionCreate"):
            result = resource_restrictions_client.restrict(
                resource_id="project-abc"
            )

        assert result is mock_restriction

    def test_restrict_emits_alpha_prerelease_warning(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        mock_response = Mock()
        resource_restrictions_client._api.resource_restrictions_create.return_value = mock_response

        with patch("arize._generated.api_client.ResourceRestrictionCreate"):
            resource_restrictions_client.restrict(resource_id="project-abc")

        assert any(
            "ALPHA" in record.message
            and "resource_restrictions.restrict" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestResourceRestrictionsClientUnrestrict:
    """Tests for ResourceRestrictionsClient.unrestrict()."""

    def test_unrestrict_calls_api_with_resource_id(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """unrestrict() should pass resource_id to resource_restrictions_delete."""
        resource_restrictions_client.unrestrict(resource_id="project-abc")

        mock_api.resource_restrictions_delete.assert_called_once_with(
            resource_id="project-abc"
        )

    def test_unrestrict_returns_none(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """unrestrict() should return None (204 No Content)."""
        mock_api.resource_restrictions_delete.return_value = None

        result = resource_restrictions_client.unrestrict(
            resource_id="project-abc"
        )

        assert result is None
