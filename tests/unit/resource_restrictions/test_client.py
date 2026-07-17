"""Unit tests for src/arize/resource_restrictions/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, create_autospec, patch

import pytest

from arize._generated.api_client import ResourceRestrictionsApi
from arize.resource_restrictions.client import ResourceRestrictionsClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock ResourceRestrictionsApi instance."""
    return create_autospec(ResourceRestrictionsApi, instance=True)


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
class TestResourceRestrictionsClientList:
    """Tests for ResourceRestrictionsClient.list()."""

    def test_list_uses_default_limit_and_no_filters(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """list() should forward the default limit and omit optional filters."""
        from arize.constants.config import DEFAULT_LIST_LIMIT

        resource_restrictions_client.list()

        mock_api.list_resource_restrictions.assert_called_once_with(
            resource_type=None,
            limit=DEFAULT_LIST_LIMIT,
            cursor=None,
        )

    def test_list_forwards_all_arguments(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """list() should forward resource_type, limit, and cursor to the API."""
        from arize.resource_restrictions.types import ResourceRestrictionType

        resource_restrictions_client.list(
            resource_type=ResourceRestrictionType.PROJECT,
            limit=25,
            cursor="next-page-token",
        )

        mock_api.list_resource_restrictions.assert_called_once_with(
            resource_type=ResourceRestrictionType.PROJECT,
            limit=25,
            cursor="next-page-token",
        )

    def test_list_returns_api_response(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """list() should return the response object from the API."""
        mock_response = Mock()
        mock_api.list_resource_restrictions.return_value = mock_response

        result = resource_restrictions_client.list()

        assert result is mock_response

    def test_list_emits_beta_prerelease_warning(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        resource_restrictions_client.list()

        assert any(
            "BETA" in record.message
            and "resource_restrictions.list" in record.message
            for record in caplog.records
        )


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
        mock_api.create_resource_restriction.return_value = mock_response

        with patch(
            "arize._generated.api_client.CreateResourceRestrictionRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            resource_restrictions_client.restrict(resource_id="project-abc")

        mock_request_cls.assert_called_once_with(resource_id="project-abc")
        mock_api.create_resource_restriction.assert_called_once_with(mock_body)

    def test_restrict_returns_resource_restriction(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """restrict() should return the ResourceRestriction from the API response."""
        mock_response = Mock()
        mock_api.create_resource_restriction.return_value = mock_response

        with patch(
            "arize._generated.api_client.CreateResourceRestrictionRequest"
        ):
            result = resource_restrictions_client.restrict(
                resource_id="project-abc"
            )

        assert result is mock_response

    def test_restrict_emits_beta_prerelease_warning(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        mock_response = Mock()
        resource_restrictions_client._api.create_resource_restriction.return_value = mock_response

        with patch(
            "arize._generated.api_client.CreateResourceRestrictionRequest"
        ):
            resource_restrictions_client.restrict(resource_id="project-abc")

        assert any(
            "BETA" in record.message
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

        mock_api.delete_resource_restriction.assert_called_once_with(
            resource_id="project-abc"
        )

    def test_unrestrict_returns_none(
        self,
        resource_restrictions_client: ResourceRestrictionsClient,
        mock_api: Mock,
    ) -> None:
        """unrestrict() should return None (204 No Content)."""
        mock_api.delete_resource_restriction.return_value = None

        result = resource_restrictions_client.unrestrict(
            resource_id="project-abc"
        )

        assert result is None
