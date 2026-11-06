"""Account-scoped SAML administration client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.pre_releases import ReleaseStage, prerelease_endpoint

if TYPE_CHECKING:
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.saml_identity_providers.types import (
        CreateMappingRequest,
        CreateRequest,
        IdentityProvider,
        ListMappingsResponse,
        ListResponse,
        RoleMapping,
        UpdateMappingRequest,
        UpdateRequest,
    )

_Request = TypeVar("_Request", bound=BaseModel)


def _request(
    model: type[_Request], value: _Request | Mapping[str, Any]
) -> _Request:
    if isinstance(value, model):
        return value
    if isinstance(value, Mapping):
        unknown = set(value) - set(model.model_fields)
        if unknown:
            raise ValueError(
                f"Unsupported SAML request fields: {', '.join(sorted(unknown))}"
            )
    return model.model_validate(value)


class SamlIdentityProvidersClient:
    """Manage SAML configurations with Enterprise account-admin access.

    This class is intended for internal SDK use. Access it through
    :class:`arize.ArizeClient`.
    """

    def __init__(
        self, *, sdk_config: SDKConfiguration, generated_client: ApiClient
    ) -> None:
        """Initialize the SAML client.

        Args:
            sdk_config: Resolved SDK configuration.
            generated_client: Shared generated API client.
        """
        from arize._generated import api_client as gen

        self._sdk_config = sdk_config
        self._api = gen.SAMLApi(generated_client)

    @prerelease_endpoint(
        key="saml_identity_providers.list", stage=ReleaseStage.ALPHA
    )
    def list(
        self, *, limit: int = DEFAULT_LIST_LIMIT, cursor: str | None = None
    ) -> ListResponse:
        """List configurations. Use the returned cursor for the next page.

        Args:
            limit: Maximum number of configurations to return, up to 100.
            cursor: Opaque cursor from a previous page.

        Returns:
            A page of identity provider summaries without metadata or mappings.

        Raises:
            ApiException: If the API request fails.
        """
        return self._api.list_saml_identity_providers(
            limit=limit, cursor=cursor
        )

    @prerelease_endpoint(
        key="saml_identity_providers.create", stage=ReleaseStage.ALPHA
    )
    def create(
        self, *, body: CreateRequest | Mapping[str, Any]
    ) -> IdentityProvider:
        """Create a configuration with one metadata source and valid login defaults.

        Args:
            body: Request model or JSON-compatible mapping. Domain verification
                flags are read-only and cannot be supplied. Non-empty space or
                project assignments require an organization.

        Returns:
            The created identity provider.

        Raises:
            ApiException: If the API request fails.
        """
        from arize.saml_identity_providers.types import CreateRequest

        request = _request(CreateRequest, body)
        return self._api.create_saml_identity_provider(
            create_saml_identity_provider_request=request
        )

    @prerelease_endpoint(
        key="saml_identity_providers.get", stage=ReleaseStage.ALPHA
    )
    def get(self, *, idp_id: str) -> IdentityProvider:
        """Read a configuration by ID, including file-managed configurations.

        Args:
            idp_id: Identity provider ID.

        Returns:
            The complete identity provider configuration.

        Raises:
            ApiException: If the API request fails.
        """
        return self._api.get_saml_identity_provider(idp_id=idp_id)

    @prerelease_endpoint(
        key="saml_identity_providers.update", stage=ReleaseStage.ALPHA
    )
    def update(
        self, *, idp_id: str, body: UpdateRequest | Mapping[str, Any]
    ) -> IdentityProvider:
        """Patch supplied fields. Omission, false, null and empty lists stay distinct.

        Args:
            idp_id: ID returned by list or create.
            body: Request model or JSON-compatible mapping. Setting
                allow_login_with_defaults to false clears all default fields.

        Returns:
            The updated identity provider.

        Raises:
            ApiException: If the API request fails.
        """
        from arize.saml_identity_providers.types import UpdateRequest

        request = _request(UpdateRequest, body)
        return self._api.update_saml_identity_provider(
            idp_id=idp_id, update_saml_identity_provider_request=request
        )

    @prerelease_endpoint(
        key="saml_identity_providers.delete", stage=ReleaseStage.ALPHA
    )
    def delete(self, *, idp_id: str) -> None:
        """Remove a disabled database configuration. Users and grants remain.

        Args:
            idp_id: Identity provider ID.

        Returns:
            None.

        Raises:
            ApiException: If the API request fails, including enforced or
                file-managed configurations.
        """
        self._api.delete_saml_identity_provider(idp_id=idp_id)

    @prerelease_endpoint(
        key="saml_identity_providers.list_role_mappings",
        stage=ReleaseStage.ALPHA,
    )
    def list_role_mappings(
        self,
        *,
        idp_id: str,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListMappingsResponse:
        """List mappings in configuration order using the returned page cursor.

        Args:
            idp_id: Identity provider ID.
            limit: Maximum number of mappings to return, up to 100.
            cursor: Opaque cursor from a previous page.

        Returns:
            A page of role mappings.

        Raises:
            ApiException: If the API request fails.
        """
        return self._api.list_saml_role_mappings(
            idp_id=idp_id, limit=limit, cursor=cursor
        )

    @prerelease_endpoint(
        key="saml_identity_providers.add_role_mapping", stage=ReleaseStage.ALPHA
    )
    def add_role_mapping(
        self, *, idp_id: str, body: CreateMappingRequest | Mapping[str, Any]
    ) -> RoleMapping:
        """Add a mapping. The server assigns its persistent ID.

        Args:
            idp_id: Identity provider ID.
            body: Request model or JSON-compatible mapping. Non-empty space
                or project assignments require an organization.

        Returns:
            The created role mapping.

        Raises:
            ApiException: If the API request fails.
        """
        from arize.saml_identity_providers.types import CreateMappingRequest

        return self._api.add_saml_role_mapping(
            idp_id=idp_id,
            create_saml_role_mapping_request=_request(
                CreateMappingRequest, body
            ),
        )

    @prerelease_endpoint(
        key="saml_identity_providers.update_role_mapping",
        stage=ReleaseStage.ALPHA,
    )
    def update_role_mapping(
        self,
        *,
        idp_id: str,
        mapping_id: str,
        body: UpdateMappingRequest | Mapping[str, Any],
    ) -> RoleMapping:
        """Patch supplied fields while keeping the mapping ID.

        Omitted fields remain unchanged. Null clears the organization only when
        the resulting mapping is valid. Empty role lists clear assignments.
        Null leaves this mapping unmanaged. Manual project roles are preserved
        only when every mapping is unmanaged.

        Args:
            idp_id: Identity provider ID.
            mapping_id: Persistent mapping ID.
            body: Request model or JSON-compatible mapping.

        Returns:
            The updated role mapping.

        Raises:
            ApiException: If the API request fails.
        """
        from arize.saml_identity_providers.types import UpdateMappingRequest

        return self._api.update_saml_role_mapping(
            idp_id=idp_id,
            mapping_id=mapping_id,
            update_saml_role_mapping_request=_request(
                UpdateMappingRequest, body
            ),
        )

    @prerelease_endpoint(
        key="saml_identity_providers.remove_role_mapping",
        stage=ReleaseStage.ALPHA,
    )
    def remove_role_mapping(self, *, idp_id: str, mapping_id: str) -> None:
        """Remove a mapping while retaining required login and admin mappings.

        Args:
            idp_id: Identity provider ID.
            mapping_id: Persistent mapping ID.

        Returns:
            None.

        Raises:
            ApiException: If the API request fails.
        """
        self._api.remove_saml_role_mapping(idp_id=idp_id, mapping_id=mapping_id)
