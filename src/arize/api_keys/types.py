"""Public type re-exports for the api_keys subdomain."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field

from arize._generated.api_client.models.api_key import ApiKey
from arize._generated.api_client.models.api_key_status import ApiKeyStatus
from arize._generated.api_client.models.api_key_type import ApiKeyType
from arize._generated.api_client.models.create_api_key_response import (
    CreateApiKeyResponse,
)
from arize._generated.api_client.models.custom_role_assignment import (
    CustomRoleAssignment,
)
from arize._generated.api_client.models.list_api_keys_response import (
    ListApiKeysResponse,
)
from arize._generated.api_client.models.organization_custom_role_assignment import (
    OrganizationCustomRoleAssignment,
)
from arize._generated.api_client.models.organization_predefined_role_assignment import (
    OrganizationPredefinedRoleAssignment,
)
from arize._generated.api_client.models.organization_role_assignment import (
    OrganizationRoleAssignment,
)
from arize._generated.api_client.models.predefined_role_assignment import (
    PredefinedRoleAssignment,
)
from arize._generated.api_client.models.refresh_api_key_response import (
    RefreshApiKeyResponse,
)
from arize._generated.api_client.models.service_api_key_created import (
    ServiceApiKeyCreated,
)
from arize._generated.api_client.models.service_key_bot_user import (
    ServiceKeyBotUser,
)
from arize._generated.api_client.models.service_key_bot_user_org_assignment import (
    ServiceKeyBotUserOrgAssignment,
)
from arize._generated.api_client.models.service_key_bot_user_space_assignment import (
    ServiceKeyBotUserSpaceAssignment,
)
from arize._generated.api_client.models.service_key_space_assignment import (
    ServiceKeySpaceAssignment,
)
from arize._generated.api_client.models.space_role_assignment import (
    SpaceRoleAssignment,
)
from arize._generated.api_client.models.user_api_key_created import (
    UserApiKeyCreated,
)
from arize._generated.api_client.models.user_role_assignment import (
    UserRoleAssignment,
)


@dataclass
class SpaceBinding:
    """Declares one space that the service key's bot user should access.

    The *binding* answers "which space?" The *role assignment* answers "with
    what role?". A role assignment is either a named predefined role (e.g.
    ``PredefinedRoleAssignment(name="MEMBER")``) or a custom RBAC role
    referenced by ID (``CustomRoleAssignment(id="<encoded-role-id>")``).

    Attributes:
        space: Space name or base64-encoded space ID.
        role: Role to assign the bot user within this space. When ``None``,
            the server defaults to the predefined ``MEMBER`` role. Use
            :class:`PredefinedRoleAssignment` for built-in roles (``ADMIN``,
            ``MEMBER``, ``READ_ONLY``) or :class:`CustomRoleAssignment` for
            custom RBAC roles.
    """

    space: str
    role: SpaceRoleAssignment | None = dataclass_field(default=None)


@dataclass
class OrgBinding:
    """Declares one organization that the service key's bot user should access.

    The *binding* answers "which org?". Each org binding contains one or more
    :class:`SpaceBinding` objects that specify which spaces within that org the
    bot user can access and with what role.

    Attributes:
        org_id: HMAC-encoded ID of the organization.
        spaces: Space bindings within this organization. At least one is
            required. All spaces must belong to this organization.
        role: Role to assign the bot user at the organization level. When
            ``None``, the server defaults to the predefined ``READ_ONLY`` role.
            Use :class:`OrganizationPredefinedRoleAssignment` for built-in
            roles or :class:`OrganizationCustomRoleAssignment` for custom RBAC
            roles.
    """

    org_id: str
    spaces: list[SpaceBinding] = dataclass_field(default_factory=list)
    role: OrganizationRoleAssignment | None = dataclass_field(default=None)

    def __post_init__(self) -> None:
        """Validate that at least one space binding is provided."""
        if not self.spaces:
            raise ValueError(
                "OrgBinding.spaces must contain at least one SpaceBinding"
            )


__all__ = [
    "ApiKey",
    "ApiKeyStatus",
    "ApiKeyType",
    "CreateApiKeyResponse",
    "CustomRoleAssignment",
    "ListApiKeysResponse",
    "OrgBinding",
    "OrganizationCustomRoleAssignment",
    "OrganizationPredefinedRoleAssignment",
    "OrganizationRoleAssignment",
    "PredefinedRoleAssignment",
    "RefreshApiKeyResponse",
    "ServiceApiKeyCreated",
    "ServiceKeyBotUser",
    "ServiceKeyBotUserOrgAssignment",
    "ServiceKeyBotUserSpaceAssignment",
    "ServiceKeySpaceAssignment",
    "SpaceBinding",
    "SpaceRoleAssignment",
    "UserApiKeyCreated",
    "UserRoleAssignment",
]
