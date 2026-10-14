"""Partial REST responses survive SDK response conversions."""

from unittest.mock import Mock

import pytest

from arize._generated.api_client.models.organization_membership import (
    OrganizationMembership as GeneratedOrganizationMembership,
)
from arize._generated.api_client.models.space_membership import (
    SpaceMembership as GeneratedSpaceMembership,
)
from arize._generated.api_client.models.task import Task as GeneratedTask
from arize._generated.api_client.models.user import User as GeneratedUser
from arize.annotation_configs.types import ListAnnotationConfigsResponse
from arize.exceptions import MissingResponseFieldError
from arize.organizations.types import OrganizationMembership
from arize.prompts.types import PromptWithVersion
from arize.spaces.types import SpaceMembership
from arize.tasks.types import ListTasksResponse, Task
from arize.users.client import UsersClient
from arize.users.types import (
    CreateUserResponse,
    CustomUserRole,
    InviteMode,
    ListUsersResponse,
    User,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "generated,wrapper",
    [
        (GeneratedUser, User),
        (GeneratedOrganizationMembership, OrganizationMembership),
        (GeneratedSpaceMembership, SpaceMembership),
        (GeneratedTask, Task),
    ],
)
def test_partial_response_conversion(generated, wrapper) -> None:
    """A wrapper does not restore required fields removed from response models."""
    raw = generated.from_dict({})
    response = wrapper.model_validate(raw, from_attributes=True)
    assert response.id is None
    assert (
        response.role is None
        if hasattr(response, "role")
        else response.name is None
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "wrapper",
    [
        ListUsersResponse,
        ListTasksResponse,
        CreateUserResponse,
        ListAnnotationConfigsResponse,
        PromptWithVersion,
    ],
)
def test_empty_response_wrapper(wrapper) -> None:
    """Response envelopes accept absent fields."""
    response = wrapper.model_validate({})
    assert all(
        getattr(response, field) is None
        for field in wrapper.model_fields
        if field != "additional_properties"
    )


@pytest.mark.unit
def test_create_user_preserves_partial_server_response(mock_sdk_config) -> None:
    """The create conversion preserves absent scalar fields from the server."""
    client = UsersClient(sdk_config=mock_sdk_config, generated_client=Mock())
    client._api = Mock()
    client._api.create_user.return_value = GeneratedUser.from_dict(
        {"email": "person@example.com"}
    )
    response = client.create(
        email="person@example.com",
        name="Person",
        role=CustomUserRole(id="role-1"),
        invite_mode=InviteMode.NONE,
    )
    assert response.email == "person@example.com"
    assert response.id is None
    assert response.name is None


@pytest.mark.unit
def test_role_missing_request_identity_has_sdk_error() -> None:
    """A domain role must retain the identity needed to reuse it in a request."""
    raw = GeneratedUser.from_dict({"role": {"type": "CUSTOM"}})
    with pytest.raises(MissingResponseFieldError, match=r"role\.id"):
        User.model_validate(raw, from_attributes=True)
