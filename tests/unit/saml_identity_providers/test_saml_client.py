"""SAML update request serialization."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from arize.saml_identity_providers.client import SamlIdentityProvidersClient


@pytest.mark.parametrize(
    "body",
    [
        {"sync_user_roles": False},
        {"allow_login_with_defaults": False, "default_organization_id": None},
        {"email_domains": ["example.com"]},
    ],
)
def test_update_preserves_supplied_fields(body, mock_sdk_config) -> None:
    client = SamlIdentityProvidersClient(
        sdk_config=mock_sdk_config, generated_client=MagicMock()
    )
    client._api = MagicMock()
    client.update(idp_id="idp", body=body)
    forwarded = client._api.update_saml_identity_provider.call_args.kwargs[
        "update_saml_identity_provider_request"
    ]
    assert forwarded.to_dict() == body


def test_update_rejects_empty_email_domains(mock_sdk_config) -> None:
    client = SamlIdentityProvidersClient(
        sdk_config=mock_sdk_config, generated_client=MagicMock()
    )
    client._api = MagicMock()

    with pytest.raises(ValidationError, match="email_domains"):
        client.update(idp_id="idp", body={"email_domains": []})

    client._api.update_saml_identity_provider.assert_not_called()


def test_update_names_unknown_fields(mock_sdk_config) -> None:
    client = SamlIdentityProvidersClient(
        sdk_config=mock_sdk_config, generated_client=MagicMock()
    )
    client._api = MagicMock()
    with pytest.raises(ValueError, match="unsupported_option"):
        client.update(idp_id="idp", body={"unsupported_option": True})
    client._api.update_saml_identity_provider.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {"is_account_admin": False},
        {"organization": None},
        {"project_roles": []},
        {"project_roles": None},
    ],
)
def test_mapping_update_preserves_supplied_fields(
    body, mock_sdk_config
) -> None:
    client = SamlIdentityProvidersClient(
        sdk_config=mock_sdk_config, generated_client=MagicMock()
    )
    client._api = MagicMock()
    client.update_role_mapping(idp_id="idp", mapping_id="mapping", body=body)
    forwarded = client._api.update_saml_role_mapping.call_args.kwargs[
        "update_saml_role_mapping_request"
    ]
    assert forwarded.to_dict() == body


@pytest.mark.parametrize("project_roles", [None, []])
def test_mapping_read_preserves_project_role_management(project_roles) -> None:
    from arize._generated.api_client.models.create_saml_role_mapping_request import (
        CreateSamlRoleMappingRequest,
    )
    from arize.saml_identity_providers.types import RoleMapping

    mapping = RoleMapping.model_validate(
        {
            "id": "mapping",
            "attributes": {"groups": "admins"},
            "is_account_admin": True,
            "space_roles": [],
            "project_roles": project_roles,
        }
    )
    body = mapping.to_dict()
    assert "project_roles_managed" not in body
    assert body["project_roles"] == project_roles
    assert (
        CreateSamlRoleMappingRequest.model_validate(body).to_dict()[
            "project_roles"
        ]
        == project_roles
    )
