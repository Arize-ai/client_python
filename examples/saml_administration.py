"""Run against a test account with Enterprise admin credentials."""

import os
from pathlib import Path

from arize import ArizeClient
from arize.utils.response import require_operation_value

client = ArizeClient()
xml = Path(os.environ["SAML_METADATA_XML_FILE"]).read_text()
created = client.saml_identity_providers.create(
    body={
        "email_domains": [os.environ["SAML_TEST_DOMAIN"]],
        "metadata_xml": xml,
        "enforce_saml": False,
        "sync_user_roles": True,
        "sign_authn": True,
        "allow_login_with_defaults": False,
        "role_mappings": [
            {
                "attributes": {"groups": "rest-admins"},
                "is_account_admin": True,
            }
        ],
    }
)
idp_id = require_operation_value(created.id, field="SamlIdentityProvider.id")
print(client.saml_identity_providers.list().model_dump())
print(client.saml_identity_providers.get(idp_id=idp_id).model_dump())
print(
    client.saml_identity_providers.update(
        idp_id=idp_id, body={"sync_user_roles": False}
    ).model_dump()
)
added = client.saml_identity_providers.add_role_mapping(
    idp_id=idp_id,
    body={
        "attributes": {"groups": "rest-secondary"},
        "is_account_admin": True,
    },
)
mapping_id = require_operation_value(added.id, field="SamlRoleMapping.id")
updated_mapping = client.saml_identity_providers.update_role_mapping(
    idp_id=idp_id,
    mapping_id=mapping_id,
    body={"attributes": {"groups": "rest-secondary-renamed"}},
)
if updated_mapping.id != mapping_id:
    raise ValueError("Mapping ID changed during an update")
print(
    client.saml_identity_providers.list_role_mappings(
        idp_id=idp_id
    ).model_dump()
)
client.saml_identity_providers.remove_role_mapping(
    idp_id=idp_id, mapping_id=mapping_id
)
client.saml_identity_providers.delete(idp_id=idp_id)
