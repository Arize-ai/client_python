# CreateSamlIdentityProviderRequest

Supply exactly one metadata source: metadata_url or metadata_xml. Supplying both or neither returns 422.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**email_domains** | **List[str]** | Unique email domains served by this identity provider. | 
**metadata_url** | **str** | HTTPS metadata URL. Supply this or metadata_xml. | [optional] 
**metadata_xml** | **str** | Metadata XML, at most 1 MiB. Supply this or metadata_url. | [optional] 
**default_organization_id** | **str** | Default organization ID, or null when default login is disabled. | [optional] 
**default_space_id** | **str** | Default space ID, or null when default login is disabled. | [optional] 
**default_organization_role_id** | **str** | Default organization role ID, or null when default login is disabled. | [optional] 
**default_space_role_id** | **str** | Default space role ID, or null when default login is disabled. | [optional] 
**enforce_saml** | **bool** | Whether users must authenticate through SAML. | 
**sync_user_roles** | **bool** | Whether login applies role mappings to existing users. | 
**sign_authn** | **bool** | Whether Arize signs SAML authentication requests sent to the identity provider. | 
**allow_login_with_defaults** | **bool** | Whether users without a matching role mapping receive default roles. | 
**role_mappings** | [**List[CreateSamlRoleMappingRequest]**](CreateSamlRoleMappingRequest.md) | Role mappings for this identity provider. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_saml_identity_provider_request import CreateSamlIdentityProviderRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateSamlIdentityProviderRequest from a JSON string
create_saml_identity_provider_request_instance = CreateSamlIdentityProviderRequest.from_json(json)
# print the JSON string representation of the object
print(CreateSamlIdentityProviderRequest.to_json())

# convert the object into a dict
create_saml_identity_provider_request_dict = create_saml_identity_provider_request_instance.to_dict()
# create an instance of CreateSamlIdentityProviderRequest from a dict
create_saml_identity_provider_request_from_dict = CreateSamlIdentityProviderRequest.from_dict(create_saml_identity_provider_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


