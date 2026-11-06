# UpdateSamlIdentityProviderRequest

Omitted fields remain unchanged. Explicit false is applied. Metadata source changes clear the other source. Disabling default login clears all four default fields; enabling it requires a complete valid set.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**email_domains** | **List[str]** | Unique email domains served by this identity provider. | [optional] 
**metadata_url** | **str** | HTTPS metadata URL. Supply this or metadata_xml. | [optional] 
**metadata_xml** | **str** | Metadata XML, at most 1 MiB. Supply this or metadata_url. | [optional] 
**default_organization_id** | **str** | Default organization ID, or null when default login is disabled. | [optional] 
**default_space_id** | **str** | Default space ID, or null when default login is disabled. | [optional] 
**default_organization_role_id** | **str** | Default organization role ID, or null when default login is disabled. | [optional] 
**default_space_role_id** | **str** | Default space role ID, or null when default login is disabled. | [optional] 
**enforce_saml** | **bool** | Whether users must authenticate through SAML. | [optional] 
**sync_user_roles** | **bool** | Whether login applies role mappings to existing users. | [optional] 
**sign_authn** | **bool** | Whether Arize signs SAML authentication requests sent to the identity provider. | [optional] 
**allow_login_with_defaults** | **bool** | Whether users without a matching role mapping receive default roles. | [optional] 

## Example

```python
from arize._generated.api_client.models.update_saml_identity_provider_request import UpdateSamlIdentityProviderRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateSamlIdentityProviderRequest from a JSON string
update_saml_identity_provider_request_instance = UpdateSamlIdentityProviderRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateSamlIdentityProviderRequest.to_json())

# convert the object into a dict
update_saml_identity_provider_request_dict = update_saml_identity_provider_request_instance.to_dict()
# create an instance of UpdateSamlIdentityProviderRequest from a dict
update_saml_identity_provider_request_from_dict = UpdateSamlIdentityProviderRequest.from_dict(update_saml_identity_provider_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


