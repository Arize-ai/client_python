# SamlIdentityProviderSummary

Identity provider summary. Use the detail endpoint to read metadata, default roles and role mappings.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Identity provider ID. | 
**email_domains** | **List[str]** | Unique email domains served by this identity provider. | 
**domain_verification** | [**List[SamlDomainVerification]**](SamlDomainVerification.md) | Current email domain verification state. | 
**enforce_saml** | **bool** | Whether users must authenticate through SAML. | 
**sync_user_roles** | **bool** | Whether login applies role mappings to existing users. | 
**sign_authn** | **bool** | Whether Arize signs SAML authentication requests sent to the identity provider. | 
**allow_login_with_defaults** | **bool** | Whether users without a matching role mapping receive default roles. | 
**is_managed_by_file** | **bool** | Whether configuration comes from a file. File configurations are read-only. | 
**created_at** | **datetime** | Creation time in UTC, or null for file-managed configuration. | [optional] 
**updated_at** | **datetime** | Last update time in UTC, or null for file-managed configuration. | [optional] 

## Example

```python
from arize._generated.api_client.models.saml_identity_provider_summary import SamlIdentityProviderSummary

# TODO update the JSON string below
json = "{}"
# create an instance of SamlIdentityProviderSummary from a JSON string
saml_identity_provider_summary_instance = SamlIdentityProviderSummary.from_json(json)
# print the JSON string representation of the object
print(SamlIdentityProviderSummary.to_json())

# convert the object into a dict
saml_identity_provider_summary_dict = saml_identity_provider_summary_instance.to_dict()
# create an instance of SamlIdentityProviderSummary from a dict
saml_identity_provider_summary_from_dict = SamlIdentityProviderSummary.from_dict(saml_identity_provider_summary_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


