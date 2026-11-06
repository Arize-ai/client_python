# ListSamlIdentityProvidersResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data** | [**List[SamlIdentityProviderSummary]**](SamlIdentityProviderSummary.md) | Identity providers in this page. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) | Pagination state for this response. | 

## Example

```python
from arize._generated.api_client.models.list_saml_identity_providers_response import ListSamlIdentityProvidersResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListSamlIdentityProvidersResponse from a JSON string
list_saml_identity_providers_response_instance = ListSamlIdentityProvidersResponse.from_json(json)
# print the JSON string representation of the object
print(ListSamlIdentityProvidersResponse.to_json())

# convert the object into a dict
list_saml_identity_providers_response_dict = list_saml_identity_providers_response_instance.to_dict()
# create an instance of ListSamlIdentityProvidersResponse from a dict
list_saml_identity_providers_response_from_dict = ListSamlIdentityProvidersResponse.from_dict(list_saml_identity_providers_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


