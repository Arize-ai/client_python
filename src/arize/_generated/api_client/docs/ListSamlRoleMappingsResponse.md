# ListSamlRoleMappingsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data** | [**List[SamlRoleMapping]**](SamlRoleMapping.md) | Role mappings in this page. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) | Pagination state for this response. | 

## Example

```python
from arize._generated.api_client.models.list_saml_role_mappings_response import ListSamlRoleMappingsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListSamlRoleMappingsResponse from a JSON string
list_saml_role_mappings_response_instance = ListSamlRoleMappingsResponse.from_json(json)
# print the JSON string representation of the object
print(ListSamlRoleMappingsResponse.to_json())

# convert the object into a dict
list_saml_role_mappings_response_dict = list_saml_role_mappings_response_instance.to_dict()
# create an instance of ListSamlRoleMappingsResponse from a dict
list_saml_role_mappings_response_from_dict = ListSamlRoleMappingsResponse.from_dict(list_saml_role_mappings_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


