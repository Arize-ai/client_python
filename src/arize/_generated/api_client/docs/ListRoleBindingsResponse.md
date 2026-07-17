# ListRoleBindingsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_bindings** | [**List[RoleBinding]**](RoleBinding.md) | A list of role bindings. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_role_bindings_response import ListRoleBindingsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListRoleBindingsResponse from a JSON string
list_role_bindings_response_instance = ListRoleBindingsResponse.from_json(json)
# print the JSON string representation of the object
print(ListRoleBindingsResponse.to_json())

# convert the object into a dict
list_role_bindings_response_dict = list_role_bindings_response_instance.to_dict()
# create an instance of ListRoleBindingsResponse from a dict
list_role_bindings_response_from_dict = ListRoleBindingsResponse.from_dict(list_role_bindings_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


