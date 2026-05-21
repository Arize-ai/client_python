# RoleBindingListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_bindings** | [**List[RoleBinding]**](RoleBinding.md) | A list of role bindings. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.role_binding_list_response import RoleBindingListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of RoleBindingListResponse from a JSON string
role_binding_list_response_instance = RoleBindingListResponse.from_json(json)
# print the JSON string representation of the object
print(RoleBindingListResponse.to_json())

# convert the object into a dict
role_binding_list_response_dict = role_binding_list_response_instance.to_dict()
# create an instance of RoleBindingListResponse from a dict
role_binding_list_response_from_dict = RoleBindingListResponse.from_dict(role_binding_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


