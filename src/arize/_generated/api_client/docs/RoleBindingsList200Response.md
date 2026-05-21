# RoleBindingsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_bindings** | [**List[RoleBinding]**](RoleBinding.md) | A list of role bindings. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.role_bindings_list200_response import RoleBindingsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of RoleBindingsList200Response from a JSON string
role_bindings_list200_response_instance = RoleBindingsList200Response.from_json(json)
# print the JSON string representation of the object
print(RoleBindingsList200Response.to_json())

# convert the object into a dict
role_bindings_list200_response_dict = role_bindings_list200_response_instance.to_dict()
# create an instance of RoleBindingsList200Response from a dict
role_bindings_list200_response_from_dict = RoleBindingsList200Response.from_dict(role_bindings_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


