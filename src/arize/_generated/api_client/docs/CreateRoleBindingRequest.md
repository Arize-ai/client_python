# CreateRoleBindingRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**user_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**resource_type** | [**RoleBindingResourceType**](RoleBindingResourceType.md) |  | 
**resource_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 

## Example

```python
from arize._generated.api_client.models.create_role_binding_request import CreateRoleBindingRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateRoleBindingRequest from a JSON string
create_role_binding_request_instance = CreateRoleBindingRequest.from_json(json)
# print the JSON string representation of the object
print(CreateRoleBindingRequest.to_json())

# convert the object into a dict
create_role_binding_request_dict = create_role_binding_request_instance.to_dict()
# create an instance of CreateRoleBindingRequest from a dict
create_role_binding_request_from_dict = CreateRoleBindingRequest.from_dict(create_role_binding_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


