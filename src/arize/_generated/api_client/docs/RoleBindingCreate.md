# RoleBindingCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_id** | **str** | A universally unique identifier | 
**user_id** | **str** | A universally unique identifier | 
**resource_type** | [**RoleBindingResourceType**](RoleBindingResourceType.md) |  | 
**resource_id** | **str** | A universally unique identifier | 

## Example

```python
from arize._generated.api_client.models.role_binding_create import RoleBindingCreate

# TODO update the JSON string below
json = "{}"
# create an instance of RoleBindingCreate from a JSON string
role_binding_create_instance = RoleBindingCreate.from_json(json)
# print the JSON string representation of the object
print(RoleBindingCreate.to_json())

# convert the object into a dict
role_binding_create_dict = role_binding_create_instance.to_dict()
# create an instance of RoleBindingCreate from a dict
role_binding_create_from_dict = RoleBindingCreate.from_dict(role_binding_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


