# RoleBinding


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the role binding. | 
**role_id** | **str** | A universally unique identifier | 
**user_id** | **str** | A universally unique identifier | 
**resource_type** | [**RoleBindingResourceType**](RoleBindingResourceType.md) |  | 
**resource_id** | **str** | A universally unique identifier | 
**created_at** | **datetime** | Timestamp when the binding was created. | 
**updated_at** | **datetime** | Timestamp when the binding was last updated. | 

## Example

```python
from arize._generated.api_client.models.role_binding import RoleBinding

# TODO update the JSON string below
json = "{}"
# create an instance of RoleBinding from a JSON string
role_binding_instance = RoleBinding.from_json(json)
# print the JSON string representation of the object
print(RoleBinding.to_json())

# convert the object into a dict
role_binding_dict = role_binding_instance.to_dict()
# create an instance of RoleBinding from a dict
role_binding_from_dict = RoleBinding.from_dict(role_binding_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


