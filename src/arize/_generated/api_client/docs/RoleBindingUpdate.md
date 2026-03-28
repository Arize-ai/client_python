# RoleBindingUpdate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_id** | **str** | A universally unique identifier | 

## Example

```python
from arize._generated.api_client.models.role_binding_update import RoleBindingUpdate

# TODO update the JSON string below
json = "{}"
# create an instance of RoleBindingUpdate from a JSON string
role_binding_update_instance = RoleBindingUpdate.from_json(json)
# print the JSON string representation of the object
print(RoleBindingUpdate.to_json())

# convert the object into a dict
role_binding_update_dict = role_binding_update_instance.to_dict()
# create an instance of RoleBindingUpdate from a dict
role_binding_update_from_dict = RoleBindingUpdate.from_dict(role_binding_update_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


