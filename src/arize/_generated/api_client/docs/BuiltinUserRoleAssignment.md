# BuiltinUserRoleAssignment

A builtin (predefined) account-level role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**UserRoleAssignmentType**](UserRoleAssignmentType.md) | Discriminator identifying this as a builtin role assignment. Must be &#x60;builtin&#x60;. | 
**name** | [**UserRole**](UserRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.builtin_user_role_assignment import BuiltinUserRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of BuiltinUserRoleAssignment from a JSON string
builtin_user_role_assignment_instance = BuiltinUserRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(BuiltinUserRoleAssignment.to_json())

# convert the object into a dict
builtin_user_role_assignment_dict = builtin_user_role_assignment_instance.to_dict()
# create an instance of BuiltinUserRoleAssignment from a dict
builtin_user_role_assignment_from_dict = BuiltinUserRoleAssignment.from_dict(builtin_user_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


